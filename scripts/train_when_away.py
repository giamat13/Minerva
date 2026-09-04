"""Train while the machine's owner is away, and stop the moment they return.

Training a 26.8M model over the v0.5.0 corpus is days of CPU time on someone's
personal computer. A fixed night window was the first approximation; asking
the machine whether anyone is actually sitting at it is a better one, because
it uses every idle afternoon as well as every night, and gets out of the way
during a 2am work session.

Presence comes from Screen Time's read-only local API (see docs/local-api.md):

    GET http://127.0.0.1:47834/status  ->  {"present": true, ...}

    python scripts/train_when_away.py

**Falling back to hours.** If that API cannot be reached - the app is not
running, the port moved - presence is unknown, and guessing either way is
wrong: assume "away" and it may grind the machine while its owner is typing;
assume "present" and it may never train at all. So it falls back to the
agreed time window (00:00-07:15), which is the behaviour that was already
acceptable, and says so in the log rather than failing silently.

**Stopping costs nothing.** The trainer is asked to stop through a stop-file,
so it finishes the step in hand and checkpoints before exiting. Killing it
would forfeit every step since the last save.

**The asymmetry is deliberate.** It yields immediately when someone appears,
but waits for a settle period before starting, so stepping away for a coffee
does not start and stop a run every two minutes.

**Sharing.** Being at the computer does not always mean the computer is busy.
When someone is here *and* there is real spare memory, training continues at
below-normal priority on fewer threads - Windows then gives it only cycles
nothing else wants, so foreground work never queues behind it. Two thresholds
rather than one: 6 GB free to start sharing, 3 GB to keep sharing, so a run
does not flicker on and off as memory moves around a single number. If
something large starts - a game, say - free memory falls through the floor and
training stops within seconds, having checkpointed.

Memory is measured as *available* rather than *free*: Windows keeps most of
RAM full of file cache and hands it to a new process on demand, so "free"
understates what a game would actually get by many gigabytes.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from datetime import time as clock_time
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from free_memory import free_memory

STATUS_URL = "http://127.0.0.1:47834/status"
#: How long the user must be away before training starts. Screen Time's own
#: idleThreshold (120s by default) already smooths brief pauses; this is on top
#: of it, so a trip to the kettle does not start a run that stops on return.
SETTLE_SECONDS = 180
#: How often presence is re-checked while training. The trainer stops at a step
#: boundary after this, so it is also roughly the worst-case delay before the
#: machine is handed back.
POLL_SECONDS = 20
#: Faster while sharing the machine: a game can claim several GB in seconds,
#: so the memory floor is only worth having if it is noticed quickly.
POLL_SECONDS_SHARED = 8


def parse_clock(value: str) -> clock_time:
    hour, _, minute = value.partition(":")
    return clock_time(int(hour), int(minute or 0))


def inside(now: clock_time, start: clock_time, end: clock_time) -> bool:
    """Is `now` inside the window? Handles windows that cross midnight."""
    if start <= end:
        return start <= now < end
    return now >= start or now < end


def seconds_until(now: datetime, target: clock_time) -> float:
    """Seconds from `now` to the next occurrence of `target`, tomorrow if past."""
    candidate = now.replace(
        hour=target.hour, minute=target.minute, second=0, microsecond=0
    )
    if candidate <= now:
        candidate += timedelta(days=1)
    return (candidate - now).total_seconds()


def presence(url: str, timeout: float = 4.0) -> bool | None:
    """True if someone is at the computer, False if not, None if unknowable.

    None is a real answer and must not be collapsed into False: a connection
    refusal means Screen Time is not running, not that nobody is here.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return None
    value = payload.get("present")
    return bool(value) if isinstance(value, bool) else None


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def available_gb() -> float | None:
    """Physical memory a new process could take right now, in GB.

    GlobalMemoryStatusEx's ullAvailPhys, not "free" memory: Windows reports
    most of RAM as in-use because it holds file cache there, and that cache is
    handed to a new process instantly. Available is therefore what a game
    launching would actually get, and free would understate it badly - on this
    machine, 1.8 GB free against 15.5 GB total, with the honest answer being a
    different number entirely. Costs microseconds, so it is safe to poll.
    """
    if not sys.platform.startswith("win"):
        return None
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status.ullAvailPhys / 2**30


class Decision(NamedTuple):
    """Whether to train right now, and why."""

    allowed: bool
    reason: str
    shared: bool
    #: True when the only thing standing in the way is free memory - which is
    #: the one obstacle this script can do something about, by reclaiming some.
    memory_blocked: bool = False


def may_train(args: argparse.Namespace, now: datetime, running: bool = False) -> Decision:
    """May training run now?

    `shared` means "the machine's owner is here, and we are only borrowing
    spare capacity" - the caller runs the trainer at low priority with fewer
    threads in that mode.

    `running` selects the memory threshold. Starting demands more headroom
    than continuing, because a run that starts and stops as memory wobbles
    around one number is worse than either.
    """
    if args.ignore_presence:
        return Decision(True, "presence checks disabled", False)

    here = presence(args.status_url)
    if here is None:
        window = inside(now.time(), args.start_t, args.end_t)
        return Decision(window, (
            f"presence API unreachable; falling back to the "
            f"{args.start}-{args.end} window"
        ), False)

    if not here:
        return Decision(True, "nobody at the computer", False)

    # Someone is here. Train only out of genuine spare capacity, and only if
    # there is enough left over that a heavy application - a game, say - can
    # still start without fighting us for memory.
    if not args.shared:
        return Decision(False, "someone is at the computer", False)

    free = available_gb()
    if free is None:
        return Decision(
            False, "someone is at the computer (cannot measure free memory)", False
        )

    floor = args.shared_floor_gb if running else args.shared_start_gb
    if free < floor:
        return Decision(False, (
            f"someone is at the computer and only {free:.1f} GB is free "
            f"(need {floor:.1f} GB to {'keep' if running else 'start'} sharing)"
        ), False, memory_blocked=True)

    return Decision(True, f"sharing: {free:.1f} GB free while you work", True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", default="00:00",
                        help="fallback window start, used only when presence is unknown")
    parser.add_argument("--to", dest="end", default="07:15",
                        help="fallback window end, used only when presence is unknown")
    parser.add_argument("--status-url", default=STATUS_URL)
    parser.add_argument("--ignore-presence", action="store_true",
                        help="train continuously, ignoring who is at the machine")
    parser.add_argument("--no-shared", dest="shared", action="store_false",
                        help="never train while someone is at the computer")
    # Defaults sized from measurement: the trainer's resident set is ~1.2 GB,
    # and the machine has 15.5 GB. Requiring 6 GB free before sharing leaves
    # roughly 4.8 GB for whatever starts next, which covers a heavy game; the
    # 3 GB floor then stops us quickly if something actually claims it.
    parser.add_argument("--shared-start-gb", type=float, default=6.0,
                        help="free memory needed before training alongside you")
    parser.add_argument("--shared-floor-gb", type=float, default=3.0,
                        help="stop sharing if free memory falls below this")
    parser.add_argument("--shared-threads", type=int, default=4,
                        help="threads to use while sharing the machine with you")
    parser.add_argument("--free-memory-every", type=float, default=30.0,
                        help="minutes between memory-reclaim attempts when blocked "
                             "on free memory; 0 disables it")
    parser.add_argument("--settle", type=int, default=SETTLE_SECONDS)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--data", default="data_v05")
    parser.add_argument("--out", default="checkpoints/swift-v05")
    parser.add_argument("--steps", type=int, default=63_232)
    # 50, not the trainer's 500. A checkpoint costs ~0.28s including the
    # fsync, and a step takes ~4.8s, so saving ten times more often is ~0.12%
    # overhead. That is what a *crash* costs - a clean shutdown costs almost
    # nothing, because the trainer catches the shutdown event and saves.
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args(argv)
    args.start_t, args.end_t = parse_clock(args.start), parse_clock(args.end)

    # The hysteresis only works one way round. With a start threshold at or
    # below the floor, a run stopped for low memory qualifies to start again
    # immediately, and training flaps on and off for as long as memory sits
    # between the two.
    if args.shared and args.shared_start_gb <= args.shared_floor_gb:
        parser.error(
            f"--shared-start-gb ({args.shared_start_gb}) must be above "
            f"--shared-floor-gb ({args.shared_floor_gb}); otherwise a run that "
            f"stops for low memory immediately qualifies to start again"
        )

    out = Path(args.out)
    stop_file = out / "STOP"
    out.mkdir(parents=True, exist_ok=True)
    # A stop-file left behind by a killed run would stop the next one instantly.
    stop_file.unlink(missing_ok=True)

    away_since: float | None = None
    last_reason = ""

    last_reclaim = 0.0

    while True:
        now = datetime.now()
        decision = may_train(args, now)

        # Memory is the one obstacle that can be argued with. When it is the
        # only thing in the way, reclaim some and re-decide immediately -
        # measured at +2.6 GB on this machine, which is the difference between
        # waiting and training. Rate-limited because it trims every process's
        # working set, which is not free for whoever is using them.
        if (
            decision.memory_blocked
            and args.free_memory_every > 0
            and time.time() - last_reclaim > args.free_memory_every * 60
        ):
            last_reclaim = time.time()
            print(f"[{now:%Y-%m-%d %H:%M}] low on memory; reclaiming", flush=True)
            if free_memory(verbose=True) > 0:
                decision = may_train(args, now)

        if not decision.allowed:
            away_since = None
            if decision.reason != last_reason:
                print(f"[{now:%Y-%m-%d %H:%M}] waiting: {decision.reason}", flush=True)
                last_reason = decision.reason
            time.sleep(POLL_SECONDS)
            continue

        # Settle before committing to a run. Sharing settles too, so a
        # momentary dip in memory use does not start a run that the next
        # sample stops again.
        if away_since is None:
            away_since = time.time()
        waited = time.time() - away_since
        if waited < args.settle:
            time.sleep(min(POLL_SECONDS, args.settle - waited))
            continue

        print(f"[{now:%Y-%m-%d %H:%M}] training: {decision.reason}", flush=True)
        last_reason = ""
        code = run_training(args, stop_file, decision.shared)
        away_since = None
        if code != 0:
            print(f"trainer exited {code}; stopping", flush=True)
            return code


def readable_checkpoint(path: Path) -> bool:
    """Can this checkpoint actually be loaded?

    Cheap insurance against the one failure the atomic-write-plus-fsync in
    save_checkpoint is meant to prevent. If it ever does happen - a failing
    disk, a filesystem that reordered the rename anyway - the run should fall
    back to the older checkpoint rather than refuse to start and lose the
    entire history.
    """
    try:
        import torch

        torch.load(path, map_location="cpu", weights_only=False)
        return True
    except Exception as exc:
        print(f"  {path.name} will not load ({type(exc).__name__}); "
              f"trying an older checkpoint", flush=True)
        return False


def pick_checkpoint(out: Path) -> Path | None:
    """The newest checkpoint that actually loads, or None to start fresh.

    `last.pt` before `best.pt`: best only moves when validation improves, so
    its step lags real progress and resuming from it silently discards work.
    A stale `.pt.tmp` is a half-written save from a crash and is removed - it
    is never a resume candidate.
    """
    for stale in out.glob("*.pt.tmp"):
        print(f"  removing half-written {stale.name} from an interrupted save",
              flush=True)
        stale.unlink(missing_ok=True)

    for name in ("last.pt", "best.pt"):
        candidate = out / name
        if candidate.exists() and readable_checkpoint(candidate):
            return candidate
    return None


def run_training(args: argparse.Namespace, stop_file: Path, shared: bool) -> int:
    """Train until the machine is wanted back, then stop the trainer cleanly.

    In shared mode the trainer runs below normal priority and on fewer
    threads. Priority is what actually keeps the machine responsive: Windows
    hands a below-normal process only the cycles nothing else wants, so
    foreground work never queues behind training. The reduced thread count is
    for cache and memory pressure, which priority does not help with.
    """
    threads = args.shared_threads if shared else args.threads
    command = [
        sys.executable, "-m", "minerva.training.trainer",
        "--data", args.data,
        "--out", args.out,
        "--steps", str(args.steps),
        "--threads", str(threads),
        "--checkpoint-interval", str(args.checkpoint_interval),
        "--stop-file", str(stop_file),
    ]
    resume = pick_checkpoint(Path(args.out))
    if resume:
        command += ["--resume", str(resume)]
    command += args.extra

    flags = 0
    if shared and hasattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS"):
        flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS
        print(f"  shared mode: {threads} threads, below-normal priority", flush=True)

    # Polled faster while sharing: a game can claim several GB in seconds, and
    # the memory floor is only useful if it is noticed quickly.
    poll = POLL_SECONDS_SHARED if shared else POLL_SECONDS

    stop_file.unlink(missing_ok=True)
    process = subprocess.Popen(command, creationflags=flags)
    try:
        while process.poll() is None:
            time.sleep(poll)
            decision = may_train(args, datetime.now(), running=True)
            # Also restart out of shared mode once the machine is free, so a
            # night does not run at four threads because it began while
            # someone was still awake.
            if not decision.allowed or decision.shared != shared:
                print(f"  yielding: {decision.reason}", flush=True)
                stop_file.touch()
                break
        # Whether it is finishing on its own or on request, let it save.
        return process.wait()
    except KeyboardInterrupt:
        stop_file.touch()
        process.wait()
        raise
    finally:
        stop_file.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
