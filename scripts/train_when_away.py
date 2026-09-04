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
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from datetime import time as clock_time
from pathlib import Path

STATUS_URL = "http://127.0.0.1:47834/status"
#: How long the user must be away before training starts. Screen Time's own
#: idleThreshold (120s by default) already smooths brief pauses; this is on top
#: of it, so a trip to the kettle does not start a run that stops on return.
SETTLE_SECONDS = 180
#: How often presence is re-checked while training. The trainer stops at a step
#: boundary after this, so it is also roughly the worst-case delay before the
#: machine is handed back.
POLL_SECONDS = 20


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


def may_train(args: argparse.Namespace, now: datetime) -> tuple[bool, str]:
    """Decide whether training may run right now, and say why."""
    if args.ignore_presence:
        return True, "presence checks disabled"
    here = presence(args.status_url)
    if here is None:
        window = inside(now.time(), args.start_t, args.end_t)
        reason = (
            f"presence API unreachable; falling back to the "
            f"{args.start}-{args.end} window"
        )
        return window, reason
    if here:
        return False, "someone is at the computer"
    return True, "nobody at the computer"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", default="00:00",
                        help="fallback window start, used only when presence is unknown")
    parser.add_argument("--to", dest="end", default="07:15",
                        help="fallback window end, used only when presence is unknown")
    parser.add_argument("--status-url", default=STATUS_URL)
    parser.add_argument("--ignore-presence", action="store_true",
                        help="train continuously, ignoring who is at the machine")
    parser.add_argument("--settle", type=int, default=SETTLE_SECONDS)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--data", default="data_v05")
    parser.add_argument("--out", default="checkpoints/swift-v05")
    parser.add_argument("--steps", type=int, default=63_232)
    # 100, not the trainer's 500: a checkpoint costs 0.3s to write, so saving
    # five times more often is 0.08% overhead and caps what an unexpected
    # shutdown can destroy at ~5 minutes instead of ~27.
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args(argv)
    args.start_t, args.end_t = parse_clock(args.start), parse_clock(args.end)

    out = Path(args.out)
    stop_file = out / "STOP"
    out.mkdir(parents=True, exist_ok=True)
    # A stop-file left behind by a killed run would stop the next one instantly.
    stop_file.unlink(missing_ok=True)

    away_since: float | None = None
    last_reason = ""

    while True:
        now = datetime.now()
        allowed, reason = may_train(args, now)

        if not allowed:
            away_since = None
            if reason != last_reason:
                print(f"[{now:%Y-%m-%d %H:%M}] waiting: {reason}", flush=True)
                last_reason = reason
            time.sleep(POLL_SECONDS)
            continue

        # Settle: be away for a while before committing to a run.
        if away_since is None:
            away_since = time.time()
        waited = time.time() - away_since
        if waited < args.settle:
            time.sleep(min(POLL_SECONDS, args.settle - waited))
            continue

        print(f"[{now:%Y-%m-%d %H:%M}] training: {reason}", flush=True)
        last_reason = ""
        code = run_training(args, stop_file)
        away_since = None
        if code != 0:
            print(f"trainer exited {code}; stopping", flush=True)
            return code


def run_training(args: argparse.Namespace, stop_file: Path) -> int:
    """Train until the user comes back, then ask the trainer to stop cleanly."""
    command = [
        sys.executable, "-m", "minerva.training.trainer",
        "--data", args.data,
        "--out", args.out,
        "--steps", str(args.steps),
        "--threads", str(args.threads),
        "--checkpoint-interval", str(args.checkpoint_interval),
        "--stop-file", str(stop_file),
    ]
    # Resume from the rolling checkpoint. `last.pt`, not `best.pt`: best lags
    # actual progress, since it only moves when validation improves, and
    # resuming from it silently discards real steps.
    last = Path(args.out) / "last.pt"
    if last.exists():
        command += ["--resume", str(last)]
    command += args.extra

    stop_file.unlink(missing_ok=True)
    process = subprocess.Popen(command)
    try:
        while process.poll() is None:
            time.sleep(POLL_SECONDS)
            allowed, reason = may_train(args, datetime.now())
            if not allowed:
                print(f"  yielding the machine: {reason}", flush=True)
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
