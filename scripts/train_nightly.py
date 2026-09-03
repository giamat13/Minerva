"""Run training only inside an allowed window, resuming across days.

Training a 23M model over the v0.5.0 corpus is days of wall clock on CPU, and
the machine it runs on is someone's actual computer. This wrapper keeps the run
inside hours that were agreed, sleeping the rest of the time and picking up
from the last checkpoint - the trainer already resumes, so a pause costs
nothing but the pause.

    python scripts/train_nightly.py --from 00:00 --to 07:15

The default --out is `checkpoints/swift-v05`, deliberately not the v0.3.0
`checkpoints/swift`: that directory holds 9.9M weights, and resuming a 23M
config from them would either fail or quietly train the wrong shape. A new
architecture gets a new directory (CLAUDE.md section 9).

Leave the window off to train continuously:

    python scripts/train_nightly.py --all-day --threads 10

Measured on this machine (23.2M params, batch 4 x seq 512): 14 threads gives
~2,520 tok/s and 10 threads ~2,365, so leaving four cores free for the desktop
costs about 6%. That is why --threads exists and why the default is not "every
core": a machine that stays usable is worth 6%.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta
from datetime import time as clock_time
from pathlib import Path


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", default="00:00")
    parser.add_argument("--to", dest="end", default="07:15")
    parser.add_argument("--all-day", action="store_true", help="ignore the window")
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--data", default="data_v05")
    parser.add_argument("--out", default="checkpoints/swift-v05")
    parser.add_argument("--steps", type=int, default=63_232)
    # 100, not the trainer's 500: a checkpoint costs 0.3s to write, so
    # saving 5x more often is 0.08% overhead, and it caps what an
    # unexpected reboot can destroy at ~5 minutes instead of ~27.
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args(argv)

    start, end = parse_clock(args.start), parse_clock(args.end)

    while True:
        now = datetime.now()
        if not args.all_day and not inside(now.time(), start, end):
            wait = seconds_until(now, start)
            print(
                f"[{now:%Y-%m-%d %H:%M}] outside the {args.start}-{args.end} window; "
                f"sleeping {wait/3600:.1f}h",
                flush=True,
            )
            time.sleep(min(wait, 900))  # re-check every 15 min so a clock change lands
            continue

        # Stop at the end of the window rather than mid-step: --max-hours makes
        # the trainer save and exit cleanly, and the next pass resumes.
        budget = None
        if not args.all_day:
            budget = seconds_until(now, end) / 3600

        command = [
            sys.executable, "-m", "minerva.training.trainer",
            "--data", args.data,
            "--out", args.out,
            "--steps", str(args.steps),
            "--threads", str(args.threads),
            "--checkpoint-interval", str(args.checkpoint_interval),
        ]
        # Resume from the rolling checkpoint when there is one. `last.pt`, not
        # `best.pt`: best lags actual progress (it only moves when validation
        # improves), and resuming from it silently discards real steps.
        last = Path(args.out) / "last.pt"
        if last.exists():
            command += ["--resume", str(last)]
        if budget:
            command += ["--max-hours", f"{budget:.2f}"]
        command += args.extra

        print(f"[{now:%Y-%m-%d %H:%M}] training"
              + (f" for {budget:.1f}h" if budget else " (no window limit)"), flush=True)
        result = subprocess.run(command)
        if result.returncode != 0:
            print(f"trainer exited {result.returncode}; stopping", flush=True)
            return result.returncode
        if args.all_day:
            return 0


if __name__ == "__main__":
    sys.exit(main())
