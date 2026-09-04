"""One-command answer to "how is the training going, and is anything broken?"

Reports both runs - the local nightly task and the CI chain - because
CLAUDE.md section 9 says progress is compared by step count from
training_log.jsonl, never by file timestamp and never by best.pt (whose step
lags real progress, since it only moves when validation improves).

    python scripts/training_status.py

Exits non-zero when something needs attention, so it is also usable as a
check rather than only as a report.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

#: Tokens consumed per optimiser step: micro_batch x grad_accum x seq_len.
TOKENS_PER_STEP = 8192
#: A Chinchilla-correct pass over the v0.5.0 corpus (~518M tokens).
TARGET_STEPS = 63_232


def read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a row torn by a kill mid-write is not a failure
    return rows


def describe_local(out_dir: Path) -> list[str]:
    problems: list[str] = []
    print("LOCAL")
    log = out_dir / "training_log.jsonl"
    rows = read_log(log)
    if not rows:
        print(f"  no training log yet at {log}")
        print("  (expected until the first night has run)")
        return problems

    step = max(r.get("step", 0) for r in rows)
    losses = [r for r in rows if r.get("val_loss") is not None]
    pct = step / TARGET_STEPS * 100
    print(f"  step        {step:,} of {TARGET_STEPS:,}  ({pct:.1f}%)")
    print(f"  tokens      {step * TOKENS_PER_STEP / 1e6:,.0f}M")
    if losses:
        first, last = losses[0], losses[-1]
        print(f"  val loss    {last['val_loss']:.4f}"
              f"  (from {first['val_loss']:.4f} at step {first.get('step', 0):,})")
        if len(losses) >= 3 and last["val_loss"] > losses[-3]["val_loss"]:
            problems.append("validation loss has risen over the last few evals")

    age = datetime.now() - datetime.fromtimestamp(log.stat().st_mtime)
    print(f"  last write  {fmt_age(age)} ago")
    # Training is presence-driven, so a quiet log only means something is wrong
    # when nobody is at the computer. Someone sitting here is the *reason* it
    # is quiet, and flagging that would make the check useless.
    here = presence()
    if here is True:
        print("  presence    someone is at the computer (training yields)")
    elif here is False:
        print("  presence    nobody at the computer (training may run)")
        if age > timedelta(hours=2):
            problems.append(
                f"nobody is at the computer but the log has not moved in "
                f"{fmt_age(age)}"
            )
    else:
        print("  presence    unknown (Screen Time API unreachable; "
              "falling back to the 00:00-07:15 window)")
    return problems


def presence(url: str = "http://127.0.0.1:47834/status") -> bool | None:
    """Whether anyone is at the computer, per Screen Time's local API.

    None when the API cannot be reached, which is not the same as "away" -
    see scripts/train_when_away.py, which makes the same distinction.
    """
    import json as _json
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=4) as response:
            payload = _json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None
    value = payload.get("present")
    return bool(value) if isinstance(value, bool) else None


def fmt_age(delta: timedelta) -> str:
    seconds = int(delta.total_seconds())
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h {seconds % 3600 // 60}m"
    return f"{seconds // 86400}d {seconds % 86400 // 3600}h"


def describe_task() -> list[str]:
    problems: list[str] = []
    if not shutil.which("powershell"):
        return problems
    query = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "$t = Get-ScheduledTask -TaskName Minerva-Background-Training "
         "-ErrorAction SilentlyContinue; "
         "if (-not $t) { 'missing' } else { "
         "$i = Get-ScheduledTaskInfo -TaskName Minerva-Background-Training; "
         "\"$($t.State)|$($i.NextRunTime)|$($i.LastTaskResult)\" }"],
        capture_output=True, text=True, timeout=60,
    )
    line = query.stdout.strip()
    if not line or line == "missing":
        problems.append("the background training task is not registered "
                        "(run scripts/install_training_task.ps1)")
        print("  task        NOT REGISTERED")
        return problems
    state, _, rest = line.partition("|")
    next_run, _, last_result = rest.partition("|")
    print(f"  task        {state}, next run {next_run.strip() or 'unknown'}")
    if state.strip() == "Disabled":
        problems.append("the background training task is disabled")
    # Benign results: 267009 "currently running", 267011 "has not yet run", and
    # 2147946720 (0x800710E0) "the operator or administrator has refused the
    # request" - which is MultipleInstances=IgnoreNew declining to start a
    # second trainer while one is already going. The 30-minute self-heal
    # trigger produces that every half hour by design, so treating it as a
    # failure would report a broken run every single day.
    if last_result.strip() not in ("", "0", "267009", "267011", "2147946720"):
        problems.append(
            f"the last training run exited with code {last_result.strip()} - "
            f"check data/training.log"
        )
    return problems


def gh_env() -> dict[str, str]:
    """gh's environment, with a token borrowed from git's credential store.

    `gh auth login` is not always done on a machine where `git push` works,
    because git keeps its own credentials. Reusing them means the status check
    works wherever pushing does, without asking anyone to log in twice.
    """
    env = dict(os.environ)
    if env.get("GH_TOKEN") or env.get("GITHUB_TOKEN"):
        return env
    try:
        filled = subprocess.run(
            ["git", "credential", "fill"],
            input="protocol=https\nhost=github.com\n\n",
            capture_output=True, text=True, timeout=30,
        )
        for line in filled.stdout.splitlines():
            if line.startswith("password="):
                env["GH_TOKEN"] = line.partition("=")[2]
                break
    except (OSError, subprocess.SubprocessError):
        pass
    return env


def describe_ci() -> list[str]:
    problems: list[str] = []
    print("\nCI")
    gh = shutil.which("gh") or r"C:\Program Files\GitHub CLI\gh.exe"
    if not Path(gh).exists():
        print("  gh CLI not found; skipping")
        return problems
    result = subprocess.run(
        [gh, "run", "list", "-R", "giamat13/Minerva", "--workflow", "train.yml",
         "--limit", "5", "--json", "databaseId,status,conclusion,createdAt"],
        capture_output=True, text=True, timeout=120, env=gh_env(),
    )
    if result.returncode != 0:
        print(f"  could not query GitHub: {result.stderr.strip()[:120]}")
        # Not knowing is itself worth flagging: the CI half of the run would
        # otherwise look fine simply because nothing could be checked.
        problems.append("could not query CI status (gh auth login, or set GH_TOKEN)")
        return problems
    runs = json.loads(result.stdout or "[]")
    if not runs:
        print("  no runs found")
        return problems

    active = [r for r in runs if r["status"] in ("queued", "in_progress")]
    print(f"  latest      {runs[0]['databaseId']}  {runs[0]['status']}"
          f"  {runs[0].get('conclusion') or ''}")
    if active:
        print(f"  in flight   {len(active)} run(s) - the chain is alive")
    else:
        recent_failure = runs[0].get("conclusion") == "failure"
        if recent_failure:
            problems.append(
                f"the newest CI run ({runs[0]['databaseId']}) failed and nothing is "
                f"queued - the chain has stopped"
            )
        else:
            problems.append("no CI run is queued or in progress - the chain may have "
                            "finished or stalled")
    return problems


def main() -> int:
    print(f"Minerva training status - {datetime.now():%Y-%m-%d %H:%M}\n")
    problems = describe_local(Path("checkpoints/swift-v05"))
    problems += describe_task()
    problems += describe_ci()

    print()
    if problems:
        print(f"NEEDS ATTENTION ({len(problems)}):")
        for item in problems:
            print(f"  - {item}")
        return 1
    print("No problems detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
