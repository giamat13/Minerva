"""Reclaim RAM with the user's QuickClean script, so training can start.

Memory is the binding constraint on this machine, not CPU: the scheduler's own
log showed 299 "waiting, not enough free RAM" entries against 10 training
starts. QuickClean trims every process's working set back to Windows, purges
the standby list and shrinks the file-system cache, which is exactly the
memory `available_gb()` counts.

    python scripts/free_memory.py

Fetched from the URL below and cached under data/cache/. Re-downloaded when
the cache is older than --max-age-hours, so a fix upstream arrives without
this having to be edited, and the SHA-256 of whatever ran is logged - a
silently changed script should be visible after the fact.

**Why this calls run_quickclean() rather than the module's main().** Two
things in main() are correct for a person at a keyboard and wrong for an
unattended task:

* It pauses on `msvcrt.getch()` when finished, which would hang the scheduler
  forever. `run_quickclean(finish=False)` does the same work without it.
* When not already elevated it relaunches itself through UAC, which would pop
  a consent dialog at whoever is using the machine - the precise interruption
  the whole presence-aware design exists to avoid.

Running unelevated means the standby purge and file-cache trim are skipped
(they need SeProfileSingleProcessPrivilege and SeIncreaseQuotaPrivilege). The
working-set trim, which is the part that actually raises available memory,
still runs for every process this user owns.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import io
import sys
import time
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path

SOURCE_URL = (
    "https://raw.githubusercontent.com/giamat13/100RAMoptimal/"
    "refs/heads/main/main.py"
)
CACHE = Path("data/cache/ramoptimal/main.py")


class _Mem(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
        (name, ctypes.c_ulonglong)
        for name in (
            "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
            "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual",
        )
    ]


def available_gb() -> float | None:
    """Physical memory a new process could take, in GB. See train_when_away."""
    if not sys.platform.startswith("win"):
        return None
    status = _Mem()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status.ullAvailPhys / 2**30


def fetch(url: str = SOURCE_URL, cache: Path = CACHE, max_age_hours: float = 24.0) -> Path | None:
    """Download the script, reusing a recent cache. None if it cannot be had."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    fresh = (
        cache.exists()
        and cache.stat().st_size > 0
        and (time.time() - cache.stat().st_mtime) < max_age_hours * 3600
    )
    if fresh:
        return cache
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = response.read()
    except Exception as exc:
        # A cached copy is better than nothing when the network is down.
        if cache.exists() and cache.stat().st_size > 0:
            print(f"  download failed ({type(exc).__name__}); using the cached copy",
                  flush=True)
            return cache
        print(f"  could not fetch {url}: {type(exc).__name__}", flush=True)
        return None
    if b"def run_quickclean" not in payload:
        print("  refusing to run: fetched file has no run_quickclean()", flush=True)
        return None
    cache.write_bytes(payload)
    return cache


def free_memory(max_age_hours: float = 24.0, verbose: bool = True) -> float:
    """Run QuickClean's reclaim steps. Returns GB freed (0.0 if it could not run).

    Never raises: this is an optimisation on the path to training, and a
    failure here must degrade to "trained a bit later", never to "the run
    stopped".
    """
    path = fetch(max_age_hours=max_age_hours)
    if path is None:
        return 0.0

    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    before = available_gb()
    if before is None:
        return 0.0

    try:
        spec = importlib.util.spec_from_file_location("quickclean", path)
        if spec is None or spec.loader is None:
            return 0.0
        module = importlib.util.module_from_spec(spec)
        # Importing defines functions and binds ctypes prototypes; the module
        # guards its own entry point behind __main__, so nothing runs yet.
        spec.loader.exec_module(module)
    except Exception as exc:
        print(f"  could not load the cleanup script ({type(exc).__name__}: {exc})",
              flush=True)
        return 0.0

    # QuickClean's three memory steps, called individually rather than through
    # run_quickclean(). Two reasons, both found by running it:
    #
    #  * run_quickclean() starts by emptying temp directories, and unelevated
    #    that raises PermissionError on C:\WINDOWS\Temp from os.listdir, which
    #    is outside its per-entry error handling - so it aborts before reaching
    #    the memory work at all.
    #  * Deleting temp files frees disk, not RAM. It is not what this is for.
    #
    # This is QuickClean's reclaim, never DeepClean: DeepClean also runs
    # `winget upgrade --all`, which is emphatically not something to trigger
    # unattended behind someone who is using the machine.
    ran = 0
    for name in ("trim_working_sets", "purge_standby_list", "trim_file_cache"):
        step = getattr(module, name, None)
        if step is None:
            continue
        try:
            # The module prints progress; that is noise in the training log.
            with redirect_stdout(io.StringIO()):
                step()
            ran += 1
        except Exception as exc:
            # Unelevated, the standby purge and cache trim lack their
            # privileges and may fail. The working-set trim is the one that
            # matters and works either way, so a failure here is not fatal.
            print(f"  {name} skipped ({type(exc).__name__})", flush=True)
    if not ran:
        return 0.0

    after = available_gb() or before
    freed = after - before
    if verbose:
        elevated = bool(getattr(module, "is_admin", lambda: False)())
        print(
            f"  freed {freed:+.2f} GB ({before:.1f} -> {after:.1f} GB)"
            f"{'' if elevated else ', unelevated so standby purge was skipped'}"
            f" [script {digest}]",
            flush=True,
        )
    return freed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-age-hours", type=float, default=24.0,
                        help="re-download the script when the cache is older than this")
    args = parser.parse_args(argv)
    before = available_gb()
    print(f"available before: {before:.2f} GB" if before else "not Windows; nothing to do")
    free_memory(max_age_hours=args.max_age_hours)
    return 0


if __name__ == "__main__":
    sys.exit(main())
