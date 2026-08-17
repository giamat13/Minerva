"""Clock and calendar tools.

A language model has no clock: its sense of "now" is frozen at training time.
These tools give it a real one, backed by the system clock and the IANA time
zone database that ships with Python (:mod:`zoneinfo`).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..base import tool

__all__ = ["current_time", "days_between"]


@tool(name="current_time")
def current_time(timezone: str = "UTC") -> str:
    """Get the current date and time in a given time zone.

    Use this whenever the answer depends on what "now" is - today's date, the
    current time somewhere, how long until a deadline. Never guess the date.

    Args:
        timezone: An IANA time zone name such as "UTC", "Asia/Jerusalem",
            "Europe/London" or "America/New_York". Defaults to UTC.
    """
    try:
        zone = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(
            f"unknown time zone {timezone!r}; use an IANA name such as "
            f"'UTC' or 'Asia/Jerusalem'"
        ) from exc

    now = datetime.now(zone)
    offset = now.utcoffset() or timedelta(0)
    total_minutes = int(offset.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    offset_text = f"UTC{sign}{abs(total_minutes) // 60:02d}:{abs(total_minutes) % 60:02d}"

    return (
        f"{now.strftime('%Y-%m-%d %H:%M:%S')} ({now.strftime('%A')}) "
        f"in {timezone} [{offset_text}]"
    )


@tool(name="days_between")
def days_between(start_date: str, end_date: str) -> str:
    """Count the number of days between two calendar dates.

    Args:
        start_date: The earlier date, in ISO format (YYYY-MM-DD).
        end_date: The later date, in ISO format (YYYY-MM-DD).
    """
    start = _parse_date(start_date, "start_date")
    end = _parse_date(end_date, "end_date")
    delta = (end - start).days
    direction = "after" if delta >= 0 else "before"
    return (
        f"{abs(delta)} day(s); {end.isoformat()} is {direction} {start.isoformat()}"
    )


def _parse_date(value: str, field: str) -> datetime:
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError(
            f"{field}={value!r} is not a valid ISO date; expected YYYY-MM-DD"
        ) from exc
