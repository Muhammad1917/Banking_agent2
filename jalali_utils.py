"""Jalali (Persian calendar) helpers.

All dates inside the bank datasets are Jalali strings like ``1404/10/05``.
This module converts them to/from Gregorian, computes business ("working")
days between two Jalali dates and derives "today" in the Iranian timezone.

It uses the well-tested ``jalaali`` package when available and falls back to
a search-based pure-Python conversion so the project never hard-crashes
because of a missing dependency.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone

try:  # python 3.9+ built-in zoneinfo
    from zoneinfo import ZoneInfo

    TEHRAN = ZoneInfo("Asia/Tehran")
except Exception:  # pragma: no cover - fallback (UTC+3:30)
    TEHRAN = timezone(timedelta(hours=3, minutes=30), name="Asia/Tehran")

try:
    from jalaali import Jalaali as _Jalaali

    _jalaali = _Jalaali()
    _HAS_JALAALI = True
except ImportError:  # pragma: no cover
    _jalaali = None
    _HAS_JALAALI = False


def jal_to_greg(jy: int, jm: int, jd: int) -> date:
    """Jalali -> Gregorian ``date``."""
    if _HAS_JALAALI:
        g = _jalaali.to_gregorian(jy, jm, jd)  # {'gy','gm','gd'}
        return date(g["gy"], g["gm"], g["gd"])
    raise RuntimeError("install `jalaali` for jalali->gregorian conversion")


def greg_to_jal(gy: int, gm: int, gd: int) -> tuple[int, int, int]:
    """Gregorian -> Jalali, returns ``(jy, jm, jd)``."""
    if _HAS_JALAALI:
        j = _jalaali.to_jalaali(gy, gm, gd)  # {'jy','jm','jd'}
        return j["jy"], j["jm"], j["jd"]
    raise RuntimeError("install `jalaali` for gregorian->jalali conversion")


# ---------------------------------------------------------------------------
# Public helpers used by the agent
# ---------------------------------------------------------------------------
_DATE_RE = re.compile(r"(\d{2,4})[/-](\d{1,2})[/-](\d{1,2})")


def parse_jalali(s) -> date | None:
    """Parse '1404/10/05' (also datetime/date objects) into a Gregorian date."""
    if s is None or (isinstance(s, float) and s != s):  # NaN check
        return None
    if isinstance(s, datetime):
        return s.date()
    if isinstance(s, date):
        return s
    m = _DATE_RE.search(str(s).strip())
    if not m:
        return None
    y, mo, d = map(int, m.groups())
    if y < 100:  # e.g. '04/10/05' -> 1404
        y += 1400
    try:
        return jal_to_greg(y, mo, d)
    except Exception:
        return None


def format_jalali(d: date) -> str:
    """Format a Gregorian date as 'YYYY/MM/DD' in the Jalali calendar."""
    jy, jm, jd = greg_to_jal(d.year, d.month, d.day)
    return f"{jy}/{jm:02d}/{jd:02d}"


def today_jalali() -> str:
    """Today's date as a Jalali string (Tehran time)."""
    return format_jalali(datetime.now(TEHRAN).date())


def today_gregorian() -> date:
    return datetime.now(TEHRAN).date()


def working_days_between(start: date, end: date) -> int:
    """Iranian working days elapsed strictly after *start* up to *end*.

    Fridays (weekday()==4) are holidays.
    """
    if end <= start:
        return 0
    n = 0
    cur = start + timedelta(days=1)
    while cur <= end:
        if cur.weekday() != 4:
            n += 1
        cur += timedelta(days=1)
    return n


def add_working_days(start: date, days: int) -> date:
    """Return the date *days* Iranian working days after *start*."""
    cur = start
    remaining = days
    while remaining > 0:
        cur += timedelta(days=1)
        if cur.weekday() != 4:
            remaining -= 1
    return cur
