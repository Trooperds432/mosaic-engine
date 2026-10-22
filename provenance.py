"""Dataset fingerprint and the NYSE trading calendar, shared by audit.py, signals.py, paper_trading.py and the bots."""
import hashlib
from datetime import date, timedelta
from functools import lru_cache

import numpy as np

import config as C


def dataset_version():
    """Hash of the size and mtime of the inputs every signal depends on: changes whenever any of them is rewritten."""
    parts = []
    for p in [C.RAW / "prices" / f"{C.BENCHMARK}.parquet", C.RAW / "facts" / "companyfacts.parquet",
              C.RAW / "insider_recent" / "form4_recent.parquet", C.PROC / "inst_ownership_cusip.parquet", C.PROC / "macro_pit.parquet",
              C.PROC / "short_interest.parquet", *sorted((C.RAW / "insider").glob("*.parquet"))[-2:]]:
        if p.exists():
            st = p.stat()
            parts.append(f"{p.name}:{st.st_size}:{int(st.st_mtime)}")
    return "ds-" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def _observed(d):
    return d - timedelta(days=1) if d.weekday() == 5 else d + timedelta(days=1) if d.weekday() == 6 else d


def _nth_weekday(y, m, wd, n):
    d = date(y, m, 1)
    d += timedelta(days=(wd - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_weekday(y, m, wd):
    d = date(y, m + 1, 1) - timedelta(days=1) if m < 12 else date(y, 12, 31)
    return d - timedelta(days=(d.weekday() - wd) % 7)


def _easter(y):
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    return date(y, month, (h + l - 7 * m + 114) % 31 + 1)


@lru_cache(maxsize=None)
def nyse_holidays(y):
    """Full-day NYSE closures under the current rules (New Year's Day is not moved back to a Friday)."""
    hol = {_nth_weekday(y, 1, 0, 3), _nth_weekday(y, 2, 0, 3), _easter(y) - timedelta(days=2), _last_weekday(y, 5, 0),
           _observed(date(y, 7, 4)), _nth_weekday(y, 9, 0, 1), _nth_weekday(y, 11, 3, 4), _observed(date(y, 12, 25))}
    ny = date(y, 1, 1)
    if ny.weekday() != 5:
        hol.add(_observed(ny))
    if y >= 2022:
        hol.add(_observed(date(y, 6, 19)))
    return frozenset(hol)


def holidays(start, end):
    return sorted(d for y in range(start.year, end.year + 1) for d in nyse_holidays(y))


def sessions_between(a, b):
    """NYSE sessions in [a, b) (numpy busday_count with exchange holidays)."""
    a, b = np.datetime64(a, "D"), np.datetime64(b, "D")
    lo, hi = (a, b) if a <= b else (b, a)
    hol = [np.datetime64(d) for d in holidays(lo.astype(object), hi.astype(object))]
    return int(np.busday_count(a, b, holidays=hol))
