"""Shared credential lookup and conservative Alpha Vantage request accounting."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import config as C


def key(name, env_path=Path("/root/.hermes/.env")):
    if os.environ.get(name):
        return os.environ[name]
    try:
        for line in Path(env_path).read_text().splitlines():
            k, sep, value = line.strip().partition("=")
            if sep and k == name:
                return value.strip().strip('"').strip("'") or None
    except OSError:
        return None
    return None


def reserve_alpha_request(limit=24):
    """Reserve before sending, including failed requests; shares the backfill ledger.

    O_EXCL serialises concurrent reservations on Windows and Linux. A stale lock
    after a crash blocks spending until inspected; it is never silently removed.
    Other applications using the same key must also respect the shared allowance.
    """
    root = C.RAW / "collect"
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".av_budget.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError("Alpha Vantage budget lock busy/stale; no request sent") from exc
    try:
        os.close(fd)
        ledger = root / "av_calls.json"
        calls = json.loads(ledger.read_text()) if ledger.exists() else {}
        day = datetime.now(timezone.utc).date().isoformat()
        used = calls.get(day, 0)
        if isinstance(used, bool) or not isinstance(used, int) or used < 0:
            raise ValueError("invalid Alpha Vantage request ledger")
        if used >= limit:
            raise ValueError(f"Alpha Vantage shared daily budget exhausted ({used}/{limit})")
        calls[day] = used + 1
        temp = ledger.with_suffix(".tmp")
        temp.write_text(json.dumps(calls))
        temp.replace(ledger)
        return used + 1
    finally:
        lock.unlink()
