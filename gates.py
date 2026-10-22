"""Freshness and identity checks for the evidence that can unlock signals. Pure functions so they can be unit-tested.

A stored audit or forward gate only counts when it belongs to the data and model being used now: anything missing,
unreadable, stale or produced for another dataset / model recipe blocks new LONG signals.
"""
from datetime import datetime, timezone

AUDIT_MAX_AGE_HOURS = 12        # the audit runs in the same nightly job, minutes before signals.py
GATE_MAX_AGE_HOURS = 4 * 24     # paper_trading.py runs after signals.py, so the gate is from the previous run (weekends)
BLOCKING_TESTS = ("truncation_invariance", "label_alignment", "duplicates_timestamps")


def _age_hours(stamp, now):
    t = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return (now - t).total_seconds() / 3600


def check_audit(doc, dataset_version, now=None):
    """(ok, reason). doc is the parsed reports/data_audit.json or None."""
    now = now or datetime.now(timezone.utc)
    if not doc:
        return False, "data audit not run or unreadable"
    try:
        age = _age_hours(doc["generated_at_utc"], now)
    except (KeyError, ValueError, TypeError):
        return False, "data audit has no valid timestamp"
    if age > AUDIT_MAX_AGE_HOURS or age < -0.1:
        return False, f"data audit is {age:.0f}h old (max {AUDIT_MAX_AGE_HOURS}h)"
    if doc.get("dataset_version") != dataset_version:
        return False, f"data audit was run on {doc.get('dataset_version')}, current data is {dataset_version}"
    blocking = doc.get("blocking") or {}
    missing = [k for k in BLOCKING_TESTS if k not in blocking]
    if missing:
        return False, "data audit incomplete: " + ", ".join(missing)
    bad = [k for k in BLOCKING_TESTS if blocking.get(k) is not True]
    if bad:
        return False, "data audit failed: " + ", ".join(bad)
    return True, None


def check_forward_gate(doc, profile, horizon, model_run_id, now=None):
    """(status, failed_checks, stats). Anything but a fresh gate for this profile, horizon and model recipe is UNVALIDATED."""
    now = now or datetime.now(timezone.utc)
    none = {"forward_ic": None, "forward_ic_se": None}
    if not doc:
        return "UNVALIDATED", ["forward gate missing or unreadable"], none
    try:
        age = _age_hours(doc["generated_at_utc"], now)
    except (KeyError, ValueError, TypeError):
        return "UNVALIDATED", ["forward gate has no valid timestamp"], none
    if age > GATE_MAX_AGE_HOURS or age < -0.1:
        return "UNVALIDATED", [f"forward gate is {age:.0f}h old (max {GATE_MAX_AGE_HOURS}h)"], none
    rid = (doc.get("model_run_ids") or {}).get(profile)
    if rid != model_run_id:
        return "UNVALIDATED", [f"forward gate is for model {rid}, current model is {model_run_id}"], none
    g = (doc.get("profiles") or {}).get(profile, {}).get(str(horizon))
    if not g:
        return "UNVALIDATED", ["forward record not yet scored"], none
    status = g.get("status") if g.get("status") in ("UNVALIDATED", "PAPER_TEST_ELIGIBLE") else "UNVALIDATED"
    return status, list(g.get("failed_checks") or []), {"forward_ic": g.get("forward_ic"), "forward_ic_se": g.get("forward_ic_se")}
