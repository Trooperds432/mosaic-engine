"""Freshness and identity checks for the evidence that can unlock signals. Pure functions so they can be unit-tested.

A stored audit or forward gate only counts when it belongs to the data and model being used now: anything missing,
unreadable, stale or produced for another dataset / model recipe blocks new LONG signals.
"""
from datetime import datetime, timezone
import math

AUDIT_MAX_AGE_HOURS = 12        # the audit runs in the same nightly job, minutes before signals.py
GATE_MAX_AGE_HOURS = 4 * 24     # paper_trading.py runs after signals.py, so the gate is from the previous run (weekends)
BLOCKING_TESTS = ("truncation_invariance", "label_alignment", "duplicates_timestamps", "historical_universe")


def joint_test_failures(result):
    """Both joint tests must have actual common-sample results; SPA retains its existing threshold."""
    if not isinstance(result, dict) or result.get('error'):
        return ['Hansen SPA / White Reality Check not computed']
    failures = []
    for key in ['spa_p', 'reality_check_p']:
        value = result.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
            failures.append(f'{key} not computed or invalid')
    if not isinstance(result.get('periods'), int) or result['periods'] < 20:
        failures.append('joint-test common sample has fewer than 20 periods')
    if not failures and result['spa_p'] > .10:
        failures.append(f"Hansen SPA p={result['spa_p']:.2f} > 0.10")
    return failures


def live_review_ready(executed, shadow, horizon=20, min_sessions=63, min_trades=20, min_t=2.0):
    """Human-review recommendation only, requiring executed AND matching forward evidence.

    Each executed model must pass independently. Unknown/event-only models have
    no matching cross-sectional IC record and cannot be promoted by this rule.
    """
    def at_least(value, threshold, strict=False):
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and (value > threshold if strict else value >= threshold))

    failures = []
    books = executed.get("by_model") or {}
    if not books:
        failures.append("executed trades lack model attribution")
    for model, book in books.items():
        if book.get("horizons") != [horizon]:
            failures.append(f"{model}: executed horizon does not match forward evidence")
        if not (at_least(book.get("n"), min_trades) and at_least(book.get("sessions_since_first"), min_sessions)
                and at_least(book.get("mean_net_excess"), 0, strict=True) and at_least(book.get("t"), min_t)):
            failures.append(f"{model}: insufficient positive executed evidence")
        prefix = "mosaic-xs-ridge-"
        profile = model[len(prefix):] if model.startswith(prefix) else None
        fwd = shadow.get(f"{profile}_h{horizon}", {}) if profile else {}
        if not (at_least(fwd.get("scored_dates"), min_sessions)
                and at_least(fwd.get("forward_ic_mean"), 0, strict=True)
                and at_least(fwd.get("forward_ic_t"), min_t)
                and at_least(fwd.get("non_overlapping_cohorts"), 3)
                and at_least(fwd.get("top_quintile_net_mean"), 0, strict=True)):
            failures.append(f"{model}: matching positive forward evidence missing")
    return not failures, failures


def _age_hours(stamp, now):
    t = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return (now - t).total_seconds() / 3600


def check_audit(doc, dataset_version, now=None, profile=None):
    """(ok, reason). doc is the parsed reports/data_audit.json or None."""
    now = now or datetime.now(timezone.utc)
    if not isinstance(doc, dict) or not doc:
        return False, "data audit not run or unreadable"
    if profile is not None and doc.get('profile') != profile:
        return False, f"data audit profile {doc.get('profile')} does not match {profile}"
    try:
        age = _age_hours(doc["generated_at_utc"], now)
    except (KeyError, ValueError, TypeError):
        return False, "data audit has no valid timestamp"
    if age > AUDIT_MAX_AGE_HOURS or age < -0.1:
        return False, f"data audit is {age:.0f}h old (max {AUDIT_MAX_AGE_HOURS}h)"
    if doc.get("dataset_version") != dataset_version:
        return False, f"data audit was run on {doc.get('dataset_version')}, current data is {dataset_version}"
    blocking = doc.get("blocking") or {}
    if not isinstance(blocking, dict):
        return False, 'data audit blocking checks are invalid'
    missing = [k for k in BLOCKING_TESTS if k not in blocking]
    if missing:
        return False, "data audit incomplete: " + ", ".join(missing)
    bad = [k for k in BLOCKING_TESTS if blocking.get(k) is not True]
    if bad:
        return False, "data audit failed: " + ", ".join(bad)
    return True, None


def check_forward_gate(doc, profile, horizon, model_run_id, now=None, recipe_id=None):
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
    if recipe_id is not None and (doc.get("recipe_ids") or {}).get(profile) != recipe_id:
        return "UNVALIDATED", ["forward record belongs to another frozen recipe"], none
    g = (doc.get("profiles") or {}).get(profile, {}).get(str(horizon))
    if not g:
        return "UNVALIDATED", ["forward record not yet scored"], none
    status = g.get("status") if g.get("status") in ("UNVALIDATED", "PAPER_TEST_ELIGIBLE") else "UNVALIDATED"
    return status, list(g.get("failed_checks") or []), {"forward_ic": g.get("forward_ic"), "forward_ic_se": g.get("forward_ic_se")}
