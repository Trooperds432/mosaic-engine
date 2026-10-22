"""One cumulative policy; historical failure cannot be overridden by forward performance."""
PAPER = "PAPER_TEST_ELIGIBLE"


def cumulative(historical, forward, *, data_ok, recipe_ok):
    failures = []
    if not data_ok:
        failures.append("current point-in-time/universe audit failed")
    if not recipe_ok:
        failures.append("recipe not frozen or changed")
    for name, stage in [("historical", historical), ("forward", forward)]:
        if not isinstance(stage, dict) or stage.get("status") not in (PAPER, "LIVE_REVIEW_ELIGIBLE") or stage.get("failed_checks"):
            failures.append(f"{name} validation has not passed")
            if isinstance(stage, dict):
                failures.extend(f"{name}: {reason}" for reason in stage.get("failed_checks", []))
    return {"status": PAPER if not failures else ("INSUFFICIENT_DATA" if not data_ok else "UNVALIDATED"), "failed_checks": failures}


def execution_target_allowed(doc):
    """Paper broker also requires a validated recipe; research shadow books remain offline."""
    policy = doc.get("deployment", {})
    return (doc.get("mode") == "VALIDATED_PAPER" and policy.get("status") == PAPER
            and not policy.get("failed_checks") and isinstance(doc.get("recipe_id"), str)
            and len(doc["recipe_id"]) == 64 and bool(doc.get("names")))
