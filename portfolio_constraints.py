"""Explicit external research constraints. Missing limits never become invented defaults."""
import json
import math
from pathlib import Path

FIELDS = ("gross_limit", "sector_cap", "turnover_cap", "beta_min", "beta_max", "max_participation", "risk_aversion", "cost_horizon_sessions")


def load(path, bot_limits):
    doc = json.loads(Path(path).read_text())
    return validate(doc, bot_limits)


def validate(doc, bot_limits):
    if not isinstance(doc, dict) or not isinstance(doc.get("source"), str) or not doc["source"].strip():
        raise ValueError("external constraints require a source")
    doc = dict(doc)
    # Preserve the newer bot-import integration. An additional approved research
    # policy can tighten a bot cap, but cannot relax a supplied bot restriction.
    for policy_key, bot_key in (("sector_cap", "sector_cap"), ("turnover_cap", "turnover_cap"),
                                ("beta_max", "max_beta"), ("max_participation", "max_adv_participation")):
        supplied = bot_limits.get(bot_key)
        if supplied is not None:
            if isinstance(supplied, bool) or not isinstance(supplied, (int, float)) or not math.isfinite(supplied):
                raise ValueError(f"invalid imported bot limit {bot_key}")
            if isinstance(doc.get(policy_key), (int, float)) and not isinstance(doc[policy_key], bool):
                doc[policy_key] = min(doc[policy_key], supplied)
    for key in FIELDS:
        v = doc.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"external constraint {key} must be finite")
    if not (0 < doc["gross_limit"] <= 1 and 0 < doc["sector_cap"] <= 1
            and 0 <= doc["turnover_cap"] <= 2 and 0 < doc["max_participation"] <= 1
            and doc["beta_min"] <= doc["beta_max"] and doc["risk_aversion"] > 0
            and doc["cost_horizon_sessions"] > 0):
        raise ValueError("external constraints outside supported long-only unleveraged ranges")
    if bot_limits.get("long_only") is not True or bot_limits.get("leverage") is not False:
        raise ValueError("optimizer supports externally confirmed long-only, unleveraged accounts")
    return doc
