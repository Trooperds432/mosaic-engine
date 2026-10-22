"""Prospective frozen-rule event evidence from immutable emitted signals; never a backfilled event study."""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config as C
import recipe
from provenance import sessions_between


def eligible_observations(records, model):
    start = pd.Timestamp(model["forward_start_utc"])
    expected_run = model["run_id"]
    kept = {}
    for signal in records:
        if (signal.get("model_id") != "insider-event" or signal.get("model_version") != model["model_version"]
                or signal.get("provenance", {}).get("research_run_id") != expected_run):
            continue
        generated = pd.Timestamp(signal["generated_at_utc"])
        if generated < start:
            continue
        sid = signal["signal_id"]
        if sid not in kept or generated < pd.Timestamp(kept[sid]["generated_at_utc"]):
            kept[sid] = signal
    return sorted(kept.values(), key=lambda s: s["generated_at_utc"])


def evaluate(records, model, market):
    from insider_model import clustered_mean
    from historical_data import active_bundle, security_for_ticker
    rows, previous = [], {}
    for s in eligible_observations(records, model):
        known = pd.Timestamp(s["generated_at_utc"]).tz_convert("America/New_York").tz_localize(None).normalize()
        entry = int(market.cal.searchsorted(known, side="right"))
        sym = s["asset"]["security_id"] if active_bundle() else security_for_ticker(s["asset"]["ticker"], known)
        j = market.col.get(sym)
        if j is None or entry == 0 or entry + model["horizon"] >= len(market.cal):
            continue
        issuer = s["asset"]["security_id"]
        if issuer in previous and entry - previous[issuer] < 20:
            continue
        ar = market.ar(j, entry, model["horizon"])
        if not np.isfinite(ar):
            continue
        rows.append({"signal_id": s["signal_id"], "entry_date": market.cal[entry], "net": ar - model["cost"]})
        previous[issuer] = entry
    values = pd.DataFrame(rows, columns=["signal_id", "entry_date", "net"])
    stats = clustered_mean(values["net"], values["entry_date"])
    elapsed = sessions_between(pd.Timestamp(model["forward_start_utc"]).date(), market.cal[-1].date())
    failed = []
    if elapsed < 63:
        failed.append(f"forward elapsed sessions {elapsed}/63")
    if len(values) < 20:
        failed.append(f"completed non-overlapping events {len(values)}/20")
    if not ((stats.get("mean") or 0) > 0 and (stats.get("t") or 0) >= 2):
        failed.append("net abnormal return not positive with month-clustered t >= 2")
    return {"status": "PAPER_TEST_ELIGIBLE" if not failed else "UNVALIDATED", "failed_checks": failed,
            "completed_events": len(values), "elapsed_sessions": elapsed, "net_statistics": stats}, values


def run():
    from insider_model import Market
    model = json.loads((C.PROC / "insider_model.json").read_text())
    ok, why = recipe.check(model)
    if not ok:
        raise ValueError(why)
    records = []
    for file in sorted((C.SIGNALS / "log").glob("*.jsonl")):
        records.extend(json.loads(line) for line in file.read_text().splitlines() if line.strip())
    gate, values = evaluate(records, model, Market())
    doc = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "recipe_id": model["recipe_id"],
           "model_run_id": model["run_id"], "gate": gate}
    temporary = C.PROC / "insider_forward_gate.tmp"
    temporary.write_text(json.dumps(doc, indent=2, allow_nan=False))
    temporary.replace(C.PROC / "insider_forward_gate.json")
    values.to_parquet(C.REPORTS / "insider_forward_events.parquet", index=False)
    return doc


if __name__ == "__main__":
    _lock = C.hold_lock()
    print(json.dumps(run(), indent=2))
