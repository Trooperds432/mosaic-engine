"""Insider event study v3 (prereg/insider_event_v3.json, registered 2026-11-19T17:58:32Z).

Reuses the v2 machinery in insider_study.py unchanged (same numeric recipe, verified at runtime) and adds the v3
scope: S&P 500 members from the LSEG bundle, entries 2016-01-04..2026-10-30, an S&P-only coverage audit with
unattested sources reported as limitations, exclusion of events with price gaps, exact acceptance times only in the
primary sample, no estimate control in the primary specification, and exclusion of events near earnings / material
agreement / M&A 8-Ks. Research only: never grants trading authority.

usage: python insider_study_v3.py --out DIR --acceptances data/processed/insider_acceptances.csv
                                  [--sensitivity-estimates data/processed/estimate_vintages_sensitivity.parquet]
"""
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import historical_data as historical
import insider_study as v2
from insider_filings import acceptance_map, load_filings
from pit_estimates import SENSITIVITY, validate

PREREG = C.ROOT / "prereg" / "insider_event_v3.json"
REG = json.loads(PREREG.read_text())
BUNDLE_ID = "8433bd236228ba351c9a"
WINDOW_START, WINDOW_END = pd.Timestamp("2016-01-04"), pd.Timestamp("2026-10-30")
CONCURRENT_ITEMS = re.compile(r"\b(?:2\.02|1\.01|2\.01)\b")
PRE_ENTRY_SESSIONS = 2
ATTESTATIONS = ["constituents_complete", "identifiers_verified", "delisted_prices_complete",
                "terminal_events_complete", "corporate_actions_verified"]
VERBATIM = ["horizon_sessions", "cooldown_sessions", "cluster_calendar_days", "minimum_close_usd", "minimum_adv_usd",
            "order_usd", "max_open_positions", "ridge_alpha", "min_train_events", "min_calibration_events",
            "embargo_sessions", "cpcv_folds", "cpcv_test_folds", "interval_coverage", "min_forward_events",
            "min_forward_calendar_days", "trials", "secondary_tests", "cost_sensitivities"]


def check_recipe():
    if not REG.get("registered_at_utc") or REG.get("status") != "REGISTERED":
        raise ValueError("v3 is not registered")
    diff = [k for k in VERBATIM if REG[k] != v2.REG[k]]
    if diff:
        raise ValueError(f"v3 recipe differs from v2 on {diff}; insider_study machinery would not implement v3")
    if C.COST_MODEL != "v2_quoted":
        raise ValueError("v3 registers cost model v2_quoted")


def concurrent_sessions(market, items_path=C.PROC / "8k_items.parquet"):
    """issuer CIK -> sorted session indices at which an earnings / material agreement / M&A 8-K became available."""
    k8 = pd.read_parquet(items_path, columns=["cik", "available_date", "items"]).dropna(subset=["available_date"])
    k8 = k8[k8["items"].fillna("").str.contains(CONCURRENT_ITEMS)]
    return {int(cik): np.sort(market.cal.searchsorted(pd.to_datetime(g["available_date"]).to_numpy(), side="left"))
            for cik, g in k8.groupby("cik")}


def price_gap(market, sid, e, x):
    last = min(x, len(market.observed) - 1)
    segment = market.opens[sid].iloc[e:last + 1]
    term = market.terminal[market.terminal["security_id"].eq(sid)]
    if len(term):
        segment = segment[segment.index <= pd.Timestamp(term.iloc[0]["last_trade_date"])]
    return bool(segment.isna().any())


def make_filter(market, sessions_8k, concurrent=True):
    def keep(events):
        window = events["entry_date"].between(WINDOW_START, WINDOW_END)
        gaps = pd.Series([price_gap(market, r.symbol, r.e, r.x) for r in events.itertuples()], index=events.index)
        near = pd.Series(False, index=events.index)
        if concurrent:
            for idx, r in events.iterrows():
                s = sessions_8k.get(int(r.cik))
                if s is not None:
                    lo, hi = np.searchsorted(s, r.e - PRE_ENTRY_SESSIONS, "left"), np.searchsorted(s, r.x, "right")
                    near.loc[idx] = hi > lo
        kept = window & ~gaps & ~near
        counts = {"outside_window": int((~window).sum()), "price_gap": int((window & gaps).sum()),
                  "concurrent_8k": int((window & ~gaps & near).sum()), "kept": int(kept.sum()),
                  "concurrent_rule_applied": concurrent}
        return events[kept].copy(), counts
    return keep


def sp500_audit(bundle, market):
    from ridge_sp500_study import audit_bundle
    result, _ = audit_bundle(bundle, pd.DatetimeIndex(market.observed), WINDOW_START, WINDOW_END)
    claim = json.loads((bundle / "coverage.json").read_text())
    unattested = [k for k in ATTESTATIONS if claim.get(k) is not True]
    return {"passed": result["passed"], "failed_checks": result["failed_checks"],
            "unattested_sources": unattested,
            "members_with_price_gaps": sorted(result["missing_prices"]),
            "unresolved_terminals": result["unresolved_terminals"],
            "input_sha256": result["input_sha256"]}


def run(out, acceptances, sensitivity_estimates=None):
    check_recipe()
    bundle, spy, sic, estimates, paths, available, hashes = v2.inspect_inputs(acceptances, sensitivity_estimates)
    if bundle is None or Path(bundle).name != BUNDLE_ID:
        raise ValueError(f"v3 registers bundle {BUNDLE_ID}; active bundle is {bundle}")
    needed = {k: v for k, v in available.items() if k not in ("verified quarterly estimate vintages",)}
    missing = [k for k, v in needed.items() if not v]
    if missing:
        raise ValueError("missing inputs: " + ", ".join(missing))
    inferred = None
    if sensitivity_estimates is not None:
        inferred = validate(pd.read_parquet(sensitivity_estimates))
        if inferred.empty or not inferred["timestamp_basis"].eq(SENSITIVITY).all():
            raise ValueError("estimate sensitivity input must contain only inferred calc-date estimates")
    out.mkdir(parents=True, exist_ok=False)
    v2.save_json(out / "registration.json", {
        "recipe": REG, "recipe_sha256": v2.digest(PREREG), "sources": v2.source_hashes() | {
            "insider_study_v3.py": v2.digest(C.ROOT / "insider_study_v3.py"),
            "quoted_spreads.py": v2.digest(C.ROOT / "quoted_spreads.py"), "data_8k.py": v2.digest(C.ROOT / "data_8k.py")},
        "input_sha256": hashes, "run_at_utc": datetime.now(timezone.utc).isoformat(), "trial_count": len(v2.TRIALS),
        "cost_config": {"cost_model": C.COST_MODEL, "fx": C.FX_FEE, "half_spread_floor": C.HALF_SPREAD,
                        "slippage": C.SLIPPAGE}})
    market = v2.StudyMarket(bundle, spy, sic)
    span = json.loads((bundle / "coverage.json").read_text())
    report = {"study": REG["id"], "registered_at_utc": REG["registered_at_utc"], "status": "UNVALIDATED",
              "evidence": "PREVIOUSLY_VIEWED_DEVELOPMENT", "reasons": [], "input_sha256": hashes}
    report["coverage_audit"] = audit = sp500_audit(bundle, market)
    report["reasons"].extend(audit["failed_checks"])
    if audit["unattested_sources"]:
        report["reasons"].append("unattested sources (registered limitation): " + ", ".join(audit["unattested_sources"]))
    from insider_acceptances import coverage as acceptance_coverage
    report["acceptance_coverage"] = acceptance_coverage(paths, acceptances)
    filings = load_filings(paths, sorted(set(market.cik.values())), acceptance_map(acceptances))
    report["filings"] = {"rows": len(filings), "exact_timestamp_rows": int(filings["accepted"].notna().sum()),
                         "date_only_rows": int(filings["accepted"].isna().sum())}
    sessions_8k = concurrent_sessions(market)
    full, no_concurrent = make_filter(market, sessions_8k), make_filter(market, sessions_8k, concurrent=False)
    report["primary"], _ = v2.history_run(market, filings, None, span, out, event_filter=full)
    report["date_only_sensitivity"], _ = v2.history_run(market, filings, None, span, out, "_date_only",
                                                        date_only=True, event_filter=full)
    report["explicit_non_plan_sensitivity"], _ = v2.history_run(market, filings, None, span, out, "_explicit_non_plan",
                                                                explicit_non_plan=True, event_filter=full)
    report["concurrent_included_sensitivity"], _ = v2.history_run(market, filings, None, span, out,
                                                                  "_concurrent_included", event_filter=no_concurrent)
    if inferred is not None:
        report["inferred_estimate_sensitivity"], _ = v2.history_run(market, filings, inferred, span, out,
                                                                    "_inferred_estimates", event_filter=full)
    report["reasons"].extend(report["primary"]["reasons"])
    if v2.inspect_inputs(acceptances, sensitivity_estimates)[-1] != hashes:
        report["reasons"].append("Input files changed during the run. Discard computed statistics and rerun.")
    report["reasons"].append("Prospective paper evidence and user review are pending")
    v2.save_json(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--acceptances", type=Path, required=True)
    parser.add_argument("--sensitivity-estimates", type=Path)
    args = parser.parse_args()
    print(json.dumps(v2.json_value(run(args.out, args.acceptances, args.sensitivity_estimates)), indent=2))


if __name__ == "__main__":
    main()
