"""Audit an S&P 500 candidate and evaluate the preregistered ridge decision."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import research as R
from historical_data import active_bundle, read_sessions
from historical_membership import reconcile_snapshots
from provenance import _content_hash, dataset_version, panel_identity
from recipe import write_exclusive
from ridge_controls import decision
from ridge_paper import benchmark

REG = C.ROOT / "prereg" / "sp500_ridge_20_v1.json"


def audit_bundle(bundle, calendar, start, end, snapshot=None, validation_rosters=None):
    def read(name):
        path = bundle / f"{name}.parquet"
        return pd.read_parquet(path) if path.is_file() else pd.read_csv(bundle / f"{name}.csv")

    coverage = json.loads((bundle / "coverage.json").read_text())
    master, membership, prices, terminal = [read(name) for name in
                                            ("security_master", "membership", "prices", "terminal")]
    prices["date"] = pd.to_datetime(prices.date)
    for column in ("start", "end", "known_at"):
        membership[column] = pd.to_datetime(membership[column])
    members = membership.loc[membership["index"].eq("SP500")].copy()
    failures = []
    for field in ("constituents_complete", "identifiers_verified", "delisted_prices_complete",
                  "terminal_events_complete", "corporate_actions_verified"):
        if coverage.get(field) is not True:
            failures.append(f"{field} requires source evidence")
    if pd.Timestamp(coverage["start"]) > start or pd.Timestamp(coverage["end"]) < end:
        failures.append("coverage dates do not span the registered study")
    sessions = calendar[(calendar >= start) & (calendar <= end)]
    if (bundle / "sessions.csv").is_file():
        independent = read_sessions(bundle / "sessions.csv")
        independent = independent[(independent >= start) & (independent <= end)]
        if not sessions.equals(independent):
            failures.append("benchmark sessions disagree with the exchange schedule")
    else:
        failures.append("independent exchange schedule missing")
    observed_rosters = []
    if validation_rosters:
        snapshots = pd.read_csv(validation_rosters, dtype={"security_id": str})
        if set(snapshots["index"]) != {"SP500"}:
            raise ValueError("comparison rosters must cover the S&P 500")
        observed_rosters = reconcile_snapshots(members, snapshots)
        if any(not row["matches"] for row in observed_rosters):
            failures.append("membership replay disagrees with observed rosters")
    else:
        failures.append("independent security-level roster reconciliation missing")
    if not len(sessions) or members.empty:
        failures.append("no S&P 500 membership or benchmark sessions in study window")
    terms = terminal.set_index("security_id")
    last_prices = prices.groupby("security_id").date.max()
    price_dates = {sid: pd.DatetimeIndex(g.date) for sid, g in prices.groupby("security_id")}
    expected_count = np.zeros(len(sessions), dtype=int)
    gaps, outcomes = {}, {}
    for row in members.itertuples():
        lo = sessions.searchsorted(max(row.start, row.known_at))
        hi = len(sessions) if pd.isna(row.end) else sessions.searchsorted(row.end)
        expected = sessions[lo:hi]
        if expected.empty:
            continue
        expected_count[lo:hi] += 1
        last = last_prices.get(row.security_id, pd.NaT)
        if pd.notna(last) and last < end and row.security_id not in terms.index:
            outcomes[row.security_id] = last.date().isoformat()
        if row.security_id in terms.index:
            term = terms.loc[row.security_id]
            expected = expected[expected <= pd.Timestamp(term["last_trade_date"])]
            if pd.Timestamp(term["last_trade_date"]) < end < pd.Timestamp(term["settlement_date"]):
                outcomes[row.security_id] = pd.Timestamp(term["last_trade_date"]).date().isoformat()
        absent = expected.difference(price_dates.get(row.security_id, pd.DatetimeIndex([])))
        if len(absent):
            gaps.setdefault(row.security_id, []).extend(absent.strftime("%Y-%m-%d"))
    empty_rosters = sessions[expected_count == 0].strftime("%Y-%m-%d").tolist()
    if gaps or outcomes or empty_rosters:
        failures.append("session coverage or terminal outcomes incomplete")
    unresolved = []
    if snapshot:
        roster = pd.read_csv(snapshot, usecols=["date", "ric"]).drop_duplicates()
        mapped = set(master["ric"])
        unresolved = roster.loc[~roster.ric.isin(mapped)].to_dict("records")
        if unresolved:
            failures.append("source roster includes unmapped instruments")
    source_files = [path for path in bundle.iterdir()
                    if path.name in {"coverage.json", "sessions.csv"} or
                    path.stem in {"security_master", "membership", "prices", "terminal"}]
    source_files.extend(path for path in (snapshot, validation_rosters) if path)
    hashes = {str(path.resolve()): _content_hash(path) for path in source_files}
    return dict(passed=not failures, failed_checks=failures, missing_prices=gaps,
                input_sha256=hashes, roster_reconciliation=observed_rosters,
                unresolved_terminals=outcomes, unpopulated_roster_sessions=empty_rosters,
                unmapped_snapshot_records=unresolved), members


def study(out, staged=None, snapshot=None, validation_rosters=None):
    spy = benchmark()
    bundle = staged or active_bundle()
    report = {"status": "UNVALIDATED", "evidence": "PREVIOUSLY_VIEWED_DEVELOPMENT"}
    if bundle is None:
        report["failed_checks"] = ["historical S&P 500 security bundle missing"]
        write_exclusive(out, report)
        return report
    reg = json.loads(REG.read_text(encoding="utf-8"))
    start, end = pd.Timestamp(reg["first_decision"]), pd.Timestamp(reg["benchmark_end"])
    spy = spy.loc[:end]
    report["preregistration_sha256"] = _content_hash(REG)
    coverage, members = audit_bundle(bundle, spy.index, start, end, snapshot, validation_rosters)
    report["coverage"] = coverage
    if not staged and coverage["passed"]:
        manifest = json.loads((bundle / "coverage.json").read_text())
        hashes = manifest.get("materialized_prices_sha256", {})
        mismatches = []
        for sid in sorted(set(members.security_id)):
            path = C.RAW / "prices" / f"{sid}.parquet"
            if not path.is_file() or _content_hash(path) != hashes.get(sid):
                mismatches.append(sid)
        if mismatches:
            coverage["passed"] = False
            coverage["failed_checks"].append("materialized prices disagree with historical import")
            coverage["materialized_mismatches"] = mismatches
    if staged or not coverage["passed"]:
        report["failed_checks"] = coverage["failed_checks"]
        write_exclusive(out, report)
        return report

    version = dataset_version()
    ok, why = panel_identity("large", version)
    if not ok:
        raise ValueError(why)
    panel_path = C.PROC / "panel_large.parquet"
    panel = pd.read_parquet(panel_path)
    panel = panel.loc[panel.date.between(start, reg["last_decision"])]
    eligible = pd.Series(False, index=panel.index)
    for sid, rows in members.groupby("security_id"):
        ix = panel.index[panel.symbol.eq(sid)]
        dates = panel.loc[ix, "date"]
        for row in rows.itertuples():
            mask = dates.ge(max(row.start, row.known_at))
            if pd.notna(row.end):
                mask &= dates.lt(row.end)
            eligible.loc[ix[mask]] = True
    panel = panel.loc[eligible].reset_index(drop=True)
    if panel.empty:
        raise ValueError("no panel rows match verified S&P 500 membership")
    dates, symbols = panel.date.to_numpy(), panel.symbol.to_numpy()
    pos = spy.index.get_indexer(dates)
    hpos = spy.index.get_loc(pd.Timestamp(reg["viewed_holdout_start"]))
    h = reg["horizon"]
    feats = reg["features"]
    expected_costs = dict(fx=C.FX_FEE, half_spread=C.HALF_SPREAD, slippage=C.SLIPPAGE)
    if (expected_costs != reg["cost_config"] or R.EMBARGO != reg["embargo_sessions"] or
            C.RIDGE_ALPHA != reg["ridge_alpha"]):
        raise ValueError("runtime configuration differs from the registered trial")
    ranked = R.preprocess(panel, feats)
    X = R.sector_neutral(ranked, panel, feats).reindex(columns=reg["columns"], fill_value=0)
    cost = R.name_costs(panel, reg["order_usd"])
    settled = R.settled_before(pos, h, hpos)
    y = np.where((settled | (pos >= hpos)) & (pos + 1 + h < len(spy)), panel.y_20, np.nan)
    registration = dict(recipe=reg, dataset_version=version,
                        panel_sha256=_content_hash(panel_path), features=feats,
                        first_date=str(start.date()), end_date=str(end.date()),
                        viewed_holdout_start=str(spy.index[hpos].date()), order_usd=reg["order_usd"],
                        ridge_alpha=C.RIDGE_ALPHA, embargo=R.EMBARGO,
                        source_sha256={name: _content_hash(C.ROOT / name) for name in
                                       ("research.py", "ridge_controls.py", "ridge_sp500_study.py",
                                        "holdings_simulation.py", "requirements.txt")},
                        bundle_sha256=coverage["input_sha256"],
                        cost_config=dict(fx=C.FX_FEE, half_spread=C.HALF_SPREAD, slippage=C.SLIPPAGE))
    write_exclusive(out.with_suffix(".registration.json"), registration)
    sectors = panel.sector.fillna("Unknown").to_numpy()
    wf_score, hp = R.walk_forward("ridge", X, y, dates, pos, h, hpos, sectors, None)
    from functools import partial
    backtest = partial(R.long_only_backtest, benchmark_returns=R.benchmark_label_returns(spy, dates, h),
                       session_positions=pos, entry_prices=panel.entry_adj_open)
    controls = R.ridge_control_report(panel, X, feats, y, dates, symbols, pos, hpos, sectors, cost,
                                     settled, wf_score, hp, backtest, {})
    wf = controls["walk_forward"]
    metrics = wf["strategies"]["ridge"]["metrics"] or {}
    ledger_ok = bool(metrics) and all(not item["missing_selected_outcomes"] for item in wf["strategies"].values())
    current = dataset_version() == version and panel_identity("large", version)[0]
    from gates import joint_test_failures, pbo_failures
    from institutional_research import compare
    vqm, _, _ = compare(panel, np.where(settled, wf_score, hp), cost, h, spy.index[hpos],
                  benchmark_returns=R.benchmark_label_returns(spy, dates, h), session_positions=pos)
    tests = controls["statistical_tests"]
    holdout = controls["viewed_holdout"]
    ic = holdout["ridge_ic"]
    held = holdout["strategies"]["ridge"]
    ledger_ok = ledger_ok and all(not item["missing_selected_outcomes"]
                                 for item in holdout["strategies"].values())
    failures = []
    if not ledger_ok:
        failures.append("incomplete selected outcomes or insufficient periods")
    if not current:
        failures.append("study inputs changed while running")
    if ic["n_dates"] < 20 or (ic["mean"] or 0) <= 0 or (ic["t"] or 0) < C.GATE_MIN_HOLDOUT_IC_T:
        failures.append("viewed holdout IC gate failed")
    if (wf["positive_ic_year_fraction"] or 0) < C.GATE_MIN_WF_POSITIVE_YEAR_FRAC:
        failures.append("positive walk-forward IC year fraction below threshold")
    if (held["metrics"] or {}).get("mean_net_excess", 0) <= 0:
        failures.append("viewed holdout net excess is nonpositive")
    if not (wf["monotonic_deciles"] and holdout["monotonic_deciles"]):
        failures.append("forecast deciles fail out-of-sample calibration")
    if (tests["dsr"] or 0) < C.GATE_MIN_DSR:
        failures.append("deflated Sharpe below threshold")
    failures.extend(joint_test_failures(tests["joint"]))
    failures.extend(pbo_failures(tests["pbo"], C.GATE_MAX_PBO))
    from risk_limits import load
    if load(refresh_if_missing=False).get("order_size_usd") != reg["order_usd"]:
        failures.append("registered order size lacks matching bot configuration")
    if (metrics.get("mean_net_excess") or 0) <= 0:
        failures.append("walk-forward net excess is nonpositive or unavailable")
    if (wf["paired_net_mean"] or 0) <= 0:
        failures.append("paired net advantage over signed equal weight is nonpositive or unavailable")
    gate_ready = not failures
    report.update(controls=controls, fixed_vqm=vqm, registration=registration, trial_count=3, failed_checks=failures)
    report["status"] = decision(current, ledger_ok, metrics.get("mean_gross_excess"),
                                metrics.get("mean_net_excess"), wf["paired_net_mean"], gate_ready)
    write_exclusive(out, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--staged", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--validation-rosters", type=Path)
    args = parser.parse_args()
    lock = C.hold_lock() if args.staged is None else None
    result = study(args.out, args.staged, args.snapshot, args.validation_rosters)
    print(json.dumps({"status": result["status"], "failed_checks": result["failed_checks"]}, indent=2))
