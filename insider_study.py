"""Research-only Form 4 v2 study. See docs/INSIDER_STUDY.md."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C
import historical_data as historical
from features import _cs_spread, sic_sector
from insider_filings import NY, acceptance_map, available_filings, filing_paths, load_filings
from pit_estimates import SENSITIVITY, read_primary_quarterly, validate
from research import name_costs
from validation_extensions import purged_combinations
from insider_evaluation import diagnostics, matched_evaluation, portfolio

PREREG = C.ROOT / "prereg" / "insider_event_v2.json"
REG = json.loads(PREREG.read_text())
H = REG["horizon_sessions"]
EMBARGO = REG["embargo_sessions"]
SIZE = REG["order_usd"]
CAPACITY = REG["max_open_positions"]
TRIALS = REG["trials"]
FEATURES = ["log_buy_to_cap", "senior", "buyers", "routine_share", "unknown_history_share"]
CONTROLS = ["mom", "rev", "log_adv", "beta", "earn_yield", "book_to_mkt", "roa", "earnings_revision"]


class StudyMarket:
    def __init__(self, bundle, spy_path, sic_path, future_dates=()):
        master, members, prices, terminal = historical.validate_bundle(*[
            pd.read_parquet(bundle / (name + ".parquet"))
            for name in ("security_master", "membership", "prices", "terminal")])
        spy = pd.read_parquet(spy_path).set_index("date").sort_index()
        self.observed = pd.DatetimeIndex(spy.index)
        if (not self.observed.is_unique or not len(spy)
                or not np.isfinite(spy[["open", "adjclose", "close"]]).all().all()
                or not spy[["open", "adjclose", "close"]].gt(0).all().all()):
            raise ValueError("invalid SPY session data")
        self.cal = self.observed.append(pd.DatetimeIndex(future_dates))
        self.spy = (spy["open"] * spy["adjclose"] / spy["close"]).reindex(self.cal)
        # Unknown future membership changes never determine an earlier decision.
        self.members = historical.membership_mask(members, self.cal, list(master["security_id"]))
        self.cik = master.set_index("security_id")["cik"].to_dict()
        sic = pd.read_parquet(sic_path).set_index("cik")["sic"]
        self.sector = {sid: sic_sector(sic.get(cik)) for sid, cik in self.cik.items()}
        self.terminal = terminal
        close = prices.pivot(index="date", columns="security_id", values="adjclose").reindex(self.cal)
        self.opens = prices.pivot(index="date", columns="security_id", values="adj_open").reindex(self.cal)
        raw_close = prices.pivot(index="date", columns="security_id", values="close").reindex(self.cal)
        volume = prices.pivot(index="date", columns="security_id", values="volume").reindex(self.cal)
        self.adv = (raw_close * volume).rolling(60, min_periods=40).mean()
        self.eligible = self.members & self.adv.ge(REG["minimum_adv_usd"]) & raw_close.ge(REG["minimum_close_usd"])
        self.cap = prices.pivot(index="date", columns="security_id", values="market_cap").reindex(self.cal)
        r = close.pct_change(fill_method=None)
        spy_r = spy["adjclose"].reindex(self.cal).pct_change(fill_method=None)
        self.beta = r.rolling(252, min_periods=200).cov(spy_r).div(spy_r.rolling(252, min_periods=200).var(), axis=0)
        self.mom = close.shift(21) / close.shift(252) - 1
        self.rev = close / close.shift(21) - 1
        vol = r.rolling(60, min_periods=40).std() * np.sqrt(252)
        hi = prices.pivot(index="date", columns="security_id", values="high").reindex(self.cal)
        lo = prices.pivot(index="date", columns="security_id", values="low").reindex(self.cal)
        spread = _cs_spread(hi, lo).rolling(60, min_periods=40).mean()
        cost_input = pd.DataFrame({"cs_spread": spread.to_numpy().ravel(),
                                   "vol_60": vol.to_numpy().ravel(), "adv_dollar": self.adv.to_numpy().ravel()})
        if C.COST_MODEL == "v2_quoted":  # prereg/spread_check_v1.json
            import quoted_spreads
            cost_input["hs_quoted"] = quoted_spreads.wide(self.cal, list(self.adv.columns)).to_numpy().ravel()
        self.costs = {size: pd.DataFrame(
            (name_costs(cost_input, size).to_numpy() + C.SLIPPAGE).reshape(self.adv.shape),
            index=self.cal, columns=self.adv.columns) for size in [SIZE, SIZE * 2]}

    def outcome(self, sid, e):
        x = e + H
        if x >= len(self.observed) or not np.isfinite(self.opens[sid].iloc[e]):
            return None
        entry = self.opens[sid].iloc[e]
        term = self.terminal[self.terminal["security_id"].eq(sid)]
        stock = historical.forward_returns(self.opens[[sid]], term, [e], [x])[0, 0]
        if not np.isfinite(stock):
            return None
        beta = self.beta[sid].iloc[e - 1]
        bench = self.spy.iloc[x] / self.spy.iloc[e] - 1
        if not np.isfinite(beta * bench):
            return None
        settlement = pd.isna(self.opens[sid].iloc[x])
        marks = self.opens[sid].iloc[e:x + 1].copy()
        if settlement:
            t = term.iloc[0]
            post_trade = marks.index > t["last_trade_date"]
            marks.loc[post_trade] = marks.loc[:t["last_trade_date"]].iloc[-1]
            marks.loc[marks.index >= t["settlement_date"]] = t["terminal_adjclose"]
        if marks.isna().any():
            return None
        return stock, stock - beta * bench, entry, settlement, marks.to_numpy()


def build_events(market, purchases):
    history, last, rows = {}, {}, []
    rejected = {"ambiguous_security": 0, "ineligible": 0, "cooldown": 0}
    for (e, cik), group in purchases.groupby(["e", "issuercik"], sort=True):
        queue = history.setdefault(cik, deque())
        if e >= len(market.cal):
            continue
        day = market.cal[e]
        cutoff = day - pd.Timedelta(days=REG["cluster_calendar_days"])
        queue.extend(zip(group["known_day"], group["owner"]))
        while queue and queue[0][0] < cutoff:
            queue.popleft()
        if e < 253:
            continue
        eligible = market.eligible.iloc[e - 1]
        ids = [sid for sid in eligible.index[eligible] if market.cik[sid] == cik]
        if len(ids) != 1:
            rejected["ambiguous_security" if ids else "ineligible"] += 1
            continue
        if e - last.get(cik, -10000) < REG["cooldown_sessions"]:
            rejected["cooldown"] += 1
            continue
        sid = ids[0]
        cap = market.cap[sid].iloc[e - 1]
        cost = market.costs[SIZE][sid].iloc[e - 1]
        if not (np.isfinite(cap) and cap > 0 and np.isfinite(cost)):
            rejected["ineligible"] += 1
            continue
        buyers = len({owner for known, owner in queue if known >= cutoff})
        value = group["value"].sum()
        rows.append({"e": int(e), "x": int(e + H), "entry_date": day, "cik": int(cik),
                     "symbol": sid, "accessions": ",".join(sorted(group["accession"].unique())),
                     "unknown_plan_lines": int(group["plan_status"].eq("unknown").sum()),
                     "log_buy_to_cap": np.log1p(10000 * value / cap),
                     "senior": int(group["senior"].any()), "buyers": buyers,
                     "routine_share": group.loc[group["class"].eq("routine"), "value"].sum() / value,
                     "unknown_history_share": group.loc[group["class"].eq("unclassified"), "value"].sum() / value,
                     "cost": cost, "double_size_cost": market.costs[SIZE * 2][sid].iloc[e - 1],
                     "beta": market.beta[sid].iloc[e - 1],
                     "primary": bool(group["senior"].any() or buyers >= 2)})
        last[cik] = e
    return pd.DataFrame(rows), rejected


def label_events(market, events):
    events = events.copy()
    events["net"] = np.nan
    outcomes = {}
    for idx, row in events.iterrows():
        outcome = market.outcome(row.symbol, row.e)
        if outcome is not None:
            stock, gross, _, settlement, _ = outcome
            events.loc[idx, "net"] = gross - row.cost * (1 + (0 if settlement else 1 + stock))
            outcomes[idx] = outcome
    return events, outcomes


def fit(events, train):
    model = make_pipeline(StandardScaler(), Ridge(alpha=REG["ridge_alpha"]))
    return model.fit(events.loc[train, FEATURES], events.loc[train, "net"])


def walk_forward(events):
    predictions = pd.Series(np.nan, index=events.index)
    bounds = pd.DataFrame(np.nan, index=events.index, columns=["low", "high"])
    for e, test in events.groupby("e", sort=True).groups.items():
        settled = events["x"].add(EMBARGO).lt(e) & events["net"].notna()
        if settled.sum() < REG["min_train_events"]:
            continue
        predictions.loc[test] = fit(events, settled).predict(events.loc[test, FEATURES])
        radius = interval_radius((predictions - events["net"]).abs()[settled].dropna())
        if radius is not None:
            bounds.loc[test, "low"] = predictions.loc[test] - radius
            bounds.loc[test, "high"] = predictions.loc[test] + radius
    return predictions, bounds


def interval_radius(errors):
    n = len(errors)
    if n < REG["min_calibration_events"]:
        return None
    k = int(np.ceil((n + 1) * REG["interval_coverage"]))
    return float(np.sort(errors)[k - 1]) if k <= n else None


def cpcv(events, last_position):
    paths = {}
    splits = []
    groups = purged_combinations(events["e"].to_numpy() - 1, H, last_position,
                                 EMBARGO, REG["cpcv_folds"], REG["cpcv_test_folds"])
    for split, (train, tests, path_ids) in enumerate(groups):
        train = train[events["net"].iloc[train].notna()]
        if len(train) < REG["min_train_events"]:
            continue
        model = fit(events, train)
        for test, path in zip(tests, path_ids):
            predicted = model.predict(events.iloc[test][FEATURES])
            paths.setdefault(int(path), []).extend(zip(test, predicted))
        splits.append({"split": split, "train": len(train), "test": sum(map(len, tests))})
    results = []
    for path, values in sorted(paths.items()):
        ids, scores = zip(*values)
        actual = events["net"].iloc[list(ids)].to_numpy()
        valid = np.isfinite(actual)
        results.append({"path": path, "scored": len(ids), "settled": int(valid.sum()),
                        "mae": float(np.abs(actual[valid] - np.asarray(scores)[valid]).mean()) if valid.any() else None})
    return {"status": "COMPUTED" if splits else "NOT_COMPUTED", "splits": splits, "paths": results}



def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_hashes():
    names = ["insider_study.py", "insider_filings.py", "insider_acceptances.py", "insider_evaluation.py", "data_form4_recent.py",
             "data_insider.py", "features.py", "historical_data.py", "pit_estimates.py",
             "research.py", "config.py", "validation_extensions.py", "requirements.txt"]
    return {name: digest(C.ROOT / name) for name in names}


def json_value(value):
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def save_json(path, data):
    payload = json.dumps(json_value(data), indent=2, allow_nan=False)
    with path.open("x", encoding="utf-8") as file:
        file.write(payload + "\n")


def interval_report(events, bounds):
    ok = bounds.notna().all(axis=1) & events["net"].notna()
    lo, hi = bounds.loc[ok, "low"], bounds.loc[ok, "high"]
    actual = events.loc[ok, "net"]
    return {"nominal_coverage": REG["interval_coverage"], "settled_intervals": int(ok.sum()),
            "coverage": float(actual.between(lo, hi).mean()) if ok.any() else None,
            "mean_width": float((hi - lo).mean()) if ok.any() else None}


def history_run(market, filings, vintages, coverage, out, suffix="", date_only=False, explicit_non_plan=False,
                event_filter=None):
    """event_filter (optional, used by insider_study_v3): events -> (kept events, exclusion counts); applied to
    candidate events before labelling, so excluded events never influence training, scoring or portfolios."""
    purchases = available_filings(filings, market.cal, date_only, explicit_non_plan)
    events, rejected = build_events(market, purchases)
    excluded = None
    if event_filter is not None and not events.empty:
        events, excluded = event_filter(events)
        events = events.reset_index(drop=True)
    report = {"candidate_events": len(events), "rejected": rejected, "reasons": [],
              "status": "UNVALIDATED", "evidence": "PREVIOUSLY_VIEWED_DEVELOPMENT"}
    if excluded is not None:
        report["scope_exclusions"] = excluded
    if events.empty:
        report["reasons"].append("No eligible purchase events")
        return report, None
    events, outcomes = label_events(market, events)
    events["score"], bounds = walk_forward(events)
    events = events.join(bounds)
    report["intervals"] = interval_report(events, bounds)
    report["unknown_plan_lines_in_events"] = int(events["unknown_plan_lines"].sum())
    report["settled_events"] = len(outcomes)
    report["cpcv"] = cpcv(events, len(market.observed))
    events.to_parquet(out / f"events{suffix}.parquet", index=False)
    sample = events[events["score"].notna() & events["x"].lt(len(market.observed))].copy()
    if sample.empty:
        report["reasons"].append("Insufficient settled data for chronological walk-forward")
        return report, events
    sample["entry_unfilled"] = [pd.isna(market.opens[r.symbol].iloc[r.e]) for r in sample.itertuples()]
    if coverage.get("start"):
        sample = sample[sample["entry_date"].ge(pd.Timestamp(coverage["start"]))]
    if sample.empty:
        report["reasons"].append("No scored events within historical source coverage")
        return report, events
    positions = slice(int(sample["e"].min()), len(market.observed))
    variants = [sample[sample["score"].gt(0)].assign(rank=lambda d: d["score"]),
                sample[sample["primary"]].assign(rank=0), sample.assign(rank=0)]
    matrix, reports = {}, {}
    for name, frame in zip(TRIALS, variants):
        summaries = {}
        for scenario, size, cost_col, multiplier in [
            ("base", SIZE, "cost", 1), ("twice_order", SIZE * 2, "double_size_cost", 1),
            ("twice_cost", SIZE, "cost", 2)
        ]:
            daily, ledger = portfolio(frame, outcomes, market.spy, positions, size, CAPACITY, H, cost_col, multiplier)
            if scenario == "base":
                matrix[name] = daily["net"]
            ledger.to_parquet(out / f"fills_{name}_{scenario}{suffix}.parquet", index=False)
            summaries[scenario] = {"orders": len(ledger), "valid_days": int(daily["net"].notna().sum()),
                                   "gross_mean_daily": daily["gross"].mean(), "net_mean_daily": daily["net"].mean(),
                                   "cost_usd": daily["cost_usd"].sum(),
                                   "statuses": ledger["status"].value_counts().to_dict() if len(ledger) else {}}
        reports[name] = summaries
    matrix = pd.DataFrame(matrix)
    matrix.to_parquet(out / f"common_returns{suffix}.parquet")
    report["variants"] = reports
    report["matched"] = {"status": "NOT_COMPUTED", "reason": "SEC company facts unavailable"}
    if (C.RAW / "facts" / "companyfacts.parquet").exists():
        report["matched"], pairs = matched_evaluation(market, sample, purchases, vintages, CONTROLS, SIZE)
        pairs.to_parquet(out / f"matched_pairs{suffix}.parquet", index=False)
    report["statistics"] = diagnostics(matrix, H, report["matched"].get("p"), C.FDR_Q)
    if report["matched"].get("p") is None:
        report["reasons"].append("Incremental value has insufficient complete matched controls")
    if matrix.isna().any().any():
        report["reasons"].append("Some selected outcomes are unresolved")
    if report["statistics"]["dsr"] is None or report["statistics"]["dsr"] < C.GATE_MIN_DSR:
        report["reasons"].append("Deflated Sharpe is missing or below the configured threshold")
    if report["statistics"]["pbo"] is None or report["statistics"]["pbo"] > C.GATE_MAX_PBO:
        report["reasons"].append("PBO is missing or above the configured threshold")
    comparable = matrix.dropna()
    if comparable.empty or not comparable.iloc[:, 0].mean() > max(0, comparable.iloc[:, 1:].mean().max()):
        report["reasons"].append("Positive net development performance versus both baselines is unproven")
    return report, events


def inspect_inputs(acceptances=None, sensitivity_estimates=None):
    bundle = historical.active_bundle()
    spy = C.RAW / "prices" / "SPY.parquet"
    sic = C.PROC / "sic.parquet"
    estimates = C.PROC / "estimate_vintages.parquet"
    facts = C.RAW / "facts" / "companyfacts.parquet"
    filings = filing_paths()
    required = {"historical security bundle": bundle is not None,
                "SPY price history": spy.exists(), "SEC SIC data": sic.exists(),
                "SEC non-derivative filings": bool(filings), "SEC company facts": facts.exists(),
                "verified quarterly estimate vintages": read_primary_quarterly(estimates) is not None}
    paths = [spy, sic, estimates, facts, *filings]
    if sensitivity_estimates is not None:
        paths.append(sensitivity_estimates)
    if bundle is not None:
        for name in ("security_master.parquet", "membership.parquet", "prices.parquet", "terminal.parquet", "coverage.json"):
            path = bundle / name
            required[name] = path.is_file()
            paths.append(path)
    if acceptances is not None:
        paths.append(acceptances)
        required["SEC acceptance CSV"] = acceptances.is_file()
    hashes = {str(p.resolve()): digest(p) for p in paths if p.is_file()}
    return bundle, spy, sic, estimates, filings, required, hashes


def run(out, acceptances=None, sensitivity_estimates=None, *, audit_only=False):
    bundle, spy, sic, estimates, paths, available, hashes = inspect_inputs(acceptances, sensitivity_estimates)
    inferred = None
    if sensitivity_estimates is not None:
        inferred = validate(pd.read_parquet(sensitivity_estimates))
        if inferred.empty or not inferred["timestamp_basis"].eq(SENSITIVITY).all():
            raise ValueError("estimate sensitivity input must contain only inferred calc-date estimates")
    registration = {"recipe": REG, "recipe_sha256": digest(PREREG), "sources": source_hashes(),
                    "input_sha256": hashes, "registered_at_utc": datetime.now(timezone.utc).isoformat(),
                    "trial_count": len(TRIALS),
                    "cost_config": {"cost_model": C.COST_MODEL, "fx": C.FX_FEE, "half_spread_floor": C.HALF_SPREAD, "slippage": C.SLIPPAGE}}
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "registration.json", registration)
    reasons = [f"Missing {name}" for name, present in available.items() if not present]
    report = {"status": "UNVALIDATED", "inputs_present": available, "input_sha256": hashes,
              "recipe_sha256": registration["recipe_sha256"], "trial_count": len(TRIALS),
              "reasons": reasons, "evidence": "PREVIOUSLY_VIEWED_DEVELOPMENT"}
    if acceptances is not None and acceptances.is_file() and paths:
        from insider_acceptances import coverage as acceptance_coverage

        report["acceptance_coverage"] = acceptance_coverage(paths, acceptances)
        count = report["acceptance_coverage"]["unmatched"]
        if count:
            reasons.append(f"{count} Form 4 accessions lack exact timestamp coverage in the supplied sidecar")
    blocking = set(available) - {"SEC company facts", "verified quarterly estimate vintages"}
    if all(available[key] for key in blocking):
        market = StudyMarket(bundle, spy, sic)
        span = json.loads((bundle / "coverage.json").read_text())
        first_decision = market.observed[min(253, len(market.observed) - 1)]
        coverage = historical.coverage_check(first_decision, market.observed[-1])
        report["historical_coverage"] = coverage
        reasons.extend(coverage["failed_checks"])
        if audit_only:
            save_json(out / "report.json", report)
            return report
        vintages = read_primary_quarterly(estimates)
        filings = load_filings(paths, sorted(set(market.cik.values())),
                               acceptance_map(acceptances) if acceptances else None)
        report["filings"] = {"rows": len(filings), "exact_timestamp_rows": int(filings["accepted"].notna().sum()),
                             "date_only_rows": int(filings["accepted"].isna().sum()),
                             "amendments": int(filings["document_type"].eq("4/A").sum()),
                             "plan_status": filings["plan_status"].value_counts().to_dict()}
        if not report["filings"]["exact_timestamp_rows"]:
            reasons.append("No accession-level SEC acceptance timestamps")
        report["primary"], _ = history_run(market, filings, vintages, span, out)
        report["date_only_sensitivity"], _ = history_run(market, filings, vintages, span, out, "_date_only", date_only=True)
        report["explicit_non_plan_sensitivity"], _ = history_run(
            market, filings, vintages, span, out, "_explicit_non_plan", explicit_non_plan=True)
        if inferred is not None:
            report["inferred_estimate_sensitivity"], _ = history_run(
                market, filings, inferred, span, out, "_inferred_estimates")
        reasons.extend(report["primary"]["reasons"])
    if inspect_inputs(acceptances, sensitivity_estimates)[-1] != hashes:
        reasons.append("Input files changed during the run. Discard computed statistics and rerun on a stable snapshot.")
    reasons.append("Prospective paper evidence and user review are pending")
    save_json(out / "report.json", report)
    return report


def checked_market(acceptances, future_dates=()):
    bundle, spy, sic, estimates, files, present, hashes = inspect_inputs(acceptances)
    missing = [name for name, exists in present.items() if not exists]
    if missing:
        raise ValueError("missing forward inputs: " + ", ".join(missing))
    market = StudyMarket(bundle, spy, sic, future_dates)
    coverage = historical.coverage_check(market.observed[min(253, len(market.observed) - 1)], market.observed[-1])
    if not coverage["passed"]:
        raise ValueError("historical coverage: " + ", ".join(coverage["failed_checks"]))
    filings = load_filings(files, sorted(set(market.cik.values())),
                           acceptance_map(acceptances) if acceptances else None)
    return market, filings, hashes


def freeze(path, acceptances=None):
    now = pd.Timestamp.now(tz="UTC")
    market, filings, hashes = checked_market(acceptances)
    if market.observed[-1] >= now.tz_convert(NY).tz_localize(None).normalize():
        raise ValueError("freeze requires benchmark history through completed prior calendar dates")
    events, _ = build_events(market, available_filings(filings, market.cal))
    if events.empty:
        raise ValueError("no eligible timestamped events")
    events, _ = label_events(market, events)
    train = events["x"].add(EMBARGO).lt(len(market.observed)) & events["net"].notna()
    if train.sum() < REG["min_train_events"]:
        raise ValueError("insufficient settled training events")
    scores, _ = walk_forward(events)
    radius = interval_radius((scores - events["net"]).abs()[train].dropna())
    if radius is None:
        raise ValueError("insufficient earlier out-of-sample residuals")
    if inspect_inputs(acceptances)[-1] != hashes:
        raise ValueError("input files changed during freeze")
    scaler, ridge = fit(events, train).steps
    save_json(path, {"status": "RESEARCH_ONLY", "frozen_at_utc": now.isoformat(), "trained_events": int(train.sum()),
                     "trained_through": market.observed[-1], "feature_order": FEATURES,
                     "recipe_sha256": digest(PREREG), "sources": source_hashes(), "input_sha256": hashes,
                     "mean": scaler[1].mean_.tolist(), "scale": scaler[1].scale_.tolist(),
                     "coef": ridge[1].coef_.tolist(), "intercept": ridge[1].intercept_, "radius": radius})


def frozen_recipe(path):
    recipe = json.loads(path.read_text())
    if (recipe["recipe_sha256"] != digest(PREREG) or recipe["sources"] != source_hashes()
            or recipe["feature_order"] != FEATURES):
        raise ValueError("current implementation differs from the frozen forward recipe")
    return recipe


def forward_score(recipe_path, sessions, out, acceptances=None):
    recipe = frozen_recipe(recipe_path)
    now = pd.Timestamp.now(tz="UTC")
    local = now.tz_convert(NY).tz_localize(None)
    entry = local.normalize()
    session_dates = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(sessions)["date"], errors="raise"))
    if (not session_dates.is_unique or not session_dates.is_monotonic_increasing
            or not session_dates.equals(session_dates.normalize()) or entry not in session_dates
            or local >= entry + pd.Timedelta(hours=9, minutes=30)):
        raise ValueError("score before 09:30 New York time on a session in the supplied exchange calendar")
    if pd.Timestamp(recipe["frozen_at_utc"]) >= now or pd.Timestamp(recipe["trained_through"]) >= entry:
        raise ValueError("frozen recipe must precede the decision")
    market, filings, hashes = checked_market(acceptances, [entry])
    i = session_dates.get_loc(entry)
    if i < 1 or market.observed[-1] != session_dates[i - 1]:
        raise ValueError("SPY inputs must end on the immediately preceding exchange session")
    overlap_start = max(market.observed[0], session_dates[0])
    observed_overlap = market.observed[(market.observed >= overlap_start) & (market.observed < entry)]
    schedule_overlap = session_dates[(session_dates >= overlap_start) & (session_dates < entry)]
    if not observed_overlap.equals(schedule_overlap):
        raise ValueError("exchange calendar differs from the benchmark history")
    events, rejected = build_events(market, available_filings(filings, market.cal))
    if not events.empty:
        events = events[events["entry_date"].eq(entry)].copy()
        standardized = (events[FEATURES].to_numpy() - recipe["mean"]) / recipe["scale"]
        events["score"] = standardized @ np.array(recipe["coef"]) + recipe["intercept"]
        events["low"] = events["score"] - recipe["radius"]
        events["high"] = events["score"] + recipe["radius"]
    if inspect_inputs(acceptances)[-1] != hashes:
        raise ValueError("input files changed during forward scoring")
    recorded = pd.Timestamp.now(tz="UTC")
    if recorded.tz_convert(NY).tz_localize(None) >= entry + pd.Timedelta(hours=9, minutes=30):
        raise ValueError("scoring finished after the pre-open deadline")
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / f"{entry.date()}.json", {"status": "RESEARCH_ONLY", "entry_date": entry,
                                             "scored_at_utc": recorded.isoformat(),
                                             "frozen_file_sha256": digest(recipe_path),
                                             "session_file_sha256": digest(sessions),
                                             "input_sha256": hashes, "rejected": rejected,
                                             "events": events.to_dict("records")})


def forward_review(recipe_path, directory, out, acceptances=None):
    recipe = frozen_recipe(recipe_path)
    recipe_hash = digest(recipe_path)
    market, _, hashes = checked_market(acceptances)
    records = [json.loads(path.read_text()) for path in sorted(directory.glob("????-??-??.json"))]
    rows, entries = [], []
    for record in records:
        day = pd.Timestamp(record["entry_date"])
        open_at = (day + pd.Timedelta(hours=9, minutes=30)).tz_localize(NY)
        scored = pd.Timestamp(record["scored_at_utc"])
        if (record["frozen_file_sha256"] != recipe_hash or not
                (pd.Timestamp(recipe["frozen_at_utc"]) < scored < open_at)
                or scored.tz_convert(NY).date() != day.date() or day in entries):
            raise ValueError("invalid prospective record timestamp or recipe")
        entries.append(day)
        if day not in market.observed:
            continue
        e = market.observed.get_loc(day)
        if e + H >= len(market.observed):
            continue
        rows.extend({**row, "e": e, "x": e + H, "entry_date": day} for row in record["events"])
    report = {"status": "UNVALIDATED", "input_sha256": hashes, "frozen_file_sha256": recipe_hash,
              "recorded_sessions": len(records), "reasons": ["User review pending"]}
    if entries and max(entries) >= market.observed[max(0, len(market.observed) - H)]:
        report["reasons"].append("Some recorded sessions have not reached their five-session exit")
    if rows:
        events, outcomes = label_events(market, pd.DataFrame(rows).sort_values(["e", "symbol"]).reset_index(drop=True))
        first = market.observed.searchsorted(min(entries))
        last = min(len(market.observed), market.observed.searchsorted(max(entries), side="right") + H)
        expected = market.observed[first:market.observed.searchsorted(max(entries), side="right")]
        if not expected.isin(entries).all():
            report["reasons"].append("Missing prospective session records")
        events["entry_unfilled"] = [pd.isna(market.opens[r.symbol].iloc[r.e]) for r in events.itertuples()]
        variants = [events[events["score"].gt(0)].assign(rank=lambda d: d["score"]),
                    events[events["primary"]].assign(rank=0), events.assign(rank=0)]
        results = {}
        for name, frame in zip(TRIALS, variants):
            daily, _ = portfolio(frame, outcomes, market.spy, slice(first, last), SIZE, CAPACITY, H)
            results[name] = float(daily["net"].mean()) if daily["net"].notna().all() else None
        report["mean_daily_net"] = results
        report["intervals"] = interval_report(events, events[["low", "high"]])
        report["settled_events"] = len(outcomes)
        if len(outcomes) < REG["min_forward_events"]:
            report["reasons"].append("Insufficient settled forward events")
        if (max(entries) - min(entries)).days < REG["min_forward_calendar_days"]:
            report["reasons"].append("Forward record is shorter than 90 calendar days")
        if any(value is None for value in results.values()) or not results[TRIALS[0]] > max(0, *(results[t] for t in TRIALS[1:])):
            report["reasons"].append("Positive net performance versus both forward baselines is unproven")
    else:
        report["reasons"].append("No settled forward scores")
    if inspect_inputs(acceptances)[-1] != hashes:
        raise ValueError("input files changed during forward review")
    save_json(out, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["historical", "freeze", "score", "review"]:
        command = commands.add_parser(name)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--acceptances", type=Path)
        if name == "historical":
            command.add_argument("--sensitivity-estimates", type=Path)
            command.add_argument("--audit-only", action="store_true")
        if name in ("score", "review"):
            command.add_argument("--recipe", type=Path, required=True)
        if name == "score":
            command.add_argument("--sessions", type=Path, required=True)
        if name == "review":
            command.add_argument("--records", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "historical":
        print(json.dumps(json_value(run(args.out, args.acceptances, args.sensitivity_estimates,
                                            audit_only=args.audit_only)), indent=2))
    elif args.command == "freeze":
        freeze(args.out, args.acceptances)
    elif args.command == "score":
        forward_score(args.recipe, args.sessions, args.out, args.acceptances)
    else:
        forward_review(args.recipe, args.records, args.out, args.acceptances)


if __name__ == "__main__":
    main()
