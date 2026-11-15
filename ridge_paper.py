"""Freeze, record and review the isolated 120-session ridge paper trial."""
import argparse
from datetime import datetime, timezone
import json
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config as C
from historical_data import active_bundle, forward_returns
from provenance import _content_hash, dataset_version, panel_identity
from recipe import digest, write_exclusive
from research import fit_predict, hac_mean, long_only_backtest, name_costs, preprocess, sector_neutral, settled_before

from ridge_controls import REG
ROOT = C.ROOT / "ledger" / "large_ridge_120_v1"
NY = ZoneInfo("America/New_York")


def benchmark():
    frame = pd.read_parquet(C.RAW / "prices" / "SPY.parquet").set_index("date").sort_index()
    if not frame.index.is_unique:
        raise ValueError("duplicate benchmark sessions")
    return frame


def transform(panel, features, columns):
    ranked = preprocess(panel, features)
    neutral = sector_neutral(ranked, panel, features)
    matrix = ranked.copy()
    matrix[features] = neutral[features]
    return matrix.reindex(columns=columns, fill_value=0)


def sources():
    return {name: _content_hash(C.ROOT / name) for name in
            ("ridge_paper.py", "research.py", "features.py", "config.py", "holdings_simulation.py",
             "historical_data.py", "provenance.py", "requirements.txt")}


def freeze(root=ROOT):
    reg = json.loads(REG.read_text(encoding="utf-8-sig"))
    version = dataset_version()
    ok, why = panel_identity("large", version)
    if not ok:
        raise ValueError(why)
    if C.RIDGE_ALPHA != reg["ridge_alpha"]:
        raise ValueError("ridge penalty differs from preregistration")
    metadata = json.loads((C.PROC / "model_large.json").read_text())
    panel_path = C.PROC / "panel_large.parquet"
    panel = pd.read_parquet(panel_path)
    cal = benchmark().index
    pos = cal.get_indexer(panel.date)
    if (pos < 0).any():
        raise ValueError("panel dates missing from benchmark")
    h = reg["horizon"]
    y = panel[f"y_{h}"].to_numpy(dtype=float)
    train = settled_before(pos, h, len(cal) - 1) & np.isfinite(y)
    if train.sum() < 5000:
        raise ValueError("insufficient settled training rows")
    features, columns = metadata["feats"], metadata["features"]
    X = transform(panel, features, columns)
    model, _ = fit_predict("ridge", X.loc[train], y[train], X.iloc[:1])
    if dataset_version() != version or not panel_identity("large", version)[0]:
        raise ValueError("training inputs changed during freeze")
    spec = dict(registration=reg, registration_sha256=_content_hash(REG), sources=sources(),
                frozen_at=datetime.now(timezone.utc).isoformat(), training_cutoff=str(cal[-1].date()),
                last_training_decision=str(panel.loc[train, "date"].max().date()),
                dataset_version=version, panel_sha256=_content_hash(panel_path), training_rows=int(train.sum()),
                features=features, columns=columns, coefficient=model.coef_.tolist(), intercept=float(model.intercept_))
    spec["recipe_id"] = digest(spec)
    write_exclusive(root / "recipe.json", spec)
    return spec


def load_recipe(root):
    spec = json.loads((root / "recipe.json").read_text(encoding="utf-8"))
    identity = spec.pop("recipe_id")
    if digest(spec) != identity or spec["registration_sha256"] != _content_hash(REG) or spec["sources"] != sources():
        raise ValueError("frozen recipe or source identity changed")
    return {**spec, "recipe_id": identity}


def record(root=ROOT):
    from features import build_panel_chunked
    from provenance import holidays

    spec = load_recipe(root)
    if (root / "decision.json").exists():
        raise ValueError("paper trial is closed")
    panel, _ = build_panel_chunked(include_labels=False, last_n_dates=1, profile="large", chunk=1200)
    day = pd.Timestamp(panel.date.max())
    day_text = day.date().isoformat()
    X = transform(panel, spec["features"], spec["columns"])
    score = X.to_numpy() @ np.asarray(spec["coefficient"]) + spec["intercept"]
    costs = name_costs(panel, spec["registration"]["order_usd"])
    now = datetime.now(timezone.utc)
    entry_candidates = pd.bdate_range(day + pd.Timedelta(days=1), periods=12)
    closed = holidays(day.date(), entry_candidates[-1].date())
    entry = next(d for d in entry_candidates if d.date() not in closed)
    if (day_text < spec["training_cutoff"] or now.date() < pd.Timestamp(spec["frozen_at"]).date() or not
            day.replace(hour=16).tz_localize(NY) <= now < entry.replace(hour=9, minute=30).tz_localize(NY)):
        raise ValueError("forecast must be recorded after the decision close and before the next open")
    rows = pd.DataFrame({"symbol": panel.symbol, "score": score, "cost": costs})
    rows = rows.loc[np.isfinite(rows.score) & np.isfinite(rows.cost) & rows.cost.ge(0)]
    if rows.empty or rows.symbol.duplicated().any():
        raise ValueError("forecast requires a unique eligible cross-section")
    doc = dict(recipe_id=spec["recipe_id"], decision=day_text, recorded_at=now.isoformat(),
               forecasts=rows.sort_values("symbol").to_dict("records"))
    write_exclusive(root / "forecasts" / f"{day_text}.json", doc)
    return doc


def review_gate(stats, net, cohorts, reg, missing=False):
    if stats["n"] < reg["min_scored_dates"] or cohorts < reg["min_settled_cohorts"] or missing:
        return "INCONCLUSIVE"
    if (stats["mean"] and stats["mean"] > 0 and stats["t"] is not None and
            np.isfinite(stats["t"]) and stats["t"] >= reg["min_hac_t"] and net is not None and net > 0):
        return "PAPER_TEST_ELIGIBLE"
    return "FAILED"


def evaluate(spec, records, spy, opens, terminal):
    reg = spec["registration"]
    h = reg["horizon"]
    cal = spy.index
    adjusted = spy.open * spy.adjclose / spy.close
    frames, correlations, schedule = [], [], []
    next_entry = 0
    seen = set()
    for doc in sorted(records, key=lambda row: row["decision"]):
        if doc["recipe_id"] != spec["recipe_id"]:
            continue
        day = pd.Timestamp(doc["decision"])
        if day in seen:
            raise ValueError("duplicate decision date")
        seen.add(day)
        e = cal.searchsorted(day, side="right")
        x = e + h
        if day.date().isoformat() < spec.get("training_cutoff", "") or day not in cal or x >= len(cal):
            continue
        recorded = pd.Timestamp(doc["recorded_at"])
        if not (recorded.tzinfo and recorded >= pd.Timestamp(spec["frozen_at"]) and
                day.replace(hour=16).tz_localize(NY) <= recorded <
                cal[e].replace(hour=9, minute=30).tz_localize(NY)):
            continue
        frame = pd.DataFrame(doc["forecasts"])
        if frame.symbol.duplicated().any():
            raise ValueError("duplicate forecast security")
        prices = opens.reindex(index=cal, columns=frame.symbol)
        bench = adjusted.iloc[x] / adjusted.iloc[e] - 1
        if not np.isfinite(bench) or min(adjusted.iloc[x], adjusted.iloc[e]) <= 0:
            raise ValueError("invalid benchmark outcome")
        frame["y"] = forward_returns(prices, terminal, [e], [x])[0] - bench
        frame["entry_price"] = prices.iloc[e].to_numpy()
        frame["benchmark"] = bench
        frame["date"], frame["position"] = day, e - 1
        frames.append(frame)
        valid = frame.dropna(subset=["y"])
        if len(valid) >= 20:
            correlations.append(valid.score.rank().corr(valid.y.rank()))
        if e >= next_entry:
            schedule.append(day)
            next_entry = x

    stats = hac_mean(correlations, h)
    stats = {k: None if v is not None and not np.isfinite(v) else v for k, v in stats.items()}
    ledgers, fills = {}, {}
    if frames:
        frame = pd.concat(frames, ignore_index=True)
        for label, multiplier in (("base", 1), ("double", reg["cost_sensitivity_multiplier"])):
            trades = []
            ledgers[label] = long_only_backtest(
                frame.score, frame.y, frame.date, frame.symbol, frame.cost, h,
                top=reg["entry_fraction"], cost_mult=multiplier, rebalance_dates=schedule,
                benchmark_returns=frame.benchmark, session_positions=frame.position,
                entry_prices=frame.entry_price, buffer_top=reg["retention_fraction"], trades=trades)
            fills[label] = trades
    base = ledgers.get("base", pd.DataFrame())
    missing = base.attrs.get("missing_selected_outcomes", [])
    net = float(base.net.mean()) if len(base) else None
    summary = {"recipe_id": spec["recipe_id"], "forward_ic": stats, "settled_cohorts": len(base),
               "forecast_records": len(seen),
               "mean_gross_excess": float(base.gross.mean()) if len(base) else None,
               "mean_turnover": float(base.turnover.mean()) if len(base) else None,
               "mean_net_excess": net, "missing_selected_outcomes": missing,
               "double_cost_mean_net_excess": float(ledgers["double"].net.mean()) if len(base) else None,
               "status": review_gate(stats, net, len(base), reg, bool(missing))}
    return summary, ledgers, fills


def review(root=ROOT):
    decision_path = root / "decision.json"
    if decision_path.exists():
        return json.loads(decision_path.read_text(encoding="utf-8"))
    spec = load_recipe(root)
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((root / "forecasts").glob("*.json"))]
    prices = {}
    for symbol in sorted({row["symbol"] for doc in records for row in doc["forecasts"]}):
        path = C.RAW / "prices" / f"{symbol}.parquet"
        if path.exists():
            price = pd.read_parquet(path, columns=["date", "adj_open"]).set_index("date")
            if not price.index.is_unique:
                raise ValueError(f"duplicate sessions in {symbol}")
            prices[symbol] = price.adj_open
    bundle = active_bundle()
    terminal = pd.read_parquet(bundle / "terminal.parquet") if bundle else pd.DataFrame()
    summary, ledgers, fills = evaluate(spec, records, benchmark(), pd.DataFrame(prices), terminal)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = root / "reviews" / stamp
    folder.mkdir(parents=True, exist_ok=False)
    for label, ledger in ledgers.items():
        ledger.to_parquet(folder / f"{label}_periods.parquet")
        pd.DataFrame(fills[label]).to_parquet(folder / f"{label}_fills.parquet")
    write_exclusive(folder / "report.json", summary)
    if summary["status"] != "INCONCLUSIVE":
        write_exclusive(decision_path, {**summary, "review": stamp})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "record", "review"))
    args = parser.parse_args()
    lock = C.hold_lock()
    result = {"freeze": freeze, "record": record, "review": review}[args.action]()
    print(json.dumps({"recipe_id": result["recipe_id"], "status": result.get("status", args.action)}, indent=2))
