"""Paper-trading validation (spec 9.1 step 5, 12.2): forward evidence nobody has optimised against.

1. Forward record  every night the ensemble rank of every security, per profile and horizon, is appended to
                   processed/forecast_history.parquet. Once a forecast's horizon has elapsed it is scored against the
                   realised excess return (open after the as-of date -> open h sessions later, minus SPY): forward IC.
2. Shadow book     a virtual equal-weight top-quintile portfolio per profile and horizon, net of each name's
                   round-trip cost; non-overlapping cohorts give the forward net return series.
3. Executed book   the bot's demo orders (/opt/t212-bot/trades.jsonl) matched to Trading 212 fills (read-only order
                   history): fill prices, FX fees, slippage vs the pre-trade price, realised return vs SPY.
4. Promotion       a model is reported READY_FOR_LIVE_REVIEW only after >= 63 sessions of paper trading, >= 20
                   closed trades, positive mean net excess with t >= 2, and forward IC > 0. This is a recommendation
                   for a human: nothing here changes a gate, the bot or live execution.
"""
import json
import math
from datetime import datetime, timezone

import httpx
import numpy as np
import pandas as pd

import config as C
import risk_limits
import gates

HIST = C.PROC / "forecast_history.parquet"
BOT_TRADES = __import__("pathlib").Path("/opt/t212-bot/trades.jsonl")
MIN_SESSIONS, MIN_TRADES, MIN_T = 63, 20, 2.0


def current_history(history, run_ids, models=None):
    """Never stamp old/unattributed observations as evidence for a new recipe."""
    if "model_run_id" not in history:
        return history.iloc[:0].copy()
    expected = history["profile"].map(run_ids)
    valid = expected.notna() & history["model_run_id"].eq(expected)
    if models is not None:
        if not {"recipe_id", "recorded_utc"} <= set(history):
            return history.iloc[:0].copy()
        expected_recipe = history["profile"].map({p: m.get("recipe_id") for p, m in models.items()})
        start = pd.to_datetime(history["profile"].map({p: m.get("forward_start_utc") for p, m in models.items()}), utc=True, errors="coerce")
        recorded = pd.to_datetime(history["recorded_utc"], utc=True, errors="coerce")
        valid &= expected_recipe.notna() & history["recipe_id"].eq(expected_recipe) & (recorded >= start)
        asof = pd.to_datetime(history["as_of"], errors="coerce")
        freeze_day = start.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
        recorded_day = recorded.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
        valid &= (asof >= freeze_day) & (asof <= recorded_day)
    return history[valid].copy()


def record():
    """Append tonight's forecasts (all names, all horizons) to the forward record."""
    f = C.REPORTS / "forecast_latest.parquet"
    sig = json.loads((C.SIGNALS / "latest.json").read_text())
    asof = {m["profile"]: m["as_of"] for m in sig.get("profiles", [])}
    run_ids = {m["profile"]: m.get("model_run_id") for m in sig.get("profiles", [])}
    fc = pd.read_parquet(f)
    if fc.empty:
        return 0
    if "model_run_id" not in fc or not fc["model_run_id"].eq(fc["profile"].map(run_ids)).all():
        raise ValueError("forecast artifact lacks matching model recipe identity; regenerate signals")
    fc["as_of"] = pd.to_datetime(fc["profile"].map(asof))
    fc["recorded_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if "recipe_id" not in fc:
        fc["recipe_id"] = None  # legacy artifacts remain diagnostic only
    if "weights_id" not in fc:
        fc["weights_id"] = None
    keep = ["as_of", "profile", "horizon", "symbol", "ensemble_rank", "decile", "exp_excess", "cost_round_trip", "gate", "recorded_utc", "model_run_id", "recipe_id", "weights_id"]
    fc = fc[keep]
    if HIST.exists():
        old = pd.read_parquet(HIST)
        if "model_run_id" not in old:
            old["model_run_id"] = None  # preserve legacy history without inventing provenance
        if "recipe_id" not in old:
            old["recipe_id"] = None
        keys = ["as_of", "profile", "horizon", "symbol", "model_run_id", "recipe_id"]
        fc = pd.concat([old, fc[~fc.set_index(keys).index.isin(old.set_index(keys).index)]], ignore_index=True)
    fc.to_parquet(HIST, index=False)
    return int(fc["as_of"].nunique())


def realised(hist):
    """Open-to-open excess return over each forecast's horizon (NaN until it has fully elapsed)."""
    if hist.empty:
        return np.array([], dtype=float)
    spy = pd.read_parquet(C.RAW / "prices" / f"{C.BENCHMARK}.parquet").drop_duplicates("date").set_index("date")
    cal = spy.index
    so = (spy["open"] * spy["adjclose"] / spy["close"]).to_numpy()
    out = np.full(len(hist), np.nan)
    for sym, idx in hist.groupby("symbol").indices.items():
        f = C.RAW / "prices" / f"{sym}.parquet"
        if not f.exists():
            continue
        px = pd.read_parquet(f, columns=["date", "adj_open"]).drop_duplicates("date").set_index("date")["adj_open"].reindex(cal).to_numpy()
        for i in idx:
            e = cal.searchsorted(hist["as_of"].iat[i], side="right")
            x = e + int(hist["horizon"].iat[i])
            if x < len(cal) and np.isfinite(px[e]) and np.isfinite(px[x]):
                out[i] = px[x] / px[e] - 1 - (so[x] / so[e] - 1)
    return out


def shadow(hist):
    res = {}
    for (prof, h), g in hist.dropna(subset=["y"]).groupby(["profile", "horizon"]):
        ic = g.groupby("as_of").apply(lambda d: d["ensemble_rank"].rank().corr(d["y"].rank()), include_groups=False).dropna()
        top = g[g["ensemble_rank"] >= 1 - C.SIGNAL_TOP_QUANTILE]
        coh = top.groupby("as_of").apply(lambda d: (d["y"] - d["cost_round_trip"]).mean(), include_groups=False)
        dates = sorted(coh.index)
        nonover = coh.loc[dates[::h]]  # cohorts h sessions apart do not overlap
        n = len(nonover)
        from research import hac_mean
        hic = hac_mean(ic, max(1, int(h)))  # daily forecasts with h-day labels overlap: Newey-West with h lags
        res[f"{prof}_h{h}"] = {"scored_dates": int(len(ic)), "forward_ic_mean": float(ic.mean()) if len(ic) else None,
                               "forward_ic_t": hic["t"], "forward_ic_se": hic["se"],
                               "top_quintile_net_mean": float(nonover.mean()) if n else None,
                               "top_quintile_net_t": float(nonover.mean() / nonover.std() * math.sqrt(n)) if n > 2 and nonover.std() > 0 else None,
                               "non_overlapping_cohorts": int(n)}
    return res


def forward_gate(sh):
    """Deployment gate from the forward record only (spec 9.1 step 5): no reused holdout.
    PAPER_TEST_ELIGIBLE when >= MIN_SESSIONS scored forecast dates, forward IC > 0 with HAC t >= MIN_T, and the
    cost-adjusted top-quintile shadow book has a positive mean over >= 3 non-overlapping cohorts."""
    gate = {}
    for key, v in sh.items():
        prof, h = key.rsplit("_h", 1)
        fails = []
        if v["scored_dates"] < MIN_SESSIONS:
            fails.append(f"forward record {v['scored_dates']}/{MIN_SESSIONS} scored dates")
        else:
            if not ((v["forward_ic_mean"] or 0) > 0 and (v["forward_ic_t"] or 0) >= MIN_T):
                fails.append(f"forward IC {v['forward_ic_mean']:+.4f} (HAC t={v['forward_ic_t'] if v['forward_ic_t'] is None else round(v['forward_ic_t'], 2)}) not > 0 with t >= {MIN_T}")
            if v["non_overlapping_cohorts"] < 3 or (v["top_quintile_net_mean"] or 0) <= 0:
                fails.append("shadow top quintile not positive after costs")
        gate.setdefault(prof, {})[h] = {"status": "PAPER_TEST_ELIGIBLE" if not fails else "UNVALIDATED", "failed_checks": fails,
                                        "forward_ic": v["forward_ic_mean"], "forward_ic_se": v.get("forward_ic_se"), "scored_dates": v["scored_dates"],
                                        "shadow_net_mean": v["top_quintile_net_mean"], "source": "forward record (paper_trading.py)"}
    return gate


def executed(run_ids=None):
    """Pair the bot's research-signal buys with later sells and Trading 212 fills."""
    if not BOT_TRADES.exists():
        return {"trades": 0}
    rows = [json.loads(l) for l in BOT_TRADES.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("dry_run")]
    skips = [r for r in rows if r["kind"] == "SKIP"]
    lim = risk_limits.load()
    kid, sec = risk_limits._env_file("/root/.hermes/.env", "T212_API_KEY_ID"), risk_limits._env_file("/root/.hermes/.env", "T212_API_SECRET")
    fills = {}
    with httpx.Client(timeout=30, auth=(kid, sec)) as c:
        host = lim["base_url"].split("/api/")[0]  # demo host (from the bot's own configuration)
        path = "/api/v0/equity/history/orders?limit=50"
        for page in range(10):  # read-only order history, newest first; endpoint allows ~6 requests/minute
            if page:
                import time
                time.sleep(11)
            j = c.get(host + path).json()
            for it in j.get("items", []):
                o, f = it.get("order", {}), it.get("fill") or {}
                fee = sum(-t["quantity"] for t in (f.get("walletImpact") or {}).get("taxes", []) if t.get("name") == "CURRENCY_CONVERSION_FEE")
                fills[o.get("id")] = {"price": f.get("price"), "qty": f.get("quantity"), "at": f.get("filledAt"), "fx_fee_gbp": fee,
                                      "net_gbp": (f.get("walletImpact") or {}).get("netValue")}
            path = j.get("nextPagePath")
            if not path:
                break
    spy = pd.read_parquet(C.RAW / "prices" / f"{C.BENCHMARK}.parquet").drop_duplicates("date").set_index("date")["adjclose"]
    trades, open_buys = [], {}
    for r in rows:
        if r["kind"] == "BUY" and r.get("reason") == "research_signal":
            open_buys[r["ticker"]] = r
        elif r["kind"] == "SELL" and r["ticker"] in open_buys:
            b = open_buys.pop(r["ticker"])
            if run_ids is not None:
                prefix = "mosaic-xs-ridge-"
                mid = b.get("model_id") or ""
                profile = mid[len(prefix):] if mid.startswith(prefix) else None
                if not profile or not b.get("research_run_id") or b["research_run_id"] != run_ids.get(profile):
                    continue  # preserve raw logs; legacy/different-recipe fills are not current-model evidence
            fb, fs = fills.get(b.get("order_id"), {}), fills.get(r.get("order_id"), {})
            if not fb.get("price") or not fs.get("price"):
                continue
            d0, d1 = pd.Timestamp(fb["at"][:10]), pd.Timestamp(fs["at"][:10])
            spy_ret = float(spy.asof(d1) / spy.asof(d0) - 1)
            gross = fs["price"] / fb["price"] - 1
            fees = (fb.get("fx_fee_gbp") or 0) + (fs.get("fx_fee_gbp") or 0)
            notional = abs(fb.get("net_gbp") or 0) or 1
            trades.append({"ticker": r["ticker"], "signal_id": b.get("signal_id"), "model_id": b.get("model_id"), "horizon": b.get("horizon"), "entry": fb["at"], "exit": fs["at"],
                           "gross_return": gross, "spy_return": spy_ret, "fees_frac": fees / notional, "net_excess": gross - spy_ret - fees / notional,
                           "slippage_vs_pretrade": fb["price"] / b["price"] - 1 if b.get("price") else None, "exit_reason": r.get("reason"),
                           "expected_net": b.get("expected_net")})
    t = pd.DataFrame(trades)
    out = {"closed_trades": int(len(t)), "open_signal_positions": sorted(open_buys), "skipped_by_pretrade_check": len(skips),
           "skip_reasons": pd.Series([s["reason"].split(" (")[0] for s in skips]).value_counts().head(5).to_dict() if skips else {}}
    if len(t):
        first = pd.Timestamp(t["entry"].min()[:10])
        out |= {"first_trade": str(first.date()), "sessions_since_first": int(np.busday_count(first.date(), datetime.now().date())),
                "mean_net_excess": float(t["net_excess"].mean()), "t": float(t["net_excess"].mean() / t["net_excess"].std() * math.sqrt(len(t))) if len(t) > 2 and t["net_excess"].std() > 0 else None,
                "hit_rate": float((t["net_excess"] > 0).mean()), "mean_fees": float(t["fees_frac"].mean()),
                "mean_slippage": float(t["slippage_vs_pretrade"].dropna().mean()) if t["slippage_vs_pretrade"].notna().any() else None,
                "mean_expected_net": float(t["expected_net"].dropna().mean()) if t["expected_net"].notna().any() else None,
                "by_model": {m: {"n": int(len(g)), "mean_net_excess": float(g["net_excess"].mean()),
                                  "horizons": sorted(g["horizon"].dropna().astype(int).unique().tolist()) if g["horizon"].notna().all() else [],
                                  "t": float(g["net_excess"].mean() / g["net_excess"].std() * math.sqrt(len(g))) if len(g) > 2 and g["net_excess"].std() > 0 else None,
                                  "sessions_since_first": __import__("provenance").sessions_between(pd.Timestamp(g["entry"].min()[:10]).date(), datetime.now(timezone.utc).date())}
                             for m, g in t.groupby("model_id")}}
        t.to_parquet(C.REPORTS / "paper_trades.parquet", index=False)
    return out


def run():
    n_dates = record()
    run_ids = {p: json.loads((C.PROC / f"model_{p}.json").read_text())["run_id"] for p in C.PROFILES if (C.PROC / f"model_{p}.json").exists()}
    models = {p: json.loads((C.PROC / f"model_{p}.json").read_text()) for p in run_ids}
    hist = current_history(pd.read_parquet(HIST), run_ids, models) if HIST.exists() else pd.DataFrame(columns=["profile", "horizon", "as_of", "symbol", "ensemble_rank", "y", "cost_round_trip"])
    hist["y"] = realised(hist)
    sh = shadow(hist)
    ex = executed(run_ids)
    gate = forward_gate(sh)
    tmp = C.PROC / "forward_gate.tmp"
    from provenance import dataset_version
    tmp.write_text(json.dumps({"generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "model_run_ids": run_ids, "recipe_ids": {p: m.get("recipe_id") for p, m in models.items()}, "dataset_version": dataset_version(),
                               "last_forecast_date": str(pd.Timestamp(hist["as_of"].max()).date()) if len(hist) else None,
                               "rule": f">= {MIN_SESSIONS} scored dates, forward IC > 0 with HAC t >= {MIN_T}, shadow top quintile net > 0",
                               "profiles": gate}, indent=1, default=float))
    tmp.replace(C.PROC / "forward_gate.json")
    promo = {}
    for key, s in sh.items():
        ok_fwd = (s["forward_ic_mean"] or 0) > 0 and (s["forward_ic_t"] or 0) >= MIN_T
        promo[key] = "INSUFFICIENT_FORWARD_HISTORY" if s["scored_dates"] < MIN_SESSIONS else ("FORWARD_EVIDENCE_POSITIVE" if ok_fwd else "FORWARD_EVIDENCE_NEGATIVE")
    ready, review_failures = gates.live_review_ready(ex, sh, C.PRIMARY_HORIZON, MIN_SESSIONS, MIN_TRADES, MIN_T)
    import recipe
    from deployment_policy import cumulative
    from signals import audit_ok
    current_dataset = dataset_version()
    for profile, model in models.items():
        audit = audit_ok(current_dataset, profile)
        frozen, reason = recipe.check(model)
        historical = model.get("horizons", {}).get(str(C.PRIMARY_HORIZON), {}).get("gate")
        forward = gate.get(profile, {}).get(str(C.PRIMARY_HORIZON))
        combined = cumulative(historical, forward, data_ok=audit[0], recipe_ok=frozen)
        if combined["failed_checks"]:
            ready = False
            review_failures.extend(f"{profile}: {x}" for x in combined["failed_checks"])
    out = {"generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "forecast_dates_recorded": n_dates,
           "shadow_book": sh, "shadow_status": promo, "executed_demo_book": ex,
           "live_review_recommendation": "READY_FOR_LIVE_REVIEW (human decision required)" if ready else "NOT_READY",
           "live_review_failed_checks": review_failures,
           "rule": f"Per current recipe/model: >= {MIN_SESSIONS} sessions and >= {MIN_TRADES} closed demo trades with mean net excess > 0 and t >= {MIN_T}; matching {C.PRIMARY_HORIZON}d forward IC > 0 with HAC t >= {MIN_T}, >= {MIN_SESSIONS} scored dates and >= 3 positive-net shadow cohorts. Never automatic."}
    (C.REPORTS / "paper_trading.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "shadow_book"}, indent=1, default=float))
    return out


if __name__ == "__main__":
    _lock = __import__("config").hold_lock()
    run()
