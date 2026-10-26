"""Market-neutral long/short research (quant hedge-fund construction) with the frozen model recipe.

Out-of-sample predictions: the ensemble members of model_<profile>.json are re-fitted with yearly expanding
walk-forward refits (same purge/embargo as research.py) plus one fit for the holdout block; predictions are cached
in data/processed/oos_preds_<profile>_h<h>.parquet.

Portfolio, rebalanced every DECISION_STEP (5) sessions and scored on non-overlapping 5-session open-to-open returns:
  * sector-neutral selection: within each sector, long the top LS_Q of the ensemble rank, short the bottom LS_Q;
    held names stay until they leave the top/bottom LS_BUFFER (turnover buffer)
  * equal weight inside each leg; leg sizes set so long beta = short beta (Blume-adjusted 252d beta) and
    long + short gross = LS_GROSS (1.5x equity)
  * shorts excluded when short interest / shares > 15% (hard to borrow); borrow fee 0.3% / 1% / 3% a year by
    short-interest tier
  * costs: max(half Corwin-Schultz spread, 3bp) + square-root impact at the order size; no FX fee (USD account),
    charged on every unit of weight traded
The same rules drive the live targets (ls_signals.py) and the Alpaca paper bot.
Also reports the long-only top-quintile strategy (the Trading 212 one) on an absolute basis next to SPY.
Output: reports/ls_research_<profile>.json and reports/ls_backtest_<profile>.parquet.
"""
import json
import math
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config as C
from research import EMBARGO, MISSING_GROUPS, date_rank, name_costs, preprocess, sector_neutral, walk_forward

import ls_rules as R  # the same selection rules the bot uses
LS_Q, LS_BUFFER, LS_GROSS, SI_MAX_SHORT = R.LS_Q, R.LS_BUFFER, R.LS_GROSS, R.SI_MAX_SHORT
REBALANCE_SESSIONS = 20  # as the Alpaca bot
EQUITY_USD = 100_000.0
PERIODS_PER_YEAR = 252 / C.DECISION_STEP


def borrow_fee(si):
    si = np.nan_to_num(si, nan=0.0)
    return np.where(si < 0.05, 0.003, np.where(si < 0.10, 0.01, 0.03))


def blume(beta):
    return 0.67 * np.nan_to_num(beta, nan=1.0) + 0.33


def oos_predictions(profile, panel, model, h, cal):
    f = C.PROC / f"oos_preds_{profile}_h{h}.parquet"
    if f.exists():
        p = pd.read_parquet(f)
        if len(p) == len(panel):
            return p["wf"].to_numpy(), p["holdout"].to_numpy()
    feats = model["feats"]
    X = preprocess(panel, feats)
    Xn = sector_neutral(X, panel, feats)
    Xm = pd.concat([Xn[feats], X[[k for k in MISSING_GROUPS if k in X]]], axis=1).reindex(columns=model["features"], fill_value=0.0)
    del X, Xn
    dates, tpos = panel["date"].to_numpy(), panel["tpos"].to_numpy()
    sectors = panel["sector"].fillna("Unknown").to_numpy()
    hpos = int(cal.searchsorted(cal[-1] - pd.DateOffset(years=C.HOLDOUT_YEARS)))
    cost = name_costs(panel, model.get("order_size_usd") or 600.0)
    y = panel[f"y_{h}"].to_numpy().astype("float64")
    wf, ho = [], []
    for n in model["horizons"][str(h)]["members"]:
        if n == "factor_resid":
            continue  # needs the style neutraliser fitted on the full panel; the remaining members carry the ensemble
        p_wf, p_ho = walk_forward(n, Xm, y, dates, tpos, h, hpos, sectors, float(cost.mean() * 2))
        wf.append(date_rank(p_wf, dates))
        ho.append(date_rank(p_ho, dates))
        print(profile, "h", h, n, "oos done", flush=True)
    wf, ho = np.nanmean(np.stack(wf), axis=0), np.nanmean(np.stack(ho), axis=0)
    pd.DataFrame({"wf": wf, "holdout": ho}).to_parquet(f, index=False)
    return wf, ho


def ls_backtest(df, spy_ret, every=None):
    """df: date, symbol, sector, score, y (5-session excess return), beta, si, cost. Selection uses the live rules
    (ls_rules.select, incl. the turnover buffer) and is redone only every REBALANCE_SESSIONS (as the Alpaca bot does);
    between rebalances the weights are held and scored on the non-overlapping 5-session returns. Returns a per-period frame."""
    every = every or max(1, REBALANCE_SESSIONS // C.DECISION_STEP)
    rows, w = [], {}
    for k, (d, g) in enumerate(df.groupby("date", sort=True)):
        rm = float(spy_ret.get(d, np.nan))
        if math.isnan(rm):
            continue
        g = g.dropna(subset=["y"])
        c = dict(zip(g["symbol"], g["cost"]))
        traded = turnover = 0.0
        if k % every == 0 or not w:
            nw = R.select(g, {s for s, x in w.items() if x > 0}, {s for s, x in w.items() if x < 0})
            if nw:
                keys = set(nw) | set(w)
                traded = sum(abs(nw.get(s, 0.0) - w.get(s, 0.0)) * c.get(s, 0.003) for s in keys)
                turnover = sum(abs(nw.get(s, 0.0) - w.get(s, 0.0)) for s in keys) / 2
                w = nw
        if not w:
            continue
        y, si = dict(zip(g["symbol"], g["y"])), dict(zip(g["symbol"], g["si"]))
        longs, shorts = [s for s, x in w.items() if x > 0], [s for s, x in w.items() if x < 0]
        gross = sum(x * (y.get(s, 0.0) + rm) for s, x in w.items())  # a held name without a label earns the market
        borrow = sum(-w[s] * float(borrow_fee([si.get(s, 0.0)])[0]) for s in shorts) / PERIODS_PER_YEAR
        rows.append({"date": d, "gross": gross, "cost": traded, "borrow": borrow, "net": gross - traded - borrow, "spy": rm,
                     "long_excess": float(np.mean([y.get(s, 0.0) for s in longs])), "short_excess": float(np.mean([y.get(s, 0.0) for s in shorts])),
                     "n_long": len(longs), "n_short": len(shorts), "gross_long": sum(w[s] for s in longs), "gross_short": -sum(w[s] for s in shorts),
                     "turnover": turnover, "rebalanced": turnover > 0})
    return pd.DataFrame(rows).set_index("date")


def long_only_abs(df, spy_ret_h, h):
    """The current long-only strategy (top quintile, equal weight, T212 costs incl. FX), absolute returns."""
    rows, prev = [], set()
    ud = np.sort(df["date"].unique())[:: max(1, h // C.DECISION_STEP)]
    for d in ud:
        g = df[(df["date"] == d)].dropna(subset=["score", "y"])
        if len(g) < 20 or d not in spy_ret_h:
            continue
        pick = g.nlargest(max(1, int(round(len(g) * C.SIGNAL_TOP_QUANTILE))), "score")
        names = dict(zip(pick["symbol"], pick["cost_fx"]))
        traded = set(names) ^ prev
        cost = sum(names.get(s, float(g["cost_fx"].mean())) for s in traded) / len(names)
        rm = spy_ret_h[d]
        rows.append({"date": d, "net": float(pick["y"].mean()) + rm - cost, "spy": rm})
        prev = set(names)
    return pd.DataFrame(rows).set_index("date")


def stats(r, spy=None, ppy=PERIODS_PER_YEAR):
    r = pd.Series(r).dropna()
    if len(r) < 8:
        return None
    eq = (1 + r).cumprod()
    yrs = len(r) / ppy
    out = {"periods": int(len(r)), "years": round(yrs, 2), "total_return": float(eq.iloc[-1] - 1), "cagr": float(eq.iloc[-1] ** (1 / yrs) - 1),
           "vol_ann": float(r.std() * math.sqrt(ppy)), "sharpe_ann": float(r.mean() / r.std() * math.sqrt(ppy)) if r.std() > 0 else None,
           "max_drawdown": float((eq / eq.cummax() - 1).min()), "hit_rate": float((r > 0).mean())}
    neg = r[r < 0]
    out["sortino_ann"] = float(r.mean() / math.sqrt((neg ** 2).mean()) * math.sqrt(ppy)) if len(neg) else None
    if spy is not None:
        s = pd.Series(spy).reindex(r.index)
        ok = s.notna()
        if ok.sum() > 8:
            b = np.polyfit(s[ok], r[ok], 1)
            out["beta_to_spy"], out["corr_to_spy"] = float(b[0]), float(np.corrcoef(s[ok], r[ok])[0, 1])
    out["by_year"] = {str(y): float((1 + g).prod() - 1) for y, g in r.groupby(pd.DatetimeIndex(r.index).year)}
    return out


def run(profile="broad"):
    model = json.loads((C.PROC / f"model_{profile}.json").read_text())
    panel = pd.read_parquet(C.PROC / f"panel_{profile}.parquet")
    spy = pd.read_parquet(C.RAW / "prices" / f"{C.BENCHMARK}.parquet").drop_duplicates("date").set_index("date")
    cal = spy.index
    panel["tpos"] = cal.get_indexer(panel["date"])
    hpos = int(cal.searchsorted(cal[-1] - pd.DateOffset(years=C.HOLDOUT_YEARS)))
    hold_start = cal[hpos]
    op = spy["adj_open"]  # dividend-adjusted opens, same convention as the labels

    def spy_fwd(h):
        i = np.arange(len(cal))
        ok = i + 1 + h < len(cal)
        return dict(zip(cal[i[ok]], op.to_numpy()[i[ok] + 1 + h] / op.to_numpy()[i[ok] + 1] - 1))

    out = {"profile": profile, "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "recipe_run_id": model["run_id"],
           "holdout_start": str(hold_start.date()), "method": __doc__.strip(),
           "params": {"q": LS_Q, "buffer": LS_BUFFER, "gross": LS_GROSS, "si_max_short": SI_MAX_SHORT, "equity_usd": EQUITY_USD,
                      "rebalance_sessions": REBALANCE_SESSIONS, "max_name_weight": R.MAX_NAME_WEIGHT}, "results": {}}
    base = panel[["date", "symbol", "sector", "beta_252", "si_ratio", "y_5", "y_20"]].copy()
    base["sector"] = base["sector"].fillna("Unknown")
    n_names = max(1.0, panel.groupby("date").size().mean() * 2 * LS_Q)
    order_usd = EQUITY_USD * LS_GROSS / n_names
    base["cost"] = name_costs(panel, order_usd).to_numpy() - C.FX_FEE
    base["cost_fx"] = name_costs(panel, model.get("order_size_usd") or 600.0).to_numpy()
    out["params"]["order_usd_per_name"] = order_usd
    s5, s20 = spy_fwd(5), spy_fwd(20)
    scores = {}
    for h in (5, 20):
        wf, ho = oos_predictions(profile, panel, model, h, cal)
        scores[h] = np.where(np.isnan(wf), ho, wf)
    scores["blend"] = np.nanmean(np.stack([scores[5], scores[20]]), axis=0)
    series = []
    for key, sc in scores.items():
        df = base.assign(score=sc, y=base["y_5"], beta=base["beta_252"], si=base["si_ratio"])
        bt = ls_backtest(df[["date", "symbol", "sector", "score", "y", "beta", "si", "cost"]], s5)
        res = {}
        for lab, part in [("walk_forward", bt[bt.index < hold_start]), ("holdout", bt[bt.index >= hold_start])]:
            st = stats(part["net"], part["spy"])
            if st:
                st.update({"mean_gross_per_period": float(part["gross"].mean()), "mean_cost_per_period": float(part["cost"].mean()),
                           "mean_borrow_per_period": float(part["borrow"].mean()), "mean_turnover": float(part["turnover"].mean()),
                           "avg_long": float(part["n_long"].mean()), "avg_short": float(part["n_short"].mean()),
                           "long_leg_excess_ann": float(part["long_excess"].mean() * PERIODS_PER_YEAR),
                           "short_leg_excess_ann": float(part["short_excess"].mean() * PERIODS_PER_YEAR)})
            res[lab] = st
        out["results"][f"ls_rank_{key}"] = res
        from research import hac_mean
        wf_stats = hac_mean(bt.loc[bt.index < hold_start, "net"], max(1, int(20 / C.DECISION_STEP)))
        ho_stats = hac_mean(bt.loc[bt.index >= hold_start, "net"], max(1, int(20 / C.DECISION_STEP)))
        failed = []
        for sample, stat in [("walk_forward", wf_stats), ("historical_holdout", ho_stats)]:
            if not ((stat.get("mean") or 0) > 0 and (stat.get("t") or 0) >= 2):
                failed.append(f"{sample} cost-adjusted mean not > 0 with HAC t >= 2")
        from historical_data import coverage_check
        coverage = coverage_check(panel["date"].min(), panel["date"].max())
        failed.extend(coverage["failed_checks"])
        out.setdefault("gates", {})[f"ls_rank_{key}"] = {"status": "PAPER_TEST_ELIGIBLE" if not failed else "UNVALIDATED",
                         "failed_checks": failed, "walk_forward_net": wf_stats, "historical_holdout_net": ho_stats}
        series.append(bt.assign(score=str(key)))
        print(profile, "L/S", key, {k: {m: round(v, 3) for m, v in (r or {}).items() if m in ("cagr", "sharpe_ann", "max_drawdown", "beta_to_spy")} for k, r in res.items()}, flush=True)
    # current long-only strategy, absolute
    lo = long_only_abs(base.assign(score=scores[20], y=base["y_20"]), s20, 20)
    ppy20 = 252 / 20
    spy_s = pd.Series(s20).reindex(lo.index)
    out["results"]["long_only_top_quintile_h20"] = {lab: {"strategy": stats(p["net"], p["spy"], ppy20), "spy": stats(p["spy"], None, ppy20)}
                                                    for lab, p in [("walk_forward", lo[lo.index < hold_start]), ("holdout", lo[lo.index >= hold_start])]}
    pd.concat(series).reset_index().to_parquet(C.REPORTS / f"ls_backtest_{profile}.parquet", index=False)
    (C.REPORTS / f"ls_research_{profile}.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    _lock = __import__("config").hold_lock()
    for p in sys.argv[1:] or ["broad"]:
        run(p)
