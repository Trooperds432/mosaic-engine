"""Matched random baselines and block-bootstrap stress (adapted from markov_hmm_vs_code_project random_baselines.py
and stress.py) - spec 9.2 "run realistic null models".

Matched random portfolios: at every rebalance date of the model's long-only backtest, pick the same number of names
from the same eligible universe, keep the same share of the previous holdings as the model's turnover implies, and
charge the same per-name costs. Thousands of such paths give the distribution a strategy with no skill would achieve;
the model's percentile in it is the test ("did it beat 95% of monkeys with the same constraints?").

Block bootstrap: resamples a strategy's per-period net returns in contiguous blocks (keeping autocorrelation) to give
distributions of mean return, Sharpe and maximum drawdown instead of one historical path.
"""
import json
import math
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config as C
from research import name_costs

N_PATHS = 2000


def random_matched(panel, h, period_mask, k_share, turnover, cost, seed=0, rebalance_dates=None):
    """Mean net excess return per rebalance period for N_PATHS random portfolios matched to the model."""
    d = panel.loc[period_mask, ["date", "symbol", f"y_{h}"]].assign(c=cost[period_mask]).dropna(subset=[f"y_{h}"])
    from research import rebalance_calendar
    # Share the model's actual schedule; filtering labels must not shift dates.
    ud = rebalance_calendar(panel['date'], h) if rebalance_dates is None else pd.DatetimeIndex(rebalance_dates)
    ud = ud[ud.isin(pd.DatetimeIndex(pd.unique(panel.loc[period_mask, 'date'])))]
    syms = {s: i for i, s in enumerate(d["symbol"].unique())}
    S = len(syms)
    rng = np.random.default_rng(seed)
    prev = np.zeros((N_PATHS, S), bool)
    prev_cost = np.zeros(S)
    rets = []
    for dt in ud:
        g = d[d["date"] == dt]
        if len(g) < 10:
            continue
        idx = g["symbol"].map(syms).to_numpy()
        y, c = g[f"y_{h}"].to_numpy(), g["c"].to_numpy()
        k = max(1, int(round(len(g) * k_share)))
        elig = np.zeros(S, bool)
        elig[idx] = True
        sel = np.zeros((N_PATHS, S), bool)
        keep_n = int(round((1 - turnover) * k)) if prev.any() else 0
        noise = rng.random((N_PATHS, S))
        if keep_n:
            kept_score = np.where(prev & elig, noise, -1.0)  # keep a random subset of still-eligible holdings
            order = np.argsort(-kept_score, axis=1)[:, :keep_n]
            rows = np.repeat(np.arange(N_PATHS), keep_n)
            ok = kept_score[rows, order.ravel()] >= 0
            sel[rows[ok], order.ravel()[ok]] = True
        need = k - sel.sum(axis=1)
        fill_score = np.where(elig & ~sel, rng.random((N_PATHS, S)), -1.0)
        order = np.argsort(-fill_score, axis=1)
        for p in np.nonzero(need > 0)[0]:
            sel[p, order[p, : need[p]]] = True
        yv, cv = np.zeros(S), np.zeros(S)
        yv[idx], cv[idx] = y, c
        gross = (sel * yv).sum(axis=1) / k
        traded = sel ^ prev
        cost_now = np.where(elig, cv, prev_cost)
        net = gross - (traded * cost_now).sum(axis=1) / k
        rets.append(net)
        prev, prev_cost = sel, np.where(elig, cv, prev_cost)
    R = np.array(rets)  # periods x paths
    return R


def block_bootstrap(series, block, reps=2000, seed=1):
    """Block bootstrap of a per-period net return series: distributions of mean, Sharpe and max drawdown."""
    x = pd.Series(series).dropna().to_numpy()
    n = len(x)
    if n < 2 * block:
        return None
    rng = np.random.default_rng(seed)
    means, srs, dds = [], [], []
    for _ in range(reps):
        starts = rng.integers(0, n - block + 1, size=math.ceil(n / block))
        s = np.concatenate([x[a:a + block] for a in starts])[:n]
        means.append(s.mean())
        srs.append(s.mean() / s.std() if s.std() > 0 else 0)
        eq = np.cumsum(s)
        dds.append((eq - np.maximum.accumulate(eq)).min())
    q = lambda v: [float(np.quantile(v, 0.05)), float(np.quantile(v, 0.5)), float(np.quantile(v, 0.95))]
    return {"periods": n, "block": block, "reps": reps, "mean_net_q05_q50_q95": q(means), "sharpe_per_period_q05_q50_q95": q(srs),
            "max_drawdown_q05_q50_q95": q(dds), "prob_mean_below_zero": float(np.mean(np.array(means) < 0))}


def run(profile="large"):
    R = json.loads((C.REPORTS / f"research_results_{profile}.json").read_text())
    panel = pd.read_parquet(C.PROC / f"panel_{profile}.parquet")
    order_usd = R["costs"]["order_size_usd"]
    cost = name_costs(panel, order_usd).to_numpy()
    spy = pd.read_parquet(C.RAW / "prices" / f"{C.BENCHMARK}.parquet").drop_duplicates("date").set_index("date")
    hold_start = pd.Timestamp(R["sample"]["holdout_start"])
    out = {"profile": profile, "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "paths": N_PATHS, "results": {}}
    for h in (5, 20):
        periods = pd.read_parquet(C.REPORTS / f'ensemble_backtest_{profile}_h{h}.parquet')
        e = R["models"][str(h)]["models"]["ensemble"]
        first_wf = min(int(y) for y in e["wf_years"]) if e["wf_years"] else None
        for label, bt, mask in [("walk_forward", e["backtest_wf"], (panel["date"] < hold_start) & (panel["date"].dt.year >= (first_wf or 0))),
                                ("holdout", e["backtest_holdout"], panel["date"] >= hold_start)]:
            if not bt:
                continue
            schedule = pd.DatetimeIndex(periods.loc[periods['sample'] == label, 'd'])
            Rr = random_matched(panel, h, mask.to_numpy(), C.SIGNAL_TOP_QUANTILE, bt["mean_turnover"], cost, seed=h,
                                rebalance_dates=schedule)
            if len(Rr) != len(schedule):
                raise ValueError('matched-random periods do not match the saved model calendar')
            path_mean = Rr.mean(axis=0)
            path_sr = Rr.mean(axis=0) / Rr.std(axis=0)
            m = bt["mean_net_excess"]
            out["results"][f"h{h}_{label}"] = {
                "model_mean_net_per_period": m, "model_periods": bt["periods"], "random_periods": int(Rr.shape[0]),
                "random_mean_net_median": float(np.median(path_mean)), "random_mean_net_q05": float(np.quantile(path_mean, 0.05)),
                "random_mean_net_q95": float(np.quantile(path_mean, 0.95)),
                "model_percentile": float((path_mean < m).mean()), "share_random_beating_model": float((path_mean > m).mean()),
                "random_sharpe_median_per_period": float(np.median(path_sr)), "matched": {"names_share": C.SIGNAL_TOP_QUANTILE, "turnover": bt["mean_turnover"], "costs": "per-name model costs"}}
            print(profile, h, label, {k: round(v, 4) if isinstance(v, float) else v for k, v in out["results"][f"h{h}_{label}"].items() if k != "matched"}, flush=True)
    # block bootstrap of the saved ensemble backtest series (written by research.py from the next re-validation on)
    for h in (5, 20):
        f = C.REPORTS / f"ensemble_backtest_{profile}_h{h}.parquet"
        if f.exists():
            s = pd.read_parquet(f)
            out["results"][f"h{h}_bootstrap"] = {lab: block_bootstrap(g["net"], block=max(2, 60 // h)) for lab, g in s.groupby("sample")}
    (C.REPORTS / f"baselines_{profile}.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    _lock = __import__("config").hold_lock()
    run(sys.argv[1] if len(sys.argv) > 1 else "large")
