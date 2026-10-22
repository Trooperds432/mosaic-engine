"""Kalman-filter dynamic betas (adapted from the Kalman pairs-trading notebook) and their walk-forward test.

State-space model per stock i:  r_it = a_it + b_it' x_t + e_it,   [a, b]_t = [a, b]_{t-1} + w_t
  x_t: SPY daily return (market beta) or [SPY, d10y, dBAA, dollar] (multi-factor betas)
  w_t ~ N(0, diag(q_a, q_b)): random-walk drift of the coefficients; e_it variance tracked by EWMA of innovations.
All stocks are filtered together (vectorised), using returns through t only (filtered, never smoothed).

Test (spec: "tested in walk-forward against the current beta"): every 21 sessions, forecast each stock's beta and
compare with the beta it actually showed over the next 63 sessions (OLS). Competitors: rolling 252-day OLS (current
engine), Blume-adjusted (0.67 b + 0.33) and Kalman. The drift q_b is chosen on 2009-2014 only; 2015+ is out of sample.
Accuracy: mean squared error; significance: Diebold-Mariano style HAC t-test on the per-date MSE difference.
Writes reports/kalman_beta.json and processed/kalman_betas_latest.parquet (nightly betas for signals/risk).
"""
import json
import math
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config as C
from features import _macro_returns, load_prices
from research import hac_mean

CALIB_END = "2014-12-31"
GRID = [1e-6, 1e-5, 1e-4, 1e-3]


def kalman_betas(R, X, q_b, q_a=1e-10, lam=0.97):
    """R: T x n returns, X: T x k regressors (no intercept). Returns T x n x (k+1) filtered [alpha, betas]."""
    T, n = R.shape
    k = X.shape[1] + 1
    theta = np.zeros((n, k))
    theta[:, 1] = 1.0  # prior beta 1
    P = np.tile(np.eye(k) * 1.0, (n, 1, 1))
    Q = np.diag([q_a] + [q_b] * (k - 1))
    Rv = np.full(n, np.nanvar(R[:60], axis=0))
    Rv = np.where(np.isfinite(Rv) & (Rv > 0), Rv, 4e-4)
    out = np.full((T, n, k), np.nan, dtype="float32")
    started = np.zeros(n, bool)
    for t in range(T):
        x = np.concatenate([[1.0], X[t]])
        if not np.isfinite(x).all():
            out[t] = np.where(started[:, None], theta, np.nan)
            continue
        P = P + Q
        y = R[t]
        ok = np.isfinite(y)
        Px = P @ x                                   # n x k
        f = Px @ x + Rv                              # n
        e = y - theta @ x
        K = Px / f[:, None]
        upd = ok
        theta[upd] += K[upd] * e[upd, None]
        P[upd] -= np.einsum("ni,nj->nij", K[upd], Px[upd])
        Rv[upd] = lam * Rv[upd] + (1 - lam) * e[upd] ** 2
        started |= ok
        out[t] = np.where(started[:, None], theta, np.nan)
    return out


def realised_beta(r, m, start, h=63):
    rr, mm = r[start:start + h], m[start:start + h]
    ok = np.isfinite(rr) & np.isfinite(mm)
    if ok.sum() < 40:
        return np.nan
    mv = mm[ok] - mm[ok].mean()
    return float((mv * (rr[ok] - rr[ok].mean())).sum() / (mv ** 2).sum())


def evaluate(R, m, betas_by_method, dates, cal):
    """Per evaluation date: MSE of each method's beta forecast vs the next-63-session realised beta."""
    rows = []
    for t in dates:
        rb = np.array([realised_beta(R[:, j], m, t + 1) for j in range(R.shape[1])])
        ok = np.isfinite(rb)
        for B in betas_by_method.values():  # score every method on the same stocks (all must have a forecast)
            ok &= np.isfinite(B[t])
        rec = {"date": cal[t], "n_stocks": int(ok.sum())}
        for name, B in betas_by_method.items():
            rec[name] = float(np.mean((B[t][ok] - rb[ok]) ** 2)) if ok.sum() > 30 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows).set_index("date")


def run(profile="large"):
    P, sym2cik = load_prices(C.PROFILES[profile]["min_dollar_adv"])
    cal = P["adjclose"].index
    R = P["adjclose"].pct_change(fill_method=None).to_numpy(dtype="float64")
    R = np.where(np.abs(R) > 0.5, np.nan, R)  # bad prints
    m = P["spy_adjclose"].pct_change().to_numpy(dtype="float64")
    syms = list(P["adjclose"].columns)
    T = len(cal)
    eval_idx = np.arange(300, T - 64, 21)
    cal_end = cal.searchsorted(pd.Timestamp(CALIB_END))
    # current engine beta: rolling 252-day OLS
    Rdf = pd.DataFrame(R, index=cal)
    ms = pd.Series(m, index=cal)
    roll = (Rdf.rolling(252, min_periods=200).cov(ms).div(ms.rolling(252, min_periods=200).var(), axis=0)).to_numpy()
    blume = 0.67 * roll + 0.33
    # choose the drift on 2009-2014 only
    calib_dates = eval_idx[eval_idx < cal_end - 64]
    sample = np.random.default_rng(3).choice(R.shape[1], size=min(400, R.shape[1]), replace=False)
    calib = {}
    for q in GRID:
        kb = kalman_betas(R[:, sample], m[:, None], q)[:, :, 1]
        calib[q] = float(evaluate(R[:, sample], m, {"k": kb, "r": roll[:, sample]}, calib_dates, cal)["k"].mean())
        print("calibration q_b", q, calib[q], flush=True)
    q_best = min(calib, key=calib.get)
    kb_all = kalman_betas(R, m[:, None], q_best)
    kalman = kb_all[:, :, 1]
    test_dates = eval_idx[eval_idx >= cal_end]
    ev = evaluate(R, m, {"rolling_252": roll, "blume": blume, "kalman": kalman}, test_dates, cal)
    res = {"profile": profile, "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "calibration_window_end": CALIB_END, "q_b_grid_mse": {str(k): v for k, v in calib.items()}, "q_b_chosen": q_best,
           "test_period": [str(cal[test_dates[0]].date()), str(cal[test_dates[-1]].date())], "evaluation_dates": int(len(ev)),
           "mse": {c: float(ev[c].mean()) for c in ["rolling_252", "blume", "kalman"]}, "mean_stocks_per_date": float(ev["n_stocks"].mean())}
    for c in ["blume", "kalman"]:
        d = (ev["rolling_252"] - ev[c]).dropna()
        res[f"{c}_vs_rolling"] = {"mse_reduction_pct": float(d.mean() / ev["rolling_252"].mean() * 100), **hac_mean(d, 3),
                                  "share_of_dates_better": float((d > 0).mean())}
    d = (ev["blume"] - ev["kalman"]).dropna()
    res["kalman_vs_blume"] = {"mse_reduction_pct": float(d.mean() / ev["blume"].mean() * 100), **hac_mean(d, 3)}
    k_ok = res["kalman_vs_rolling"]["t"] is not None and res["kalman_vs_rolling"]["t"] > 2
    b_ok = res["kalman_vs_blume"]["t"] is not None and res["kalman_vs_blume"]["t"] > 0
    res["winner"] = "kalman" if k_ok and b_ok else ("blume" if (res["blume_vs_rolling"]["t"] or 0) > 2 else "rolling_252")
    print(json.dumps({k: res[k] for k in ["mse", "kalman_vs_rolling", "blume_vs_rolling", "kalman_vs_blume", "winner"]}, indent=1, default=float), flush=True)

    # multi-factor Kalman betas (market, rates, credit, dollar) for the features / risk model
    mr = _macro_returns(cal)
    Xm = np.column_stack([m, mr["dy10"].to_numpy(), mr["dbaa"].to_numpy(), mr["dollar"].to_numpy()])
    mf = kalman_betas(R, Xm, q_best)
    last = pd.DataFrame({"symbol": syms, "beta_kalman": kalman[-1], "beta_rolling_252": roll[-1], "beta_blume": blume[-1],
                         "beta_mkt_mf": mf[-1, :, 1], "beta_rates_kf": mf[-1, :, 2], "beta_credit_kf": mf[-1, :, 3], "beta_dollar_kf": mf[-1, :, 4],
                         "as_of": str(cal[-1].date()), "q_b": q_best})
    last.to_parquet(C.PROC / f"kalman_betas_latest_{profile}.parquet", index=False)
    np.save(C.PROC / f"kalman_beta_history_{profile}.npy", kalman.astype("float32"))
    (C.PROC / f"kalman_beta_symbols_{profile}.json").write_text(json.dumps({"symbols": syms, "dates": [str(d.date()) for d in cal]}))
    res["latest_examples"] = last.set_index("symbol").loc[[s for s in ["AAPL", "NVDA", "KO", "JPM", "XOM"] if s in syms],
                                                          ["beta_kalman", "beta_rolling_252", "beta_blume"]].round(3).to_dict(orient="index")
    (C.REPORTS / f"kalman_beta_{profile}.json").write_text(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    _lock = __import__("config").hold_lock()
    run(sys.argv[1] if len(sys.argv) > 1 else "large")
