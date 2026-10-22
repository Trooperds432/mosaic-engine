"""Regime research beyond the HMM (spec section 8): Markov-switching regression, change-point detection and
regime-dependent factor regressions.

1. Markov-switching regression
   A 2-regime Markov-switching model of weekly SPY log returns (switching mean and variance, statsmodels
   MarkovRegression) is refitted each January on data before that year; FILTERED regime probabilities are then run
   forward with those frozen parameters (no smoothing, no look-ahead). A regime-switching ridge learns one weight
   vector per regime by weighting each training date by its regime probability; a forecast blends the two weight
   sets with the current filtered probability. It is walk-forward tested against the unconditional ridge on the
   same rows (difference in IC, HAC t), and evaluated once on the holdout.
2. Change-point detection
   PELT (ruptures, mean-shift cost, BIC-style penalty) on the daily factor returns and on the per-date IC series of
   the strongest features; reports each break with mean before/after and a Welch t-test.
3. Regime-dependent factor regressions
   Each risk-model factor return regressed on regime dummies (filtered HMM high-vol state; observable trend x VIX
   states): annualised mean per regime and a HAC t-test of the difference.
Research output only (reports/regime_research_<profile>.json); nothing here changes signals or the bot.
"""
import json
import math
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import ruptures as rpt
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import Ridge
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

import config as C
from features import FEATURES
from research import EMBARGO, MISSING_GROUPS, hac_mean, ic_series, preprocess, sector_neutral

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------- 1. Markov-switching regression
def ms_filtered(spy_ac):
    """Weekly SPY log returns -> filtered P(high-variance regime), refitted each year on prior data only."""
    w = np.log(spy_ac.resample("W-FRI").last()).diff().dropna() * 100
    out = pd.Series(np.nan, index=w.index)
    params_by_year = {}
    for yr in sorted(set(w.index.year))[5:]:
        train = w[w.index.year < yr]
        try:
            res = MarkovRegression(train.to_numpy(), k_regimes=2, trend="c", switching_variance=True).fit(disp=False)
        except Exception:
            continue
        sig = [res.params[f"sigma2[{k}]"] if f"sigma2[{k}]" in res.params.index else res.params.iloc[-2 + k] for k in range(2)] \
            if hasattr(res.params, "index") else res.params[-2:]
        hi = int(np.argmax(sig))
        upto = w[w.index.year <= yr]
        filt = MarkovRegression(upto.to_numpy(), k_regimes=2, trend="c", switching_variance=True).filter(res.params)
        probs = np.asarray(filt.filtered_marginal_probabilities)[:, hi]
        sel = upto.index.year == yr
        out.loc[upto.index[sel]] = probs[sel]
        params_by_year[int(yr)] = {"sigma2": [float(x) for x in sig], "high_vol_regime": hi}
    return out, params_by_year


def ms_ridge_walk_forward(Xm, y, dates, tpos, h, hpos, p_hi):
    """Regime-switching ridge vs unconditional ridge, yearly expanding refits, purged and embargoed."""
    years = pd.DatetimeIndex(dates).year
    ok = ~np.isnan(y)
    pre = tpos < hpos
    p = np.clip(np.nan_to_num(p_hi, nan=0.5), 0, 1)
    pred_ms, pred_base = np.full(len(y), np.nan), np.full(len(y), np.nan)
    X = Xm.to_numpy(dtype="float32")

    def fit_pair(train):
        lo, hi = np.nanquantile(y[train], [C.WINSOR_Q, 1 - C.WINSOR_Q])
        yc = np.clip(y[train], lo, hi)
        a = C.RIDGE_ALPHA * train.sum() / 1000
        base = Ridge(alpha=a).fit(X[train], yc)
        m_hi = Ridge(alpha=a).fit(X[train], yc, sample_weight=p[train] + 1e-3)
        m_lo = Ridge(alpha=a).fit(X[train], yc, sample_weight=1 - p[train] + 1e-3)
        return base, m_hi, m_lo
    for yr in range(years.min() + C.WF_MIN_TRAIN_YEARS, years.max() + 1):
        test = pre & (years == yr)
        if not test.any():
            continue
        train = ok & (tpos + 1 + h + EMBARGO < tpos[test].min())
        if train.sum() < 5000:
            continue
        base, m_hi, m_lo = fit_pair(train)
        pred_base[test] = base.predict(X[test])
        pred_ms[test] = p[test] * m_hi.predict(X[test]) + (1 - p[test]) * m_lo.predict(X[test])
    hold = tpos >= hpos
    base, m_hi, m_lo = fit_pair(ok & (tpos + 1 + h + EMBARGO < hpos))
    hb, hm = np.full(len(y), np.nan), np.full(len(y), np.nan)
    hb[hold] = base.predict(X[hold])
    hm[hold] = p[hold] * m_hi.predict(X[hold]) + (1 - p[hold]) * m_lo.predict(X[hold])
    coef_gap = float(np.corrcoef(m_hi.coef_, m_lo.coef_)[0, 1])
    return pred_ms, pred_base, hm, hb, coef_gap, dict(zip(Xm.columns, (m_hi.coef_ - m_lo.coef_).round(5)))


# ---------------------------------------------------------------- 1b. 2- vs 3-state HMM
def hmm_k(spy_ac, vix, k):
    """k-state Gaussian HMM on (SPY return, VIX log change), refit each January on prior data; returns filtered
    P(highest-variance state) per day and the out-of-sample log-likelihood of each year under the prior-years fit."""
    from hmmlearn.hmm import GaussianHMM
    from scipy import stats as st
    obs = pd.DataFrame({"r": spy_ac.pct_change(), "v": np.log(vix).diff()}).dropna()
    p_hi = pd.Series(np.nan, index=obs.index)
    oos_ll, bic = {}, {}
    for yr in sorted(set(obs.index.year))[3:]:
        tr = obs[obs.index.year < yr].to_numpy()
        try:
            m = GaussianHMM(n_components=k, covariance_type="full", n_iter=300, random_state=7).fit(tr)
        except Exception:
            continue
        hi = int(np.argmax([m.covars_[i][0, 0] for i in range(k)]))
        sel = obs.index[obs.index.year <= yr]
        X = obs.loc[sel].to_numpy()
        logB = np.column_stack([st.multivariate_normal(m.means_[i], m.covars_[i], allow_singular=True).logpdf(X) for i in range(k)])
        a = m.startprob_
        filt, ll = np.empty((len(X), k)), 0.0
        mask = sel.year == yr
        for t in range(len(X)):
            pred = a @ m.transmat_ if t else m.startprob_
            w = pred * np.exp(logB[t] - logB[t].max())
            if mask[t]:
                ll += float(np.log(w.sum()) + logB[t].max())  # one-step-ahead predictive log-likelihood
            a = w / w.sum()
            filt[t] = a
        p_hi.loc[sel[mask]] = filt[mask, hi]
        oos_ll[int(yr)] = ll
        n_par = k * (k - 1) + k * 2 + k * 3 + (k - 1)
        bic[int(yr)] = float(n_par * np.log(len(tr)) - 2 * m.score(tr))
    return p_hi, oos_ll, bic


# ---------------------------------------------------------------- 1c. observable Markov chain (markov-hedge-fund-method)
def observable_markov(spy_ac, holdout_start, window=20, threshold=0.05, min_train=252, cost_per_switch=C.ROUND_TRIP_COST / 2):
    """Bull/Bear/Sideways labels from the trailing `window`-day SPY return; 3x3 transition matrix counted from labels
    known before t only; signal = P(next=Bull|state) - P(next=Bear|state). Long/short (as in the source script) and
    long/cash (what a Trading 212 Invest account can do), net of a cost on every position change, vs holding SPY."""
    close = spy_ac.dropna()
    rr = close.pct_change(window)
    lab = pd.Series(1, index=close.index)
    lab[rr > threshold], lab[rr < -threshold] = 2, 0
    lab = lab[rr.notna()].to_numpy()
    idx = close.index[rr.notna()]
    ret = close.pct_change().reindex(idx).to_numpy()
    counts = np.zeros((3, 3))
    for i in range(min_train - 1):
        counts[lab[i], lab[i + 1]] += 1
    sig = np.full(len(lab), np.nan)
    for t in range(min_train, len(lab) - 1):
        P = counts / np.where(counts.sum(axis=1, keepdims=True) == 0, 1, counts.sum(axis=1, keepdims=True))
        sig[t] = P[lab[t], 2] - P[lab[t], 0]
        counts[lab[t - 1], lab[t]] += 1
    sig = pd.Series(sig, index=idx)
    nxt = pd.Series(ret, index=idx).shift(-1)
    fwd20 = (1 + pd.Series(ret, index=idx)).rolling(20).apply(np.prod, raw=True).shift(-20) - 1
    out = {"window": window, "threshold": threshold, "state_shares": {k: float(v) for k, v in pd.Series(lab).map({0: "bear", 1: "sideways", 2: "bull"}).value_counts(normalize=True).items()}}
    for name, pos in [("long_short", np.sign(sig)), ("long_cash", (sig > 0).astype(float).where(sig.notna()))]:
        sw = pos.diff().abs().fillna(0)
        strat = (pos * nxt - sw * cost_per_switch).dropna()
        for label, m in [("discovery", strat.index < holdout_start), ("holdout", strat.index >= holdout_start)]:
            x, b = strat[m], nxt.reindex(strat.index)[m]
            out[f"{name}_{label}"] = {"days": int(len(x)), "sharpe_ann": float(x.mean() / x.std() * math.sqrt(252)) if x.std() > 0 else None,
                                      "spy_sharpe_ann": float(b.mean() / b.std() * math.sqrt(252)), "ann_return": float(x.mean() * 252),
                                      "spy_ann_return": float(b.mean() * 252), "switches_per_year": float(sw[x.index].sum() / len(x) * 252),
                                      "share_in_market": float((pos.reindex(x.index) > 0).mean())}
    d = pd.DataFrame({"s": sig, "f": fwd20}).dropna()
    d = d[d.index < holdout_start]
    out["signal_vs_next_20d_spy_return_discovery"] = hac_mean((d["s"] - d["s"].mean()) * (d["f"] - d["f"].mean()) / (d["s"].std() * d["f"].std()), 20)
    return out


# ---------------------------------------------------------------- 2. change points
def change_points(series, name, min_size=26):
    s = pd.Series(series).dropna()
    if len(s) < 3 * min_size:
        return None
    x = s.to_numpy().reshape(-1, 1)
    pen = 3 * np.log(len(x)) * float(np.var(x))
    bks = rpt.Pelt(model="l2", min_size=min_size).fit(x).predict(pen=pen)[:-1]
    segs, edges = [], [0] + bks + [len(s)]
    for a, b in zip(edges[:-1], edges[1:]):
        seg = s.iloc[a:b]
        segs.append({"from": str(seg.index[0].date()), "to": str(seg.index[-1].date()), "n": int(len(seg)), "mean": float(seg.mean())})
    breaks = []
    for k, i in enumerate(bks):
        before, after = s.iloc[edges[k]:i], s.iloc[i:edges[k + 2]]
        t = stats.ttest_ind(before, after, equal_var=False)
        breaks.append({"date": str(s.index[i].date()), "mean_before": float(before.mean()), "mean_after": float(after.mean()),
                       "welch_t": float(t.statistic), "p": float(t.pvalue)})
    return {"series": name, "observations": int(len(s)), "breaks": breaks, "segments": segs}


# ---------------------------------------------------------------- 3. regime-dependent factor regressions
def factor_regimes(F, reg):
    out = {}
    d = F.join(reg, how="inner").dropna(subset=["p_high_vol"])
    d["stress"] = (d["p_high_vol"] >= 0.5).astype(float)
    for f in [c for c in F.columns if not c.startswith("sec_")]:
        y = d[f].to_numpy()
        res = sm.OLS(y, sm.add_constant(d["stress"].to_numpy())).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
        calm, stress = d.loc[d["stress"] == 0, f], d.loc[d["stress"] == 1, f]
        rec = {"calm_ann_mean": float(calm.mean() * 252), "stress_ann_mean": float(stress.mean() * 252),
               "calm_ann_vol": float(calm.std() * math.sqrt(252)), "stress_ann_vol": float(stress.std() * math.sqrt(252)),
               "difference_t_hac": float(res.tvalues[1]), "days_calm": int(len(calm)), "days_stress": int(len(stress))}
        by_obs = d.groupby("regime")[f].agg(lambda s: s.mean() * 252)
        rec["observable_regime_ann_mean"] = {k: float(v) for k, v in by_obs.items()}
        out[f] = rec
    return out


# ---------------------------------------------------------------- main
def run(profile="large", h=None):
    h = h or C.PRIMARY_HORIZON
    panel = pd.read_parquet(C.PROC / f"panel_{profile}.parquet")
    model = json.loads((C.PROC / f"model_{profile}.json").read_text())
    feats = model["feats"]
    spy = pd.read_parquet(C.RAW / "prices" / f"{C.BENCHMARK}.parquet").drop_duplicates("date").set_index("date")
    cal = spy.index
    panel["tpos"] = cal.get_indexer(panel["date"])
    X = preprocess(panel, feats)
    Xn = sector_neutral(X, panel, feats)
    Xm = pd.concat([Xn[feats], X[[k for k in MISSING_GROUPS if k in X]]], axis=1)
    dates, tpos = panel["date"].to_numpy(), panel["tpos"].to_numpy()
    hpos = int(cal.searchsorted(cal[-1] - pd.DateOffset(years=C.HOLDOUT_YEARS)))
    y = panel[f"y_{h}"].to_numpy().astype("float64")
    lags = math.ceil(h / C.DECISION_STEP)
    R = {"profile": profile, "horizon": h, "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    # 1. Markov-switching regime probabilities and regime-switching ridge
    p_ms, ms_params = ms_filtered(spy["adjclose"])
    p_rows = p_ms.reindex(pd.DatetimeIndex(dates), method="ffill").to_numpy()  # latest weekly filtered probability known at t
    pm, pb, hm, hb, coef_corr, coef_diff = ms_ridge_walk_forward(Xm, y, dates, tpos, h, hpos, p_rows)
    ic_ms, ic_base = ic_series(pm, y, dates), ic_series(pb, y, dates)
    d = (ic_ms - ic_base).dropna()
    dh = (ic_series(hm, y, dates) - ic_series(hb, y, dates)).dropna()
    top_diff = sorted(coef_diff.items(), key=lambda kv: -abs(kv[1]))[:10]
    R["markov_switching"] = {
        "method": "2-regime MarkovRegression on weekly SPY log returns (switching mean+variance), yearly refit, filtered probabilities; ridge weights per regime",
        "current_p_high_vol": float(p_ms.dropna().iloc[-1]), "as_of": str(p_ms.dropna().index[-1].date()),
        "regime_params_by_year": ms_params, "share_of_weeks_high_vol": float((p_ms.dropna() >= 0.5).mean()),
        "walk_forward": {"ic_regime_switching": float(ic_ms.mean()), "ic_unconditional": float(ic_base.mean()), "difference": hac_mean(d, lags)},
        "holdout": {"ic_regime_switching": float(ic_series(hm, y, dates).mean()), "ic_unconditional": float(ic_series(hb, y, dates).mean()), "difference": hac_mean(dh, lags)},
        "weight_vector_correlation_high_vs_low": coef_corr,
        "largest_weight_differences_high_minus_low": dict(top_diff),
        "improves": bool((hac_mean(d, lags)["t"] or 0) > 2 and d.mean() > 0)}
    print("markov-switching:", {k: R["markov_switching"][k] for k in ["current_p_high_vol", "walk_forward", "holdout", "improves", "weight_vector_correlation_high_vs_low"]}, flush=True)

    # 1b. 2- vs 3-state HMM: out-of-sample likelihood and ranking improvement from the 3-state stress probability
    vix = pd.read_parquet(C.RAW / "prices" / "^VIX.parquet").drop_duplicates("date").set_index("date")["close"].reindex(cal).ffill()
    hmm = {}
    for k in (2, 3):
        ph, ll, bic = hmm_k(spy["adjclose"], vix, k)
        hmm[k] = (ph, ll, bic)
    common = sorted(set(hmm[2][1]) & set(hmm[3][1]))
    d_ll = pd.Series({y: hmm[3][1][y] - hmm[2][1][y] for y in common})
    p3 = hmm[3][0].reindex(pd.DatetimeIndex(dates)).to_numpy()
    pm3, pb3, hm3, hb3, corr3, _ = ms_ridge_walk_forward(Xm, y, dates, tpos, h, hpos, p3)
    d3 = (ic_series(pm3, y, dates) - ic_series(pb3, y, dates)).dropna()
    R["hmm_2_vs_3_states"] = {"oos_loglik_3_minus_2_by_year": d_ll.round(2).to_dict(), "years_3_state_better": int((d_ll > 0).sum()), "years": int(len(d_ll)),
                              "mean_oos_loglik_gain": float(d_ll.mean()), "t": float(d_ll.mean() / d_ll.std() * math.sqrt(len(d_ll))) if len(d_ll) > 2 else None,
                              "mean_bic_2": float(np.mean(list(hmm[2][2].values()))), "mean_bic_3": float(np.mean(list(hmm[3][2].values()))),
                              "current_p_stress_3_state": float(hmm[3][0].dropna().iloc[-1]),
                              "ranking_test_3_state_switching_ridge": {"wf_ic_difference": hac_mean(d3, lags), "improves": bool((hac_mean(d3, lags)["t"] or 0) > 2 and d3.mean() > 0)}}
    print("hmm 2 vs 3:", {k: R["hmm_2_vs_3_states"][k] for k in ["years_3_state_better", "years", "mean_oos_loglik_gain", "t", "current_p_stress_3_state"]},
          R["hmm_2_vs_3_states"]["ranking_test_3_state_switching_ridge"], flush=True)

    # 1c. observable Markov chain on SPY
    R["observable_markov"] = observable_markov(spy["adjclose"], cal[hpos])  # source default: +-5% over 20 days
    R["observable_markov"]["threshold_sensitivity_2pct"] = {k: v for k, v in observable_markov(spy["adjclose"], cal[hpos], threshold=0.02).items() if k.startswith(("long_", "signal"))}
    om = R["observable_markov"]
    print("observable markov:", {k: (round(v["sharpe_ann"], 2) if v.get("sharpe_ann") is not None else None, round(v["spy_sharpe_ann"], 2)) for k, v in om.items() if isinstance(v, dict) and "sharpe_ann" in v},
          "signal corr t", round(om["signal_vs_next_20d_spy_return_discovery"]["t"] or 0, 2), flush=True)

    # 2. change points: factor returns (daily) and the per-date IC of the strongest features
    F = pd.read_parquet(C.PROC / f"factor_returns_{profile}.parquet")
    cps = []
    for f in [c for c in F.columns if not c.startswith("sec_")]:
        r = change_points(F[f].rolling(63).mean().iloc[::21], f"factor:{f} (63d mean, monthly samples)", min_size=6)
        if r:
            cps.append(r)
    res = json.loads((C.REPORTS / f"research_results_{profile}.json").read_text())
    strong = sorted([u for u in res["univariate"] if u["horizon"] == h and u["variant"] == "raw" and u.get("t") is not None],
                    key=lambda u: -abs(u["t"]))[:12]
    for u in strong:
        f = u["feature"]
        ic = ic_series(X[f].where(panel[f].notna()).to_numpy(), y, dates)
        r = change_points(ic, f"ic:{f}", min_size=26)
        if r:
            cps.append(r)
    R["change_points"] = cps
    print("change points:", sum(len(c["breaks"]) for c in cps), "breaks across", len(cps), "series", flush=True)

    # 3. regime-dependent factor regressions (filtered HMM + observable regimes from research.py)
    reg = pd.read_parquet(C.PROC / "regime_history.parquet")
    R["factor_regimes"] = factor_regimes(F, reg)
    sig = {f: v for f, v in R["factor_regimes"].items() if abs(v["difference_t_hac"]) >= 2}
    print("factor premia that differ by regime (|t|>=2):", {f: round(v["difference_t_hac"], 2) for f, v in sig.items()}, flush=True)

    (C.REPORTS / f"regime_research_{profile}.json").write_text(json.dumps(R, indent=1, default=float))
    with (C.LEDGER / "research_runs.jsonl").open("a") as fh:
        fh.write(json.dumps({"run_id": f"regimes-{profile}-{R['generated_at_utc']}", "note": "Markov-switching ridge (new model, holdout evaluated once), change points, regime factor regressions",
                             "ms_wf_ic_diff_t": R["markov_switching"]["walk_forward"]["difference"]["t"], "ms_improves": R["markov_switching"]["improves"]}) + "\n")
    return R


if __name__ == "__main__":
    _lock = __import__("config").hold_lock()
    run(sys.argv[1] if len(sys.argv) > 1 else "large")
