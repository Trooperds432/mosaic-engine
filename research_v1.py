"""Statistical validation of the feature library and the combined cross-sectional model.

Protocol: discovery sample (all pre-holdout dates) -> univariate IC / quantile tests with HAC
inference and BH-FDR -> purged, embargoed expanding-window walk-forward ridge -> single evaluation
on the untouched final holdout -> calibration from out-of-sample predictions only -> cost-adjusted
long-only backtest -> PSR / DSR / PBO -> validation gate.
"""
import hashlib
import itertools
import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import Ridge

import config as C
from features import FEATURES, FUND_FEATURES

FEATS = list(FEATURES)
EMBARGO = 5


def config_hash():
    keys = ["HORIZONS", "PRIMARY_HORIZON", "DECISION_STEP", "HOLDOUT_YEARS", "WF_MIN_TRAIN_YEARS", "RIDGE_ALPHA",
            "WINSOR_Q", "N_QUANTILES", "ROUND_TRIP_COST", "MIN_MCAP_USD", "MIN_ADV_SHARES", "FEATURE_SET_VERSION"]
    blob = json.dumps({k: getattr(C, k) for k in keys} | {"features": FEATS}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ------------------------------------------------------------- preprocessing
def preprocess(panel):
    """Per-date cross-sectional winsorise + percentile rank to [-0.5, 0.5]. Uses only the same
    date's cross-section, so no information crosses time. Missing -> 0 (cross-sectional median)
    with a fundamentals-missing indicator."""
    X = pd.DataFrame(index=panel.index)
    g = panel.groupby("date")
    for f in FEATS:
        x = panel[f].astype("float64").replace([np.inf, -np.inf], np.nan)
        lo = g[f].transform(lambda s: s.quantile(C.WINSOR_Q))
        hi = g[f].transform(lambda s: s.quantile(1 - C.WINSOR_Q))
        x = x.clip(lo, hi)
        r = x.groupby(panel["date"]).rank(pct=True) - 0.5
        X[f] = r.fillna(0.0).astype("float32")
    X["fund_missing"] = panel[FUND_FEATURES].isna().any(axis=1).astype("float32")
    return X


def sector_neutral(X, panel):
    Xn = X.copy()
    for f in FEATS:
        m = X[f].groupby([panel["date"], panel["sector"]]).transform("mean")
        Xn[f] = (X[f] - m).astype("float32")
    return Xn


# ------------------------------------------------------------- inference helpers
def hac_mean(series, lags):
    s = pd.Series(series).dropna()
    if len(s) < 10:
        return {"mean": None, "t": None, "p": None, "n": int(len(s))}
    res = sm.OLS(s.to_numpy(), np.ones(len(s))).fit(cov_type="HAC", cov_kwds={"maxlags": int(lags)})
    return {"mean": float(res.params[0]), "se": float(res.bse[0]), "t": float(res.tvalues[0]),
            "p": float(res.pvalues[0]), "n": int(len(s))}


def ic_series(score, y, dates):
    df = pd.DataFrame({"s": score, "y": y, "d": dates}).dropna()
    df = df[df.groupby("d")["y"].transform("count") >= 20]
    rs = df.groupby("d")["s"].rank()
    ry = df.groupby("d")["y"].rank()
    tmp = pd.DataFrame({"rs": rs, "ry": ry, "d": df["d"]})
    return tmp.groupby("d").apply(lambda g: g["rs"].corr(g["ry"]), include_groups=False).dropna()


def ic_summary(ic, h):
    lags = max(1, math.ceil(h / C.DECISION_STEP))
    out = hac_mean(ic, lags)
    out.update({"ic_std": float(ic.std()) if len(ic) > 1 else None,
                "ic_ir": float(ic.mean() / ic.std()) if len(ic) > 1 and ic.std() > 0 else None,
                "pos_frac": float((ic > 0).mean()) if len(ic) else None,
                "n_dates": int(len(ic)), "effective_n": int(len(ic) * C.DECISION_STEP / max(h, C.DECISION_STEP)),
                "nw_lags": lags})
    if out.get("mean") is not None:
        out["ci95"] = [out["mean"] - 1.96 * out["se"], out["mean"] + 1.96 * out["se"]]
    return out


def quantile_table(score, y, dates, q=C.N_QUANTILES):
    df = pd.DataFrame({"s": score, "y": y, "d": dates}).dropna()
    df = df[df.groupby("d")["y"].transform("count") >= 20]  # same minimum cross-section as ic_series
    df["q"] = df.groupby("d")["s"].transform(lambda s: pd.qcut(s.rank(method="first"), q, labels=False))
    per = df.groupby(["d", "q"])["y"].mean().unstack()
    means = per.mean()
    return per, means


def bh_fdr(pvals, q):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    thresh = q * (np.arange(1, n + 1)) / n
    passed = p[order] <= thresh
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    rej = np.zeros(n, bool)
    rej[order[:k]] = True
    adj = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    padj = np.empty(n)
    padj[order] = np.minimum(adj, 1)
    return rej, padj


def sharpe_stats(r):
    r = pd.Series(r).dropna()
    if len(r) < 8 or r.std() == 0:
        return None
    return {"sr": float(r.mean() / r.std()), "skew": float(stats.skew(r)), "kurt": float(stats.kurtosis(r, fisher=False)),
            "n": int(len(r))}


def psr(sr, sr0, n, skew, kurt):
    den = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr))
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(n - 1) / den))


def deflated_sr(st, trial_srs):
    n_trials = len(trial_srs)
    if st is None or n_trials < 2:
        return None, None
    v = float(np.var(trial_srs, ddof=1))
    g = 0.5772156649
    sr0 = math.sqrt(v) * ((1 - g) * stats.norm.ppf(1 - 1 / n_trials) + g * stats.norm.ppf(1 - 1 / (n_trials * math.e)))
    return psr(st["sr"], sr0, st["n"], st["skew"], st["kurt"]), sr0


def pbo_cscv(M, S=10):
    """Probability of backtest overfitting (Bailey et al.) via combinatorially symmetric CV.
    M: periods x strategies matrix of returns."""
    M = M.dropna(axis=1, how="all").dropna()
    T, N = M.shape
    if N < 3 or T < S * 4:
        return None
    blocks = np.array_split(np.arange(T), S)
    lam = []
    for comb in itertools.combinations(range(S), S // 2):
        ins = np.concatenate([blocks[i] for i in comb])
        oos = np.concatenate([blocks[i] for i in range(S) if i not in comb])
        a, b = M.iloc[ins], M.iloc[oos]
        sr_is = a.mean() / a.std()
        sr_oos = b.mean() / b.std()
        best = sr_is.idxmax()
        w = sr_oos.rank().loc[best] / (N + 1)
        lam.append(math.log(w / (1 - w)))
    return float(np.mean(np.array(lam) <= 0))


# ------------------------------------------------------------- portfolio backtest
def long_only_backtest(score, y, dates, h, top=C.SIGNAL_TOP_QUANTILE, offset=0):
    """Equal-weight top-quantile long-only, rebalanced every h sessions (non-overlapping),
    excess vs SPY, net of turnover x one-way cost."""
    df = pd.DataFrame({"s": score, "y": y, "d": dates, "sym": SYMS.reindex(score.index).to_numpy()}).dropna(subset=["s", "y"])
    ud = np.sort(df["d"].unique())
    step = max(1, h // C.DECISION_STEP)
    keep = set(ud[offset::step])
    df = df[df["d"].isin(keep)]
    rows, prev = [], set()
    one_way = C.ROUND_TRIP_COST / 2
    for d, g in df.groupby("d"):
        k = max(1, int(round(len(g) * top)))
        pick = g.nlargest(k, "s")
        names = set(pick["sym"])
        turnover = 1.0 if not prev else len(names.symmetric_difference(prev)) / (2 * len(names))
        gross = pick["y"].mean()
        rows.append({"d": d, "gross": gross, "net": gross - 2 * turnover * one_way, "turnover": turnover, "n": k})
        prev = names
    return pd.DataFrame(rows).set_index("d")


SYMS = pd.Series(dtype=object)


# ------------------------------------------------------------- walk-forward model
def fit_ridge(X, y):
    lo, hi = np.nanquantile(y, [C.WINSOR_Q, 1 - C.WINSOR_Q])  # thresholds from training labels only
    m = Ridge(alpha=C.RIDGE_ALPHA * len(y) / 1000).fit(X, np.clip(y, lo, hi))
    return m


def walk_forward(X, y, dates, tpos, h, holdout_start_pos):
    """Expanding-window, refit yearly. Training rows are purged so that every training label ends
    (entry t+1, exit t+1+h) strictly before the test block begins, plus an embargo."""
    ok = ~np.isnan(y)
    years = pd.DatetimeIndex(dates).year
    first_year = years.min() + C.WF_MIN_TRAIN_YEARS
    preds = pd.Series(np.nan, index=X.index)
    coefs = {}
    pre = tpos < holdout_start_pos
    for yr in range(first_year, years.max() + 1):
        test = pre & (years == yr)
        if not test.any():
            continue
        start = tpos[test].min()
        train = ok & (tpos + 1 + h + EMBARGO < start)
        if train.sum() < 5000:
            continue
        m = fit_ridge(X[train], y[train])
        preds[test] = m.predict(X[test])
        coefs[yr] = dict(zip(X.columns, m.coef_.round(6)))
    # untouched holdout: one fit on all pre-holdout labels that end before the holdout
    hold = tpos >= holdout_start_pos
    train = ok & (tpos + 1 + h + EMBARGO < holdout_start_pos)
    m = fit_ridge(X[train], y[train])
    hpred = pd.Series(np.nan, index=X.index)
    hpred[hold] = m.predict(X[hold])
    return preds, hpred, coefs, dict(zip(X.columns, m.coef_.round(6)))


def calibration(pred, y, dates, buckets=C.CAL_BUCKETS):
    df = pd.DataFrame({"p": pred, "y": y, "d": dates}).dropna()
    df = df[df.groupby("d")["y"].transform("count") >= 20]
    df["b"] = df.groupby("d")["p"].transform(lambda s: pd.qcut(s.rank(method="first"), buckets, labels=False))
    tab = df.groupby("b")["y"].agg(mean="mean", std="std", n="count",
                                   q025=lambda s: s.quantile(0.025), q05=lambda s: s.quantile(0.05),
                                   q975=lambda s: s.quantile(0.975), p_pos=lambda s: (s > 0).mean(),
                                   es05=lambda s: s[s <= s.quantile(0.05)].mean())
    return tab, df


# ------------------------------------------------------------- main
def run():
    global SYMS
    panel = pd.read_parquet(C.PROC / "panel.parquet")
    spy = pd.read_parquet(C.RAW / "prices" / f"{C.BENCHMARK}.parquet").set_index("date")
    cal = spy.index
    panel["tpos"] = cal.get_indexer(panel["date"])
    SYMS = panel["symbol"]
    X = preprocess(panel)
    Xn = sector_neutral(X, panel)
    dates = panel["date"].to_numpy()
    tpos = panel["tpos"].to_numpy()
    holdout_start = cal[-1] - pd.DateOffset(years=C.HOLDOUT_YEARS)
    hpos = int(cal.searchsorted(holdout_start))
    pre = tpos < hpos
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + config_hash()
    R = {"run_id": run_id, "config_hash": config_hash(), "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "sample": {"first_date": str(pd.Timestamp(dates.min()).date()), "last_date": str(pd.Timestamp(dates.max()).date()),
                    "holdout_start": str(cal[hpos].date()), "rows": int(len(panel)), "symbols": int(panel["symbol"].nunique()),
                    "decision_dates": int(panel["date"].nunique()),
                    "avg_names_per_date": float(panel.groupby("date").size().mean())},
         "costs": {"round_trip": C.ROUND_TRIP_COST, "fx_fee_per_side": C.FX_FEE, "half_spread": C.HALF_SPREAD, "slippage": C.SLIPPAGE}}

    # ---- regimes (observable, filtered: only data up to t)
    vix = pd.read_parquet(C.RAW / "prices" / "^VIX.parquet").set_index("date")["close"].reindex(cal).ffill()
    spy_ac = spy["adjclose"]
    trend = spy_ac > spy_ac.rolling(200).mean()
    vix_hi = vix > vix.rolling(1260, min_periods=252).median()
    regime = (np.where(trend, "uptrend", "downtrend").astype(object) + "/" + np.where(vix_hi, "high_vol", "low_vol").astype(object))
    regime = pd.Series(regime, index=cal)
    panel["regime"] = regime.reindex(panel["date"]).to_numpy()
    R["current_regime"] = {"state": regime.iloc[-1], "spy_above_200d": bool(trend.iloc[-1]), "vix": float(vix.iloc[-1]),
                           "vix_5y_median": float(vix.rolling(1260, min_periods=252).median().iloc[-1]),
                           "as_of": str(cal[-1].date())}

    # ---- univariate discovery tests (pre-holdout)
    uni, pvals, keys = [], [], []
    for f in FEATS:
        for h in C.HORIZONS:
            y = panel[f"y_{h}"].to_numpy()
            for variant, XX in [("raw", X), ("sector_neutral", Xn)]:
                ic = ic_series(XX[f].where(pre), np.where(pre, y, np.nan), dates)
                s = ic_summary(ic, h)
                rec = {"feature": f, "family": FEATURES[f][0], "hyp_sign": FEATURES[f][1], "horizon": h, "variant": variant, **s}
                if variant == "raw":
                    _, qm = quantile_table(XX[f].where(pre), np.where(pre, y, np.nan), dates)
                    rec["quantile_means"] = [float(v) for v in qm.to_numpy()]
                    rec["top_minus_bottom"] = float(qm.iloc[-1] - qm.iloc[0])
                    rec["monotonicity"] = float(stats.spearmanr(np.arange(len(qm)), qm.to_numpy())[0])
                uni.append(rec)
                pvals.append(s["p"] if s["p"] is not None else 1.0)
    rej, padj = bh_fdr(pvals, C.FDR_Q)
    for rec, r_, pa in zip(uni, rej, padj):
        rec["fdr_reject"] = bool(r_)
        rec["p_bh"] = float(pa)
        rec["sign_matches_hypothesis"] = (rec["mean"] is not None and np.sign(rec["mean"]) == rec["hyp_sign"])
    R["univariate"] = uni
    n_hyp = len(uni)

    # ---- models per horizon
    Xm = pd.concat([Xn[FEATS], X[["fund_missing"]]], axis=1)
    R["models"] = {}
    oos_pred = {}
    final_models = {}
    for h in C.HORIZONS:
        y = panel[f"y_{h}"].to_numpy().astype("float64")
        wf, hold, coefs, hold_coef = walk_forward(Xm, y, dates, tpos, h, hpos)
        ic_wf = ic_series(wf, y, dates)
        ic_ho = ic_series(hold, y, dates)
        yearly = ic_wf.groupby(pd.DatetimeIndex(ic_wf.index).year).mean()
        # baselines evaluated on the same OOS dates
        base_mom = ic_series(X["mom_12_1"].where(~np.isnan(wf.to_numpy()) | (tpos >= hpos)), y, dates)
        eq = sum(FEATURES[f][1] * Xn[f] for f in FEATS) / len(FEATS)
        base_eq_ho = ic_series(eq.where(tpos >= hpos), y, dates)
        coef_df = pd.DataFrame(coefs).T
        stability = {f: float((np.sign(coef_df[f]) == np.sign(coef_df[f].median())).mean()) for f in coef_df.columns} if len(coef_df) else {}
        mres = {"horizon": h, "walk_forward": ic_summary(ic_wf, h), "holdout": ic_summary(ic_ho, h),
                "wf_years": {int(k): float(v) for k, v in yearly.items()},
                "wf_positive_year_frac": float((yearly > 0).mean()) if len(yearly) else None,
                "baseline_mom_12_1_oos_ic": float(base_mom.mean()) if len(base_mom) else None,
                "baseline_equal_weight_holdout_ic": float(base_eq_ho.mean()) if len(base_eq_ho) else None,
                "holdout_coefficients": hold_coef, "coef_sign_stability": stability,
                "wf_first_test_year": int(min(coefs)) if coefs else None}
        # regime-conditioned OOS IC
        reg = pd.Series(regime.reindex(ic_wf.index).to_numpy(), index=ic_wf.index)
        mres["regime_ic"] = {k: {"mean": float(v.mean()), "n": int(len(v)), "t": hac_mean(v, max(1, math.ceil(h / C.DECISION_STEP)))["t"]}
                             for k, v in ic_wf.groupby(reg) if len(v) >= 10}
        combined = wf.fillna(hold)
        oos_pred[h] = combined
        tab, _ = calibration(combined, y, dates)
        tab_wf, _ = calibration(wf, y, dates)
        tab_ho, _ = calibration(hold, y, dates)
        mono_wf = stats.spearmanr(tab_wf.index, tab_wf["mean"])[0] if len(tab_wf) > 2 else np.nan
        mono_ho = stats.spearmanr(tab_ho.index, tab_ho["mean"])[0] if len(tab_ho) > 2 else np.nan
        mres["calibration"] = {"table": tab.reset_index().to_dict(orient="records"),
                               "wf_monotonicity": float(mono_wf), "holdout_monotonicity": float(mono_ho),
                               "status": "HISTORICAL_DECILE_CALIBRATED" if (mono_wf >= 0.7 and mono_ho >= 0.5) else "NOT_CALIBRATED"}
        # cost-adjusted long-only backtests (WF period and holdout separately)
        bt_wf = long_only_backtest(wf, pd.Series(y, index=X.index), dates, h)
        bt_ho = long_only_backtest(hold, pd.Series(y, index=X.index), dates, h)
        ppy = 252 / max(h, 1)

        def bt_summary(bt):
            if len(bt) < 4:
                return None
            st = sharpe_stats(bt["net"])
            return {"periods": int(len(bt)), "mean_gross_excess": float(bt["gross"].mean()), "mean_net_excess": float(bt["net"].mean()),
                    "mean_turnover": float(bt["turnover"].mean()), "avg_names": float(bt["n"].mean()),
                    "net_sharpe_ann": float(st["sr"] * math.sqrt(ppy)) if st else None,
                    "net_t_hac": hac_mean(bt["net"], 2)["t"], "hit_rate": float((bt["net"] > 0).mean()),
                    "max_drawdown_excess": float((bt["net"].cumsum() - bt["net"].cumsum().cummax()).min())}
        mres["backtest_wf"] = bt_summary(bt_wf)
        mres["backtest_holdout"] = bt_summary(bt_ho)
        mres["_bt_oos"] = pd.concat([bt_wf, bt_ho])["net"]
        R["models"][h] = mres
        # production model: fit on every labelled row (incl. holdout) after validation
        okp = ~np.isnan(y)
        final_models[h] = fit_ridge(Xm[okp], y[okp])

    # ---- overfitting diagnostics at the primary horizon
    h = C.PRIMARY_HORIZON
    yv = pd.Series(panel[f"y_{h}"].to_numpy(), index=X.index)
    trial_rets = {}
    for f in FEATS:
        bt = long_only_backtest((FEATURES[f][1] * Xn[f]).where(pre), yv.where(pre), dates, h)
        if len(bt):
            trial_rets[f] = bt["net"]
    M = pd.DataFrame(trial_rets)
    trial_srs = [s["sr"] for s in (sharpe_stats(M[c]) for c in M) if s]
    bt_oos = R["models"][h]["_bt_oos"]
    st = sharpe_stats(bt_oos)
    n_trials = len(trial_srs) + len(C.HORIZONS)
    dsr, sr0 = deflated_sr(st, trial_srs + [st["sr"]] if st else trial_srs)
    R["overfitting"] = {"primary_horizon": h, "oos_periods": int(len(bt_oos.dropna())),
                        "oos_sr_per_period": st["sr"] if st else None,
                        "psr_vs_zero": psr(st["sr"], 0, st["n"], st["skew"], st["kurt"]) if st else None,
                        "deflated_sharpe_ratio": dsr, "dsr_benchmark_sr": sr0, "n_trials_for_dsr": n_trials,
                        "pbo_cscv_univariate_strategies": pbo_cscv(M),
                        "train_vs_test_ic_degradation": None}
    wf_ic = R["models"][h]["walk_forward"]["mean"]
    ho_ic = R["models"][h]["holdout"]["mean"]
    if wf_ic is not None and ho_ic is not None:
        R["overfitting"]["train_vs_test_ic_degradation"] = wf_ic - ho_ic

    # ---- validation gate (pre-registered thresholds in config.py)
    for hh, m in R["models"].items():
        reasons = []
        ho, wf = m["holdout"], m["walk_forward"]
        if ho["mean"] is None or ho["n_dates"] < 20:
            status = "INSUFFICIENT_DATA"
            reasons.append("holdout too short")
        else:
            if not (ho["mean"] > 0 and (ho["t"] or 0) >= C.GATE_MIN_HOLDOUT_IC_T):
                reasons.append(f"holdout IC t={ho['t']:.2f} < {C.GATE_MIN_HOLDOUT_IC_T}" if ho["t"] is not None else "holdout IC n/a")
            if (m["wf_positive_year_frac"] or 0) < C.GATE_MIN_WF_POSITIVE_YEAR_FRAC:
                reasons.append(f"walk-forward positive-IC years {m['wf_positive_year_frac']:.0%} < {C.GATE_MIN_WF_POSITIVE_YEAR_FRAC:.0%}")
            bh = m["backtest_holdout"]
            if not bh or bh["mean_net_excess"] <= 0:
                reasons.append("holdout long-only net excess <= 0 after costs")
            if hh == C.PRIMARY_HORIZON and (dsr is None or dsr < C.GATE_MIN_DSR):
                reasons.append(f"deflated Sharpe {dsr if dsr is None else round(dsr, 3)} < {C.GATE_MIN_DSR}")
            if m["calibration"]["status"] != "HISTORICAL_DECILE_CALIBRATED":
                reasons.append("forecast deciles not monotonic out-of-sample")
            wf_sig = wf["t"] is not None and wf["t"] >= 2 and wf["mean"] > 0
            status = "PAPER_TEST_ELIGIBLE" if not reasons else ("RESEARCH_CANDIDATE" if wf_sig else "UNVALIDATED")
        m["gate"] = {"status": status, "failed_checks": reasons}
        m.pop("_bt_oos", None)

    R["tested_hypotheses"] = n_hyp + 3 * len(C.HORIZONS)  # univariate tests + model, 2 baselines per horizon

    # ---- persist model + results
    model = {"model_id": C.MODEL_ID, "model_version": C.MODEL_VERSION, "feature_set_version": C.FEATURE_SET_VERSION,
             "run_id": run_id, "config_hash": config_hash(), "features": list(Xm.columns),
             "coefficients": {h: dict(zip(Xm.columns, m.coef_.tolist())) for h, m in final_models.items()},
             "intercepts": {h: float(m.intercept_) for h, m in final_models.items()},
             "calibration": {h: R["models"][h]["calibration"] for h in C.HORIZONS},
             "gate": {h: R["models"][h]["gate"] for h in C.HORIZONS},
             "validation": {h: {"oos_ic": R["models"][h]["holdout"]["mean"], "oos_ic_se": R["models"][h]["holdout"].get("se"),
                                "net_sharpe": (R["models"][h]["backtest_holdout"] or {}).get("net_sharpe_ann")} for h in C.HORIZONS},
             "deflated_sharpe_ratio": dsr, "tested_hypotheses": R["tested_hypotheses"],
             "trained_through": str(pd.Timestamp(dates.max()).date())}
    (C.PROC / "model.json").write_text(json.dumps(model, indent=1, default=float))
    (C.REPORTS / "research_results.json").write_text(json.dumps(R, indent=1, default=str))
    with (C.LEDGER / "research_runs.jsonl").open("a") as f:
        f.write(json.dumps({"run_id": run_id, "config_hash": config_hash(), "at": R["generated_at_utc"],
                            "holdout_start": R["sample"]["holdout_start"], "tested_hypotheses": R["tested_hypotheses"],
                            "gate": {h: R["models"][h]["gate"]["status"] for h in C.HORIZONS},
                            "holdout_ic": {h: R["models"][h]["holdout"]["mean"] for h in C.HORIZONS}}) + "\n")
    print(json.dumps({h: {"wf_ic": R["models"][h]["walk_forward"]["mean"], "wf_t": R["models"][h]["walk_forward"]["t"],
                          "ho_ic": R["models"][h]["holdout"]["mean"], "ho_t": R["models"][h]["holdout"]["t"],
                          "ho_net": (R["models"][h]["backtest_holdout"] or {}).get("mean_net_excess"),
                          "gate": R["models"][h]["gate"]} for h in C.HORIZONS}, indent=1, default=str))
    print("overfitting:", R["overfitting"])
    return R


if __name__ == "__main__":
    run()
