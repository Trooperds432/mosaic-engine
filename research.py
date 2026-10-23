"""Statistical validation of the feature library and the model suite (research v2).

Protocol (sections 5-9): discovery sample (all pre-holdout dates) -> univariate IC / quantile tests in raw,
industry-neutral and style-neutral (sector + size + beta) form with HAC inference, stationary-bootstrap CIs and
BH-FDR -> purged, embargoed expanding-window walk-forward of every model -> leakage-free ensemble -> one
evaluation on the untouched holdout -> decile calibration + Mondrian conformal coverage -> cost-adjusted
long-only backtests with a per-name cost model -> robustness splits, cost and parameter sensitivity (walk-
forward period only) -> White Reality Check / Hansen SPA, PSR / DSR / PBO / MinTRL -> factor-adjusted alpha
and regression diagnostics -> regime (observable + filtered HMM) conditioning test -> validation gate.

Preprocessing is per-date cross-sectional (percentile ranks), so no scaling parameter crosses time; ranks make
winsorisation unnecessary. Model-specific fitting (label clipping, calibration) happens inside each training
fold only; hyper-parameters are fixed in config/this file, not tuned on the holdout.
"""
import hashlib
from functools import partial
import itertools
import json
import math
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNet, HuberRegressor, LogisticRegression, Ridge
from sklearn.preprocessing import SplineTransformer

import config as C
from features import FEATURES

EMBARGO = 5
MODELS_DIR = C.ROOT / "models"
GAM_FEATS = ["mom_12_1", "ret_1m", "vol_60", "log_mcap", "earn_yield", "book_to_mkt", "gross_prof", "sue", "ins_net_90", "io_breadth_chg"]
FULL_SUITE_H = (5, 20)
MISSING_GROUPS = {"fund_missing": ["earn_yield", "book_to_mkt", "fcf_yield", "gross_prof", "roa"], "io_missing": ["io_ratio"],
                  "si_missing": ["si_ratio"], "sue_missing": ["sue"]}
RNG = np.random.default_rng(20261019)


def config_hash(feats, prof):
    keys = ["HORIZONS", "PRIMARY_HORIZON", "DECISION_STEP", "HOLDOUT_YEARS", "WF_MIN_TRAIN_YEARS", "RIDGE_ALPHA",
            "N_QUANTILES", "FX_FEE", "HALF_SPREAD", "SLIPPAGE", "FEATURE_SET_VERSION"]
    blob = json.dumps({k: getattr(C, k) for k in keys} | {"features": feats, "profile": prof}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ------------------------------------------------------------- preprocessing
def preprocess(panel, feats):
    """Per-date percentile rank in [-0.5, 0.5]; missing -> 0 (the cross-sectional median) plus group indicators."""
    X = pd.DataFrame(index=panel.index)
    g = panel["date"]
    for f in feats:
        x = panel[f].astype("float64").replace([np.inf, -np.inf], np.nan)
        X[f] = (x.groupby(g).rank(pct=True) - 0.5).fillna(0.0).astype("float32")
    for k, cols in MISSING_GROUPS.items():
        cols = [c for c in cols if c in panel]
        if cols:
            X[k] = panel[cols].isna().all(axis=1).astype("float32")
    return X


def sector_neutral(X, panel, feats):
    Xn = X.copy()
    key = [panel["date"], panel["sector"]]
    for f in feats:
        Xn[f] = (X[f] - X[f].groupby(key).transform("mean")).astype("float32")
    return Xn


class StyleNeutraliser:
    """Per-date projection off [1, size rank, beta rank, sector dummies] (conditional neutralisation, 5.1)."""

    def __init__(self, X, panel):
        self.blocks = []
        for d, idx in panel.groupby("date").indices.items():
            Z = [np.ones(len(idx)), X["log_mcap"].to_numpy()[idx], X["beta_252"].to_numpy()[idx]]
            sec = panel["sector"].fillna("Unknown").to_numpy()[idx]
            for s in np.unique(sec)[1:]:
                Z.append((sec == s).astype(float))
            Q, _ = np.linalg.qr(np.column_stack(Z))
            self.blocks.append((idx, Q))

    def __call__(self, col):
        out = np.empty(len(col), dtype="float32")
        v = col.to_numpy(dtype="float64")
        for idx, Q in self.blocks:
            x = v[idx]
            out[idx] = x - Q @ (Q.T @ x)
        return pd.Series(out, index=col.index)


# ------------------------------------------------------------- inference helpers
def safe(fn, *a, **k):
    """Run a diagnostic; return {'error': ...} instead of aborting the research run."""
    try:
        return fn(*a, **k)
    except Exception as ex:
        print("diagnostic error", fn.__name__, repr(ex)[:200], flush=True)
        return {"error": repr(ex)[:300]}


def hac_mean(series, lags):
    s = pd.Series(series).dropna()
    if len(s) < 10:
        return {"mean": None, "se": None, "t": None, "p": None, "n": int(len(s))}
    res = sm.OLS(s.to_numpy(), np.ones(len(s))).fit(cov_type="HAC", cov_kwds={"maxlags": int(lags)})
    return {"mean": float(res.params[0]), "se": float(res.bse[0]), "t": float(res.tvalues[0]), "p": float(res.pvalues[0]), "n": int(len(s))}


def stationary_bootstrap_idx(n, block, reps):
    """Politis-Romano stationary bootstrap indices (geometric block lengths, mean `block`)."""
    p = 1 / max(block, 1)
    starts = RNG.integers(n, size=(reps, n))
    new = RNG.random((reps, n)) < p
    new[:, 0] = True
    out = np.empty((reps, n), dtype=np.int64)
    for r in range(reps):
        cur = starts[r, 0]
        for t in range(n):
            cur = starts[r, t] if new[r, t] else (cur + 1) % n
            out[r, t] = cur
    return out


def boot_ci(series, block, reps=1000):
    s = pd.Series(series).dropna().to_numpy()
    if len(s) < 20:
        return None
    m = s[stationary_bootstrap_idx(len(s), block, reps)].mean(axis=1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def ic_series(score, y, dates, min_n=20):
    df = pd.DataFrame({"s": np.asarray(score, dtype="float64"), "y": np.asarray(y, dtype="float64"), "d": dates}).dropna()
    df = df[df.groupby("d")["y"].transform("count") >= min_n]
    if not len(df):
        return pd.Series(dtype=float)
    df["rs"] = df.groupby("d")["s"].rank()
    df["ry"] = df.groupby("d")["y"].rank()
    df["rs"] -= df.groupby("d")["rs"].transform("mean")
    df["ry"] -= df.groupby("d")["ry"].transform("mean")
    g = df.assign(xy=df["rs"] * df["ry"], xx=df["rs"] ** 2, yy=df["ry"] ** 2).groupby("d")
    out = g["xy"].sum() / np.sqrt(g["xx"].sum() * g["yy"].sum())
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def ic_summary(ic, h, boot=False):
    lags = max(1, math.ceil(h / C.DECISION_STEP))
    out = hac_mean(ic, lags)
    out.update({"ic_std": float(ic.std()) if len(ic) > 1 else None,
                "ic_ir": float(ic.mean() / ic.std()) if len(ic) > 1 and ic.std() > 0 else None,
                "pos_frac": float((ic > 0).mean()) if len(ic) else None,
                "n_dates": int(len(ic)), "effective_n": int(len(ic) * C.DECISION_STEP / max(h, C.DECISION_STEP)), "nw_lags": lags})
    if out.get("mean") is not None:
        out["ci95_hac"] = [out["mean"] - 1.96 * out["se"], out["mean"] + 1.96 * out["se"]]
        if boot:
            out["ci95_block_bootstrap"] = boot_ci(ic, 2 * lags)
    return out


def quantile_stats(score, y, dates, sectors, q=C.N_QUANTILES):
    df = pd.DataFrame({"s": score, "y": y, "d": dates, "sec": sectors}).dropna()
    df = df[df.groupby("d")["y"].transform("count") >= 4 * q]
    if not len(df):
        return None
    df["q"] = df.groupby("d")["s"].transform(lambda s: pd.qcut(s.rank(method="first"), q, labels=False))
    per = df.groupby(["d", "q"])["y"].mean().unstack()
    means = per.mean()
    top, bot = df[df["q"] == q - 1], df[df["q"] == 0]
    contrib = {}  # sector contributions to the top-minus-bottom spread
    for sec in df["sec"].unique():
        t = top[top["sec"] == sec].groupby("d")["y"].sum() / top.groupby("d").size()
        b = bot[bot["sec"] == sec].groupby("d")["y"].sum() / bot.groupby("d").size()
        contrib[sec] = float(t.reindex(per.index).fillna(0).mean() - b.reindex(per.index).fillna(0).mean())
    return {"quantile_means": [float(v) for v in means.to_numpy()], "top_minus_bottom": float(means.iloc[-1] - means.iloc[0]),
            "monotonicity": float(stats.spearmanr(np.arange(len(means)), means.to_numpy())[0]), "sector_contribution": contrib}


def bh_fdr(pvals, q):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, n + 1) / n
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
    return {"sr": float(r.mean() / r.std()), "skew": float(stats.skew(r)), "kurt": float(stats.kurtosis(r, fisher=False)), "n": int(len(r))}


def psr(sr, sr0, n, skew, kurt):
    den = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr))
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(n - 1) / den))


def deflated_sr(st, trial_srs):
    if st is None or len(trial_srs) < 2:
        return None, None
    v = float(np.var(trial_srs, ddof=1))
    g = 0.5772156649
    n = len(trial_srs)
    sr0 = math.sqrt(v) * ((1 - g) * stats.norm.ppf(1 - 1 / n) + g * stats.norm.ppf(1 - 1 / (n * math.e)))
    return psr(st["sr"], sr0, st["n"], st["skew"], st["kurt"]), sr0


def min_track_record(st, sr_target=0.0, conf=0.95):
    """Bailey & Lopez de Prado MinTRL: periods needed for PSR(sr_target) >= conf (None if SR <= target)."""
    if st is None or st["sr"] <= sr_target:
        return None
    z = stats.norm.ppf(conf)
    return float(1 + (1 - st["skew"] * st["sr"] + (st["kurt"] - 1) / 4 * st["sr"] ** 2) * (z / (st["sr"] - sr_target)) ** 2)


def pbo_cscv(M, S=10):
    # strategies start at different dates: keep periods where >= 80% are live, missing -> 0 (no position)
    M = M.dropna(axis=1, how="all")
    M = M[M.notna().mean(axis=1) >= 0.8].fillna(0)
    T, N = M.shape
    if N < 3 or T < S * 4:
        return None
    blocks = np.array_split(np.arange(T), S)
    lam = []
    for comb in itertools.combinations(range(S), S // 2):
        ins = np.concatenate([blocks[i] for i in comb])
        oos = np.concatenate([blocks[i] for i in range(S) if i not in comb])
        a, b = M.iloc[ins], M.iloc[oos]
        best = (a.mean() / a.std()).idxmax()
        w = (b.mean() / b.std()).rank().loc[best] / (N + 1)
        lam.append(math.log(w / (1 - w)))
    return float(np.mean(np.array(lam) <= 0))


def reality_check_spa(M, block=4, reps=1000):
    """White (2000) Reality Check and Hansen (2005) consistent SPA p-values for H0: no strategy has positive
    expected net excess return. M: periods x strategies (non-overlapping)."""
    from validation_extensions import multiple_comparison
    return multiple_comparison(M, block=block, reps=reps)


# ------------------------------------------------------------- costs
def name_costs(panel, order_usd):
    """Per-row one-way cost: FX fee + max(half Corwin-Schultz spread, config half-spread) + square-root impact
    0.7 x daily vol x sqrt(order $ / ADV $) at the bot's actual order size."""
    hs = (panel["cs_spread"] / 2).clip(lower=C.HALF_SPREAD, upper=0.02).fillna(C.HALF_SPREAD)
    dvol = (panel["vol_60"] / math.sqrt(252)).fillna(0.02)
    imp = 0.7 * dvol * np.sqrt(order_usd / panel["adv_dollar"].clip(lower=1e5))
    return (C.FX_FEE + hs + imp).astype("float64")


def rebalance_calendar(dates, h):
    """Anchor the schedule to input decision dates, before filtering any strategy's scores."""
    if h < 1:
        raise ValueError("backtest horizon must be positive")
    return pd.DatetimeIndex(pd.unique(dates)).sort_values()[::max(1, h // C.DECISION_STEP)]


def benchmark_label_returns(spy, dates, h):
    """Same adjusted t+1-open to t+1+h-open benchmark as features.labels."""
    positions = spy.index.get_indexer(pd.to_datetime(dates))
    if (positions < 0).any():
        raise ValueError('decision date not present in benchmark calendar')
    prices = (spy['open'] * spy['adjclose'] / spy['close']).to_numpy(dtype=float)
    result = np.full(len(positions), np.nan)
    valid = positions + 1 + h < len(prices)
    result[valid] = prices[positions[valid] + 1 + h] / prices[positions[valid] + 1] - 1
    return result


def long_only_backtest(score, y, dates, syms, cost, h, top=C.SIGNAL_TOP_QUANTILE, cost_mult=1.0, bench=None,
                       rebalance_dates=None, holdings=None, *, benchmark_returns=None, session_positions=None,
                       entry_prices=None, trades=None, buffer_top=None):
    """Equal-weight top-quantile long-only, rebalanced every h sessions (non-overlapping), excess vs SPY (or vs
    `bench`), net of per-name one-way costs on every name bought or sold."""
    df = pd.DataFrame({"s": np.asarray(score, dtype="float64"), "y": np.asarray(y, dtype="float64"), "d": dates, "sym": syms,
                       "c": np.asarray(cost, dtype="float64")})
    if benchmark_returns is not None:
        from holdings_simulation import simulate
        df['d'] = pd.to_datetime(df['d'])
        df['b'] = np.asarray(benchmark_returns, dtype=float)
        df['relative_benchmark'] = 0. if bench is None else np.asarray(bench, dtype=float)
        df['entry_price'] = np.nan if entry_prices is None else np.asarray(entry_prices, dtype=float)
        if session_positions is None:
            ordinal = {d: i * C.DECISION_STEP for i, d in enumerate(sorted(df.d.unique()))}
            df['position'] = df.d.map(ordinal)
        else:
            df['position'] = np.asarray(session_positions)
        df['exit_position'] = df.position + h
        schedule = rebalance_calendar(dates, h) if rebalance_dates is None else pd.DatetimeIndex(rebalance_dates)
        return simulate(df, schedule, top, cost_mult=cost_mult, buffer_top=buffer_top, holdings=holdings, trades=trades)
    if buffer_top is not None or trades is not None:
        raise ValueError('drift/buffer/fill ledger requires actual benchmark returns')
    if bench is not None:
        df["y"] = df["y"] - np.asarray(bench, dtype="float64")
    if not 0 < top <= 1 or cost_mult < 0:
        raise ValueError("invalid portfolio fraction or cost multiplier")
    df['d'] = pd.to_datetime(df['d'])
    if df.duplicated(['d', 'sym']).any():
        raise ValueError("duplicate security/decision-date rows")
    keep = rebalance_calendar(dates, h) if rebalance_dates is None else pd.DatetimeIndex(rebalance_dates)
    if not keep.is_unique or not keep.is_monotonic_increasing:
        raise ValueError("rebalance dates must be unique and ordered")
    # Select without consulting future outcomes. Never replace a missing outcome
    # with a lower-ranked security whose return happens to be available.
    df = df[np.isfinite(df['s']) & np.isfinite(df['c']) & (df['c'] >= 0)]
    df = df[df["d"].isin(keep)]
    rows, prev, missing = [], {}, []
    for d, g in df.groupby("d"):
        k = max(1, int(round(len(g) * top)))
        pick = g.sort_values(['s', 'sym'], ascending=[False, True]).head(k)
        if not np.isfinite(pick['y']).all():
            missing.append(str(pd.Timestamp(d).date()))
            continue
        names = dict(zip(pick["sym"], pick["c"]))
        w = 1 / len(names)
        traded = set(names) ^ set(prev)
        c = sum(names.get(s, prev.get(s, 0)) for s in traded) * w * cost_mult if prev else sum(names.values()) * w * cost_mult
        gross = pick["y"].mean()
        rows.append({"d": d, "gross": gross, "net": gross - c, "turnover": 1.0 if not prev else len(traded) / (2 * len(names)), "n": k, "cost": c})
        if holdings is not None:
            holdings.extend({"d": d, "symbol": r.sym, "weight": w, "score": float(r.s),
                             "realized_excess": float(r.y), "one_way_cost": float(r.c)} for r in pick.itertuples())
        prev = names
    result = pd.DataFrame(rows).set_index("d") if rows else pd.DataFrame(columns=["gross", "net", "turnover", "n", "cost"])
    result.attrs['missing_selected_outcome_dates'] = missing
    result.attrs['accounting'] = 'legacy_excess_return_approximation'
    return result


def bt_summary(bt, h):
    if len(bt) < 4:
        return None
    st = sharpe_stats(bt["net"])
    result = {"periods": int(len(bt)), "mean_gross_excess": float(bt["gross"].mean()), "mean_net_excess": float(bt["net"].mean()),
            "mean_cost": float(bt["cost"].mean()), "mean_turnover": float(bt["turnover"].mean()), "avg_names": float(bt["n"].mean()),
            "net_sharpe_ann": float(st["sr"] * math.sqrt(252 / max(h, 1))) if st else None, "net_t_hac": hac_mean(bt["net"], 2)["t"],
            "hit_rate": float((bt["net"] > 0).mean()), "max_drawdown_excess": float((bt["net"].cumsum() - bt["net"].cumsum().cummax()).min()),
            'accounting': bt.attrs.get('accounting', 'unknown'),
            'missing_selected_outcome_dates': bt.attrs.get('missing_selected_outcome_dates', [])}
    if 'nav_end' in bt:
        nav = np.r_[float(bt.nav_start.iloc[0]), bt.nav_end.to_numpy()]
        result.update(portfolio_total_return=float(nav[-1] / nav[0] - 1),
                      max_drawdown_portfolio=float(np.min(nav / np.maximum.accumulate(nav) - 1)),
                      mean_transaction_cost=float(bt.transaction_cost.mean()))
    return {k: None if isinstance(v, float) and not np.isfinite(v) else v for k, v in result.items()}


# ------------------------------------------------------------- models
class GAM:
    """Additive spline terms for a pre-specified feature subset plus linear terms for the rest (ridge-penalised)."""

    def __init__(self, cols):
        self.cols = cols
        self.sp = SplineTransformer(n_knots=5, degree=3)
        self.r = Ridge(alpha=50.0)

    def _z(self, X, fit=False):
        cols = [c for c in self.cols if c in X]
        Z = self.sp.fit_transform(X[cols].to_numpy(dtype="float32")) if fit else self.sp.transform(X[cols].to_numpy(dtype="float32"))
        return np.hstack([Z.astype("float32"), X.to_numpy(dtype="float32")])

    def fit(self, X, y):
        self.r.fit(self._z(X, True), y)
        return self

    def predict(self, X):
        return self.r.predict(self._z(X))


class Hierarchical:
    """Partial pooling: global ridge + per-sector ridge on the global residual with 20x stronger shrinkage."""

    def __init__(self, sectors_train, alpha):
        self.sec, self.alpha = sectors_train, alpha

    def fit(self, X, y):
        self.g = Ridge(alpha=self.alpha).fit(X, y)
        res = y - self.g.predict(X)
        self.s = {s: Ridge(alpha=self.alpha * 20).fit(X[self.sec == s], res[self.sec == s]) for s in np.unique(self.sec) if (self.sec == s).sum() > 2000}
        return self

    def predict(self, X, sectors):
        p = self.g.predict(X)
        for s, mdl in self.s.items():
            m = sectors == s
            if m.any():
                p[m] += mdl.predict(X[m])
        return p


class CalibratedLogit:
    """P(excess > round-trip cost): logistic on the first 80% of the time-ordered training rows, isotonic
    calibration on the last 20%."""

    def __init__(self, threshold):
        self.th = threshold

    def fit(self, X, y):
        cut = int(len(y) * 0.8)
        lab = (y > self.th).astype(int)
        self.lr = LogisticRegression(C=0.05, max_iter=300).fit(X[:cut], lab[:cut])
        self.iso = IsotonicRegression(out_of_bounds="clip").fit(self.lr.predict_proba(X[cut:])[:, 1], lab[cut:])
        return self

    def predict(self, X):
        return self.iso.predict(self.lr.predict_proba(X)[:, 1])


for _cls in (GAM, Hierarchical, CalibratedLogit):  # pickle under the module name, not __main__, so signals.py can load them
    _cls.__module__ = "research"


def make_model(name, n_rows, sectors=None, cost=None):
    a = C.RIDGE_ALPHA * n_rows / 1000
    if name in ("ridge", "factor_resid"):
        return Ridge(alpha=a)
    if name == "enet":
        return ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=2000)
    if name == "huber":
        return HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=300)
    if name == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=500, subsample=0.7,
                                 subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0, verbose=-1, n_jobs=2)
    if name == "gam":
        return GAM(GAM_FEATS)
    if name == "hier":
        return Hierarchical(sectors, a)
    if name == "logit":
        return CalibratedLogit(cost)
    raise ValueError(name)


def fit_predict(name, Xtr, ytr, Xte, sec_tr=None, sec_te=None, cost=None):
    lo, hi = np.nanquantile(ytr, [C.WINSOR_Q, 1 - C.WINSOR_Q])  # label clipping thresholds from training labels only
    yc = np.clip(ytr, lo, hi)
    if name == "huber" and len(yc) > 150_000:  # IRLS on a deterministic, evenly spaced subsample
        sel = np.linspace(0, len(yc) - 1, 150_000).astype(int)
        Xtr, yc, ytr = Xtr.iloc[sel], yc[sel], ytr[sel]
        sec_tr = sec_tr[sel] if sec_tr is not None else None
    m = make_model(name, len(yc), sec_tr, cost)
    if name == "logit":
        m.fit(Xtr.to_numpy(dtype="float32"), ytr)
    elif name in ("gam", "lgbm"):
        m.fit(Xtr, yc)
    else:
        m.fit(Xtr.to_numpy(dtype="float32"), yc)
    if name == "hier":
        return m, m.predict(Xte.to_numpy(dtype="float32"), sec_te)
    if name in ("gam", "lgbm"):
        return m, m.predict(Xte)
    return m, m.predict(Xte.to_numpy(dtype="float32"))


def walk_forward(name, X, y, dates, tpos, h, hpos, sectors, cost):
    """Yearly expanding refits; training labels must end (t+1+h) before the test block starts minus an embargo."""
    ok = ~np.isnan(y)
    years = pd.DatetimeIndex(dates).year
    first_year = years.min() + C.WF_MIN_TRAIN_YEARS
    preds = np.full(len(y), np.nan)
    pre = tpos < hpos
    for yr in range(first_year, years.max() + 1):
        test = pre & (years == yr)
        if not test.any():
            continue
        train = ok & (tpos + 1 + h + EMBARGO < tpos[test].min())
        if train.sum() < 5000:
            continue
        _, p = fit_predict(name, X[train], y[train], X[test], sectors[train], sectors[test], cost)
        preds[test] = p
    hold = tpos >= hpos
    train = ok & (tpos + 1 + h + EMBARGO < hpos)
    _, p = fit_predict(name, X[train], y[train], X[hold], sectors[train], sectors[hold], cost)
    hp = np.full(len(y), np.nan)
    hp[hold] = p
    return preds, hp


def date_rank(p, dates):
    return pd.Series(p).groupby(np.asarray(dates)).rank(pct=True).to_numpy() - 0.5


def calibration(pred, y, dates, buckets=C.CAL_BUCKETS):
    df = pd.DataFrame({"p": pred, "y": y, "d": dates}).dropna()
    df = df[df.groupby("d")["y"].transform("count") >= 2 * buckets]
    if not len(df):
        return pd.DataFrame()
    df["b"] = df.groupby("d")["p"].transform(lambda s: pd.qcut(s.rank(method="first"), buckets, labels=False))
    return df.groupby("b")["y"].agg(mean="mean", std="std", n="count", q025=lambda s: s.quantile(0.025), q05=lambda s: s.quantile(0.05),
                                    q975=lambda s: s.quantile(0.975), p_pos=lambda s: (s > 0).mean(),
                                    es05=lambda s: s[s <= s.quantile(0.05)].mean())


def conformal_coverage(table, pred, y, dates, buckets=C.CAL_BUCKETS):
    """Mondrian (per-decile) split-conformal: 95% intervals from walk-forward deciles, coverage on the holdout."""
    df = pd.DataFrame({"p": pred, "y": y, "d": dates}).dropna()
    df = df[df.groupby("d")["y"].transform("count") >= 2 * buckets]
    if not len(df) or not len(table):
        return None
    df["b"] = df.groupby("d")["p"].transform(lambda s: pd.qcut(s.rank(method="first"), buckets, labels=False))
    lo, hi = df["b"].map(table["q025"]), df["b"].map(table["q975"])
    return {"nominal": 0.95, "holdout_coverage": float(((df["y"] >= lo) & (df["y"] <= hi)).mean()), "n": int(len(df))}


# ------------------------------------------------------------- regimes
def hmm_filtered(cal, spy_ac, vix):
    """2-state Gaussian HMM on (SPY daily return, VIX log change), refit each January on prior data only; returns
    the FILTERED P(high-vol state) from the forward recursion (never smoothed posteriors)."""
    from hmmlearn.hmm import GaussianHMM
    obs = pd.DataFrame({"r": spy_ac.pct_change(), "v": np.log(vix).diff()}, index=cal).dropna()
    p_hv = pd.Series(np.nan, index=cal)
    years = sorted(set(obs.index.year))
    for yr in years[3:]:
        tr = obs[obs.index.year < yr].to_numpy()
        m = GaussianHMM(n_components=2, covariance_type="full", n_iter=200, random_state=7).fit(tr)
        hv = int(np.argmax([m.covars_[k][0, 0] for k in range(2)]))
        sel = obs.index[obs.index.year <= yr]
        Xo = obs.loc[sel].to_numpy()
        logB = np.column_stack([stats.multivariate_normal(m.means_[k], m.covars_[k], allow_singular=True).logpdf(Xo) for k in range(2)])
        alpha = m.startprob_
        filt = np.empty((len(Xo), 2))
        for t in range(len(Xo)):
            a = (alpha @ m.transmat_ if t else m.startprob_) * np.exp(logB[t] - logB[t].max())
            alpha = a / a.sum()
            filt[t] = alpha
        mask = sel.year == yr
        p_hv.loc[sel[mask]] = filt[mask, hv]
    return p_hv


# ------------------------------------------------------------- factor-adjusted alpha
def factor_alpha(bt, h, profile):
    f = C.PROC / f"factor_returns_{profile}.parquet"
    if not f.exists() or len(bt) < 12:
        return None
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.stattools import durbin_watson, jarque_bera
    F = pd.read_parquet(f)
    cal = F.index
    rows = []
    for d in bt.index:
        i = cal.searchsorted(d) + 1
        rows.append(((1 + F.iloc[i:i + h]).prod() - 1).rename(d))
    Fx = pd.DataFrame(rows)[["market", "size", "value", "momentum", "quality", "volatility", "beta"]]
    Xc = sm.add_constant(Fx).to_numpy()
    res = sm.OLS(bt["net"].to_numpy(), Xc).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    return {"alpha_per_period": float(res.params[0]), "alpha_t_hac": float(res.tvalues[0]), "r2": float(res.rsquared),
            "loadings": dict(zip(Fx.columns, [float(v) for v in res.params[1:]])), "durbin_watson": float(durbin_watson(res.resid)),
            "jarque_bera_p": float(jarque_bera(res.resid)[1]), "breusch_pagan_p": float(het_breuschpagan(res.resid, Xc)[1]),
            "vif": {c: float(variance_inflation_factor(Xc, i + 1)) for i, c in enumerate(Fx.columns)}, "periods": int(len(bt))}


# ------------------------------------------------------------- main
def run(profile=None):
    prof = C.PROFILES[profile] if profile else C.PROFILE
    pname = prof["name"]
    panel = pd.read_parquet(C.PROC / f"panel_{pname}.parquet")
    feats = [f for f in FEATURES if f in panel and panel[f].notna().mean() > 0.02]
    # snapshot features (options / estimates collected nightly since 2026-10-20) need a long enough history to be tested
    from features import SNAPSHOT_FEATURES, SNAPSHOT_MIN_YEARS
    span = {f: (panel.loc[panel[f].notna(), "date"].max() - panel.loc[panel[f].notna(), "date"].min()).days / 365.25 for f in feats if f in SNAPSHOT_FEATURES}
    feats = [f for f in feats if f not in SNAPSHOT_FEATURES or span.get(f, 0) >= SNAPSHOT_MIN_YEARS]
    from features import NON_PIT_FEATURES
    feats = [f for f in feats if f not in NON_PIT_FEATURES]  # cannot be reconstructed point-in-time
    spy = pd.read_parquet(C.RAW / "prices" / f"{C.BENCHMARK}.parquet").drop_duplicates("date").set_index("date")
    cal = spy.index
    panel["tpos"] = cal.get_indexer(panel["date"])
    X = preprocess(panel, feats)
    Xn = sector_neutral(X, panel, feats)
    style = StyleNeutraliser(X, panel)
    dates, tpos = panel["date"].to_numpy(), panel["tpos"].to_numpy()
    syms, sectors = panel["symbol"].to_numpy(), panel["sector"].fillna("Unknown").to_numpy()
    hpos = int(cal.searchsorted(cal[-1] - pd.DateOffset(years=C.HOLDOUT_YEARS)))
    pre, hm = tpos < hpos, tpos >= hpos
    from risk_limits import load as load_limits
    order_usd = load_limits().get("order_size_usd") or 600.0
    cost = name_costs(panel, order_usd)
    rt_cost = float(cost.mean() * 2)
    chash = config_hash(feats, pname)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{pname}-" + chash
    R = {"run_id": run_id, "profile": pname, "config_hash": chash, "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "features": feats, "sample": {"first_date": str(pd.Timestamp(dates.min()).date()), "last_date": str(pd.Timestamp(dates.max()).date()),
                                       "holdout_start": str(cal[hpos].date()), "rows": int(len(panel)), "symbols": int(panel["symbol"].nunique()),
                                       "decision_dates": int(panel["date"].nunique()), "avg_names_per_date": float(panel.groupby("date").size().mean())},
         "costs": {"model": "FX fee + max(half Corwin-Schultz spread, config half-spread) + 0.7*daily vol*sqrt(order/ADV$)", "order_size_usd": order_usd,
                   "mean_one_way": float(cost.mean()), "median_one_way": float(cost.median()), "fixed_round_trip_v1": C.ROUND_TRIP_COST,
                   'accounting': 'self-financing adjusted-return holdings; all dollar rebalances and liquidation charged',
                   'cost_column': 'gross minus net return drag; transaction_cost separately records paid fees / starting NAV'}}
    print("panel", R["sample"], flush=True)
    from historical_data import coverage_check
    R["historical_integrity"] = coverage_check(R["sample"]["first_date"], R["sample"]["last_date"])

    # ---- multicollinearity (discovery)
    corr = X.loc[pre, feats].corr()
    R["multicollinearity"] = {"pairs_abs_corr_gt_0_7": [(a, b, float(corr.loc[a, b])) for i, a in enumerate(feats) for b in feats[i + 1:] if abs(corr.loc[a, b]) > 0.7],
                              "vif": dict(zip(feats, np.diag(np.linalg.pinv(corr.fillna(0).to_numpy())).round(2).tolist()))}

    # ---- regimes: observable rules + filtered HMM
    vix = pd.read_parquet(C.RAW / "prices" / "^VIX.parquet").drop_duplicates("date").set_index("date")["close"].reindex(cal).ffill()
    spy_ac = spy["adjclose"]
    trend = spy_ac > spy_ac.rolling(200).mean()
    vix_hi = vix > vix.rolling(1260, min_periods=252).median()
    regime = pd.Series(np.where(trend, "uptrend", "downtrend").astype(object) + "/" + np.where(vix_hi, "high_vol", "low_vol").astype(object), index=cal)
    p_hv = hmm_filtered(cal, spy_ac, vix)
    panel["regime"] = regime.reindex(panel["date"]).to_numpy()
    panel["p_hv"] = p_hv.reindex(panel["date"]).to_numpy()
    R["current_regime"] = {"state": regime.iloc[-1], "spy_above_200d": bool(trend.iloc[-1]), "vix": float(vix.iloc[-1]),
                           "vix_5y_median": float(vix.rolling(1260, min_periods=252).median().iloc[-1]),
                           "hmm_p_high_vol_filtered": float(p_hv.dropna().iloc[-1]) if p_hv.notna().any() else None, "as_of": str(cal[-1].date())}
    pd.DataFrame({"regime": regime, "p_high_vol": p_hv}).to_parquet(C.PROC / "regime_history.parquet")
    print("regime", R["current_regime"], flush=True)

    # ---- univariate discovery tests
    uni, pvals = [], []
    for f in feats:
        variants = {"raw": X[f], "sector_neutral": Xn[f]}
        if f not in ("log_mcap", "beta_252"):
            variants["style_neutral"] = style(X[f])
        miss = panel[f].isna().to_numpy()
        for h in C.HORIZONS:
            yy = np.where(pre & ~miss, panel[f"y_{h}"].to_numpy(), np.nan)
            for vname, xs in variants.items():
                ic = ic_series(xs.to_numpy(), yy, dates)
                s = ic_summary(ic, h, boot=(h == C.PRIMARY_HORIZON))
                rec = {"feature": f, "family": FEATURES[f][0], "hyp_sign": FEATURES[f][1], "horizon": h, "variant": vname, **s}
                if vname == "raw" and h == C.PRIMARY_HORIZON:
                    rec.update(quantile_stats(xs.to_numpy(), yy, dates, sectors) or {})
                    rec["rolling_ic_52"] = {str(k.date()): float(v) for k, v in ic.rolling(52, min_periods=26).mean().dropna().iloc[::13].items()}
                    reg = pd.Series(regime.reindex(ic.index).to_numpy(), index=ic.index)
                    rec["regime_ic"] = {k: float(v.mean()) for k, v in ic.groupby(reg) if len(v) >= 10}
                p = s["p"]  # one-sided in the pre-specified direction; two-sided when no sign was hypothesised
                if p is not None and rec["hyp_sign"] != 0 and s["t"] is not None:
                    p = float(stats.norm.sf(s["t"] * rec["hyp_sign"]))
                uni.append(rec)
                pvals.append(p if p is not None else 1.0)
        print("univariate", f, flush=True)
    rej, padj = bh_fdr(pvals, C.FDR_Q)
    for rec, r_, pa, p in zip(uni, rej, padj, pvals):
        rec["p_test"], rec["fdr_reject"], rec["p_bh"] = p, bool(r_), float(pa)
        rec["sign_matches_hypothesis"] = rec["mean"] is not None and (rec["hyp_sign"] == 0 or np.sign(rec["mean"]) == rec["hyp_sign"])
    R["univariate"] = uni

    # ---- model suite
    miss_cols = [k for k in MISSING_GROUPS if k in X]
    Xm = pd.concat([Xn[feats], X[miss_cols]], axis=1)
    del Xn
    R["models"], final, strategy_returns = {}, {}, {}
    MODELS_DIR.mkdir(exist_ok=True)
    years = pd.DatetimeIndex(dates).year
    for h in C.HORIZONS:
        y = panel[f"y_{h}"].to_numpy().astype("float64")
        benchmark_returns = benchmark_label_returns(spy, dates, h)
        backtest = partial(long_only_backtest, benchmark_returns=benchmark_returns, session_positions=tpos,
                           entry_prices=panel['entry_adj_open'].to_numpy() if 'entry_adj_open' in panel else None)
        names = ["ridge", "enet", "huber", "lgbm", "gam", "hier", "logit", "factor_resid"] if h in FULL_SUITE_H else ["ridge"]
        y_res = None
        if "factor_resid" in names:  # residual return after style + sector exposures, per date
            y_res = style(pd.Series(np.nan_to_num(y), index=X.index)).to_numpy().astype("float64")
            y_res[np.isnan(y)] = np.nan
        oos, hold = {}, {}
        for name in names:
            wf, hp = walk_forward(name, Xm, y_res if name == "factor_resid" else y, dates, tpos, h, hpos, sectors, rt_cost)
            oos[name], hold[name] = wf, hp
            print("model", h, name, "wf IC", round(float(ic_series(wf, y, dates).mean()), 4), flush=True)
        ics = {n: ic_series(oos[n], y, dates) for n in names}
        yearly = {n: ics[n].groupby(pd.DatetimeIndex(ics[n].index).year).mean() for n in names}
        ranks = {n: date_rank(np.where(hm, hold[n], oos[n]), dates) for n in names}
        # leakage-free ensemble: in year t, average the ranks of models whose walk-forward IC in years < t was positive
        ens = np.full(len(y), np.nan)
        members_by_year = {}
        for yr in sorted(set(years[pre])):
            mem = [n for n in names if yearly[n][yearly[n].index < yr].mean() > 0] or names
            members_by_year[int(yr)] = mem
            m = pre & (years == yr)
            stack = np.stack([ranks[n][m] for n in mem])
            if not np.isnan(stack).all():
                ens[m] = np.nanmean(stack, axis=0)
        hold_members = [n for n in names if ics[n].mean() > 0] or names
        ens[hm] = np.nanmean(np.stack([ranks[n][hm] for n in hold_members]), axis=0)
        res = {"horizon": h, "models": {}, "ensemble_members_by_year": members_by_year, "holdout_ensemble_members": hold_members}
        for n in names + ["ensemble"]:
            p = ens if n == "ensemble" else np.where(hm, hold[n], oos[n])
            wfm, hpm = np.where(pre, p, np.nan), np.where(hm, p, np.nan)
            icw, ich = ic_series(wfm, y, dates), ic_series(hpm, y, dates)
            yr_ic = icw.groupby(pd.DatetimeIndex(icw.index).year).mean()
            bt_w = backtest(wfm, y, dates, syms, cost, h)
            res["models"][n] = {"walk_forward": ic_summary(icw, h, boot=True), "holdout": ic_summary(ich, h, boot=True),
                                "wf_years": {int(k): float(v) for k, v in yr_ic.items()}, "wf_positive_year_frac": float((yr_ic > 0).mean()) if len(yr_ic) else None,
                                "backtest_wf": bt_summary(bt_w, h), "backtest_holdout": bt_summary(backtest(hpm, y, dates, syms, cost, h), h)}
            if h == C.PRIMARY_HORIZON and len(bt_w):
                strategy_returns[f"model:{n}"] = bt_w["net"]
            if n == "ensemble" and h in FULL_SUITE_H:  # saved for baselines.py (matched random paths, block bootstrap)
                ledger, fills = [], []
                bt_w = backtest(wfm, y, dates, syms, cost, h, holdings=ledger, trades=fills)
                for r in ledger + fills:
                    r['sample'] = 'walk_forward'
                hold_ledger, hold_fills = [], []
                bt_h = backtest(hpm, y, dates, syms, cost, h, holdings=hold_ledger, trades=hold_fills)
                for r in hold_ledger + hold_fills:
                    r['sample'] = 'holdout'
                pd.DataFrame(ledger + hold_ledger).to_parquet(C.REPORTS / f'ensemble_holdings_{pname}_h{h}.parquet', index=False)
                pd.DataFrame(fills + hold_fills).to_parquet(C.REPORTS / f'ensemble_fills_{pname}_h{h}.parquet', index=False)
                pd.concat([bt_w.assign(sample="walk_forward"), bt_h.assign(sample="holdout")]).reset_index().to_parquet(
                    C.REPORTS / f"ensemble_backtest_{pname}_h{h}.parquet", index=False)
        if len(names) > 1:
            res["models"]["ensemble"]["model_disagreement_mean_rank_std"] = float(np.nanmean(np.nanstd(np.stack([ranks[n] for n in names]), axis=0)))
        e = res["models"]["ensemble"]
        if h == C.PRIMARY_HORIZON:
            from validation_extensions import cpcv_ridge
            from factor_persistence import summary as persistence_summary
            from institutional_research import write_comparison
            comparison = write_comparison(panel, ens, cost, h, cal[hpos], pname, C.REPORTS,
                                          {'run_id': run_id, 'historical_integrity': R['historical_integrity'],
                                           'order_size_usd': order_usd}, top=C.SIGNAL_TOP_QUANTILE,
                                          benchmark_returns=benchmark_returns, session_positions=tpos)
            res['fixed_vqm_comparison'] = comparison
            # Preserve genuine OOS scores for reproducible comparisons. Final
            # production weights trained on all labels cannot reconstruct these.
            pd.DataFrame({'date': dates, 'symbol': syms, 'ensemble_score': ens,
                          'sample': np.where(pre, 'walk_forward', 'holdout')}).to_parquet(
                C.REPORTS / f'oos_scores_{pname}_h{h}.parquet', index=False)
            res["ridge_cpcv"] = cpcv_ridge(Xm, y, dates, tpos, h, hpos, embargo=EMBARGO)
            e["factor_persistence_wf"] = persistence_summary(ens[pre], y[pre], dates[pre], syms[pre])
        tab_wf = calibration(np.where(pre, ens, np.nan), y, dates)
        tab_ho = calibration(np.where(hm, ens, np.nan), y, dates)
        tab = tab_wf  # exported expectations use walk-forward outcomes only: the reused holdout never feeds production numbers
        mono_wf = stats.spearmanr(tab_wf.index, tab_wf["mean"])[0] if len(tab_wf) > 2 else np.nan
        mono_ho = stats.spearmanr(tab_ho.index, tab_ho["mean"])[0] if len(tab_ho) > 2 else np.nan
        e["calibration"] = {"table": tab.reset_index().to_dict(orient="records"), "wf_monotonicity": float(mono_wf), "holdout_monotonicity": float(mono_ho),
                            "status": "HISTORICAL_DECILE_CALIBRATED" if (mono_wf >= 0.7 and mono_ho >= 0.5) else "NOT_CALIBRATED",
                            "conformal": conformal_coverage(tab_wf, np.where(hm, ens, np.nan), y, dates)}
        if "logit" in names:
            pl = np.where(hm, hold["logit"], np.nan)
            okb = ~np.isnan(pl) & ~np.isnan(y)
            lab = (y[okb] > rt_cost).astype(float)
            rel = pd.DataFrame({"p": pl[okb], "y": lab})
            rel["b"] = pd.cut(rel["p"], [0, .3, .4, .45, .5, .55, .6, .7, 1], include_lowest=True).astype(str)
            e["probability_model"] = {"target": f"P(excess > mean round-trip cost {rt_cost:.4f})", "holdout_brier": float(np.mean((pl[okb] - lab) ** 2)),
                                      "holdout_base_rate_brier": float(np.mean((lab.mean() - lab) ** 2)),
                                      "reliability": rel.groupby("b").agg(p=("p", "mean"), freq=("y", "mean"), n=("y", "size")).reset_index().to_dict(orient="records")}
        if h == C.PRIMARY_HORIZON:
            wfe = np.where(pre, ens, np.nan)
            try:  # diagnostics: a failure here is recorded, it must not discard the fitted models
                # regime conditioning: ridge + feature x (filtered P(high vol) - 0.5) vs the unconditional ridge
                ph = pd.Series(panel["p_hv"].fillna(0.5).to_numpy() - 0.5, index=X.index)
                Xr = pd.concat([Xm, Xm[feats].mul(ph, axis=0).add_suffix("_xhv").astype("float32")], axis=1)
                wf_r, _ = walk_forward("ridge", Xr, y, dates, tpos, h, hpos, sectors, None)
                del Xr
                d_ic = (ic_series(wf_r, y, dates) - ics["ridge"]).dropna()
                dstat = hac_mean(d_ic, math.ceil(h / C.DECISION_STEP))
                res["regime_conditioning"] = {"method": "ridge + feature x (filtered HMM P(high vol) - 0.5) vs unconditional ridge, walk-forward IC difference",
                                              "wf_ic_difference": dstat, "improves": bool((dstat["mean"] or 0) > 0 and (dstat["t"] or 0) > 2)}
                # robustness on walk-forward ensemble predictions (holdout untouched)
                wfe = np.where(pre, ens, np.nan)
                rob = {}

                def terc(c):
                    # percentile-rank terciles work for any cross-section size (early dates can hold very few names)
                    r = panel.groupby("date")[c].rank(pct=True)
                    return pd.cut(r, [0, 1 / 3, 2 / 3, 1], labels=["low", "mid", "high"], include_lowest=True).astype(str)
                splits = [("cap_tercile", terc("mcap")), ("liquidity_tercile", terc("adv_dollar")), ("sector", panel["sector"].fillna("Unknown")),
                          ("regime", panel["regime"])]
                if "earn_recent" in panel:
                    splits.append(("earnings_window", panel["earn_recent"].map({1.0: "earnings", 0.0: "no_earnings"})))
                for label, key in splits:
                    out = {}
                    for k in pd.unique(key.dropna()):
                        icx = ic_series(np.where((key == k).to_numpy(), wfe, np.nan), y, dates, min_n=10)
                        if len(icx) >= 20:
                            out[str(k)] = {"mean_ic": float(icx.mean()), "t": hac_mean(icx, math.ceil(h / C.DECISION_STEP))["t"], "n_dates": int(len(icx))}
                    rob[label] = out
                rob["cost_sensitivity_wf_net_per_period"] = {f"x{m}": (bt_summary(backtest(wfe, y, dates, syms, cost, h, cost_mult=m), h) or {}).get("mean_net_excess") for m in [0, 0.5, 1, 2, 3]}
                rob["top_quantile_sensitivity_wf_net"] = {str(q): (bt_summary(backtest(wfe, y, dates, syms, cost, h, top=q), h) or {}).get("mean_net_excess") for q in [0.1, 0.2, 0.3]}
                rob["benchmark_equal_weight_universe_wf"] = bt_summary(backtest(wfe, y, dates, syms, cost, h, bench=panel.groupby("date")[f"y_{h}"].transform("mean").to_numpy()), h)
                sens = {str(C.RIDGE_ALPHA): float(ics["ridge"].mean())}
                for a in [1.0, 100.0]:
                    old, C.RIDGE_ALPHA = C.RIDGE_ALPHA, a
                    wf_a, _ = walk_forward("ridge", Xm, y, dates, tpos, h, hpos, sectors, None)
                    C.RIDGE_ALPHA = old
                    sens[str(a)] = float(ic_series(wf_a, y, dates).mean())
                rob["ridge_alpha_sensitivity_wf_ic"] = sens
                res["robustness"] = rob
                e["factor_adjusted_alpha_wf"] = factor_alpha(backtest(wfe, y, dates, syms, cost, h), h, pname)
                e["factor_adjusted_alpha_holdout"] = factor_alpha(backtest(np.where(hm, ens, np.nan), y, dates, syms, cost, h), h, pname)
            except Exception as ex:
                res.setdefault("diagnostic_errors", []).append(repr(ex)[:300])
                print("diagnostic error", repr(ex)[:300], flush=True)
            for f in feats:  # signed single-feature strategies join the model strategies for RC/SPA/DSR/PBO
                bt = backtest(np.where(pre, (FEATURES[f][1] or 1) * Xm[f].to_numpy(), np.nan), y, dates, syms, cost, h)
                if len(bt):
                    strategy_returns[f"feat:{f}"] = bt["net"]
        R["models"][h] = res
        # production fits on every labelled row, saved for signals.py
        okp = ~np.isnan(y)
        for n in names:
            yt = y_res if n == "factor_resid" else y
            m, _ = fit_predict(n, Xm[okp & ~np.isnan(yt)], yt[okp & ~np.isnan(yt)], Xm.iloc[:2], sectors[okp & ~np.isnan(yt)], sectors[:2], rt_cost)
            joblib.dump(m, MODELS_DIR / f"{pname}_h{h}_{n}.joblib")
        final[h] = {"members": hold_members, "names": names}
        print("horizon done", h, {k: (v["walk_forward"]["mean"], v["holdout"]["mean"]) for k, v in res["models"].items()}, flush=True)

    # ---- overfitting diagnostics at the primary horizon
    h = C.PRIMARY_HORIZON
    M = pd.DataFrame(strategy_returns)
    trial_srs = [s["sr"] for s in (sharpe_stats(M[c]) for c in M) if s]
    e = R["models"][h]["models"]["ensemble"]
    st = sharpe_stats(M["model:ensemble"]) if "model:ensemble" in M else None
    dsr, sr0 = deflated_sr(st, trial_srs)
    R["overfitting"] = {"primary_horizon": h, "strategies_tested": int(M.shape[1]), "wf_periods": int(len(M)),
                        "ensemble_wf_sr_per_period": st["sr"] if st else None,
                        "psr_vs_zero": psr(st["sr"], 0, st["n"], st["skew"], st["kurt"]) if st else None,
                        "deflated_sharpe_ratio": dsr, "dsr_benchmark_sr": sr0, "min_track_record_periods": min_track_record(st),
                        "pbo_cscv": safe(pbo_cscv, M), "reality_check_spa": safe(reality_check_spa, M, block=max(2, h // C.DECISION_STEP)),
                        "train_vs_test_ic_degradation": (e["walk_forward"]["mean"] - e["holdout"]["mean"]) if e["walk_forward"]["mean"] is not None and e["holdout"]["mean"] is not None else None}
    print("overfitting", R["overfitting"], flush=True)

    # ---- validation gate on the ensemble (thresholds pre-set in config.py)
    for hh, res in R["models"].items():
        m = res["models"]["ensemble"]
        reasons = []
        for sample in ['backtest_wf', 'backtest_holdout']:
            if (m.get(sample) or {}).get('missing_selected_outcome_dates'):
                reasons.append(f'{sample}: selected outcomes missing; ledger incomplete')
        if not R["historical_integrity"]["passed"]:
            reasons.extend(R["historical_integrity"]["failed_checks"])
        ho, wf = m["holdout"], m["walk_forward"]
        if ho["mean"] is None or ho["n_dates"] < 20:
            status = "INSUFFICIENT_DATA"
            reasons.append("holdout too short")
        else:
            if not (ho["mean"] > 0 and (ho["t"] or 0) >= C.GATE_MIN_HOLDOUT_IC_T):
                reasons.append(f"holdout IC t={ho['t']:.2f} < {C.GATE_MIN_HOLDOUT_IC_T}")
            if (m["wf_positive_year_frac"] or 0) < C.GATE_MIN_WF_POSITIVE_YEAR_FRAC:
                reasons.append(f"walk-forward positive-IC years {m['wf_positive_year_frac']:.0%} < {C.GATE_MIN_WF_POSITIVE_YEAR_FRAC:.0%}")
            if not m["backtest_holdout"] or m["backtest_holdout"]["mean_net_excess"] <= 0:
                reasons.append("holdout long-only net excess <= 0 after per-name costs")
            if hh == C.PRIMARY_HORIZON:
                if dsr is None or dsr < C.GATE_MIN_DSR:
                    reasons.append(f"deflated Sharpe {dsr if dsr is None else round(dsr, 3)} < {C.GATE_MIN_DSR}")
                spa = R["overfitting"]["reality_check_spa"]
                from gates import joint_test_failures
                reasons.extend(joint_test_failures(spa))
            if m["calibration"]["status"] != "HISTORICAL_DECILE_CALIBRATED":
                reasons.append("forecast deciles not monotonic out-of-sample")
            wf_sig = wf["t"] is not None and wf["t"] >= 2 and wf["mean"] > 0
            status = "PAPER_TEST_ELIGIBLE" if not reasons else ("RESEARCH_CANDIDATE" if wf_sig else "UNVALIDATED")
        m["gate"] = {"status": status, "failed_checks": reasons}
    R["tested_hypotheses"] = len(uni) + sum(len(r["models"]) for r in R["models"].values()) + 13

    model = {"model_id": f"{C.MODEL_ID}-{pname}", "model_version": C.MODEL_VERSION, "feature_set_version": C.FEATURE_SET_VERSION,
             "run_id": run_id, "profile": pname, "config_hash": chash, "features": list(Xm.columns), "feats": feats,
             "horizons": {h: {"members": final[h]["members"], "names": final[h]["names"], "calibration": R["models"][h]["models"]["ensemble"]["calibration"],
                              "gate": R["models"][h]["models"]["ensemble"]["gate"],
                              "validation": {"oos_ic": R["models"][h]["models"]["ensemble"]["holdout"]["mean"],
                                             "oos_ic_se": R["models"][h]["models"]["ensemble"]["holdout"].get("se"),
                                             "net_sharpe": (R["models"][h]["models"]["ensemble"]["backtest_holdout"] or {}).get("net_sharpe_ann")}}
                          for h in C.HORIZONS},
             "deflated_sharpe_ratio": dsr, "tested_hypotheses": R["tested_hypotheses"], "trained_through": str(pd.Timestamp(dates.max()).date()),
             "current_regime": R["current_regime"], "order_size_usd": order_usd}
    (C.PROC / f"model_{pname}.json").write_text(json.dumps(model, indent=1, default=float))
    (C.REPORTS / f"research_results_{pname}.json").write_text(json.dumps(R, indent=1, default=str))
    with (C.LEDGER / "research_runs.jsonl").open("a") as f:
        f.write(json.dumps({"run_id": run_id, "profile": pname, "config_hash": chash, "at": R["generated_at_utc"], "holdout_start": R["sample"]["holdout_start"],
                            "tested_hypotheses": R["tested_hypotheses"], "note": "research v2 (fs-2.0, model suite); large-cap holdout previously viewed by the v1 run",
                            "gate": {h: R["models"][h]["models"]["ensemble"]["gate"]["status"] for h in C.HORIZONS},
                            "holdout_ic": {h: R["models"][h]["models"]["ensemble"]["holdout"]["mean"] for h in C.HORIZONS}}) + "\n")
    print(json.dumps({h: {"ens_wf_ic": R["models"][h]["models"]["ensemble"]["walk_forward"]["mean"], "ens_ho_ic": R["models"][h]["models"]["ensemble"]["holdout"]["mean"],
                          "gate": R["models"][h]["models"]["ensemble"]["gate"]} for h in C.HORIZONS}, indent=1, default=str))
    return R


if __name__ == "__main__":
    # Run the canonical module's factories: script-local classes labelled
    # 'research' are different objects and cannot be pickled under that name.
    import research as canonical
    _lock = canonical.C.hold_lock()
    canonical.run(sys.argv[1] if len(sys.argv) > 1 else None)
