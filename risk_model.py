"""Multifactor risk model, volatility forecasts, stress tests and a constrained optimiser (spec section 10).

  r_t = B_t f_t + e_t   daily cross-sectional WLS (weights sqrt mcap) of stock returns on standardised style
                        exposures (beta, size, value, momentum, quality, low-vol, liquidity) + sector dummies.
                        Exposures come from the PIT panel (decision dates) carried forward, so B_t is known at t.
  Sigma = B F B' + D    F: scikit-learn Ledoit-Wolf factor covariance over the last 756 observations;
                        D: EWMA residual variance (half-life 63d).
Per-name vol: EWMA (lambda .94) and GARCH(1,1) (arch, Student-t) where 500+ observations exist.
Stress: historical replay of 2008-09/10, 2020-02/03, 2022-H1, 2024-08 using actual returns (factor-model proxy
when a name lacks history) plus factor-shock scenarios; correlation spike = mean pairwise correlation in the
window vs the prior year.
Optimiser (cvxpy): max w'mu - (lambda/2) w'Sigma w - c'|w - w0|, long-only, per-name cap = bot POSITION_PCT,
sector cap, turnover cap, sum(w) <= 1 (no leverage). Only run when mu comes from a validated model; otherwise
the proposal is NO_SIGNAL. Proposals are research output, never orders.
"""
import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config as C
from features import load_prices, sic_sector

STYLES = ["beta_252", "log_mcap", "book_to_mkt", "mom_12_1", "roa", "vol_60", "amihud"]
STYLE_NAMES = {"beta_252": "beta", "log_mcap": "size", "book_to_mkt": "value", "mom_12_1": "momentum", "roa": "quality",
               "vol_60": "volatility", "amihud": "illiquidity"}
SCENARIOS = {"gfc_2008": ("2008-09-12", "2008-10-27"), "covid_2020": ("2020-02-19", "2020-03-23"),
             "rates_2022": ("2022-01-03", "2022-06-16"), "carry_unwind_2024": ("2024-07-16", "2024-08-07")}


def _z(x):
    x = x.clip(x.quantile(0.01), x.quantile(0.99))
    return (x - x.mean()) / x.std() if x.std() > 0 else x * 0


def exposures(panel):
    """Standardised exposures per decision date (cross-sectional z-scores; missing -> 0)."""
    E = panel[["date", "symbol", "sector", "mcap"]].copy()
    for s in STYLES:
        E[STYLE_NAMES[s]] = panel.groupby("date")[s].transform(_z).fillna(0).astype("float32")
    return E


def factor_returns(P, E):
    cal = P["adjclose"].index
    r = P["adjclose"].pct_change(fill_method=None)
    dates = np.sort(E["date"].unique())
    sectors = sorted(E["sector"].dropna().unique())
    fnames = ["market"] + list(STYLE_NAMES.values()) + [f"sec_{s}" for s in sectors[1:]]
    F, resid_sq = [], {}
    by_date = {d: g for d, g in E.groupby("date")}
    cur = None
    for i, d in enumerate(cal):
        if d in by_date:
            cur = by_date[d]
        if cur is None or i == 0:
            continue
        g = cur
        y = r.loc[d, g["symbol"]].to_numpy(dtype="float64")
        ok = np.isfinite(y) & np.isfinite(g["mcap"].to_numpy())
        if ok.sum() < 50:
            continue
        X = np.column_stack([np.ones(ok.sum())] + [g[c].to_numpy()[ok] for c in STYLE_NAMES.values()] +
                            [(g["sector"].to_numpy()[ok] == s).astype(float) for s in sectors[1:]])
        w = np.sqrt(np.clip(g["mcap"].to_numpy()[ok], 1e8, None))
        yy = np.clip(y[ok], -0.5, 0.5)
        beta, *_ = np.linalg.lstsq(X * w[:, None], yy * w, rcond=None)
        F.append(pd.Series(beta, index=fnames, name=d))
        if i >= len(cal) - 504:  # specific risk only needs the last two years
            e = yy - X @ beta
            for s, v in zip(g["symbol"].to_numpy()[ok], e):
                resid_sq.setdefault(s, []).append((d, v * v))
    Fdf = pd.DataFrame(F)
    return Fdf, resid_sq, fnames


def ewma_cov(X, halflife):
    lam = 0.5 ** (1 / halflife)
    Xc = X - X.mean()
    w = lam ** np.arange(len(X))[::-1]
    w = w / w.sum()
    S = (Xc.to_numpy() * w[:, None]).T @ Xc.to_numpy()
    # shrink toward the diagonal (intensity from the ratio of off-diagonal noise; capped)
    D = np.diag(np.diag(S))
    n_eff = 1 / (w ** 2).sum()
    k = min(1.0, X.shape[1] / n_eff * 5)
    return pd.DataFrame((1 - k) * S + k * D, index=X.columns, columns=X.columns), k


def ledoit_wolf_cov(X):
    """Actual data-estimated shrinkage; never substitute zero for missing returns."""
    from sklearn.covariance import LedoitWolf
    if not X.columns.is_unique:
        raise ValueError("factor names must be unique")
    data = X.replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 2 or not len(data.columns):
        raise ValueError("at least two complete factor-return observations required")
    fit = LedoitWolf().fit(data.to_numpy(dtype=float))
    return pd.DataFrame(fit.covariance_, index=X.columns, columns=X.columns), float(fit.shrinkage_)


def garch_vol(ret):
    from arch import arch_model
    x = ret.dropna().iloc[-1500:] * 100
    if len(x) < 500:
        return None
    try:
        res = arch_model(x, vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(disp="off")
        return float(math.sqrt(res.forecast(horizon=1).variance.iloc[-1, 0]) / 100 * math.sqrt(252))
    except Exception:
        return None


def build(profile=None):
    prof = C.PROFILES[profile] if profile else C.PROFILE
    panel = pd.read_parquet(C.PROC / f"panel_{prof['name']}.parquet", columns=["date", "symbol", "sector", "mcap"] + STYLES)
    E = exposures(panel)
    P, _ = load_prices(prof["min_dollar_adv"], symbols=set(E["symbol"]))
    for k in ["high", "low", "volume", "adj_open", "close"]:
        P.pop(k, None)
    Fdf, resid_sq, fnames = factor_returns(P, E)
    Fcov, shrink = ledoit_wolf_cov(Fdf.iloc[-756:])
    lam = 0.5 ** (1 / 63)
    spec = {}
    for s, v in resid_sq.items():
        v = v[-504:]
        w = lam ** np.arange(len(v))[::-1]
        spec[s] = float(np.sqrt(np.dot(w, [x for _, x in v]) / w.sum() * 252)) if len(v) >= 60 else None
    last = E[E["date"] == E["date"].max()].set_index("symbol")
    r = P["adjclose"].pct_change(fill_method=None)
    ewma = {s: float(np.sqrt((r[s].dropna().iloc[-500:] ** 2).ewm(alpha=0.06).mean().iloc[-1] * 252)) for s in last.index if r[s].notna().sum() > 60}
    Fdf.to_parquet(C.PROC / f"factor_returns_{prof['name']}.parquet")
    model = {"profile": prof["name"], "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "factors": fnames,
             "factor_cov_annual": (Fcov * 252).to_dict(), "shrinkage_intensity": shrink, "specific_vol_annual": spec,
             "covariance_estimator": "sklearn.covariance.LedoitWolf", "covariance_window_observations": 756,
             "covariance_complete_observations": int(Fdf.iloc[-756:].replace([np.inf, -np.inf], np.nan).dropna().shape[0]),
             "ewma_vol_annual": ewma, "exposures_asof": str(E["date"].max().date()),
             "exposures": last[["sector"] + list(STYLE_NAMES.values())].reset_index().to_dict(orient="records"),
             "factor_return_stats": {f: {"ann_mean": float(Fdf[f].mean() * 252), "ann_vol": float(Fdf[f].std() * math.sqrt(252))} for f in fnames}}
    (C.PROC / f"risk_model_{prof['name']}.json").write_text(json.dumps(model, default=float))
    print("factor model:", len(Fdf), "days,", len(fnames), "factors, shrinkage", round(shrink, 3))
    return model


class RiskModel:
    def __init__(self, profile=None):
        prof = C.PROFILES[profile] if profile else C.PROFILE
        self.m = json.loads((C.PROC / f"risk_model_{prof['name']}.json").read_text())
        self.F = pd.DataFrame(self.m["factor_cov_annual"])
        self.f = self.m["factors"]
        self.X = pd.DataFrame(self.m["exposures"]).set_index("symbol")
        self.prof = prof

    def B(self, syms):
        X = self.X.reindex(syms)
        cols = {"market": np.ones(len(syms))}
        for s in STYLE_NAMES.values():
            cols[s] = X[s].fillna(0).to_numpy()
        for f in self.f:
            if f.startswith("sec_"):
                cols[f] = (X["sector"] == f[4:]).astype(float).to_numpy()
        return pd.DataFrame(cols, index=syms)[self.f]

    def sigma(self, syms):
        B = self.B(syms)
        d = np.array([(self.m["specific_vol_annual"].get(s) or 0.35) ** 2 for s in syms])
        return B.to_numpy() @ self.F.loc[self.f, self.f].to_numpy() @ B.to_numpy().T + np.diag(d), B

    def decompose(self, w):
        """w: Series of portfolio weights (fraction of capital)."""
        syms = list(w.index)
        S, B = self.sigma(syms)
        wv = w.to_numpy()
        var = float(wv @ S @ wv)
        vol = math.sqrt(var)
        mrc = S @ wv / vol
        crc = wv * mrc
        expo = pd.Series(B.to_numpy().T @ wv, index=self.f)
        fvar = float(expo.to_numpy() @ self.F.loc[self.f, self.f].to_numpy() @ expo.to_numpy())
        sec = self.X.reindex(syms)["sector"].fillna("Unknown")
        return {"vol_annual": vol, "factor_share_of_var": fvar / var if var else None,
                "exposures": {k: float(v) for k, v in expo.items() if not k.startswith("sec_")},
                "sector_weights": w.groupby(sec.to_numpy()).sum().to_dict(),
                "component_risk": dict(zip(syms, (crc / vol).round(4).tolist())) if vol else {},
                "marginal_risk": dict(zip(syms, mrc.round(4).tolist()))}


def stress(w, P, rm):
    cal = P["adjclose"].index
    out = {}
    Fdf = pd.read_parquet(C.PROC / f"factor_returns_{rm.prof['name']}.parquet")
    B = rm.B(list(w.index))
    for name, (a, b) in SCENARIOS.items():
        ia, ib = cal.searchsorted(pd.Timestamp(a)), cal.searchsorted(pd.Timestamp(b))
        if ib <= ia:
            continue
        pnl, proxy = 0.0, []
        fr = Fdf.loc[(Fdf.index >= cal[ia]) & (Fdf.index <= cal[ib])]
        fcum = (1 + fr).prod() - 1
        for s, wt in w.items():
            px = P["adjclose"][s] if s in P["adjclose"] else None
            if px is not None and pd.notna(px.iat[ia]) and pd.notna(px.iat[ib]):
                pnl += wt * (px.iat[ib] / px.iat[ia] - 1)
            else:
                pnl += wt * float(B.loc[s].to_numpy() @ fcum.reindex(B.columns).fillna(0).to_numpy())
                proxy.append(s)
        win = P["adjclose"][list(w.index)].iloc[ia:ib + 1].pct_change(fill_method=None)
        pre = P["adjclose"][list(w.index)].iloc[max(0, ia - 252):ia].pct_change(fill_method=None)

        def mpc(x):
            c = x.corr().to_numpy()
            iu = np.triu_indices_from(c, 1)
            v = c[iu]
            v = v[np.isfinite(v)]
            return float(v.mean()) if len(v) else None
        out[name] = {"window": [a, b], "portfolio_return": pnl, "proxied_names": proxy,
                     "spy_return": float(P["spy_adjclose"].iat[ib] / P["spy_adjclose"].iat[ia] - 1),
                     "avg_pairwise_corr_window": mpc(win) if len(w) > 1 else None, "avg_pairwise_corr_prior_year": mpc(pre) if len(w) > 1 else None}
    fs = rm.F.loc[rm.f, rm.f]
    expo = pd.Series(rm.B(list(w.index)).to_numpy().T @ w.to_numpy(), index=rm.f)
    for f, k in [("market", -3), ("momentum", -3), ("size", -3), ("value", 3), ("volatility", 3)]:
        sd = math.sqrt(fs.loc[f, f] / 252 * 21)  # one-month shock of k standard deviations
        out[f"factor_shock_{f}_{k:+d}sd_1m"] = {"portfolio_return": float(expo[f] * k * sd)}
    return out


def tail_and_liquidity(w, P, order_usd):
    r = P["adjclose"][list(w.index)].pct_change(fill_method=None).iloc[-504:]
    port = (r.fillna(0) * w.to_numpy()).sum(axis=1)
    q = port.quantile(0.025)
    gap = (P["adj_open"][list(w.index)] / P["adjclose"][list(w.index)].shift(1) - 1).iloc[-756:] if "adj_open" in P else None
    adv = (P["close"] * P["volume"]).rolling(60).mean().iloc[-1] if "close" in P and "volume" in P else None
    names = {}
    for s in w.index:
        names[s] = {"gap_q01": float(gap[s].quantile(0.01)) if gap is not None and gap[s].notna().sum() > 100 else None,
                    "order_pct_adv": float(order_usd / adv[s]) if adv is not None and adv[s] > 0 else None}
    return {"es_97_5_daily": float(port[port <= q].mean()), "var_97_5_daily": float(q),
            "max_drawdown_2y": float(((1 + port).cumprod() / (1 + port).cumprod().cummax() - 1).min()), "names": names}


def optimise(mu, rm, w0, limits, cost=None, constraints=None, betas=None, adv_dollars=None):
    """Research proposal under explicit external limits; includes liquidation of non-candidates.

    Alpha/covariance are annualised. Transaction costs are converted to annual
    units using the externally supplied planning horizon. No limits are guessed.
    """
    import cvxpy as cp
    from portfolio_constraints import validate
    constraints = validate(constraints, limits)
    syms = list(mu.index.union(w0.index, sort=False))
    if not syms:
        return pd.Series(dtype=float)
    if not mu.index.is_unique or not w0.index.is_unique:
        raise ValueError("duplicate portfolio symbols")
    if betas is None or adv_dollars is None:
        raise ValueError("per-name raw beta and dollar ADV required")
    beta = betas.reindex(syms).to_numpy(dtype=float)
    adv = adv_dollars.reindex(syms).to_numpy(dtype=float)
    equity = limits.get("account_total_usd")
    cap = limits.get("position_pct")
    if (not np.isfinite(beta).all() or not np.isfinite(adv).all() or (adv <= 0).any()
            or equity is None or not math.isfinite(equity) or equity <= 0
            or cap is None or not math.isfinite(cap) or not 0 < cap <= 1):
        raise ValueError("finite beta, positive ADV/equity and imported position cap required")
    alpha = mu.reindex(syms, fill_value=0).to_numpy(dtype=float)
    w0v = w0.reindex(syms, fill_value=0).to_numpy(dtype=float)
    if not np.isfinite(alpha).all() or not np.isfinite(w0v).all() or (w0v < 0).any():
        raise ValueError("finite alpha and nonnegative current holdings required")
    if cost is None:
        raise ValueError("per-name transaction costs required")
    c = cost.reindex(syms).to_numpy(dtype=float)
    if not np.isfinite(c).all() or (c < 0).any():
        raise ValueError("finite nonnegative costs required for candidates and exits")
    S, _ = rm.sigma(syms)
    if not np.isfinite(S).all():
        raise ValueError("finite risk covariance required")
    S = (S + S.T) / 2 + 1e-8 * np.eye(len(syms))
    w = cp.Variable(len(syms))
    annual_cost = c * 252 / constraints["cost_horizon_sessions"]
    obj = alpha @ w - constraints["risk_aversion"] / 2 * cp.quad_form(w, cp.psd_wrap(S)) - annual_cost @ cp.abs(w - w0v)
    cons = [w >= 0, w <= cap, cp.sum(w) <= constraints["gross_limit"],
            cp.sum(cp.abs(w - w0v)) <= constraints["turnover_cap"],
            beta @ w >= constraints["beta_min"], beta @ w <= constraints["beta_max"],
            cp.abs(w - w0v) * equity <= constraints["max_participation"] * adv]
    candidates = set(mu.index)
    for i, sym in enumerate(syms):
        if sym not in candidates:
            cons.append(w[i] == 0)
    sec = rm.X.reindex(syms)["sector"]
    if sec.isna().any():
        raise ValueError("sector coverage required for all candidates and holdings")
    for sector in set(sec):
        cons.append(cp.sum(w[(sec == sector).to_numpy()]) <= constraints["sector_cap"])
    prob = cp.Problem(cp.Maximize(obj), cons)
    prob.solve(solver="CLARABEL")
    if prob.status != cp.OPTIMAL or w.value is None:
        return None
    # Check solver tolerances before export; retain full precision so rounding
    # cannot increase a weight through a cap or erase liquidation turnover.
    if any(np.max(np.asarray(cn.violation())) > 1e-6 for cn in cons):
        return None
    return pd.Series(np.maximum(w.value, 0), index=syms)


if __name__ == "__main__":
    _lock = __import__("config").hold_lock()
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else None)


# ---------------------------------------------------------------- t-copula tail dependence (Copulas notebook, extended)
def t_copula(x, y, dfs=(2, 3, 4, 6, 8, 12, 20, 30)):
    """Bivariate Student-t copula on pseudo-observations: rho from Kendall's tau, nu by maximum likelihood.
    Returns rho, nu, lower-tail dependence lambda, and AIC vs the Gaussian copula (lambda = 0)."""
    from scipy import stats as st
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 250:
        return None
    n = len(d)
    u = d["x"].rank().to_numpy() / (n + 1)
    v = d["y"].rank().to_numpy() / (n + 1)
    tau = st.kendalltau(d["x"], d["y"]).statistic
    rho = float(np.sin(np.pi * tau / 2))
    S = np.array([[1, rho], [rho, 1]])
    best = None
    for nu in dfs:
        a, b = st.t.ppf(u, nu), st.t.ppf(v, nu)
        ll = float(np.sum(st.multivariate_t(loc=[0, 0], shape=S, df=nu).logpdf(np.column_stack([a, b])) - st.t.logpdf(a, nu) - st.t.logpdf(b, nu)))
        if best is None or ll > best[1]:
            best = (nu, ll)
    za, zb = st.norm.ppf(u), st.norm.ppf(v)
    ll_g = float(np.sum(st.multivariate_normal(mean=[0, 0], cov=S).logpdf(np.column_stack([za, zb])) - st.norm.logpdf(za) - st.norm.logpdf(zb)))
    nu = best[0]
    lam = float(2 * st.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1))
    return {"rho": rho, "nu": float(nu), "tail_dependence": lam, "aic_t": 2 * 2 - 2 * best[1], "aic_gaussian": 2 * 1 - 2 * ll_g,
            "t_preferred": bool(2 * 2 - 2 * best[1] < 2 * 1 - 2 * ll_g), "obs": int(n)}


def tail_dependence_report(symbols, P, years=3):
    """Each name's lower-tail dependence with SPY and the basket's average pairwise tail dependence."""
    r = P["adjclose"][symbols].pct_change(fill_method=None).iloc[-252 * years:]
    m = P["spy_adjclose"].pct_change().iloc[-252 * years:]
    vs_mkt = {s: t_copula(r[s], m) for s in symbols}
    pairs = []
    for i, a in enumerate(symbols):
        for b in symbols[i + 1:]:
            res = t_copula(r[a], r[b])
            if res:
                pairs.append(res["tail_dependence"])
    vm = {k: v for k, v in vs_mkt.items() if v}
    return {"window_years": years, "vs_spy": vm,
            "mean_tail_dependence_vs_spy": float(np.mean([v["tail_dependence"] for v in vm.values()])) if vm else None,
            "share_t_preferred_over_gaussian": float(np.mean([v["t_preferred"] for v in vm.values()])) if vm else None,
            "mean_pairwise_tail_dependence": float(np.mean(pairs)) if pairs else None, "pairs": len(pairs)}
