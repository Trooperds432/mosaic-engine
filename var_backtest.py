"""VaR / expected-shortfall backtesting of competing volatility models (spec 10.2), adapted from the GARCH and EWMA
VaR notebooks with formal tests added.

Series: SPY, an equal-weight portfolio of the large-cap universe (rebalanced daily, point-in-time membership), and a
fixed random sample of individual large-cap stocks (to choose the per-name model used in signals).
Models (1-day ahead, forecasts use data through t-1 only):
  historical   250-day empirical quantile / tail mean
  ewma         RiskMetrics lambda=0.94, normal quantiles
  garch_t      GARCH(1,1) with Student-t errors on a rolling 1000-day window, parameters refit every 21 sessions,
               variance updated daily with the frozen parameters in between
  factor       portfolio variance b'F b + s^2: b = rolling 252-day loadings on the risk-model factor returns,
               F = EWMA factor covariance, s^2 = EWMA residual variance (portfolio series only)
Tests at 95% and 99%: Kupiec proportion-of-failures (unconditional coverage), Christoffersen independence (breach
clustering) and conditional coverage; ES at 97.5% by the McNeil-Frey exceedance-residual t-test.
Winner: the model that passes coverage and independence at both levels with the lowest quantile (tick) loss;
written to processed/vol_model.json for signals.py and the bot's sizing.
"""
import json
import math
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

import config as C

warnings.filterwarnings("ignore")
LEVELS = (0.95, 0.99)          # tested VaR levels
ES_LEVEL = 0.975               # expected-shortfall level (Basel FRTB)
ALL = LEVELS + (ES_LEVEL,)
START = "2010-01-01"


# ---------------------------------------------------------------- models: return (sigma or None, var dict, es)
def historical(r):
    q = {a: -r.rolling(250).quantile(1 - a).shift(1) for a in ALL}
    es = -r.rolling(250).apply(lambda x: x[x <= np.quantile(x, 1 - ES_LEVEL)].mean(), raw=True).shift(1)
    return q, es


def ewma(r, lam=0.94):
    v = (r ** 2).ewm(alpha=1 - lam, adjust=False).mean().shift(1)
    s = np.sqrt(v)
    q = {a: -stats.norm.ppf(1 - a) * s for a in ALL}
    z = stats.norm.ppf(1 - ES_LEVEL)
    es = s * stats.norm.pdf(z) / (1 - ES_LEVEL)
    return q, es, s


def garch_t(r, window=1000, refit=21):
    from arch import arch_model
    x = r.dropna() * 100
    sig = pd.Series(np.nan, index=x.index)
    nu = pd.Series(np.nan, index=x.index)
    params = None
    v = None
    for i in range(window, len(x)):
        if params is None or (i - window) % refit == 0:
            try:
                res = arch_model(x.iloc[i - window:i], vol="GARCH", p=1, q=1, dist="t", mean="Constant", rescale=False).fit(disp="off")
                params = res.params
                v = float(res.conditional_volatility.iloc[-1] ** 2)
                prev = float(x.iloc[i - 1] - params["mu"])
            except Exception:
                continue
        else:
            prev = float(x.iloc[i - 1] - params["mu"])
        v = params["omega"] + params["alpha[1]"] * prev ** 2 + params["beta[1]"] * v  # sigma^2 for day i from data through i-1
        sig.iat[i] = math.sqrt(v) / 100
        nu.iat[i] = params["nu"]
    mu = 0.0
    q, es = {}, pd.Series(np.nan, index=x.index)
    for a in ALL:
        tq = stats.t.ppf(1 - a, nu) * np.sqrt((nu - 2) / nu)  # standardised-t quantile
        q[a] = -(mu + sig * tq)
    tq = stats.t.ppf(1 - ES_LEVEL, nu)
    es = sig * np.sqrt((nu - 2) / nu) * (stats.t.pdf(tq, nu) / (1 - ES_LEVEL)) * ((nu + tq ** 2) / (nu - 1))
    return {a: q[a].reindex(r.index) for a in ALL}, es.reindex(r.index), sig.reindex(r.index)


def factor_model(r, F, lam=0.97):
    """Rolling 252-day loadings on factor returns; EWMA factor covariance and residual variance, all through t-1."""
    F = F.reindex(r.index)
    cols = ["market", "beta", "size", "value", "momentum", "quality", "volatility", "illiquidity"]
    X = F[cols]
    sig = pd.Series(np.nan, index=r.index)
    cov = None
    res_v = None
    for i in range(300, len(r)):
        win = slice(i - 252, i)
        y, Xw = r.iloc[win], X.iloc[win]
        ok = y.notna() & Xw.notna().all(axis=1)
        if ok.sum() < 200:
            continue
        A = np.column_stack([np.ones(ok.sum()), Xw[ok].to_numpy()])
        b, *_ = np.linalg.lstsq(A, y[ok].to_numpy(), rcond=None)
        e = y[ok].to_numpy() - A @ b
        if cov is None or i % 21 == 0:
            cov = Xw[ok].ewm(alpha=1 - lam).cov().iloc[-len(cols):].to_numpy()
            res_v = float(pd.Series(e).pow(2).ewm(alpha=1 - lam).mean().iloc[-1])
        sig.iat[i] = math.sqrt(max(b[1:] @ cov @ b[1:] + res_v, 1e-12))
    q = {a: -stats.norm.ppf(1 - a) * sig for a in ALL}
    z = stats.norm.ppf(1 - ES_LEVEL)
    return q, sig * stats.norm.pdf(z) / (1 - ES_LEVEL), sig


# ---------------------------------------------------------------- tests
def kupiec(hits, a):
    n, x = len(hits), int(hits.sum())
    p = 1 - a
    if n == 0:
        return None
    ph = x / n
    ll0 = (n - x) * math.log(1 - p) + x * math.log(p)
    ll1 = (n - x) * math.log(1 - ph) + x * math.log(ph) if 0 < x < n else 0.0
    lr = -2 * (ll0 - ll1)
    return {"breaches": x, "expected": n * p, "rate": ph, "lr": lr, "p": float(1 - stats.chi2.cdf(lr, 1))}


def christoffersen(hits):
    h = hits.astype(int).to_numpy()
    a, b = h[:-1], h[1:]
    n00, n01 = int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n10, n11 = int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum())
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def ll(pp, k0, k1):
        return (k0 * math.log(1 - pp) if k0 and pp < 1 else 0) + (k1 * math.log(pp) if k1 and pp > 0 else 0)
    lr = -2 * (ll(p, n00 + n10, n01 + n11) - ll(p01, n00, n01) - ll(p11, n10, n11))
    return {"lr": lr, "p": float(1 - stats.chi2.cdf(lr, 1)), "p_breach_after_breach": p11, "p_breach_after_no_breach": p01}


def tick_loss(r, q, a):
    """Quantile (pinball) loss of the VaR forecast; lower is better."""
    u = r + q  # return minus the (negative) quantile
    tau = 1 - a
    return float(np.mean(np.where(u < 0, (tau - 1) * u, tau * u)))


def es_test(r, var975, es, sig):
    d = pd.DataFrame({"r": r, "v": var975, "es": es, "s": sig}).dropna()
    d = d[d.index >= START]
    exc = d[-d["r"] > d["v"]]
    if len(exc) < 10:
        return None
    resid = ((-exc["r"]) - exc["es"]) / exc["s"]
    t = resid.mean() / resid.std() * math.sqrt(len(resid))
    return {"exceedances": int(len(exc)), "mean_std_residual": float(resid.mean()), "t": float(t), "p_one_sided": float(1 - stats.norm.cdf(t))}


def run_series(name, r, F=None):
    r = r.dropna()
    h_q, h_es = historical(r)
    h_s = r.rolling(250).std().shift(1)
    e_q, e_es, e_s = ewma(r)
    g_q, g_es, g_s = garch_t(r)
    models = {"historical": (h_q, h_es, h_s), "ewma": (e_q, e_es, e_s), "garch_t": (g_q, g_es, g_s)}
    if F is not None:
        f_q, f_es, f_s = factor_model(r, F)
        models["factor"] = (f_q, f_es, f_s)
    res = {}
    for m, (q, es, s) in models.items():
        rec = {}
        for a in LEVELS:
            d = pd.DataFrame({"r": r, "q": q[a]}).dropna()
            d = d[d.index >= START]
            hits = d["r"] < -d["q"]
            k, c = kupiec(hits, a), christoffersen(hits)
            rec[str(a)] = {"kupiec": k, "christoffersen": c, "cc_p": float(1 - stats.chi2.cdf(k["lr"] + c["lr"], 2)),
                           "tick_loss": tick_loss(d["r"].to_numpy(), d["q"].to_numpy(), a), "n": int(len(d))}
        rec["es_975"] = es_test(r, q[ES_LEVEL], es, s)
        rec["passes"] = all(rec[str(a)]["kupiec"]["p"] > 0.05 and rec[str(a)]["christoffersen"]["p"] > 0.05 for a in LEVELS)
        res[m] = rec
    return res


def pick(res_by_series):
    """Rank models: share of series where coverage+independence pass at both levels, then mean 99% tick loss."""
    score = {}
    for m in {m for r in res_by_series.values() for m in r}:
        recs = [r[m] for r in res_by_series.values() if m in r]
        score[m] = (np.mean([x["passes"] for x in recs]), -np.mean([x["0.99"]["tick_loss"] for x in recs]))
    return max(score, key=score.get), score


def run():
    spy = pd.read_parquet(C.RAW / "prices" / f"{C.BENCHMARK}.parquet").drop_duplicates("date").set_index("date")["adjclose"]
    r_spy = spy.pct_change()
    F = pd.read_parquet(C.PROC / "factor_returns_large.parquet")
    panel = pd.read_parquet(C.PROC / "panel_large.parquet", columns=["date", "symbol"])
    members = panel.groupby("date")["symbol"].apply(set)
    syms = sorted(panel["symbol"].unique())
    px = {}
    for s in syms:
        f = C.RAW / "prices" / f"{s}.parquet"
        if f.exists():
            px[s] = pd.read_parquet(f, columns=["date", "adjclose"]).drop_duplicates("date").set_index("date")["adjclose"]
    R = pd.DataFrame(px).reindex(spy.index).pct_change(fill_method=None)
    # equal-weight portfolio of names eligible at the latest decision date on or before each day (point in time)
    mem = members.reindex(R.index, method="ffill")
    ew = pd.Series([R.loc[d, list(m & set(R.columns))].mean() if isinstance(m, set) else np.nan for d, m in mem.items()], index=R.index)
    out = {"generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "evaluation_start": START,
           "levels": list(LEVELS), "es_level": ES_LEVEL, "series": {}}
    out["series"]["SPY"] = run_series("SPY", r_spy, F)
    print("SPY", {m: (v["passes"], round(v["0.99"]["kupiec"]["rate"], 4), round(v["0.99"]["christoffersen"]["p"], 3)) for m, v in out["series"]["SPY"].items()}, flush=True)
    out["series"]["EW_large_cap_portfolio"] = run_series("EW", ew, F)
    print("EW", {m: (v["passes"], round(v["0.99"]["kupiec"]["rate"], 4)) for m, v in out["series"]["EW_large_cap_portfolio"].items()}, flush=True)
    rng = np.random.default_rng(7)
    live = [s for s in members.iloc[-1] if s in R.columns and R[s].notna().sum() > 2500]
    sample = sorted(rng.choice(live, size=min(40, len(live)), replace=False))
    stocks = {}
    for s in sample:
        stocks[s] = run_series(s, R[s])
    out["stock_sample"] = {"symbols": sample, "pass_rate": {m: float(np.mean([v[m]["passes"] for v in stocks.values()])) for m in ["historical", "ewma", "garch_t"]},
                           "mean_tick_loss_99": {m: float(np.mean([v[m]["0.99"]["tick_loss"] for v in stocks.values()])) for m in ["historical", "ewma", "garch_t"]},
                           "breach_rate_99": {m: float(np.mean([v[m]["0.99"]["kupiec"]["rate"] for v in stocks.values()])) for m in ["historical", "ewma", "garch_t"]}}
    print("stocks", out["stock_sample"], flush=True)
    win_port, sc_port = pick({k: v for k, v in out["series"].items()})
    win_name, sc_name = pick(stocks)
    out["winner"] = {"portfolio": win_port, "single_name": win_name, "scores_portfolio": {k: list(v) for k, v in sc_port.items()},
                     "scores_single_name": {k: list(v) for k, v in sc_name.items()}}
    (C.REPORTS / "var_backtest.json").write_text(json.dumps(out, indent=1, default=float))
    (C.PROC / "vol_model.json").write_text(json.dumps({"single_name": win_name, "portfolio": win_port, "chosen_at": out["generated_at_utc"],
                                                         "source": "reports/var_backtest.json"}, indent=1))
    print("winner:", out["winner"], flush=True)
    return out




# ---------------------------------------------------------------- nightly per-name forecasts with the winning model
def forecast_names(symbols, window=1000):
    """1-day volatility, 99% VaR and 97.5% ES per symbol from the model chosen by the backtest (GARCH-t or a fallback)."""
    from arch import arch_model
    choice = json.loads((C.PROC / "vol_model.json").read_text())["single_name"] if (C.PROC / "vol_model.json").exists() else "ewma"
    rows = []
    for sym in symbols:
        f = C.RAW / "prices" / f"{sym}.parquet"
        if not f.exists():
            continue
        r = pd.read_parquet(f, columns=["date", "adjclose"]).drop_duplicates("date").set_index("date")["adjclose"].pct_change().dropna().iloc[-window:]
        if len(r) < 250:
            continue
        rec = {"symbol": sym, "model": choice, "as_of": str(r.index[-1].date())}
        try:
            if choice == "garch_t" and len(r) >= 500:
                res = arch_model(r * 100, vol="GARCH", p=1, q=1, dist="t", mean="Constant", rescale=False).fit(disp="off")
                s = math.sqrt(res.forecast(horizon=1).variance.iloc[-1, 0]) / 100
                nu = float(res.params["nu"])
                k = math.sqrt((nu - 2) / nu)
                var99 = -s * stats.t.ppf(0.01, nu) * k
                tq = stats.t.ppf(1 - ES_LEVEL, nu)
                es = s * k * stats.t.pdf(tq, nu) / (1 - ES_LEVEL) * (nu + tq ** 2) / (nu - 1)
                ew = float(np.sqrt((r ** 2).ewm(alpha=0.06, adjust=False).mean().iloc[-1]))
                if nu < 2.5 or s > 3 * ew or s > 0.2:  # unstable fit (degenerate tails or bad prints): use EWMA instead
                    raise ValueError("unstable GARCH fit")
                rec.update(sigma_1d=s, nu=nu, var99_1d=var99, es975_1d=es)
            else:
                raise ValueError("fallback")
        except Exception:
            s = float(np.sqrt((r ** 2).ewm(alpha=0.06, adjust=False).mean().iloc[-1]))
            rec.update(model="ewma_fallback", sigma_1d=s, nu=None, var99_1d=2.326 * s, es975_1d=s * stats.norm.pdf(stats.norm.ppf(0.025)) / 0.025)
        rows.append(rec)
    df = pd.DataFrame(rows)
    tmp = C.PROC / "vol_forecasts.tmp"
    df.to_parquet(tmp, index=False)
    tmp.replace(C.PROC / "vol_forecasts.parquet")
    return df


if __name__ == "__main__":
    _lock = __import__("config").hold_lock()
    if "--forecast" in sys.argv:  # nightly: forecasts for every symbol the engine scores tonight
        fc = pd.read_parquet(C.REPORTS / "forecast_latest.parquet", columns=["symbol"]) if (C.REPORTS / "forecast_latest.parquet").exists() else None
        syms = sorted(set(fc["symbol"])) if fc is not None else []
        out = forecast_names(syms)
        print(f"vol forecasts: {len(out)} symbols, models {out['model'].value_counts().to_dict()}")
    else:
        run()
