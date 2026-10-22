"""Commodity and energy research module (spec 4C / 4E statistical models). Research output only: the bot trades
equities, so nothing here becomes a signal directly.

Data: Yahoo continuous front-month futures (roll gaps included; term structure is NOT observable from this
source), CFTC disaggregated COT (+3d availability), EIA weekly crude stocks (+5d), FRED NY Harbor product prices.
Discovery sample ends at the research holdout start; the holdout is reported separately.

Tests per commodity (monthly returns, HAC standard errors):
  seasonality      calendar-month dummies: joint Wald test, discovery vs holdout sign agreement
  ts_momentum      12-1 month return sign -> next-month return (Moskowitz-Ooi-Pedersen)
  cot_crowding     managed-money net % OI 3y z-score -> next 4-week return (contrarian hypothesis)
Energy-specific:
  inventory        EIA crude stocks vs 5-year same-week mean (z) -> WTI return over the following week
  crack_coint      Engle-Granger and Johansen tests for WTI, RBOB-equivalent gasoline and heating oil (monthly logs);
                   error-correction half-life when cointegrated, with a pre/post-2016 stability split
Equity separation: energy/materials stock returns regressed on SPY and the matching commodity (monthly): the
commodity loading is the operational exposure, the SPY loading the equity-market beta.
"""
import json
import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

import config as C
from features import load_prices, sector_flags

HOLDOUT = None


def hac(y, X, lags=3):
    res = sm.OLS(y, sm.add_constant(X, has_constant="add"), missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return res


def monthly(fut):
    return {c: g.set_index("date")["close"].resample("ME").last() for c, g in fut.groupby("commodity")}


def run():
    global HOLDOUT
    spy = pd.read_parquet(C.RAW / "prices" / f"{C.BENCHMARK}.parquet").drop_duplicates("date").set_index("date")
    HOLDOUT = spy.index[-1] - pd.DateOffset(years=C.HOLDOUT_YEARS)
    fut = pd.read_parquet(C.RAW / "alt" / "futures.parquet")
    fut = fut[fut["close"] > 0]
    M = monthly(fut)
    out = {"holdout_start": str(HOLDOUT.date()), "commodities": {}}
    cot = pd.read_parquet(C.RAW / "alt" / "cot.parquet")
    for c, px in M.items():
        r = np.log(px).diff().dropna()
        r = r[r.abs() < 0.5]  # drop roll/expiry artefacts larger than 50% in a month
        rec = {"months": int(len(r)), "first": str(r.index[0].date())}
        disc, ho = r[r.index < HOLDOUT], r[r.index >= HOLDOUT]
        D = pd.get_dummies(disc.index.month, prefix="m", drop_first=True, dtype=float).set_index(disc.index)
        res = sm.OLS(disc.to_numpy(), sm.add_constant(D).to_numpy()).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        k = D.shape[1]
        R = np.hstack([np.zeros((k, 1)), np.eye(k)])
        w = res.wald_test(R, scalar=True)
        mm_d = disc.groupby(disc.index.month).mean()
        mm_h = ho.groupby(ho.index.month).mean()
        both = mm_d.index.intersection(mm_h.index)
        rec["seasonality"] = {"wald_p_discovery": float(w.pvalue), "month_means_discovery": {int(m): float(v) for m, v in mm_d.items()},
                              "holdout_sign_agreement": float((np.sign(mm_d[both]) == np.sign(mm_h[both])).mean()) if len(both) else None}
        mom = np.log(px).diff(12).shift(1) - np.log(px).diff(1).shift(1)
        dfm = pd.DataFrame({"y": r, "x": np.sign(mom.reindex(r.index))}).dropna()
        for lab, sub in [("discovery", dfm[dfm.index < HOLDOUT]), ("holdout", dfm[dfm.index >= HOLDOUT])]:
            if len(sub) > 12:
                m = hac(sub["y"], sub["x"])
                rec.setdefault("ts_momentum", {})[lab] = {"coef_monthly": float(m.params.iloc[1]), "t": float(m.tvalues.iloc[1]), "n": int(len(sub))}
        cc = cot[cot["commodity"] == c].sort_values("report_date")
        if len(cc) > 200:
            z = (cc["mm_net_pct_oi"] - cc["mm_net_pct_oi"].rolling(156, min_periods=52).mean()) / cc["mm_net_pct_oi"].rolling(156, min_periods=52).std()
            daily = fut[fut["commodity"] == c].set_index("date")["close"]
            z.index = cc["available_date"]
            zz = z.dropna()
            fwd = []
            for d in zz.index:
                i = daily.index.searchsorted(d)
                fwd.append(math.log(daily.iloc[min(i + 20, len(daily) - 1)] / daily.iloc[i]) if i < len(daily) - 1 else np.nan)
            dz = pd.DataFrame({"z": zz.to_numpy(), "y": fwd}, index=zz.index).dropna()
            dz = dz[dz["y"].abs() < 0.5]
            for lab, sub in [("discovery", dz[dz.index < HOLDOUT]), ("holdout", dz[dz.index >= HOLDOUT])]:
                if len(sub) > 30:
                    m = hac(sub["y"], sub["z"], lags=4)
                    rec.setdefault("cot_crowding", {})[lab] = {"coef_per_z_4w": float(m.params.iloc[1]), "t": float(m.tvalues.iloc[1]), "n": int(len(sub)),
                                                                 "hypothesis": "negative (crowded longs underperform)"}
        out["commodities"][c] = rec

    # EIA inventory surprise -> next-week WTI
    mp = pd.read_parquet(C.PROC / "macro_pit.parquet")
    cr = mp[mp["series"] == "crude_stocks_ex_spr"].sort_values("obs_date").copy()
    cr["woy"] = cr["obs_date"].dt.isocalendar().week.astype(int)
    cr["z"] = [(v - h.mean()) / h.std() if len(h) >= 4 and h.std() > 0 else np.nan
               for v, h in ((r.value, cr[(cr["woy"] == r.woy) & (cr["obs_date"] < r.obs_date) & (cr["obs_date"] >= r.obs_date - pd.Timedelta(days=1830))]["value"]) for r in cr.itertuples())]
    cr["dz"] = cr["z"].diff()
    wti = fut[fut["commodity"] == "wti"].set_index("date")["close"]
    ys = []
    for d in cr["available_date"]:
        i = wti.index.searchsorted(d)
        ys.append(math.log(wti.iloc[i + 5] / wti.iloc[i]) if 0 < i < len(wti) - 5 and wti.iloc[i] > 0 and wti.iloc[i + 5] > 0 else np.nan)
    cr["y"] = ys
    ci = cr.dropna(subset=["dz", "y"])
    ci = ci[(ci["available_date"] >= "2000-01-01") & (ci["y"].abs() < 0.4)]
    inv = {}
    for lab, sub in [("discovery", ci[ci["available_date"] < HOLDOUT]), ("holdout", ci[ci["available_date"] >= HOLDOUT])]:
        m = hac(sub["y"], sub["dz"], lags=2)
        inv[lab] = {"coef_per_z_change_1w": float(m.params.iloc[1]), "t": float(m.tvalues.iloc[1]), "n": int(len(sub))}
    out["eia_inventory_surprise"] = inv | {"hypothesis": "negative (unexpected builds lower crude)", "measure": "weekly change in 5y same-week z of crude stocks ex SPR"}

    # cointegration of crude and products (monthly logs, $/bbl)
    from data_macro import daily
    cal = spy.index
    mac = daily(cal)
    lv = pd.DataFrame({"wti": np.log(mac["DCOILWTICO"].where(mac["DCOILWTICO"] > 0)), "gasoline": np.log(mac["DGASNYH"] * 42),
                       "heating_oil": np.log(mac["DHOILNYH"] * 42)}).resample("ME").last().dropna()
    co = {}
    for lab, sub in [("full", lv), ("pre_2016", lv[lv.index < "2016-01-01"]), ("post_2016", lv[lv.index >= "2016-01-01"])]:
        eg = {p: float(coint(sub["wti"], sub[p])[1]) for p in ["gasoline", "heating_oil"]}
        jo = coint_johansen(sub.to_numpy(), det_order=0, k_ar_diff=2)
        rank = int(sum(jo.lr1 > jo.cvt[:, 1]))
        spread = sub["gasoline"] - sub["wti"]
        dl = spread.diff().dropna()
        lag = spread.shift(1).loc[dl.index]
        b = sm.OLS(dl.to_numpy(), sm.add_constant(lag.to_numpy())).fit().params[1]
        co[lab] = {"engle_granger_p": eg, "johansen_rank_95": rank, "gasoline_crack_ecm_coef": float(b),
                   "half_life_months": float(math.log(0.5) / math.log(1 + b)) if -1 < b < 0 else None, "months": int(len(sub))}
    out["crude_products_cointegration"] = co

    # equity vs commodity exposure separation (monthly)
    P, sym2cik = load_prices(C.PROFILES["broad"]["min_dollar_adv"])
    sic = pd.read_parquet(C.PROC / "sic.parquet").set_index("cik")["sic"]
    mr = P["adjclose"].resample("ME").last().pct_change(fill_method=None).iloc[-120:]
    ms = P["spy_adjclose"].resample("ME").last().pct_change().iloc[-120:]
    cm = {c: M[c].pct_change().reindex(mr.index) for c in ["wti", "copper", "gold", "corn", "natgas"] if c in M}
    sep = {}
    for grp, flag, comm in [("energy", "energy", "wti"), ("materials_metals", "commodity", "copper")]:
        names = [s for s, c in sym2cik.items() if sector_flags(sic.get(c))[flag]]
        loads = []
        for s in names:
            d = pd.DataFrame({"y": mr[s], "spy": ms, "cmd": cm[comm]}).dropna()
            if len(d) >= 60:
                m = sm.OLS(d["y"], sm.add_constant(d[["spy", "cmd"]])).fit()
                loads.append((m.params["spy"], m.params["cmd"], m.rsquared))
        if loads:
            a = np.array(loads)
            sep[grp] = {"names": len(loads), "commodity": comm, "median_equity_beta": float(np.median(a[:, 0])), "median_commodity_loading": float(np.median(a[:, 1])),
                        "median_r2": float(np.median(a[:, 2])), "window_months": 120}
    out["equity_vs_commodity_exposure"] = sep
    (C.REPORTS / "commodity_research.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: (v if k != "commodities" else {c: {kk: vv for kk, vv in r.items() if kk in ("ts_momentum", "cot_crowding")} for c, r in v.items()})
                      for k, v in out.items()}, indent=1, default=str)[:4000])
    return out


if __name__ == "__main__":
    _lock = __import__("config").hold_lock()
    run()
