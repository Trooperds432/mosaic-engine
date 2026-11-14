"""Commodity and energy research module (spec 4C / 4E statistical models). Research output only: the bot trades
equities, so nothing here becomes a signal directly.

Data: Yahoo continuous front-month futures (roll gaps included; term structure is NOT observable from this
source), CFTC disaggregated COT (+3d availability), timestamped EIA crude-stock snapshots, FRED NY Harbor product prices.
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
import argparse
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


def event_returns(prices, availability, horizon):
    """Use the first close strictly after the availability date and a full exit horizon."""
    prices = prices.sort_index()
    if not prices.index.is_unique or horizon < 1:
        raise ValueError("unique price sessions and a positive horizon required")
    entry = prices.index.searchsorted(pd.DatetimeIndex(availability), side="right")
    exit_position = entry + horizon
    result = pd.DataFrame({"entry": pd.NaT, "exit": pd.NaT, "y": np.nan}, index=range(len(entry)))
    valid = exit_position < len(prices)
    e, x = entry[valid], exit_position[valid]
    start, end = prices.iloc[e].to_numpy(), prices.iloc[x].to_numpy()
    valid[valid] = np.isfinite(start) & np.isfinite(end) & (start > 0) & (end > 0)
    result.loc[valid, "entry"] = prices.index[entry[valid]]
    result.loc[valid, "exit"] = prices.index[exit_position[valid]]
    result.loc[valid, "y"] = np.log(prices.iloc[exit_position[valid]].to_numpy() / prices.iloc[entry[valid]].to_numpy())
    return result


def event_splits(frame, holdout):
    return [("discovery", frame[frame["exit"].lt(holdout)]),
            ("holdout", frame[frame["entry"].ge(holdout)])]


def inventory_observations(frame):
    """One latest inventory observation per availability date, with as-of history."""
    d = frame[frame["series"].eq("crude_stocks_ex_spr")
              & frame["source"].eq("eia_observed_snapshot")].copy()
    d = d[d["available_date"].notna() & d["obs_date"].notna()
          & d["available_date"].gt(d["obs_date"]) & np.isfinite(d["value"])]
    d = d.sort_values(["available_date", "obs_date"], kind="stable")
    history, rows = {}, []
    for available, batch in d.groupby("available_date", sort=True):
        for row in batch.itertuples():
            history[row.obs_date] = row.value
        obs = max(history)
        week = obs.isocalendar().week
        baseline = pd.Series([value for day, value in history.items()
                              if obs - pd.Timedelta(days=1830) <= day < obs and day.isocalendar().week == week],
                             dtype=float)
        z = (history[obs] - baseline.mean()) / baseline.std() if len(baseline) >= 4 and baseline.std() > 0 else np.nan
        rows.append({"available_date": available, "obs_date": obs, "z": z})
    return pd.DataFrame(rows, columns=["available_date", "obs_date", "z"])


def contract_curves(decision, session):
    from futures_curves import curve_at, validate

    paths = sorted((C.PROC / "futures_contracts").glob("*.parquet"))
    if not paths:
        return {"status": "NOT_COMPUTED", "reason": "timestamped individual contract settlements required"}
    observations = pd.concat([pd.read_parquet(path, filters=[("session", "=", pd.Timestamp(session))])
                              for path in paths], ignore_index=True).drop_duplicates()
    curves = curve_at(validate(observations), decision, session)
    computed = any(curve["status"] == "COMPUTED" for curve in curves.values())
    return {"status": "COMPUTED" if computed else "NOT_COMPUTED",
            "decision_utc": pd.Timestamp(decision).tz_convert("UTC").isoformat(),
            "session": str(pd.Timestamp(session).date()), "curves": curves}


def run():
    global HOLDOUT
    spy = pd.read_parquet(C.RAW / "prices" / f"{C.BENCHMARK}.parquet").drop_duplicates("date").set_index("date")
    HOLDOUT = spy.index[-1] - pd.DateOffset(years=C.HOLDOUT_YEARS)
    fut = pd.read_parquet(C.RAW / "alt" / "futures.parquet")
    M = monthly(fut)
    M = {commodity: prices.where(prices.gt(0)) for commodity, prices in M.items()}
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
            zz = cc.assign(z=z).dropna(subset=["z"]).reset_index(drop=True)
            dz = zz.join(event_returns(daily, zz["available_date"], 20)).dropna(subset=["y"])
            for lab, sub in event_splits(dz, HOLDOUT):
                if len(sub) > 30:
                    m = hac(sub["y"], sub["z"], lags=4)
                    rec.setdefault("cot_crowding", {})[lab] = {"coef_per_z_4w": float(m.params.iloc[1]), "t": float(m.tvalues.iloc[1]), "n": int(len(sub)),
                                                                 "hypothesis": "negative (crowded longs underperform)"}
        out["commodities"][c] = rec

    # EIA inventory surprise -> next-week WTI
    mp = pd.read_parquet(C.PROC / "macro_pit.parquet")
    cr = inventory_observations(mp)
    cr["dz"] = cr["z"].diff()
    wti = fut[fut["commodity"] == "wti"].set_index("date")["close"]
    ci = cr.join(event_returns(wti, cr["available_date"], 5)).dropna(subset=["dz", "y"])
    inv = {}
    for lab, sub in event_splits(ci, HOLDOUT):
        if len(sub) < 30 or sub["dz"].nunique() < 2:
            inv[lab] = {"status": "NOT_COMPUTED", "n": len(sub),
                        "reason": "need at least 30 settled timestamped observations and variation"}
            continue
        m = hac(sub["y"], sub["dz"], lags=2)
        inv[lab] = {"status": "COMPUTED", "coef_per_z_change_1w": float(m.params.iloc[1]),
                    "t": float(m.tvalues.iloc[1]), "n": int(len(sub))}
    out["eia_inventory_surprise"] = inv | {"hypothesis": "negative (unexpected builds lower crude)",
                                           "measure": "change in as-of same-week inventory z-score"}

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve-session", help="exchange session for the settlement curve")
    parser.add_argument("--as-of", help="timezone-aware decision timestamp")
    args = parser.parse_args()
    if bool(args.curve_session) != bool(args.as_of):
        parser.error("--curve-session and --as-of must be supplied together")
    if args.curve_session:
        print(json.dumps(contract_curves(args.as_of, args.curve_session), indent=2, allow_nan=False))
    else:
        _lock = C.hold_lock()
        run()
