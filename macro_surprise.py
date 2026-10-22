"""ARIMA-based macro surprises (spec 4D "released vs expected"; adapted from the Auto-ARIMA notebook).

No free point-in-time consensus exists, so the "expected" value is a statistical forecast made only from data
published before each release:
  for each release of observation T (its first ALFRED vintage), y_T = the reported change (value(T) minus value(T-1),
  both as published in that vintage; log-% changes for CPI, IP, claims), forecast by an ARIMA(p,d,q) chosen by AIC
  (p, q <= 2; d from an ADF test) fitted on the first-release changes of all earlier observations.
  surprise = y_T - forecast;  z = surprise / std of the previous 36 surprises.
Available on the ALFRED vintage date (conservative: real releases can be earlier).
Output: processed/macro_surprises.parquet (+ reports/macro_surprise.json with forecast accuracy vs a random walk).
"""
import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

import config as C

warnings.filterwarnings("ignore")
SERIES = {"UNRATE": "diff", "PAYEMS": "diff", "CPIAUCSL": "logdiff", "INDPRO": "logdiff", "ICSA": "logdiff", "IPG3344S": "logdiff"}
MIN_OBS = 48


def releases(series, how):
    """One row per observation's first publication: change as reported in that vintage."""
    v = pd.read_parquet(C.RAW / "macro" / f"alfred_{series}.parquet")
    v["obs_date"] = pd.to_datetime(v["obs_date"])
    v["vdate"] = pd.to_datetime(v["vintage"])
    v = v.sort_values(["vdate", "obs_date"])
    first = v.sort_values("vdate").drop_duplicates("obs_date", keep="first")[["obs_date", "vdate", "value"]]
    rows = []
    for r in first.itertuples():
        same = v[(v["vdate"] == r.vdate) & (v["obs_date"] < r.obs_date)]
        if same.empty:
            continue
        prev = same.sort_values("obs_date").iloc[-1]["value"]
        y = (r.value - prev) if how == "diff" else 100 * np.log(r.value / prev)
        rows.append({"obs_date": r.obs_date, "available_date": r.vdate, "value": r.value, "change": y})
    return pd.DataFrame(rows).sort_values("obs_date").reset_index(drop=True)


def arima_forecast(hist):
    med = np.median(hist)
    mad = np.median(np.abs(hist - med)) * 1.4826 or np.std(hist)
    hist = np.clip(hist, med - 5 * mad, med + 5 * mad)  # robust to one-off shocks (e.g. 2020) dominating the fit
    d = 0 if adfuller(hist, autolag="AIC")[1] < 0.05 else 1
    best = None
    for p in range(3):
        for q in range(3):
            try:
                fit = SARIMAX(hist, order=(p, d, q), trend="c" if d == 0 else "n").fit(disp=False)
            except Exception:
                continue
            if best is None or fit.aic < best[0]:
                best = (fit.aic, fit, (p, d, q))
    if best is None:
        return np.nan, None
    return float(best[1].forecast(1)[0]), best[2]


def run():
    out, summary = [], {}
    for s, how in SERIES.items():
        try:
            rel = releases(s, how)
        except FileNotFoundError:
            continue
        fc, orders = [], []
        for i in range(len(rel)):
            if i < MIN_OBS:
                fc.append(np.nan)
                orders.append(None)
                continue
            f, o = arima_forecast(rel["change"].iloc[:i].to_numpy())  # only earlier first releases
            fc.append(f)
            orders.append(str(o))
        rel["forecast"] = fc
        rel["order"] = orders
        rel["surprise"] = rel["change"] - rel["forecast"]
        rel["z"] = rel["surprise"] / rel["surprise"].rolling(36, min_periods=12).std().shift(1)
        rel["series"] = s
        e_arima = rel["surprise"].dropna()
        e_rw = (rel["change"] - rel["change"].shift(1)).loc[e_arima.index]
        normal = ~rel.loc[e_arima.index, "obs_date"].between("2020-01-01", "2021-12-31")
        summary[s] = {"releases": int(len(rel)), "forecasted": int(e_arima.size), "rmse_arima": float(np.sqrt((e_arima ** 2).mean())),
                      "rmse_random_walk": float(np.sqrt((e_rw ** 2).mean())),
                      "mae_arima": float(e_arima.abs().median()), "mae_random_walk": float(e_rw.abs().median()),
                      "rmse_arima_ex_2020_21": float(np.sqrt((e_arima[normal] ** 2).mean())), "rmse_random_walk_ex_2020_21": float(np.sqrt((e_rw[normal] ** 2).mean())), "latest": {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in rel.iloc[-1][["obs_date", "available_date", "change", "forecast", "surprise", "z"]].items()}}
        print(s, {k: round(summary[s][k], 4) for k in ["releases", "mae_arima", "mae_random_walk", "rmse_arima_ex_2020_21", "rmse_random_walk_ex_2020_21"]}, "latest z", round(float(rel["z"].iloc[-1]), 2) if pd.notna(rel["z"].iloc[-1]) else None, flush=True)
        out.append(rel)
    df = pd.concat(out, ignore_index=True)
    df.to_parquet(C.PROC / "macro_surprises.parquet", index=False)
    (C.REPORTS / "macro_surprise.json").write_text(json.dumps({"generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "series": summary}, indent=1, default=float))
    return df


def daily_z(cal, max_age_days=31):
    """Date-level growth and inflation surprise indices on the trading calendar (latest release within max_age_days)."""
    f = C.PROC / "macro_surprises.parquet"
    out = pd.DataFrame(index=cal)
    if not f.exists():
        return out
    d = pd.read_parquet(f).dropna(subset=["z"])
    sign = {"PAYEMS": 1, "UNRATE": -1, "INDPRO": 1, "ICSA": -1, "IPG3344S": 1, "CPIAUCSL": 1}
    zs = {}
    for s, g in d.groupby("series"):
        g = g.sort_values("available_date").drop_duplicates("available_date", keep="last")
        z = g.set_index("available_date")["z"].clip(-4, 4) * sign.get(s, 1)
        age = pd.Series(g["available_date"].to_numpy(), index=g["available_date"])
        zz = z.reindex(z.index.union(cal)).ffill().reindex(cal)
        aa = age.reindex(age.index.union(cal)).ffill().reindex(cal)
        zz[(cal - pd.DatetimeIndex(aa)).days > max_age_days] = np.nan
        zs[s] = zz
    z = pd.DataFrame(zs, index=cal)
    out["growth_surprise_z"] = z[[c for c in ["PAYEMS", "UNRATE", "INDPRO", "ICSA"] if c in z]].mean(axis=1)
    out["inflation_surprise_z"] = z["CPIAUCSL"] if "CPIAUCSL" in z else np.nan
    return out


if __name__ == "__main__":
    _lock = __import__("config").hold_lock()
    run()
