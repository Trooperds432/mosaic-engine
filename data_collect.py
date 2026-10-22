"""Collectors that build point-in-time history the free tiers do not provide (run nightly by mosaic-collect.timer).

  av_earnings   Alpha Vantage EARNINGS: every quarter's reported EPS vs the consensus estimate at report time,
                with report date and time (1996+). Free key: 25 requests/day, so 24 companies per day, largest
                first. Available: report date if pre-market, next session if post-market.
  options       Cboe delayed quotes (15-min delayed, after the close): per stock ATM implied vol (~30d and ~90d),
                term slope, 25-delta risk reversal (skew), put/call volume and open-interest ratios.
  estimates     Nasdaq analyst API: consensus EPS for the current quarter/year, its value 1 week and 1 month ago
                (revision momentum), number of estimates, mean rating.
  recs          Finnhub recommendation trends (monthly buy/hold/sell counts), weekly.
Snapshots are stored per date with available_date = snapshot date; they become backtestable only as history
accumulates (features.py excludes them from models until they span >= 3 years).
"""
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone

import httpx
import numpy as np
import pandas as pd

import config as C

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/130 Safari/537.36"}
NQ = UA | {"Accept": "application/json", "Origin": "https://www.nasdaq.com", "Referer": "https://www.nasdaq.com/"}
OUT = C.RAW / "collect"
AV_DAILY = 24


def _atomic(df, dest):
    """Write then rename, so the nightly feature build never reads a half-written file."""
    tmp = dest.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(dest)


def _key(name):
    v = os.environ.get(name)
    if v:
        return v
    for line in open("/root/.hermes/.env"):
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip()
    return None


def universe_by_size():
    """Currently evaluated symbols, largest market cap first (from the latest broad panel date)."""
    p = pd.read_parquet(C.PROC / "panel_broad.parquet", columns=["date", "symbol", "mcap"])
    last = p[p["date"] == p["date"].max()].sort_values("mcap", ascending=False)
    return list(last["symbol"])


# ---------------------------------------------------------------- Alpha Vantage earnings vs consensus
def av_earnings():
    d = OUT / "av_earnings"
    d.mkdir(parents=True, exist_ok=True)
    key = _key("ALPHA_VANTAGE_API_KEY")
    calls_f = OUT / "av_calls.json"
    calls = json.loads(calls_f.read_text()) if calls_f.exists() else {}
    today = datetime.now(timezone.utc).date().isoformat()
    used = calls.get(today, 0)
    todo = [s for s in universe_by_size() if not (d / f"{s}.json").exists()]
    got = 0
    with httpx.Client(timeout=60) as c:
        for s in todo:
            if used >= AV_DAILY:
                break
            r = c.get("https://www.alphavantage.co/query", params={"function": "EARNINGS", "symbol": s.replace("-", "."), "apikey": key})
            used += 1
            j = r.json() if r.status_code == 200 else {}
            if "quarterlyEarnings" in j:
                (d / f"{s}.json").write_text(json.dumps(j["quarterlyEarnings"]))
                got += 1
            elif "Information" in j or "Note" in j:
                print("AV limit:", (j.get("Information") or j.get("Note"))[:120], flush=True)
                break
            else:
                (d / f"{s}.json").write_text("[]")  # no coverage for this symbol
            time.sleep(1.5)
    calls[today] = used
    calls_f.write_text(json.dumps(calls))
    rows = []
    for f in d.glob("*.json"):
        for q in json.loads(f.read_text()):
            rows.append({"symbol": f.stem, **q})
    df = pd.DataFrame(rows)
    if len(df):
        for k in ["reportedEPS", "estimatedEPS", "surprise", "surprisePercentage"]:
            df[k] = pd.to_numeric(df[k], errors="coerce")
        df["reportedDate"] = pd.to_datetime(df["reportedDate"], errors="coerce")
        # pre-market reports are known at that day's close; post-market (or unknown) ones from the next session
        df["available_date"] = df["reportedDate"] + pd.to_timedelta((df["reportTime"].fillna("") != "pre-market").astype(int), unit="D")
        _atomic(df, C.PROC / "av_earnings.parquet")
    done = len(list(d.glob("*.json")))
    print(f"av_earnings: fetched {got} today ({used}/{AV_DAILY} calls), {done} of {done + len(todo) - got} symbols covered", flush=True)


# ---------------------------------------------------------------- Cboe options snapshot
OPT_RE = re.compile(r"^(.+?)(\d{6})([CP])(\d{8})$")


def option_metrics(sym, d):
    spot = d.get("current_price") or d.get("close")
    if not spot:
        return None
    rows = []
    for o in d["options"]:
        m = OPT_RE.match(o["option"])
        if not m:
            continue
        exp = datetime.strptime(m.group(2), "%y%m%d").date()
        rows.append((exp, m.group(3), int(m.group(4)) / 1000, o.get("iv") or 0, o.get("delta"), o.get("volume") or 0, o.get("open_interest") or 0))
    if not rows:
        return None
    x = pd.DataFrame(rows, columns=["exp", "cp", "k", "iv", "delta", "vol", "oi"])
    x["dte"] = (pd.to_datetime(x["exp"]) - pd.Timestamp(date.today())).dt.days
    x = x[(x["dte"] >= 7) & (x["iv"] > 0.01) & (x["iv"] < 5)]

    def atm(target):
        if not len(x):
            return np.nan, None
        e = x.loc[(x["dte"] - target).abs().idxmin(), "exp"]
        g = x[x["exp"] == e]
        k = g.loc[(g["k"] - spot).abs().idxmin(), "k"]
        return float(g[g["k"] == k]["iv"].mean()), g
    iv30, g30 = atm(30)
    iv90, _ = atm(90)
    rr = np.nan
    if g30 is not None and g30["delta"].notna().any():
        p = g30[(g30["cp"] == "P") & g30["delta"].notna()]
        cc = g30[(g30["cp"] == "C") & g30["delta"].notna()]
        if len(p) and len(cc):
            rr = float(p.loc[(p["delta"] + 0.25).abs().idxmin(), "iv"] - cc.loc[(cc["delta"] - 0.25).abs().idxmin(), "iv"])
    allx = pd.DataFrame(rows, columns=["exp", "cp", "k", "iv", "delta", "vol", "oi"])
    cv, pv = allx.loc[allx["cp"] == "C", "vol"].sum(), allx.loc[allx["cp"] == "P", "vol"].sum()
    co, po = allx.loc[allx["cp"] == "C", "oi"].sum(), allx.loc[allx["cp"] == "P", "oi"].sum()
    return {"symbol": sym, "spot": spot, "atm_iv_30": iv30, "atm_iv_90": iv90, "term_slope": iv90 / iv30 - 1 if iv30 and iv90 else np.nan,
            "risk_reversal_25d": rr, "pc_volume_ratio": pv / cv if cv else np.nan, "pc_oi_ratio": po / co if co else np.nan,
            "option_volume": cv + pv, "open_interest": co + po, "contracts": len(rows), "quote_time": d.get("last_trade_time")}


def options(symbols):
    today = date.today().isoformat()
    dest = OUT / "options" / f"{today}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out, err = [], 0
    with httpx.Client(timeout=60, headers=UA, follow_redirects=True) as c:
        def one(s):
            for i in range(3):
                try:
                    r = c.get(f"https://cdn.cboe.com/api/global/delayed_quotes/options/{s.replace('-', '.')}.json")
                    if r.status_code == 200:
                        return option_metrics(s, r.json()["data"])
                    if r.status_code in (403, 404):
                        return None
                except Exception:
                    pass
                time.sleep(2 * (i + 1))
            return None
        with ThreadPoolExecutor(4) as ex:
            for m in ex.map(one, symbols):
                if m:
                    out.append(m)
                else:
                    err += 1
    df = pd.DataFrame(out)
    df["snapshot_date"] = pd.Timestamp(today)
    _atomic(df, dest)
    print(f"options: {len(df)} symbols snapshotted, {err} without listed options or unavailable", flush=True)


# ---------------------------------------------------------------- Nasdaq estimates snapshot
def _num(v):
    try:
        return float(str(v).replace("$", "").replace(",", ""))
    except (TypeError, ValueError):
        return np.nan


def estimates(symbols):
    today = date.today().isoformat()
    dest = OUT / "estimates" / f"{today}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out = []
    with httpx.Client(timeout=30, headers=NQ) as c:
        def one(s):
            rec = {"symbol": s}
            try:
                m = c.get(f"https://api.nasdaq.com/api/analyst/{s.replace('-', '.')}/estimate-momentum").json().get("data") or {}
                ch = m.get("changeInConsensus") or {}
                for k, lab in [("currentData", "now"), ("weekData", "1w"), ("monthData", "1m")]:
                    v = ch.get(k) or {}
                    rec[f"qtr_mean_{lab}"], rec[f"yr_mean_{lab}"] = _num(v.get("qtrMean")), _num(v.get("yrMean"))
                time.sleep(0.2)
                f = c.get(f"https://api.nasdaq.com/api/analyst/{s.replace('-', '.')}/earnings-forecast").json().get("data") or {}
                rows = ((f.get("quarterlyForecast") or {}).get("rows") or [])
                if rows:
                    rec["qtr_estimates_n"] = _num(rows[0].get("noOfEstimates"))
                    rec["qtr_eps_high"], rec["qtr_eps_low"] = _num(rows[0].get("highEPSForecast")), _num(rows[0].get("lowEPSForecast"))
                time.sleep(0.2)
                r = c.get(f"https://api.nasdaq.com/api/analyst/{s.replace('-', '.')}/ratings").json().get("data") or {}
                rec["mean_rating"] = r.get("meanRatingType")
                rec["upgrades_downgrades"] = len(r.get("upgradesDowngrades") or [])
            except Exception as e:
                rec["error"] = repr(e)[:100]
            return rec
        with ThreadPoolExecutor(3) as ex:
            out = list(ex.map(one, symbols))
    df = pd.DataFrame(out)
    df["snapshot_date"] = pd.Timestamp(today)
    for lab in ["qtr", "yr"]:
        df[f"{lab}_rev_1w"] = df[f"{lab}_mean_now"] / df[f"{lab}_mean_1w"] - 1
        df[f"{lab}_rev_1m"] = df[f"{lab}_mean_now"] / df[f"{lab}_mean_1m"] - 1
    df["dispersion"] = (df["qtr_eps_high"] - df["qtr_eps_low"]) / df["qtr_mean_now"].abs()
    _atomic(df, dest)
    print(f"estimates: {int(df['qtr_mean_now'].notna().sum())} of {len(df)} symbols with consensus data", flush=True)


# ---------------------------------------------------------------- Finnhub recommendation trends
def recs(symbols):
    key = _key("FINNHUB_API_KEY")
    dest = C.PROC / "finnhub_recommendations.parquet"
    rows = []
    with httpx.Client(timeout=30) as c:
        for s in symbols:
            r = c.get("https://finnhub.io/api/v1/stock/recommendation", params={"symbol": s.replace("-", "."), "token": key})
            if r.status_code == 429:
                time.sleep(30)
                continue
            if r.status_code == 200 and isinstance(r.json(), list):
                for x in r.json():
                    rows.append({**x, "symbol": s, "collected": date.today().isoformat()})
            time.sleep(1.05)  # free tier: 60 calls/minute
    new = pd.DataFrame(rows)
    if dest.exists():
        new = pd.concat([pd.read_parquet(dest), new], ignore_index=True)
    # a month's counts can be revised later; keep the first collected version (as first seen)
    new = new.sort_values("collected").drop_duplicates(["symbol", "period"], keep="first")
    _atomic(new, dest)
    print(f"recommendations: {len(rows)} rows fetched, {new['symbol'].nunique()} symbols stored", flush=True)


def run(parts):
    syms = universe_by_size()
    weekday = date.today().weekday() < 5
    for p in parts:
        try:
            if p == "av":
                av_earnings()
            elif p == "options" and weekday:
                options(syms)
            elif p == "estimates" and weekday:
                estimates(syms)
            elif p == "recs" and date.today().weekday() == 4:  # weekly, Fridays
                recs(syms)
        except Exception as e:
            print(p, "ERROR", repr(e)[:300], flush=True)


if __name__ == "__main__":
    run(sys.argv[1:] or ["av", "options", "estimates", "recs"])
