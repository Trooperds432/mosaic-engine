"""Convert a CodeBook S&P 500 survivorship export into a historical_data.py bundle directory.

Inputs (from CodeBook, see docs/HISTORICAL_DATA.md):
  spx_members.csv           month-end S&P 500 constituent snapshots (date, ric, ticker, ...)
  mosaic_hist_static.csv    per-RIC CIK, name, retire date, exchange, ticker (LSEG TR fields)
  mosaic_hist_prices.csv    daily split-adjusted OHLCV, daily total return, issuer market cap and
                            security shares outstanding per RIC (LSEG TR fields)
  mosaic_sp500_dated_returns.csv.gz  separate Instrument, CalcDate, ReturnDate, Total Return export

Treatment, stated rather than hidden:
  * Membership comes from month-end snapshots: an interval starts at the first snapshot that lists the
    security and ends (exclusive) at the first later snapshot that does not. Intra-month joins are seen
    late and intra-month removals stay eligible until month end. known_at = start.
    No membership is inferred after the final snapshot.
  * close/open/high/low are LSEG split-adjusted prices. adjclose is a total-return index built from
    independently dated LSEG daily total returns; adj_open = open * adjclose / close.
  * market_cap is LSEG issuer market cap; issuers with several securities in the bundle get it split by
    shares outstanding. Missing cap observations are excluded.
  * No terminal payouts are supplied, so terminal.csv is empty and delisted outcomes stay unknown.
  * Only SP500 membership is supplied. Every completeness attestation in coverage.json is false.
Usage: python lseg_bundle.py SOURCE_DIR OUTPUT_DIR
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from historical_data import validate_bundle

SOURCE = "lseg_codebook"


def _first(series):
    s = series.dropna()
    return s.iloc[0] if len(s) else np.nan


def reference_by_ric(raw):
    raw = raw.copy()
    cik = pd.to_numeric(raw["CIK Number"], errors="coerce")
    raw["CIK Number"] = cik.where(np.isfinite(cik) & cik.gt(0) & cik.mod(1).eq(0))
    counts = raw.groupby("Instrument")["CIK Number"].nunique()
    ambiguous = sorted(counts[counts.gt(1)].index)
    unique = raw.loc[~raw["Instrument"].isin(ambiguous)]
    return unique.groupby("Instrument").agg(_first), ambiguous


def security_ids(static, members):
    tick = members.groupby("ric")["ticker"].agg(_first)
    rows = []
    for ric, r in static.iterrows():
        base = r.get("Ticker Symbol")
        if not isinstance(base, str) or not base.strip():
            base = tick.get(ric, ric.split(".")[0])
        base = re.sub(r"[^A-Za-z0-9]", "_", str(base).strip()).strip("_") or re.sub(r"[^A-Za-z0-9]", "_", ric)
        sid = base
        if "^" in ric:
            retire = pd.to_datetime(r.get("RetireDate"), errors="coerce")
            sid = f"{base}_{retire:%Y%m%d}" if pd.notna(retire) else f"{base}_{re.sub(r'[^A-Za-z0-9]', '', ric.split('^')[1])}"
        rows.append((ric, sid, base))
    ids = pd.DataFrame(rows, columns=["ric", "security_id", "ticker"]).set_index("ric")
    reserved = {C.BENCHMARK, *C.REGIME_SERIES}
    clash = ids["security_id"].duplicated(keep=False) | ids["security_id"].isin(reserved)
    ids.loc[clash, "security_id"] = [f"{s}_{re.sub(r'[^A-Za-z0-9]', '', ric)}" for ric, s in ids.loc[clash, "security_id"].items()]
    if ids["security_id"].duplicated().any():
        raise ValueError("could not derive unique security IDs")
    return ids


def membership(members, ids):
    snaps = pd.DatetimeIndex(sorted(members["date"].unique()))
    rows = []
    for ric, g in members.loc[members["ric"].isin(ids.index)].groupby("ric"):
        present = set(g["date"])
        start = None
        for d in snaps:
            if d in present and start is None:
                start = d
            elif d not in present and start is not None:
                rows.append((ids.at[ric, "security_id"], start, d))
                start = None
        if start is not None:
            rows.append((ids.at[ric, "security_id"], start, snaps[-1] + pd.Timedelta(days=1)))
    m = pd.DataFrame(rows, columns=["security_id", "start", "end"])
    m["index"], m["known_at"], m["source"] = "SP500", m["start"], f"{SOURCE}:spx_members month-end snapshots"
    return m[["security_id", "index", "start", "end", "known_at", "source"]]


def dated_returns(raw):
    r = raw.rename(columns={"Instrument": "ric", "ReturnDate": "date", "Total Return": "tr"})
    for col in ["CalcDate", "date"]:
        r[col] = pd.to_datetime(r[col], errors="coerce").dt.normalize()
    r["tr"] = pd.to_numeric(r["tr"], errors="coerce")
    r = r.loc[r["date"].notna() & r["date"].eq(r["CalcDate"]) &
              np.isfinite(r["tr"]) & r["tr"].gt(-100), ["ric", "date", "tr"]]
    r = r.drop_duplicates()
    return r.loc[~r.duplicated(["ric", "date"], keep=False)]


def prices(raw, ids, master, returns):
    p = raw.rename(columns={"Instrument": "ric", "Date": "date", "Price Close": "close", "Price Open": "open",
                            "Price High": "high", "Price Low": "low", "Volume": "volume",
                            "Company Market Cap": "issuer_cap", "Outstanding Shares": "shares"})
    p = p.loc[p["ric"].isin(ids.index)].drop(columns="Total Return", errors="ignore").copy()
    p["date"] = pd.to_datetime(p["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    for c in ["close", "open", "high", "low", "volume", "issuer_cap", "shares"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    n0 = len(p)
    p = p.dropna(subset=["date", "close", "open", "high", "low", "volume"])
    p = p[(p[["close", "open", "high", "low"]] > 0).all(axis=1) & (p["volume"] >= 0) & (p["high"] >= p["low"])]
    p = p.drop_duplicates()
    p = p.loc[~p.duplicated(["ric", "date"], keep=False)].sort_values(["ric", "date"])
    dropped = n0 - len(p)
    p["security_id"] = p["ric"].map(ids["security_id"])
    p["ticker"] = p["ric"].map(ids["ticker"])

    r = dated_returns(returns)
    r = r.loc[r["ric"].isin(p["ric"])]
    p = p.merge(r, on=["ric", "date"], how="outer", validate="one_to_one")
    p = p.sort_values(["ric", "date"])
    missing_returns = int((p["close"].notna() & p["tr"].isna()).sum())
    missing_prices = int((p["close"].isna() & p["tr"].notna()).sum())
    # Use the uninterrupted dated series from the first verified observation.
    # An interior hole cannot be bridged with a price-only or fabricated return.
    has_return = p["tr"].notna()
    anchored = has_return & p["close"].notna()
    started = anchored.groupby(p["ric"]).cummax()
    broken = (started & ~has_return).groupby(p["ric"]).cummax()
    n_before_returns = int(p["close"].notna().sum())
    p = p.loc[has_return & started & ~broken].copy()
    log_growth = np.log1p(p["tr"] / 100)
    log_index = log_growth.groupby(p["ric"]).cumsum()
    # A dated return can bridge an absent OHLCV row. The session stays
    # unavailable for execution and must not become a synthetic price.
    p["log_index"] = log_index
    p = p.loc[p["close"].notna()].copy()
    retained_prices = len(p)
    log_index = p.pop("log_index")
    p["adjclose"] = np.exp(log_index - log_index.groupby(p["ric"]).transform("last")) * (
        p.groupby("ric")["close"].transform("last")
    )
    p["adj_open"] = p["open"] * p["adjclose"] / p["close"]
    if not (np.isfinite(p[["adjclose", "adj_open"]]) &
            p[["adjclose", "adj_open"]].gt(0)).all().all():
        raise ValueError("dated returns produced invalid adjusted prices")

    p["cik"] = p["security_id"].map(master.set_index("security_id")["cik"])
    multi = master["cik"].duplicated(keep=False)
    multi_ids = set(master.loc[multi, "security_id"])
    p["market_cap"] = p["issuer_cap"]
    if multi_ids:
        m = p["security_id"].isin(multi_ids)
        total = p[m].groupby(["cik", "date"])["shares"].transform("sum")
        p.loc[m, "market_cap"] = p.loc[m, "issuer_cap"] * p.loc[m, "shares"] / total
    no_cap = ~np.isfinite(p["market_cap"]) | (p["market_cap"] <= 0)
    p = p[~no_cap]
    # one ticker per date
    dup = p.duplicated(["ticker", "date"], keep=False)
    p.loc[dup, "ticker"] = p.loc[dup, "security_id"]
    p["source"] = SOURCE
    cols = ["security_id", "date", "ticker", "open", "high", "low", "close", "volume", "adjclose", "adj_open", "market_cap", "source"]
    return p[cols], {"rows_in": int(n0), "rows_dropped_invalid": int(dropped), "rows_dropped_no_market_cap": int(no_cap.sum()),
                     "rows_missing_dated_return": missing_returns,
                     "dated_returns_without_ohlcv": missing_prices,
                     "rows_excluded_by_return_coverage": n_before_returns - retained_prices,
                     "multi_security_issuers": sorted(multi_ids)}


def build(src, out):
    src, out = Path(src), Path(out)
    if out.exists():
        raise FileExistsError(out)
    members = pd.read_csv(src / "spx_members.csv")
    members["date"] = pd.to_datetime(members["date"])
    constituent_rics = set(members["ric"].dropna())
    raw_static = pd.read_csv(src / "mosaic_hist_static.csv")
    raw_static = raw_static.loc[raw_static["Instrument"].isin(constituent_rics)]
    static, ambiguous_cik = reference_by_ric(raw_static)
    no_reference = sorted(constituent_rics - set(raw_static["Instrument"]))
    cik = pd.to_numeric(static["CIK Number"], errors="coerce")
    no_cik = sorted(static.index[cik.isna() | (cik <= 0)])
    static = static[cik.notna() & (cik > 0)]
    ids = security_ids(static, members)
    master = pd.DataFrame({"security_id": ids["security_id"], "cik": cik.loc[ids.index].astype("int64"),
                           "ticker": ids["ticker"], "exchange": static["Exchange Name"].fillna("UNKNOWN"),
                           "name": static["Company Common Name"], "ric": ids.index,
                           "source": SOURCE}).reset_index(drop=True)
    mem = membership(members, ids)
    returns = pd.read_csv(src / "mosaic_sp500_dated_returns.csv.gz")
    px, stats = prices(pd.read_csv(src / "mosaic_hist_prices.csv"), ids, master, returns)
    usable_ids = set(px["security_id"])
    no_usable_prices = sorted(master.loc[~master["security_id"].isin(usable_ids), "ric"])
    master = master[master["security_id"].isin(usable_ids)]
    mem = mem[mem["security_id"].isin(master["security_id"])]
    terminal = pd.DataFrame(columns=["security_id", "last_trade_date", "settlement_date", "terminal_adjclose", "source"])
    if px.empty:
        raise ValueError("no usable prices after reconciling dated returns and OHLCV")
    validate_bundle(master.copy(), mem.copy(), px.copy(), terminal.copy())
    out.mkdir(parents=True, exist_ok=False)
    master.to_csv(out / "security_master.csv", index=False)
    mem.to_csv(out / "membership.csv", index=False, date_format="%Y-%m-%d")
    px.to_csv(out / "prices.csv", index=False, date_format="%Y-%m-%d")
    terminal.to_csv(out / "terminal.csv", index=False)
    coverage = {
        "start": str(max(members["date"].min(), px["date"].min()).date()),
        "end": str(min(members["date"].max(), px["date"].max()).date()),
        "source_date_ranges": {
            "membership": [str(members["date"].min().date()), str(members["date"].max().date())],
            "prices": [str(px["date"].min().date()), str(px["date"].max().date())],
        },
        "indices": ["SP500"],
        "constituents_complete": False, "delisted_prices_complete": False,
        "terminal_events_complete": False, "identifiers_verified": False,
        "source": "LSEG CodeBook (Workspace) export, converted by lseg_bundle.py",
        "limitations": [
            "S&P 500 only; Nasdaq-100 and Russell 2000 membership not supplied",
            "membership approximated from month-end snapshots and capped at the final snapshot",
            "no terminal payouts; delisted outcomes after the last trade are unknown",
            "CIKs are LSEG reference data, not independently verified against SEC",
            "adjclose built from uninterrupted, explicitly dated LSEG daily total returns",
            "price histories can end early when dated returns are missing",
        ],
        "conversion": {**stats, "rics_without_reference": no_reference,
                       "rics_without_cik": no_cik, "rics_with_conflicting_ciks": ambiguous_cik,
                       "rics_without_usable_prices": no_usable_prices,
                       "constituent_rics_in_source": len(constituent_rics), "securities": int(len(master)),
                       "membership_intervals": int(len(mem))},
    }
    (out / "coverage.json").write_text(json.dumps(coverage, indent=2))
    print(json.dumps({k: v for k, v in coverage["conversion"].items() if k != "multi_security_issuers"}, indent=1))
    return out


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
