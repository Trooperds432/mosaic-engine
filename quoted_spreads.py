"""Quoted half-spreads for cost model v2 (prereg/spread_check_v1.json).

build: LSEG end-of-day BID/ASK (lseg_quotes_pull.py output) -> data/processed/quoted_spread.parquet with
security_id, date, hs_quoted = trailing 60-session mean of (ask-bid)/(2*mid), at least 40 valid days.
The value on date t uses quotes up to and including t's close, the same timing as cs_spread.

attach / wide: point-in-time lookup of the latest value at or before each date, no older than
QUOTE_MAX_STALENESS_DAYS calendar days; otherwise NaN (callers fall back to Corwin-Schultz).

usage: python quoted_spreads.py BIDASK_CSV
"""
import functools
import sys

import numpy as np
import pandas as pd

import config as C

WINDOW, MIN_DAYS, MAX_REL = 60, 40, 0.20
BUNDLE_MASTER = C.RAW / "historical" / "8433bd236228ba351c9a" / "security_master.parquet"


def from_quotes(bidask):
    """bidask columns: ric|security_id, date, bid, ask. Returns key, date, hs_quoted."""
    key = "security_id" if "security_id" in bidask else "ric"
    q = bidask.dropna(subset=["date", "bid", "ask"]).copy()
    q["date"] = pd.to_datetime(q["date"]).astype("datetime64[ns]")
    q = q[(q.bid > 0) & (q.ask > q.bid)]
    mid = (q.bid + q.ask) / 2
    q = q[(q.ask - q.bid) / mid < MAX_REL]
    q["q"] = (q.ask - q.bid) / (2 * ((q.bid + q.ask) / 2))
    q = q.sort_values([key, "date"]).drop_duplicates([key, "date"])
    q["hs_quoted"] = q.groupby(key)["q"].transform(lambda s: s.rolling(WINDOW, min_periods=MIN_DAYS).mean())
    return q.dropna(subset=["hs_quoted"])[[key, "date", "hs_quoted"]].reset_index(drop=True)


def build(bidask_csv, master_path=BUNDLE_MASTER, out=None):
    out = out or C.QUOTED_SPREAD_FILE
    table = from_quotes(pd.read_csv(bidask_csv))
    master = pd.read_parquet(master_path)[["security_id", "ric"]]
    table = table.merge(master, on="ric", how="inner")[["security_id", "date", "hs_quoted"]]
    table.to_parquet(out, index=False)
    return table


@functools.lru_cache(maxsize=2)
def _load(path, mtime):
    t = pd.read_parquet(path)
    t["date"] = pd.to_datetime(t["date"]).astype("datetime64[ns]")
    return t.sort_values("date")


def table():
    p = C.QUOTED_SPREAD_FILE
    if not p.exists():
        return None
    return _load(str(p), p.stat().st_mtime)


def attach(panel, quotes=None):
    """Series of hs_quoted aligned to panel.index, from panel['symbol'] (security_id) and panel['date']."""
    quotes = table() if quotes is None else quotes
    if quotes is None or not {"symbol", "date"} <= set(panel.columns):
        return pd.Series(np.nan, index=panel.index)
    left = pd.DataFrame({"symbol": panel["symbol"].astype(str).to_numpy(),
                         "date": pd.to_datetime(panel["date"]).astype("datetime64[ns]").to_numpy(),
                         "_pos": np.arange(len(panel))}).sort_values("date")
    right = quotes.rename(columns={"security_id": "symbol"})[["symbol", "date", "hs_quoted"]]
    right = right.assign(symbol=right["symbol"].astype(str)).sort_values("date")
    m = pd.merge_asof(left, right, on="date", by="symbol", direction="backward",
                      tolerance=pd.Timedelta(days=C.QUOTE_MAX_STALENESS_DAYS))
    return pd.Series(m.sort_values("_pos")["hs_quoted"].to_numpy(), index=panel.index)


def wide(calendar, columns, quotes=None):
    """Date x security frame of hs_quoted on `calendar`, same staleness rule as attach."""
    quotes = table() if quotes is None else quotes
    cal = pd.DatetimeIndex(calendar)
    if quotes is None:
        return pd.DataFrame(np.nan, index=cal, columns=columns)
    w = quotes[quotes["security_id"].isin(columns)].pivot(index="date", columns="security_id", values="hs_quoted")
    w = w.reindex(columns=columns)
    union = w.index.union(cal)
    have = w.reindex(union)
    stamps = pd.DataFrame(np.repeat(union.values[:, None], len(columns), axis=1), index=union, columns=columns)
    age = stamps - stamps.where(have.notna()).ffill()
    filled = have.ffill().where(age.le(pd.Timedelta(days=C.QUOTE_MAX_STALENESS_DAYS)))
    return filled.reindex(cal)


if __name__ == "__main__":
    t = build(sys.argv[1])
    print(len(t), "rows,", t["security_id"].nunique(), "securities,", t["date"].min().date(), "to", t["date"].max().date())
