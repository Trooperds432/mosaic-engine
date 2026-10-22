"""Macro, rates, credit and energy fundamentals with point-in-time availability dates.

  revised series: FRED initial-release API observations, available the next day
  NFCI, mortgage and EIA snapshots: available only after their actual ingestion
  legacy sampled vintages are excluded from historical claims
Output: processed/macro_pit.parquet (series, obs_date, value, available_date, vintage, source).
"""
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import httpx
import pandas as pd

from config import PROC, RAW

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
ALFRED = "https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={}&vintage_date={}"
EIA = "https://www.eia.gov/dnav/pet/hist_xls/{}w.xls"
MARKET = ["DGS10", "DGS2", "DTB3", "T10Y3M", "T10Y2Y", "DFII10", "BAA10Y", "VIXCLS", "DCOILWTICO", "DHHNGSP",
          "DGASNYH", "DHOILNYH", "DTWEXBGS", "T5YIE", "MORTGAGE30US", "NFCI"]
REVISED = ["UNRATE", "PAYEMS", "CPIAUCSL", "INDPRO", "IPG3344S", "A34SNO", "PCOPPUSDM", "ICSA"]
EIA_SERIES = {"WCESTUS1": "crude_stocks_ex_spr", "WGTSTUS1": "gasoline_stocks", "WDISTUS1": "distillate_stocks",
              "WPULEUS3": "refinery_utilization", "WCRFPUS2": "crude_production"}
OUT = RAW / "macro"


def get(client, url, tries=5):
    for i in range(tries):
        try:
            r = client.get(url)
            if r.status_code == 200:
                return r
        except httpx.HTTPError:
            pass
        time.sleep(2 ** i)
    raise RuntimeError(url)


def fred_csv(client, s, tries=4):
    """FRED series as a two-column frame; retries when FRED returns an error page instead of CSV."""
    for i in range(tries):
        try:
            d = pd.read_csv(io.StringIO(get(client, FRED.format(s)).text))
            if d.shape[1] == 2 and len(d) > 10:
                return d
        except (pd.errors.ParserError, pd.errors.EmptyDataError, RuntimeError):
            pass
        time.sleep(5 * 2 ** i)
    return None


def market(client):
    rows = []
    prev = pd.read_parquet(PROC / "macro_pit.parquet") if (PROC / "macro_pit.parquet").exists() else None
    for s in MARKET:
        d = fred_csv(client, s)
        if d is None:  # keep yesterday's copy of this series rather than failing the whole nightly run
            old = prev[(prev["series"] == s) & (prev["source"] == "fred")] if prev is not None else None
            if old is None or old.empty:
                raise RuntimeError(f"FRED {s}: no valid response and no cached copy")
            print(f"WARNING FRED {s}: no valid CSV after retries, kept cached copy through {old['obs_date'].max().date()}", flush=True)
            rows.append(old)
            continue
        d.columns = ["obs_date", "value"]
        d["value"] = pd.to_numeric(d["value"], errors="coerce")
        d = d.dropna()
        d["obs_date"] = pd.to_datetime(d["obs_date"])
        lag = 6 if s in ("NFCI", "MORTGAGE30US") else 1  # weekly series are published days after the reference date
        d["available_date"] = d["obs_date"] + pd.Timedelta(days=lag)
        d["series"], d["vintage"], d["source"] = s, "latest", "fred"
        if s in ("NFCI", "MORTGAGE30US"):
            d["available_date"] = d["available_date"].clip(lower=pd.Timestamp(date.today()) + pd.Timedelta(days=1))
            d["source"] = "fred_observed_snapshot"
        rows.append(d)
        time.sleep(0.3)
    return pd.concat(rows)


def revised(client, start="2005-01-01"):
    """True initial releases only; sampled legacy caches are never relabelled."""
    from macro_vintages import initial_releases
    key = __import__("provider_access").key("FRED_API_KEY")
    rows = []
    for series in REVISED:
        cache = OUT / f"initial_{series}.parquet"
        if key:
            frame = initial_releases(client, series, key, start)
            temporary = cache.with_suffix(".tmp")
            frame.to_parquet(temporary, index=False)
            temporary.replace(cache)
        elif cache.exists():
            frame = pd.read_parquet(cache)
            if not frame["source"].eq("fred_initial_release_api").all():
                raise ValueError("unverified initial-release cache")
        else:
            # No history is preferable to silently including a revised value.
            print(f"NOT_COMPUTED {series}: FRED_API_KEY or verified initial-release cache required", flush=True)
            continue
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["series", "obs_date", "value", "available_date", "vintage", "source", "pit_eligible"])


def eia(client):
    rows = []
    for code, name in EIA_SERIES.items():
        r = get(client, EIA.format(code))
        x = pd.read_excel(io.BytesIO(r.content), sheet_name="Data 1", skiprows=2)
        x = x.iloc[:, :2]
        x.columns = ["obs_date", "value"]
        x["obs_date"] = pd.to_datetime(x["obs_date"], errors="coerce")
        x["value"] = pd.to_numeric(x["value"], errors="coerce")
        x = x.dropna()
        x["available_date"] = x["obs_date"] + pd.Timedelta(days=5)
        x["series"], x["vintage"], x["source"] = name, "latest", "eia_wpsr"
        x["available_date"] = x["available_date"].clip(lower=pd.Timestamp(date.today()) + pd.Timedelta(days=1))
        x["source"] = "eia_observed_snapshot"
        rows.append(x)
        time.sleep(0.5)
    return pd.concat(rows)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60, headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True) as c:
        parts = [market(c), eia(c), revised(c)]
    df = pd.concat(parts, ignore_index=True)
    if (PROC / "macro_pit.parquet").exists():
        old = pd.read_parquet(PROC / "macro_pit.parquet")
        old = old[old["source"].isin(["fred_observed_snapshot", "eia_observed_snapshot"])]
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(["series", "obs_date", "available_date", "source"], keep="first")
    df.to_parquet(PROC / "macro_pit.parquet", index=False)
    print(df.groupby("series").agg(n=("value", "size"), first=("obs_date", "min"), last=("obs_date", "max")).to_string())


def daily(calendar):
    """Wide frame on the trading calendar: latest value whose available_date <= session date."""
    m = pd.read_parquet(PROC / "macro_pit.parquet")
    # Old sampled ALFRED values are not guaranteed initial releases.
    m = m[~m["source"].eq("alfred_first_release")]
    m = m[~m["source"].eq("eia_wpsr") & ~(m["source"].eq("fred") & m["series"].isin(["NFCI", "MORTGAGE30US"]))]
    out = {}
    for s, g in m.groupby("series"):
        g = g.sort_values("available_date").drop_duplicates("available_date", keep="last")
        idx = g.set_index("available_date")["value"]
        out[s] = idx.reindex(idx.index.union(calendar)).ffill().reindex(calendar)
    return pd.DataFrame(out, index=calendar).reindex(columns=list(dict.fromkeys(MARKET + REVISED + list(EIA_SERIES.values()))))


if __name__ == "__main__":
    run()
