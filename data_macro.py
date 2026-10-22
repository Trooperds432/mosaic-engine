"""Macro, rates, credit and energy fundamentals with point-in-time availability dates.

  market series (FRED, never revised): available the session after the observation date
  revised series (ALFRED): value as published in twice-monthly vintages since 2005; an observation is
      available from the first vintage that contains it (conservative: real release can be earlier)
  EIA weekly petroleum status (week ending Friday, released Wednesday): available week end + 5 days
Output: processed/macro_pit.parquet (series, obs_date, value, available_date, vintage, source).
"""
import io
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
        rows.append(d)
        time.sleep(0.3)
    return pd.concat(rows)


def revised(client, start="2005-01-01"):
    vint = [d for d in pd.date_range(start, date.today(), freq="SMS")]
    rows = []
    for s in REVISED:
        cache = OUT / f"alfred_{s}.parquet"
        have = pd.read_parquet(cache) if cache.exists() else pd.DataFrame()
        done = set(have["vintage"]) if len(have) else set()
        def one(vs):
            try:
                d = pd.read_csv(io.StringIO(get(client, ALFRED.format(s, vs)).text))
            except Exception:
                return None
            if d.shape[1] < 2:
                return None
            d.columns = ["obs_date", "value"]
            d["value"] = pd.to_numeric(d["value"], errors="coerce")
            d = d.dropna().tail(24)  # recent observations as known at this vintage
            d["vintage"] = vs
            return d
        todo = [v.strftime("%Y-%m-%d") for v in vint if v.strftime("%Y-%m-%d") not in done]
        with ThreadPoolExecutor(6) as ex:  # ALFRED answers slowly (~4s); a few concurrent requests
            new = [d for d in ex.map(one, todo) if d is not None]
        allv = pd.concat([have] + new, ignore_index=True) if new else have
        allv.to_parquet(cache, index=False)
        allv["obs_date"] = pd.to_datetime(allv["obs_date"])
        allv["vdate"] = pd.to_datetime(allv["vintage"])
        # first-release value: the earliest vintage that contains each observation
        first = allv.sort_values("vdate").drop_duplicates("obs_date", keep="first")
        first = first.assign(available_date=first["vdate"], series=s, source="alfred_first_release")
        rows.append(first[["obs_date", "value", "available_date", "series", "vintage", "source"]])
        print(s, len(first), "observations", flush=True)
    return pd.concat(rows)


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
        rows.append(x)
        time.sleep(0.5)
    return pd.concat(rows)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60, headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True) as c:
        parts = [market(c), eia(c), revised(c)]
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(PROC / "macro_pit.parquet", index=False)
    print(df.groupby("series").agg(n=("value", "size"), first=("obs_date", "min"), last=("obs_date", "max")).to_string())


def daily(calendar):
    """Wide frame on the trading calendar: latest value whose available_date <= session date."""
    m = pd.read_parquet(PROC / "macro_pit.parquet")
    out = {}
    for s, g in m.groupby("series"):
        g = g.sort_values("available_date").drop_duplicates("available_date", keep="last")
        idx = g.set_index("available_date")["value"]
        out[s] = idx.reindex(idx.index.union(calendar)).ffill().reindex(calendar)
    return pd.DataFrame(out, index=calendar)


if __name__ == "__main__":
    run()
