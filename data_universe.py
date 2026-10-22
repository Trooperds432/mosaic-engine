"""Build the candidate pool: current and historical S&P 500 members, current NASDAQ-100 and all NYSE/Nasdaq-listed
common stocks,
mapped to SEC CIKs. Index membership only defines the candidate pool; eligibility on any date is
decided point-in-time from market cap and volume."""
import io
import json

import httpx
import pandas as pd

from config import HISTORY_START, PROC, SEC_UA

HIST_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
WIKI = {"sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "ndx": "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"}


def _tables(url):
    r = httpx.get(url, headers={"User-Agent": SEC_UA}, timeout=60, follow_redirects=True)
    r.raise_for_status()
    return pd.read_html(io.StringIO(r.text))


def build():
    from historical_data import active_bundle
    if active_bundle():
        raise ValueError("historical bundle is active: update it with historical_data.py, not the current-listing collector")
    sp = _tables(WIKI["sp500"])
    cur = sp[0][["Symbol", "Security", "GICS Sector"]].rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"})
    cur["source"] = "sp500_current"
    # Historical membership (fja05680/sp500, compiled from S&P change announcements)
    hist = pd.read_csv(HIST_URL, parse_dates=["start_date", "end_date"])
    removed = hist[hist["end_date"].notna() & (hist["end_date"] >= HISTORY_START)][["ticker", "end_date"]]
    removed = removed.rename(columns={"end_date": "removed_on"})
    removed["source"] = "sp500_removed"

    ndx_tbl = next(t for t in _tables(WIKI["ndx"]) if "Ticker" in t.columns or "Symbol" in t.columns)
    tcol = "Ticker" if "Ticker" in ndx_tbl.columns else "Symbol"
    ndx = pd.DataFrame({"ticker": ndx_tbl[tcol], "source": "ndx_current"})

    # every NYSE/Nasdaq-listed operating company today (covers the Russell 2000 range; whether a name is
    # eligible on a date is decided point-in-time by market cap and liquidity, not by this list)
    ex = httpx.get("https://www.sec.gov/files/company_tickers_exchange.json", headers={"User-Agent": SEC_UA}, timeout=60).json()
    ex = pd.DataFrame(ex["data"], columns=ex["fields"])
    ex = ex[ex["exchange"].isin(["NYSE", "Nasdaq"])]
    t = ex["ticker"].astype(str).str.upper()
    suffix = t.str.split("-").str[1].fillna("")
    common = (~t.str.contains("-") | suffix.isin(["A", "B", "C"])) & \
        ~((t.str.len() == 5) & t.str[-1].isin(["W", "U", "R"]) & (ex["exchange"] == "Nasdaq"))
    listed = pd.DataFrame({"ticker": t[common], "source": "listed_current"})

    allp = pd.concat([cur, removed, ndx, listed], ignore_index=True)
    allp["ticker"] = allp["ticker"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)  # SEC style: BRK-B

    sec = httpx.get("https://www.sec.gov/files/company_tickers.json", headers={"User-Agent": SEC_UA}, timeout=60).json()
    sec = pd.DataFrame(sec.values()).rename(columns={"cik_str": "cik", "title": "sec_name"})
    sec["ticker"] = sec["ticker"].str.upper()

    pool = allp.groupby("ticker").agg(sources=("source", lambda s: ",".join(sorted(set(s)))),
                                      sector=("sector", "first"), removed_on=("removed_on", "max")).reset_index()
    pool = pool.merge(sec, on="ticker", how="left")
    pool["yahoo"] = pool["ticker"].str.replace(".", "-", regex=False)
    audit = {
        "candidates": int(len(pool)),
        "listed_current": int(pool["sources"].str.contains("listed_current").sum()),
        "sp500_removed_since_start": int(pool["sources"].str.contains("sp500_removed").sum()),
        "removed_without_current_sec_ticker": int((pool["sources"].str.contains("sp500_removed") & pool["cik"].isna()).sum()),
    }
    pool = pool[pool["cik"].notna()].copy()
    pool["cik"] = pool["cik"].astype(int)
    # one security per issuer, preferring index-sourced tickers, then SEC's primary (first-listed) ticker
    pool["pri"] = (~pool["sources"].str.contains("sp500|ndx")).astype(int)
    pool = pool.sort_values(["pri"], kind="stable").drop_duplicates("cik", keep="first").drop(columns="pri")
    audit["kept_with_cik"] = int(len(pool))
    PROC.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(PROC / "universe.parquet", index=False)
    (PROC / "universe_audit.json").write_text(json.dumps(audit, indent=1))
    print(audit)


if __name__ == "__main__":
    build()
