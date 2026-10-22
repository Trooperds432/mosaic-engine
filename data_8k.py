"""8-K filing items from SEC submissions (full history incl. paged files). Item 2.02 = results of
operations (earnings release), 2.01 = completed acquisition/disposition. acceptanceDateTime is the
public timestamp; filings accepted after 17:30 ET carry the next business day's filing date."""
import json
import sys
import time

import httpx
import pandas as pd

from config import PROC, RAW, SEC_UA
from data_fundamentals import _get

OUT = RAW / "8k"
COLS = ["accessionNumber", "filingDate", "acceptanceDateTime", "form", "items"]


def accepted_et(ts):
    """True acceptance time in New York time. The submissions API's acceptanceDateTime applies the Eastern offset
    twice (EDGAR index shows 16:30:28 ET, API shows 00:30:28Z next day), so subtract one offset before converting."""
    ts = pd.to_datetime(ts, utc=True)
    off = ts.dt.tz_convert("America/New_York").map(lambda x: x.utcoffset() if pd.notna(x) else pd.NaT)
    return (ts + off).dt.tz_convert("America/New_York")


def frame(block, cik):
    df = pd.DataFrame({k: block.get(k, []) for k in COLS})
    df = df[df["form"].isin(["8-K", "8-K/A"])].copy()
    df["cik"] = cik
    return df


def run(refresh=False):
    OUT.mkdir(parents=True, exist_ok=True)
    uni = pd.read_parquet(PROC / "universe.parquet")
    errors = 0
    with httpx.Client(timeout=60, headers={"User-Agent": SEC_UA}) as c:
        for i, cik in enumerate(uni["cik"].astype(int)):
            dest = OUT / f"{cik}.parquet"
            if dest.exists() and not refresh:
                continue
            try:
                fetch_cik(cik, c, dest)
            except Exception as e:
                errors += 1
                print("ERR", cik, e, flush=True)
            time.sleep(0.12)
            if i % 100 == 0:
                print(i, flush=True)
    build_items(errors)


def fetch_cik(cik, client, dest, recent_only=False):
    """One company's 8-K history (all pages, or only the recent page merged into the existing file)."""
    j = _get(client, f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
    parts = [frame(j["filings"]["recent"], cik)]
    if recent_only and dest.exists():
        parts.append(pd.read_parquet(dest)[COLS + ["cik"]].astype({"filingDate": str, "acceptanceDateTime": str}))
    else:
        for f in j["filings"].get("files", []):
            time.sleep(0.12)
            parts.append(frame(_get(client, f"https://data.sec.gov/submissions/{f['name']}").json(), cik))
    df = pd.concat(parts, ignore_index=True).drop_duplicates("accessionNumber")
    df["filingDate"] = pd.to_datetime(df["filingDate"])
    df["acceptanceDateTime"] = pd.to_datetime(df["acceptanceDateTime"], utc=True, errors="coerce")
    df.to_parquet(dest, index=False)


def build_items(errors=0):
    allk = pd.concat([pd.read_parquet(f) for f in sorted(OUT.glob("*.parquet"))], ignore_index=True)
    allk["accepted_et"] = accepted_et(allk["acceptanceDateTime"])
    # public at the session of acceptance if accepted before the 16:00 ET close, otherwise the next session
    allk["available_date"] = allk["accepted_et"].dt.tz_localize(None).dt.normalize() + pd.to_timedelta((allk["accepted_et"].dt.hour >= 16).astype(int), unit="D")
    allk.to_parquet(PROC / "8k_items.parquet", index=False)
    n202 = int(allk["items"].fillna("").str.contains("2.02").sum())
    (PROC / "8k_audit.json").write_text(json.dumps({"rows": len(allk), "ciks": int(allk["cik"].nunique()),
                                                    "item_2_02": n202, "errors": errors}, indent=1))
    print("done", len(allk), "item 2.02:", n202, "errors:", errors)


if __name__ == "__main__":
    run(refresh="--refresh" in sys.argv)
