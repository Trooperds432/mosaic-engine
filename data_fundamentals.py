"""As-reported fundamentals from SEC XBRL companyfacts, plus SIC industry codes.
Every value keeps its `filed` date: a value is usable at t only if filed before t, and when a
period is restated the original and restated filings are both kept (as-reported vs latest)."""
import json
import time
from datetime import datetime, timezone

import httpx
import pandas as pd

from config import PROC, RAW, SEC_UA

TAGS = {
    "dei": ["EntityCommonStockSharesOutstanding"],
    "us-gaap": ["NetIncomeLoss", "StockholdersEquity", "Assets", "GrossProfit", "Revenues",
                "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "CostOfRevenue",
                "CostOfGoodsAndServicesSold", "OperatingIncomeLoss", "NetCashProvidedByUsedInOperatingActivities",
                "PaymentsToAcquirePropertyPlantAndEquipment", "WeightedAverageNumberOfDilutedSharesOutstanding",
                # earnings surprise (SUE), sector modules 4A-4E, quality and leverage
                "EarningsPerShareDiluted", "EarningsPerShareBasic", "InventoryNet", "CashAndCashEquivalentsAtCarryingValue",
                "ResearchAndDevelopmentExpense", "LongTermDebt", "LongTermDebtNoncurrent", "LiabilitiesCurrent", "AssetsCurrent",
                "Liabilities", "InterestExpense", "SellingGeneralAndAdministrativeExpense", "DepreciationDepletionAndAmortization",
                "IncomeTaxExpenseBenefit", "Deposits", "InterestIncomeExpenseNet", "LoansAndLeasesReceivableNetReportedAmount",
                "ProvisionForLoanLeaseAndOtherLosses", "ProvisionForLoanAndLeaseLosses", "ProvisionForCreditLosses"],
}
FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "40-F", "10-KT"}


def facts(cik, r):
    j = r.json().get("facts", {})
    rows = []
    for ns, tags in TAGS.items():
        for tag in tags:
            for unit, obs in j.get(ns, {}).get(tag, {}).get("units", {}).items():
                for o in obs:
                    if o.get("form") in FORMS:
                        rows.append({"cik": cik, "tag": tag, "unit": unit, "start": o.get("start"), "end": o["end"],
                                     "val": o["val"], "filed": o["filed"], "form": o["form"], "fy": o.get("fy"),
                                     "fp": o.get("fp"), "accn": o["accn"]})
    return rows


def _get(client, url, tries=6):
    """SEC endpoints intermittently drop TLS connections; retry with backoff."""
    for i in range(tries):
        try:
            r = client.get(url)
            if r.status_code in (429, 500, 502, 503):
                raise httpx.HTTPError(f"status {r.status_code}")
            return r
        except httpx.HTTPError:
            time.sleep(2 ** i)
    raise RuntimeError(f"gave up on {url}")


def fetch_cik(cik, client, dest, stamp):
    """Download one company's XBRL companyfacts into its per-CIK parquet."""
    r = _get(client, f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
    df = pd.DataFrame(facts(cik, r) if r.status_code == 200 else [],
                      columns=["cik", "tag", "unit", "start", "end", "val", "filed", "form", "fy", "fp", "accn"])
    df["fy"] = pd.to_numeric(df["fy"], errors="coerce").astype("Int64")
    for col in ["start", "end", "filed"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["val"] = pd.to_numeric(df["val"], errors="coerce").astype("float64")
    df["ingested_utc"] = stamp
    df.to_parquet(dest, index=False)


def merge(per):
    """Stream the per-CIK files into one parquet (concatenating ~6,000 frames in pandas exhausts 3 GB RAM)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    files = sorted(per.glob("*.parquet"))
    ts = pa.timestamp("us")  # some filings carry garbage dates outside the ns range
    schema = pa.schema([("cik", pa.int64()), ("tag", pa.string()), ("unit", pa.string()), ("start", ts), ("end", ts),
                        ("val", pa.float64()), ("filed", ts), ("form", pa.string()), ("fy", pa.int64()), ("fp", pa.string()),
                        ("accn", pa.string()), ("ingested_utc", pa.string())])
    rows = 0
    with pq.ParquetWriter(RAW / "facts" / "companyfacts.parquet", schema) as w:
        for f in files:
            t = pq.read_table(f)
            if t.num_rows:
                t = t.select([c for c in schema.names if c in t.column_names])
                w.write_table(pa.table({c: (t[c].cast(schema.field(c).type) if c in t.column_names else pa.nulls(t.num_rows, schema.field(c).type))
                                        for c in schema.names}, schema=schema))
                rows += t.num_rows
    return pd.DataFrame({"rows": [rows]}).assign(cik=0)


def run(refresh=False, merge_only=False):
    """Caches one parquet per CIK so an interrupted run resumes; `refresh` re-downloads."""
    uni = pd.read_parquet(PROC / "universe.parquet")
    per = RAW / "facts" / "by_cik"
    per.mkdir(parents=True, exist_ok=True)
    sic_f = PROC / "sic.parquet"
    sic = pd.read_parquet(sic_f).to_dict("records") if sic_f.exists() and not refresh else []
    have_sic = {r["cik"] for r in sic}
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    errors = 0
    with httpx.Client(timeout=60, headers={"User-Agent": SEC_UA}) as c:
        for i, cik in enumerate(uni["cik"].astype(int)):
            dest = per / f"{cik}.parquet"
            try:
                if refresh or not dest.exists():
                    fetch_cik(cik, c, dest, stamp)
                    time.sleep(0.12)
                if cik not in have_sic:
                    s = _get(c, f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
                    if s.status_code == 200:
                        sj = s.json()
                        sic.append({"cik": cik, "sic": sj.get("sic"), "sic_desc": sj.get("sicDescription"),
                                    "exchanges": ",".join(e or "" for e in sj.get("exchanges", []))})
                        if len(sic) % 200 == 0:  # checkpoint so an interrupted run keeps what it fetched
                            pd.DataFrame(sic).to_parquet(sic_f, index=False)
                    time.sleep(0.3)  # shares SEC's 10 req/s budget with the Form 4 downloader
            except Exception as e:
                errors += 1
                print("ERR", cik, e, flush=True)
            if i % 100 == 0:
                print(i, flush=True)
    pd.DataFrame(sic).to_parquet(sic_f, index=False)
    df = merge(per)
    (PROC / "facts_audit.json").write_text(json.dumps({"rows": int(df["rows"].iloc[0]), "ciks": len(list(per.glob("*.parquet"))),
                                                       "sic_rows": len(sic), "errors": errors}, indent=1))
    print("done", int(df["rows"].iloc[0]))


if __name__ == "__main__":
    import sys
    run(refresh="--refresh" in sys.argv)
