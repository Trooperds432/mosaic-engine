"""Alternative / sector datasets, each stored with a point-in-time availability date.

  futures   Yahoo continuous front-month futures (CL, NG, RB, HO, GC, SI, HG, ZC, ZW, ZS, KC, SB, CT): daily
            close, available the next session. Continuous series roll-splice: returns across rolls include
            the roll gap (term structure not observable from this source).
  cot       CFTC disaggregated futures-only Commitments of Traders: managed-money and producer net positions
            as of Tuesday, released Friday 15:30 ET -> available report date + 3 days.
  short     FINRA consolidated short interest (exchange-listed, 2020+), settlement date -> FINRA publishes
            ~8 business days later; available settlement + 12 calendar days (conservative).
  trials    ClinicalTrials.gov v2 studies whose lead sponsor matches a universe health-care company: phase,
            status, start / primary-completion / first-posted / results-posted dates. Registry rows are
            edited in place. Archive each observed snapshot; historical phase features
            only use snapshots acquired before the decision date.
  fda       openFDA drugs@FDA submissions (approvals) by sponsor: action date = available date.
"""
import json
import re
import sys
import time
from datetime import datetime, timezone
from uuid import uuid4

import httpx
import pandas as pd

from config import PROC, RAW

OUT = RAW / "alt"
FUTURES = {"CL=F": "wti", "NG=F": "natgas", "RB=F": "rbob", "HO=F": "heating_oil", "GC=F": "gold", "SI=F": "silver",
           "HG=F": "copper", "ZC=F": "corn", "ZW=F": "wheat", "ZS=F": "soybeans", "KC=F": "coffee", "SB=F": "sugar", "CT=F": "cotton"}
COT = {"067651": "wti", "023651": "natgas", "111659": "rbob", "022651": "heating_oil", "088691": "gold", "084691": "silver",
       "085692": "copper", "002602": "corn", "001602": "wheat", "005602": "soybeans", "083731": "coffee", "080732": "sugar", "033661": "cotton"}
SUFFIX = re.compile(r"\b(INC|CORP|CORPORATION|CO|COMPANY|LTD|LIMITED|PLC|HOLDINGS?|GROUP|N\.?V|S\.?A|AG|SE|LLC|LP|THE|CLASS [A-Z])\b\.?")


def norm(name):
    n = re.sub(r"[^A-Z0-9 ]", " ", str(name).upper().replace("&", " AND "))
    return re.sub(r"\s+", " ", SUFFIX.sub(" ", n)).strip()


def get(c, url, **kw):
    for i in range(5):
        try:
            r = c.get(url, **kw)
            if r.status_code == 200:
                return r
            if r.status_code in (400, 404):
                return r
        except httpx.HTTPError:
            pass
        time.sleep(2 ** i)
    raise RuntimeError(url)


def futures(c):
    rows = []
    for sym, name in FUTURES.items():
        j = get(c, f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
                params={"period1": 946684800, "period2": int(time.time()), "interval": "1d"}).json()["chart"]["result"][0]
        d = pd.DataFrame({"date": pd.to_datetime(j["timestamp"], unit="s").normalize(), "close": j["indicators"]["quote"][0]["close"]}).dropna()
        d["commodity"] = name
        d["available_date"] = d["date"] + pd.Timedelta(days=1)
        rows.append(d)
        time.sleep(0.4)
    df = pd.concat(rows)
    df.to_parquet(OUT / "futures.parquet", index=False)
    return f"futures {len(df)} rows"


def cot(c):
    codes = ",".join(f"'{k}'" for k in COT)
    rows, off = [], 0
    while True:
        r = get(c, "https://publicreporting.cftc.gov/resource/72hh-3qpy.json", params={
            "$where": f"cftc_contract_market_code in({codes})", "$limit": 50000, "$offset": off,
            "$select": "report_date_as_yyyy_mm_dd,cftc_contract_market_code,open_interest_all,m_money_positions_long_all,"
                       "m_money_positions_short_all,prod_merc_positions_long,prod_merc_positions_short"}).json()
        rows += r
        if len(r) < 50000:
            break
        off += 50000
    d = pd.DataFrame(rows)
    d["report_date"] = pd.to_datetime(d["report_date_as_yyyy_mm_dd"])
    for k in ["open_interest_all", "m_money_positions_long_all", "m_money_positions_short_all", "prod_merc_positions_long", "prod_merc_positions_short"]:
        d[k] = pd.to_numeric(d[k], errors="coerce")
    d["commodity"] = d["cftc_contract_market_code"].map(COT)
    d["mm_net_pct_oi"] = (d["m_money_positions_long_all"] - d["m_money_positions_short_all"]) / d["open_interest_all"]
    d["prod_net_pct_oi"] = (d["prod_merc_positions_long"] - d["prod_merc_positions_short"]) / d["open_interest_all"]
    d["available_date"] = d["report_date"] + pd.Timedelta(days=3)
    d = d.groupby(["commodity", "report_date"], as_index=False).first()
    d.to_parquet(OUT / "cot.parquet", index=False)
    return f"cot {len(d)} rows {d['report_date'].min().date()}..{d['report_date'].max().date()}"


def short_interest(c):
    uni = pd.read_parquet(PROC / "universe.parquet")
    tick = set(uni["ticker"].str.replace("-", ".", regex=False)) | set(uni["ticker"])
    dest = OUT / "short_interest"
    dest.mkdir(exist_ok=True)
    url = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
    dates = pd.date_range("2020-01-01", pd.Timestamp.today(), freq="SMS")
    n = 0
    for d0 in dates:
        # settlement dates are mid-month and month-end business days; probe the nearby days
        for d in [d0 + pd.Timedelta(days=k) for k in range(-3, 14)]:
            f = dest / f"{d.date()}.parquet"
            if f.exists():
                break
            rows, off = [], 0
            while True:
                r = c.post(url, json={"limit": 5000, "offset": off, "compareFilters": [
                    {"compareType": "EQUAL", "fieldName": "settlementDate", "fieldValue": str(d.date())}]},
                           headers={"Accept": "application/json"})
                if r.status_code != 200 or not r.text.strip():
                    break
                page = r.json()
                rows += page
                if len(page) < 5000:
                    break
                off += 5000
            if rows:
                x = pd.DataFrame(rows)
                x = x[x["symbolCode"].isin(tick)]
                x.to_parquet(f, index=False)
                n += 1
                break
            time.sleep(0.2)
    files = sorted(dest.glob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["settlement_date"] = pd.to_datetime(df["settlementDate"])
    df["available_date"] = df["settlement_date"] + pd.Timedelta(days=12)
    keep = ["symbolCode", "settlement_date", "available_date", "currentShortPositionQuantity", "averageDailyVolumeQuantity", "daysToCoverQuantity"]
    df[keep].to_parquet(PROC / "short_interest.parquet", index=False)
    return f"short interest {len(df)} rows over {len(files)} settlement dates"


def trials(c):
    uni = pd.read_parquet(PROC / "universe.parquet")
    sic = pd.read_parquet(PROC / "sic.parquet").set_index("cik")["sic"]
    hc = uni[pd.to_numeric(uni["cik"].map(sic), errors="coerce").between(2830, 2836) | pd.to_numeric(uni["cik"].map(sic), errors="coerce").between(8730, 8734)]
    rows = []
    fields = "NCTId,LeadSponsorName,Phase,OverallStatus,StartDate,PrimaryCompletionDate,StudyFirstPostDate,ResultsFirstPostDate,Condition,InterventionType"
    for cik, name in zip(hc["cik"], hc["sec_name"]):
        q = norm(name)
        if len(q) < 3:
            continue
        token = None
        while True:
            params = {"query.lead": q, "pageSize": 1000, "fields": fields}
            if token:
                params["pageToken"] = token
            r = get(c, "https://clinicaltrials.gov/api/v2/studies", params=params)
            if r.status_code != 200 or not r.text.startswith("{"):
                break  # unparseable sponsor query: skip this company
            j = r.json()
            for s in j.get("studies", []):
                p = s.get("protocolSection", {})
                ident, stat, design = p.get("identificationModule", {}), p.get("statusModule", {}), p.get("designModule", {})
                sponsor = p.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name", "")
                if not norm(sponsor).startswith(q.split(" ")[0]):
                    continue
                rows.append({"cik": int(cik), "nct": ident.get("nctId"), "sponsor": sponsor, "phase": ",".join(design.get("phases", []) or []),
                             "status": stat.get("overallStatus"), "start": (stat.get("startDateStruct") or {}).get("date"),
                             "primary_completion": (stat.get("primaryCompletionDateStruct") or {}).get("date"),
                             "first_posted": (stat.get("studyFirstPostDateStruct") or {}).get("date"),
                             "results_posted": (stat.get("resultsFirstPostDateStruct") or {}).get("date"),
                             "conditions": ";".join(p.get("conditionsModule", {}).get("conditions", [])[:5])})
            token = j.get("nextPageToken")
            if not token:
                break
        time.sleep(0.2)
    d = pd.DataFrame(rows)
    for k in ["start", "primary_completion", "first_posted", "results_posted"]:
        d[k] = pd.to_datetime(d[k], errors="coerce", format="mixed")
    if d.empty:
        raise ValueError("empty clinical registry response; preserve prior data")
    observed = datetime.now(timezone.utc)
    snapshot_id = observed.strftime("%Y%m%dT%H%M%S%f") + "_" + uuid4().hex[:8]
    d["observed_at_utc"] = observed.isoformat()
    d["snapshot_id"] = snapshot_id
    archive = OUT / "clinical_snapshots"
    archive.mkdir(parents=True, exist_ok=True)
    d.to_parquet(archive / f"{snapshot_id}.parquet", index=False)
    d.to_parquet(PROC / "clinical_trials.parquet", index=False)
    return f"trials {len(d)} studies for {d['cik'].nunique()} companies"


def fda(c):
    uni = pd.read_parquet(PROC / "universe.parquet")
    sic = pd.read_parquet(PROC / "sic.parquet").set_index("cik")["sic"]
    hc = uni[pd.to_numeric(uni["cik"].map(sic), errors="coerce").between(2830, 2836)]
    rows = []
    for cik, name in zip(hc["cik"], hc["sec_name"]):
        q = norm(name).split(" ")[0]
        if len(q) < 4:
            continue
        r = get(c, "https://api.fda.gov/drug/drugsfda.json", params={"search": f'sponsor_name:"{q}"', "limit": 1000})
        if r.status_code != 200:
            continue
        for app in r.json().get("results", []):
            for s in app.get("submissions", []):
                if s.get("submission_status") == "AP":
                    rows.append({"cik": int(cik), "sponsor": app.get("sponsor_name"), "application": app.get("application_number"),
                                 "type": s.get("submission_type"), "class": (s.get("submission_class_code") or ""),
                                 "action_date": s.get("submission_status_date")})
        time.sleep(0.3)
    d = pd.DataFrame(rows)
    d["action_date"] = pd.to_datetime(d["action_date"], format="%Y%m%d", errors="coerce")
    d.to_parquet(PROC / "fda_approvals.parquet", index=False)
    return f"fda {len(d)} approvals for {d['cik'].nunique()} companies"


def run(parts):
    OUT.mkdir(parents=True, exist_ok=True)
    fns = {"futures": futures, "cot": cot, "short": short_interest, "trials": trials, "fda": fda}
    with httpx.Client(timeout=120, headers={"User-Agent": "Mozilla/5.0"}) as c:
        for p in parts or list(fns):
            try:
                print(p, fns[p](c), flush=True)
            except Exception as e:
                print(p, "ERROR", repr(e)[:300], flush=True)


if __name__ == "__main__":
    run(sys.argv[1:])
