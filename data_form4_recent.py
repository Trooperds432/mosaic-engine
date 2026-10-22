"""Form 4 filings newer than the latest SEC quarterly insider dataset, parsed from EDGAR XML into
the same columns as data_insider.py so historical and live features share one definition."""
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import httpx
import pandas as pd

from config import PROC, RAW, SEC_UA

OUT = RAW / "insider_recent" / "form4_recent.parquet"
SEEN = RAW / "insider_recent" / "seen_accessions.txt"  # includes filings with no non-derivative rows


def coverage_end():
    qs = sorted(p.stem for p in (RAW / "insider").glob("*.parquet"))
    y, q = map(int, qs[-1].split("q"))
    return pd.Timestamp(year=y, month=3 * q, day=1) + pd.offsets.MonthEnd(0)


def _t(node, path):
    x = node.find(path)
    return x.text.strip() if x is not None and x.text else None


def parse(xml, acc, filed, cik):
    root = ET.fromstring(xml)
    own = root.find("reportingOwner")
    rel = own.find("reportingOwnerRelationship") if own is not None else None
    flags = []
    if rel is not None:
        for tag, name in [("isDirector", "Director"), ("isOfficer", "Officer"), ("isTenPercentOwner", "TenPercentOwner"), ("isOther", "Other")]:
            if (_t(rel, tag) or "").lower() in ("1", "true"):
                flags.append(name)
    base = {"accession_number": acc, "filing_date": filed, "issuercik": cik,
            "issuertradingsymbol": _t(root, "issuer/issuerTradingSymbol"), "document_type": _t(root, "documentType"),
            "period_of_report": _t(root, "periodOfReport"), "aff10b5one": _t(root, "aff10b5One"),
            "rptownercik": _t(own, "reportingOwnerId/rptOwnerCik") if own is not None else None,
            "rptownername": _t(own, "reportingOwnerId/rptOwnerName") if own is not None else None,
            "rptowner_relationship": ",".join(flags), "rptowner_title": _t(rel, "officerTitle") if rel is not None else None}
    notes = {f.get("id"): (f.text or "") for f in root.findall("footnotes/footnote")}
    from data_insider import DATE_RE, PLAN_RE
    rows = []
    for tr in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        txt = " | ".join(notes.get(x.get("id"), "") for x in tr.iter("footnoteId"))
        plan = bool(PLAN_RE.search(txt))
        adopted = DATE_RE.search(txt) if plan else None
        rows.append({**base, "trans_date": _t(tr, "transactionDate/value"), "trans_code": _t(tr, "transactionCoding/transactionCode"),
                     "trans_shares": _t(tr, "transactionAmounts/transactionShares/value"),
                     "trans_pricepershare": _t(tr, "transactionAmounts/transactionPricePerShare/value"),
                     "trans_acquired_disp_cd": _t(tr, "transactionAmounts/transactionAcquiredDisposedCode/value"),
                     "shrs_ownd_folwng_trans": _t(tr, "postTransactionAmounts/sharesOwnedFollowingTransaction/value"),
                     "direct_indirect_ownership": _t(tr, "ownershipNature/directOrIndirectOwnership/value"),
                     "footnote_txt": txt[:600], "fn_10b5_1": plan, "plan_adopted": adopted.group(1) if adopted else None})
    return rows


def _get(client, url, tries=6):
    for i in range(tries):
        try:
            r = client.get(url)
            if r.status_code in (429, 500, 502, 503):
                raise httpx.HTTPError(f"status {r.status_code}")
            return r
        except httpx.HTTPError:
            time.sleep(2 ** i)
    raise RuntimeError(f"gave up on {url}")


def _frame(rows):
    new = pd.DataFrame(rows)
    if len(new):
        for col in ["trans_shares", "trans_pricepershare", "shrs_ownd_folwng_trans"]:
            new[col] = pd.to_numeric(new[col], errors="coerce")
        for col in ["filing_date", "trans_date", "period_of_report"]:
            new[col] = pd.to_datetime(new[col], errors="coerce")
        if "plan_adopted" in new:
            new["plan_adopted"] = pd.to_datetime(new["plan_adopted"], errors="coerce", format="mixed")
        new["issuercik"] = new["issuercik"].astype("Int64")
        new["dataset"] = "edgar_form4_xml"
        new["ingested_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return new


def _save(old, rows):
    new = _frame(rows)
    allr = pd.concat([old, new], ignore_index=True) if len(old) else new
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if len(allr):
        allr.to_parquet(OUT, index=False)
    return allr


def _mark(accs):
    if accs:
        with SEEN.open("a") as f:
            f.write("".join(a + "\n" for a in accs))


def run():
    since = coverage_end()
    uni = pd.read_parquet(PROC / "universe.parquet")
    old = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame()
    seen = set(old["accession_number"]) if len(old) else set()
    if SEEN.exists():
        seen |= set(SEEN.read_text().split())
    fetched = []
    old = old[old["filing_date"] > since] if len(old) else old
    rows, n_new = [], 0
    with httpx.Client(timeout=30, headers={"User-Agent": SEC_UA}) as c:
        for i, cik in enumerate(uni["cik"].astype(int)):
            if i and i % 50 == 0:  # checkpoint: a restart skips accessions already saved
                old, rows = _save(old, rows), []
                _mark(fetched)
                fetched = []
                print(i, n_new, flush=True)
            try:
                rec = _get(c, f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()["filings"]["recent"]
            except Exception as e:
                print("ERR submissions", cik, e, flush=True)
                continue
            time.sleep(0.11)
            for form, fd, acc, doc in zip(rec["form"], rec["filingDate"], rec["accessionNumber"], rec["primaryDocument"]):
                if form not in ("4", "4/A") or pd.Timestamp(fd) <= since or acc in seen:
                    continue
                url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc.split('/')[-1]}"
                try:
                    r = _get(c, url)
                    if r.status_code == 200:
                        rows += parse(r.content, acc, fd, cik)
                        fetched.append(acc)
                        n_new += 1
                except Exception as e:
                    print("ERR form4", url, e, flush=True)
                time.sleep(0.11)
    allr = _save(old, rows)
    _mark(fetched)
    print(f"form4 recent: coverage after {since.date()}, new filings={n_new}, rows total={len(allr)}")


if __name__ == "__main__":
    run()
