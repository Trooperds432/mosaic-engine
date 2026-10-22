"""Incremental filing updates (spec 12.3: fundamentals refresh "on verified new disclosures").

Reads SEC's daily form index for the last few business days, finds universe companies that filed a 10-Q, 10-K
(or amendments, 20-F/40-F) or an 8-K, and re-downloads only those companies' XBRL facts and 8-K histories, then
rebuilds the merged files. A filing therefore reaches the models the evening it is filed instead of at Friday's
full refresh. Features still apply the filed <= t-1 rule, so nothing becomes visible earlier than before.
"""
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone

import httpx
import pandas as pd

import data_8k
import data_fundamentals as DF
from config import LEDGER, PROC, RAW, SEC_UA

IDX = "https://www.sec.gov/Archives/edgar/daily-index/{y}/QTR{q}/form.{d}.idx"
FACT_FORMS = ("10-Q", "10-K", "10-Q/A", "10-K/A", "10-KT", "20-F", "40-F")
DONE = LEDGER / "filing_updates_done.txt"


def filers(client, day):
    r = client.get(IDX.format(y=day.year, q=(day.month - 1) // 3 + 1, d=day.strftime("%Y%m%d")))
    if r.status_code != 200:
        return None
    out = []
    for line in r.text.splitlines():
        m = re.match(r"^(\S+(?: \S+)?)\s{2,}.*?\s{2,}(\d{3,10})\s+(\d{8})\s+(edgar/\S+)", line)
        if m:
            out.append((m.group(1).strip(), int(m.group(2))))
    return out


def run(days=5):
    ciks = set(pd.read_parquet(PROC / "universe.parquet")["cik"].astype(int))
    done = set(DONE.read_text().split()) if DONE.exists() else set()
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    facts, eights, checked = set(), set(), []
    with httpx.Client(timeout=60, headers={"User-Agent": SEC_UA}) as c:
        today = date.today()
        recent = [today - timedelta(days=k) for k in range(0, 15) if (today - timedelta(days=k)).weekday() < 5][:days + 1]
        for d in recent:
            key = d.isoformat()
            if key in done:
                continue
            rows = filers(c, d)
            time.sleep(0.2)
            if rows is None:  # index not published yet (today before ~22:00 ET, or a holiday)
                continue
            for form, cik in rows:
                if cik in ciks:
                    if form in FACT_FORMS:
                        facts.add(cik)
                    elif form.startswith("8-K"):
                        eights.add(cik)
            checked.append(key)
            if d < today:
                done.add(key)  # a past day's index is final; today's is re-read next run
        print(f"days {checked}: {len(facts)} companies with new 10-Q/10-K, {len(eights)} with new 8-Ks", flush=True)
        for cik in sorted(facts):
            try:
                DF.fetch_cik(cik, c, RAW / "facts" / "by_cik" / f"{cik}.parquet", stamp)
            except Exception as e:
                print("ERR facts", cik, e, flush=True)
            time.sleep(0.15)
        for cik in sorted(eights | facts):  # a periodic report usually comes with an earnings 8-K
            try:
                data_8k.fetch_cik(cik, c, data_8k.OUT / f"{cik}.parquet", recent_only=True)
            except Exception as e:
                print("ERR 8-K", cik, e, flush=True)
            time.sleep(0.15)
    if facts:
        DF.merge(RAW / "facts" / "by_cik")
    if eights or facts:
        data_8k.build_items()
    DONE.write_text("\n".join(sorted(done)) + "\n")
    print(f"updated facts for {len(facts)} and 8-Ks for {len(eights | facts)} companies", flush=True)


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
