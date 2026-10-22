"""Event-driven insider updates (spec 12.3): poll EDGAR's live Form 4 feed during the US day, parse new filings
for universe issuers with the same parser as data_form4_recent.py, and refresh only the insider-event signals in
signals/latest.json. Cross-sectional signals keep their daily cadence; fundamentals are not refreshed here.
Never places orders. Every refresh is also written to signals/log/ (append-only).
"""
import fcntl
import json
import re
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

import config as C
from data_form4_recent import OUT, SEEN, _frame, _mark, parse

FEED = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4&company=&dateb=&owner=include&start={}&count=100&output=atom"
NY = ZoneInfo("America/New_York")
POLL_SECONDS = 600


def poll(client, ciks, seen):
    """New (accession, issuer cik, filed date) entries for universe issuers, newest first."""
    new = []
    for start in (0, 100, 200, 300):
        t = client.get(FEED.format(start)).text
        entries = re.findall(r"<title>4(?:/A)? - .*?\((\d{10})\) \(Issuer\)</title>.*?AccNo:&lt;/b&gt; ([\d-]+).*?<updated>(\d{4}-\d{2}-\d{2})", t, re.S)
        if not entries:
            break
        stop = False
        for cik, acc, d in entries:
            if acc in seen:
                stop = True
                continue
            if int(cik) in ciks:
                new.append((acc, int(cik), d))
        if stop:
            break
        time.sleep(0.5)
    return new


def fetch_xml(client, cik, acc):
    j = client.get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/index.json").json()
    xml = next((f["name"] for f in j["directory"]["item"] if f["name"].endswith(".xml") and not f["name"].startswith("xsl")), None)
    return client.get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{xml}").content if xml else None


def refresh_insider_signals():
    import insider_model
    import regime
    from signals import iso, session_close_utc, dataset_version
    f = C.SIGNALS / "latest.json"
    if not f.exists():
        return 0
    doc = json.loads(f.read_text())
    asof_day = pd.Timestamp(doc["as_of_utc"][:10])
    gen = datetime.now(timezone.utc)
    stale = (pd.Timestamp(gen.astimezone(NY).date()) - asof_day).days > 5
    sigs = insider_model.live_signals(asof_day, gen, dataset_version(), not stale, doc.get("data_audit_passed", False))
    reg = regime.field()
    for s in sigs:
        s["regime"] = reg
    keep = [s for s in doc["signals"] if s["model_id"] != "insider-event"]
    doc["signals"] = sorted(keep + sigs, key=lambda s: -(s["forecast"]["expected_net_excess_return"] or -9))
    doc["n_insider_event_signals"] = len(sigs)
    doc["n_long"] = sum(s["signal"]["direction"] == "LONG" for s in doc["signals"])
    doc["insider_refreshed_at_utc"] = iso(gen)
    (C.SIGNALS / "log" / f"intraday_{gen.strftime('%Y%m%dT%H%M%S')}.jsonl").write_text("".join(json.dumps(s) + "\n" for s in sigs))
    tmp = C.SIGNALS / "latest.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(f)
    return len(sigs)


def main():
    ciks = set(pd.read_parquet(C.PROC / "universe.parquet")["cik"].astype(int))
    with httpx.Client(timeout=30, headers={"User-Agent": C.SEC_UA}) as client:
        while True:
            now = datetime.now(NY)
            lock = open(C.ROOT / ".lock", "a")
            try:  # same lock as run_daily.sh: never write the Form 4 file while the daily job does
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except OSError:
                locked = False
            if locked and now.weekday() < 5 and 6 <= now.hour < 22:
                try:
                    seen = set(SEEN.read_text().split()) if SEEN.exists() else set()
                    if OUT.exists():
                        seen |= set(pd.read_parquet(OUT, columns=["accession_number"])["accession_number"])
                    rows, got = [], []
                    for acc, cik, d in poll(client, ciks, seen):
                        xml = fetch_xml(client, cik, acc)
                        if xml:
                            rows += parse(xml, acc, d, cik)
                            got.append(acc)
                        time.sleep(0.2)
                    if rows:
                        new = _frame(rows)
                        new["dataset"] = "edgar_form4_xml_intraday"
                        old = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame()
                        pd.concat([old, new], ignore_index=True).to_parquet(OUT, index=False)
                    _mark(got)
                    buys = sum(1 for r in rows if r.get("trans_code") == "P")
                    n = refresh_insider_signals() if buys else 0
                    print(f"{now:%Y-%m-%d %H:%M} new filings={len(got)} purchases={buys} insider signals={n}", flush=True)
                except Exception as e:
                    print(f"{now:%Y-%m-%d %H:%M} error {e!r}"[:300], flush=True)
            if locked:
                fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
