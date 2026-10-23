"""Filing-text similarity ("Lazy Prices", Cohen, Malloy & Nguyen, Journal of Finance 2020).

For every 10-K and 10-Q of the universe's issuers (from TEXT_START), the primary document is downloaded from EDGAR,
reduced to plain words and compared with the same issuer's filing of the same type about one year earlier
(10-K with the previous 10-K; 10-Q with the same fiscal quarter's 10-Q of the prior year):
  sim_cos      cosine similarity of word-count vectors (1 = unchanged wording)
  sim_jac      Jaccard similarity of the word sets
  sim_rf_cos   cosine similarity of the 10-K Item 1A "Risk Factors" section (10-K only, NaN when not located)
  len_chg      log(words / comparator words)
The paper finds that "changers" (low similarity) underperform over the following year, with no reaction on the filing
date; changes to risk factors are the most informative. Point-in-time: a filing is usable from the first session after
its EDGAR acceptance time (data_8k.accepted_et corrects the API's doubled timezone offset).

Cleaned text is stored gzip-compressed under data/raw/text/<cik>/<accession>.txt.gz (re-used for incremental updates
and future text features); per-issuer results under data/raw/text/<cik>.parquet and the merged table in
data/processed/text_similarity.parquet. Requests stay below the SEC fair-access limit (10/s) with a contact UA.
Usage: data_text.py [--max-ciks N] [--refresh]   (own lock: .text.lock, never blocks the nightly job)
"""
import fcntl
import gzip
import html
import math
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import httpx
from datetime import datetime
import numpy as np
import pandas as pd

import config as C
from data_8k import accepted_et

TEXT_START = "2011-01-01"
OUT = C.RAW / "text"
FORMS = ("10-K", "10-Q")
MIN_GAP_DAYS, MAX_GAP_DAYS = 300, 430
RATE = 8.0      # requests per second across all threads (SEC fair-access limit is 10)
WORKERS = 4


class Limiter:
    def __init__(self, rate):
        self.gap, self.next, self.lock = 1.0 / rate, time.monotonic(), threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.gap
        time.sleep(max(0.0, t - now))


LIMIT = Limiter(RATE)
TAGS = re.compile(r"<[^>]+>")
HIDDEN = re.compile(r"<(script|style|ix:header|head)\b.*?</\1\s*>", re.S | re.I)
WORD = re.compile(r"[a-z]{2,}")
RF_START = re.compile(r"item\s*1a\.?\s*[\-:—–]?\s*risk\s+factors", re.I)
RF_END = re.compile(r"item\s*(1b\.?\s*[\-:—–]?\s*unresolved|1c\.?\s*[\-:—–]?\s*cybersecurity|2\.?\s*[\-:—–]?\s*properties)", re.I)


def clean(raw):
    t = HIDDEN.sub(" ", raw)
    t = TAGS.sub(" ", t)
    t = html.unescape(t).replace("\xa0", " ")
    return re.sub(r"\s+", " ", t).strip()


def words(text):
    return WORD.findall(text.lower())


def risk_factors(text):
    """Longest Item 1A ... Item 1B/1C/2 span: the table of contents yields a short match, the section a long one."""
    best = ""
    for m in RF_START.finditer(text):
        e = RF_END.search(text, m.end())
        seg = text[m.end(): e.start()] if e else ""
        if len(seg) > len(best):
            best = seg
    return best if len(best) > 2000 else ""


def cosine(a, b):
    if not a or not b:
        return np.nan
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na, nb = math.sqrt(sum(v * v for v in a.values())), math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else np.nan


def jaccard(a, b):
    if not a or not b:
        return np.nan
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def filings(client, cik):
    """All 10-K / 10-Q (no amendments) since TEXT_START from the submissions API, incl. paged history."""
    LIMIT.wait()
    j = client.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
    blocks = [j["filings"]["recent"]]
    for f in j["filings"].get("files", []):
        if f.get("filingTo", "9999") >= TEXT_START:
            LIMIT.wait()
            blocks.append(client.get(f"https://data.sec.gov/submissions/{f['name']}").json())
    cols = ["accessionNumber", "filingDate", "acceptanceDateTime", "form", "primaryDocument", "reportDate"]
    df = pd.concat([pd.DataFrame({k: b.get(k, []) for k in cols}) for b in blocks], ignore_index=True)
    df = df[df["form"].isin(FORMS) & (df["filingDate"] >= TEXT_START) & (df["primaryDocument"].fillna("") != "")]
    return df.drop_duplicates("accessionNumber").sort_values("filingDate").reset_index(drop=True)


def document(client, cik, acc, doc):
    path = OUT / str(cik) / f"{acc}.txt.gz"
    if path.exists():
        return gzip.decompress(path.read_bytes()).decode()
    for i in range(4):
        try:
            LIMIT.wait()
            r = client.get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}")
            if r.status_code == 200:
                text = clean(r.text)
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(gzip.compress(text.encode(), 6))
                tmp.replace(path)
                return text
            if r.status_code == 404:
                return None
        except httpx.HTTPError:
            pass
        time.sleep(2 ** (i + 1))
    return None


def process_cik(client, cik):
    fl = filings(client, cik)
    if fl.empty:
        return pd.DataFrame()
    fl["filed"] = pd.to_datetime(fl["filingDate"])
    fl["accepted_et"] = accepted_et(fl["acceptanceDateTime"])
    vec, rows = {}, []
    for r in fl.itertuples():
        text = document(client, cik, r.accessionNumber, r.primaryDocument)
        if not text:
            continue
        w = words(text)
        rf = Counter(words(risk_factors(text))) if r.form == "10-K" else Counter()
        vec[r.accessionNumber] = (r.form, r.filed, Counter(w), rf, len(w))
        prev = [(a, v) for a, v in vec.items() if v[0] == r.form and MIN_GAP_DAYS <= (r.filed - v[1]).days <= MAX_GAP_DAYS]
        rec = {"cik": cik, "accession": r.accessionNumber, "form": r.form, "filed": r.filed, "report_date": r.reportDate,
               "accepted_et": r.accepted_et, "words": len(w), "comparator": None, "sim_cos": np.nan, "sim_jac": np.nan,
               "sim_rf_cos": np.nan, "len_chg": np.nan}
        if prev:
            a, p = min(prev, key=lambda kv: abs((r.filed - kv[1][1]).days - 365))
            cur = vec[r.accessionNumber]
            rec.update(comparator=a, sim_cos=cosine(cur[2], p[2]), sim_jac=jaccard(cur[2], p[2]),
                       sim_rf_cos=cosine(cur[3], p[3]) if cur[3] and p[3] else np.nan,
                       len_chg=math.log(cur[4] / p[4]) if cur[4] and p[4] else np.nan)
        rows.append(rec)
        for a in [a for a, v in vec.items() if (r.filed - v[1]).days > MAX_GAP_DAYS]:  # keep memory bounded
            del vec[a]
    return pd.DataFrame(rows)


def merge():
    parts = [pd.read_parquet(f) for f in sorted(OUT.glob("*.parquet"))]
    parts = [p for p in parts if len(p)]
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    # usable from the first calendar day after acceptance (New York time): never on the filing session itself
    df["available_date"] = (pd.to_datetime(df["accepted_et"].astype(str).str[:10]) + pd.Timedelta(days=1)).astype("datetime64[us]")
    tmp = C.PROC / "text_similarity.parquet.tmp"
    df.to_parquet(tmp, index=False)
    tmp.replace(C.PROC / "text_similarity.parquet")
    return df


def run(max_ciks=None, refresh=False, only=None):
    OUT.mkdir(parents=True, exist_ok=True)
    lock = open(C.ROOT / ".text.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("data_text already running", flush=True)
        return None
    uni = pd.read_parquet(C.PROC / "panel_broad.parquet", columns=["cik"]) if (C.PROC / "panel_broad.parquet").exists() else None
    ciks = sorted(set(uni["cik"].dropna().astype(int))) if uni is not None else []
    if only is not None:  # incremental: issuers with a new 10-K/10-Q (cached documents are re-used, new ones downloaded)
        ciks, refresh = [c for c in ciks if c in only], True
    todo = [c for c in ciks if refresh or not (OUT / f"{c}.parquet").exists()][: max_ciks or None]
    done = [0]
    client = httpx.Client(headers={"User-Agent": C.SEC_UA}, timeout=60, follow_redirects=True,
                          limits=httpx.Limits(max_connections=WORKERS))

    def one(cik):
        dest = OUT / f"{cik}.parquet"
        try:
            df = process_cik(client, cik)
        except Exception as e:  # one issuer must not stop the backfill
            print(f"cik {cik}: {e.__class__.__name__}: {e}"[:200], flush=True)
            return
        tmp = dest.with_suffix(f".{threading.get_ident()}.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(dest)
        done[0] += 1
        if done[0] % 25 == 0:
            print(f"{done[0]}/{len(todo)} issuers done ({datetime.now():%H:%M})", flush=True)

    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(one, todo))
    client.close()
    df = merge()
    print(f"text similarity: {0 if df is None else len(df)} filings from {0 if df is None else df['cik'].nunique()} issuers", flush=True)
    return df


if __name__ == "__main__":
    n = int(sys.argv[sys.argv.index("--max-ciks") + 1]) if "--max-ciks" in sys.argv else None
    if "--queue" in sys.argv:
        q = C.LEDGER / "text_queue.txt"
        only = {int(x) for x in q.read_text().split()} if q.exists() else set()
        if only:
            if run(only=only) is not None:
                q.write_text("")
        else:
            print("text queue empty", flush=True)
    else:
        run(max_ciks=n, refresh="--refresh" in sys.argv)
