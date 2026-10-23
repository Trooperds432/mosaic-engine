"""Investor attention from English Wikipedia page views (free alternative data, Wikimedia REST API, from July 2015).

Mapping: Wikidata links companies to their SEC CIK (property P5531) and to their English Wikipedia article; issuers
without a CIK link fall back to an exchange ticker (P414 / P249) match on NYSE or Nasdaq. Daily user page views per
article are cached per issuer in data/raw/wiki/<cik>.parquet and merged into data/processed/wiki_views.parquet.
Point-in-time: a UTC day's count is published shortly after midnight UTC, so features at decision date t use days
<= t-1 only. Known gaps: article renames split history (the current title only), and mapping uses today's Wikidata.
Usage: data_wiki.py [--max N] [--days D]  (D: incremental refresh window, default full history for new issuers)
"""
import sys
import time
import urllib.parse
from datetime import date, timedelta

import httpx
import pandas as pd

import config as C

OUT = C.RAW / "wiki"
START = "20150701"
SPARQL = "https://query.wikidata.org/sparql"
VIEWS = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{t}/daily/{a}/{b}"
Q_CIK = """SELECT ?cik ?article WHERE { ?item wdt:P5531 ?cik . ?article schema:about ?item ;
  schema:isPartOf <https://en.wikipedia.org/> . }"""
Q_TICKER = """SELECT ?ticker ?article WHERE { VALUES ?ex { wd:Q13677 wd:Q82059 } ?item p:P414 ?st . ?st ps:P414 ?ex ;
  pq:P249 ?ticker . ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> . }"""
PAUSE = 0.1


def sparql(client, q):
    r = client.get(SPARQL, params={"query": q, "format": "json"}, headers={"Accept": "application/sparql-results+json"}, timeout=300)
    r.raise_for_status()
    return [{k: v["value"] for k, v in b.items()} for b in r.json()["results"]["bindings"]]


def mapping(client):
    uni = pd.read_parquet(C.PROC / "universe.parquet")
    uni["cik"] = uni["cik"].astype(int)
    by_cik = {}
    for b in sparql(client, Q_CIK):
        try:
            by_cik.setdefault(int(b["cik"]), b["article"].rsplit("/wiki/", 1)[1])
        except ValueError:
            continue
    by_ticker = {}
    for b in sparql(client, Q_TICKER):
        by_ticker.setdefault(b["ticker"].upper().replace("-", "."), b["article"].rsplit("/wiki/", 1)[1])
    tick_col = "ticker" if "ticker" in uni else uni.columns[0]
    rows = []
    for r in uni.drop_duplicates("cik").itertuples():
        t = by_cik.get(r.cik)
        src = "wikidata_cik"
        if t is None:
            t, src = by_ticker.get(str(getattr(r, tick_col)).upper().replace("-", ".")), "wikidata_ticker"
        if t:
            rows.append({"cik": r.cik, "article": urllib.parse.unquote(t), "source": src})
    m = pd.DataFrame(rows)
    m.to_parquet(C.PROC / "wiki_map.parquet", index=False)
    return m


def views(client, article, a, b):
    url = VIEWS.format(t=urllib.parse.quote(article.replace(" ", "_"), safe=""), a=a, b=b)
    for i in range(4):
        try:
            r = client.get(url)
            time.sleep(PAUSE)
            if r.status_code == 404:
                return pd.DataFrame(columns=["date", "views"])
            if r.status_code == 200:
                items = r.json().get("items", [])
                return pd.DataFrame({"date": pd.to_datetime([x["timestamp"][:8] for x in items]), "views": [x["views"] for x in items]})
        except httpx.HTTPError:
            pass
        time.sleep(2 ** (i + 1))
    return None


def run(max_n=None, days=None):
    OUT.mkdir(parents=True, exist_ok=True)
    end = (date.today() - timedelta(days=1)).strftime("%Y%m%d")
    with httpx.Client(headers={"User-Agent": C.SEC_UA}, timeout=60, follow_redirects=True) as c:
        m = mapping(c)
        print(f"wikipedia articles mapped for {len(m)} issuers ({(m['source'] == 'wikidata_cik').sum()} by CIK)", flush=True)
        done = 0
        for r in m.itertuples():
            dest = OUT / f"{r.cik}.parquet"
            old = pd.read_parquet(dest) if dest.exists() else None
            if old is not None and len(old) and old.attrs.get("article", r.article) != r.article:
                old = None
            if old is not None and len(old):
                a = (old["date"].max() - pd.Timedelta(days=days or 3)).strftime("%Y%m%d")
            else:
                a = START
            if a > end:
                continue
            new = views(c, r.article, a, end)
            if new is None:
                continue
            df = pd.concat([old, new]) if old is not None else new
            df = df.drop_duplicates("date", keep="last").sort_values("date")
            df["article"] = r.article
            tmp = dest.with_suffix(".tmp")
            df.to_parquet(tmp, index=False)
            tmp.replace(dest)
            done += 1
            if done % 250 == 0:
                print(f"{done}/{len(m)} issuers", flush=True)
            if max_n and done >= max_n:
                break
    parts = []
    for f in OUT.glob("*.parquet"):
        d = pd.read_parquet(f, columns=["date", "views"])
        d["cik"] = int(f.stem)
        parts.append(d)
    if parts:
        allv = pd.concat(parts, ignore_index=True)
        tmp = C.PROC / "wiki_views.parquet.tmp"
        allv.to_parquet(tmp, index=False)
        tmp.replace(C.PROC / "wiki_views.parquet")
        print(f"wiki views: {len(allv)} rows, {allv['cik'].nunique()} issuers, through {allv['date'].max().date()}", flush=True)


if __name__ == "__main__":
    n = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else None
    d = int(sys.argv[sys.argv.index("--days") + 1]) if "--days" in sys.argv else None
    run(n, d)
