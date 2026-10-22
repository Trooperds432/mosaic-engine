"""SEC Insider Transactions Data Sets (Forms 3/4/5, quarterly, 2006+).

Per quarter, for universe issuers:
  raw/insider/{q}.parquet        non-derivative transactions + filing metadata + linked footnotes
  raw/insider_deriv/{q}.parquet  derivative transactions (options, RSUs, conversions; codes M, A, F, ...)
Footnotes are linked through each transaction's *_FN columns. A transaction counts as a 10b5-1 plan trade
if the 2023+ checkbox (AFF10B5ONE) is set OR a linked footnote mentions Rule 10b5-1; the plan adoption
date is parsed from that footnote when stated. Dataset URLs are read from SEC's index page (recent
quarters moved to a different path).
"""
import io
import re
import sys
import time
import zipfile
from datetime import datetime, timezone

import httpx
import pandas as pd

from config import PROC, RAW, SEC_UA

INDEX = "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets"
OUT, OUT_D = RAW / "insider", RAW / "insider_deriv"
PLAN_RE = re.compile(r"10b5-?1", re.I)
DATE_RE = re.compile(r"adopted[^.]{0,80}?(?:on|dated)?\s*((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s*\d{4}|\d{1,2}/\d{1,2}/\d{2,4})", re.I)


def links(client):
    html = client.get(INDEX).text
    out = {}
    for h in re.findall(r'href="(/files/[^"]*?(\d{4}q\d)_form345\.zip)"', html):
        out[h[1]] = "https://www.sec.gov" + h[0]
    return out


def _read(z, name, keep=None):
    member = next(n for n in z.namelist() if n.rsplit("/", 1)[-1].upper() == name.upper())
    with z.open(member) as f:
        return pd.read_csv(f, sep="\t", dtype=str, on_bad_lines="skip", quoting=3,
                           usecols=(lambda c: c in keep) if keep else None)


def _link_footnotes(tr, fn):
    """Attach plan flag, adoption date and text of the footnotes each transaction row references."""
    fncols = [c for c in tr.columns if c.endswith("_FN")]
    ids = tr[fncols].fillna("").agg(",".join, axis=1).str.replace(r"\s", "", regex=True)
    refs = ids.str.split(",").explode()
    refs = refs[refs != ""]
    key = tr.loc[refs.index, "ACCESSION_NUMBER"].to_numpy() + "|" + refs.to_numpy()
    fmap = dict(zip(fn["ACCESSION_NUMBER"] + "|" + fn["FOOTNOTE_ID"], fn["FOOTNOTE_TXT"].fillna("")))
    txt = pd.Series([fmap.get(k, "") for k in key], index=refs.index)
    joined = txt.groupby(level=0).agg(" | ".join).reindex(tr.index).fillna("")
    tr["footnote_txt"] = joined.str.slice(0, 600)
    tr["fn_10b5_1"] = joined.str.contains(PLAN_RE)
    tr["plan_adopted"] = pd.to_datetime(joined.where(tr["fn_10b5_1"]).str.extract(DATE_RE)[0], errors="coerce", format="mixed")
    return tr.drop(columns=fncols)


def quarter(q, url, ciks, client):
    dest = OUT / f"{q}.parquet"
    if dest.exists():
        return "cached"
    r = client.get(url)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    sub = _read(z, "SUBMISSION.tsv", {"ACCESSION_NUMBER", "FILING_DATE", "PERIOD_OF_REPORT", "DATE_OF_ORIG_SUB", "DOCUMENT_TYPE",
                                      "ISSUERCIK", "ISSUERTRADINGSYMBOL", "AFF10B5ONE"})
    sub = sub[pd.to_numeric(sub["ISSUERCIK"], errors="coerce").isin(ciks)]
    acc = set(sub["ACCESSION_NUMBER"])
    own = _read(z, "REPORTINGOWNER.tsv", {"ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"})
    own = own[own["ACCESSION_NUMBER"].isin(acc)]
    n_own = own.groupby("ACCESSION_NUMBER").size().rename("n_owners")
    own = own.drop_duplicates("ACCESSION_NUMBER").join(n_own, on="ACCESSION_NUMBER")
    fn = _read(z, "FOOTNOTES.tsv")
    fn = fn[fn["ACCESSION_NUMBER"].isin(acc)]
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = {}
    for table, dest_dir in [("DERIV_TRANS.tsv", OUT_D), ("NONDERIV_TRANS.tsv", OUT)]:  # non-derivative last: its file marks the quarter done
        tr = _read(z, table)
        tr = tr[tr["ACCESSION_NUMBER"].isin(acc)].copy()
        tr = _link_footnotes(tr, fn)
        df = tr.merge(sub, on="ACCESSION_NUMBER").merge(own, on="ACCESSION_NUMBER", how="left")
        df.columns = [c.lower() for c in df.columns]
        for c in ["trans_shares", "trans_pricepershare", "shrs_ownd_folwng_trans", "conv_exercise_price", "undlyng_sec_shares",
                  "trans_total_value", "valu_ownd_folwng_trans"]:
            if c in df:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["issuercik"] = pd.to_numeric(df["issuercik"], errors="coerce").astype("Int64")
        for c in ["filing_date", "trans_date", "period_of_report", "date_of_orig_sub", "deemed_execution_date", "excercise_date", "expiration_date"]:
            if c in df:
                df[c] = pd.to_datetime(df[c], format="%d-%b-%Y", errors="coerce")
        df["dataset"] = f"sec_form345_{q}"
        df["ingested_utc"] = stamp
        df.to_parquet(dest_dir / f"{q}.parquet", index=False)
        out[table] = len(df)
    return out


def run(start_year=2006, refresh_all=False):
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_D.mkdir(parents=True, exist_ok=True)
    ciks = set(pd.read_parquet(PROC / "universe.parquet")["cik"].astype(int))
    with httpx.Client(timeout=300, headers={"User-Agent": SEC_UA}, follow_redirects=True) as c:
        avail = links(c)
        qs = sorted(q for q in avail if int(q[:4]) >= start_year)
        for q in qs:
            # always refresh the latest two quarters (SEC republishes them); refresh_all rebuilds everything
            if (refresh_all or q in qs[-2:]) and (OUT / f"{q}.parquet").exists():
                (OUT / f"{q}.parquet").unlink()
            for i in range(4):
                try:
                    print(q, quarter(q, avail[q], ciks, c), flush=True)
                    break
                except Exception as e:
                    print(q, "ERROR", e, flush=True)
                    time.sleep(5 * (i + 1))
            time.sleep(0.5)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run(int(args[0]) if args else 2006, refresh_all="--refresh-all" in sys.argv)
