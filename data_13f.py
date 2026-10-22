"""SEC Form 13F data sets -> per-period (filer, CUSIP, shares) holdings, then per-CUSIP ownership aggregates.

Point-in-time rule: for report period p only original 13F-HR filings filed on or before p + 45 days
(the statutory deadline) are used, and the aggregate becomes usable on p + 46 days. Late filers and
amendments are excluded rather than back-filled. Options (PUTCALL set) and principal-amount (PRN) rows
are dropped. 13F omits shorts, non-13(f) securities and small managers: these are not fund flows.
"""
import io
import re
import sys
import time
import zipfile
from datetime import datetime, timezone

import httpx
import numpy as np
import pandas as pd

from config import PROC, RAW, SEC_UA

INDEX = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
OUT = RAW / "13f"
DEADLINE = pd.Timedelta(days=45)


def links(client):
    html = client.get(INDEX).text
    return sorted(set("https://www.sec.gov" + h for h in re.findall(r'href="(/files/[^"]*form13f\.zip)"', html)))


def process(url, client):
    name = url.rsplit("/", 1)[-1].replace(".zip", "")
    done = OUT / "done" / f"{name}.ok"
    if done.exists():
        return "cached"
    r = None
    for i in range(6):
        try:
            r = client.get(url)
            if r.status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(2 ** i)
    if r is None or r.status_code != 200:
        return f"failed {getattr(r, 'status_code', None)}"
    z = zipfile.ZipFile(io.BytesIO(r.content))
    member = {n.rsplit("/", 1)[-1].upper(): n for n in z.namelist()}  # some years nest files in a folder
    with z.open(member["SUBMISSION.TSV"]) as f:
        sub = pd.read_csv(f, sep="\t", dtype=str, usecols=["ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT"])
    sub = sub[sub["SUBMISSIONTYPE"] == "13F-HR"].copy()
    sub["fd"] = pd.to_datetime(sub["FILING_DATE"], format="%d-%b-%Y", errors="coerce")
    sub["period"] = pd.to_datetime(sub["PERIODOFREPORT"], format="%d-%b-%Y", errors="coerce")
    sub = sub[(sub["fd"] <= sub["period"] + DEADLINE) & sub["period"].notna()]
    acc = sub.set_index("ACCESSION_NUMBER")[["CIK", "period"]]
    parts, names = [], []
    with z.open(member["INFOTABLE.TSV"]) as f:
        for ch in pd.read_csv(f, sep="\t", dtype=str, chunksize=500_000, quoting=3, on_bad_lines="skip",
                              usecols=["ACCESSION_NUMBER", "NAMEOFISSUER", "TITLEOFCLASS", "CUSIP", "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL"]):
            ch = ch[ch["ACCESSION_NUMBER"].isin(acc.index) & ch["PUTCALL"].isna() & (ch["SSHPRNAMTTYPE"] == "SH")]
            ch = ch.join(acc, on="ACCESSION_NUMBER")
            ch["shares"] = pd.to_numeric(ch["SSHPRNAMT"], errors="coerce")
            ch["CUSIP"] = ch["CUSIP"].str.upper().str.strip()
            names.append(ch[["CUSIP", "NAMEOFISSUER", "TITLEOFCLASS"]].drop_duplicates("CUSIP"))
            parts.append(ch.groupby(["period", "CIK", "CUSIP"], as_index=False)["shares"].sum())
    hold = pd.concat(parts).groupby(["period", "CIK", "CUSIP"], as_index=False)["shares"].sum()
    for p, g in hold.groupby("period"):
        dest = OUT / "holdings" / f"{p.date()}__{name}.parquet"
        g.assign(CIK=g["CIK"].astype(int)).to_parquet(dest, index=False)
    pd.concat(names).drop_duplicates("CUSIP").to_parquet(OUT / "names" / f"{name}.parquet", index=False)
    done.touch()
    return f"{len(hold)} rows, periods {sorted(set(hold['period'].dt.date.astype(str)))}"


def aggregate():
    """Per (period, CUSIP): holders, shares held, HHI, and Chen-Hong-Stein breadth change among filers
    present in both periods."""
    files = sorted((OUT / "holdings").glob("*.parquet"))
    periods = sorted({f.name.split("__")[0] for f in files})
    rows, prev = [], None
    for p in periods:
        h = pd.concat([pd.read_parquet(f) for f in files if f.name.startswith(p + "__")])
        h["CUSIP"] = h["CUSIP"].astype(str).str.zfill(9)
        h = h.groupby(["CIK", "CUSIP"], as_index=False)["shares"].sum()
        h["CIK"] = h["CIK"].astype("int32")
        h["CUSIP"] = h["CUSIP"].astype("category")
        filers = set(h["CIK"])
        g = h.groupby("CUSIP", observed=True)["shares"]
        tot = g.sum()
        agg = pd.DataFrame({"n_holders": g.size(), "inst_shares": tot,
                            "hhi": (h["shares"] / h["CUSIP"].map(tot).astype(float)).pow(2).groupby(h["CUSIP"], observed=True).sum()})
        agg.index = agg.index.astype(str)
        if prev is not None:
            ph, pfilers = prev
            both = filers & pfilers
            cur_b = h[h["CIK"].isin(both)].groupby("CUSIP", observed=True).size()
            prv_b = ph[ph["CIK"].isin(both)].groupby("CUSIP", observed=True).size()
            cur_b.index, prv_b.index = cur_b.index.astype(str), prv_b.index.astype(str)
            idx = cur_b.index.union(prv_b.index)
            agg = agg.reindex(agg.index.union(idx))
            agg["d_breadth"] = (cur_b.reindex(idx, fill_value=0) - prv_b.reindex(idx, fill_value=0)) / max(len(both), 1)
            mg = h[["CIK", "CUSIP"]].merge(ph[["CIK", "CUSIP"]], how="outer", indicator=True)
            agg["n_new"] = mg.loc[mg["_merge"] == "left_only", "CUSIP"].value_counts().reindex(agg.index).fillna(0)
            agg["n_exit"] = mg.loc[mg["_merge"] == "right_only", "CUSIP"].value_counts().reindex(agg.index).fillna(0)
            del mg
        agg.index = agg.index.astype(str)
        agg.index.name = "CUSIP"
        agg["period"] = pd.Timestamp(p)
        agg["available_date"] = pd.Timestamp(p) + DEADLINE + pd.Timedelta(days=1)
        agg["n_filers_total"] = len(filers)
        rows.append(agg.reset_index())
        prev = (h, filers)
        print("agg", p, len(agg), flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(PROC / "inst_ownership_cusip.parquet", index=False)
    return out


def run():
    for d in ["holdings", "names", "done"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=600, headers={"User-Agent": SEC_UA}, follow_redirects=True) as c:
        for u in links(c):
            print(u.rsplit("/", 1)[-1], process(u, c), flush=True)
            time.sleep(1)
    aggregate()
    print("done", datetime.now(timezone.utc).isoformat(timespec="seconds"))


if __name__ == "__main__":
    aggregate() if "--aggregate" in sys.argv else run()
