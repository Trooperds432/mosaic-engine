"""Accession-level SEC Form 4 research records, including immutable amendments."""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config as C
from features import INS_COLS

NY = "America/New_York"
SIGNATURE = ["issuercik", "owner", "trans_date", "trans_code", "trans_shares", "trans_pricepershare"]
EXTRA = ["accepted_at_utc", "transaction_line", "nonderiv_trans_sk"]


def accession_key(series):
    return series.astype("string").str.replace("-", "", regex=False)


def acceptance_map(path):
    """Join SEC submission timestamps by accession, never by ticker or filing date."""
    d = pd.read_csv(path, dtype="string")
    keys = accession_key(d["accession_number"])
    raw = d["accepted_at_utc"]
    dates = pd.to_datetime(raw, utc=True, format="mixed", errors="coerce")
    if (not keys.str.fullmatch(r"\d{18}").fillna(False).all() or dates.isna().any()
            or not raw.str.contains(r"(?:Z|[+-]\d\d:\d\d)$", regex=True).fillna(False).all()):
        raise ValueError("acceptance CSV requires SEC accessions and explicit timezone timestamps")
    pairs = pd.DataFrame({"key": keys, "accepted": dates}).drop_duplicates()
    if pairs["key"].duplicated().any():
        raise ValueError("conflicting SEC acceptance timestamps for an accession")
    return pairs.set_index("key")["accepted"]


def filing_paths():
    files = sorted((C.RAW / "insider").glob("*.parquet"))
    recent = C.RAW / "insider_recent" / "form4_recent.parquet"
    return files + ([recent] if recent.is_file() else [])


def load_filings(paths, ciks, timestamps=None):
    frames = []
    for path in paths:
        schema = pq.read_schema(path)
        columns = [col for col in INS_COLS + EXTRA if col in schema.names]
        table = pq.read_table(path, columns=columns, filters=[
            ("trans_code", "in", ["P", "S"]), ("issuercik", "in", list(ciks))])
        frames.append(table.to_pandas().reindex(columns=INS_COLS + EXTRA))
    if not frames:
        return pd.DataFrame(columns=INS_COLS + EXTRA)
    return normalize(pd.concat(frames, ignore_index=True), timestamps)


def normalize(frame, timestamps=None):
    d = frame.copy()
    d["accession"] = accession_key(d["accession_number"])
    raw = d.get("accepted_at_utc", pd.Series(index=d.index, dtype="string"))
    zoned = raw.astype("string").str.contains(r"(?:Z|[+-]\d\d:\d\d)$", regex=True).fillna(False)
    d["accepted"] = pd.to_datetime(raw.where(zoned), utc=True, errors="coerce", format="mixed")
    if timestamps is not None and not timestamps.empty:
        mapped = d["accession"].map(timestamps)
        if (mapped.notna() & d["accepted"].notna() & mapped.ne(d["accepted"])).any():
            raise ValueError("SEC acceptance sidecar conflicts with collected timestamps")
        d["accepted"] = d["accepted"].fillna(mapped)
    # A timestamp from the recent XML copy also dates the quarterly copy.
    known = d.loc[d["accepted"].notna(), ["accession", "accepted"]].drop_duplicates()
    if known["accession"].duplicated().any():
        raise ValueError("conflicting timestamps within the same SEC accession")
    if not known.empty:
        d["accepted"] = d["accession"].map(known.set_index("accession")["accepted"])
    d["file_day"] = pd.to_datetime(d["filing_date"], errors="coerce").dt.normalize()
    d["accept_day"] = d["accepted"].dt.tz_convert(NY).dt.tz_localize(None).dt.normalize()
    d["trans_date"] = pd.to_datetime(d["trans_date"], errors="coerce").dt.normalize()
    d["owner"] = (d["rptownercik"].astype("string").str.strip().str.lstrip("0")
                  .replace("", pd.NA).fillna(d["rptownername"].astype("string").str.strip().str.upper()))
    for col in ("trans_shares", "trans_pricepershare"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["value"] = d["trans_shares"] * d["trans_pricepershare"]
    flag = d["aff10b5one"].astype("string").str.lower()
    disclosed = flag.isin(["1", "true", "y", "yes"]) | d["fn_10b5_1"].astype("string").str.lower().isin(["1", "true"])
    d["plan_status"] = np.select([disclosed, flag.isin(["0", "false", "n", "no"])],
                                  ["plan", "non_plan"], default="unknown")
    d["senior"] = d["rptowner_title"].astype("string").str.contains(
        r"\bCEO\b|CHIEF EXECUTIVE|\bCFO\b|CHIEF FINANCIAL", case=False, na=False)
    d = d[d["document_type"].isin(["4", "4/A"]) & d["value"].between(0, 5e9, inclusive="neither")
          & d["trans_shares"].gt(0) & d["trans_pricepershare"].gt(0)
          & d["accession"].notna() & d["trans_date"].notna() & d["owner"].notna()].copy()
    # Prefer quarterly records where both feeds contain the same accession.
    # Keep every separately identified SEC line, even when its economics match.
    line = d.get("non_deriv_trans_sk", d.get("nonderiv_trans_sk", pd.Series(index=d.index, dtype=object)))
    xml = d.get("transaction_line", pd.Series(index=d.index, dtype=object))
    d["line"] = line.astype("string").fillna(xml.astype("string"))
    has_quarter = line.notna().groupby(d["accession"]).transform("any")
    d = d[~has_quarter | xml.isna()]
    return d.drop_duplicates(["accession", "line", *SIGNATURE]).reset_index(drop=True)


def available_filings(filings, calendar, date_only=False, explicit_non_plan=False):
    d = filings.copy()
    d["known_day"] = d["accept_day"].fillna(d["file_day"]) if date_only else d["accept_day"]
    # An amendment remains in the audit trail. An amended line cannot safely be
    # counted as a fresh purchase without a verified link to the original line.
    d = d[d["known_day"].notna() & d["document_type"].eq("4")
          & d["trans_date"].le(d["known_day"])].copy()
    d["e"] = calendar.searchsorted(d["known_day"], side="right")
    d = d.sort_values(["e", "known_day", "accepted", "accession", "line"], kind="stable")
    # Within a session all eligible filings are visible. Routine history uses
    # earlier public filings only, including prior sales.
    history = {}
    classification = []
    for row in d.itertuples():
        key = (row.issuercik, row.owner)
        months = history.setdefault(key, set())
        year, month = row.trans_date.year, row.trans_date.month
        prior = {year - i for i in (1, 2, 3)}
        if not prior <= {y for y, _ in months}:
            classification.append("unclassified")
        else:
            classification.append("routine" if all((y, month) in months for y in prior) else "opportunistic")
        months.add((year, month))
    d["class"] = classification
    allowed = ["non_plan"] if explicit_non_plan else ["non_plan", "unknown"]
    return d[d["trans_code"].eq("P") & d["plan_status"].isin(allowed)]
