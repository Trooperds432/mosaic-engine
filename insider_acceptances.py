"""Build the Form 4 acceptance sidecar from a local SEC submissions ZIP."""
import argparse
import csv
import json
import time
from pathlib import Path
import re
from zipfile import ZipFile

import httpx
import pandas as pd
import pyarrow.parquet as pq

import config as C
from insider_filings import accession_key, filing_paths


def match_table(table, wanted, matched, conflicts=None):
    keys = table.get("accessionNumber", [])
    forms = table.get("form", [])
    times = table.get("acceptanceDateTime")
    if times is None:
        return
    for accession, form, stamp in zip(keys, forms, times, strict=True):
        key = accession.replace("-", "")
        if key not in wanted or form not in {"4", "4/A"} or not stamp:
            continue
        accepted = pd.Timestamp(stamp)
        if pd.isna(accepted) or accepted.tzinfo is None:
            continue
        normalized = accepted.tz_convert("UTC").isoformat()
        if conflicts is not None and key in conflicts:
            conflicts[key].add(normalized)
            continue
        if key in matched and matched[key] != normalized:
            if conflicts is None:
                raise ValueError(f"conflicting SEC acceptance timestamps for {key}")
            conflicts[key] = {matched.pop(key), normalized}
            continue
        matched[key] = normalized


def archive_acceptances(archive, wanted_by_issuer, conflicts=None):
    """Match accessions in both recent and archived SEC submission arrays."""
    matched = {}
    if not wanted_by_issuer:
        return matched
    with ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            name = Path(entry.filename).name
            issuer = re.fullmatch(r"CIK([0-9]{10})(?:-submissions-[0-9]+)?[.]json", name)
            if not issuer or entry.is_dir():
                continue
            wanted = wanted_by_issuer.get(int(issuer[1]))
            if not wanted:
                continue
            with bundle.open(entry) as stream:
                document = json.load(stream)
            match_table(document.get("filings", {}).get("recent", document), wanted, matched, conflicts)
    return matched


def issuer_acceptances(client, cik, wanted, matched, pause=time.sleep):
    """Fetch SEC metadata by issuer, following historical pages if needed."""
    def get(filename):
        if not re.fullmatch(r"CIK[0-9]{10}(?:-submissions-[0-9]+)?[.]json", filename):
            raise ValueError("invalid SEC submissions page name")
        pause(.35)
        response = client.get(f"https://data.sec.gov/submissions/{filename}")
        response.raise_for_status()
        return response.json()

    filings = get(f"CIK{cik:010d}.json")["filings"]
    match_table(filings["recent"], wanted, matched)
    for page in filings.get("files", []):
        if wanted <= matched.keys():
            break
        match_table(get(page["name"]), wanted, matched)


def build_from_api(filings, output, limit=None):
    parts = [pd.read_parquet(path, columns=["issuercik", "accession_number"],
                            filters=[("document_type", "in", ["4", "4/A"]), ("trans_code", "in", ["P", "S"])])
             for path in filings]
    if not parts:
        raise ValueError("No local Form 4 filings")
    records = pd.concat(parts, ignore_index=True).dropna().drop_duplicates()
    groups = list(records.groupby("issuercik", sort=True))
    if limit is not None:
        if limit < 1:
            raise ValueError("issuer limit must be positive")
        groups.sort(key=lambda pair: (-len(pair[1]), int(pair[0])))
        groups = groups[:limit]
    wanted, matched, failures = set(), {}, {}
    with httpx.Client(timeout=20, headers={"User-Agent": C.SEC_UA}) as client:
        for cik, group in groups:
            keys = set(accession_key(group["accession_number"]))
            wanted.update(keys)
            try:
                issuer_acceptances(client, int(cik), keys, matched)
            except httpx.HTTPError as exc:
                failures[str(cik)] = type(exc).__name__
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (403, 429):
                    failures[str(cik)] = f"HTTP {exc.response.status_code}; stopped collection"
                    wanted.update(accession_key(records["accession_number"]))
                    break
    return write_sidecar(output, wanted, matched) | {"issuer_errors": failures, "selected_issuers": len(groups)}


def build_sidecar(archive, filings, output):
    wanted_by_issuer = {}
    for path in filings:
        table = pq.read_table(path, columns=["issuercik", "accession_number"], filters=[("trans_code", "in", ["P", "S"])])
        rows = table.to_pandas().dropna()
        for cik, group in rows.groupby("issuercik"):
            wanted_by_issuer.setdefault(int(cik), set()).update(accession_key(group["accession_number"]).unique())
    wanted = set().union(*wanted_by_issuer.values())
    if not wanted:
        raise ValueError("No Form 4 purchase or sale accessions found in local filings")
    conflicts = {}
    matched = archive_acceptances(archive, wanted_by_issuer, conflicts)
    report = write_sidecar(output, wanted, matched)
    if conflicts:
        conflict_path = Path(output).with_suffix(".conflicts.json")
        with conflict_path.open("x", encoding="utf-8") as stream:
            json.dump({key: sorted(values) for key, values in sorted(conflicts.items())}, stream, indent=2)
        report.update(conflicting_accessions=len(conflicts), conflict_file=str(conflict_path))
    return report


def write_sidecar(output, wanted, matched):
    if not matched:
        raise ValueError("No timezone-qualified acceptance timestamps matched local accessions")
    with Path(output).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["accession_number", "accepted_at_utc"])
        writer.writerows(sorted(matched.items()))
    return {"requested_accessions": len(wanted), "matched_accessions": len(matched),
            "unmatched_accessions": len(wanted - matched.keys())}


def coverage(filings, sidecar):
    """Measure timestamp coverage by quarterly source and unique accession."""
    from insider_filings import acceptance_map

    sidecar_times = acceptance_map(sidecar)
    accepted_keys = set(sidecar_times.index)
    seen, verified = set(), set()
    quarters = []
    for path in filings:
        columns = ["accession_number"]
        if "accepted_at_utc" in pq.read_schema(path).names:
            columns.append("accepted_at_utc")
        table = pq.read_table(path, columns=columns, filters=[
            ("trans_code", "in", ["P", "S"]), ("document_type", "in", ["4", "4/A"])])
        rows = table.to_pandas()
        accessions = accession_key(rows["accession_number"])
        keys = set(accessions.dropna().unique())
        matched = keys & accepted_keys
        if "accepted_at_utc" in rows:
            raw = rows["accepted_at_utc"].astype("string")
            zoned = raw.str.contains(r"(?:Z|[+-][0-9]{2}:[0-9]{2})$", regex=True).fillna(False)
            times = pd.to_datetime(raw.where(zoned), utc=True, format="mixed", errors="coerce")
            pairs = pd.DataFrame({"key": accessions, "time": times}).dropna().drop_duplicates()
            if pairs["key"].duplicated().any():
                raise ValueError(f"conflicting embedded timestamps in {path.name}")
            mapped = pairs["key"].map(sidecar_times) if not sidecar_times.empty else pd.NaT
            if (pd.notna(mapped) & pairs["time"].ne(mapped)).any():
                raise ValueError(f"embedded timestamps disagree with the sidecar in {path.name}")
            matched.update(pairs["key"])
        seen.update(keys)
        verified.update(matched)
        quarters.append({"source": path.name, "accessions": len(keys),
                         "matched": len(matched), "unmatched": len(keys - matched)})
    return {"accessions": len(seen), "matched": len(verified),
            "unmatched": len(seen - verified), "sources": quarters}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, nargs="?")
    parser.add_argument("--sec-api", action="store_true")
    parser.add_argument("--issuer-limit", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--coverage", type=Path, help="audit an existing timestamp sidecar")
    args = parser.parse_args()
    if args.coverage:
        if args.archive or args.sec_api or args.issuer_limit is not None or args.out:
            parser.error("--coverage is a read-only mode")
        print(json.dumps(coverage(filing_paths(), args.coverage), indent=2))
        return
    if not args.out:
        parser.error("--out is required to build a sidecar")
    if bool(args.archive) == args.sec_api or (args.issuer_limit is not None and not args.sec_api):
        parser.error("choose one archive or --sec-api; --issuer-limit applies only to API mode")
    if args.out.exists():
        parser.error("output already exists")
    if args.sec_api:
        result = build_from_api(filing_paths(), args.out, args.issuer_limit)
    else:
        result = build_sidecar(args.archive, filing_paths(), args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
