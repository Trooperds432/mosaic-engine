import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import httpx
import pandas as pd

from insider_acceptances import build_sidecar, issuer_acceptances, match_table
from insider_filings import acceptance_map


class InsiderAcceptanceArchiveTests(unittest.TestCase):
    def test_recent_and_archived_accessions_have_independent_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, source, output = (root / name for name in ("sec.zip", "filings.parquet", "out.csv"))
            accessions = [f"0000000000-24-00000{i}" for i in range(1, 5)]
            pd.DataFrame({"accession_number": accessions, "issuercik": [1, 1, 1, 1], "trans_code": ["P", "P", "S", "M"]}).to_parquet(source)
            with ZipFile(archive, "w") as z:
                z.writestr("CIK0000000001.json", json.dumps({"filings": {"recent": {
                    "accessionNumber": accessions[:1], "form": ["4"], "acceptanceDateTime": ["2024-02-01T22:00:00Z"]
                }}}))
                z.writestr("CIK0000000002.json", json.dumps({"filings": {"recent": {
                    "accessionNumber": accessions[:1], "form": ["4"], "acceptanceDateTime": ["2024-02-01T18:00:00Z"]
                }}}))
                z.writestr("CIK0000000001-submissions-001.json", json.dumps({
                    "accessionNumber": accessions[1:3], "form": ["4/A", "4"],
                    "acceptanceDateTime": ["2024-02-02T14:00:00-05:00", "2024-02-02T14:00:00"]
                }))
            self.assertEqual(build_sidecar(archive, [source], output),
                             {"requested_accessions": 3, "matched_accessions": 2, "unmatched_accessions": 1})
            mapped = acceptance_map(output)
            self.assertEqual(mapped.iloc[0], pd.Timestamp("2024-02-01T22:00Z"))
            self.assertEqual(mapped.iloc[1], pd.Timestamp("2024-02-02T19:00Z"))
            with self.assertRaises(FileExistsError):
                build_sidecar(archive, [source], output)

    def test_api_follows_older_submission_page_for_missing_accession(self):
        seen = []
        accession = "0000000001-24-000001"
        key = accession.replace("-", "")

        def response(request):
            name = request.url.path.rsplit("/", 1)[-1]
            seen.append(name)
            if len(seen) == 1:
                return httpx.Response(200, json={"filings": {"recent": {}, "files": [
                    {"name": "CIK0000000001-submissions-001.json"},
                    {"name": "CIK0000000001-submissions-002.json"}]}})
            return httpx.Response(200, json={"accessionNumber": [accession], "form": ["4/A"],
                                             "acceptanceDateTime": ["2024-02-02T14:00:00-05:00"]})

        matched = {}
        with httpx.Client(transport=httpx.MockTransport(response)) as client:
            issuer_acceptances(client, 1, {key}, matched, pause=lambda _: None)
        self.assertEqual(seen, ["CIK0000000001.json", "CIK0000000001-submissions-001.json"])
        self.assertEqual(matched, {key: "2024-02-02T19:00:00+00:00"})

    def test_conflicting_accession_stays_excluded_after_duplicate(self):
        key = "000000000124000001"
        matched, conflicts = {}, {}
        for stamp in ["2024-02-02T14:00Z", "2024-02-02T19:00Z", "2024-02-02T14:00Z"]:
            table = {"accessionNumber": [key], "form": ["4"], "acceptanceDateTime": [stamp]}
            match_table(table, {key}, matched, conflicts)
        self.assertEqual(matched, {})
        self.assertEqual(len(conflicts[key]), 2)
