import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import pandas as pd
import config as C
from lseg_data import import_bundle, observations_asof, read_bundle


class LsegImportTests(unittest.TestCase):
    def bundle(self, root, *, backdated=False, corrupt=False, unsafe=False):
        csv = b"Instrument,observation_date,EPS\nAAPL.O,2020-01-01,2.0\n"
        item = {"name": "consensus_history", "status": "OBSERVED", "universe": "AAPL.O",
                "file": "data.csv", "sha256": hashlib.sha256(csv).hexdigest(),
                "observed_at_utc": "2026-01-02T12:00:00+00:00",
                "available_at_utc": "2020-01-01T00:00:00+00:00" if backdated else "2026-01-02T12:00:00+00:00",
                "availability_basis": "observed_snapshot", "historical_pit_certified": False,
                "source_id": "LSEG/test", "rows": 1, "columns": ["Instrument", "observation_date", "EPS"]}
        doc = {"schema_version": "1.0", "provider": "LSEG CodeBook", "historical_pit_certified": False,
               "universe_scope": "test", "datasets": [item]}
        path = root / "sample.zip"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("manifest.json", json.dumps(doc))
            z.writestr("data.csv", csv.replace(b"2.0", b"9.0") if corrupt else csv)
            if unsafe:
                z.writestr("../escape.csv", csv)
        return path

    def test_actual_archive_time_controls_historical_visibility_and_repeat_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(C, "RAW", root / "raw"), patch.object(C, "PROC", root / "proc"), patch.object(C, "REPORTS", root / "reports"):
                path = self.bundle(root)
                output = import_bundle(path)
                self.assertEqual(import_bundle(path), output)
                self.assertTrue(observations_asof("2020-12-31T23:59:59Z").empty)
                rows = observations_asof("2026-01-03T00:00:00Z", "consensus_history")
                self.assertEqual(rows["value"].tolist(), ["2.0"])
                self.assertFalse(rows["historical_pit_certified"].any())
                self.assertEqual(len(list((C.PROC / "lseg").glob("*.parquet"))), 1)
                with self.assertRaises(ValueError):
                    observations_asof("2026-01-03")

    def test_rejects_backdating_tampering_and_path_traversal(self):
        for case in ["backdated", "corrupt", "unsafe"]:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    read_bundle(self.bundle(Path(tmp), **{case: True}))


if __name__ == "__main__":
    unittest.main()
