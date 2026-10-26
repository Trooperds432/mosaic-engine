"""Full-universe CodeBook requests stay importable: unique names, bounded batches, explicit universes."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lseg_codebook as L  # noqa: E402


class FullRequestTests(unittest.TestCase):
    def test_batches_names_and_universes(self):
        rics = [f"T{i}.N" for i in range(450)]
        jobs = L.full_requests({"rics": rics}, end="2026-10-26", snapshot_batch=200, history_batch=10, membership_from=2025)
        names = [j[0] for j in jobs]
        self.assertEqual(len(names), len(set(names)))
        hist = [j for j in jobs if j[0].startswith("consensus_hist_")]
        self.assertEqual(len(hist), 45)
        self.assertTrue(all(len(j[2]) <= 10 and "TR.EPSMean.calcdate" in j[3] for j in hist))
        self.assertEqual(sorted(r for j in hist for r in j[2]), sorted(rics))
        members = [j for j in jobs if j[0].startswith("members_")]
        self.assertEqual(len(members), 2 * 7)  # 2025 Q1-Q4 and 2026 Q1-Q3 for two indices
        self.assertTrue(all(j[2] in (".SPX", ".NDX") for j in members))


if __name__ == "__main__":
    unittest.main()
