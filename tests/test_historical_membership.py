import unittest

import pandas as pd

from historical_data import membership_mask
from historical_membership import reconcile_snapshots, reconstruct


class MembershipReplayTests(unittest.TestCase):
    def test_late_announced_join_and_exclusive_removal(self):
        seed = pd.DataFrame([dict(security_id="A", index="SP500", known_at="2020-01-01", source="roster")])
        changes = pd.DataFrame([
            dict(security_id="B", index="SP500", effective_date="2020-01-02", known_at="2020-01-03",
                 action="join", source="join notice"),
            dict(security_id="A", index="SP500", effective_date="2020-01-04", known_at="2020-01-03",
                 action="leave", source="removal notice"),
        ])
        intervals = reconstruct(seed, changes, "2020-01-01", "2020-01-04")
        actual = membership_mask(intervals, pd.date_range("2020-01-01", periods=4), ["A", "B"])
        self.assertEqual(actual["A"].tolist(), [True, True, True, False])
        self.assertEqual(actual["B"].tolist(), [False, False, True, True])
        with self.assertRaisesRegex(ValueError, "removal was not known"):
            reconstruct(seed, changes.assign(known_at=["2020-01-03", "2020-01-05"]), "2020-01-01", "2020-01-04")

    def test_reconciles_membership_at_effective_and_known_dates(self):
        intervals = pd.DataFrame([
            ("A", "NASDAQ100", "2020-01-01", "2020-02-01", "2020-01-01"),
            ("B", "NASDAQ100", "2020-02-01", None, "2020-02-02"),
        ], columns=["security_id", "index", "start", "end", "known_at"])
        for column in ["start", "end", "known_at"]:
            intervals[column] = pd.to_datetime(intervals[column])
        snapshots = pd.DataFrame([
            ("NASDAQ100", "2020-01-31", "A"),
            ("NASDAQ100", "2020-02-01", "B"),
            ("NASDAQ100", "2020-02-02", "A"),
        ], columns=["index", "date", "security_id"])
        first, second, third = reconcile_snapshots(intervals, snapshots)
        self.assertTrue(first["matches"])
        self.assertEqual(second["missing"], ["B"])
        self.assertEqual(third["missing"], ["A"])
        self.assertEqual(third["unexpected"], ["B"])
