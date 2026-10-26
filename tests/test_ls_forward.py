"""Forward record of the long/short book: next-open entry, drift, one-off costs, borrow, recipe restarts, gate."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
import ls_forward as F  # noqa: E402


class ForwardBookTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "prices").mkdir()
        self.cal = pd.bdate_range("2026-01-05", periods=10)
        px = {"AAA": np.linspace(100, 110, 10), "BBB": np.full(10, 50.0), C.BENCHMARK: np.full(10, 400.0)}
        for s, p in px.items():
            pd.DataFrame({"date": self.cal, "adj_open": p}).to_parquet(root / "prices" / f"{s}.parquet")
        self.p = patch.object(C, "RAW", root)
        self.p.start()

    def tearDown(self):
        self.p.stop()
        self.tmp.cleanup()

    def test_entry_next_open_drift_cost_and_borrow(self):
        st = F._new_state("r1", "m1")
        with patch.object(F.R, "select", return_value={"AAA": 0.5, "BBB": -0.5}):
            F.rebalance(st, pd.DataFrame(), str(self.cal[1].date()), self.cal, {"AAA": 0.001, "BBB": 0.001})
        self.assertEqual(st["last_marked"], str(self.cal[2].date()))  # entered at the next session's open
        st["si"] = {"BBB": 0.12}  # 3% borrow tier
        F.mark(st, self.cal)
        self.assertEqual(len(st["sessions"]), 7)
        r = pd.Series([x["net"] for x in st["sessions"]])
        self.assertLess(r.iloc[0], r.iloc[1])  # entry cost (0.1% of 1.0 traded) charged once, in the first session
        long_gain = 0.5 * (110 / np.linspace(100, 110, 10)[2] - 1)
        borrow = 0.5 * 0.03 / 252 * 7
        self.assertAlmostEqual(st["equity"], 1.0 + long_gain - 0.001 - borrow, delta=1e-4)

    def test_pending_when_entry_open_not_available(self):
        st = F._new_state("r1", "m1")
        with patch.object(F.R, "select", return_value={"AAA": 0.5, "BBB": -0.5}):
            F.rebalance(st, pd.DataFrame(), str(self.cal[-1].date()), self.cal, {})
        self.assertIsNotNone(st["pending"])
        self.assertEqual(st["positions"], {})

    def test_gate_needs_frozen_recipe_sessions_rebalances_and_positive_t(self):
        st = F._new_state("r1", "m1")
        st["sessions"] = [{"date": "x", "net": 0.001 + 0.0001 * (i % 3)} for i in range(80)]
        st["rebalances"] = 4
        self.assertTrue(F.evaluate(st, True)["strategy_forward_validated"])
        self.assertFalse(F.evaluate(st, False)["strategy_forward_validated"])
        st["rebalances"] = 2
        self.assertFalse(F.evaluate(st, True)["strategy_forward_validated"])
        st["rebalances"], st["sessions"] = 4, st["sessions"][:40]
        self.assertFalse(F.evaluate(st, True)["strategy_forward_validated"])


if __name__ == "__main__":
    unittest.main()
