import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

import insider_study_v3 as v3


def market():
    cal = pd.bdate_range("2016-01-04", periods=40)
    opens = pd.DataFrame(100.0, index=cal, columns=["A", "B", "T"])
    opens.loc[cal[12], "B"] = np.nan                      # B has a gap inside an event window
    opens.loc[cal[20]:, "T"] = np.nan                     # T stops trading after a terminal event
    terminal = pd.DataFrame({"security_id": ["T"], "last_trade_date": [cal[19]]})
    return SimpleNamespace(cal=cal, observed=cal, opens=opens, terminal=terminal)


def events(m, rows):
    return pd.DataFrame([{"e": e, "x": e + 5, "entry_date": m.cal[e] if e < len(m.cal) else pd.Timestamp("2027-01-01"),
                          "symbol": s, "cik": c} for s, c, e in rows])


class ScopeRuleTests(unittest.TestCase):
    def test_recipe_matches_v2_and_is_registered(self):
        v3.check_recipe()
        self.assertEqual(v3.REG["status"], "REGISTERED")

    def test_price_gap_excluded_but_terminal_end_is_not_a_gap(self):
        m = market()
        ev = events(m, [("A", 1, 10), ("B", 2, 10), ("T", 3, 17)])
        kept, counts = v3.make_filter(m, {}, concurrent=False)(ev)
        self.assertEqual(sorted(kept["symbol"]), ["A", "T"])
        self.assertEqual(counts["price_gap"], 1)

    def test_concurrent_8k_window_runs_two_sessions_before_entry_to_exit(self):
        m = market()
        ev = events(m, [("A", 1, 10), ("A", 1, 25), ("A", 1, 30)])
        sessions = {1: np.array([8, 27])}
        kept, counts = v3.make_filter(m, sessions)(ev)
        # e=10: window [8, 15] contains 8; e=25: [23, 30] contains 27; e=30: [28, 35] contains neither
        self.assertEqual(kept["e"].tolist(), [30])
        self.assertEqual(counts["concurrent_8k"], 2)

    def test_concurrent_boundaries(self):
        m = market()
        ev = events(m, [("A", 1, 10)])
        flt = v3.make_filter(m, {1: np.array([7])})        # 3 sessions before entry: outside the window
        self.assertEqual(len(flt(ev)[0]), 1)
        flt = v3.make_filter(m, {1: np.array([15])})       # exit session (x = e + 5): inside
        self.assertEqual(len(flt(ev)[0]), 0)
        flt = v3.make_filter(m, {1: np.array([16])})       # after exit: outside
        self.assertEqual(len(flt(ev)[0]), 1)
        flt = v3.make_filter(m, {2: np.array([10])})       # another issuer: ignored
        self.assertEqual(len(flt(ev)[0]), 1)

    def test_window_bounds(self):
        m = market()
        ev = events(m, [("A", 1, 10)])
        ev.loc[0, "entry_date"] = pd.Timestamp("2026-11-02")
        kept, counts = v3.make_filter(m, {}, concurrent=False)(ev)
        self.assertTrue(kept.empty)
        self.assertEqual(counts["outside_window"], 1)


if __name__ == "__main__":
    unittest.main()
