"""NYSE calendar and long/short selection rules."""
import sys
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ls_rules as R  # noqa: E402
from provenance import nyse_holidays, sessions_between  # noqa: E402


class CalendarTests(unittest.TestCase):
    def test_known_holidays(self):
        h = nyse_holidays(2026)
        for d in [date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3), date(2026, 5, 25),
                  date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26), date(2026, 12, 25)]:
            self.assertIn(d, h, d)
        self.assertEqual(len(h), 10)
        self.assertIn(date(2024, 3, 29), nyse_holidays(2024))   # Good Friday
        self.assertNotIn(date(2021, 12, 31), nyse_holidays(2022))  # New Year on a Saturday is not observed on Friday

    def test_sessions_skip_holidays(self):
        self.assertEqual(sessions_between("2026-11-25", "2026-11-30"), 2)  # Wed, (Thanksgiving), Fri
        self.assertEqual(sessions_between("2026-10-19", "2026-10-26"), 5)


class SelectionTests(unittest.TestCase):
    def frame(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({"symbol": [f"S{i}" for i in range(n)], "sector": np.repeat(["A", "B", "C", "D"], n // 4),
                             "score": rng.random(n), "beta": rng.uniform(0.5, 1.5, n), "si": rng.uniform(0, 0.1, n)})

    def test_beta_neutral_and_gross(self):
        for n, full in [(2000, True), (200, False)]:  # small universes hit the per-name cap: gross shrinks, neutrality holds
            g = self.frame(n)
            w = R.select(g)
            beta = dict(zip(g["symbol"], R.blume(g["beta"])))
            gross = sum(abs(x) for x in w.values())
            if full:
                self.assertAlmostEqual(gross, R.LS_GROSS, places=6)
            else:
                self.assertLess(gross, R.LS_GROSS)
            self.assertAlmostEqual(sum(x * beta[s] for s, x in w.items()), 0.0, places=6)
            self.assertTrue(all(abs(x) <= R.MAX_NAME_WEIGHT + 1e-12 for x in w.values()))

    def test_hard_to_borrow_and_unshortable_names_are_not_shorted(self):
        g = self.frame()
        g.loc[g["score"] < 0.2, "si"] = 0.5
        w = R.select(g)
        self.assertFalse(any(x < 0 and s in set(g.loc[g["si"] > R.SI_MAX_SHORT, "symbol"]) for s, x in w.items()))
        w = R.select(self.frame(), can_short=lambda s: False)
        self.assertEqual(w, {})

    def test_buffer_keeps_holdings(self):
        g = self.frame()
        keep = g.loc[g.groupby("sector")["score"].rank(pct=True).between(0.7, 0.85), "symbol"].iloc[0]
        self.assertNotIn(keep, R.select(g))
        self.assertGreater(R.select(g, held_long={keep})[keep], 0)


if __name__ == "__main__":
    unittest.main()
