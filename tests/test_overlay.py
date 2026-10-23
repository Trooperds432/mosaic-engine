"""Analyst overlay: vetoes only remove, expire on schedule, are logged, and are scored."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
import ls_rules as R  # noqa: E402
import overlay as O  # noqa: E402


def sig(ticker, direction):
    return {"asset": {"ticker": ticker}, "signal": {"direction": direction, "reason": None}}


class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.p = [patch.object(O, "DIR", root / "overlay"), patch.object(O, "VETOES", root / "overlay" / "vetoes.csv"),
                  patch.object(O, "NOTES", root / "notes.jsonl"), patch.object(O, "ACTIONS", root / "actions.jsonl"),
                  patch.object(C, "LEDGER", root)]
        for p in self.p:
            p.start()
        self.root = root

    def tearDown(self):
        for p in self.p:
            p.stop()
        self.tmp.cleanup()

    def test_veto_needs_reason_and_valid_code(self):
        with self.assertRaises(ValueError):
            O.veto("AAPL", "accounting", "  ")
        with self.assertRaises(ValueError):
            O.veto("AAPL", "gut_feeling", "x")

    def test_veto_only_removes_longs_and_is_logged(self):
        v = O.veto("brk-b", "pending_event", "merger vote next week", side="long")
        act = O.active_vetoes()
        self.assertIn("BRK.B", act)
        sigs = [sig("BRK.B", "LONG"), sig("BRK.B", "NO_SIGNAL"), sig("MSFT", "LONG")]
        self.assertEqual(O.apply_to_signals(sigs, act), 1)
        self.assertEqual([s["signal"]["direction"] for s in sigs], ["NO_SIGNAL", "NO_SIGNAL", "LONG"])
        self.assertIn(v["veto_id"], sigs[0]["signal"]["reason"])
        self.assertEqual(json.loads((self.root / "actions.jsonl").read_text().splitlines()[0])["event"], "veto")

    def test_short_veto_does_not_touch_longs_and_lift_ends_it(self):
        v = O.veto("XYZ", "data_error", "bad share count", side="short")
        sigs = [sig("XYZ", "LONG")]
        self.assertEqual(O.apply_to_signals(sigs, O.active_vetoes()), 0)
        O.lift(v["veto_id"], "fixed")
        self.assertEqual(O.active_vetoes(), {})

    def test_expiry_counts_exchange_sessions(self):
        self.assertEqual(O.add_sessions("2026-11-25", 1), "2026-11-27")  # skips Thanksgiving
        v = O.veto("ABC", "governance", "auditor resigned", days=5)
        self.assertNotIn("ABC", O.active_vetoes(v["expires"]))
        self.assertIn("ABC", O.active_vetoes(v["start"]))

    def test_long_short_book_excludes_vetoed_side_even_if_held(self):
        rng = np.random.default_rng(1)
        g = pd.DataFrame({"symbol": [f"S{i}" for i in range(400)], "sector": np.repeat(list("ABCD"), 100), "score": rng.random(400),
                          "beta": 1.0, "si": 0.01})
        top = g.loc[g.groupby("sector")["score"].idxmax(), "symbol"].iloc[0]
        bottom = g.loc[g.groupby("sector")["score"].idxmin(), "symbol"].iloc[0]
        self.assertGreater(R.select(g)[top], 0)
        g["veto_long"], g["veto_short"] = g["symbol"].eq(top), g["symbol"].eq(bottom)
        w = R.select(g, held_long={top}, held_short={bottom})
        self.assertNotIn(top, w)
        self.assertNotIn(bottom, w)

    def test_scorecard_signs_value_added(self):
        prices = self.root / "prices"
        prices.mkdir()
        d = pd.bdate_range("2026-01-02", periods=30)
        pd.DataFrame({"date": d, "adj_open": np.linspace(100, 90, 30)}).to_parquet(prices / "AAA.parquet")
        pd.DataFrame({"date": d, "adj_open": np.full(30, 100.0)}).to_parquet(prices / f"{C.BENCHMARK}.parquet")
        pd.DataFrame([{"veto_id": "v1", "created_utc": "", "ticker": "AAA", "side": "long", "reason": "accounting", "note": "x",
                       "analyst": "me", "start": "2026-01-02", "expires": "2026-02-10", "lifted_utc": "", "lift_note": ""}]).to_csv(O.VETOES.parent.mkdir(parents=True) or O.VETOES, index=False)
        with patch.object(C, "RAW", self.root), patch.object(C, "REPORTS", self.root):
            out = O.score()
        self.assertGreater(out["vetoes"][0]["value_added"], 0)  # the vetoed long fell: the veto helped


if __name__ == "__main__":
    unittest.main()
