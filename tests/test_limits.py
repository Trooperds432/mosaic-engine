"""Optimiser limits are taken from the bot when it defines them and labelled as engine defaults otherwise."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class LimitTests(unittest.TestCase):
    def test_bot_values_win_and_defaults_are_labelled(self):
        try:
            from risk_model import ENGINE_DEFAULT_LIMITS, resolve_limits
        except ImportError as e:  # risk_model needs the full research stack; CI installs only numpy/pandas
            self.skipTest(str(e))
        vals, src = resolve_limits({"sector_cap": 0.2})
        self.assertEqual((vals["sector_cap"], src["sector_cap"]), (0.2, "bot"))
        for k in ENGINE_DEFAULT_LIMITS:
            if k != "sector_cap":
                self.assertEqual(src[k], "ENGINE_DEFAULT_NOT_BOT_SUPPLIED")


if __name__ == "__main__":
    unittest.main()
