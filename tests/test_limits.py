"""Explicit external policy replaces invented optimiser defaults."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class LimitTests(unittest.TestCase):
    def test_missing_limits_have_no_engine_default_fallback(self):
        from portfolio_constraints import validate
        with self.assertRaises(ValueError):
            validate({}, {"long_only": True, "leverage": False})

    def test_supplied_policy_preserves_values_and_source(self):
        from portfolio_constraints import validate
        policy = {"source": "approved external policy", "gross_limit": .8, "sector_cap": .2,
                  "turnover_cap": .4, "beta_min": 0, "beta_max": 1.1, "max_participation": .005,
                  "risk_aversion": 3, "cost_horizon_sessions": 20}
        result = validate(policy, {"long_only": True, "leverage": False})
        self.assertEqual(result, policy)
        restricted = validate(policy, {"long_only": True, "leverage": False, "sector_cap": .1, "max_beta": .7})
        self.assertEqual(restricted["sector_cap"], .1)
        self.assertEqual(restricted["beta_max"], .7)
        self.assertEqual(policy["sector_cap"], .2)  # validation never mutates approved policy


if __name__ == "__main__":
    unittest.main()
