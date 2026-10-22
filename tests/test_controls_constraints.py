import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import insider_model
from paper_trading import current_history
import paper_trading
import provenance
import config
from unittest.mock import patch
from portfolio_constraints import load
from risk_model import optimise


class InsiderControlsTests(unittest.TestCase):
    def test_match_returns_control_company_covariates(self):
        # Event A's beta/reversal are deliberately very different from B's.
        m = SimpleNamespace(sym_of={1: "A", 2: "B"}, syms=["A", "B"], sector=np.array(["Tech", "Tech"]),
                            dv=np.array([[30e6, 40e6]]), close=np.array([[10., 20.]]),
                            mom=np.array([[.1, .2]]), rev=np.array([[.9, -.4]]), beta=np.array([[3., .5]]),
                            ar=lambda j, e, h: .07)
        ev = pd.DataFrame([{"e": 1, "j": 0, "sector": "Tech", "fdate": pd.Timestamp("2026-01-02"),
                            "entry_date": pd.Timestamp("2026-01-05")}], index=[17])
        buys = pd.DataFrame({"issuercik": [1], "filing_date": pd.to_datetime(["2026-01-02"])})
        control = insider_model.matched_controls(m, ev, buys, 5, details=True)
        self.assertEqual(control.loc[17, "symbol"], "B")
        self.assertEqual(control.loc[17, "beta"], .5)
        self.assertEqual(control.loc[17, "rev"], -.4)
        self.assertAlmostEqual(control.loc[17, "ldv"], np.log(40e6))

    def test_sparse_regression_reports_not_computed(self):
        cols = ["ar", "mom", "rev", "ldv", "beta", "sector", "entry_date"]
        ev = pd.DataFrame([[.1, .2, .3, 20., 1., "Tech", pd.Timestamp("2026-01-02")]], columns=cols)
        result = insider_model.incremental_regression(ev, ev)
        self.assertIsNone(result["t"])
        self.assertIn("NOT_COMPUTED", result["status"])


class ForwardIdentityTests(unittest.TestCase):
    def test_record_deduplicates_per_security_and_preserves_recipe_boundaries(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            signals_dir, reports = root / "signals", root / "reports"
            signals_dir.mkdir()
            reports.mkdir()
            fc = pd.DataFrame([{"profile": "broad", "model_run_id": "new", "symbol": "A", "horizon": 20,
                                "ensemble_rank": .9, "decile": 9, "exp_excess": .03, "cost_round_trip": .0046, "gate": "UNVALIDATED"}])
            fc.to_parquet(reports / "forecast_latest.parquet")
            (signals_dir / "latest.json").write_text(json.dumps({"profiles": [{"profile": "broad", "as_of": "2026-10-22", "model_run_id": "new"}]}))
            history_path = root / "history.parquet"
            with patch.object(config, "SIGNALS", signals_dir), patch.object(config, "REPORTS", reports), patch.object(paper_trading, "HIST", history_path):
                paper_trading.record()
                paper_trading.record()
                self.assertEqual(len(pd.read_parquet(history_path)), 1)
                fc["symbol"] = "B"
                fc.to_parquet(reports / "forecast_latest.parquet")
                paper_trading.record()
                self.assertEqual(len(pd.read_parquet(history_path)), 2)
                fc["model_run_id"] = "stale"
                fc.to_parquet(reports / "forecast_latest.parquet")
                with self.assertRaises(ValueError):
                    paper_trading.record()
                self.assertEqual(len(pd.read_parquet(history_path)), 2)

    def test_old_or_unknown_recipes_do_not_unlock_new_recipe(self):
        history = pd.DataFrame({"profile": ["broad", "broad", "broad", "large"],
                                "model_run_id": ["old", "new", None, "other"], "value": [1, 2, 3, 4]})
        current = current_history(history, {"broad": "new"})
        self.assertEqual(current["value"].tolist(), [2])
        self.assertTrue(current_history(history.drop(columns="model_run_id"), {"broad": "new"}).empty)

    def test_fingerprint_changes_when_nonbenchmark_input_changes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            raw, proc = root / "data/raw", root / "data/processed"
            (raw / "prices").mkdir(parents=True)
            proc.mkdir(parents=True)
            for name in ("features.py", "pit_clinical.py", "config.py"):
                (root / name).write_text("source")
            price = raw / "prices/AAPL.parquet"
            price.write_bytes(b"old")
            with patch.object(config, "ROOT", root), patch.object(config, "RAW", raw), patch.object(config, "PROC", proc):
                before = provenance.dataset_version()
                price.write_bytes(b"changed non-benchmark price")
                self.assertNotEqual(before, provenance.dataset_version())


POLICY = {"source": "test external policy", "gross_limit": .8, "sector_cap": .6, "turnover_cap": 1.,
          "beta_min": 0, "beta_max": .4, "max_participation": .01, "risk_aversion": 1., "cost_horizon_sessions": 20}
LIMITS = {"long_only": True, "leverage": False, "position_pct": .6, "account_total_usd": 100000.}


class OptimizerTests(unittest.TestCase):
    def test_missing_or_invalid_external_limits_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "policy.json"
            for policy in ({}, {**POLICY, "gross_limit": None}, {**POLICY, "max_participation": -1}):
                path.write_text(json.dumps(policy))
                with self.assertRaises(ValueError):
                    load(path, LIMITS)

    def test_beta_participation_and_exit_turnover_are_enforced(self):
        class Risk:
            X = pd.DataFrame({"sector": ["Tech", "Energy", "Tech"]}, index=["A", "B", "OLD"])
            def sigma(self, syms):
                return np.eye(len(syms)) * .01, None
        mu = pd.Series({"A": 1., "B": .8})
        held = pd.Series({"OLD": .1})
        cost = pd.Series({"A": .001, "B": .001, "OLD": .001})
        beta = pd.Series({"A": 2., "B": .5, "OLD": 1.})
        adv = pd.Series({"A": 1e6, "B": 5e6, "OLD": 2e6})
        w = optimise(mu, Risk(), held, LIMITS, cost, POLICY, beta, adv)
        self.assertIsNotNone(w)
        self.assertLessEqual(w["A"], .100001)  # $10k participation / $100k equity
        self.assertLessEqual(float(w @ beta.reindex(w.index)), .400001)
        self.assertLess(w["OLD"], 1e-6)
        self.assertLessEqual(float((w - held.reindex(w.index, fill_value=0)).abs().sum()), 1.000001)
        infeasible = optimise(mu, Risk(), held, LIMITS, cost, {**POLICY, "turnover_cap": .05}, beta, adv)
        self.assertIsNone(infeasible)  # liquidating OLD alone needs 10%, cannot be ignored


if __name__ == "__main__":
    unittest.main()
