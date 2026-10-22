import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
import pandas as pd

import config as C
import historical_data as H
import recipe
from deployment_policy import cumulative, execution_target_allowed
from macro_vintages import initial_releases
from pit_estimates import validate as validate_estimates, revision


class HistoricalTests(unittest.TestCase):
    def test_constituents_are_filtered_by_effective_and_known_dates(self):
        members = pd.DataFrame([{"security_id": "S1", "start": "2020-01-01", "end": "2020-01-05", "known_at": "2020-01-02"},
                                {"security_id": "S2", "start": "2020-01-01", "end": None, "known_at": "2020-01-04"}])
        days = pd.date_range("2020-01-01", periods=6)
        mask = H.membership_mask(members, days, ["S1", "S2", "UNLISTED"])
        self.assertEqual(mask["S1"].tolist(), [False, True, True, True, False, False])
        self.assertEqual(mask["S2"].tolist(), [False, False, False, True, True, True])
        self.assertFalse(mask["UNLISTED"].any())

    def test_verified_total_loss_is_minus_one_and_unknown_exit_is_nan(self):
        days = pd.date_range("2020-01-01", periods=5)
        opens = pd.DataFrame({"FAILED": [10, 10, np.nan, np.nan, np.nan], "UNKNOWN": [10, 10, np.nan, np.nan, np.nan]}, index=days)
        terminal = pd.DataFrame([{"security_id": "FAILED", "last_trade_date": days[1], "settlement_date": days[3], "terminal_adjclose": 0}])
        returns = H.forward_returns(opens, terminal, np.array([1, 1]), np.array([2, 4]))
        self.assertTrue(np.isnan(returns[0]).all())
        self.assertEqual(returns[1, 0], -1)
        self.assertTrue(np.isnan(returns[1, 1]))

    def test_partial_terminal_payout_is_held_as_cash(self):
        days = pd.date_range("2020-01-01", periods=4)
        op = pd.DataFrame({"S": [10, 10, np.nan, np.nan]}, index=days)
        terminal = pd.DataFrame([{"security_id": "S", "last_trade_date": days[1], "settlement_date": days[2], "terminal_adjclose": 4}])
        self.assertAlmostEqual(H.forward_returns(op, terminal, [1], [3])[0, 0], -.6)

    def test_missing_historical_bundle_cannot_pass_coverage(self):
        with tempfile.TemporaryDirectory() as d, patch.object(C, "PROC", Path(d)):
            self.assertFalse(H.coverage_check("2005-01-01", "2026-01-01")["passed"])


class MacroTests(unittest.TestCase):
    def test_initial_release_api_and_publication_latency(self):
        def handler(request):
            self.assertEqual(request.url.params["output_type"], "4")
            self.assertEqual(request.url.params["realtime_start"], "1776-07-04")
            return httpx.Response(200, json={"count": 1, "observations": [{"date": "2020-01-01", "realtime_start": "2020-02-07", "value": "3.6"}]})
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            result = initial_releases(client, "UNRATE", "test-key")
        self.assertEqual(result.iloc[0]["available_date"], pd.Timestamp("2020-02-08"))
        self.assertEqual(result.iloc[0]["source"], "fred_initial_release_api")
        self.assertEqual(result.iloc[0]["value"], 3.6)

    def test_api_error_or_no_key_cannot_become_an_initial_release(self):
        with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(403))) as client:
            for key in [None, "denied"]:
                with self.assertRaises(ValueError):
                    initial_releases(client, "UNRATE", key)


class EstimateTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([{"security_id": "S", "fiscal_period": "2021FY", "consensus_eps": x,
                              "available_at_utc": stamp, "observed_at_utc": stamp, "timestamp_basis": "observed_snapshot", "source": "provider"}
                             for x, stamp in [(2, "2020-01-01T12:00Z"), (3, "2020-02-03T12:00Z"), (100, "2020-03-01T12:00Z")]])

    def test_estimate_revision_uses_same_period_and_only_known_observations(self):
        frame = validate_estimates(self.frame())
        self.assertEqual(revision(frame, "S", "2020-02-04T16:00Z"), .5)
        self.assertTrue(np.isnan(revision(frame, "S", "2019-12-31T16:00Z")))
        frame.loc[1, "fiscal_period"] = "2022FY"
        self.assertTrue(np.isnan(revision(frame, "S", "2020-02-04T16:00Z")))

    def test_snapshot_cannot_be_backdated(self):
        frame = self.frame()
        frame.loc[0, "observed_at_utc"] = "2020-06-01T00:00Z"
        with self.assertRaises(ValueError):
            validate_estimates(frame)


class PolicyTests(unittest.TestCase):
    def test_all_stages_are_required(self):
        good = {"status": "PAPER_TEST_ELIGIBLE", "failed_checks": []}
        self.assertEqual(cumulative(good, good, data_ok=True, recipe_ok=True)["status"], "PAPER_TEST_ELIGIBLE")
        for historical, forward, data, frozen in [(None, good, True, True), (good, None, True, True), (good, good, False, True), (good, good, True, False)]:
            self.assertNotEqual(cumulative(historical, forward, data_ok=data, recipe_ok=frozen)["status"], "PAPER_TEST_ELIGIBLE")

    def test_paper_broker_cannot_trade_exploratory_targets(self):
        doc = {"mode": "RESEARCH_ONLY", "names": [{"ticker": "A"}], "recipe_id": "a" * 64,
               "deployment": {"status": "PAPER_TEST_ELIGIBLE", "failed_checks": []}}
        self.assertFalse(execution_target_allowed(doc))
        doc["mode"] = "VALIDATED_PAPER"
        self.assertTrue(execution_target_allowed(doc))
        doc["deployment"]["failed_checks"] = ["failed validation"]
        self.assertFalse(execution_target_allowed(doc))


class RecipeTests(unittest.TestCase):
    def test_code_weights_and_contract_are_frozen_independently_of_refit_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            proc = root / "data/processed"
            proc.mkdir(parents=True)
            (root / "models").mkdir()
            (root / "requirements.txt").write_text("numpy==2.5.4")
            (root / "features.py").write_text("version = 1")
            weights = root / "models/broad_h20_ridge.joblib"
            weights.write_bytes(b"weights")
            model = {"model_id": "mosaic-xs-ridge-broad", "profile": "broad", "run_id": "run-1", "feats": ["value"], "features": ["value"],
                     "horizons": {"20": {"names": ["ridge"], "members": ["ridge"], "gate": {"status": "UNVALIDATED"}}}}
            path = proc / "model_broad.json"
            path.write_text(json.dumps(model))
            with patch.object(C, "ROOT", root), patch.object(C, "PROC", proc):
                identity = recipe.freeze("broad")
                model = json.loads(path.read_text())
                self.assertEqual(model["recipe_id"], identity)
                self.assertTrue(recipe.check(model)[0])
                model["weights_refit_utc"] = "tomorrow"
                self.assertTrue(recipe.check(model)[0])
                weights.write_bytes(b"modified")
                self.assertFalse(recipe.check(model)[0])
                recipe.snapshot_weights(model)
                self.assertTrue(recipe.check(model)[0])
                (root / "features.py").write_text("version = 2")
                self.assertFalse(recipe.check(model)[0])
                with self.assertRaises(ValueError):
                    recipe.freeze("broad")


if __name__ == "__main__":
    unittest.main()
