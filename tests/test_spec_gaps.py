"""Specification regressions using synthetic data and temporary files; no network or broker access."""
import copy
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import gates
import publication
from pit_clinical import phase3_asof


class PromotionTests(unittest.TestCase):
    def test_positive_executed_book_without_forward_evidence_cannot_promote(self):
        book = {"closed_trades": 30, "sessions_since_first": 80, "mean_net_excess": .02, "t": 3}
        self.assertFalse(gates.live_review_ready(book, {})[0])

    def test_missing_failed_or_unrelated_forward_evidence_blocks(self):
        book = {"closed_trades": 30, "sessions_since_first": 80, "mean_net_excess": .02, "t": 3,
                "by_model": {"mosaic-xs-ridge-broad": {"n": 30, "sessions_since_first": 80, "mean_net_excess": .02, "t": 3, "horizons": [20]}}}
        good = {"broad_h20": {"scored_dates": 70, "forward_ic_mean": .03, "forward_ic_t": 3,
                               "non_overlapping_cohorts": 4, "top_quintile_net_mean": .01}}
        self.assertTrue(gates.live_review_ready(book, good)[0])
        for key, val in [("forward_ic_mean", -.01), ("forward_ic_t", 1), ("scored_dates", 62),
                         ("top_quintile_net_mean", -.01), ("forward_ic_mean", float("nan"))]:
            bad = copy.deepcopy(good)
            bad["broad_h20"][key] = val
            self.assertFalse(gates.live_review_ready(book, bad)[0], key)
        self.assertFalse(gates.live_review_ready(book, {"large_h20": good["broad_h20"]})[0])
        bad_book = copy.deepcopy(book)
        bad_book["by_model"]["mosaic-xs-ridge-broad"]["t"] = float("nan")
        self.assertFalse(gates.live_review_ready(bad_book, good)[0])
        bad_book["by_model"]["mosaic-xs-ridge-broad"].update(t=3, horizons=[5])
        self.assertFalse(gates.live_review_ready(bad_book, good)[0])


def signal():
    schema = publication.SCHEMA
    # Every property in the actual required contract is supplied; unknowns stay null.
    s = {k: None for k in schema["required"]}
    s.update(schema_version="2.0", generated_at_utc="2026-10-22T20:00:00Z", as_of_utc="2026-10-22T20:00:00Z")
    for group in ("asset", "forecast", "validation", "risk", "signal", "provenance", "regime"):
        props = schema["properties"][group]
        s[group] = {k: None for k in props["required"]}
    s["asset"]["asset_class"] = "Equity"
    s["validation"]["status"] = "UNVALIDATED"
    s["risk"].update(liquidity_approved=False, data_quality_approved=False)
    s["signal"].update(direction="NO_SIGNAL", deployment_eligibility="UNVALIDATED", key_drivers=[], contradictory_evidence=[])
    s["provenance"]["source_ids"] = []
    return s


class PublicationTests(unittest.TestCase):
    def test_invalid_or_nonfinite_update_preserves_existing_publication(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            publication.publish({"signals": [signal()]}, root, "first")
            original = (root / "latest.json").read_bytes()
            for value in ("not numeric", float("nan"), float("inf")):
                s = signal()
                s["forecast"]["expected_excess_return"] = value
                with self.assertRaises(ValueError):
                    publication.publish({"signals": [s]}, root, "bad")
                self.assertEqual((root / "latest.json").read_bytes(), original)
            self.assertEqual(len(list((root / "log").glob("*"))), 1)

    def test_logs_are_exclusive_and_nulls_survive(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            publication.publish({"signals": [signal()]}, root, "same")
            with self.assertRaises(FileExistsError):
                publication.publish({"signals": [signal()]}, root, "same")
            self.assertIsNone(json.loads((root / "latest.json").read_text())["signals"][0]["forecast"]["predictive_std"])


class ClinicalTests(unittest.TestCase):
    def test_revised_snapshot_cannot_change_earlier_features(self):
        old = {"snapshot_id": "old", "observed_at_utc": "2026-01-02T12:00:00Z", "cik": 1,
               "nct": "N1", "phase": "PHASE3", "primary_completion": "2027-01-01"}
        revised = {**old, "snapshot_id": "new", "observed_at_utc": "2026-01-05T12:00:00Z", "phase": "PHASE2"}
        dates = pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-06"])
        a = phase3_asof(pd.DataFrame([old]), dates, [1, 2])
        b = phase3_asof(pd.DataFrame([old, revised]), dates, [1, 2])
        self.assertTrue(np.isnan(b.loc[dates[0], 1]))
        self.assertEqual(a.loc[dates[1], 1], b.loc[dates[1], 1])
        self.assertEqual(b.loc[dates[1], 1], 1)
        self.assertEqual(b.loc[dates[2], 1], 0)
        self.assertTrue(b[2].isna().all())

    def test_unversioned_registry_history_stays_missing(self):
        df = phase3_asof(pd.DataFrame([{"cik": 1, "phase": "PHASE3"}]), pd.to_datetime(["2020-01-01"]), [1])
        self.assertTrue(df.isna().all().all())


class IntradayContractTests(unittest.TestCase):
    def test_nightly_publication_uses_same_contract(self):
        import config
        import signals
        s = signal()
        day = pd.Timestamp("2026-10-22")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            reports = root / "reports"
            reports.mkdir()
            fc = pd.DataFrame([{"profile": "broad", "model_run_id": "new", "symbol": "A", "horizon": 20}])
            result = ([s], fc, {"profile": "broad", "as_of": "2026-10-22", "model_run_id": "new", "evaluated": 1,
                                  "model_gate_primary": {"status": "UNVALIDATED"}}, day, True, True)
            with patch.object(config, "SIGNALS", root / "signals"), patch.object(config, "REPORTS", reports), \
                    patch.object(config, "PROC", root / "processed"), patch.object(signals, "dataset_version", return_value="ds-test"), \
                    patch.object(signals, "audit_ok", return_value=(True, None)), patch.object(signals.regime, "field", return_value={}), \
                    patch.object(signals, "profile_signals", return_value=result):
                signals.run(profiles=["broad"])
            published = json.loads((root / "signals/latest.json").read_text())
            publication.validate_signal(published["signals"][0])
            self.assertEqual(pd.read_parquet(reports / "forecast_latest.parquet")["model_run_id"].tolist(), ["new"])

    def test_intraday_revalidates_and_blocks_older_longs_on_failed_audit(self):
        import config
        import insider_model
        import intraday_watch
        import regime
        import signals
        s = signal()
        s["model_id"] = "mosaic-xs-ridge-broad"
        s["signal"].update(direction="LONG", deployment_eligibility="PAPER_TEST_ELIGIBLE")
        s["validation"]["status"] = "PAPER_TEST_ELIGIBLE"
        s["risk"].update(liquidity_approved=True, data_quality_approved=True)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            doc = {"as_of_utc": "2026-10-22T20:00:00Z", "signals": [s]}
            (root / "latest.json").write_text(json.dumps(doc))
            with patch.object(config, "SIGNALS", root), patch.object(signals, "dataset_version", return_value="new"), \
                    patch.object(signals, "audit_ok", return_value=(False, "dataset changed")), \
                    patch.object(insider_model, "live_signals", return_value=[]), patch.object(regime, "field", return_value={}):
                intraday_watch.refresh_insider_signals()
            updated = json.loads((root / "latest.json").read_text())
            self.assertEqual(updated["signals"][0]["signal"]["direction"], "NO_SIGNAL")
            self.assertFalse(updated["signals"][0]["risk"]["data_quality_approved"])
            self.assertEqual(updated["signals"][0]["signal"]["reason"], "dataset changed")


if __name__ == "__main__":
    unittest.main()
