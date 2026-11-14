import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import config as C
from provenance import _content_hash, panel_identity


class PanelIdentityTests(unittest.TestCase):
    def test_stale_input_and_changed_panel_are_ineligible(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            panel = root / "panel_large.parquet"
            panel.write_bytes(b"original")
            (root / "panel_meta_large.json").write_text(json.dumps({
                "profile": "large", "feature_set_version": C.FEATURE_SET_VERSION,
                "dataset_version": "v1", "panel_sha256": _content_hash(panel)}))
            with patch.object(C, "PROC", root):
                self.assertTrue(panel_identity("large", "v1")[0])
                self.assertFalse(panel_identity("large", "v2")[0])
                panel.write_bytes(b"modified")
                self.assertFalse(panel_identity("large", "v1")[0])

    def test_baseline_refuses_mixed_research_snapshot(self):
        from baselines import verify_research_inputs

        with TemporaryDirectory() as directory:
            root = Path(directory)
            panel = root / "panel_large.parquet"
            panel.write_bytes(b"original")
            report = {"profile": "large", "panel_provenance": {
                "dataset_version": "v1", "panel_sha256": _content_hash(panel)}}
            with patch.object(C, "PROC", root), patch("provenance.dataset_version", return_value="v1"):
                verify_research_inputs(report, "large")
                panel.write_bytes(b"different")
                with self.assertRaisesRegex(ValueError, "panel differs"):
                    verify_research_inputs(report, "large")

    def test_baseline_refuses_portfolio_returns_from_a_different_run(self):
        import pandas as pd
        from baselines import load_research_periods

        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "ensemble_backtest_large_h20.parquet"
            pd.DataFrame({"net": [.01], "research_run_id": ["run-a"]}).to_parquet(path)
            report = {"run_id": "run-a", "ensemble_periods_sha256": {"20": _content_hash(path)}}
            with patch.object(C, "REPORTS", root):
                self.assertEqual(load_research_periods(report, "large", 20)["net"].tolist(), [.01])
                with self.assertRaisesRegex(ValueError, "different research run"):
                    load_research_periods({**report, "run_id": "run-b"}, "large", 20)
                pd.DataFrame({"net": [.02], "research_run_id": ["run-a"]}).to_parquet(path)
                with self.assertRaisesRegex(ValueError, "do not match"):
                    load_research_periods(report, "large", 20)

    def test_refit_panel_preserves_previous_snapshot_when_inputs_change(self):
        import pandas as pd
        from refit import publish_refit_panel

        with TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {"profile": "large", "feature_set_version": C.FEATURE_SET_VERSION}
            initial = pd.DataFrame({"value": [1]})
            with patch.object(C, "PROC", root), patch("provenance.dataset_version", return_value="v1"):
                publish_refit_panel(initial, metadata, "large", "v1")
                self.assertTrue(panel_identity("large", "v1")[0])
                with self.assertRaisesRegex(ValueError, "inputs changed"):
                    publish_refit_panel(initial.assign(value=2), metadata, "large", "v0")
                self.assertTrue(panel_identity("large", "v1")[0])
                self.assertEqual(pd.read_parquet(root / "panel_large.parquet")["value"].tolist(), [1])
