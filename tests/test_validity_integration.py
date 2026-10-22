import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
import pandas as pd

import config as C
import features
import historical_data as H
import pit_estimates
from insider_forward import eligible_observations
from paper_trading import current_history
from provider_access import key, reserve_alpha_request
from provider_history import collect


class EstimateImportTests(unittest.TestCase):
    def test_archive_and_active_table(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "estimates.csv"
            pd.DataFrame([{"security_id": "A", "fiscal_period": "2026Q4", "consensus_eps": 2.0,
                           "available_at_utc": "2026-01-01T12:00:00Z", "observed_at_utc": "2026-01-01T12:00:00Z",
                           "timestamp_basis": "observed_snapshot", "source": "test"}]).to_csv(source, index=False)
            with patch.object(C, "RAW", root / "raw"), patch.object(C, "PROC", root / "processed"):
                target = pit_estimates.import_csv(source)
                self.assertEqual(pd.read_parquet(target).iloc[0]["consensus_eps"], 2.0)
                self.assertEqual(len(list((C.RAW / "estimate_vintages").glob("*.csv"))), 1)
                pit_estimates.import_csv(source)
                self.assertEqual(len(list((C.RAW / "estimate_vintages").glob("*.csv"))), 1)


class HistoricalImportTests(unittest.TestCase):
    def fixture(self, source):
        master = pd.DataFrame([{"security_id": "OLD", "cik": 1, "ticker": "REUSED", "exchange": "NYSE", "source": "vendor"},
                               {"security_id": "NEW", "cik": 2, "ticker": "REUSED", "exchange": "NYSE", "source": "vendor"}])
        membership = pd.DataFrame([{"security_id": sid, "index": idx, "start": "2020-01-01" if sid == "OLD" else "2020-01-03", "end": None,
                                    "known_at": "2020-01-01", "source": "vendor"} for sid in ["OLD", "NEW"] for idx in H.INDICES])
        rows = []
        for sid in ["OLD", "NEW"]:
            for day in pd.date_range("2020-01-01" if sid == "OLD" else "2020-01-03", periods=2 if sid == "OLD" else 3):
                rows.append({"security_id": sid, "ticker": "REUSED", "date": day, "open": 10., "high": 11., "low": 9., "close": 10.,
                             "adjclose": 10., "adj_open": 10., "volume": 2e6, "market_cap": 2e10, "source": "vendor"})
        prices = pd.DataFrame(rows)
        terminal = pd.DataFrame([{"security_id": "OLD", "last_trade_date": "2020-01-02", "settlement_date": "2020-01-03",
                                   "terminal_adjclose": 0., "source": "verified bankruptcy payout"}])
        for name, frame in zip(["security_master", "membership", "prices", "terminal"], [master, membership, prices, terminal]):
            frame.to_csv(source / f"{name}.csv", index=False)
        (source / "coverage.json").write_text(json.dumps({"source": "test fixture, not actual market history", "indices": sorted(H.INDICES),
            "start": "2020-01-01", "end": "2020-01-05", "constituents_complete": True, "identifiers_verified": True,
            "terminal_events_complete": True, "delisted_prices_complete": True}))

    def test_import_preserves_reused_tickers_and_delisted_identifiers(self):
        with tempfile.TemporaryDirectory() as d:
            root, source = Path(d), Path(d) / "source"
            source.mkdir()
            self.fixture(source)
            raw, proc = root / "data/raw", root / "data/processed"
            with patch.object(C, "ROOT", root), patch.object(C, "RAW", raw), patch.object(C, "PROC", proc):
                bundle = H.import_bundle(source)
                universe = pd.read_parquet(proc / "universe.parquet")
                self.assertEqual(set(universe["yahoo"]), {"OLD", "NEW"})
                self.assertTrue(H.coverage_check("2020-01-01", "2020-01-05")["passed"])
                self.assertEqual(H.security_for_ticker("REUSED", "2020-01-04"), "NEW")
                self.assertEqual(H.ticker_for_security("OLD", "2020-01-02"), "REUSED")
                self.assertEqual(len(pd.read_parquet(bundle / "security_master.parquet")), 2)

    def test_feature_labels_no_longer_forward_fill_missing_exits(self):
        cal = pd.date_range("2020-01-01", periods=4)
        p = {"adj_open": pd.DataFrame({"S": [10, 10, np.nan, np.nan]}, index=cal),
             "spy_open": pd.Series([100, 100, 102, 103], index=cal)}
        with patch("historical_data.active_bundle", return_value=None), patch.object(features, "HORIZONS", [1]):
            self.assertTrue(np.isnan(features.labels(p, np.array([0]))[1][0, 0]))


class ForwardBoundaryTests(unittest.TestCase):
    def test_no_backfilled_forecast_or_old_recipe_can_enter_fresh_forward_test(self):
        model = {"run_id": "run", "recipe_id": "new", "forward_start_utc": "2026-01-05T12:00:00Z"}
        hist = pd.DataFrame({"profile": ["broad"] * 4, "model_run_id": ["run"] * 4, "recipe_id": ["new", "old", "new", "new"],
            "recorded_utc": ["2026-01-06T22:00:00Z"] * 4, "as_of": pd.to_datetime(["2026-01-06", "2026-01-06", "2026-01-02", "2026-01-07"])})
        self.assertEqual(current_history(hist, {"broad": "run"}, {"broad": model}).index.tolist(), [0])

    def test_event_forward_deduplicates_and_excludes_before_freeze(self):
        model = {"run_id": "run", "model_version": "1.2.0", "forward_start_utc": "2026-01-05T00:00:00Z"}
        base = {"model_id": "insider-event", "model_version": "1.2.0", "signal_id": "one", "provenance": {"research_run_id": "run"}}
        records = [{**base, "generated_at_utc": t} for t in ["2026-01-06T00:00:00Z", "2026-01-07T00:00:00Z"]]
        records.append({**base, "signal_id": "old", "generated_at_utc": "2026-01-01T00:00:00Z"})
        selected = eligible_observations(records, model)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["generated_at_utc"], "2026-01-06T00:00:00Z")


class InventoryAvailabilityTests(unittest.TestCase):
    def test_macro_inventory_excludes_legacy_rows(self):
        calendar = pd.bdate_range("2020-01-01", periods=15)
        observations = pd.to_datetime([f"{year}-01-04" for year in range(2015, 2021)])
        frame = pd.DataFrame({"series": "crude_stocks_ex_spr", "obs_date": observations,
                              "available_date": observations + pd.Timedelta(days=5),
                              "value": np.arange(6) * 10. + 100., "source": "eia_wpsr"})
        market = pd.DataFrame(1., index=calendar, columns=["DGS10", "BAA10Y", "T10Y3M", "IPG3344S", "A34SNO", "DGASNYH", "DHOILNYH", "DCOILWTICO"])
        with tempfile.TemporaryDirectory() as directory, patch.object(features, "RAW", Path(directory)), \
             patch("data_macro.daily", return_value=market), patch("features.pd.read_parquet", return_value=frame), \
             patch("features._series", return_value=pd.Series(1., index=calendar)), \
             patch("macro_surprise.daily_z", return_value=pd.DataFrame(index=calendar)):
            self.assertTrue(features.macro_state(calendar)["crude_inv_z"].isna().all())
            frame["source"] = "eia_observed_snapshot"
            frame["available_date"] = pd.Timestamp("2020-01-09")
            actual = features.macro_state(calendar)["crude_inv_z"]
            self.assertTrue(actual.loc[:"2020-01-08"].isna().all())
            self.assertGreater(actual.loc["2020-01-09"], 0.)


class LongShortExportTests(unittest.TestCase):
    def test_numpy_selection_weights_export_as_json_numbers(self):
        from ls_signals import strategy_preview
        preview = strategy_preview({"A": np.float64(.25), "B": np.float64(-.25)})
        exported = json.loads(json.dumps(preview, allow_nan=False))
        self.assertEqual(exported, {"n_long": 1, "n_short": 1, "gross": .5, "net": 0.})


class ProviderTests(unittest.TestCase):
    def test_env_takes_precedence_over_root_config_without_printing_secrets(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config"
            path.write_text('ALPHA_VANTAGE_API_KEY="file-secret"\n')
            with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "environment-secret"}):
                self.assertEqual(key("ALPHA_VANTAGE_API_KEY", path), "environment-secret")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(key("ALPHA_VANTAGE_API_KEY", path), "file-secret")

    def test_free_budget_prevents_overrun_and_counts_reservations(self):
        with tempfile.TemporaryDirectory() as d, patch.object(C, "RAW", Path(d)):
            self.assertEqual(reserve_alpha_request(limit=2), 1)
            self.assertEqual(reserve_alpha_request(limit=2), 2)
            with self.assertRaises(ValueError):
                reserve_alpha_request(limit=2)
            ledger = json.loads((Path(d) / "collect/av_calls.json").read_text())
            self.assertEqual(list(ledger.values()), [2])

    def test_finnhub_denial_never_generates_a_coverage_archive(self):
        with tempfile.TemporaryDirectory() as d, patch.object(C, "RAW", Path(d)), patch("provider_history.credential", return_value="secret"):
            with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(403))) as client:
                with self.assertRaises(ValueError):
                    collect("finnhub", symbol="^GSPC", client=client)
            self.assertFalse((Path(d) / "provider_history").exists())


if __name__ == "__main__":
    unittest.main()
