import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import insider_study as study
from data_form4_recent import parse
from insider_evaluation import covariates, diagnostics, portfolio
from insider_filings import acceptance_map, available_filings, normalize
from validation_extensions import purged_combinations


def filing(acc, owner, accepted, day="2024-07-03", kind="4", line=1):
    return {"accession_number": f"{acc:018d}", "rptownercik": str(owner), "rptownername": None,
            "accepted_at_utc": accepted, "filing_date": day, "trans_date": day,
            "document_type": kind, "transaction_line": line, "trans_code": "P",
            "trans_shares": 100, "trans_pricepershare": 20, "issuercik": 1,
            "aff10b5one": None, "fn_10b5_1": False, "rptowner_title": "CFO"}


class InsiderStudyTests(unittest.TestCase):
    def test_acceptance_holiday_amendments_and_same_session(self):
        calendar = pd.DatetimeIndex(["2024-07-03", "2024-07-05", "2024-07-08"])
        first = filing(1, 8, "2024-07-04T00:10:00Z")  # 20:10 ET on July 3
        second = filing(2, 9, "2024-07-04T14:00:00Z")
        amendment = filing(3, 8, "2024-07-05T16:00:00Z", kind="4/A")
        date_only = filing(4, 10, None)
        later = filing(5, 11, "2024-07-05T15:00:00Z")
        parsed = normalize(pd.DataFrame([first, first, second, amendment, date_only, later]))
        self.assertEqual(len(parsed), 5)
        self.assertEqual(parsed["document_type"].eq("4/A").sum(), 1)
        purchases = available_filings(parsed, calendar)
        self.assertEqual(set(purchases.loc[purchases["e"].eq(1), "owner"]), {"8", "9"})
        self.assertEqual(set(available_filings(parsed, calendar, date_only=True)
                             .query("e == 1")["owner"]), {"8", "9", "10"})
        self.assertEqual(len(available_filings(parsed, calendar, explicit_non_plan=True)), 0)
        self.assertEqual(set(purchases.query("e == 2")["owner"]), {"11"})

    def test_quarter_with_no_acceptance_timestamps_stays_date_only(self):
        frame = normalize(pd.DataFrame([filing(1, 8, None)]))
        self.assertTrue(frame["accepted"].isna().all())
        calendar = pd.DatetimeIndex(["2024-07-03", "2024-07-05"])
        self.assertTrue(available_filings(frame, calendar).empty)
        self.assertEqual(available_filings(frame, calendar, date_only=True)["e"].tolist(), [1])

    def test_equity_insider_history_cannot_see_late_or_same_day_filings(self):
        from features import classify_insiders

        frame = pd.DataFrame({
            "issuercik": [1, 1, 1, 2], "owner": ["x"] * 4,
            "trans_date": pd.to_datetime(["2023-01-01", "2024-01-01", "2023-02-01", "2020-01-01"]),
            "filing_date": pd.to_datetime(["2024-01-05", "2024-01-03", "2024-01-03", "2020-01-02"]),
            "value": [100., 200., 300., 10000.],
        })
        actual = classify_insiders(frame).set_index("value")
        self.assertTrue(np.isnan(actual.at[200., "prior_median_value"]))
        self.assertTrue(np.isnan(actual.at[300., "prior_median_value"]))
        self.assertEqual(actual.at[100., "prior_median_value"], 250.)

    def test_equity_uses_acceptance_day_without_changing_filing_date(self):
        from features import insider_availability

        frame = pd.DataFrame([filing(1, 8, None), filing(2, 9, None)])
        sidecar = pd.Series([pd.Timestamp("2024-07-05T00:10:00Z")],
                            index=["000000000000000001"])
        available = insider_availability(frame, sidecar)
        self.assertEqual(available.tolist(), [pd.Timestamp("2024-07-04"), pd.Timestamp("2024-07-03")])
        self.assertEqual(frame["filing_date"].tolist(), ["2024-07-03", "2024-07-03"])

    def test_equity_lookbacks_stay_missing_when_a_quarter_is_absent(self):
        from features import insider_features

        dates = pd.DatetimeIndex(["2024-05-01", "2024-10-01"])
        files = [Path("2024q2.parquet"), Path("2024q3.parquet")]
        result = insider_features(pd.DataFrame({"issuercik": []}), dates, [1], files)
        self.assertTrue(np.isnan(result["buy_usd_90"].iloc[0, 0]))
        self.assertEqual(result["buy_usd_90"].iloc[1, 0], 0)
        self.assertEqual(result["buy_usd_180"].iloc[1, 0], 0)
        self.assertTrue(result["net_usd_365"].isna().all().all())

    def test_timestamp_sidecar_and_recent_xml(self):
        with tempfile.TemporaryDirectory() as root:
            sidecar = Path(root) / "acceptances.csv"
            sidecar.write_text("accession_number,accepted_at_utc\n0000000000-00-000001,2024-07-04T00:10:00Z\n")
            d = normalize(pd.DataFrame([filing(1, 8, None)]), acceptance_map(sidecar))
            self.assertEqual(d["accept_day"].iloc[0], pd.Timestamp("2024-07-03"))
        xml = b"<ownershipDocument><nonDerivativeTable><nonDerivativeTransaction/></nonDerivativeTable></ownershipDocument>"
        row = parse(xml, "a", "2024-07-03", 1, "2024-07-04T00:10:00Z")[0]
        self.assertEqual(row["accepted_at_utc"], "2024-07-04T00:10:00Z")
        self.assertEqual(row["transaction_line"], 1)

    def test_walk_forward_uses_only_settled_labels_and_residuals(self):
        entries = np.arange(20, 120, 2)
        frame = pd.DataFrame({key: np.sin(entries / (i + 2)) for i, key in enumerate(study.FEATURES)})
        frame["e"] = entries
        frame["x"] = entries + study.H
        frame["net"] = entries / 10000
        with patch.dict(study.REG, min_train_events=4, min_calibration_events=20):
            pred, bounds = study.walk_forward(frame)
            future = frame.copy()
            future.loc[future["x"].add(study.EMBARGO).ge(90), "net"] = 999
            changed, changed_bounds = study.walk_forward(future)
        earlier = frame["e"].le(90)
        self.assertTrue(bounds.loc[earlier, "low"].notna().any())
        pd.testing.assert_series_equal(pred[earlier], changed[earlier])
        pd.testing.assert_frame_equal(bounds.loc[earlier], changed_bounds.loc[earlier])
        for train, tests, _ in purged_combinations(entries - 1, study.H, 130, study.EMBARGO):
            for test in tests:
                overlap = (entries[train, None] <= entries[test] + study.H) & (
                    entries[train, None] + study.H >= entries[test])
                self.assertFalse(overlap.any())

    def test_cost_ledger_reconciles_and_terminal_cash_has_no_exit_order(self):
        days = pd.bdate_range("2024-01-01", periods=8)
        spy = pd.Series(100., index=days)
        events = pd.DataFrame({"e": [0, 0], "rank": [1, 0], "symbol": ["a", "b"],
                               "cost": [.01, .01], "beta": [0., 0.]})
        outcomes = {0: (.1, .1, 100., False, np.linspace(100, 110, 6)),
                    1: (-1., -1., 100., True, np.array([100., 90., 90., 90., 90., 0.]))}
        daily, fills = portfolio(events, outcomes, spy, slice(None), 600, 2, 5)
        stressed, stress_fills = portfolio(events, outcomes, spy, slice(None), 600, 2, 5, multiplier=2)
        self.assertAlmostEqual(daily["net"].sum() * 1200, fills["net_pnl_usd"].sum())
        self.assertAlmostEqual(fills.loc[0, "exit_cost_usd"], 6.6)
        self.assertEqual(fills.loc[1, "exit_cost_usd"], 0)
        self.assertEqual(stress_fills["symbol"].tolist(), fills["symbol"].tolist())
        self.assertLess(stressed["net"].sum(), daily["net"].sum())
        self.assertIsNone(diagnostics(pd.DataFrame({name: daily["net"] for name in study.TRIALS}),
                                      5, None, .1)["dsr"])

    def test_covariates_use_matched_security_and_preserve_missing_revision(self):
        from types import SimpleNamespace
        day = pd.Timestamp("2024-07-05")
        market = SimpleNamespace(cal=pd.DatetimeIndex([day - pd.Timedelta(days=2), day]),
                                 cik={"a": 1, "b": 2})
        for name in ("cap", "mom", "rev", "adv", "beta"):
            setattr(market, name, pd.DataFrame({"a": [10.], "b": [40.]}))
        facts = {key: pd.DataFrame({1: [1.], 2: [2.]}, index=[day]) for key in ("ni", "equity", "assets")}
        event = covariates(market, "a", 1, facts, None)
        matched = covariates(market, "b", 1, facts, None)
        self.assertEqual(matched["mom"], 40.)
        self.assertEqual(matched["earn_yield"], .05)
        self.assertNotEqual(event["earn_yield"], matched["earn_yield"])
        self.assertTrue(np.isnan(matched["earnings_revision"]))


if __name__ == "__main__":
    unittest.main()

