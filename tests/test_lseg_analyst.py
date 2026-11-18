"""Point-in-time rules for the pre-registered LSEG analyst and ETF-flow features (prereg/lseg_analyst_flow_v1.json)."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import features


def eod(day):
    return (pd.Timestamp(day) + pd.Timedelta(hours=23, minutes=59, seconds=59)).tz_localize("America/New_York").tz_convert("UTC")


class AnalystFeatureTests(unittest.TestCase):
    def run_features(self, analyst=None, vintages=None, etf=None, dates=None, syms=("S1",), sector=None):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            if analyst is not None:
                analyst.to_parquet(root / "lseg_analyst.parquet", index=False)
            if vintages is not None:
                vintages.to_parquet(root / "estimate_vintages.parquet", index=False)
            if etf is not None:
                etf.to_parquet(root / "etf_flows.parquet", index=False)
            with patch.object(features, "PROC", root):
                return features.analyst_features(pd.DatetimeIndex(dates), list(syms), sector or {s: "Technology" for s in syms})

    def test_price_target_revision_is_not_visible_on_its_calc_date(self):
        a = pd.DataFrame({"security_id": "S1", "measure": "pt", "value": [100.0, 110.0],
                          "available_at_utc": [eod("2020-01-31"), eod("2020-02-28")]})
        out = self.run_features(analyst=a, dates=["2020-02-28", "2020-03-02"])["pt_rev_1m_pit"]["S1"]
        self.assertTrue(np.isnan(out.iloc[0]))  # calc date itself: the February value is not yet available
        self.assertAlmostEqual(out.iloc[1], np.log(1.1))

    def test_stale_or_gapped_values_are_missing(self):
        a = pd.DataFrame({"security_id": "S1", "measure": "rec", "value": [2.0, 1.5],
                          "available_at_utc": [eod("2020-01-31"), eod("2020-02-28")]})
        out = self.run_features(analyst=a, dates=["2020-03-02", "2020-05-01"])["rec_rev_1m_pit"]["S1"]
        self.assertAlmostEqual(out.iloc[0], 0.5)  # rating improved from 2.0 to 1.5 -> positive
        self.assertTrue(np.isnan(out.iloc[1]))  # latest value more than 45 days old
        gapped = a.assign(available_at_utc=[eod("2019-11-29"), eod("2020-02-28")])
        self.assertTrue(np.isnan(self.run_features(analyst=gapped, dates=["2020-03-02"])["rec_rev_1m_pit"]["S1"].iloc[0]))

    def test_dispersion_requires_three_estimates(self):
        v = pd.DataFrame({"security_id": "S1", "consensus_eps": [2.0, 2.0], "eps_std": [0.2, 0.4], "n_estimates": [5.0, 2.0],
                          "available_at_utc": [eod("2020-01-31"), eod("2020-02-28")]})
        v = v.assign(fiscal_period="2020Q4", observed_at_utc=v["available_at_utc"],
                     timestamp_basis="observed_snapshot", source="test")
        out = self.run_features(vintages=v, dates=["2020-02-03", "2020-03-02"])["est_dispersion_pit"]["S1"]
        self.assertAlmostEqual(out.iloc[0], 0.1)
        self.assertTrue(np.isnan(out.iloc[1]))  # latest vintage has only two estimates

    def test_etf_flow_uses_only_prior_sessions(self):
        days = pd.bdate_range("2020-01-01", periods=30)
        shares = np.full(30, 100.0)
        shares[25] = 110.0  # creation on session 25
        etf = pd.DataFrame({"etf": "XLK", "date": days, "shares": shares, "nav": 10.0})
        out = self.run_features(etf=etf, dates=[days[25], days[26]])["sector_etf_flow_20d"]["S1"]
        self.assertAlmostEqual(out.iloc[0], 0.0)  # decision on the creation day cannot see it
        self.assertAlmostEqual(out.iloc[1], 0.1)  # next session: 10 shares x NAV 10 / AUM 1000
        other = self.run_features(etf=etf, dates=[days[26]], sector={"S1": "Unknown"})["sector_etf_flow_20d"]["S1"]
        self.assertTrue(np.isnan(other.iloc[0]))


if __name__ == "__main__":
    unittest.main()
