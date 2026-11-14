import unittest

import numpy as np
import pandas as pd

from commodity_research import event_returns, event_splits, inventory_observations


class CommodityTimingTests(unittest.TestCase):
    def test_entry_after_release_full_exit_and_purged_boundary(self):
        dates = pd.bdate_range("2024-01-01", periods=9)
        prices = pd.Series(np.arange(100., 109.), index=dates)
        outcomes = event_returns(prices, dates[[0, 2, 4, 7]], 2)
        self.assertEqual(outcomes.loc[0, "entry"], dates[1])
        self.assertAlmostEqual(outcomes.loc[0, "y"], np.log(103 / 101))
        self.assertTrue(pd.isna(outcomes.loc[3, "y"]))
        split = dict(event_splits(outcomes, dates[4]))
        self.assertEqual(list(split["discovery"].index), [0])
        self.assertEqual(list(split["holdout"].index), [2])

    def test_inventory_baseline_excludes_future_and_legacy_vintages(self):
        records = []
        for year, value in [(2016, 90), (2017, 100), (2018, 110), (2019, 120), (2020, 130)]:
            day = pd.Timestamp(f"{year}-01-04")
            records.append(dict(series="crude_stocks_ex_spr", source="eia_observed_snapshot",
                                obs_date=day, available_date=day + pd.Timedelta(days=5), value=value))
        records.append({**records[-2], "value": 999, "available_date": pd.Timestamp("2020-02-01")})
        records.append({**records[-3], "source": "eia_wpsr", "value": -999})
        frame = pd.DataFrame(records)
        initial = inventory_observations(frame.iloc[:5])
        revised = inventory_observations(frame)
        self.assertTrue(np.isfinite(initial.iloc[-1]["z"]))
        pd.testing.assert_frame_equal(initial, revised.iloc[:5])
        self.assertNotEqual(initial.iloc[-1]["z"], revised.iloc[-1]["z"])
