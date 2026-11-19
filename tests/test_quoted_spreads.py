import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import config as C
import quoted_spreads as Q
from research import name_costs


def quotes(n=70, half=0.0001, sid="A"):
    dates = pd.bdate_range("2020-01-01", periods=n)
    mid = 100.0
    return pd.DataFrame({"security_id": sid, "date": dates, "bid": mid * (1 - half), "ask": mid * (1 + half)})


class QuotedSpreadTests(unittest.TestCase):
    def test_trailing_mean_needs_forty_days_and_drops_bad_quotes(self):
        q = quotes()
        q.loc[5, "ask"] = q.loc[5, "bid"]          # locked quote dropped
        q.loc[6, "bid"] = 0                         # missing bid dropped
        t = Q.from_quotes(q)
        self.assertEqual(t["date"].min(), q["date"].iloc[41])  # 40 valid days after two drops
        np.testing.assert_allclose(t["hs_quoted"], 0.0001, rtol=1e-9)

    def test_attach_is_point_in_time_and_stale_quotes_fall_back(self):
        t = Q.from_quotes(quotes())
        last = t["date"].max()
        panel = pd.DataFrame({"symbol": ["A", "A", "A", "B"],
                              "date": [t["date"].min() - pd.Timedelta(days=1), last + pd.Timedelta(days=3),
                                       last + pd.Timedelta(days=C.QUOTE_MAX_STALENESS_DAYS + 1), last]},
                             index=[10, 11, 12, 13])
        out = Q.attach(panel, t)
        self.assertEqual(list(out.index), [10, 11, 12, 13])
        self.assertTrue(np.isnan(out[10]))          # no look-ahead before the first value
        self.assertAlmostEqual(out[11], 0.0001)
        self.assertTrue(np.isnan(out[12]))          # older than the staleness limit
        self.assertTrue(np.isnan(out[13]))          # no quotes for B

    def test_wide_matches_attach(self):
        t = pd.concat([Q.from_quotes(quotes(sid="A")), Q.from_quotes(quotes(half=0.0003, sid="B"))])
        cal = pd.bdate_range("2020-01-01", periods=90)
        w = Q.wide(cal, ["A", "B", "C"], t)
        long = w.stack(future_stack=True).rename("w").reset_index()
        long.columns = ["date", "symbol", "w"]
        a = Q.attach(long, t)
        np.testing.assert_allclose(long["w"].to_numpy(), a.to_numpy(), equal_nan=True)
        self.assertTrue(w["C"].isna().all())

    def test_name_costs_v2_uses_floored_quote_and_falls_back_to_corwin_schultz(self):
        panel = pd.DataFrame({"cs_spread": [0.01, 0.01, 0.01], "vol_60": [0.2] * 3, "adv_dollar": [1e9] * 3,
                              "hs_quoted": [0.0001, 0.0010, np.nan]})
        impact = 0.7 * 0.2 / np.sqrt(252) * np.sqrt(600 / 1e9)
        with patch.object(C, "COST_MODEL", "v2_quoted"):
            v2 = name_costs(panel, 600.).to_numpy()
        with patch.object(C, "COST_MODEL", "v1_cs"):
            v1 = name_costs(panel, 600.).to_numpy()
        np.testing.assert_allclose(v2, C.FX_FEE + impact + np.array([C.HALF_SPREAD, 0.0010, 0.005]))
        np.testing.assert_allclose(v1, C.FX_FEE + impact + 0.005)
        with patch.object(C, "COST_MODEL", "other"), self.assertRaises(ValueError):
            name_costs(panel, 600.)


if __name__ == "__main__":
    unittest.main()
