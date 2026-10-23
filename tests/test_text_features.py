"""Filing-text features: point-in-time availability, expiry, and the similarity measures."""
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class SimilarityTests(unittest.TestCase):
    def test_measures(self):
        try:
            import data_text as T
        except ImportError as e:  # needs httpx; CI installs only numpy/pandas
            self.skipTest(str(e))
        a, b = Counter("the risk of loss is low".split()), Counter("the risk of loss is high".split())
        self.assertAlmostEqual(T.cosine(a, a), 1.0)
        self.assertLess(T.cosine(a, b), 1.0)
        self.assertAlmostEqual(T.jaccard(a, b), 5 / 7)
        text = "Item 1A. Risk Factors see below Item 1B. Unresolved " + "Item 1A Risk Factors " + "x " * 1500 + "Item 2. Properties"
        self.assertGreater(len(T.risk_factors(text)), 2000)
        self.assertEqual(T.clean("<p>A&amp;B</p><script>x()</script>"), "A&B")


class FeatureTests(unittest.TestCase):
    def test_point_in_time_and_expiry(self):
        try:
            import features as F
        except ImportError as e:
            self.skipTest(str(e))
        cal = pd.bdate_range("2024-01-01", "2024-12-31")
        df = pd.DataFrame({"cik": [7], "available_date": [pd.Timestamp("2024-02-10")], "sim_cos": [0.9], "sim_jac": [0.5],
                           "sim_rf_cos": [np.nan], "len_chg": [0.1]})
        with tempfile.TemporaryDirectory() as d, patch.object(F, "PROC", Path(d)):
            df.to_parquet(Path(d) / "text_similarity.parquet", index=False)
            dec_idx = cal.get_indexer(pd.to_datetime(["2024-02-09", "2024-02-12", "2024-08-28", "2024-09-02"]))
            out = F.text_features(dec_idx, cal, [7])
        s = out["text_sim_cos"][7].to_numpy()
        self.assertTrue(np.isnan(s[0]))            # before availability
        self.assertAlmostEqual(float(s[1]), 0.9, places=5)  # first session on/after availability
        self.assertAlmostEqual(float(s[2]), 0.9, places=5)  # 200 days later still valid
        self.assertTrue(np.isnan(s[3]))            # expired after 200 days
        self.assertTrue(np.isnan(out["text_sim_rf"][7].to_numpy()).all())


class WikiTests(unittest.TestCase):
    def test_uses_only_days_before_decision(self):
        try:
            import features as F
        except ImportError as e:
            self.skipTest(str(e))
        cal = pd.bdate_range("2024-01-01", "2024-06-28")
        days = pd.date_range("2023-10-01", "2024-06-30")
        views = pd.Series(100.0, index=days)
        views[pd.Timestamp("2024-05-01")] = 100000.0  # spike on the decision day itself
        df = pd.DataFrame({"cik": 7, "date": days, "views": views.to_numpy()})
        with tempfile.TemporaryDirectory() as d, patch.object(F, "PROC", Path(d)):
            df.to_parquet(Path(d) / "wiki_views.parquet", index=False)
            dec_idx = cal.get_indexer(pd.to_datetime(["2024-05-01", "2024-05-02"]))
            out = F.wiki_features(dec_idx, cal, [7])
        a = out["wiki_attn_abn"][7].to_numpy()
        self.assertAlmostEqual(float(a[0]), 0.0, places=5)  # same-day spike is not yet known
        self.assertGreater(float(a[1]), 1.0)               # visible the next day
        self.assertAlmostEqual(float(out["wiki_attn_level"][7].to_numpy()[0]), float(np.log1p(100)), places=4)


if __name__ == "__main__":
    unittest.main()
