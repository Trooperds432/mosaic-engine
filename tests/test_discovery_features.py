import unittest

import numpy as np
import pandas as pd

from research import EMBARGO, discovery_features, settled_before, training_cost


class DiscoveryFeatureTests(unittest.TestCase):
    def test_future_rows_do_not_qualify_a_short_snapshot_history(self):
        dates = pd.to_datetime(["2020-01-01", "2020-02-01", "2024-01-01"])
        panel = pd.DataFrame({"date": dates, "mom_12_1": [1., 2., 3.],
                              "eps_revision_pit_31d": [.1, .2, .3],
                              "roa": [None, None, .4]})
        self.assertIn("eps_revision_pit_31d", discovery_features(panel))
        self.assertEqual(discovery_features(panel.iloc[:2]), ["mom_12_1"])

    def test_label_touching_next_block_after_embargo_is_excluded(self):
        boundary, horizon = 100, 20
        last_excluded = boundary - 1 - horizon - EMBARGO
        self.assertEqual(settled_before([last_excluded - 1, last_excluded, 100],
                                        horizon, boundary).tolist(), [True, False, False])

    def test_probability_cost_uses_only_training_rows(self):
        mask = np.array([True, True, False])
        self.assertAlmostEqual(training_cost([.01, .02, 100], mask), .03)
        self.assertAlmostEqual(training_cost([.01, .02, .001], mask), .03)

    def test_probability_fit_excludes_labels_touching_calibration(self):
        from research import CalibratedLogit

        rng = np.random.default_rng(12)
        positions = np.repeat(np.arange(60), 3)
        x = rng.normal(size=(len(positions), 3))
        y = np.where(np.arange(len(x)) % 2 == 0, .05, -.05)
        baseline = CalibratedLogit(0.).fit(x, y, positions=positions, horizon=5)
        purge = (positions < 48) & ~settled_before(positions, 5, 48)
        modified = y.copy()
        modified[purge] *= -1
        actual = CalibratedLogit(0.).fit(x, modified, positions=positions, horizon=5)
        np.testing.assert_array_equal(actual.lr.coef_, baseline.lr.coef_)
        np.testing.assert_array_equal(actual.predict(x), baseline.predict(x))
