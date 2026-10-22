import json
import unittest
import runpy
import tempfile
from pathlib import Path
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from arch.bootstrap import SPA, RealityCheck
from sklearn.covariance import LedoitWolf

from factor_persistence import summary, quantile_turnover
from risk_model import ledoit_wolf_cov
from validation_extensions import multiple_comparison, purged_combinations, cpcv_ridge


class ResearchEntrypointTests(unittest.TestCase):
    def test_cli_uses_importable_model_classes_and_saves_them(self):
        import research
        rng = np.random.default_rng(37)
        X = pd.DataFrame(rng.normal(size=(120, len(research.GAM_FEATS))), columns=research.GAM_FEATS)
        y = np.tile([-.02, .03], 60)
        sectors = np.repeat('tech', len(X))
        def evaluate(profile):
            self.assertEqual(profile, 'large')
            with tempfile.TemporaryDirectory() as directory:
                for name in ['gam', 'hier', 'logit']:
                    model, expected = research.fit_predict(name, X, y, X, sectors, sectors, cost=0.)
                    target = Path(directory) / (name + '.joblib')
                    joblib.dump(model, target)
                    restored = joblib.load(target)
                    values = X.to_numpy(dtype='float32')
                    actual = restored.predict(X) if name == 'gam' else (restored.predict(values, sectors) if name == 'hier' else restored.predict(values))
                    np.testing.assert_allclose(actual, expected)
        with patch.object(research, 'run', side_effect=evaluate) as execute, \
             patch.object(research.C, 'hold_lock', return_value=None), patch('sys.argv', ['research.py', 'large']):
            runpy.run_path(str(Path(research.__file__)), run_name='__main__')
        execute.assert_called_once_with('large')


class MaintainedBootstrapTests(unittest.TestCase):
    def test_loss_sign_and_pvalues_match_upstream_with_common_periods(self):
        matrix = pd.DataFrame(np.random.default_rng(21).normal(.001, .02, size=(60, 3)), columns=list('ABC'))
        matrix.loc[0, 'A'] = np.nan
        actual = multiple_comparison(matrix, block=4, reps=99)
        common = matrix.dropna()
        spa = SPA(np.zeros(59), -common, block_size=4, reps=99, studentize=True, seed=7)
        rc = RealityCheck(np.zeros(59), -common, block_size=4, reps=99, studentize=False, seed=7)
        spa.compute()
        rc.compute()
        self.assertEqual(actual['spa_p'], float(spa.pvalues['consistent']))
        self.assertEqual(actual['reality_check_p'], float(rc.pvalues['upper']))
        self.assertEqual(actual['periods'], 59)
        self.assertEqual(actual['excluded_periods'], 1)
        self.assertEqual(actual, multiple_comparison(matrix, block=4, reps=99))
        json.dumps(actual, allow_nan=False)

    def test_insufficient_common_history_not_zero_imputed(self):
        matrix = pd.DataFrame({'A': [1.] * 30 + [np.nan] * 30, 'B': [np.nan] * 30 + [2.] * 30})
        self.assertIsNone(multiple_comparison(matrix))

    def test_unordered_input_rejected(self):
        with self.assertRaises(ValueError):
            multiple_comparison(pd.DataFrame({'A': [1., 2.], 'B': [2., 3.]}, index=[1, 0]))


class CovarianceTests(unittest.TestCase):
    def test_true_ledoit_wolf_and_psd_for_collinear_factors(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=100)
        frame = pd.DataFrame({'market': x, 'style': x * 2, 'other': rng.normal(size=100)})
        frame.loc[0, 'style'] = np.nan
        actual, shrink = ledoit_wolf_cov(frame)
        upstream = LedoitWolf().fit(frame.dropna())
        np.testing.assert_allclose(actual, upstream.covariance_)
        self.assertEqual(shrink, upstream.shrinkage_)
        self.assertGreaterEqual(np.linalg.eigvalsh(actual).min(), -1e-12)
        self.assertEqual(list(actual.columns), list(frame.columns))

    def test_missing_covariance_history_rejected(self):
        with self.assertRaises(ValueError):
            ledoit_wolf_cov(pd.DataFrame({'market': [np.nan, np.nan]}))


class FactorPersistenceTests(unittest.TestCase):
    def test_name_turnover_and_rank_reversal_independent_expected_values(self):
        dates = np.repeat(pd.to_datetime(['2020-01-01', '2020-01-02']), 5)
        securities = list('ABCDE') * 2
        scores = [1., 2., 3., 4., 5., 5., 4., 3., 2., 1.]
        actual = summary(scores, np.arange(10) / 100, dates, securities, lags=(1,))
        result = actual['persistence']['1']
        self.assertAlmostEqual(result['rank_autocorrelation'], -1.)
        self.assertEqual(result['top_quantile_turnover'], 1.)
        self.assertEqual(result['turnover_pairs'], 1)
        json.dumps(actual, allow_nan=False)

    def test_missing_bucket_date_is_not_skipped(self):
        index = pd.MultiIndex.from_product([pd.date_range('2020-01-01', periods=3), ['A']], names=['date', 'asset'])
        series = pd.Series([5., np.nan, 5.], index=index)
        result = quantile_turnover(series, 5)
        self.assertTrue(result.isna().all())

    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError):
            summary([1., 2.], [0., 0.], ['2020-01-01'] * 2, ['A'] * 2)


class CombinatorialTests(unittest.TestCase):
    def test_session_label_purge_and_asset_groups(self):
        positions = np.repeat(np.arange(0, 200, 5), 2)
        splits = list(purged_combinations(positions, 10, 160, embargo=5, n_folds=4))
        self.assertEqual(len(splits), 6)
        for train, tests, path_ids in splits:
            self.assertTrue(np.all(positions[train] + 16 < 160))
            for group in tests:
                self.assertFalse(set(positions[train]) & set(positions[group]))
                for train_position in positions[train]:
                    self.assertTrue(all(not (train_position + 1 <= p + 11 and train_position + 11 >= p + 1) for p in positions[group]))
                for position in np.unique(positions[group]):
                    self.assertEqual(sum(positions[group] == position), 2)
            self.assertEqual(len(path_ids), 2)

    def test_irregular_decision_dates_use_session_distance(self):
        positions = np.repeat([0, 1, 2, 3, 20, 21, 40, 41, 60, 61, 80, 81], 2)
        for train, tests, _ in purged_combinations(positions, 7, 100, n_folds=3):
            for group in tests:
                lo, hi = positions[group].min() + 1, positions[group].max() + 8
                self.assertFalse(np.any((positions[train] + 1 <= hi) & (positions[train] + 8 >= lo)))

    def test_real_ridge_paths_never_consume_holdout_labels(self):
        rng = np.random.default_rng(12)
        positions = np.repeat(np.arange(80) * 5, 25)
        dates = np.repeat(pd.bdate_range('2020-01-01', periods=80), 25)
        X = pd.DataFrame(rng.normal(size=(len(positions), 3)), columns=list('xyz'))
        y = X['x'].to_numpy() * .01 + rng.normal(0, .001, len(X))
        args = dict(embargo=2, n_folds=4, n_test_folds=2, min_train=50)
        actual = cpcv_ridge(X, y, dates, positions, 5, 350, **args)
        contaminated = y.copy()
        contaminated[positions >= 350] = 1e9
        repeated = cpcv_ridge(X, contaminated, dates, positions, 5, 350, **args)
        self.assertEqual(actual, repeated)
        self.assertEqual(actual['status'], 'COMPUTED')
        self.assertEqual(len(actual['splits']), 6)
        self.assertEqual(len(actual['paths']), 3)
        self.assertTrue(all(path['mean_ic'] > .9 for path in actual['paths']))
        json.dumps(actual, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
