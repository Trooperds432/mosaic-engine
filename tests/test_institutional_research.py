import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from research import long_only_backtest


class CalendarComparisonTests(unittest.TestCase):
    def test_missing_first_score_does_not_shift_rebalance_calendar(self):
        dates = pd.bdate_range('2020-01-01', periods=9)
        full = long_only_backtest(np.arange(9.), np.ones(9), dates, ['A'] * 9, np.zeros(9), 20)
        delayed = long_only_backtest([np.nan] + list(np.arange(8.)), np.ones(9), dates, ['A'] * 9, np.zeros(9), 20)
        self.assertEqual(list(delayed.index), list(full.index[1:]))

    def test_explicit_calendar_and_missing_selected_outcome(self):
        dates = pd.to_datetime(['2020-01-01'] * 2 + ['2020-01-02'] * 2)
        # Never replace the highest-ranked stock because its outcome is missing.
        bt = long_only_backtest([2, 1, 2, 1], [np.nan, .1, .02, .1], dates, ['A', 'B'] * 2,
                                np.zeros(4), 20, top=.5, rebalance_dates=pd.unique(dates))
        self.assertEqual(list(bt.index), [pd.Timestamp('2020-01-02')])

    def test_common_calendar_produces_comparable_multiple_testing_periods(self):
        from validation_extensions import multiple_comparison
        dates = pd.bdate_range('2020-01-01', periods=104)
        rng = np.random.default_rng(22)
        values = rng.normal(.001, .02, len(dates))
        early = long_only_backtest(np.ones(len(dates)), values, dates, ['A'] * len(dates), np.zeros(len(dates)), 20)
        late = long_only_backtest([np.nan] * 8 + [1.] * (len(dates) - 8), values * .8,
                                 dates, ['A'] * len(dates), np.zeros(len(dates)), 20)
        result = multiple_comparison(pd.DataFrame({'early': early.net, 'late': late.net}), reps=19)
        self.assertEqual(result['periods'], 24)
        self.assertEqual(result['excluded_periods'], 2)

    def test_random_baseline_uses_explicit_model_dates(self):
        import baselines
        dates = pd.bdate_range('2020-01-01', periods=12)
        panel = pd.DataFrame({'date': np.repeat(dates, 10), 'symbol': list('ABCDEFGHIJ') * 12,
                              'y_20': np.repeat(np.arange(12.) / 100, 10)})
        with patch.object(baselines, 'N_PATHS', 5):
            result = baselines.random_matched(panel, 20, np.ones(len(panel), bool), .2, 1.,
                                              np.zeros(len(panel)), rebalance_dates=dates[[1, 5, 9]])
        self.assertEqual(result.shape, (3, 5))
        np.testing.assert_allclose(result, np.repeat([[.01], [.05], [.09]], 5, axis=1))


class InstitutionalBenchmarkTests(unittest.TestCase):
    def panel(self):
        from institutional_research import THEMES
        dates = np.repeat(pd.bdate_range('2020-01-01', periods=40), 10)
        frame = pd.DataFrame({'date': dates, 'symbol': list('ABCDEFGHIJ') * 40, 'sector': 'tech'})
        for theme in THEMES.values():
            for feature, sign in theme.items():
                frame[feature] = np.tile(np.arange(10.) * sign, 40)
        return frame

    def test_fixed_signs_equal_theme_weights_and_no_missing_imputation(self):
        from institutional_research import THEMES, fixed_vqm
        panel = self.panel()
        score = fixed_vqm(panel)
        self.assertGreater(score.iloc[9], score.iloc[0])
        contradictory = panel.copy()
        for feature in THEMES['quality']:
            contradictory[feature] *= -1
        # Five quality inputs still receive only ONE theme's weight, equal to
        # the three value inputs and the two momentum inputs.
        self.assertAlmostEqual(fixed_vqm(contradictory).iloc[0], (.1 + 1. + .1) / 3)
        self.assertAlmostEqual(fixed_vqm(contradictory).iloc[9], (1. + .1 + 1.) / 3)
        panel.loc[9, 'leverage'] = np.nan
        self.assertTrue(np.isnan(fixed_vqm(panel).iloc[9]))
        before_future_change = fixed_vqm(panel).copy()
        # Future changes cannot change an earlier score.
        panel.loc[panel.date > panel.date.min(), 'roa'] = -1e9
        np.testing.assert_allclose(fixed_vqm(panel).iloc[:9], before_future_change.iloc[:9])

    def test_paired_report_identical_universe_dates_costs_and_saved_names(self):
        from institutional_research import compare
        panel = self.panel()
        panel['y_20'] = np.tile(np.arange(10.) / 100, 40)
        panel.loc[0, 'roa'] = np.nan
        dates = pd.unique(panel.date)
        report, periods, holdings = compare(panel, np.tile(np.arange(10.), 40), np.full(len(panel), .001),
                                             20, dates[20], top=.2)
        self.assertEqual(report['eligible_rows'], len(panel) - 1)
        self.assertGreater(report['samples']['holdout']['paired_periods'], 0)
        self.assertAlmostEqual(report['samples']['holdout']['mosaic_minus_vqm_net']['mean'], 0.)
        for _, group in periods.groupby('sample'):
            self.assertEqual(group.groupby('strategy').size().nunique(), 1)
        self.assertEqual(set(holdings.strategy), {'mosaic', 'vqm'})
        self.assertTrue((holdings.weight > 0).all())

    def test_missing_outcome_is_reported_and_removed_from_both_portfolios(self):
        from institutional_research import compare
        panel = self.panel()
        panel['y_20'] = .02
        panel.loc[9, 'y_20'] = np.nan
        report, periods, _ = compare(panel, np.tile(np.arange(10.), 40), np.full(len(panel), .001),
                                    20, pd.unique(panel.date)[20])
        self.assertEqual(report['status'], 'INCOMPLETE_OUTCOMES')
        self.assertEqual(report['samples']['walk_forward']['unassessable_periods'], ['2020-01-01'])
        self.assertFalse((periods.d == panel.date.min()).any())

    def test_cli_replay_writes_strict_json_periods_and_actual_selected_names(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            panel = self.panel()
            panel['y_20'] = np.tile(np.arange(10.) / 100, 40)
            panel['cs_spread'], panel['vol_60'], panel['adv_dollar'] = .001, .2, 1e7
            panel.to_parquet(path / 'panel.parquet', index=False)
            boundary = pd.unique(panel.date)[20]
            scores = panel[['date', 'symbol']].assign(ensemble_score=np.tile(np.arange(10.), 40))
            scores['sample'] = np.where(scores.date < boundary, 'walk_forward', 'holdout')
            scores.to_parquet(path / 'scores.parquet', index=False)
            result = subprocess.run([sys.executable, str(root / 'institutional_research.py'),
                                     '--panel', str(path / 'panel.parquet'), '--scores', str(path / 'scores.parquet'),
                                     '--profile', 'large', '--holdout-start', str(pd.Timestamp(boundary).date()), '--order-usd', '600',
                                     '--output-dir', str(path / 'reports')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((path / 'reports/institutional_comparison_large.json').read_text())
            json.dumps(report, allow_nan=False)
            holdings = pd.read_parquet(path / 'reports/institutional_holdings_large.parquet')
            self.assertEqual(set(holdings.symbol), {'I', 'J'})
            self.assertFalse(report['historical_integrity']['passed'])


class ExecutionEvidenceTests(unittest.TestCase):
    def fills(self):
        return pd.DataFrame([
            dict(fill_id='a', order_id='o1', symbol='A', side='buy', quantity=10., fill_price=101.,
                 arrival_mid=100., arrival_at='2026-01-02T14:30:00Z', submitted_at='2026-01-02T14:30:01Z',
                 filled_at='2026-01-02T14:30:02Z', fees_usd=1., fx_cost_usd=0., modeled_cost_bps=50., environment='paper'),
            dict(fill_id='b', order_id='o2', symbol='B', side='sell', quantity=10., fill_price=99.,
                 arrival_mid=100., arrival_at='2026-01-02T14:30:00Z', submitted_at='2026-01-02T14:30:01Z',
                 filled_at='2026-01-02T14:30:02Z', fees_usd=0., fx_cost_usd=0., modeled_cost_bps=50., environment='live')])

    def test_side_sign_fees_and_separate_paper_live_evidence(self):
        from execution_costs import audit
        result = audit(self.fills(), min_orders=1, min_days=1)
        self.assertAlmostEqual(result['environments']['paper']['observed_cost_bps'], 110.)
        self.assertAlmostEqual(result['environments']['live']['observed_cost_bps'], 100.)
        self.assertFalse(result['environments']['paper']['sufficient_live_evidence'])
        self.assertTrue(result['environments']['live']['sufficient_live_evidence'])
        self.assertFalse(result['changes_cost_assumptions'])

    def test_duplicates_future_reference_and_stale_reference_rejected(self):
        from execution_costs import audit
        f = self.fills()
        with self.assertRaises(ValueError):
            audit(pd.concat([f, f.iloc[:1]]))
        for value in ['2026-01-02T14:30:02Z', '2026-01-02T14:00:00Z']:
            f = self.fills()
            f.loc[0, 'arrival_at'] = value
            with self.assertRaises(ValueError):
                audit(f)

    def test_partial_fills_do_not_inflate_order_sample_size(self):
        from execution_costs import audit
        f = self.fills().iloc[:1]
        second = f.copy()
        second['fill_id'] = 'partial'
        result = audit(pd.concat([f, second]), min_orders=2, min_days=1)
        self.assertEqual(result['environments']['paper']['orders'], 1)
        self.assertFalse(result['environments']['paper']['sufficient_live_evidence'])

    def test_cli_and_empty_evidence(self):
        from execution_costs import audit
        fills = self.fills()
        self.assertEqual(audit(fills.iloc[:0])['status'], 'NO_EXECUTION_EVIDENCE')
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            fills.to_csv(path / 'fills.csv', index=False)
            result = subprocess.run([sys.executable, str(root / 'execution_costs.py'), str(path / 'fills.csv'),
                                     '--output', str(path / 'audit.json')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((path / 'audit.json').read_text())
            self.assertEqual(len(report['source_sha256']), 64)
            self.assertFalse(report['environments']['live']['sufficient_live_evidence'])


if __name__ == '__main__':
    unittest.main()
