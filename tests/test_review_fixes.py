import gzip
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import config as C
import gates
from provenance import dataset_version


class DependencyTests(unittest.TestCase):
    def test_text_content_archive_and_source_each_invalidate_old_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, proc = root / 'data/raw', root / 'data/processed'
            raw.mkdir(parents=True)
            proc.mkdir(parents=True)
            for name in ['features.py', 'pit_clinical.py', 'config.py', 'data_text.py']:
                (root / name).write_text('# source\n')
            with patch.object(C, 'ROOT', root), patch.object(C, 'RAW', raw), patch.object(C, 'PROC', proc):
                old = dataset_version()
                evidence = {'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'dataset_version': old,
                            'profile': 'broad', 'blocking': {k: True for k in gates.BLOCKING_TESTS}}
                text = proc / 'text_similarity.parquet'
                text.write_bytes(b'first')
                current = dataset_version()
                self.assertNotEqual(old, current)
                self.assertFalse(gates.check_audit(evidence, current, profile='broad')[0])
                stamp = text.stat()
                text.write_bytes(b'other')  # Same length, even with restored mtime.
                os.utime(text, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                updated = dataset_version()
                self.assertNotEqual(current, updated)
                archive = raw / 'text/1/filing.txt.gz'
                archive.parent.mkdir(parents=True)
                archive.write_bytes(gzip.compress(b'filing'))
                archived = dataset_version()
                self.assertNotEqual(updated, archived)
                (root / 'data_text.py').write_text('# new extraction method\n')
                self.assertNotEqual(archived, dataset_version())


class ProfileAuditTests(unittest.TestCase):
    def test_audit_writer_keeps_independent_profile_evidence(self):
        import audit
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc, reports = root / 'processed', root / 'reports'
            proc.mkdir()
            reports.mkdir()
            for profile in C.PROFILES:
                pd.DataFrame({'date': pd.to_datetime(['2020-01-01'])}).to_parquet(proc / f'panel_{profile}.parquet')
            good = {'passed': True, 'worst': 0}
            with patch.object(C, 'PROC', proc), patch.object(C, 'REPORTS', reports), \
                 patch('provenance.dataset_version', return_value='ds-test'), \
                 patch.object(audit, 'truncation_invariance', return_value=good), \
                 patch.object(audit, 'label_alignment', return_value=good), \
                 patch.object(audit, 'duplicates_and_timestamps', return_value=good), \
                 patch.object(audit, 'diagnostics', return_value={}), \
                 patch('historical_data.coverage_check', return_value=good):
                audit.run('large')
                audit.run('broad')
            self.assertEqual(json.loads((reports / 'data_audit_large.json').read_text())['profile'], 'large')
            self.assertEqual(json.loads((reports / 'data_audit_broad.json').read_text())['profile'], 'broad')

    def test_passing_large_audit_cannot_authorise_broad(self):
        import signals
        doc = {'profile': 'large', 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
               'dataset_version': 'ds-test', 'blocking': {k: True for k in gates.BLOCKING_TESTS}}
        with tempfile.TemporaryDirectory() as directory, patch.object(C, 'REPORTS', Path(directory)):
            (C.REPORTS / 'data_audit_large.json').write_text(json.dumps(doc))
            self.assertTrue(signals.audit_ok('ds-test', 'large')[0])
            self.assertFalse(signals.audit_ok('ds-test', 'broad')[0])
            (C.REPORTS / 'data_audit_broad.json').write_text(json.dumps(doc))
            self.assertFalse(signals.audit_ok('ds-test', 'broad')[0])
            doc['profile'] = 'broad'
            (C.REPORTS / 'data_audit_broad.json').write_text(json.dumps(doc))
            self.assertTrue(signals.audit_ok('ds-test', 'broad')[0])

    def test_intraday_uses_each_profiles_audit_and_cannot_reapprove_old_dataset_signals(self):
        import insider_model
        import intraday_watch
        import regime
        import signals
        from tests.test_spec_gaps import signal
        entries = []
        for profile, dataset in [('large', 'current'), ('broad', 'current'), ('large', 'old')]:
            s = signal()
            s['model_id'] = C.MODEL_ID + '-' + profile
            s['signal'].update(direction='LONG', deployment_eligibility='PAPER_TEST_ELIGIBLE')
            s['risk']['data_quality_approved'] = True
            s['provenance']['dataset_version'] = dataset
            entries.append(s)
        def evidence(dataset, profile=None):
            return (True, None) if profile == 'large' else (False, 'broad audit missing')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'latest.json').write_text(json.dumps({'as_of_utc': '2026-10-22T20:00:00Z', 'signals': entries}))
            with patch.object(C, 'SIGNALS', root), patch.object(signals, 'dataset_version', return_value='current'), \
                 patch.object(signals, 'audit_ok', side_effect=evidence), \
                 patch.object(insider_model, 'live_signals', return_value=[]), patch.object(regime, 'field', return_value={}):
                intraday_watch.refresh_insider_signals()
            updated = json.loads((root / 'latest.json').read_text())['signals']
            self.assertEqual(updated[0]['signal']['direction'], 'LONG')
            self.assertEqual(updated[1]['signal']['direction'], 'NO_SIGNAL')
            self.assertEqual(updated[2]['signal']['direction'], 'NO_SIGNAL')


class RiskFreshnessTests(unittest.TestCase):
    def test_republished_stale_name_future_and_nonfinite_var_block(self):
        from risk_freshness import position_weight
        now = datetime(2026, 10, 23, 15, tzinfo=timezone.utc)
        doc = {'generated_at_utc': now.isoformat(), 'names': {'A': {'as_of': '2026-10-22', 'var99_1d': .2, 'model': 'test'}}}
        self.assertAlmostEqual(position_weight(doc, 'A', .09, .01, now=now)[0], .05)
        for day in ['2026-10-01', '2026-10-24', '2026-10-23', None]:
            doc['names']['A']['as_of'] = day
            self.assertEqual(position_weight(doc, 'A', .09, .01, now=now)[0], 0.)
        doc['names']['A']['as_of'] = '2026-10-22'
        for value in [float('inf'), float('nan'), -1, 0]:
            doc['names']['A']['var99_1d'] = value
            self.assertEqual(position_weight(doc, 'A', .09, .01, now=now)[0], 0.)
        doc['names']['A']['var99_1d'] = .2
        self.assertEqual(position_weight(doc, 'A', .09, .01, now=now, signal_asof='2026-10-23T20:00:00Z')[0], 0.)
        doc['generated_at_utc'] = (now + timedelta(hours=1)).isoformat()
        self.assertEqual(position_weight(doc, 'A', .09, .01, now=now)[0], 0.)


class HoldingsLedgerTests(unittest.TestCase):
    def test_joint_tests_require_both_real_results_and_sufficient_common_history(self):
        good = {'spa_p': .05, 'reality_check_p': .2, 'periods': 30}
        self.assertEqual(gates.joint_test_failures(good), [])
        for bad in [None, {'spa_p': .05}, {**good, 'reality_check_p': float('nan')}, {**good, 'periods': 0}]:
            self.assertTrue(gates.joint_test_failures(bad))

    def test_drift_rebalance_uses_total_returns_including_benchmark(self):
        from research import long_only_backtest
        dates = pd.to_datetime(['2020-01-02'] * 2 + ['2020-01-09'] * 2)
        fills, holdings = [], []
        bt = long_only_backtest([2, 1] * 2, [.1, -.1, 0, 0], dates, ['A', 'B'] * 2,
                                np.zeros(4), 5, top=1., benchmark_returns=[.1, .1, 0, 0],
                                session_positions=[0, 0, 5, 5], rebalance_dates=pd.unique(dates),
                                entry_prices=[100, 100, 120, 100], holdings=holdings, trades=fills)
        self.assertAlmostEqual(bt.iloc[0].nav_end, 1.1)
        self.assertAlmostEqual(bt.iloc[1].nav_start, 1.1)
        drift = [f for f in fills if f['d'] == dates[-1] and f['phase'] == 'rebalance']
        self.assertEqual({(f['symbol'], f['side']) for f in drift}, {('A', 'sell'), ('B', 'buy')})
        for f in drift:
            self.assertAlmostEqual(f['notional_nav_units'], .05)
        self.assertAlmostEqual(bt.iloc[-1].nav_end, 1.1)
        self.assertTrue(all(f['side'] == 'sell' for f in fills if f['phase'] == 'liquidation'))

    def test_fee_ledger_reconciles_nav_and_reweights_retained_names(self):
        from research import long_only_backtest
        dates = pd.to_datetime(['2020-01-02'] * 4 + ['2020-01-09'] * 2)
        fills = []
        bt = long_only_backtest([4, 3, 2, 1, 2, 1], np.zeros(6), dates, ['A', 'B', 'C', 'D', 'A', 'B'],
                                np.full(6, .01), 5, top=.5, benchmark_returns=np.zeros(6),
                                session_positions=[0] * 4 + [5] * 2, rebalance_dates=pd.unique(dates), trades=fills)
        retained = [f for f in fills if f['symbol'] == 'A' and f['d'] == dates[-1] and f['phase'] == 'rebalance']
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0]['side'], 'buy')
        self.assertAlmostEqual(1 - bt.iloc[-1].nav_end, sum(f['fee_nav_units'] for f in fills))
        np.testing.assert_allclose(bt.gross - bt.cost, bt.net)
        self.assertAlmostEqual(float(np.prod(1 + bt.portfolio_return)), bt.iloc[-1].nav_end)

    def test_fixed_buffer_reduces_name_switching_without_looking_at_outcomes(self):
        from research import long_only_backtest
        dates = pd.to_datetime(['2020-01-02'] * 3 + ['2020-01-09'] * 3)
        common = dict(score=[3, 2, 1, 2, 3, 1], y=np.zeros(6), dates=dates, syms=['A', 'B', 'C'] * 2,
                      cost=np.full(6, .001), h=5, top=1 / 3, benchmark_returns=np.zeros(6),
                      session_positions=[0] * 3 + [5] * 3, rebalance_dates=pd.unique(dates))
        normal, buffered = [], []
        standard = long_only_backtest(**common, trades=normal)
        lower = long_only_backtest(**common, trades=buffered, buffer_top=2 / 3)
        self.assertLess(len(buffered), len(normal))
        self.assertGreater(lower.iloc[-1].nav_end, standard.iloc[-1].nav_end)

    def test_missing_selected_outcome_stops_chain_instead_of_substituting_survivor(self):
        from research import long_only_backtest
        dates = pd.to_datetime(['2020-01-02'] * 2)
        bt = long_only_backtest([2, 1], [np.nan, .5], dates, ['A', 'B'], [0, 0], 5, top=.5,
                                benchmark_returns=[0, 0], rebalance_dates=pd.unique(dates))
        self.assertTrue(bt.empty)
        self.assertEqual(bt.attrs['missing_selected_outcome_dates'], ['2020-01-02'])

    def test_vector_random_paths_use_same_drift_and_liquidation_accounting(self):
        import baselines
        from research import long_only_backtest
        dates = pd.to_datetime(['2020-01-02'] * 10 + ['2020-01-09'] * 10)
        symbols = [f'S{i}' for i in range(10)] * 2
        returns = np.array([.1, -.1] * 5 + [0.] * 10)
        costs = np.full(20, .001)
        benchmark = np.array([.1] * 10 + [0.] * 10)
        positions = np.array([0] * 10 + [5] * 10)
        frame = pd.DataFrame({'date': dates, 'symbol': symbols, 'y_5': returns})
        expected = long_only_backtest(np.ones(20), returns, dates, symbols, costs, 5, top=1.,
                                     benchmark_returns=benchmark, session_positions=positions,
                                     rebalance_dates=pd.unique(dates))
        with patch.object(baselines, 'N_PATHS', 3):
            actual = baselines.random_matched(frame, 5, np.ones(20, bool), 1., .2, costs,
                                              rebalance_dates=pd.unique(dates), benchmark_returns=benchmark,
                                              session_positions=positions)
        np.testing.assert_allclose(actual, np.repeat(expected.net.to_numpy()[:, None], 3, axis=1), atol=1e-12)

    def test_self_financing_comparison_writes_reconcilable_fill_files(self):
        from institutional_research import THEMES, write_comparison
        dates = pd.bdate_range('2020-01-01', periods=40)
        panel = pd.DataFrame({'date': np.repeat(dates, 10), 'symbol': list('ABCDEFGHIJ') * 40, 'sector': 'tech',
                              'y_20': np.tile(np.arange(10.) / 100, 40), 'entry_adj_open': 100.})
        for theme in THEMES.values():
            for feature, sign in theme.items():
                panel[feature] = np.tile(np.arange(10.) * sign, 40)
        with tempfile.TemporaryDirectory() as directory:
            report = write_comparison(panel, np.tile(np.arange(10.), 40), np.full(len(panel), .001),
                                      20, dates[20], 'large', directory,
                                      benchmark_returns=np.zeros(len(panel)), session_positions=np.repeat(np.arange(40) * 5, 10))
            json.dumps(report, allow_nan=False)
            self.assertIn('lower_turnover_diagnostic', report)
            fills = pd.read_parquet(Path(directory) / 'institutional_fills_large.parquet')
            periods = pd.read_parquet(Path(directory) / 'institutional_periods_large.parquet')
            self.assertGreater(len(fills), 0)
            for (sample, strategy), g in periods.groupby(['sample', 'strategy']):
                ledger = fills[(fills['sample'] == sample) & (fills.strategy == strategy)]
                for _, period in g.iterrows():
                    fee = ledger.loc[ledger.d == period.d, 'fee_nav_units'].sum()
                    self.assertAlmostEqual(fee / period.nav_start, period.transaction_cost)


if __name__ == '__main__':
    unittest.main()
