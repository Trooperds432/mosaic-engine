"""Tests for the 12 November 2026 audit-response changes (docs/AUDIT_RESPONSE_2026-11-12.md)."""
import hashlib
import importlib
import os
import tempfile
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import config as C
import gates


class MomentumOrthogonalTests(unittest.TestCase):
    def panel(self):
        rng = np.random.default_rng(3)
        dates = np.repeat(pd.date_range('2020-01-03', periods=4, freq='7D'), 40)
        mom = rng.normal(size=len(dates))
        sue = 0.8 * mom + rng.normal(size=len(dates))  # surprise strongly loaded on momentum
        sue[::9] = np.nan
        mom[5::11] = np.nan
        return pd.DataFrame({'date': dates, 'symbol': [f'S{i}' for i in range(40)] * 4, 'mom_12_1': mom, 'sue': sue})

    def test_residual_is_uncorrelated_with_momentum_rank_on_every_date(self):
        from features import orthogonal_residual
        panel = self.panel()
        out = orthogonal_residual(panel, 'sue')
        ok = panel.mom_12_1.notna() & panel.sue.notna()
        self.assertTrue(out[~ok].isna().all())
        self.assertTrue(out[ok].notna().all())
        for _, g in panel[ok].groupby('date'):
            rx = g.mom_12_1.rank(pct=True) - 0.5
            self.assertAlmostEqual(float(np.corrcoef(rx, out[g.index])[0, 1]), 0.0, places=5)
            self.assertAlmostEqual(float(out[g.index].mean()), 0.0, places=5)
        raw = panel[ok].groupby('date').apply(lambda g: g.sue.rank().corr(g.mom_12_1.rank())).mean()
        self.assertGreater(raw, 0.4)  # the input really was momentum-loaded

    def test_added_to_panel_and_registered(self):
        from features import FEATURES, add_momentum_orthogonal
        panel = self.panel()
        add_momentum_orthogonal(panel)
        self.assertIn('sue_xmom', panel)
        self.assertNotIn('sue_consensus_xmom', panel)  # source column absent -> not invented
        self.assertEqual(FEATURES['sue_xmom'][:2], ('earnings', 1))
        self.assertIn('sue_consensus_xmom', FEATURES)
        self.assertEqual(C.FEATURE_SET_VERSION, 'fs-2.7')

    def test_small_cross_sections_stay_missing(self):
        from features import orthogonal_residual
        panel = pd.DataFrame({'date': pd.to_datetime(['2020-01-03'] * 2), 'mom_12_1': [1., 2.], 'sue': [1., 0.]})
        self.assertTrue(orthogonal_residual(panel, 'sue').isna().all())


class BufferedEnsembleTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        n_dates, n_names = 12, 50
        self.dates = np.repeat(pd.date_range('2015-01-02', periods=n_dates, freq='28D'), n_names)
        persistent = np.tile(rng.normal(size=n_names), n_dates)
        self.scores = persistent + 0.7 * rng.normal(size=len(self.dates))
        self.y = rng.normal(0, .05, size=len(self.dates))
        self.syms = np.array([f'S{i:02d}' for i in range(n_names)] * n_dates)
        self.cost = np.full(len(self.dates), .003)
        positions = np.repeat(np.arange(n_dates) * 20, n_names)
        from research import long_only_backtest
        self.backtest = partial(long_only_backtest, benchmark_returns=np.zeros(len(self.dates)), session_positions=positions,
                                rebalance_dates=pd.unique(self.dates))

    def test_buffer_lowers_turnover_and_cost_and_records_prereg(self):
        from research import TURNOVER_PREREG, bt_summary, buffered_ensemble
        holdout = np.full(len(self.scores), np.nan)
        holdout[-150:] = self.scores[-150:]
        out = buffered_ensemble(self.backtest, self.scores, holdout, self.y, self.dates, self.syms, self.cost, 20)
        s = out['summary']
        plain = bt_summary(self.backtest(self.scores, self.y, self.dates, self.syms, self.cost, 20, top=.2), 20)
        self.assertEqual(s['name'], 'model:ensemble_buffered')
        self.assertEqual(s['prereg_sha256'], hashlib.sha256(TURNOVER_PREREG.read_bytes()).hexdigest())
        self.assertLess(s['backtest_wf']['mean_turnover'], plain['mean_turnover'])
        self.assertLess(s['backtest_wf']['mean_cost'], plain['mean_cost'])
        self.assertEqual(len(out['wf_net']), plain['periods'])

    def test_registered_horizon_is_enforced(self):
        from research import buffered_ensemble
        with self.assertRaises(ValueError):
            buffered_ensemble(self.backtest, self.scores, self.scores, self.y, self.dates, self.syms, self.cost, 5)

    def test_invalid_prereg_rejected(self):
        from research import turnover_prereg
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'bad.json'
            p.write_text('{"id": "x", "variant": {"entry_quantile": 0.4, "buffer_top": 0.2}}')
            with self.assertRaises(ValueError):
                turnover_prereg(p)


class PBOGateTests(unittest.TestCase):
    def test_fails_closed_and_thresholds(self):
        self.assertEqual(gates.pbo_failures(.3, C.GATE_MAX_PBO), [])
        self.assertEqual(gates.pbo_failures(.5, C.GATE_MAX_PBO), [])
        self.assertEqual(gates.pbo_failures(.71, C.GATE_MAX_PBO), ['PBO 0.71 > 0.50'])
        for missing in [None, {'error': 'x'}, float('nan'), 1.5, True]:
            self.assertEqual(gates.pbo_failures(missing, C.GATE_MAX_PBO), ['PBO (CSCV) not computed or invalid'])


class PaperTradingPathTests(unittest.TestCase):
    def test_env_path_and_missing_log_reported(self):
        import paper_trading
        with tempfile.TemporaryDirectory() as d:
            missing = Path(d) / 'trades.jsonl'
            with patch.dict(os.environ, {'MOSAIC_T212_TRADES': str(missing)}):
                module = importlib.reload(paper_trading)
                self.assertEqual(module.BOT_TRADES, missing)
                out = module.executed()
        importlib.reload(paper_trading)
        self.assertEqual(out['trade_log'], 'NOT_FOUND')
        self.assertEqual(out['trades'], 0)


if __name__ == '__main__':
    unittest.main()
