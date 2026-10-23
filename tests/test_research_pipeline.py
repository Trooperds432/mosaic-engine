"""Small synthetic full research run: real ridge fits, costs, ledgers and joint tests."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import config as C
import research
from institutional_research import THEMES


class ResearchPipelineTests(unittest.TestCase):
    def test_full_run_routes_actual_benchmark_returns_into_comparison_and_joint_tests(self):
        rng = np.random.default_rng(81)
        calendar = pd.bdate_range('2009-01-01', periods=1800)
        prices = 100 * np.exp(np.cumsum(rng.normal(.0002, .01, len(calendar))))
        spy = pd.DataFrame({'date': calendar, 'open': prices, 'close': prices, 'adjclose': prices})
        decisions = calendar[260::5]
        names = [f'S{i}' for i in range(60)]
        panel = pd.DataFrame({'date': np.repeat(decisions, 60), 'symbol': names * len(decisions),
                              'sector': np.tile(np.repeat(['tech', 'energy', 'health'], 20), len(decisions)),
                              'entry_adj_open': 100., 'cs_spread': .001, 'vol_60': .2,
                              'adv_dollar': 1e7, 'mcap': 2e10})
        for theme in THEMES.values():
            for feature in theme:
                panel[feature] = rng.normal(size=len(panel))
        panel['log_mcap'] = rng.normal(23, 1, len(panel))
        panel['beta_252'] = rng.uniform(.5, 1.5, len(panel))
        panel['y_20'] = .005 * panel.mom_12_1 + rng.normal(0, .03, len(panel))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, proc, reports, ledger = [root / d for d in ['raw', 'proc', 'reports', 'ledger']]
            for d in [raw / 'prices', proc, reports, ledger]:
                d.mkdir(parents=True)
            spy.to_parquet(raw / 'prices/SPY.parquet', index=False)
            pd.DataFrame({'date': calendar, 'close': 20 + rng.random(len(calendar))}).to_parquet(raw / 'prices/^VIX.parquet', index=False)
            panel.to_parquet(proc / 'panel_large.parquet', index=False)
            with patch.object(C, 'ROOT', root), patch.object(C, 'RAW', raw), patch.object(C, 'PROC', proc), \
                 patch.object(C, 'REPORTS', reports), patch.object(C, 'LEDGER', ledger), patch.object(C, 'HORIZONS', [20]), \
                 patch.object(C, 'WF_MIN_TRAIN_YEARS', 1), patch.object(research, 'MODELS_DIR', root / 'models'), \
                 patch.object(research, 'FULL_SUITE_H', ()), patch('risk_limits.load', return_value={'order_size_usd': 600}), \
                 patch.object(research, 'hmm_filtered', return_value=pd.Series(.5, index=calendar)), \
                 patch('historical_data.coverage_check', return_value={'passed': False, 'failed_checks': ['synthetic data only']}), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = research.run('large')
            primary = result['models'][20]
            self.assertEqual(primary['models']['ensemble']['backtest_holdout']['accounting'], 'self_financing_adjusted_return_drift')
            self.assertTrue((reports / 'institutional_fills_large.parquet').exists())
            self.assertTrue((reports / 'oos_scores_large_h20.parquet').exists())
            self.assertEqual(result['overfitting']['reality_check_spa']['implementation'], 'arch.bootstrap')
            self.assertGreaterEqual(result['overfitting']['reality_check_spa']['periods'], 20)
            self.assertIn('synthetic data only', primary['models']['ensemble']['gate']['failed_checks'])
            report = json.loads((reports / 'institutional_comparison_large.json').read_text())
            json.dumps(report, allow_nan=False)
            self.assertEqual(report['cost_convention'], 'self-financing with weight drift and terminal liquidation')
            self.assertTrue((root / 'models/large_h20_ridge.joblib').exists())


if __name__ == '__main__':
    unittest.main()
