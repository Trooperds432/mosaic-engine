"""Fixed economic benchmark and paired diagnostics; never a deployment approval.

This is Mosaic's own transparent VQM recipe, not a replica of a manager's
proprietary model. Every component must be observed; no outcome informs scoring.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

THEMES = {
    'value': {'earn_yield': 1, 'book_to_mkt': 1, 'fcf_yield': 1},
    'quality': {'gross_prof': 1, 'roa': 1, 'roic': 1, 'accruals': -1, 'leverage': -1},
    'momentum': {'mom_12_1': 1, 'mom_6_1': 1},
}
RECIPE = {'version': 'vqm-1', 'themes': THEMES, 'theme_weights': {k: 1 / 3 for k in THEMES},
          'scaling': 'within decision date and sector percentile ranks',
          'missing_policy': 'all ten components required, no neutral imputation',
          'fit': 'none; fixed signs and equal theme weights; no holdout tuning'}


def fixed_vqm(panel):
    key = [panel['date'], panel['sector'].fillna('Unknown')]
    themes = []
    complete = pd.Series(True, index=panel.index)
    for components in THEMES.values():
        ranks = []
        for feature, sign in components.items():
            raw = pd.to_numeric(panel[feature], errors='coerce').replace([np.inf, -np.inf], np.nan)
            complete &= raw.notna()
            ranks.append((sign * raw).groupby(key).rank(pct=True))
        themes.append(pd.concat(ranks, axis=1).mean(axis=1))
    return pd.concat(themes, axis=1).mean(axis=1).where(complete)


def compare(panel, mosaic_score, cost, h, holdout_start, top=.2, *, benchmark_returns=None, session_positions=None,
            buffer_top=None, trade_ledger=None):
    """Compare fixed VQM and OOS Mosaic on identical dates, eligible names and costs.

    Selection eligibility uses only feature/score/cost availability, never labels.
    Missing selected outcomes make that period unassessable for BOTH strategies.
    A result with exclusions is explicitly incomplete, not repaired with zeros.
    """
    from research import bt_summary, hac_mean, long_only_backtest, rebalance_calendar
    score = np.asarray(mosaic_score, dtype=float)
    costs = np.asarray(cost, dtype=float)
    vqm = fixed_vqm(panel).to_numpy()
    eligible = np.isfinite(score) & np.isfinite(vqm) & np.isfinite(costs) & (costs >= 0)
    dates = panel['date'].to_numpy()
    outcomes = panel[f'y_{h}'].to_numpy()
    schedule = rebalance_calendar(dates, h)
    accounting = {'benchmark_returns': benchmark_returns, 'session_positions': session_positions,
                  'entry_prices': panel['entry_adj_open'].to_numpy() if 'entry_adj_open' in panel else None,
                  'buffer_top': buffer_top}
    out = {'recipe': RECIPE, 'horizon': h, 'top_fraction': top, 'eligible_rows': int(eligible.sum()),
           'excluded_rows': int((~eligible).sum()), 'samples': {},
           'status': 'DIAGNOSTIC_ONLY', 'eligibility_uses_future_returns': False,
           'cost_convention': 'self-financing with weight drift and terminal liquidation' if benchmark_returns is not None else 'legacy excess-return approximation; benchmark returns not supplied',
           'buffer_top': buffer_top,
           'approval': 'No automatic promotion. Reused holdout is descriptive, not fresh confirmation.'}
    period_frames, holding_frames = [], []
    for sample, mask in [('walk_forward', dates < np.datetime64(holdout_start)),
                         ('holdout', dates >= np.datetime64(holdout_start))]:
        cal = schedule[schedule < holdout_start] if sample == 'walk_forward' else schedule[schedule >= holdout_start]
        portfolios, ledger, fill_ledgers, missing = {}, {}, {}, set()
        for name, scores in [('mosaic', score), ('vqm', vqm)]:
            records, fills = [], []
            bt = long_only_backtest(np.where(mask & eligible, scores, np.nan), outcomes, dates,
                                    panel['symbol'].to_numpy(), costs, h, top=top,
                                    rebalance_dates=cal, holdings=records,
                                    trades=fills if benchmark_returns is not None else None, **accounting)
            portfolios[name], ledger[name] = bt, records
            fill_ledgers[name] = fills
            missing.update(bt.attrs['missing_selected_outcome_dates'])
        common = portfolios['mosaic'].index.intersection(portfolios['vqm'].index)
        expected = cal[cal.isin(pd.DatetimeIndex(pd.unique(dates[mask & eligible])))]
        if benchmark_returns is not None:
            measurable = dates[np.isfinite(benchmark_returns)]
            if len(measurable):
                expected = expected[expected <= pd.Timestamp(measurable.max())]
            else:
                expected = expected[:0]
        omitted = expected.difference(common)
        # Re-run both after omissions, so each uses the same predecessor holdings.
        if len(omitted):
            for name, scores in [('mosaic', score), ('vqm', vqm)]:
                records, fills = [], []
                portfolios[name] = long_only_backtest(np.where(mask & eligible, scores, np.nan), outcomes, dates,
                                                      panel['symbol'].to_numpy(), costs, h, top=top,
                                                      rebalance_dates=common, holdings=records,
                                                      trades=fills if benchmark_returns is not None else None, **accounting)
                ledger[name] = records
                fill_ledgers[name] = fills
        paired = {name: bt.loc[common] for name, bt in portfolios.items()}
        delta = paired['mosaic']['net'] - paired['vqm']['net']
        delta_stats = hac_mean(delta, 2)
        delta_stats['mean'] = float(delta.mean()) if len(delta) else None
        delta_stats = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in delta_stats.items()}
        result = {'paired_periods': len(common), 'unassessable_periods': [str(d.date()) for d in omitted],
                  'missing_selected_outcome_dates': sorted(missing),
                  'mosaic': bt_summary(paired['mosaic'], h), 'vqm': bt_summary(paired['vqm'], h),
                  'mosaic_minus_vqm_net': delta_stats,
                  'cost_sensitivity': {}}
        for multiple in [0., .5, 1., 2.]:
            scenario = {}
            for name, scores in [('mosaic', score), ('vqm', vqm)]:
                if benchmark_returns is not None:
                    bt = long_only_backtest(np.where(mask & eligible, scores, np.nan), outcomes, dates,
                                            panel['symbol'].to_numpy(), costs, h, top=top, rebalance_dates=common,
                                            cost_mult=multiple, **accounting)
                    scenario[name] = float(bt.net.mean()) if len(bt) else None
                else:
                    bt = paired[name]
                    scenario[name] = float((bt.gross - multiple * bt.cost).mean()) if len(bt) else None
            result['cost_sensitivity'][str(multiple)] = scenario
        out['samples'][sample] = result
        if len(omitted):
            out['status'] = 'INCOMPLETE_OUTCOMES'
        for name, bt in paired.items():
            period_frames.append(bt.rename_axis('d').reset_index().assign(sample=sample, strategy=name))
            holding_frames.extend(dict(r, sample=sample, strategy=name) for r in ledger[name] if r['d'] in common)
            if trade_ledger is not None:
                trade_ledger.extend(dict(r, sample=sample, strategy=name) for r in fill_ledgers[name] if r['d'] in common)
    return out, pd.concat(period_frames, ignore_index=True), pd.DataFrame(holding_frames)


def write_comparison(panel, scores, costs, h, holdout_start, profile, output_dir, metadata=None, top=.2,
                     benchmark_returns=None, session_positions=None):
    """Shared writer for the full research run and replay of saved OOS scores."""
    trades = []
    report, periods, holdings = compare(panel, scores, costs, h, pd.Timestamp(holdout_start), top=top,
                                       benchmark_returns=benchmark_returns, session_positions=session_positions, trade_ledger=trades)
    if benchmark_returns is not None:
        buffered, _, _ = compare(panel, scores, costs, h, pd.Timestamp(holdout_start), top=top,
                                  benchmark_returns=benchmark_returns, session_positions=session_positions,
                                  buffer_top=max(top, .4))
        report['lower_turnover_diagnostic'] = {'policy': 'retain existing names within top 40%, up to target name count',
                                              'samples': buffered['samples'], 'status': buffered['status'],
                                              'selection': 'fixed diagnostic; never selects production rules from holdout performance'}
    report.update(metadata or {})
    report['profile'] = profile
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f'institutional_comparison_{profile}.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    periods.to_parquet(output_dir / f'institutional_periods_{profile}.parquet', index=False)
    holdings.to_parquet(output_dir / f'institutional_holdings_{profile}.parquet', index=False)
    if benchmark_returns is not None:
        pd.DataFrame(trades).to_parquet(output_dir / f'institutional_fills_{profile}.parquet', index=False)
    return report


def main():
    parser = argparse.ArgumentParser(description='Replay VQM comparison from genuine saved OOS scores, never final fitted weights.')
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--benchmark-prices', type=Path, help='benchmark parquet with date/open/close/adjclose for drift accounting')
    parser.add_argument('--profile', choices=['large', 'broad'], required=True)
    parser.add_argument('--horizon', type=int, default=20)
    parser.add_argument('--holdout-start', required=True)
    parser.add_argument('--order-usd', type=float, required=True)
    parser.add_argument('--top-fraction', type=float, default=.2)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.order_usd <= 0 or not np.isfinite(args.order_usd):
        parser.error('order size must be finite and positive')
    panel = pd.read_parquet(args.panel)
    scores = pd.read_parquet(args.scores)
    if panel.duplicated(['date', 'symbol']).any() or scores.duplicated(['date', 'symbol']).any():
        raise ValueError('duplicate panel or score keys')
    expected_sample = np.where(scores.date < pd.Timestamp(args.holdout_start), 'walk_forward', 'holdout')
    if 'sample' not in scores or not np.array_equal(scores['sample'], expected_sample):
        raise ValueError('OOS score sample labels differ from holdout boundary')
    panel = panel.merge(scores[['date', 'symbol', 'ensemble_score']], on=['date', 'symbol'], how='left', validate='one_to_one')
    from historical_data import coverage_check
    from research import name_costs, benchmark_label_returns
    benchmark_returns, positions = None, None
    if args.benchmark_prices is not None:
        spy = pd.read_parquet(args.benchmark_prices).drop_duplicates('date').set_index('date').sort_index()
        benchmark_returns = benchmark_label_returns(spy, panel.date, args.horizon)
        positions = spy.index.get_indexer(panel.date)
    report = write_comparison(panel, panel.ensemble_score, name_costs(panel, args.order_usd), args.horizon,
                              args.holdout_start, args.profile, args.output_dir,
                              {'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.panel, args.scores, *([args.benchmark_prices] if args.benchmark_prices else [])]},
                               'historical_integrity': coverage_check(str(panel.date.min().date()), str(panel.date.max().date())),
                               'order_size_usd': args.order_usd}, top=args.top_fraction,
                              benchmark_returns=benchmark_returns, session_positions=positions)
    print(json.dumps({'status': report['status'], 'profile': args.profile, 'output_dir': str(args.output_dir)}))


if __name__ == '__main__':
    main()
