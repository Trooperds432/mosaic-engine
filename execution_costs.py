"""Audit normalized execution evidence against costs quoted BEFORE submission.

Input CSV: one row per actual fill, not order acknowledgement. No credentials or
broker writes. Signed implementation shortfall includes spread/impact together;
fees and FX are added once. Paper fills never qualify as live cost evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ['fill_id', 'order_id', 'symbol', 'side', 'quantity', 'fill_price', 'arrival_mid',
          'arrival_at', 'submitted_at', 'filled_at', 'fees_usd', 'fx_cost_usd',
          'modeled_cost_bps', 'environment']


def audit(fills, min_orders=100, min_days=20, max_reference_age_seconds=60):
    if min_orders < 1 or min_days < 1 or max_reference_age_seconds <= 0:
        raise ValueError('positive evidence thresholds required')
    missing = set(FIELDS) - set(fills.columns)
    if missing:
        raise ValueError(f'missing execution fields: {sorted(missing)}')
    d = fills[FIELDS].copy()
    if d.isna().any().any():
        raise ValueError('execution fields must be complete')
    if d.duplicated(['environment', 'fill_id']).any():
        raise ValueError('duplicate fill IDs')
    if not d.environment.isin(['paper', 'live']).all() or not d.side.isin(['buy', 'sell']).all():
        raise ValueError('invalid execution environment or side')
    for name in ['fill_id', 'order_id', 'symbol']:
        if d[name].astype(str).str.strip().eq('').any():
            raise ValueError('empty execution identifier')
    for name in ['quantity', 'fill_price', 'arrival_mid', 'fees_usd', 'fx_cost_usd', 'modeled_cost_bps']:
        d[name] = pd.to_numeric(d[name], errors='raise')
        if not np.isfinite(d[name]).all():
            raise ValueError('nonfinite execution values')
    if (d[['quantity', 'fill_price', 'arrival_mid']] <= 0).any().any() or (d[['fees_usd', 'fx_cost_usd', 'modeled_cost_bps']] < 0).any().any():
        raise ValueError('invalid price, quantity or cost')
    for name in ['arrival_at', 'submitted_at', 'filled_at']:
        # Require explicit zones: interpreting a local timestamp as UTC is unsafe.
        if not d[name].astype(str).str.contains(r'(?:Z|[+-]\d{2}:\d{2})$', regex=True).all():
            raise ValueError('execution timestamps require explicit time zones')
        d[name] = pd.to_datetime(d[name], utc=True, errors='raise')
    age = (d.submitted_at - d.arrival_at).dt.total_seconds()
    if (age < 0).any() or (age > max_reference_age_seconds).any() or (d.filled_at < d.submitted_at).any():
        raise ValueError('future/stale arrival reference or fill before submission')
    # All partial fills of one order must carry one immutable arrival benchmark
    # and model quote. Fees/FX must be allocated per fill, not repeated per order.
    invariant = ['symbol', 'side', 'arrival_mid', 'arrival_at', 'submitted_at', 'modeled_cost_bps']
    if (d.groupby(['environment', 'order_id'])[invariant].nunique() > 1).any().any():
        raise ValueError('inconsistent partial-fill order benchmark')
    sign = d.side.map({'buy': 1., 'sell': -1.})
    d['notional'] = d.quantity * d.arrival_mid
    d['shortfall_usd'] = sign * d.quantity * (d.fill_price - d.arrival_mid)
    d['observed_usd'] = d.shortfall_usd + d.fees_usd + d.fx_cost_usd
    d['modeled_usd'] = d.notional * d.modeled_cost_bps / 10000
    out = {'schema_version': 1, 'changes_cost_assumptions': False, 'environments': {},
           'policy': {'min_orders': min_orders, 'min_days': min_days,
                      'max_reference_age_seconds': max_reference_age_seconds},
           'limitations': 'Filled quantity only; excludes unfilled-order opportunity costs and borrow. '
                          'Sample sufficiency is diagnostic, not authorization to recalibrate or trade.'}
    for environment, g in d.groupby('environment'):
        orders = int(g.order_id.nunique())
        days = int(g.submitted_at.dt.tz_convert('America/New_York').dt.date.nunique())
        notional = float(g.notional.sum())
        observed = float(g.observed_usd.sum() / notional * 10000)
        modeled = float(g.modeled_usd.sum() / notional * 10000)
        out['environments'][environment] = {
            'fills': len(g), 'orders': orders, 'trading_dates': days, 'arrival_notional_usd': notional,
            'observed_cost_bps': observed, 'modeled_cost_bps': modeled,
            'observed_minus_modeled_bps': observed - modeled,
            'sufficient_live_evidence': environment == 'live' and orders >= min_orders and days >= min_days}
    if not len(d):
        out['status'] = 'NO_EXECUTION_EVIDENCE'
    else:
        out['status'] = 'DIAGNOSTIC_ONLY'
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('fills_csv', type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--min-orders', type=int, default=100)
    p.add_argument('--min-days', type=int, default=20)
    args = p.parse_args()
    result = audit(pd.read_csv(args.fills_csv, dtype={'fill_id': str, 'order_id': str, 'symbol': str}),
                   min_orders=args.min_orders, min_days=args.min_days)
    result['source_sha256'] = hashlib.sha256(args.fills_csv.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
