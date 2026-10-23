"""Self-financing long-only research ledger, using adjusted next-open returns.

Fills are modeled cash flows in initial-NAV units, not broker executions. Costs
are charged on all dollar changes, including drift rebalances and liquidation.
"""
import numpy as np
import pandas as pd


def post_fee_nav(previous, weights, rates, before):
    """Solve the self-financing equation for one portfolio or a batch of paths."""
    previous, weights, rates = np.asarray(previous), np.asarray(weights), np.asarray(rates)
    before = np.asarray(before)
    value = before.copy()
    for _ in range(12):
        signs = np.sign(weights * np.expand_dims(value, -1) - previous)
        numerator = before + np.sum(previous * signs * rates, axis=-1)
        denominator = 1 + np.sum(weights * signs * rates, axis=-1)
        candidate = numerator / denominator
        residual = candidate + np.sum(np.abs(weights * np.expand_dims(candidate, -1) - previous) * rates, axis=-1) - before
        if np.all(np.abs(residual) <= np.maximum(before, 1) * 1e-12):
            return candidate
        value = candidate
    raise ValueError('self-financing cost equation did not converge')


def simulate(frame, schedule, top, cost_mult=1., buffer_top=None, holdings=None, trades=None):
    if buffer_top is not None and not top <= buffer_top <= 1:
        raise ValueError('holding buffer must be between entry fraction and one')
    if not 0 < top <= 1 or not np.isfinite(cost_mult) or cost_mult < 0:
        raise ValueError('invalid selection/cost policy')
    if frame.duplicated(['d', 'sym']).any():
        raise ValueError('duplicate security/decision-date rows')
    if not schedule.is_unique or not schedule.is_monotonic_increasing:
        raise ValueError('rebalance schedule must be ordered and unique')
    groups = {d: g for d, g in frame[frame.d.isin(schedule)].groupby('d')}
    cash, values, last_costs, rows, missing = 1., {}, {}, [], []
    periods = []
    for d in schedule:
        g = groups.get(d)
        if g is None:
            continue
        # Whole-period benchmark unavailable = horizon not yet measurable.
        if not np.isfinite(g.b).any():
            continue
        usable = g[np.isfinite(g.s) & np.isfinite(g.c) & (g.c >= 0)].copy()
        if usable.empty:
            continue
        usable = usable.sort_values(['s', 'sym'], ascending=[False, True])
        periods.append((d, usable, g))
    for i, (d, g, all_names) in enumerate(periods):
        if i + 1 < len(periods) and int(periods[i + 1][1].position.iloc[0]) < int(g.exit_position.iloc[0]):
            raise ValueError('overlapping holding periods: provide a non-overlapping decision calendar')
        k = max(1, int(round(len(g) * top)))
        if buffer_top is not None and values:
            retained = g.iloc[:max(k, int(np.ceil(len(g) * buffer_top)))].loc[lambda x: x.sym.isin(values)].head(k)
            pick = pd.concat([retained, g[~g.sym.isin(retained.sym)].head(k - len(retained))])
        else:
            pick = g.head(k)
        # Never replace a chosen name because its future outcome is unavailable.
        if not np.isfinite(pick[['y', 'b', 'relative_benchmark']]).all().all():
            missing.append(str(pd.Timestamp(d).date()))
            break  # an unknown held outcome breaks the NAV chain; no silent bridge
        benchmark = float(pick.b.iloc[0])
        relative = float(pick.relative_benchmark.iloc[0])
        if pick.b.nunique() != 1 or pick.relative_benchmark.nunique() != 1:
            raise ValueError('one benchmark return per period required')
        returns = dict(zip(pick.sym, pick.y + benchmark))
        if any(r < -1 - 1e-6 for r in returns.values()):
            raise ValueError('asset total return below minus 100 percent')
        returns = {s: max(-1., float(r)) for s, r in returns.items()}
        before = cash + sum(values.values())
        if not before > 0:
            raise ValueError('portfolio has no remaining equity')
        w = 1. / k
        names = sorted(set(values) | set(pick.sym))
        previous = np.array([values.get(s, 0.) for s in names])
        weights = np.array([w if s in returns else 0. for s in names])
        current_costs = all_names.set_index('sym').c.to_dict()
        rates = np.array([current_costs.get(s, last_costs.get(s, np.nan)) for s in names], dtype=float) * cost_mult
        if not np.isfinite(rates).all() or (rates < 0).any() or (rates >= 1).any():
            raise ValueError('missing or invalid modeled execution cost')
        # Post-fee target NAV solves x + sum(abs(w*x - held_value)*rate) = NAV.
        invested = float(post_fee_nav(previous, weights, rates, before))
        target = weights * invested
        deltas = target - previous
        fee = float(np.abs(deltas) @ rates)
        current_values = {s: float(v) for s, v in zip(names, target) if s in returns}
        cash = max(0., before - fee - sum(current_values.values()))
        prices = all_names.set_index('sym').entry_price.to_dict()
        if trades is not None:
            for s, old, new, delta, rate in zip(names, previous, target, deltas, rates):
                if abs(delta) <= before * 1e-12:
                    continue
                price = prices.get(s, np.nan)
                trades.append({'d': d, 'symbol': s, 'phase': 'rebalance', 'side': 'buy' if delta > 0 else 'sell',
                               'nav_before': before, 'value_before': float(old), 'value_after': float(new),
                               'notional_nav_units': float(abs(delta)), 'fee_nav_units': float(abs(delta) * rate),
                               'modeled_rate': float(rate), 'adjusted_price': float(price) if np.isfinite(price) and price > 0 else None,
                               'adjusted_units': float(abs(delta) / price) if np.isfinite(price) and price > 0 else None,
                               'cost_source': 'current_decision' if s in current_costs else 'previous_decision_estimate'})
        if holdings is not None:
            holdings.extend({'d': d, 'symbol': r.sym, 'weight': w, 'score': float(r.s),
                             'value_nav_units': current_values[r.sym], 'realized_excess': float(r.y),
                             'realized_total_return': returns[r.sym], 'one_way_cost': float(r.c)} for r in pick.itertuples())
        values = {s: value * (1 + returns[s]) for s, value in current_values.items()}
        last_costs.update({s: float(current_costs[s]) for s in returns})
        gross_total = float(np.mean(list(returns.values())))
        ending = cash + sum(values.values())
        liquidation = 0.
        # Fixed h-session labels only connect when the next entry is this exit.
        # Cash gaps and sample end require liquidation, never fictitious carry.
        contiguous = i + 1 < len(periods) and int(periods[i + 1][1].position.iloc[0]) == int(g.exit_position.iloc[0])
        if not contiguous:
            for s, value in values.items():
                exit_fee = value * last_costs[s] * cost_mult
                liquidation += exit_fee
                if trades is not None and value > 1e-15:
                    trades.append({'d': d, 'symbol': s, 'phase': 'liquidation', 'side': 'sell',
                                   'nav_before': ending, 'value_before': value, 'value_after': 0.,
                                   'notional_nav_units': value, 'fee_nav_units': exit_fee,
                                   'modeled_rate': last_costs[s] * cost_mult,
                                   'adjusted_price': None, 'adjusted_units': None,
                                   'cost_source': 'last_decision_estimate'})
            ending -= liquidation
            cash, values = ending, {}
        reported_gross = gross_total - benchmark - relative
        net = ending / before - 1 - benchmark - relative
        rows.append({'d': d, 'gross': reported_gross, 'net': net, 'cost': reported_gross - net,
                     'transaction_cost': (fee + liquidation) / before,
                     'turnover': float(np.abs(deltas).sum() / (2 * before)), 'n': k,
                     'nav_start': before, 'nav_end': ending, 'portfolio_return': ending / before - 1,
                     'benchmark_return': benchmark + relative})
    # If an interior unknown outcome stopped the chain, close the last assessed
    # period at its exit rather than silently leaving an unpriced position open.
    if values and rows:
        fee = sum(v * last_costs[s] * cost_mult for s, v in values.items())
        last = rows[-1]
        for s, value in values.items():
            if trades is not None and value > 1e-15:
                trades.append({'d': last['d'], 'symbol': s, 'phase': 'liquidation', 'side': 'sell',
                               'nav_before': last['nav_end'], 'value_before': value, 'value_after': 0.,
                               'notional_nav_units': value, 'fee_nav_units': value * last_costs[s] * cost_mult,
                               'modeled_rate': last_costs[s] * cost_mult, 'adjusted_price': None, 'adjusted_units': None,
                               'cost_source': 'last_decision_estimate'})
        last['nav_end'] -= fee
        last['portfolio_return'] -= fee / last['nav_start']
        last['net'] -= fee / last['nav_start']
        last['cost'] += fee / last['nav_start']
        last['transaction_cost'] += fee / last['nav_start']
    result = pd.DataFrame(rows).set_index('d') if rows else pd.DataFrame(columns=[
        'gross', 'net', 'cost', 'transaction_cost', 'turnover', 'n', 'nav_start', 'nav_end', 'portfolio_return', 'benchmark_return'])
    result.attrs.update(accounting='self_financing_adjusted_return_drift', missing_selected_outcome_dates=missing,
                        fills='modeled_not_broker', terminal_liquidation=True)
    return result
