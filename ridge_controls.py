"""Prespecified controls shared by the historical ridge evaluations."""
from pathlib import Path

import numpy as np
import pandas as pd


REG = Path(__file__).resolve().parent / "prereg" / "large_ridge_120_v1.json"


def signed_composite(transformed, raw, directions):
    signs = pd.Series({name: sign for name, sign in directions.items() if sign}, dtype=float)
    ranks = transformed[signs.index].where(np.isfinite(raw[signs.index]))
    return ranks.mul(signs).mean(axis=1).to_numpy()


def shuffled_training(X, dates, symbols, seed):
    """Align a single permutation of date blocks by permanent security."""
    dates, symbols = np.asarray(dates), np.asarray(symbols)
    days = np.unique(dates)
    order = np.random.default_rng(seed).permutation(days)
    source_dates = order[np.searchsorted(days, dates)]
    lookup = X.set_axis(pd.MultiIndex.from_arrays([dates, symbols]))
    shuffled = lookup.reindex(pd.MultiIndex.from_arrays([source_dates, symbols]))
    keep = lookup.index.get_indexer(shuffled.index) >= 0
    return shuffled.iloc[keep].set_axis(X.index[keep]), keep


def decision(coverage_ok, ledger_ok, gross, net, paired_net, existing_gate):
    if not coverage_ok or not ledger_ok or gross is None:
        return "UNVALIDATED"
    if gross <= 0:
        return "FAILED"
    if net is not None and paired_net is not None and net > 0 and paired_net > 0 and existing_gate:
        return "ELIGIBLE_FOR_FURTHER_REVIEW"
    return "UNVALIDATED"
