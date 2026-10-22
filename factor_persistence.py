# Copyright 2017 Quantopian, Inc.
# Licensed under Apache-2.0; see third_party/ALPHALENS_LICENSE.
# Adapted from Alphalens Reloaded performance.py at
# 1bc181de4aafbe33437ed0e362ff96aa7ce20acd.
# Modifications: preserve the complete observed date grid, reject duplicate
# security-dates, and leave missing/empty quantiles unknown rather than skipping dates.
"""Alphalens signal persistence diagnostics on Mosaic decision observations."""
import numpy as np
import pandas as pd


def quantile_turnover(quantile_factor, quantile, period=1):
    dates = quantile_factor.index.get_level_values("date").unique().sort_values()
    selected = quantile_factor[quantile_factor == quantile]
    sets = selected.groupby(level="date").apply(lambda group: set(group.index.get_level_values("asset"))).reindex(dates)
    previous = sets.shift(period)
    return pd.Series([len(now - old) / len(now) if isinstance(now, set) and isinstance(old, set) and now else np.nan
                      for now, old in zip(sets, previous)], index=dates, name=quantile)


def factor_rank_autocorrelation(factor_data, period=1):
    ranks = factor_data.groupby(level="date")["factor"].rank().unstack("asset")
    return ranks.corrwith(ranks.shift(period), axis=1).rename(period)


def summary(scores, returns, dates, securities, lags=(1, 4, 12), quantiles=5):
    frame = pd.DataFrame({"date": pd.to_datetime(dates), "asset": securities, "factor": scores, "forward_return": returns})
    if frame.duplicated(["date", "asset"]).any():
        raise ValueError("factor diagnostics require unique security-date observations")
    frame = frame.set_index(["date", "asset"]).sort_index()
    frame["factor"] = frame["factor"].where(np.isfinite(frame["factor"]))
    frame["forward_return"] = frame["forward_return"].where(np.isfinite(frame["forward_return"]))
    ranks = frame.groupby(level="date")["factor"].rank(pct=True)
    frame["quantile"] = np.ceil(ranks * quantiles).clip(1, quantiles)
    finite = lambda series: float(series.mean()) if series.notna().any() else None
    persistence = {}
    for lag in lags:
        if isinstance(lag, bool) or not isinstance(lag, int) or lag < 1:
            raise ValueError("persistence lags must be positive decision-observation counts")
        ac = factor_rank_autocorrelation(frame, lag)
        turnover = quantile_turnover(frame["quantile"], quantiles, lag)
        persistence[str(lag)] = {"rank_autocorrelation": finite(ac), "top_quantile_turnover": finite(turnover),
                                 "rank_pairs": int(ac.notna().sum()), "turnover_pairs": int(turnover.notna().sum())}
    quantile_means = frame.groupby([pd.Grouper(level="date"), "quantile"])["forward_return"].mean().unstack("quantile")
    return {"implementation": "alphalens_reloaded_adapted", "lag_unit": "observed_decision_dates",
            "periods": int(frame.index.get_level_values("date").nunique()), "persistence": persistence,
            "gross_excess_by_quantile": {str(int(q)): finite(quantile_means[q]) for q in quantile_means},
            "note": "gross label returns; membership turnover is not portfolio dollar turnover or a transaction-cost estimate"}
