# Third-party integrations

`factor_persistence.py` adapts `quantile_turnover` and
`factor_rank_autocorrelation` from Alphalens Reloaded, copyright 2017 Quantopian,
Inc., Apache-2.0. Source: https://github.com/stefan-jansen/alphalens-reloaded/blob/1bc181de4aafbe33437ed0e362ff96aa7ce20acd/src/alphalens/performance.py.
The complete licence is retained in `third_party/ALPHALENS_LICENSE`. The inspected
upstream revision has no NOTICE file. Modifications preserve all observed dates,
leave empty quantiles unknown, reject duplicate security-dates, and provide a
Mosaic-specific research summary. No upstream branding implies endorsement.

The following are imported installed dependencies, not copied source:

- `arch==8.0.0`: SPA/RealityCheck with stationary resampling, in addition to the
  existing GARCH usage. https://github.com/bashtage/arch; licence in the distribution.
- `skfolio==1.9.13`: combinatorial date partitions and backtest path IDs.
  https://github.com/skfolio/skfolio; BSD-3-Clause.
- `scikit-learn==1.9.1`: LedoitWolf covariance estimation, in addition to existing
  prediction models. https://github.com/scikit-learn/scikit-learn; BSD-3-Clause.

Reviewed revisions, integration decisions and the tested installed implementation
hashes are recorded in `third_party/upstream_review.json`. Those GitHub review
snapshots are distinct from the pinned PyPI versions used at runtime.
