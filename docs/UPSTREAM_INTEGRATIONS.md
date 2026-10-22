# GitHub review and Mosaic integrations

Reviewed on 22 October 2026 against Mosaic release a8effd2. Selection criteria:
does it close a concrete gap, is the source inspectable, is reuse authorised by
its licence, does it fit Python 3.13 and the existing architecture, and can the
integration be tested without making historical performance claims?

| Repository | Decision and observable change |
| --- | --- |
| [arch](https://github.com/bashtage/arch) | Reuse the already pinned package's SPA and RealityCheck. Replaces Mosaic's simplified standardisation with maintained stationary-bootstrap tests. Missing strategy returns are excluded on a common sample, never turned into flat returns. |
| [skfolio](https://github.com/skfolio/skfolio) | Add pinned CombinatorialPurgedCV partitions and path assignment. Mosaic splits unique decision dates, maps every security on those dates together, and purges actual label intervals using trading-session positions. |
| [Alphalens Reloaded](https://github.com/stefan-jansen/alphalens-reloaded) | Adapt two small Apache-licensed diagnostics instead of importing the full plotting/analytics stack: rank persistence and top-quantile name turnover. |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) | Use the already pinned LedoitWolf covariance estimator for factor risk. Replaces the heuristic shrinkage intensity with a data-estimated intensity. |
| [cvxportfolio](https://github.com/cvxgrp/cvxportfolio) | Not copied. Mosaic already has a constrained CVXPY optimiser. Copying its GPL-3.0 implementation would introduce a separate redistribution decision without resolving the present data gaps. |
| [Qlib](https://github.com/microsoft/qlib) | Not integrated. A full platform replacement would add substantial migration work; it does not supply Mosaic's required verified delisting/estimate history. |
| [mlfinlab](https://github.com/hudson-and-thames/mlfinlab) | No code selected. Used inspectable maintained alternatives for the required functions. |
| [Community S&P history](https://github.com/shardul0701/SP500-Survivorship-bias-data-2004-2026) | Not redistributed or certified. No declared licence appeared in repository metadata. Membership history alone supplies neither terminal returns nor Nasdaq/Russell coverage. |

Exact reviewed revisions and installed package hashes are in
`third_party/upstream_review.json`. Copied-source notices and licence are in
`THIRD_PARTY_NOTICES.md` and `third_party/ALPHALENS_LICENSE`.

## What changes on the next research run

Install the pinned requirements, then use the existing manual research workflow.
`research.py` now records a fixed ridge CPCV diagnostic at the primary horizon,
ensemble persistence on walk-forward observations, and upstream SPA/RC results.
`report.py` renders CPCV coverage and the persistence/turnover table. CPCV uses
six date folds, two test folds per combination (15 fits and five paths), fixed
ridge hyperparameters and the existing five-session embargo. Label intervals
touching a test label interval are excluded. Data reaching the holdout boundary,
including its embargo, never enters these fits.

CPCV deliberately allows some later discovery folds to train earlier discovery
test folds. It is a stability stress test, not an expanding chronological backtest
or a substitute for the prospective record. It does not set ensemble weights,
approve deployment, or spend the holdout. It adds fitting time to manual research;
it is not run nightly and does not change existing trained weights on disk.

Alphalens lags count observed decision dates, not trading sessions. Quantile
returns are gross label excess returns. Name-membership turnover is not dollar
turnover and does not replace Mosaic's transaction-cost model. Missing bucket
observations remain unknown; the comparison does not skip across empty dates.

`risk_model.build()` now estimates factor covariance with genuine Ledoit-Wolf
shrinkage on the last 756 factor-return observations (complete rows only), rather
than the former EWMA/heuristic blend. Specific risk remains the existing EWMA
estimate. This changes risk estimation, not an approved bot constraint. Existing
stored risk models retain their previous covariance until rebuilt. The old
`ewma_cov()` helper remains available for explicit historical comparisons.

The statistical-test sample can shrink when strategies have different coverage.
The report records the common sample size, excluded periods, seed and bootstrap
parameters. Neither additional diagnostics nor a better-conditioned covariance
matrix establishes increased investment returns.

## Verification and rollout

All 65 tests pass in the local Python 3.13 environment. New tests exercise actual
upstream packages, end-to-end synthetic ridge paths, holdout-label contamination,
cross-security/date leakage, irregular session gaps, expected signal rank reversal
and turnover, missing buckets, genuine covariance shrinkage and strict JSON.
Compilation, dependency consistency and diff checks also pass.

No production dataset, fresh vendor data, model refit, or investment performance
evaluation was run. These changes are on the local upstream-research-integrations
branch; this integration batch has not been pushed or deployed. Existing validity
gates remain in place. Source/dependency changes invalidate a previously frozen
recipe, so rollout requires revalidation and a newly approved forward recipe.
