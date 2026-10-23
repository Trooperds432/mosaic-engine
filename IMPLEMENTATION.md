# Specification implementation — integrity and validation tranche

For the subsequent provenance, profile-audit, risk-freshness and holdings-ledger
changes, see [the 23 October engineering fixes](docs/REVIEW_FIXES_2026-10-23.md).

Baseline: 7548fd77e5c0c2297d67fd9b6da2c48b95e73c27. Scope: local implementation;
no publication, deployment, broker requests, model fitting or production data migration performed.

GitHub integration: reconciled with 25f6cf3. Its updated insider artifacts, research ledger,
non-PIT feature exclusion and imported bot limits are retained. The stricter publication,
recipe-specific readiness and external-policy optimizer supersede overlapping implementations.
Supplied bot caps cannot be relaxed by the research policy. Existing saved results were
produced by their recorded older code and have not been regenerated for this implementation.

## Implemented

| Specification | Behavior | Evidence |
|---|---|---|
| 2.2 / 4B | Archive timestamped registry snapshots; phase/completion revisions cannot backfill earlier decisions. No archive coverage is missing, not zero. FDA issuer/source gaps also remain missing. | ClinicalTests |
| 2.2 | Audit fingerprint covers every price file, insider history, alternative/collected inputs, identity maps and feature/config source. | ForwardIdentityTests |
| 3.3 | Matched-control regression receives the actual control company's momentum, reversal, ADV, beta and sector. Only complete event/control pairs enter; sparse samples produce NOT_COMPUTED. | InsiderControlsTests |
| 9 / 12.2 | Live-review recommendation requires matching positive forward IC with HAC t, sufficient scored dates, positive cost-adjusted shadow cohorts, and executed evidence per model at the matching horizon. | PromotionTests |
| 9 / 12.2 | Histories retain model recipe IDs. Missing/old recipe observations cannot unlock a new recipe. Forecast artifact identity is checked before recording. | ForwardIdentityTests, real Parquet round trip |
| 9 / 12.2 | Executed promotion evidence excludes legacy orders without recipe attribution and orders from other recipes. Trading 212 BUY logs add research_run_id only; order sizing/execution settings are unchanged. | Source inspection; broker fill reconciliation not executed |
| 10.3 | External policy supplies gross, sector, turnover, beta, participation and risk-aversion limits. Missing policy yields NOT_COMPUTED. Existing holdings/forced exits count toward turnover and capacity. | OptimizerTests, real CLARABEL solve and infeasible solve |
| 12.3 / 14 | Shared nightly/intraday strict JSON and Schema validation, exclusive history files and atomic latest publication. NaN/Infinity rejected. Intraday refresh rechecks audit; failed audit demotes retained LONGs too. | PublicationTests, IntradayContractTests |

The research engine still cannot authorise live execution. The recommendation is for human review only.
No performance improvement is claimed from these changes.

## Compatibility and adoption

1. Install requirements.txt in an isolated Python 3.13 environment. Run `python -m unittest discover -s tests -v`.
2. Feature semantics are now fs-2.1. Previously trained recipes containing bio_ph3_trials are blocked from LONG
   output until revalidated under the corrected semantics. Existing generated reports have not been rewritten.
3. Insider analysis version is 1.1.0. The frozen trigger/threshold file is unchanged; older stored model versions
   are blocked until the corrected control analysis is run. Re-running does not create a new untouched holdout.
4. The next successful trial collection starts the archive. Current trial records cannot repair earlier history.
   There is no retroactive reconstruction of trial revisions or issuer coverage.
5. Legacy forward histories remain on disk, but unattributed rows no longer qualify for promotion. A frozen recipe's
   subsequent weight refits keep its run ID; changing the recipe requires a new validation run ID.
6. Copy portfolio_constraints.example.json to portfolio_constraints.json ONLY after supplying approved external
   policy values and its source. Nulls deliberately disable proposals. Constraints support long-only, unleveraged
   accounts. No arbitrary new live limits are installed. Costs are annualised using the supplied planning horizon.
7. The expanded fingerprint invalidates older audit artifacts. Regenerate an audit against current inputs before
   signals. Intraday collection invalidates the preceding audit, so new and retained LONGs are blocked until a
   fresh matching audit exists. The default event watcher does not automatically run that full audit.
8. The Trading 212 instrumentation field is required for new executed-recipe evidence. Historical fills remain in
   original logs but cannot be credited to a recipe without a recorded research_run_id. No service was restarted.

## Still missing / not resolved

| Area | Remaining requirement / dependency |
|---|---|
| Historical universe | Historical S&P/Nasdaq/Russell constituents, delisted prices/returns, security-level identifier history and corporate-action reconciliation. Requires suitable historical datasets and verified mappings. |
| Tech/semis | Foundry/wafer/memory/bookings/lead-time data, hyperscaler spending and supplier/customer analysis. Public-source discovery and timestamped archives remain to be implemented. |
| Biotech | Historical protocol/outcome vintages, efficacy/safety endpoints, clinical success base rates and failure-return cohorts, reimbursement/patents. No clinical probabilities invented. |
| Energy/agriculture | NOAA/weather/degree days, USDA/USGS balances, LNG/outages, contract-specific futures curves and roll handling, geospatial inputs. |
| Financials | Broader regulatory capital, loan/deposit mix, funding, payments/AUM and business-model controls. |
| Options/estimates | Existing collectors need sufficient genuinely historical or forward-collected data; the three-year snapshot exclusion remains. |
| Research protocol | Complete per-hypothesis mechanism/provenance registry, observation-level availability records, historical-vintage revisions across all sources, full A–J per-feature evidence tables. |
| Insider validation | Value, quality, earnings-revision controls; merger/guidance controls; amendment reconciliation. Correct control covariates do not establish full incremental alpha. |
| Macro | Twice-monthly ALFRED observations remain earliest *sampled* vintages, not guaranteed original releases; EIA/FRED revision limitations remain. |
| Statistical validation | Independent review of conformal/SPA/PBO implementations, feature-selection timing and effective sample sizes; no new untouched historical test is created. |
| Risk/execution | Live fill/FX benchmarking, full latency/decay/capacity modelling, borrow permissions and general multi-asset optimisation. |
| Catalysts | Missing calendars, priced-in expectations and conditional failure/downside distributions. |
| Operations | Full Linux/VPS pipeline, collectors, broker reconciliation, service permissions, restart/recovery and production-scale memory/performance have not been tested locally. |

The specification is not fully implemented. This tranche addresses concrete integrity gaps first; broader sector
and multi-asset work is still outstanding. Positive unseen performance and enough forward observations cannot be
created by adding code.
