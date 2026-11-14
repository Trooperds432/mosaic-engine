# Response to the external literature audit — 12 November 2026

An external "executive summary" audit said Mosaic lacks the safeguards recommended in the backtest-overfitting
literature, and proposed about ten tickets. Each claim was checked against the code. **Most are false.** The
safeguards exist, and they are doing their job: they reject a forecast that has no robust gross edge. This note
records the evidence, the few genuine gaps, and the changes made for those gaps.

No performance improvement is claimed. Results from the changes below need the next VPS research run.

## What the evidence says

From `docs/BACKTEST_2026-10-22.md` (20-session horizon, net excess vs SPY per period):

| Universe | Sample | Gross | Modelled cost | Net |
| --- | --- | ---: | ---: | ---: |
| large | walk-forward | −0.170% | 0.372% | −0.541% |
| large | reused holdout | +0.166% | 0.392% | −0.226% |
| broad | walk-forward | +0.006% | 0.450% | −0.444% |
| broad | reused holdout | +0.296% | 0.504% | −0.208% |

- The walk-forward cost sweep is still negative at **zero** cost in the 20 Oct run (large −0.28%, broad −0.09%).
- Hansen SPA p is 0.96 (large) and 0.64 (broad); deflated Sharpe ≈ 0.000; holdout IC HAC t is 1.17 / 1.78.
- The FX fee (0.15% each way; every Trading 212 trade converts from GBP) is the largest cost component.

More statistical machinery cannot create an edge that the data does not show. The audit's own diagnosis of
the symptoms (negative SPY-relative returns, UNVALIDATED, NO_SIGNAL) is right. Its explanation, missing
safeguards, is wrong: these outputs are the safeguards working.

## Claim-by-claim

| Audit claim | Verdict | Evidence |
| --- | --- | --- |
| No CPCV; walk-forward only | FALSE | `validation_extensions.py:36` `purged_combinations` (skfolio CombinatorialPurgedCV, interval purge + embargo); `:64` `cpcv_ridge`; run at the primary horizon in `research.py` (15 splits / 5 paths on 22 Oct). It is a discovery stress test, not a gate, by design |
| No deflated Sharpe / PBO | FALSE | `research.py` `psr`, `deflated_sr`, `min_track_record`, `pbo_cscv`; trial set = every model plus every signed single-feature strategy; DSR ≥ 0.90 is a gate |
| No multiple-testing correction | FALSE | Benjamini-Hochberg `research.py` `bh_fdr` (q = 0.10, `config.FDR_Q`) over every univariate test; White Reality Check and Hansen SPA via `arch` (`validation_extensions.py:12`, gated by `gates.joint_test_failures`). Holm/Bonferroni are absent on purpose: BH plus SPA/RC is the more appropriate pair |
| No significance gating | FALSE | Historical gate in `research.run`: holdout IC HAC t ≥ 2, ≥ 60% positive WF years, holdout net > 0, DSR ≥ 0.90, SPA p ≤ 0.10, monotone deciles. Forward gate `paper_trading.forward_gate`; combined by `deployment_policy.cumulative`; enforced in `signals.py` |
| Walk-forward does not purge/embargo | FALSE | `research.walk_forward`: training labels must end `EMBARGO` = 5 sessions before the test block (same for the holdout fit) |
| No transaction-cost model | FALSE | `research.name_costs`: FX fee + max(½ Corwin-Schultz, 3 bp) + 0.7·σ·√(order/ADV) at the bot's order size; self-financing ledger `holdings_simulation.simulate` charges every traded dollar; cost × {0, 0.5, 1, 2, 3} sweep |
| No point-in-time data | FALSE (survivorship PARTIAL) | Fundamentals as filed, t−1 availability (`features.fundamentals_asof`); first-reported EPS for SUE; `pit_estimates.py`, `pit_clinical.py`, `macro_vintages.py` (FRED first releases); Form 4 keyed on filing date. Survivorship is the real gap — see below |
| No insider composite | FALSE | 14 insider features (`features.py`, `insider_*`), including the Cohen-Malloy-Pomorski opportunistic/routine split (`classify_insiders`), roles, clusters, 10b5-1 share; separate pre-registered event model `insider_model.py` + `prereg/insider_event_v1.json` |
| No forecast uncertainty | MOSTLY FALSE | Walk-forward decile calibration with quantiles and expected shortfall, Mondrian split-conformal coverage, logit P(excess > cost) Brier/reliability, per-signal `prediction_interval_95`. Missing: stock-conditional quantile models |
| No regime models | PARTIAL | HMM P(high-vol) interaction ridge and Markov-switching ridge (`regimes.py`) are tested; neither improves walk-forward IC (`ms_improves: false`), so production stays unconditional |
| No provenance / schema | FALSE | `publication.py` jsonschema validation against `schema/signal_v2.schema.json`, atomic publication; `provenance.dataset_version`; every signal carries model, run, recipe, feature-set and dataset ids |
| Earnings surprise not orthogonalised to momentum | TRUE | Fixed — see change 1 |
| Paper trading shows NO_SIGNAL / no trades (implied bug) | FALSE | Intentional fail-closed: no LONG without a passing gate; `paper_trading.py` read a hard-coded VPS path, so a local run always showed 0 trades — fixed in change 4 |

Several citations in the audit could not be verified (for example "Müller et al. (2026)" and some 2025–2026
SSRN titles). The recommendations above do not depend on them.

## Changes made

1. **Momentum-orthogonal earnings features** (`features.py`). `sue_xmom` and `sue_consensus_xmom` are the
   per-date OLS residuals of rank(SUE) on rank(`mom_12_1`). They use the same date's cross-section only, so
   there is no new look-ahead. They are computed after symbol chunks are joined. Both enter the BH-FDR
   univariate tests and the single-feature trial set, so DSR/SPA penalise them. Feature set is now **fs-2.2**:
   the panel must be rebuilt, and source changes invalidate existing frozen recipes.
2. **Pre-registered low-turnover variant** (`prereg/cross_section_turnover_v1.json`, `research.buffered_ensemble`).
   It enters from the top 20% and holds while a name stays in the top 40%, at the 20-session horizon, using the
   same cost model and ledger. Its walk-forward net series joins `strategy_returns` as `model:ensemble_buffered`,
   so it counts as an extra trial. The prereg sha256 is stored in the results and in `ledger/research_runs.jsonl`.
   The gate still evaluates the unbuffered ensemble. The report shows gross / cost / net / turnover for both.
3. **PBO gate** (`gates.pbo_failures`, `config.GATE_MAX_PBO = 0.50`). PBO above 0.50, or not computed, now fails
   the primary-horizon gate. This only tightens the gate.
4. **Configurable trade log** (`paper_trading.py`). `MOSAIC_T212_TRADES` overrides `/opt/t212-bot/trades.jsonl`.
   A missing log reports `trade_log: NOT_FOUND` instead of a silent zero.

Tests: `tests/test_audit_response.py` (8 tests). Full suite: 114 tests pass, 1 skipped.

Conformal coverage remains a reported figure, not a gate. Decile monotonicity already gates calibration, and the
conformal implementation still awaits the independent review listed in `IMPLEMENTATION.md`.

## Not changed, and why

- **Thresholds**: no gate threshold was relaxed.
- **Holdout**: the 2-year holdout has been viewed by several runs. Any variant judged on it is snooped, which is
  why change 2 is pre-registered and counted as a trial. Only forward time produces a clean test.
- **Historical universe**: verified historical constituents and delisted returns are a data dependency
  (`historical_data.py`, `audit.historical_universe`). Code cannot supply them.
- **Quantile regression / regime switching in production**: not justified while the point forecast itself has
  no robust walk-forward edge.

## Next steps (VPS)

1. Rebuild panels (`features.py large|broad`) so fs-2.2 columns exist.
2. Run `research.py large` and `research.py broad`; regenerate the report.
3. Read the buffered-variant row and the new feature tests once. Do not iterate on parameters without
   registering `cross-section-turnover-v2`.
