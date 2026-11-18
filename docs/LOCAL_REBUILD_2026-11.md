# Local rebuild and survivorship-corrected re-evaluation (13–15 November 2026)

**Status: the cross-sectional model's hypothesis is not supported.** Correcting for survivorship removed
the only signs of edge, which had appeared in the reused holdout at longer horizons. None of the nine
new or newly populated features survives the false-discovery correction, and the one feature with
|t| ≥ 3 has the opposite sign to its pre-registration. All results are exploratory, because the holdout
had already been reused before this rebuild.

## What changed between runs

| Run | Universe | Data added | Feature set |
|---|---|---|---|
| A. Yahoo baseline | Current Yahoo/SEC listings (survivor-biased) | Prices, SEC fundamentals, Form 4, 8-K, 13F, FINRA SI, FRED market series, EIA, CoT, trials/FDA, Wikipedia | fs-2.7 |
| B. Survivorship bundle | LSEG S&P 500 month-end members, Dec 2015 – Oct 2026, 721 securities including delisted names | A + `lseg_bundle.py` import (CIKs resolved against SEC) | fs-2.7 |
| C. Bundle + new data | As B | B + FRED first releases (FRED_API_KEY) + LSEG calc-dated EPS, price-target and recommendation history + sector-ETF flows (`prereg/lseg_analyst_flow_v1.json`) | fs-2.8 |

Run C sample: 2016-01-05 to 2026-10-29, 545 decision dates, 413 securities, about 178 names per date.
The holdout starts on 2024-11-12.

## Ensemble results (walk-forward IC / holdout IC; holdout t in brackets)

| Horizon | A. Yahoo | B. Bundle | C. Bundle + new data |
|---|---|---|---|
| 1 | −0.003 / 0.006 (0.58) | −0.005 / 0.013 (1.02) | −0.009 / 0.011 (0.86) |
| 5 | 0.008 / 0.035 | −0.002 / 0.005 (0.49) | −0.004 / 0.007 (0.83) |
| 20 | 0.009 / 0.028 (1.18) | 0.008 / 0.003 (0.19) | 0.011 / 0.010 (0.60) |
| 60 | 0.000 / 0.071 | 0.015 / 0.010 (0.32) | 0.015 / 0.009 (0.28) |
| 120 | −0.004 / 0.095 | 0.005 / −0.007 (−0.24) | 0.007 / −0.005 (−0.16) |

20-session gate (primary horizon):

| | A | B | C |
|---|---|---|---|
| Deflated Sharpe | 0.0 | 0.236 | 0.012 |
| Hansen SPA p | 0.98 | not computed | not computed |
| PBO | — | not computed | not computed |
| Holdout long-only net excess > 0 | no | no | no |

The A-run holdout ICs at 60 and 120 sessions (0.071 and 0.095) disappear once delisted names are
included. They were survivorship artefacts, so the planned 120-session variant is dropped.

SPA and PBO can't be computed in B and C. The walk-forward backtest selects securities that later
delisted, and the bundle has no terminal payout for them. The engine refuses to impute a return, so
the strategy ledger is incomplete. That is the correct, fail-closed behaviour.

## New and newly populated features (run C, univariate cross-sectional IC, HAC t, raw variant)

| Feature | h=1 | h=5 | h=20 | h=60 | h=120 | Pre-registered sign |
|---|---|---|---|---|---|---|
| `eps_revision_pit_31d` | 0.41 | 1.91 | 0.78 | 0.04 | 0.77 | + |
| `est_dispersion_pit` | 0.24 | −0.37 | −0.30 | 0.19 | 0.28 | − |
| `pt_rev_1m_pit` | −0.06 | −0.56 | −0.99 | −0.16 | 0.90 | + |
| `rec_rev_1m_pit` | −1.04 | −2.27 | **−3.20** | −1.34 | 0.71 | + |
| `sector_etf_flow_20d` | −1.20 | −1.23 | −2.38 | −0.95 | −0.92 | 0 (reported only) |
| `growth_surprise_tilt` | −0.81 | −0.73 | −2.00 | 0.03 | 0.38 | 0 |
| `inflation_surprise_tilt` | −1.74 | −0.59 | 0.12 | 0.40 | −0.41 | 0 |

`semi_cycle` and `semi_orders` cover about 7% of rows (semiconductor names only) and produced no
usable tests.

**Pre-registered verdict:** none of the new features passes. The bar was walk-forward t ≥ 3 and
BH-FDR at q = 0.10, among 1,385 univariate tests. No test in the whole run survives BH-FDR.

`rec_rev_1m_pit` at 20 sessions has t = −3.20, the opposite of its pre-registered sign: upgrades were
followed by lower returns. A single sign-reversed result among 1,385 tests that fails FDR is not a
finding. Treating it as a signal would be a new hypothesis, and it would need its own registration and
forward test.

## Run D: terminal (delisting) outcomes added (15 November)

All 118 ended bundle securities were classified from SEC submissions (8-K items within ±30 days of the
last trade, and Form 25-NSE). The list is in `ledger_local/delisting_classification_20261115.csv`.

| Class | Count | Terminal payout |
|---|---|---|
| Merger / acquisition (8-K 2.01 or 5.01, or Form 25) | 108 | **Proxy:** last adjusted close, settled one business day after the last trade. This is not the contractual consideration. |
| Chapter 11 (8-K 1.03): FTR, DO, CHK, VAL, MNK, ENDP, SIVB, BBBY | 8 | 0 (equity cancelled; any warrants valued at 0) |
| FDIC receivership: SBNY, FRC | 2 | 0, settled on the seizure date |

**Estimate features empty in runs D and E.** Between runs C and D, `pit_estimates` started classifying
LSEG CodeBook calc-date consensus as `inferred_calc_date` (sensitivity only). `features.py` excludes that
class, so `eps_revision_pit_31d` and `est_dispersion_pit` are missing on every row of the D/E panel. Their
run C results above used the calc-date data. Price-target and recommendation revisions are unaffected.

`terminal_events_complete` stays false, and the method is written into the bundle's `coverage.json`.
Only one ended security remains unresolved (LVLT, which has no LSEG prices).

| Horizon | Walk-forward IC | Holdout IC (t) |
|---|---|---|
| 1 | −0.008 | 0.013 (1.00) |
| 5 | −0.006 | 0.006 (0.74) |
| 20 | 0.012 | 0.009 (0.56) |
| 60 | 0.014 | 0.009 (0.29) |
| 120 | 0.005 | −0.008 (−0.24) |

20-session results: 103 strategies, 111 walk-forward periods.

| Metric | Result |
|---|---|
| Deflated Sharpe | 0.013 (gate 0.9) |
| PBO (CSCV) | **0.17**, now computable; passes the 0.5 limit |
| Hansen SPA / White RC | Not computed in run D; computed in run E (see below): SPA p 0.64, fails |
| Strategy ledger | Complete; no missing selected outcomes |

**Pre-registered controls** (`ridge_controls.py`, 20 sessions, walk-forward):

| Strategy | Gross / period | Net / period | Net HAC t |
|---|---|---|---|
| Ridge | +0.13% | −0.25% | −0.88 |
| Equal-weight signed composite | +0.13% | −0.21% | −0.65 |
| Time-shuffled training (negative control) | +0.11% | −0.16% | −0.57 |

Ridge minus equal-weight, paired: −0.03% per period, 95% CI [−0.64%, +0.54%]. The model is
indistinguishable from both the equal-weight baseline and the shuffled null.

**SPA not computed in run D:** `multiple_comparison` uses only periods where every strategy has a return.
Some feature strategies are empty in most periods, so fewer than 20 common periods remained and the test
failed closed.

## Run E: pre-registered SPA coverage rule (15 November)

Rule: `prereg/spa_coverage_v1.json`, registered at 20:50 UTC before any SPA p-value on real data. A
strategy enters SPA / Reality Check only if it has returns in ≥ 90% of the periods where the ensemble has
a return. The ensemble must be present. Then the existing complete-common-period rule applies (no zero
imputation, ≥ 20 periods). The gate threshold (SPA p ≤ 0.10) is unchanged. DSR still counts all 103
strategies.

One amendment, recorded in the file before any real result: as first written, coverage was measured over
all periods. Model strategies start only after the minimum training window, so the ensemble itself covered
46% of periods on the synthetic test and the rule would always have failed closed. Coverage is now measured
on the ensemble's own walk-forward window.

| Item | Result |
|---|---|
| Evaluation window | 48 periods (the ensemble's walk-forward periods at h=20) |
| Strategies admitted | 96 of 103 |
| Excluded (coverage 17% each) | `macd_hist`, `r2_mkt_252`, `vol_ratio_20_60`, `credit_tilt`, `opt_mkt_term_tilt`, `opt_mkt_skew_tilt`, `growth_surprise_tilt` |
| White Reality Check p | 0.68 |
| **Hansen SPA p** | **0.64: fails the 0.10 gate** |
| Best strategy in the set | `feat:energy_crack_chg`, +0.38% net per period |

Excluding seven strategies makes the test less conservative, and it still fails. Over 96 strategies there
is no evidence that any one has positive expected net excess return over the benchmark. Everything else
matches run D (DSR 0.013, PBO 0.17, identical ICs).

20-session gate failures for the ensemble: order size is a placeholder; four source attestations are
missing; holdout IC t 0.56 < 2.0; holdout net excess ≤ 0; DSR 0.013 < 0.9; SPA p 0.64 > 0.10. PBO passes.
Status: `UNVALIDATED`.

## Insider event forward record: blocked

`insider_study.py freeze` stops with `missing forward inputs: verified quarterly estimate vintages`.
`pit_estimates.primary_quarterly` admits only `vendor_archive` quarterly vintages with documented
availability and correction-policy evidence. LSEG CodeBook calc-date exports are classified
`inferred_calc_date` (sensitivity only). Daily scoring also needs fresh prices for bundle security IDs,
which Yahoo can't supply while a bundle is active.

Unblocking it needs one of two things:
- a documented quarterly estimate archive;
- a new pre-registration that runs without that control.

It also needs a scheduled price refresh for bundle IDs.

## Data quality

- **Bundle prices vs Yahoo:** total-return CAGR agrees to 0.01 percentage points (AAPL, MSFT, JPM, XOM).
  Mean absolute daily difference is under 1 bp.
- **CIK resolution:** LSEG returned several CIKs for 81 RICs and none for 9. These were resolved through
  SEC's ticker map, SEC's issuer list, or SEC submissions (the filer of the 10-K/10-Q and Form 4s).
  First Republic and Signature Bank filed with bank regulators, so their CIK is an identifier only.
  The log is at `scratchpad/sp500_src2/cik_resolution_log.csv`.
- **LSEG restates estimate, target and EPS history for later splits.** Revisions are unaffected; the
  levels are not as-reported values.
- **Coverage check (`historical_data.coverage_check`):** fails on missing attestations, missing
  Nasdaq-100 and Russell 2000 membership, 117 ended securities without terminal outcomes, and short
  price gaps. These are known, disclosed limitations, not defects.
- **Not available from LSEG on this entitlement:** option chains and implied-vol history (NotEntitled),
  Argus LNG freight, MarketPsych. No DRAM spot series was found.

## Limitations

- S&P 500 only, from December 2015; membership at month-end resolution.
- Merger payouts are a proxy (last adjusted close), not the contractual consideration. LVLT has no outcome.
- CIKs are LSEG data resolved against SEC, not certified.
- Order size is a placeholder ($600); the T212 bot configuration is not on this machine.
- The holdout was reused before this rebuild, so every number here is exploratory.

## Recommendation

1. **Stop developing the cross-sectional ridge ensemble on this universe.** Log it as a failed
   hypothesis. Further feature additions would be more trials against an exhausted holdout.
2. **Terminal payouts: done (run D), with proxy merger values.** PBO (0.17) and SPA (p 0.64, run E) are
   now computed on the survivorship-corrected universe. Both are consistent with no skill.
3. **Prioritise the pre-registered insider event model and its forward record** (`insider_forward.py`).
   It is judged on prospective data, not on the reused holdout. It is currently blocked (see above).
4. **Controls: done.** Equal-weight baseline, time-shuffled negative control and known-answer test are in
   `ridge_controls.py`. Ridge is indistinguishable from both.

## Reproduce

Logs are in `ledger_local/`:
- A: `research_large_baseline_yahoo.log`
- B: `research_large_bundle.log`
- C: `research_large_fs28.log`
- D: `research_large_terminal.log`
- E: `research_large_spa.log`

Pre-import backup: `data_backup_2026-11-14_pre_lseg_bundle/`.

Converters: `lseg_bundle.py`, `lseg_estimates.py`. Feature definitions: `features.analyst_features`.
Pre-registrations: `prereg/lseg_analyst_flow_v1.json`, `prereg/spa_coverage_v1.json`.
