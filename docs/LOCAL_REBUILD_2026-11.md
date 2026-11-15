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
- No delisting or terminal payouts.
- CIKs are LSEG data resolved against SEC, not certified.
- Order size is a placeholder ($600); the T212 bot configuration is not on this machine.
- The holdout was reused before this rebuild, so every number here is exploratory.

## Recommendation

1. **Stop developing the cross-sectional ridge ensemble on this universe.** Log it as a failed
   hypothesis. Further feature additions would be more trials against an exhausted holdout.
2. **Close the terminal-payout gap** (delisting cash or stock consideration) before any further
   historical claim. Without it, SPA and PBO can't be computed on a survivorship-corrected universe.
3. **Prioritise the pre-registered insider event model and its forward record** (`insider_forward.py`).
   It is judged on prospective data, not on the reused holdout.
4. **Add the three missing controls** from `docs/MOSAIC_PROMPT_REVIEW.md`: an equal-weight composite
   baseline, a time-shuffled negative control, and a known-answer regression test.

## Reproduce

Logs are in `ledger_local/`:
- A: `research_large_baseline_yahoo.log`
- B: `research_large_bundle.log`
- C: `research_large_fs28.log`

Pre-import backup: `data_backup_2026-11-14_pre_lseg_bundle/`.

Converters: `lseg_bundle.py`, `lseg_estimates.py`. Feature definitions: `features.analyst_features`.
Pre-registration: `prereg/lseg_analyst_flow_v1.json`.
