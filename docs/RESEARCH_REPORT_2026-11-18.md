# Mosaic Engine: research report (18 November 2026)

This report covers everything established from the external audit (12 November) to the pre-registered
SPA run (15 November), plus the earlier October results that remain relevant. Every number below comes
from a file in this repository or from `ledger_local/`; the source is named next to each table.

**Standard applied.** Results are judged against the QR.md checklist: point-in-time data, survivorship
correction, costs at the actual order size, counted trials, pre-registration before results, and
fail-closed gates. "Working" below means *survives those tests*, not *looked good once*.

---

## 1. Executive summary

1. **The cross-sectional ridge model has no detectable skill.** On the survivorship-corrected S&P 500
   universe (runs B–E) the holdout IC is about zero at every horizon, the deflated Sharpe ratio is 0.013
   (gate 0.90), Hansen SPA p is 0.64 (gate 0.10), and the model is statistically indistinguishable from
   both an equal-weight composite and a time-shuffled null. It should be logged as a failed hypothesis.
2. **The only promising signs of edge in the earlier runs were survivorship artefacts.** On the
   survivor-biased Yahoo universe (run A), holdout IC at 60 and 120 sessions was 0.071 and 0.095. With
   delisted names included, both fell to about 0.01 and −0.01.
3. **No new data source added a validated signal.** LSEG estimate, price-target, recommendation and
   ETF-flow features, plus FRED first-release macro surprises, were pre-registered and tested. None of
   1,385 univariate tests survives BH-FDR. The single |t| ≥ 3 result has the wrong sign.
4. **The strongest raw effect in the project is insider buying, gross of costs.** CEO/CFO open-market
   purchases show a 1-session abnormal return of +0.44% (t 6.65) in discovery and +0.57% (t 2.33) in
   the viewed holdout. The effect is consistent with the literature. It does not survive costs in the
   v1 model (net −0.24% per event in discovery), and the redesigned v2 study has not yet produced a
   historical result because its inputs were blocked.
5. **Costs are the binding constraint, and the cost estimate itself needs checking.** The modelled
   one-way cost averages 0.42% (round trip about 0.8%) at a $600 order. Most of it is the Corwin-Schultz
   spread estimate (0.26% one way), not the FX fee (0.15%). A 24 bp median half-spread is very likely
   far above actual quoted spreads in S&P 500 names. If real costs are near FX plus a few bp, short-
   horizon insider buys would clear costs gross in both samples. The ridge model would still not,
   because it does not beat its shuffled null even gross. This is unverified (§7).
6. **The validation framework works.** It caught the survivorship artefact, refused to impute missing
   delisting returns, refused to compute SPA on an inadequate sample until a rule was pre-registered, and
   reclassified calc-dated estimates as sensitivity-only. Every one of these fail-closed behaviours
   prevented a false positive.

**Status of every tradable strategy: `UNVALIDATED`. Nothing has trading authority.**

---

## 2. What appears to be working

Ranked by strength of evidence. "Working" is split into *signals* (predict returns) and *tools*
(do their job correctly but do not by themselves make money).

### 2.1 Signals

| Rank | Candidate | Evidence | Why it is not yet validated |
|---|---|---|---|
| 1 | **Insider purchases (CEO/CFO, clusters), 1–5 sessions** | Gross abnormal returns positive in discovery *and* viewed holdout (table §6.1); matches Seyhun 1986, Lakonishok-Lee 2001, Cohen-Malloy-Pomorski 2012; event-minus-matched-control +0.27% (t 2.43); incremental regression t 2.16 after momentum, reversal, ADV, beta, sector | Net of costs negative in v1 (discovery −0.24%, t −1.28; holdout −0.09%, t −0.31). Evaluated on the survivor-biased universe. Holdout already viewed. v2 not yet run |
| 2 | **Value/quality/momentum composite (`vqm-1`), concentrated** | Viewed holdout net +1.02% per 20 sessions, t 1.69, Sharpe 1.60 (24 periods) | Walk-forward net −0.06% (t −0.24, 48 periods). Sign flips between samples; the holdout was viewed; t < 2. Treat as noise until a forward record says otherwise |
| 3 | **Concentrated model book (~13 names), walk-forward** | Walk-forward net +0.41% per period, t 0.95 (48 periods); turnover-buffered variant +0.37%, t 0.85, at half the cost | Holdout net −0.08%. Not significant in either sample |

None of these clears the pre-registered bar (t ≥ 3 for new signals, holdout net > 0, DSR ≥ 0.90,
SPA p ≤ 0.10). The insider effect is the only one where the gross effect is consistent across samples
and independently documented, so it is the only line worth further investment.

### 2.2 Tools and models that do their job

| Component | Result | Source |
|---|---|---|
| **GARCH-t VaR / ES** | Best of four models for both the portfolio and single names (tick-loss ranking). Historical VaR fails Christoffersen independence on SPY (breaches cluster: p 8.8e-8) | `reports/var_backtest.json` |
| **Blume-adjusted beta** | Beats 252-day rolling beta: forecast MSE −4.0%, t 3.22, better on 64% of dates. Kalman beta is worse (MSE +28%, t −9.07) | `reports/kalman_beta_large.json` |
| **3-state HMM regimes** | Higher out-of-sample log-likelihood than 2 states in 16 of 19 years (mean gain 18.3). Describes volatility well; **does not** improve return forecasts (`ms_improves: false`) | `reports/regime_research_large.json` |
| **Macro release nowcast (ARIMA)** | Unemployment-rate changes: RMSE 0.165 vs 0.213 for a random walk, excluding 2020–21 | `reports/macro_surprise.json` |
| **Crude-products cointegration** | Engle-Granger p 0.002 (gasoline), 0.010 (heating oil); crack-spread error-correction half-life 2.2 months | `reports/commodity_research.json` (not tested as a trade) |
| **Validation stack** | DSR/PSR/MinTRL, CSCV PBO, Hansen SPA and White RC, BH-FDR, CPCV, purge and embargo, per-name costs, self-financing ledger, panel provenance hashing, pre-registration | §5 |
| **Data rebuild** | Survivorship bundle of 721 S&P 500 securities (2015–2026) with delisting outcomes; SEC acceptance timestamps for 725,905 of 725,909 Form 4 filings | §4 |

---

## 3. What is not working (falsified or unsupported)

| Hypothesis | Result | Status |
|---|---|---|
| Ridge ensemble ranks large caps (20 sessions) | Holdout IC 0.009 (t 0.56); DSR 0.013; SPA p 0.64; equal to equal-weight and shuffled null | **Failed** |
| Longer horizon (120 sessions) fixes costs (`large_ridge_120_v1`) | Run A holdout IC 0.095 was survivorship bias; corrected −0.008 (t −0.24) | **Dropped** |
| Market-neutral long/short (`ls_rank_5`) | Walk-forward Sharpe −1.80, CAGR −11.8%; holdout Sharpe −0.75. Gross +0.017% vs cost 0.254% per period | **Failed** (costs about 15× gross) |
| LSEG analyst revisions (`pt_rev_1m_pit`, `rec_rev_1m_pit`) | Max \|t\| 0.99 (price target); recommendations t −3.20, opposite to the registered sign | **Not supported** |
| Estimate dispersion | \|t\| ≤ 0.37 | **Not supported** |
| EPS revision 31-day | Peak t 1.91 (5 sessions) | **Not supported** |
| Sector ETF flows (sign 0, exploratory) | −2.38 at 20 sessions; reported only | **No prior; not a finding** |
| FRED growth / inflation surprise tilts | \|t\| ≤ 2.0 | **Not supported** |
| EIA crude inventory surprise | Discovery t −1.72 (expected sign), holdout t +1.19 (wrong sign) | **Not supported** |
| Commodity seasonality (e.g. coffee) | Wald p 0.87 | **Not supported** |
| Regime-conditional ridge (HMM, Markov switching) | No walk-forward IC improvement | **Not supported** |
| Random-portfolio baseline (h=5, 21 Oct, Yahoo universe) | 97.8% of matched random portfolios beat the model walk-forward | **Model worse than random** |
| Discretionary overlay | 0 vetoes scored; no evidence either way | **Untested** |
| Paper trading | 4 forecast dates, 0 trades (fail-closed gates) | **No record** |

---

## 4. Data: what was rebuilt and how

The VPS was lost, so all data was rebuilt locally. Source priority: LSEG CodeBook > Yahoo >
Databento (paid; not used) > SEC.

### 4.1 Sources in use

| Data | Source | Notes |
|---|---|---|
| Prices (survivorship) | LSEG `TR.PriceClose` (split-adjusted) + `TR.TotalReturn` (daily %) | Bundle `data/raw/historical/8433bd236228ba351c9a` |
| Index membership | LSEG S&P 500 month-end members, Dec 2015 – Oct 2026 | Monthly resolution only |
| Delisting outcomes | SEC submissions (8-K items 1.03 / 2.01 / 5.01 / 3.01, Form 25-NSE) | 118 classified, §4.3 |
| Fundamentals | SEC company facts (as filed, t−1 availability) | |
| Insider trades | SEC Form 4 quarterly + recent XML; acceptance timestamps from SEC submissions | 725,905 / 725,909 matched |
| 8-K, 13F, FINRA short interest | SEC / FINRA | |
| Macro | FRED first releases (ALFRED vintages); key in Windows env var only | |
| Estimates | LSEG `TR.EPSMean` FY1 monthly, calc-dated, 111,046 rows, 713 securities | **Sensitivity only** (§4.4) |
| Price targets, recommendations | LSEG `TR.PriceTargetMean`, `TR.RecMean` monthly, 2012+ | |
| Sector ETF flows | LSEG shares outstanding × NAV for sector SPDRs | XLC from 2018, XLRE from 2015 |
| Energy, CoT, trials/FDA, Wikipedia | EIA, CFTC, ClinicalTrials.gov/FDA, Wikimedia | |

**Not available on the current LSEG entitlement:** option chains and implied-vol history
(NotEntitled), Argus LNG freight, MarketPsych. No DRAM spot series was found. Databento would cover
options but costs money and has not been authorised.

### 4.2 Survivorship bundle

- 722 securities in the S&P 500 membership history; 721 with prices (LVLT has none).
- CIKs: LSEG returned several CIKs for 81 RICs and none for 9. These were resolved against SEC's
  ticker map, SEC's issuer list, then SEC submissions (who filed the 10-K/10-Q and Form 4s). 17 names
  were verified by hand (e.g. ADT, AET, BBBY, NLSN, RHT, TWX, WFM, FRC, SBNY). First Republic and
  Signature Bank filed with bank regulators, so their CIK is an identifier only.
- Bundle prices vs Yahoo: total-return CAGR agrees within 0.01 pp (AAPL, MSFT, JPM, XOM); mean absolute
  daily difference under 1 bp.
- All four source attestations (`constituents_complete`, `delisted_prices_complete`,
  `terminal_events_complete`, `identifiers_verified`) are **false**. These are honest: LSEG does not
  certify them, so the coverage check fails and every gate result carries those failures.

### 4.3 Delisting (terminal) outcomes

| Class | Count | Payout |
|---|---|---|
| Merger / acquisition | 108 | **Proxy:** last adjusted close, settled the next business day. Not the contractual consideration |
| Chapter 11 (FTR, DO, CHK, VAL, MNK, ENDP, SIVB, BBBY) | 8 | 0 |
| FDIC receivership (SBNY 2023-03-13, FRC 2023-05-01) | 2 | 0 on the seizure date |

Before this, the engine refused to impute returns for selected names that later delisted, so PBO and
SPA could not be computed. After it, the strategy ledger is complete.

### 4.4 Estimate provenance (important)

LSEG CodeBook exports give a *calc date* for each consensus value, but not a documented
first-availability time or correction policy. `pit_estimates.py` now classifies them as
`inferred_calc_date` (sensitivity only). Consequences:

- `eps_revision_pit_31d` and `est_dispersion_pit` are **empty** in runs D and E. Their run C results
  used the calc-date data.
- The insider v2 primary study needs documented quarterly vendor vintages
  (`availability_evidence`, `correction_policy_evidence`). Until one exists, its forward record is
  blocked.
- LSEG also restates estimate, target and EPS history for later splits. Revisions are unaffected;
  levels are not as-reported.

---

## 5. Validation framework

| Control | Implementation | Gate |
|---|---|---|
| Walk-forward with purge and embargo | `research.walk_forward`, 5-session embargo, yearly refits | |
| CPCV | `validation_extensions.purged_combinations` (skfolio), 15 splits / 5 paths | Stress test only |
| Holdout | From 2024-11-12; **already reused**, so all results are exploratory | Holdout IC HAC t ≥ 2; net excess > 0 |
| Deflated Sharpe | Every model and signed single-feature strategy counted (103 at h=20) | DSR ≥ 0.90 |
| PBO | CSCV, S = 10 blocks, periods with ≥ 80% of strategies live | PBO ≤ 0.50 |
| Hansen SPA / White RC | `arch`, stationary bootstrap, common periods, coverage rule `spa_coverage_v1` | SPA p ≤ 0.10 |
| Multiple testing | BH-FDR q = 0.10 over all univariate tests; t ≥ 3 for new features | |
| Costs | FX 0.15% each way + max(½ Corwin-Schultz spread, 3 bp) + 0.7·σ·√(order/ADV) at $600 | Net, not gross |
| Ledger | Self-financing, weight drift, terminal liquidation; no imputation of missing outcomes | Ledger complete |
| Controls | Equal-weight signed composite, time-shuffled training (seed 7), known-answer test | Reported |
| Provenance | `dataset_version` fingerprint; panel identity; aborts if inputs change mid-run | Must pass |
| Pre-registration | `prereg/*.json` written before results; amendments timestamped | |

---

## 6. Detailed results

### 6.1 Insider purchases (event study, `reports/insider_event_study.json`)

Abnormal return (%) and t-stat, keyed on the filing date. Holdout from 2024-10-16. Survivor-biased
Yahoo universe; gross of costs.

| Event | Sample | 1 day | 5 days | 20 days | 60 days |
|---|---|---|---|---|---|
| CEO/CFO buy | Discovery (n 1,449) | **+0.44 (6.65)** | +0.72 (4.45) | +0.62 (1.64) | +2.76 (3.42) |
| CEO/CFO buy | Holdout (n 201) | **+0.57 (2.33)** | +0.98 (2.55) | +0.21 (0.29) | −0.78 (−0.44) |
| Cluster buy | Discovery (n 2,013) | +0.28 (3.72) | +0.45 (2.43) | +0.45 (1.10) | +2.47 (3.95) |
| Cluster buy | Holdout (n 217) | +0.35 (2.30) | +0.94 (2.22) | +0.59 (0.80) | −1.64 (−0.84) |
| Any buy | Discovery (n 6,377) | +0.20 (7.11) | +0.29 (3.46) | +0.31 (1.52) | +1.41 (3.80) |
| Any buy | Holdout (n 732) | +0.12 (1.23) | +0.23 (0.99) | +0.50 (1.11) | +0.49 (0.36) |
| Discretionary sell | Discovery (n 24,872) | −0.04 (−3.78) | −0.06 (−1.81) | −0.04 (−0.53) | +0.40 (2.54) |
| Discretionary sell | Holdout (n 2,309) | −0.07 (−1.26) | −0.04 (−0.25) | −0.09 (−0.21) | −0.40 (−0.49) |

Reading: the short-horizon purchase effect is present in both samples for CEO/CFO and cluster buys.
The 60-day effect reverses in the holdout. Sells carry little information, as the literature predicts.

**v1 tradable model (`prereg/insider_event_v1.json`, 22 Oct run, `reports/insider_model_results.json`):**

| Metric | Value |
|---|---|
| Events | 6,054 (discovery 5,055, holdout 999) |
| Discovery gross | +0.22% per event (t 1.22) |
| Discovery net | −0.24% (t −1.28), hit rate 46% |
| Holdout net | −0.09% (t −0.31) |
| Event minus matched control | +0.27% (t 2.43) passes |
| Incremental regression | +0.24% (t 2.16) passes; earnings-revision control not computed |
| CEO/CFO only, 1 session | Discovery net −0.16% (t −2.32); holdout net +0.13% (t 1.46) |
| Gate | `UNVALIDATED`: discovery net t < 3, DSR < 0.90, holdout net ≤ 0 |

The 0.46% gap between gross and net per event is the v1 fixed round-trip cost assumption.

**v2 design (`prereg/insider_event_v2.json`):** 5-session horizon, 20-session cooldown, $20m minimum
ADV, $5 minimum price, 20 open positions, ridge on event features with CPCV, exact SEC acceptance
timestamps (entry at the first regular open on a later New York date), beta-adjusted returns, cost
sensitivities at 2× order size and 2× execution cost. Three trials: `ridge_positive_net`,
`ceo_cfo_or_cluster`, `any_purchase`. Forward review: ≥ 90 days and ≥ 100 settled events, positive
after costs against both baselines, plus user review.

**v2 status:** every historical attempt (`data/research/insider-v2-*`, `ledger_local/insider_v2_*`)
stopped on missing inputs, so v2 has **no statistics yet**. The last blocker list:
verified quarterly estimate vintages missing; 4 of 725,909 accessions without exact timestamps; the
four unattested bundle checks. `insider_study.py freeze` refuses to start the forward record without
estimate vintages.

### 6.2 Cross-sectional ridge model: runs A–E

| Run | Universe | Added | Feature set |
|---|---|---|---|
| A | Yahoo/SEC current listings (survivor-biased) | Base data | fs-2.7 |
| B | LSEG S&P 500 bundle, 721 securities | Survivorship correction | fs-2.7 |
| C | As B | FRED first releases, LSEG estimates/targets/recs, ETF flows | fs-2.8 |
| D | As C | Delisting outcomes; estimate reclassification | fs-2.8 |
| E | As D | Pre-registered SPA coverage rule | fs-2.8 |

Sample (C): 2016-01-05 to 2026-10-29, 545 decision dates, 413 securities, about 178 names per date.

Walk-forward IC / holdout IC (holdout t):

| Horizon | A | B | C | D / E |
|---|---|---|---|---|
| 1 | −0.003 / 0.006 (0.58) | −0.005 / 0.013 (1.02) | −0.009 / 0.011 (0.86) | −0.008 / 0.013 (1.00) |
| 5 | 0.008 / 0.035 | −0.002 / 0.005 (0.49) | −0.004 / 0.007 (0.83) | −0.006 / 0.006 (0.74) |
| 20 | 0.009 / 0.028 (1.18) | 0.008 / 0.003 (0.19) | 0.011 / 0.010 (0.60) | 0.012 / 0.009 (0.56) |
| 60 | 0.000 / 0.071 | 0.015 / 0.010 (0.32) | 0.015 / 0.009 (0.28) | 0.014 / 0.009 (0.29) |
| 120 | −0.004 / 0.095 | 0.005 / −0.007 (−0.24) | 0.007 / −0.005 (−0.16) | 0.005 / −0.008 (−0.24) |

20-session overfitting diagnostics:

| | A | B | C | D | E |
|---|---|---|---|---|---|
| Deflated Sharpe | 0.0 | 0.236 | 0.012 | 0.013 | 0.013 |
| PBO | — | not computable | not computable | **0.17** | 0.17 |
| Hansen SPA p | 0.98 | not computed | not computed | not computed | **0.64** |
| White RC p | — | — | — | — | 0.68 |
| Holdout net excess > 0 | no | no | no | no | no |

Run E SPA detail: 48 periods (the ensemble's walk-forward window), 96 of 103 strategies admitted. Seven
excluded with 17% coverage each: `macd_hist`, `r2_mkt_252`, `vol_ratio_20_60`, `credit_tilt`,
`opt_mkt_term_tilt`, `opt_mkt_skew_tilt`, `growth_surprise_tilt`. The best strategy in the set was
`feat:energy_crack_chg` at +0.38% net per period, not significant after the joint test.

Pre-registered controls (run D, 20 sessions, 111 walk-forward periods):

| Strategy | Gross / period | Net / period | Net HAC t |
|---|---|---|---|
| Ridge | +0.13% | −0.25% | −0.88 |
| Equal-weight signed composite | +0.13% | −0.21% | −0.65 |
| Time-shuffled training (null) | +0.11% | −0.16% | −0.57 |

Ridge minus equal-weight: −0.03% per period, 95% CI [−0.64%, +0.54%].

Gate failures for the ensemble at 20 sessions (run E): order size is a placeholder; four attestations
missing; holdout IC t 0.56 < 2.0; holdout net ≤ 0; DSR 0.013 < 0.90; SPA p 0.64 > 0.10.

### 6.3 New feature tests (run C, univariate HAC t)

| Feature | h=1 | h=5 | h=20 | h=60 | h=120 | Registered sign |
|---|---|---|---|---|---|---|
| `eps_revision_pit_31d` | 0.41 | 1.91 | 0.78 | 0.04 | 0.77 | + |
| `est_dispersion_pit` | 0.24 | −0.37 | −0.30 | 0.19 | 0.28 | − |
| `pt_rev_1m_pit` | −0.06 | −0.56 | −0.99 | −0.16 | 0.90 | + |
| `rec_rev_1m_pit` | −1.04 | −2.27 | **−3.20** | −1.34 | 0.71 | + |
| `sector_etf_flow_20d` | −1.20 | −1.23 | −2.38 | −0.95 | −0.92 | 0 |
| `growth_surprise_tilt` | −0.81 | −0.73 | −2.00 | 0.03 | 0.38 | 0 |
| `inflation_surprise_tilt` | −1.74 | −0.59 | 0.12 | 0.40 | −0.41 | 0 |

No test of 1,385 survives BH-FDR at q = 0.10.

### 6.4 Portfolio construction comparison (`reports/institutional_comparison_large.json`, 15 Nov)

Top 20% concentrated book, 20 sessions, self-financing with weight drift, $600 orders. Status
`DIAGNOSTIC_ONLY`.

| Book | Sample | Periods | Gross | Cost | Net | Net t | Turnover |
|---|---|---|---|---|---|---|---|
| Model ranking | Walk-forward | 48 | +0.77% | 0.36% | +0.41% | 0.95 | 38% |
| Model ranking | Holdout | 24 | +0.38% | 0.47% | −0.08% | −0.15 | 46% |
| `vqm-1` (fixed value/quality/momentum) | Walk-forward | 48 | +0.22% | 0.28% | −0.06% | −0.24 | 32% |
| `vqm-1` | Holdout | 24 | +1.29% | 0.26% | +1.02% | 1.69 | 27% |
| Model, turnover buffer (`cross_section_turnover_v1`) | Walk-forward | 48 | +0.53% | 0.17% | +0.37% | 0.85 | 16% |

The turnover buffer halved costs as pre-registered, but gross fell too, so net did not improve.
Neither book is consistent across samples.

### 6.5 Earlier VPS-era backtest (22 October, Yahoo universe, `docs/BACKTEST_2026-10-22.md`)

| Universe | Sample | Gross | Cost | Net (per 20 sessions vs SPY) |
|---|---|---|---|---|
| large | Walk-forward | −0.170% | 0.372% | −0.541% |
| large | Holdout | +0.166% | 0.392% | −0.226% |
| broad | Walk-forward | +0.006% | 0.450% | −0.444% |
| broad | Holdout | +0.296% | 0.504% | −0.208% |

The walk-forward was negative even at zero cost.

---

## 7. Why costs dominate

Per-name one-way cost in the run D/E panel (`research.name_costs`, $600 order; decomposed from
`data/processed/panel_large.parquet`):

| Component | Mean one-way | Notes |
|---|---|---|
| FX conversion (Trading 212, GBP→USD) | 0.150% | Every trade converts |
| Half spread: max(½ Corwin-Schultz, 3 bp floor) | 0.263% (median 0.237%) | No row sits at the 3 bp floor |
| Square-root impact, 0.7·σ·√(600/ADV) | 0.003% | Negligible at $600 |
| **Total** | **0.42%** (median 0.39%) | Round trip about 0.8% |

At about 40% turnover per 20 sessions this gives the 0.36–0.50% per-period costs seen in every
cross-sectional run. The insider v1 model used a separate fixed round trip of 0.46%.

> **Update 19 November: verified.** LSEG end-of-day quotes give a median quoted half-spread of 1.26 bp,
> against 23.7 bp from Corwin-Schultz. That is 19× too high, on 99.998% of rows (pre-registered check in
> `prereg/spread_check_v1.json`; results in `docs/LOCAL_REBUILD_2026-11.md`). Under the pre-registered
> rule, cost model v2 replaces it: max(quoted, 3 bp floor). The mean one-way cost falls from 0.42% to
> 0.19%. v2 is not yet wired into the code, and nothing has been rerun under it.

**The spread term was the open question.** Corwin-Schultz estimates spreads from daily high-low ranges.
In liquid S&P 500 names, actual quoted half-spreads are typically a few bp, so a 24 bp median is
probably an overstatement driven by intraday volatility. That has not been verified here. It matters
because:

| Case | Round trip | CEO/CFO buy, 1 session, holdout gross +0.57% | Ridge, gross +0.13% / period |
|---|---|---|---|
| Current model | ~0.80% | Negative net | Negative net |
| FX + 3 bp floor only | ~0.36% | About +0.2% net | About −0.01% net at 40% turnover |

Even at the lower cost, the ridge model fails, because its gross return equals the time-shuffled null
(+0.13% vs +0.11%). The insider result is where the cost estimate decides the outcome.

Implication: verify spreads against actual bid/ask data before drawing any cost conclusion. If the
cost model is changed, register the change before rerunning anything, because changing costs after
seeing results is a form of tuning. Reducing turnover by buffering was tried and lowered gross
proportionally.

---

## 8. Mosaic prompt-pack review (summary of `docs/MOSAIC_PROMPT_REVIEW.md`)

The LLM "Mosaic Theory" prompt pack is an unvalidated scoring rule: hand-set weights (insiders 35%,
weather 40%, FDA 40%), fixed thresholds, a discretionary +1 bias toward defense, nuclear and NVDA, and
tiles built from overlapping data added together. Verified gaps and fixes:

1. Weights and thresholds never validated → log variants; DSR, PBO, t > 3 (Bailey & López de Prado;
   Harvey-Liu-Zhu).
2. Unequal fixed weights → equal or shrunk-to-equal weights (DeMiguel et al. 2009). The engine now
   has an equal-weight baseline; ridge does not beat it.
3. Correlated tiles summed → cluster or orthogonalise.
4. All insider trades weighted alike → routine trades weight 0 (Cohen-Malloy-Pomorski); engine has
   `ins_opp_net_90`.
5. Raw short interest → days-to-cover (Hong et al.); engine has `si_days_to_cover`.
6. Loose options reading → defined 25-delta smirk (Xing-Zhang-Zhao 2010); history not entitled.
7. Published effects treated as current → haircut about 50% (McLean-Pontiff 2016).
8. LLM-supplied figures → every number from a tool call against a primary source.

The project's own results confirm the review's central point: signal families with published edges did
not survive costs and survivorship correction in this universe.

---

## 9. External audit response (summary of `docs/AUDIT_RESPONSE_2026-11-12.md`)

Most audit claims (no CPCV, no DSR/PBO, no multiple-testing correction, no purge, no cost model, no
point-in-time data, no insider composite) were false; the code already had them. True or partly
true: survivorship (fixed by the LSEG bundle), earnings surprise not orthogonalised to momentum (fixed),
regime models (tested; no improvement). The audit's symptoms were correct; its diagnosis was not. The
negative returns and `UNVALIDATED` statuses were the safeguards working.

---

## 10. Pre-registrations and trial accounting

| ID | Registered | Hypothesis | Outcome |
|---|---|---|---|
| `insider-event-v1` | 2026-10-19 | Insider purchases earn positive net abnormal return over 1–5 sessions | Gross yes; net no. `UNVALIDATED` |
| `insider-event-v2` | Nov 2026 (no timestamp in file) | Redesigned, cost-aware insider event model with exact timestamps | **Not yet run** (inputs blocked) |
| `cross-section-turnover-v1` | 2026-11-12 | Buffering cuts cost without cutting gross | Cost halved; gross fell; net t 0.85 |
| `large-ridge-120-v1` | Nov 2026 | 120-session holds earn positive net | Dropped (survivorship artefact) |
| `sp500-large-ridge-20-decision-v1` | Nov 2026 (no timestamp in file) | Ridge ranking earns gross excess on verified S&P 500 members | Run D: gross +0.13% per period, equal to the shuffled null; net −0.25% |
| `lseg-analyst-flow-v1` | 2026-11-15 | Five LSEG analyst/flow features | None supported |
| `spa-coverage-v1` | 2026-11-15 20:50 UTC | SPA admission rule (≥ 90% coverage on the ensemble window) | Computed: SPA p 0.64 |

The holdout (from 2024-11-12; insider from 2024-10-16) has been viewed repeatedly. Only prospective
paper records can now provide clean confirmation.

Amendments made after registration, all before results: `lseg-analyst-flow-v1` implementation notes
(staleness window, ETF lag); `spa-coverage-v1` coverage window (the original definition always failed
closed).

---

## 11. Open blockers

| Blocker | Effect | Options |
|---|---|---|
| No documented quarterly estimate archive | Insider v2 primary study and forward record cannot start | (a) Source a vendor archive with documented availability and correction policy. (b) Pre-register a v3 that drops the estimate control. (c) Run v2 with `--sensitivity-estimates` as a diagnostic only |
| No daily price refresh for bundle security IDs | Forward scoring cannot run while a bundle is active (Yahoo is disabled then) | Scheduled LSEG CodeBook pull for bundle RICs |
| Bundle attestations false | Every gate carries four failures | Needs a certified source (e.g. CRSP), or accept as a disclosed limitation |
| Nasdaq-100 / Russell 2000 membership missing | Coverage check fails; universe is S&P 500 only | LSEG membership pulls for the other indices |
| Merger payouts are proxies | Terminal returns approximate | Pull deal consideration (cash/stock terms) per merger |
| Order size placeholder | Cost model uses $600 assumption | Restore T212 bot configuration |
| Option history not entitled | `opt_*` features cannot be backtested | Databento (paid, needs approval) or accumulate snapshots |

---

## 12. Recommendations

1. **Retire the cross-sectional ridge ensemble** on this universe. Further features would be more
   trials against an exhausted holdout. Keep the code and the validation stack.
2. **Make insider events the single research line.** It is the only effect that is economically
   motivated, documented in the literature, and present gross in both samples here. Concretely:
   - decide between blocker options (a), (b) or (c) above, then run v2 historically on the corrected
     bundle with delisting outcomes;
   - set up the daily LSEG price refresh so the forward record can start;
   - judge it only on the prospective record (≥ 90 days, ≥ 100 settled events, net positive).
3. **Verify the spread estimate before anything else on costs.** Compare Corwin-Schultz half-spreads
   with actual bid/ask quotes for a sample of S&P 500 names; check whether LSEG CodeBook provides
   historical bid/ask on this entitlement. If they differ materially, pre-register a corrected cost model
   (spread source, fallback, floor) before rerunning. This one measurement decides whether the
   insider effect clears costs at $600 orders.
4. **Keep the risk tools in use.** GARCH-t VaR and Blume beta are validated and should be the defaults
   for sizing and risk limits.
5. **Do not trade the VQM or concentrated-book holdout numbers.** They flip sign between samples. If
   they're kept, freeze the recipe and let a forward record decide.

---

## 13. Limitations

- S&P 500 only, from December 2015, month-end membership.
- Merger terminal payouts are proxies; LVLT has no outcome.
- CIKs are LSEG data resolved against SEC, not certified.
- Estimates are calc-dated and split-restated; sensitivity only.
- Insider event study and v1 model ran on the survivor-biased Yahoo universe (October, VPS).
- The holdout was reused before the rebuild, so every historical number is exploratory.
- Order size is a $600 placeholder.

---

## 14. Reproduce

| Item | Location |
|---|---|
| Rebuild narrative, runs A–E | `docs/LOCAL_REBUILD_2026-11.md` |
| Research logs | `ledger_local/research_large_{baseline_yahoo,bundle,fs28,terminal,spa}.log` |
| Latest results | `reports/research_results_large.json` |
| Delisting classification | `ledger_local/delisting_classification_20261115.csv` |
| Insider v2 attempts | `data/research/insider-v2-*`, `ledger_local/insider_v2_*` |
| Pre-registrations | `prereg/*.json` |
| Converters | `lseg_bundle.py`, `lseg_estimates.py` |
| Controls | `ridge_controls.py` |
| SPA rule | `validation_extensions.multiple_comparison`, `config.SPA_MIN_COVERAGE` |
| Pull requests | Trooperds432/mosaic-engine #1 (merged), #2 (merged), #3 (open) |
| Pre-import backup | `data_backup_2026-11-14_pre_lseg_bundle/` |
