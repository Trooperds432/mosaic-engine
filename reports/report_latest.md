# Mosaic Quantamental Research Report — 2026-10-23

Feature set `fs-2.1` · model `mosaic-xs-ridge` v2.0.0 · generated 2026-10-24 01:07 UTC · evidence tags: [OBS] observation, [CALC] calculated statistic, [MODEL] model estimate, [HYP] hypothesis, [UNVER] unverified

## A. Executive summary

- **Regime [CALC]:** uptrend/high_vol; filtered HMM P(high-vol) = 0.00.
- **Data audit [CALC]:** FAILED — all LONG signals blocked (look-ahead truncation, label alignment, duplicates, timestamps).
- **Universe `large` [OBS]:** mcap ≥ $10bn, ADV ≥ 1000k shares and ≥ $5m, point-in-time; 896 names ever eligible, 328 per date; 2009-05-04 → 2026-10-16, holdout from 2024-10-21.
  - **Ensemble at 20d [CALC]:** walk-forward IC -0.0011 (t=-0.14), holdout IC 0.0080 (t=0.43); holdout net excess/period -0.78%. Gate: **UNVALIDATED** — holdout IC t=0.43 < 2.0; walk-forward positive-IC years 55% < 60%; holdout long-only net excess <= 0 after per-name costs; deflated Sharpe 0.0 < 0.9; Hansen SPA p=0.96 > 0.10; forecast deciles not monotonic out-of-sample
- **Universe `broad` [OBS]:** mcap ≥ $0.3bn, ADV ≥ 200k shares and ≥ $2m, point-in-time; 3064 names ever eligible, 1321 per date; 2009-04-20 → 2026-10-16, holdout from 2024-10-21.
  - **Ensemble at 20d [CALC]:** walk-forward IC 0.0063 (t=1.03), holdout IC 0.0280 (t=2.08); holdout net excess/period -0.59%. Gate: **UNVALIDATED** — holdout long-only net excess <= 0 after per-name costs; deflated Sharpe 0.0 < 0.9; Hansen SPA p=0.64 > 0.10; forecast deciles not monotonic out-of-sample
- **Signals:** 0 LONG of 2931 evaluated (5 insider-event). A research thesis is not execution approval; the bot's own risk checks always apply.

## B. Evidence and source audit

| Dataset | Timestamp rule | Source | Quality notes | PIT-eligible |
|---|---|---|---|---|
| Daily OHLCV, split/dividend adjusted | close t; entry t+1 open | Yahoo chart API | vendor survivorship: no delisted prices (0% delisting return) | yes |
| Fundamentals (XBRL, as-reported) | filed ≤ t−1 | SEC companyfacts | first-filed value used; restated series kept separately | yes |
| Insider Forms 3/4/5 + footnotes + derivative table | filed ≤ t−1 | SEC insider data sets, EDGAR XML | 10b5-1 from checkbox (2023+) or footnote | yes |
| 13F holdings | period end + 46d, on-time originals only | SEC 13F data sets | CUSIP→issuer by name match; omits shorts | yes |
| Short interest | settlement + 12d | FINRA consolidated (2020+) | short history | yes (2020+) |
| Macro (rates, credit, dollar, oil products) | obs + 1d | FRED | market series, unrevised | yes |
| Macro (UNRATE, PAYEMS, CPI, INDPRO, semis IP, orders) | first ALFRED vintage containing the value | ALFRED | twice-monthly vintages (conservative lag) | yes |
| EIA petroleum status | week end + 5d | EIA history files |  | yes |
| CFTC COT (disaggregated) | report + 3d | CFTC Socrata |  | yes |
| Commodity futures | close + 1d | Yahoo continuous front month | roll gaps; no term structure | yes |
| Earnings dates | 8-K item 2.02 acceptance time | SEC submissions |  | yes |
| Clinical trials | first-posted date | ClinicalTrials.gov | records edited in place: only first-posted/results-posted PIT-safe | partial |
| FDA approvals | action date | openFDA drugs@FDA | approvals only | yes |
| Options / implied volatility / estimate revisions history | — | not available free | NOT_COMPUTED | no |

Audit [CALC]: truncation invariance worst relative difference 0.00000000 over 3 dates; label max error 0.00000161; duplicate insider rows 0; restated periods 6.06%; series ending early 1; rows with stale fundamentals 6.53%. Missing-not-at-random IC: fundamentals 0.0011, sue -0.0077, 13f -0.0034.

History being collected [OBS]: Cboe options snapshots 4 days, Nasdaq estimate snapshots 4 days (per-stock options/estimate features enter models after 3 years); Alpha Vantage consensus-surprise backfill 94 companies (24/day, largest first). VIX term structure and CBOE SKEW: full history in use.

## C. Factor and sector Mosaic dashboard (discovery sample, sector-neutral, primary horizon) [CALC]

| Family | Feature | Hyp. | IC | HAC t | Style-neutral IC | BH p | FDR | Q5−Q1 |
|---|---|---|---|---|---|---|---|---|
| earnings | gm_change | + [HYP] | -0.0145 | -1.89 | -0.0140 | 1.000 | — | -0.23% |
| earnings | sue | + [HYP] | 0.0075 | 1.43 | 0.0103 | 0.397 | — | 0.23% |
| earnings | eps_growth | + [HYP] | 0.0054 | 1.06 | 0.0086 | 0.517 | — | 0.23% |
| earnings | inv_to_sales_chg | − [HYP] | 0.0043 | 0.77 | 0.0037 | 1.000 | — | 0.20% |
| earnings | op_margin_change | + [HYP] | -0.0042 | -0.65 | -0.0018 | 0.993 | — | -0.27% |
| event | earn_recent | ± [HYP] | -0.0018 | -0.38 | -0.0015 | 0.983 | — | -0.01% |
| insider | ins_buy_x_drawdown | + [HYP] | -0.0074 | -1.36 | -0.0108 | 1.000 | — | -0.00% |
| insider | ins_net_90 | + [HYP] | -0.0040 | -0.87 | -0.0058 | 1.000 | — | -0.05% |
| insider | ins_sponsor_sell_90 | − [HYP] | -0.0055 | -0.80 | -0.0029 | 0.653 | — | -0.05% |
| insider | ins_routine_net_90 | ± [HYP] | 0.0049 | 0.76 | -0.0031 | 0.857 | — | -0.04% |
| insider | ins_dir_buy_90 | + [HYP] | -0.0041 | -0.72 | -0.0057 | 0.996 | — | 0.11% |
| insider | ins_net_180 | + [HYP] | -0.0033 | -0.71 | -0.0047 | 0.996 | — | -0.16% |
| insider | ins_buyers_90 | + [HYP] | -0.0039 | -0.70 | -0.0083 | 0.993 | — | -0.03% |
| insider | ins_conviction | + [HYP] | 0.0030 | 0.30 | 0.0031 | 0.822 | — | -0.01% |
| insider | ins_purchase_intensity | + [HYP] | -0.0020 | -0.22 | -0.0028 | 0.933 | — | -0.01% |
| insider | ins_opp_net_90 | + [HYP] | -0.0008 | -0.17 | -0.0010 | 0.933 | — | -0.09% |
| insider | ins_surprise | + [HYP] | -0.0007 | -0.17 | 0.0007 | 0.933 | — | -0.03% |
| insider | ins_officer_buy_180 | + [HYP] | 0.0009 | 0.13 | -0.0047 | 0.857 | — | -0.17% |
| insider | ins_plan_sale_share | ± [HYP] | -0.0005 | -0.12 | 0.0019 | 1.000 | — | -0.05% |
| insider | ins_sellers_90 | − [HYP] | 0.0004 | 0.08 | 0.0035 | 0.915 | — | -0.03% |
| institutional | io_hhi | − [HYP] | -0.0076 | -1.46 | -0.0083 | 0.382 | — | -0.08% |
| institutional | io_breadth_chg | + [HYP] | 0.0065 | 0.75 | 0.0076 | 0.668 | — | 0.19% |
| institutional | io_holders_chg | + [HYP] | -0.0021 | -0.24 | -0.0001 | 0.933 | — | 0.09% |
| institutional | io_ratio | ± [HYP] | -0.0006 | -0.12 | 0.0009 | 1.000 | — | -0.13% |
| investment | share_issuance | − [HYP] | -0.0155 | -2.52 | -0.0147 | 0.132 | — | -0.41% |
| investment | capex_intensity | − [HYP] | -0.0143 | -2.50 | -0.0144 | 0.134 | — | -0.31% |
| investment | asset_growth | − [HYP] | -0.0015 | -0.26 | 0.0004 | 0.835 | — | -0.01% |
| liquidity | cs_spread | + [HYP] | -0.0083 | -0.86 | -0.0133 | 1.000 | — | 0.01% |
| liquidity | turnover | − [HYP] | -0.0020 | -0.27 | -0.0031 | 0.834 | — | 0.23% |
| liquidity | amihud | + [HYP] | -0.0008 | -0.12 | 0.0008 | 0.923 | — | -0.11% |
| macro | rates_tilt | + [HYP] | -0.0086 | -1.59 | -0.0059 | 1.000 | — | -0.19% |
| macro | credit_tilt | + [HYP] | 0.0069 | 1.34 | 0.0042 | 0.416 | — | 0.24% |
| macro | beta_credit | ± [HYP] | 0.0051 | 0.86 | 0.0039 | 0.834 | — | 0.24% |
| macro | beta_rates | ± [HYP] | 0.0042 | 0.68 | 0.0008 | 0.889 | — | 0.06% |
| macro | beta_dollar | ± [HYP] | 0.0031 | 0.49 | 0.0058 | 0.948 | — | 0.04% |
| momentum | mom_6_1 | + [HYP] | -0.0053 | -0.63 | -0.0022 | 0.993 | — | 0.17% |
| momentum | dist_52w_high | + [HYP] | 0.0047 | 0.45 | 0.0080 | 0.776 | — | -0.01% |
| momentum | mom_12_1 | + [HYP] | 0.0042 | 0.43 | 0.0096 | 0.778 | — | 0.25% |
| positioning | si_ratio | − [HYP] | -0.0330 | -2.21 | -0.0369 | 0.172 | — | -0.45% |
| positioning | si_days_to_cover | − [HYP] | -0.0088 | -1.51 | -0.0105 | 0.363 | — | -0.43% |
| positioning | si_change | − [HYP] | 0.0029 | 0.60 | 0.0021 | 0.993 | — | 0.05% |
| quality | gross_prof | + [HYP] | 0.0170 | 2.07 | 0.0178 | 0.196 | — | 0.29% |
| quality | roic | + [HYP] | 0.0132 | 1.91 | 0.0107 | 0.223 | — | 0.21% |
| quality | leverage | − [HYP] | 0.0109 | 1.60 | 0.0100 | 1.000 | — | 0.17% |
| quality | roa | + [HYP] | 0.0112 | 1.51 | 0.0118 | 0.361 | — | 0.13% |
| quality | accruals | − [HYP] | -0.0045 | -1.00 | -0.0039 | 0.548 | — | -0.20% |
| reversal | ret_1m | − [HYP] | -0.0123 | -1.83 | -0.0134 | 0.252 | — | -0.15% |
| risk | vol_60 | − [HYP] | -0.0067 | -0.57 | -0.0134 | 0.733 | — | 0.10% |
| risk | beta_252 | − [HYP] | 0.0046 | 0.34 | NOT_COMPUTED | 0.957 | — | 0.26% |
| sector_energy | energy_oil_tilt | + [HYP] | 0.0455 | 0.80 | 0.0455 | 0.651 | — | 0.78% |
| sector_energy | energy_inv_surprise | + [HYP] | -0.0376 | -0.58 | -0.0371 | 0.993 | — | -0.69% |
| sector_energy | energy_cot_z | − [HYP] | -0.0273 | -0.51 | -0.0270 | 0.751 | — | -0.90% |
| sector_fin | fin_nim_proxy | + [HYP] | -0.1524 | -1.45 | -0.1524 | 1.000 | — | -1.45% |
| sector_fin | fin_provision_ratio | − [HYP] | -0.0409 | -0.44 | -0.0409 | 0.776 | — | -0.48% |
| sector_fin | fin_deposit_growth | + [HYP] | 0.0069 | 0.18 | 0.0069 | 0.854 | — | -2.24% |
| sector_fin | fin_slope_tilt | + [HYP] | NOT_COMPUTED | NOT_COMPUTED | 0.1215 | 1.000 | — | 0.16% |
| sector_health | bio_cash_runway | + [HYP] | 0.1313 | 3.21 | 0.0146 | 0.032 | ✓ | 1.53% |
| sector_health | bio_fda_appr_365 | + [HYP] | 0.0374 | 1.39 | -0.0212 | 0.406 | — | 1.08% |
| sector_health | bio_ph3_trials | + [HYP] | 0.0363 | 1.26 | -0.0219 | 0.450 | — | 0.62% |
| sector_health | bio_rnd_intensity | ± [HYP] | NOT_COMPUTED | NOT_COMPUTED | NOT_COMPUTED | 1.000 | — | -12.33% |
| sector_materials | mat_commodity_tilt | + [HYP] | 0.0061 | 0.35 | 0.0059 | 0.811 | — | 0.44% |
| sector_tech | semi_cycle | + [HYP] | 0.0282 | 0.41 | -0.0173 | 0.790 | — | 2.51% |
| sector_tech | semi_orders | + [HYP] | 0.0282 | 0.41 | -0.0173 | 0.790 | — | 2.51% |
| size | log_mcap | − [HYP] | -0.0036 | -0.58 | NOT_COMPUTED | 0.733 | — | -0.14% |
| technical | bb_pctb | − [HYP] | -0.0083 | -1.53 | -0.0089 | 0.352 | — | -0.12% |
| technical | rsi_14 | − [HYP] | -0.0075 | -1.14 | -0.0081 | 0.481 | — | -0.15% |
| technical | px_sma200 | + [HYP] | -0.0046 | -0.49 | -0.0019 | 0.976 | — | 0.09% |
| technical | macd_hist | + [HYP] | -0.0021 | -0.38 | -0.0039 | 0.972 | — | -0.12% |
| value | fcf_yield | + [HYP] | 0.0128 | 1.92 | 0.0100 | 0.223 | — | 0.38% |
| value | earn_yield | + [HYP] | 0.0094 | 1.32 | 0.0075 | 0.426 | — | 0.14% |
| value | book_to_mkt | + [HYP] | -0.0012 | -0.18 | -0.0035 | 0.933 | — | 0.01% |

IC = per-date Spearman correlation with forward SPY-excess return; Newey-West lags = h/step; BH-FDR q=0.1 over 1055 tests (one-sided where a sign was pre-specified). Sector features are NaN outside their sector. Contributions to forecasts are not reported where the gate failed.
New idiosyncratic / lottery / stationarity features, `large`, discovery period only [CALC]: 7 of 165 tests pass BH-FDR jointly with all 1220 tests: r2_mkt_252 (style_neutral, 120d, IC +0.060, t=5.0).
New idiosyncratic / lottery / stationarity features, `broad`, discovery period only [CALC]: 17 of 150 tests pass BH-FDR jointly with all 1205 tests: text_sim_jac (style_neutral, 20d, IC +0.012, t=4.4), text_sim_cos (style_neutral, 20d, IC +0.007, t=2.6).
Mutual information (k-NN, permutation null, BH-FDR) [CALC]: 27 of 71 features show significant dependence; strongest: beta_252 (z=13.5), gross_prof (z=7.7), log_mcap (z=4.7), ins_net_90 (z=5.0), fin_slope_tilt (z=4.9), beta_rates (z=4.8), capex_intensity (z=5.2), gm_change (z=2.4). Non-linear only (MI significant, IC not): beta_252, gross_prof, log_mcap, ins_net_90, fin_slope_tilt, beta_rates, capex_intensity, gm_change, op_margin_change, inv_to_sales_chg, io_hhi, ins_purchase_intensity, dist_52w_high, roa, io_holders_chg, fin_provision_ratio, si_change, energy_cot_z, semi_orders, earn_recent, energy_inv_surprise, energy_oil_tilt, ins_surprise, px_sma200, bio_fda_appr_365, fcf_yield.
Multicollinearity: 13 feature pairs with |ρ| > 0.7; max VIF 17.3.

## D. Insider activity and ownership

Event study [CALC] (beta-adjusted CAR from first open after filing; SEs clustered by month):

| Event | Sample | N | 1d | 5d | 20d | 60d |
|---|---|---|---|---|---|---|
| cluster_buy | discovery | 2013 | 0.28% (t=3.72) | 0.45% (t=2.43) | 0.45% (t=1.10) | 2.47% (t=3.95) |
| cluster_buy | holdout | 217 | 0.35% (t=2.30) | 0.94% (t=2.22) | 0.59% (t=0.80) | -1.64% (t=-0.84) |
| ceo_cfo_buy | discovery | 1449 | 0.44% (t=6.65) | 0.72% (t=4.45) | 0.62% (t=1.64) | 2.76% (t=3.42) |
| ceo_cfo_buy | holdout | 201 | 0.57% (t=2.33) | 0.98% (t=2.55) | 0.21% (t=0.29) | -0.78% (t=-0.44) |
| any_buy | discovery | 6377 | 0.20% (t=7.11) | 0.29% (t=3.46) | 0.31% (t=1.52) | 1.41% (t=3.80) |
| any_buy | holdout | 732 | 0.12% (t=1.23) | 0.23% (t=0.99) | 0.50% (t=1.11) | 0.49% (t=0.36) |
| discr_sell | discovery | 24872 | -0.04% (t=-3.78) | -0.06% (t=-1.81) | -0.04% (t=-0.53) | 0.40% (t=2.54) |
| discr_sell | holdout | 2309 | -0.07% (t=-1.26) | -0.04% (t=-0.25) | -0.09% (t=-0.21) | -0.40% (t=-0.49) |

Pre-registered insider event model (`insider-event-v1`) [CALC]: gate **UNVALIDATED**; discovery net -0.24% (t=-1.28), vs matched controls 0.27% (t=2.43), incremental regression t=2.16.

Largest open-market purchases filed in the last 30 days [OBS]:

| Filed | Ticker | Insider | Role | Class | Value $ | Plan |
|---|---|---|---|---|---|---|
| 2026-10-16 | SAFX | EEME Energy SPV I, LLC | nan | unclassified | 142,083,451 |  |
| 2026-10-01 | UUU | AULT MILTON C III | nan | opportunistic | 92,412,000 |  |
| 2026-10-05 | LEN | BERKSHIRE HATHAWAY INC | nan | routine | 67,453,950 |  |
| 2026-09-30 | LEN | BUFFETT WARREN E | nan | opportunistic | 52,120,753 |  |
| 2026-09-25 | LEN | BUFFETT WARREN E | nan | opportunistic | 43,518,878 |  |
| 2026-10-16 | PAM | Mindlin Marcos Marcelo | nan | unclassified | 35,139,300 |  |
| 2026-10-05 | LEN | BERKSHIRE HATHAWAY INC | nan | routine | 35,118,616 |  |
| 2026-10-05 | LEN | BERKSHIRE HATHAWAY INC | nan | routine | 33,852,327 |  |
| 2026-09-25 | LEN | BUFFETT WARREN E | nan | opportunistic | 32,170,598 |  |
| 2026-10-09 | BBD | Alvarez Rubens Aguiar | nan | unclassified | 29,494,260 |  |
| 2026-10-05 | LEN | BERKSHIRE HATHAWAY INC | nan | routine | 27,071,964 |  |
| 2026-10-09 | GGR | Yin Chung-Yao | nan | unclassified | 26,517,849 |  |
| 2026-09-25 | LEN | BUFFETT WARREN E | nan | opportunistic | 25,352,361 |  |
| 2026-09-30 | CRBD | NIPPON LIFE INSURANCE CO | nan | unclassified | 25,125,357 | 10b5-1 |
| 2026-10-13 | GME | Cohen Ryan | President, CEO and Chairman | unclassified | 20,979,000 |  |

Net discretionary $ (buys − non-plan sales), last 30d, by insider class [CALC]: opportunistic $-105.5m, routine $185.9m, unclassified $-2,039.3m. Routine/opportunistic per Cohen-Malloy-Pomorski using only each insider's earlier trades.
Cluster indicator [CALC]: 71 issuers with ≥2 distinct buyers in the last 30 days.

13F [CALC]: ownership ratio, breadth change (Chen-Hong-Stein), holder-count change and HHI enter the factor tests in section C. 13F is filed up to 45 days after quarter end, omits short positions and non-13(f) securities, and holding-value changes are not flows.

## E. Model performance and validation [CALC]

**Profile `large`** — 288,595 security-dates; costs per name (mean one-way 0.43% at $450 orders).

| h | Model | WF IC | WF t | HO IC | HO t | HO IC 95% (block bootstrap) | HO eff. N | WF +yrs | HO net/period | HO net SR | Gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ensemble | -0.0017 | -0.40 | 0.0034 | 0.38 | [-0.013, 0.021] | 99 | 0.55 | -0.24% | -7.15 | UNVALIDATED |
| 5 | ensemble | 0.0033 | 0.62 | 0.0084 | 0.82 | [-0.012, 0.029] | 99 | 0.45 | -0.29% | -2.10 | UNVALIDATED |
| 20 | ridge | -0.0027 | -0.37 | 0.0068 | 0.38 | [-0.027, 0.036] | 24 | 0.45 | -0.64% | -0.93 |  |
| 20 | enet | -0.0072 | -0.88 | 0.0076 | 0.41 | [-0.029, 0.041] | 24 | 0.36 | -0.61% | -0.87 |  |
| 20 | huber | 0.0013 | 0.19 | 0.0049 | 0.24 | [-0.034, 0.043] | 24 | 0.55 | -0.71% | -0.97 |  |
| 20 | lgbm | -0.0020 | -0.25 | 0.0042 | 0.22 | [-0.023, 0.032] | 24 | 0.55 | -0.69% | -0.99 |  |
| 20 | gam | 0.0038 | 0.57 | 0.0070 | 0.38 | [-0.027, 0.038] | 24 | 0.73 | -0.54% | -0.73 |  |
| 20 | hier | -0.0024 | -0.23 | 0.0268 | 1.47 | [-0.006, 0.059] | 24 | 0.64 | -0.16% | -0.27 |  |
| 20 | logit | -0.0095 | -1.54 | -0.0013 | -0.11 | [-0.024, 0.019] | 24 | 0.18 | -0.43% | -0.57 |  |
| 20 | factor_resid | 0.0032 | 0.41 | 0.0111 | 0.69 | [-0.02, 0.041] | 24 | 0.64 | -0.57% | -0.89 |  |
| 20 | ensemble | -0.0011 | -0.14 | 0.0080 | 0.43 | [-0.027, 0.043] | 24 | 0.55 | -0.78% | -1.10 | UNVALIDATED |
| 60 | ensemble | -0.0125 | -1.19 | 0.0370 | 1.56 | [-0.004, 0.078] | 7 | 0.36 | -1.42% | -0.78 | UNVALIDATED |
| 120 | ensemble | -0.0201 | -1.38 | 0.0449 | 4.17 | [0.028, 0.062] | 3 | 0.27 | -3.06% | NOT_COMPUTED | UNVALIDATED |

Overfitting: 80 strategies; deflated Sharpe 0.000, PSR 0.000, PBO 0.381, White RC p 0.987, Hansen SPA p 0.960, MinTRL NOT_COMPUTED periods, WF→holdout IC degradation -0.0091.
Factor-adjusted alpha (holdout long-only vs risk-model factors): -0.18%/period (HAC t=-0.80), R² 0.74; diagnostics: Durbin-Watson 2.29, Jarque-Bera p 0.760, Breusch-Pagan p 0.122, max VIF 2.1.
Calibration: decile monotonicity WF -0.35, holdout -0.35 → NOT_CALIBRATED; Mondrian conformal 95% interval holdout coverage 87.69%.
Probability model P(excess > mean round-trip cost 0.0085): holdout Brier 0.2430 vs base-rate 0.2429.
Model disagreement (mean cross-model rank SD): 0.159.
Regime conditioning (filtered HMM interactions) vs unconditional: ΔIC 0.0029 (t=1.45) → no reliable improvement; unconditional model kept.
Markov-switching regression: current filtered P(high-vol) 0.03; regime weight sets correlate 0.35; walk-forward ΔIC 0.0027 (t=0.71), holdout ΔIC 0.0041 (t=1.08) → not adopted.
Structural breaks (p<0.001): factor:value (63d mean, monthly samples) 2025-09-16 (-0.000→+0.000); factor:volatility (63d mean, monthly samples) 2026-02-17 (-0.000→+0.001); factor:illiquidity (63d mean, monthly samples) 2013-03-08 (+0.000→-0.000); ic:gross_prof 2011-02-22 (-0.085→+0.114); ic:gross_prof 2012-05-01 (+0.114→-0.012); ic:gross_prof 2016-12-30 (-0.012→+0.074); ic:gross_prof 2020-10-09 (+0.074→+0.004); ic:credit_tilt 2025-03-20 (+0.017→-0.083).
HMM 2 vs 3 states: 3-state better out-of-sample likelihood in 16/19 years (t=3.65) → live regime uses 3 states; ranking improvement t=-0.08 (not adopted for forecasts).
Factor returns that differ by regime (|t|≥2): market calm 15.56% vs stress -45.49% (t=-2.22).

Matched random portfolios (2000 paths; same names count, rebalancing, turnover and per-name costs):

| Test | Model net/period | Random median | Random 5–95% | Model percentile |
|---|---|---|---|---|
| h5 walk forward | -0.30% | -0.27% | [-0.29%, -0.24%] | 2.20% |
| h5 holdout | -0.29% | -0.27% | [-0.34%, -0.19%] | 32.60% |
| h20 walk forward | -0.62% | -0.46% | [-0.57%, -0.35%] | 1.05% |
| h20 holdout | -0.78% | -0.63% | [-0.94%, -0.31%] | 22.20% |
Observable Markov chain on SPY (±5% over 20 days): long/cash Sharpe 0.45 vs SPY 0.61 (discovery), 0.75 vs 0.99 (holdout); signal vs next 20d return t=-0.68 → not adopted.

Robustness (walk-forward ensemble IC by split):

| Split | Group | IC | t | Dates |
|---|---|---|---|---|
| cap_tercile | high | 0.0145 | 1.64 | 544 |
| cap_tercile | mid | -0.0112 | -1.25 | 544 |
| cap_tercile | low | -0.0052 | -0.57 | 544 |
| liquidity_tercile | high | 0.0100 | 0.89 | 544 |
| liquidity_tercile | low | 0.0009 | 0.10 | 544 |
| liquidity_tercile | mid | -0.0110 | -1.24 | 544 |
| regime | downtrend/high_vol | 0.0027 | 0.22 | 91 |
| regime | uptrend/high_vol | -0.0239 | -1.64 | 172 |
| regime | uptrend/low_vol | 0.0116 | 1.19 | 281 |
| earnings_window | earnings | -0.0114 | -0.71 | 346 |
| earnings_window | no_earnings | -0.0003 | -0.03 | 544 |
| sector | Materials | -0.0280 | -1.58 | 544 |
| sector | Energy | -0.0343 | -1.70 | 544 |
| sector | Utilities | 0.0240 | 1.46 | 544 |
| sector | Other | 0.0024 | 0.15 | 544 |
| sector | Technology | -0.0073 | -0.51 | 544 |
| sector | Health Care | 0.0376 | 2.70 | 544 |
| sector | Industrials | -0.0103 | -0.74 | 544 |
| sector | Financials | 0.0063 | 0.46 | 544 |
| sector | Consumer | 0.0068 | 0.40 | 544 |
| sector | Consumer Staples | -0.0189 | -1.10 | 544 |
| sector | Communication | -0.0197 | -0.42 | 139 |
| sector | Real Estate | 0.0098 | 0.53 | 544 |

Cost sensitivity (WF net/period at cost multiples): x0 -0.28%, x0.5 -0.45%, x1 -0.62%, x2 -0.96%, x3 -1.30%. Top-quantile sensitivity: 0.1 -0.82%, 0.2 -0.62%, 0.3 -0.43%. Ridge α sensitivity (WF IC): 10.0 -0.0027, 1.0 0.0001, 100.0 -0.0060. Equal-weight-universe benchmark WF net/period -0.46%.

**Profile `broad`** — 1,164,785 security-dates; costs per name (mean one-way 0.54% at $450 orders).

| h | Model | WF IC | WF t | HO IC | HO t | HO IC 95% (block bootstrap) | HO eff. N | WF +yrs | HO net/period | HO net SR | Gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ensemble | -0.0034 | -0.86 | 0.0057 | 0.62 | [-0.011, 0.025] | 99 | 0.45 | -0.30% | -7.47 | UNVALIDATED |
| 5 | ensemble | 0.0086 | 2.09 | 0.0245 | 2.10 | [0.003, 0.045] | 99 | 0.73 | -0.30% | -1.62 | RESEARCH_CANDIDATE |
| 20 | ridge | -0.0019 | -0.29 | 0.0264 | 1.90 | [0.002, 0.05] | 24 | 0.36 | -0.51% | -0.59 |  |
| 20 | enet | -0.0065 | -0.90 | 0.0256 | 1.76 | [-0.001, 0.05] | 24 | 0.36 | -0.50% | -0.58 |  |
| 20 | huber | 0.0083 | 1.45 | 0.0295 | 1.51 | [-0.008, 0.069] | 24 | 0.55 | -0.60% | -0.76 |  |
| 20 | lgbm | 0.0050 | 0.77 | 0.0127 | 0.72 | [-0.017, 0.042] | 24 | 0.45 | -0.91% | -1.05 |  |
| 20 | gam | 0.0018 | 0.31 | 0.0286 | 2.40 | [0.009, 0.046] | 24 | 0.55 | -0.41% | -0.49 |  |
| 20 | hier | -0.0003 | -0.04 | 0.0285 | 1.83 | [0.001, 0.054] | 24 | 0.64 | -0.47% | -0.53 |  |
| 20 | logit | 0.0012 | 0.16 | 0.0100 | 0.49 | [-0.03, 0.046] | 24 | 0.27 | -0.09% | -0.08 |  |
| 20 | factor_resid | 0.0168 | 2.85 | 0.0297 | 1.71 | [-0.009, 0.062] | 24 | 0.73 | -0.50% | -0.70 |  |
| 20 | ensemble | 0.0063 | 1.03 | 0.0280 | 2.08 | [0.002, 0.05] | 24 | 0.64 | -0.59% | -0.74 | UNVALIDATED |
| 60 | ensemble | -0.0148 | -1.49 | 0.0485 | 3.05 | [0.017, 0.08] | 7 | 0.27 | -0.42% | -0.24 | UNVALIDATED |
| 120 | ensemble | -0.0277 | -1.89 | 0.0639 | 4.05 | [0.042, 0.085] | 3 | 0.36 | -0.84% | NOT_COMPUTED | UNVALIDATED |

Overfitting: 80 strategies; deflated Sharpe 0.000, PSR 0.030, PBO 0.099, White RC p 0.608, Hansen SPA p 0.635, MinTRL NOT_COMPUTED periods, WF→holdout IC degradation -0.0217.
Factor-adjusted alpha (holdout long-only vs risk-model factors): -0.57%/period (HAC t=-1.41), R² 0.66; diagnostics: Durbin-Watson 2.10, Jarque-Bera p 0.121, Breusch-Pagan p 0.089, max VIF 3.4.
Calibration: decile monotonicity WF 0.21, holdout 0.07 → NOT_CALIBRATED; Mondrian conformal 95% interval holdout coverage 90.69%.
Probability model P(excess > mean round-trip cost 0.0108): holdout Brier 0.2414 vs base-rate 0.2416.
Model disagreement (mean cross-model rank SD): 0.167.
Regime conditioning (filtered HMM interactions) vs unconditional: ΔIC 0.0064 (t=2.56) → improves.
Markov-switching regression: current filtered P(high-vol) 0.03; regime weight sets correlate 0.52; walk-forward ΔIC 0.0050 (t=1.10), holdout ΔIC 0.0103 (t=1.58) → not adopted.
Structural breaks (p<0.001): factor:value (63d mean, monthly samples) 2025-09-16 (-0.000→+0.001); factor:illiquidity (63d mean, monthly samples) 2021-07-12 (+0.000→-0.000); ic:fcf_yield 2021-02-11 (+0.013→+0.118); ic:fcf_yield 2022-05-26 (+0.118→+0.030); ic:capex_intensity 2025-08-13 (-0.017→+0.067); ic:capex_intensity 2026-03-19 (+0.067→-0.068); ic:share_issuance 2021-02-04 (-0.009→-0.049); ic:bio_ph3_trials 2017-08-14 (+0.031→-0.092).
HMM 2 vs 3 states: 3-state better out-of-sample likelihood in 16/19 years (t=3.65) → live regime uses 3 states; ranking improvement t=0.71 (not adopted for forecasts).
Factor returns that differ by regime (|t|≥2): market calm 18.39% vs stress -55.58% (t=-2.21).

Matched random portfolios (2000 paths; same names count, rebalancing, turnover and per-name costs):

| Test | Model net/period | Random median | Random 5–95% | Model percentile |
|---|---|---|---|---|
| h5 walk forward | -0.30% | -0.29% | [-0.31%, -0.27%] | 9.45% |
| h5 holdout | -0.30% | -0.35% | [-0.41%, -0.30%] | 95.10% |
| h20 walk forward | -0.52% | -0.46% | [-0.54%, -0.38%] | 13.20% |
| h20 holdout | -0.59% | -0.62% | [-0.84%, -0.39%] | 56.55% |
Observable Markov chain on SPY (±5% over 20 days): long/cash Sharpe 0.45 vs SPY 0.61 (discovery), 0.75 vs 0.99 (holdout); signal vs next 20d return t=-0.68 → not adopted.

Robustness (walk-forward ensemble IC by split):

| Split | Group | IC | t | Dates |
|---|---|---|---|---|
| cap_tercile | high | 0.0064 | 0.88 | 544 |
| cap_tercile | low | 0.0158 | 2.93 | 544 |
| cap_tercile | mid | 0.0056 | 1.01 | 544 |
| liquidity_tercile | high | 0.0026 | 0.32 | 544 |
| liquidity_tercile | mid | 0.0061 | 0.95 | 544 |
| liquidity_tercile | low | 0.0146 | 2.73 | 544 |
| regime | downtrend/high_vol | 0.0011 | 0.12 | 91 |
| regime | uptrend/high_vol | -0.0003 | -0.03 | 172 |
| regime | uptrend/low_vol | 0.0120 | 1.56 | 281 |
| earnings_window | earnings | 0.0028 | 0.31 | 534 |
| earnings_window | no_earnings | 0.0070 | 1.15 | 544 |
| sector | Industrials | 0.0087 | 0.99 | 544 |
| sector | Energy | -0.0235 | -1.51 | 544 |
| sector | Materials | -0.0082 | -0.83 | 544 |
| sector | Utilities | 0.0157 | 1.39 | 544 |
| sector | Technology | 0.0061 | 0.72 | 544 |
| sector | Consumer | 0.0014 | 0.14 | 544 |
| sector | Other | 0.0182 | 2.11 | 544 |
| sector | Health Care | 0.0225 | 2.60 | 544 |
| sector | Real Estate | -0.0006 | -0.05 | 544 |
| sector | Financials | 0.0031 | 0.29 | 544 |
| sector | Consumer Staples | -0.0028 | -0.23 | 544 |
| sector | Communication | 0.0160 | 1.00 | 544 |
| sector | Unknown | -0.0038 | -0.10 | 105 |

Cost sensitivity (WF net/period at cost multiples): x0 -0.09%, x0.5 -0.30%, x1 -0.52%, x2 -0.94%, x3 -1.37%. Top-quantile sensitivity: 0.1 -0.60%, 0.2 -0.52%, 0.3 -0.46%. Ridge α sensitivity (WF IC): 10.0 -0.0019, 1.0 0.0001, 100.0 -0.0086. Equal-weight-universe benchmark WF net/period -0.43%.

## F. Forecast distribution (top ensemble decile, out-of-sample history)

| Profile | Horizon | Expected excess [MODEL] | 95% interval | P(excess>0) | 5% quantile | Calibration |
|---|---|---|---|---|---|---|
| large | 1 | NOT_COMPUTED | NOT_COMPUTED | NOT_COMPUTED | NOT_COMPUTED | NOT_CALIBRATED |
| large | 5 | NOT_COMPUTED | NOT_COMPUTED | NOT_COMPUTED | NOT_COMPUTED | NOT_CALIBRATED |
| large | 20 | NOT_COMPUTED | NOT_COMPUTED | NOT_COMPUTED | NOT_COMPUTED | NOT_CALIBRATED |
| large | 60 | NOT_COMPUTED | NOT_COMPUTED | NOT_COMPUTED | NOT_COMPUTED | NOT_CALIBRATED |
| large | 120 | NOT_COMPUTED | NOT_COMPUTED | NOT_COMPUTED | NOT_COMPUTED | NOT_CALIBRATED |
| Profile | Horizon | Expected excess [MODEL] | 95% interval | P(excess>0) | 5% quantile | Calibration |
|---|---|---|---|---|---|---|
| broad | 1 | NOT_COMPUTED | NOT_COMPUTED | NOT_COMPUTED | NOT_COMPUTED | NOT_CALIBRATED |
| broad | 5 | NOT_COMPUTED | NOT_COMPUTED | NOT_COMPUTED | NOT_COMPUTED | NOT_CALIBRATED |
| broad | 20 | NOT_COMPUTED | NOT_COMPUTED | NOT_COMPUTED | NOT_COMPUTED | NOT_CALIBRATED |
| broad | 60 | NOT_COMPUTED | NOT_COMPUTED | NOT_COMPUTED | NOT_COMPUTED | NOT_CALIBRATED |
| broad | 120 | NOT_COMPUTED | NOT_COMPUTED | NOT_COMPUTED | NOT_COMPUTED | NOT_CALIBRATED |

Fields stay NOT_COMPUTED unless the ensemble's deciles were monotonic both in walk-forward and holdout.

## G. Risk and implementation

- Bot limits imported read-only [OBS]: position 9.00% of account (≈ $600), stop-loss 8.00%, long-only True, leverage False, account ≈ $6662.
- Cost model [MODEL]: FX 0.15% per side + Corwin-Schultz half-spread + 0.7·σ·√(order/ADV) impact at the bot's order size.
- Bot's current demo holdings: no open positions.
- Optimiser proposal: NO_SIGNAL — no validated LONG signals today; nothing to optimise

VaR / ES backtest since 2010 [CALC] (Kupiec = right number of breaches; Christoffersen = breaches not clustered; ES t > 2 = shortfall underestimated):

| Series | Model | 95% breaches | Kupiec p | 99% breaches | Kupiec p | Christoffersen p | ES t | Passes |
|---|---|---|---|---|---|---|---|---|
| SPY | historical | 5.09% | 0.789 | 1.47% | 0.004 | 0.000 | 2.31 | — |
| SPY | ewma | 5.75% | 0.028 | 2.37% | 0.000 | 0.012 | 6.58 | — |
| SPY | garch_t | 5.49% | 0.148 | 1.33% | 0.043 | 0.007 | 1.51 | — |
| SPY | factor | 4.97% | 0.920 | 2.33% | 0.000 | 0.000 | 6.32 | — |
| EW_large_cap_portfolio | historical | 5.14% | 0.678 | 1.42% | 0.010 | 0.000 | 2.51 | — |
| EW_large_cap_portfolio | ewma | 5.80% | 0.020 | 2.20% | 0.000 | 0.005 | 6.31 | — |
| EW_large_cap_portfolio | garch_t | 5.39% | 0.299 | 1.21% | 0.238 | 0.001 | 0.96 | — |
| EW_large_cap_portfolio | factor | 4.94% | 0.849 | 2.10% | 0.000 | 0.000 | 6.44 | — |

40-stock sample pass rate: historical 2.50%, ewma 0.00%, garch_t 60.00%. **Winner: garch_t** → per-name volatility in signals and the bot's VaR-capped sizing (1-day 99% VaR ≤ 1% of the account).
- Beta forecasts 2015+ [CALC]: MSE rolling-252 0.541, Blume 0.520, Kalman 0.693 (Kalman vs rolling t=-9.07) → **blume** used for signal betas.
- Tail dependence, t-copula [CALC] (reference basket: top 20 of tonight's forecasts (not holdings)): mean lower-tail dependence with SPY 0.096, mean pairwise 0.052; t-copula preferred to Gaussian for 90.00% of names.
- Factor risk model [MODEL]: Σ = BFB′ + D with 20 factors (market, 7 styles, sectors); EWMA F (126d half-life) shrunk 0.28 toward diagonal.
- GARCH(1,1) volatility is computed per name on request (risk_model.garch_vol); signals use EWMA (λ≈0.94) volatility.

## H. Trading-bot signal decision

**NO_SIGNAL for all 2931 entries.** Model gates — `large`: UNVALIDATED; `broad`: UNVALIDATED. Each entry's `signal.reason` lists every failed check.

**Paper trading (Trading 212 demo) [CALC]:** 0 closed signal trades, open none, 0 entries skipped by the pre-trade price check; forward record 4 nightly forecast dates. Live-review recommendation: **NOT_READY** (Per current recipe/model: >= 63 sessions and >= 20 closed demo trades with mean net excess > 0 and t >= 2.0; matching 20d forward IC > 0 with HAC t >= 2.0, >= 63 scored dates and >= 3 positive-net shadow cohorts. Never automatic.).

## I. Screener — top candidates

No security passed validation, so no ranked list is issued. Showing unsupported rankings would violate the spec.

## Sector modules and catalysts

- Energy/commodities [CALC]: crude–gasoline Engle-Granger p 0.0022, Johansen rank 3, crack half-life 2.2 months; EIA inventory-surprise effect on next-week WTI t=-1.72 (discovery) / 1.19 (holdout). energy: equity beta 0.99, wti loading 0.32; materials_metals: equity beta 0.81, copper loading 0.30.
- Commodity seasonality significant in discovery and ≥2/3 sign agreement in holdout: natgas, rbob (RBOB continuous series includes seasonal-grade rolls).
- Macro surprises vs ARIMA forecast from earlier first releases (spec 4D) [MODEL]: UNRATE: NOT_COMPUTED: verified initial releases unavailable; PAYEMS: NOT_COMPUTED: verified initial releases unavailable; CPIAUCSL: NOT_COMPUTED: verified initial releases unavailable; INDPRO: NOT_COMPUTED: verified initial releases unavailable; ICSA: NOT_COMPUTED: verified initial releases unavailable; IPG3344S: NOT_COMPUTED: verified initial releases unavailable.
- Biotech [CALC]: 680 issuers; 25 with < 1 year cash runway; registry cohorts PHASE1: 16.45% terminated/withdrawn (n=7239), PHASE2: 24.59% terminated/withdrawn (n=6186), PHASE3: 14.58% terminated/withdrawn (n=5625). Clinical success probabilities: NOT_COMPUTED (licensed outcome data required).
- Catalyst calendar next 90 days [OBS/UNVER]: 1290 events — earnings 1065, trial_primary_completion 195, eia_wpsr 13, macro_release 13, fomc 2, fda_adcomm 2. Not computed: NOT_COMPUTED: bls.gov blocks scripted access; dates not verified; NOT_COMPUTED: no machine-readable OPEC+ schedule; NOT_COMPUTED: usda.gov blocks scripted access; NOT_COMPUTED: DEF 14A meeting dates not parsed.
- Historical responses [CALC]: earnings median |abnormal| 2.74% (p90 11.43%), drift d2–20 after up/down reactions 0.40% / 0.26%; FOMC days |SPY| ×1.29 normal; FDA approvals 2-day abnormal mean 0.26% (n=575).

## J. Limitations and reproduction

- Models: ridge, elastic net, Huber, LightGBM, GAM (splines on 10 pre-specified features), hierarchical sector-pooled ridge, isotonic-calibrated logistic (P(excess > cost)), factor-residual ridge; ensemble members chosen each year from earlier walk-forward ICs only. Hyper-parameters fixed, not tuned on the holdout.
- Validation: yearly expanding walk-forward with purge (h+1) and 5-session embargo; 2-year holdout. The large-cap holdout was already evaluated by research v1; every run is appended to `ledger/research_runs.jsonl`, and paper trading is the only untouched test left.
- Known biases/gaps: no delisted-company prices (vendor survivorship), 321 removed S&P names without current tickers excluded; no point-in-time analyst estimate history (only forward consensus in the calendar); no options/IV data; 13F CUSIP mapping by issuer name; ALFRED vintages twice-monthly; ClinicalTrials records are edited in place; continuous futures include roll gaps.
- Section 13 B's per-feature observed-value table is held in the JSON (`key_drivers`) rather than printed for every security.
- Versions: feature set fs-2.1; run ids `20261020T084512Z-large-019cad03683b`, `20261020T085623Z-broad-3faa10e07a3c`; schema `schema/signal_v2.schema.json`.
- Next tests: 3+ months of paper trading with the bot reading LONG signals only; licensed estimate-revision and options data; delisted-price data (CRSP/Norgate).
