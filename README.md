# Mosaic research engine (spec: Institutional Quantamental Mosaic Theory v2.0)

Local integrity/validation implementation: see [IMPLEMENTATION.md](IMPLEMENTATION.md) for changed behavior,
tests, migration requirements and remaining specification gaps. Model/data changes require revalidation;
existing reports are historical artifacts, not results from the updated implementation.

Research, forecasting and validation only. It never places orders, changes bot limits or enables live trading.

## Bot interface

`signals/latest.json` is replaced atomically after each daily run:

```
{ "schema_version": "2.0", "as_of_utc", "research_run_id", "model_gate_primary", "insider_coverage_through",
  "n_evaluated", "n_long", "signals": [ <schema/signal_v2.schema.json objects> ] }
```

A consumer should act only on entries where `signal.direction == "LONG"` and `expires_at_utc` is in the
future, and it should still run its own sizing, risk and kill-switch checks. Units: decimal returns over
`forecast.horizon_sessions`, in excess of SPY, net figures after the 0.46% round-trip cost model. Null means
not estimated, never zero. Every run is also written to `signals/log/` and never overwritten.

Since 2026-10-20 `t212-bot` (demo account only) trades this file: it buys LONG signals that are PAPER_TEST_ELIGIBLE or
LIVE_REVIEW_ELIGIBLE, unexpired, liquidity/data-quality approved and with positive net expected return; it holds each for
the signal's horizon (8% stop, 9% per name, max 11 names). Stale (> 4 days) or invalid files block new entries.

## Jobs

| What | How | When |
|---|---|---|
| Data refresh + signals + report | `run_daily.sh` (`mosaic-engine.service`) | `mosaic-engine.timer`, Mon–Fri 22:40 UK; fundamentals fully refreshed on Fridays |
| Re-validation (panel, tests, model.json, event study) | `run_research.sh` | manual only; each run spends the holdout and is appended to `ledger/research_runs.jsonl` |

## Layout

`data_*.py` ingestion · `features.py` point-in-time panel · `research.py` IC/HAC/FDR, purged walk-forward,
holdout, PSR/DSR/PBO, gate · `event_study.py` insider CARs · `signals.py` export + JSON-Schema validation ·
`report.py` report sections A–J (`reports/report_latest.md`) · `config.py` all thresholds.

## Insider event model (`insider_model.py`)

Pre-registered in `prereg/insider_event_v1.json` (hash logged in `ledger/preregistrations.log`): CEO/CFO or
cluster open-market purchase → entry at the next open → hold 5 sessions. Its signals use `model_id:
"insider-event"` and appear in `latest.json` only on days with qualifying filings, so a ticker can have one
cross-sectional and one insider-event entry. Its returns are **beta-adjusted abnormal returns** (stock − β·SPY),
not plain SPY-excess. The frozen rule cannot be re-tuned: any change needs a new `insider_event_v2.json`.
Earnings dates for the controls come from 8-K item 2.02 (`data_8k.py`, refreshed Fridays).

## v2 additions (2026-10-19)

| Spec section | Module | What it does |
|---|---|---|
| 0 universe / limits | `data_universe.py`, `config.PROFILES`, `risk_limits.py` | all NYSE/Nasdaq common stocks screened point-in-time; profiles `large` ($10bn) and `broad` ($300m, Russell 2000 range); bot limits imported read-only |
| 2 data | `data_macro.py`, `data_alt.py`, `data_13f.py`, `data_insider.py`, `data_8k.py` | FRED + ALFRED first-release vintages, EIA, CFTC COT, FINRA short interest, futures, ClinicalTrials.gov, openFDA, 13F, Form 4 footnotes + derivative table |
| 2.2 PIT tests | `audit.py` | truncation-invariance (look-ahead), label alignment, duplicates, timestamps (blocking); survivorship, restatement, MNAR, stale, corporate actions (diagnostic) |
| 3.2 insider | `features.py` | role-split, purchase intensity, conviction, Cohen-Malloy-Pomorski routine/opportunistic, plan-sale share |
| 4A-4E sector | `features.py`, `commodity_research.py`, `biotech.py` | semis cycle, biotech runway/trials/FDA, energy oil/inventory/crack/COT, bank NIM/deposits/provisions, commodity tilts; commodity seasonality/momentum/COT/cointegration |
| 5-9 research | `research.py` | raw/industry/style-neutral ICs, bootstrap CIs, FDR, 8-model suite + leakage-free ensemble, conformal intervals, calibrated P(excess>cost), filtered-HMM regimes, robustness, cost/parameter sensitivity, White RC / Hansen SPA, DSR/PBO/MinTRL, factor-adjusted alpha |
| 10 risk | `risk_model.py`, `portfolio.py` | BFB'+D factor model, EWMA/GARCH vol, stress replays, ES, liquidity/gap risk, cvxpy optimiser under bot limits (proposal only) |
| 11 catalysts | `catalysts.py` | 90-day calendar (earnings, FDA adcomm, FOMC, BEA, EIA, trial completions) + historical responses |
| 12.3 event-driven | `intraday_watch.py` (`mosaic-intraday.service`) | polls EDGAR live Form 4 feed every 10 min, refreshes insider-event signals |
| 13/14 | `report.py`, `signals.py`, `regime.py` | evidence-tagged A-J report; per-signal regime field in the schema |

## Freshness (2026-10-20)

Continuous: `t212-bot` (orders 9:35-15:50 ET) and `mosaic-intraday` (Form 4s every 10 min, Mon-Fri 06:00-22:00 ET).
Nightly (Mon-Fri 22:40 UK): prices, insider data, macro, futures/COT, and `data_updates.py`, which reads SEC's daily
form index and re-downloads facts/8-Ks only for companies that filed a 10-Q/10-K/8-K in the last 5 business days,
so filings reach the models the evening they are filed. Fridays add the full fundamentals/8-K/13F/short-interest/trial
refresh. Models are applied nightly; retraining (`run_research.sh`) stays manual because each retrain re-spends the holdout.

## Paper trading and real-time execution safeguards (2026-10-20)

- `paper_trading.py` (nightly): forward record of every forecast (`processed/forecast_history.parquet`), forward IC and a
  cost-adjusted top-quintile shadow book per profile/horizon, and the executed demo book (bot orders matched to
  Trading 212 fills: price, FX fee, slippage, return vs SPY). Output `reports/paper_trading.json`; promotion rule
  (>= 63 sessions, >= 20 closed trades, mean net > 0, t >= 2, forward IC > 0) only *recommends* live review.
- `t212-bot`: Finnhub WebSocket real-time stop-loss on held names (confirmed by a second live quote), pre-trade check
  (fresh live quote within 3% of the as-of close, else SKIP), 15-minute cycle kept as fallback.

## Options and analyst-estimate data (2026-10-20)

- Market-level, usable now: `^VIX9D/^VIX3M/^VIX6M/^VVIX/^SKEW` (Yahoo, 10y+) -> `opt_mkt_term_tilt`, `opt_mkt_skew_tilt`.
- `data_collect.py` via `mosaic-collect.timer` (daily 21:20 UK):
  - Alpha Vantage EARNINGS backfill, 24 companies/day (free key), largest first -> `sue_consensus` (surprise vs consensus).
  - Cboe delayed options snapshots (weekdays) -> `opt_iv_rv`, `opt_term_slope`, `opt_skew_rr`, `opt_pc_ratio`.
  - Nasdaq analyst snapshots (weekdays) -> `est_rev_1m`, `est_dispersion`; Finnhub recommendation trends (Fridays).
- Snapshot features are excluded from models until their history spans 3 years (`features.SNAPSHOT_MIN_YEARS`).
- Yahoo's options/quoteSummary endpoints block this server (429), hence Cboe/Nasdaq. An Alpha Vantage premium key would
  unlock historical options (2008+) and finish the earnings backfill in ~1.5 h.

## Regime research (2026-10-20): `regimes.py`, run per profile in `run_research.sh`

Markov-switching regression (2-regime MarkovRegression on weekly SPY, yearly refit, filtered probabilities; ridge
weights per regime blended by the current probability), PELT change-point detection on factor returns and feature ICs,
and factor returns by regime (filtered HMM + observable). Output `reports/regime_research_<profile>.json`, report
section E. First run: regime-specific weights differ (corr 0.35 large / 0.52 broad) but do not improve out-of-sample
IC significantly (WF t 0.7 / 1.1, holdout t 1.1 / 1.6) -> not adopted; many feature ICs break around Feb 2021.

## Econometrics additions from the user's notebooks (2026-10-20)

| Module | What | Result / use |
|---|---|---|
| `var_backtest.py` | Historical, EWMA, GARCH(1,1)-t, factor VaR/ES; Kupiec, Christoffersen, McNeil-Frey | GARCH-t wins (60% of stocks pass vs 0-3%); nightly `--forecast` -> `processed/vol_forecasts.parquet`, `signals/risk_latest.json`; signals' `forecast_volatility`; bot caps each position at 1-day 99% VaR <= 1% of account (`BOT_POSITION_VAR_LIMIT`) |
| `kalman_beta.py` | Vectorised Kalman betas vs rolling-252 vs Blume, drift chosen on 2009-14 | Kalman 28% worse, Blume 4% better (t=3.2) -> signals report Blume-adjusted beta |
| `macro_surprise.py` | Auto-ARIMA "expected" value from earlier first releases; surprise z | Beats random walk on all 6 series; features `growth_surprise_tilt`, `inflation_surprise_tilt` (next re-validation) |
| `screening.py` | k-NN mutual information, permutation null, vol-scaled returns | Diagnostic only |
| `risk_model.t_copula` | t-copula tail dependence vs SPY and pairwise | In portfolio risk report |
| `regimes.hmm_k` | 2- vs 3-state HMM | 3 states better OOS likelihood 16/19 years -> live regime uses 3 states; no ranking gain |

## Known limitations and errata (2026-10-22, after an external code review)
Read every backtest number in `reports/` with these qualifications.

- **Survivorship bias.** The universe starts from current SEC/exchange listings plus historical S&P 500 members, and
  Yahoo has no prices for most delisted stocks. Historical samples are therefore biased towards survivors; results
  are optimistic by an unknown amount until delisting returns and historical listings (e.g. CRSP / Norgate) are added.
- **The 2-year holdout is not independent.** It has been inspected in several research runs (ledger/research_runs.jsonl)
  and was used in model-selection diagnostics. It must not be quoted as an untouched test. Only the forward record
  (paper_trading.py, from 2026-10-19) is independent; it alone sets the gate.
- **Calibration** used walk-forward *and* holdout outcomes up to 2026-10-22; from the next re-validation the exported
  expected returns use walk-forward outcomes only (`research.py`).
- **Insider pre-registration timestamp.** `prereg/insider_event_v1.json` says `registered_utc 14:30`, but the run in
  ledger/research_runs.jsonl is stamped 14:14:27 and already carries the file's SHA-256 (ed1edd44…). The rule text was
  therefore fixed *before* the run, but the stated registration time is wrong, and the rule was written after the
  exploratory event study of the same day. Treat it as a rule frozen before its own test, not as a blind pre-registration.
  The file is left unchanged so the recorded hash still verifies.
- **Backtest execution.** Long-only backtests use equal weights reset each period (no weight drift), open-to-open
  returns and modelled (not observed) costs.

Fixes made in response to the review:
- the Trading 212 bot skips an entry when the VaR forecast is missing, stale, unreadable or non-positive (it used to fall
  back to full size);
- `signals.py` only accepts a data audit produced in the same run for the same dataset version (`provenance.py`), and
  a forward gate that is at most 4 days old and was produced for the current model recipe (`gates.py`);
- the collectors take their own lock (`.collect.lock`) instead of claiming the engine lock;
- session counts use the NYSE holiday calendar (`provenance.sessions_between`);
- a FRED error page no longer stops the nightly job (retry, then keep the cached copy of that series);
- the long/short per-name cap now scales both legs together, keeping the book beta-neutral (found by the new tests);
- `requirements.txt` (pinned), `tests/` (`python -m unittest discover -s tests`) and a GitHub Actions workflow.
