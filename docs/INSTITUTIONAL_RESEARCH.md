# Institutional research comparison and execution evidence

This implements public research principles, not a hedge fund's proprietary model
or a claim of institutional performance. Man Numeric describes fundamental
reasoning, combined equity signals, risk control and implementation. AQR motivates
quality signals and checking cost assumptions against real executions. Neither
source specifies Mosaic's approval thresholds.

Sources: [Man Numeric](https://www.man.com/numeric),
[AQR Quality Minus Junk](https://www.aqr.com/insights/research/working-paper/quality-minus-junk),
[AQR Trading Costs](https://www.aqr.com/insights/research/working-paper/trading-costs).

## Fixed value–quality–momentum benchmark

`institutional_research.py` defines recipe `vqm-1`. It percentile-ranks signed
features within decision date and sector, averages within each theme, then gives
each theme one third of the score:

| Theme | Components |
| --- | --- |
| Value | earnings yield, book-to-market, free-cash-flow yield |
| Quality | gross profitability, ROA, ROIC, negative accruals, negative leverage |
| Momentum | 12–1 and 6–1 month momentum |

All ten components are required; no missing-component neutral imputation.
The recipe is fixed, with no holdout-fitted signs or weights. It uses Mosaic's
existing point-in-time panel and cannot repair errors in the underlying data.

Each full `research.py large` / `research.py broad` run now compares the primary
horizon ensemble's OOS scores with VQM on the same eligible rows, calendar, top
fraction and name costs. Walk-forward and holdout are reported separately, with
paired net-return differences and HAC uncertainty where sample size permits.
Both receive 0x, 0.5x, 1x and 2x cost scenarios.

Outputs in `reports/`:

- `institutional_comparison_<profile>.json`: methodology, sample/exclusion counts,
  historical-coverage check and paired results (also embedded in research JSON).
- `institutional_periods_<profile>.parquet`: each strategy's period returns/costs.
- `institutional_holdings_<profile>.parquet`: actual simulated selections,
  scores, target weights and outcomes for each assessed period.
- `institutional_fills_<profile>.parquet`: modeled cash flows and fees for
  rebalances and liquidation under drift accounting.
- `oos_scores_<profile>_h20.parquet`: genuine OOS ensemble scores for replay.

The comparison subset requires complete VQM features; it can differ from the
original full-model universe. A full-model advantage does not isolate which
extra signal caused it. Selected names are simulated decisions, not broker fills.
Missing selected outcomes stop the measurable NAV chain. Both strategies are
compared on the common assessed prefix, and the report is marked
`INCOMPLETE_OUTCOMES`; no lower-ranked survivor or zero return is substituted.

Full research runs use self-financing drift accounting: adjusted next-open total
returns mark holdings between decisions, costs apply to every dollar adjustment,
including continuing names, and target amounts are solved after fees. Cash gaps
and sample ends incur modeled liquidation fees. `cost` is gross-minus-net return
drag; `transaction_cost` is paid fees divided by starting NAV. Records use initial
NAV units and adjusted-price units, not raw broker share quantities. Estimated
rates remain assumptions, including last-known rates used at liquidation. Dividend
effects are implicit in adjusted total returns. This is not an Alpaca long/short
execution backtest or a reconstruction of actual broker fills.

A fixed lower-turnover diagnostic retains existing names within the top 40%, up
to the original name count. It is reported separately and never chooses production
rules using holdout results. Cost scenarios rerun the fee/NAV accounting, rather
than subtracting a scaled fee from an unchanged path.

All research model/single-feature strategies now anchor rebalance dates BEFORE
filtering scores/labels. Different first-score dates no longer shift calendar
phase. SPA/Reality Check still require complete common periods with no zero
imputation. Matched random baselines use saved ensemble dates and fail if their
period count differs. New random paths use the same drift/fee/liquidation engine;
membership retention targets approximate the model's average turnover. Realized
dollar turnover remains path-dependent. Unknown eligible outcomes produce
`INSUFFICIENT_DATA`, not a survivor-only random comparison.

Replay genuine saved OOS predictions, never predictions from final weights fitted
on the holdout:

```bash
python institutional_research.py --profile large \
  --panel data/processed/panel_large.parquet \
  --scores reports/oos_scores_large_h20.parquet \
  --benchmark-prices data/raw/prices/SPY.parquet \
  --holdout-start 2024-10-21 --order-usd 600 --output-dir reports/replay
```

Use the ORIGINAL run's holdout boundary, horizon, top fraction and order size.
Replay records input hashes and checks sample labels against the boundary. The
caller must supply the original panel and genuine OOS predictions.
Omitting `--benchmark-prices` explicitly produces the legacy approximation; that
compatibility mode is not comparable to the new self-financing research results.

## Execution-cost evidence

`execution_costs.py` audits normalized CSV fills against estimates quoted BEFORE
submission. It performs no broker calls and changes no cost setting. An order
acknowledgement is not a fill; the current bot order log alone is insufficient.
Obtain broker fill records plus retained pre-order quotes/model estimates.
Missing quotes cannot be reconstructed from later quotes or daily closes.

Required columns (one row per actual fill):

```text
fill_id,order_id,symbol,side,quantity,fill_price,arrival_mid,arrival_at,submitted_at,filled_at,fees_usd,fx_cost_usd,modeled_cost_bps,environment
```

Use `buy`/`sell`, `paper`/`live`, explicitly zoned ISO timestamps, positive prices
and quantities, and nonnegative costs. Allocate fees and FX ONCE per fill, in
USD. `modeled_cost_bps` is the complete one-way estimate for that account, including
applicable fees/FX. Do not add spread/impact to observed shortfall a second time.

Arrival midpoint must be timestamped at/before submission and at most 60 seconds
earlier. Partial fills must share one order's arrival benchmark and model quote.
Duplicate fill IDs, inconsistent benchmarks, stale/future references and fills
before submission are rejected.

Observed cost = signed arrival-to-fill shortfall + fees + FX, divided by arrival
notional. Buy/sell signs are opposite; favorable fills may have negative shortfall.
Aggregates are notional-weighted and separated by paper/live environment. Partial
fills do not inflate independent order counts. The audit excludes unfilled-order
opportunity costs and stock-borrow expense.

```bash
python execution_costs.py /private/path/fills.csv \
  --output /private/path/cost_audit.json --min-orders 100 --min-days 20
```

Thresholds are configurable research choices, not universal hedge-fund rules.
Paper fills never qualify as live evidence. Sufficient live samples do not
automatically recalibrate costs: review order size, liquidity, market conditions,
venue/feed and account currency before adopting a new recipe. No executions or
quotes are invented when the bot has not traded.

## Validation and rollout

Historical-universe, data-audit, recipe, historical and forward gates remain
binding. Changed sources require revalidation and a new forward recipe. Reused
holdouts are descriptive, not fresh confirmation. Missing delisted-security data
still needs a verified source. This work does not establish alpha or enable
trading; push/deployment and a fresh historical run are separate operations.
