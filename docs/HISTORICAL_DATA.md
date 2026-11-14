# Historical data import

Run `python historical_data.py DIRECTORY` under the engine environment. This is
an explicit replacement of the historical bundle, not a merge with current listings.
Preserve a backup of the existing data directory before migration.

Supply these CSV files, with headers even when a table is empty:

| File | Columns |
| --- | --- |
| security_master.csv | security_id,cik,ticker,exchange,source |
| membership.csv | security_id,index,start,end,known_at,source |
| prices.csv | security_id,date,ticker,open,high,low,close,volume,adjclose,adj_open,market_cap,source |
| terminal.csv | security_id,last_trade_date,settlement_date,terminal_adjclose,source |

Use permanent alphanumeric/underscore security IDs, distinct for reused tickers
and share classes. CIK identifies an issuer, not a security. `market_cap` must be
security-specific. All prices and terminal payouts must share the same adjusted
price basis. A verified zero terminal payout is valid; an unknown payout is not zero.
Membership end is exclusive; eligibility begins only when both start and known_at
have been reached. Index names are SP500, NASDAQ100 and RUSSELL2000.

Also supply coverage.json with start/end dates, indices (all three names), and
boolean constituents_complete, delisted_prices_complete, terminal_events_complete,
identifiers_verified. These are provider attestations, not independent proof of
completeness. The audit additionally checks actual members, prices and ended series.
Retain the vendor methodology and source evidence for review. Structural validation
does not establish that no omitted security exists.

The importer archives the source by content hash and keeps security IDs in feature
and research files. Broker and external data boundaries translate to current tickers.
Unknown forward exit prices remain missing unless a verified terminal payout has
settled by the scoring date. No stale-price forward filling supplies returns.

The cross-sectional feature path supports shared issuer CIKs. The insider event
path currently rejects multiple securities per CIK until an explicit primary-security
mapping is supplied; it must not silently choose a share class.

## Estimate history

Run `python pit_estimates.py COMPLETE_EXPORT.csv`. Columns are security_id,
fiscal_period,consensus_eps,available_at_utc,observed_at_utc,timestamp_basis,source.
Use UTC timestamps. timestamp_basis is vendor_archive or observed_snapshot.
Observed snapshots cannot have availability earlier than observation. The import
replaces the active estimate table and archives the export; include the full history.
Revision controls compare the same fiscal period using information available before
the event decision. Missing history blocks the complete incremental-value test.

## Provider discovery

Credentials are read from the environment, then `/root/.hermes/.env`; no secrets
belong in code. Collectors are opt-in and do not run during ordinary research:

```
python provider_history.py alphavantage --day 2020-01-02 --state delisted
python provider_history.py finnhub --symbol ^GSPC
python provider_history.py exa --query "historical index constituent archive methodology"
```

Alpha Vantage collection shares the engine's 24-call daily reservation ledger with
earnings backfill. Other software outside this ledger can still consume quota.
Listing history is documented only after 2010-01-01 and supplies no guarantee of
delisted prices or terminal payouts. Finnhub access depends on entitlement; 403
fails explicitly. Exa provides discovery evidence, not certified numerical history.
Raw responses require identity and coverage review before the bundle import.

The coverage audit checks that actual prices reach the requested end date and
compares each eligible membership interval with benchmark sessions when the
benchmark file is available. Missing prices within those intervals and ended
series without a verified terminal outcome fail the audit. When benchmark data
are unavailable, the bundle check identifies its fallback calendar explicitly.
Observed security dates cannot establish that a session missing from the entire
export was covered.

The fs-2.3 equity panel can use an imported estimate archive via
`eps_revision_pit_31d`. It joins permanent security IDs from the active
historical bundle and uses only estimates available by midnight New York time
on the decision date. Both estimates must refer to the same fiscal period.
When several fiscal periods share the latest publication timestamp and no
unique target can be inferred, the revision stays missing. The research model
applies the existing minimum-history requirement to this feature.

For independent session reconciliation, include `sessions.csv` in the
vendor directory with columns `date,source`. Supply ordered, unique ISO
dates from an exchange schedule spanning the backtest window. The audit
compares the benchmark dates against this schedule and reports missing and
extra sessions. Use an independent schedule because deriving it from the
price files would conceal gaps shared by those files.

To prepare membership intervals from sourced constituent changes, use
`python historical_membership.py roster.csv changes.csv --start YYYY-MM-DD
--end YYYY-MM-DD --out membership.csv`. The starting roster requires
`security_id,index,known_at,source`. The change file additionally requires
`effective_date,action`, with action set to join or leave. Use permanent
security IDs and independently verified availability dates. The command
replays the supplied records and does not establish source completeness.

## LSEG CodeBook research export

Run `python lseg_bundle.py SOURCE_DIR OUTPUT_DIR` to build a staged S&P 500
bundle. Supply the month-end snapshots (`spx_members.csv`), issuer reference
data (`mosaic_hist_static.csv`), and OHLCV and capitalization export
(`mosaic_hist_prices.csv`). Request `TR.TotalReturn.CalcDate`,
`TR.TotalReturn.Date`, and `TR.TotalReturn` separately at daily frequency.
Save these as `mosaic_sp500_dated_returns.csv.gz` with columns
`Instrument,CalcDate,ReturnDate,Total Return`.

The converter joins by instrument and date, rejects stale dates and conflicting
duplicate observations, and stops each series when a dated total return is missing after the first
matched observation. A dated return without OHLCV contributes to compounding
and leaves the session unpriced. Conversion counts are
saved in `coverage.json`. This export covers monthly S&P 500 membership and
contains no verified terminal payouts. The resulting coverage attestations
remain false.
