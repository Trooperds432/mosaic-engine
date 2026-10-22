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
