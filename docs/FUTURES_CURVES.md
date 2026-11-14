# Individual futures contracts

For a point-in-time settlement curve, import a CSV with commodity, contract_id,
session, expiry, settlement, available_at_utc, currency, price_unit, and source.
Session and expiry are exchange calendar dates. Availability is the actual
publication or first-observed time with an explicit offset. Revised settlements
require their own availability timestamps.

    python futures_curves.py contracts.csv
    python commodity_research.py --curve-session 2026-10-26 --as-of 2026-10-27T00:00:00Z

The report uses the nearest two visible unexpired expiries from the requested
session. It reports front minus second in the contract's price unit and an
annualized log slope when both prices are positive. This slope is a curve
measure. A realized roll strategy requires contract-level fills, roll rules,
and trading costs. Missing contracts and ambiguous comparisons remain
NOT_COMPUTED.
