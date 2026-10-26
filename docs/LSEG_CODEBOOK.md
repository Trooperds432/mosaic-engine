# LSEG CodeBook research data

Run `lseg_codebook.py` in an already signed-in CodeBook Python notebook. It uses
the hosted session and contains no credentials. The default is a bounded
entitlement/coverage sample, not a full universe. Edit `default_requests()` or
pass an explicit request list to `export(requests=...)` to expand it.

The exporter requests:

| Specification | Dataset | Treatment |
|---|---|---|
| 0 / 2.2 | S&P 500, Nasdaq-100 and Russell joiners/leavers | Actual dated events only; not complete starting membership |
| 2.2 | RIC, ISIN and organisation PermID | Current reference data; not historical identifier mapping |
| 2.2 | Dividend announcement/ex/pay dates, currency and unadjusted amount | Corporate-action reconciliation evidence; not automatically applied to prices |
| 3.3 / 4A / 5 | EPS/revenue consensus history, range and forward EPS dispersion | Retained as retrieved vendor history; not certified first-release vintages |
| 4A | Forward capital-expenditure consensus | Observed snapshot; not realised hyperscaler expenditure |
| 3.3 / 11 | Dated company guidance and document type | Archived source text; no automatic merger/guidance event adjustment |
| 4D | Bank customer deposits as reported | Observed snapshot; broader capital, margin and loan-mix fields remain missing |
| 4C / 4E | Individual WTI, gas and gold contracts | Access test; failed contracts stay unavailable, no continuous-price substitution |

Download the resulting ZIP from CodeBook's file browser, then run:

```sh
python lseg_data.py /path/to/mosaic_lseg_TIMESTAMP.zip
```

The importer verifies content hashes, identities, dimensions, timestamps and safe
archive paths before creating immutable content-addressed raw archives and
processed Parquet files. Reimporting an identical archive is idempotent.
`reports/lseg_coverage.json` describes the latest imported bundle, including
failures and field-level non-null counts. Every previous bundle stays archived.
The dependency registry fingerprints both the archives and transformed data.

## Full-universe capture (repeat on a schedule)

Upload `lseg_codebook.py` and `lseg_universe.json` (2,352 RICs built from the broad profile) to the CodeBook file
browser, then run one cell:

```python
import json
from lseg_codebook import export, full_requests
export(requests=full_requests(json.load(open("lseg_universe.json"))))
```

About 406 bounded requests: identifiers and FY1 consensus snapshots (batches of 200), monthly FY1 EPS/revenue
consensus history since 2012 with each value's calculation date (batches of 10), S&P 500 / Nasdaq-100 joiners and
leavers since 2009, and quarter-end constituent lists since 2009. A batch LSEG rejects is recorded as UNAVAILABLE;
re-run it with a smaller `history_batch`. Download the ZIP and import it with `lseg_data.py` as above. Each capture is
an observed snapshot; repeated weekly captures build a genuinely point-in-time estimate archive.

## Research use and availability

```python
from lseg_data import observations_asof
known = observations_asof("2026-10-27T16:00:00Z", "consensus_snapshot")
```

This returns a long table with dataset, RIC, source row, field, original value,
observation date when present, retrieval/availability timestamps and archive hash.
Numeric fields retain vendor labels and values. Units, currency, fiscal-period
alignment and permanent-identifier mapping must be reviewed before calculations.
Some requested fields can be unavailable even when a request succeeds: inspect
requested fields, returned columns and non-null counts together.

All retrieved data have availability at acquisition. The historical period end,
guidance date, dividend announcement date or estimate calculation date does not
silently become a verified first-available timestamp. No historical alpha feature
is populated from this archive before retrieval. Repeated captures can build a
forward archive; vendor history requires a separate documented PIT review before
it can support earlier performance claims. Current reference identifiers are not
backdated onto historical constituents.

The new datasets are implemented in the research store and as-of query API.
They are deliberately not automatically inserted into `features.FEATURES`,
production price files, trained models or deployment gates. Neither the minimum
snapshot history requirement nor the research validation controls are relaxed.

## Remaining sources

This connector does not create missing clinical protocol/outcome vintages,
clinical success cohorts, reimbursement/patent events, foundry utilisation,
wafer capacity, equipment bookings, memory prices, supplier/customer revenue
links, NOAA/weather/NDVI/soil moisture, USDA/USGS balances, LNG flows/outages,
historical borrow data, full derivatives Greeks, contemporaneous macro consensus
or original macro releases. These require independently identified datasets,
entitlements and timestamp validation. The general CodeBook API is not proof that
these specific datasets are available in this account.

No order is placed, no live settings are changed, and no predictive performance
is claimed from downloading these observations.

## Verified acquisition: 26 October 2026

The bounded CodeBook run imported 15 datasets and 4,408 non-missing field values:
six-company identifiers and current consensus, six EPS/revenue consensus histories,
36 guidance records, 163 dividend records, three bank customer-deposit observations,
and four dated index-change datasets. The equity sample is AAPL, MSFT, NVDA, JPM,
XOM and PFE; the bank sample is JPM, BAC and Citi.

S&P 500 returned 137 additions and 139 removals; Nasdaq-100 returned 68 additions
and 71 removals for the requested 2020-to-23-October-2026 interval. These are vendor
change records, not a reconstructed complete membership table. Russell returned
undated placeholder records and was rejected. All six selected individual-futures
requests failed; their cause is recorded as request/content unavailable, not
assumed to be a subscription denial. Customer-deposit values returned no reporting
date, so only acquisition-time visibility is supported.

The actual imported store returned zero rows for a 2020 decision and 4,408 rows
for a decision after acquisition. Both new safety tests passed; the full suite
ran 102 tests with 101 passing and one platform-related skip. Existing audits
must be regenerated because the dependency fingerprint now includes this source.

Local archive SHA-256:
`8819d953eb147f529c336ec140062618d743ba8970670c3b8044832ba8177118`.
Licensed CSV/Parquet data stay under ignored `data/`; no upload, push or deployment
was performed.
