# Form 4 purchase study

The study is a separate research command. It writes only to the explicit output path. It uses the v2 preregistration and does not publish bot signals.

## Inputs

- Import the historical security, membership, adjusted price and terminal payout bundle as described in [HISTORICAL_DATA.md](HISTORICAL_DATA.md). Supply the SPY price file, SEC SIC mapping and company facts in their usual Mosaic locations.
- Use the quarterly SEC non-derivative transaction data and the recent XML collection. The recent collector saves the SEC submission acceptance time alongside its accession. To enrich older data, supply `--acceptances PATH` with CSV columns `accession_number,accepted_at_utc`. Use EDGAR Submissions metadata, including older submission files where needed, and preserve the timezone. Retain the source export alongside the CSV. [SEC Submissions API](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
- Import certified estimate vintages using the existing estimate importer. Current consensus and historical reported EPS are not substitutes.
- Capture the data directory at a stable point before running the study. Inputs and code are hashed. The runner checks for changes to historical inputs during evaluation.

## Historical evaluation

```powershell
python insider_study.py historical --acceptances C:\data\acceptances.csv --out data/research/insider-v2-run-001
```

Use a fresh output directory for each attempt. Registration is written before evaluation. Reports cover the primary exact-timestamp study and separate date-only and explicit-non-plan sensitivity runs. Parquet files hold the event scores, matched pairs, common net return series and per-scenario fill ledgers.

A filing may first contribute at the next regular session open on a later New York date. Exact timestamp coverage, unknown 10b5-1 status and amendment counts are reported separately. Amendments are retained in the source and counted in the audit, but they do not create purchases without a verified transaction link. If several share classes of an issuer meet the eligibility screen on the same decision, the study skips the event and counts the ambiguity.

The study estimates beta-adjusted returns using a fixed entry hedge and fixed order sleeves. It models entry and exit costs from pre-entry estimates. A verified terminal payout settles in cash and incurs no simulated exit trade. Portfolio series are daily P&L divided by fixed reference capital, with zero P&L on cash days. DSR and PBO use common five-session blocks and are development diagnostics. Coverage of the 95% prediction intervals is empirical because financial residuals are dependent. The historical run remains unvalidated even if point estimates are positive.

## Prospective paper record

Supply a trusted regular-session exchange calendar CSV with a `date` column, including at least the previous session and the scoring date. Keep the calendar and input snapshots for audit. Run the scoring command on each session before 09:30 New York time with completed price data through the preceding session. The scorer records empty sessions as well as candidate scores. Use one dedicated records directory per frozen recipe.

```powershell
python insider_study.py freeze --acceptances C:\data\acceptances.csv --out data/research/insider-v2-run-001/frozen.json
python insider_study.py score --recipe data/research/insider-v2-run-001/frozen.json --sessions C:\data\sessions.csv --acceptances C:\data\acceptances.csv --out data/research/insider-v2-run-001/forward
python insider_study.py review --recipe data/research/insider-v2-run-001/frozen.json --records data/research/insider-v2-run-001/forward --acceptances C:\data\acceptances.csv --out data/research/insider-v2-run-001/review.json
```

Files use exclusive creation to prevent accidental overwrites. Retain them in a separately controlled archive if independent verification of prospective timestamps is required. Review uses settled observations and reports missing sessions and outstanding forward criteria. User review is required before any change in trading policy.


To create the sidecar from the SEC bulk submissions archive, keep the
downloaded ZIP and run `python insider_acceptances.py submissions.zip --out
acceptances.csv` after collecting the local quarterly filing data. The command
matches recent and archived submission records by accession and reports
matched and unmatched counts. Unmatched filings stay date-only.

If the bulk archive is unavailable, use the SEC submissions API directly.
For a bounded coverage check, run
`python insider_acceptances.py --sec-api --issuer-limit 25 --out acceptances.csv`.
The limit selects issuers with the most distinct purchase and sale accessions.
Omit it to consider every issuer present in the local filing snapshot. API
mode checks historical submission pages as needed and reports issuer errors.
Use a fresh output path for each snapshot.

The bulk archive can contain different timestamps for the same accession in
different filer records. The importer excludes those accessions from the
timestamped sidecar and writes the conflicting timestamps to a companion
`.conflicts.json` file. Resolve them against the original filing before use.

Bulk matching uses the issuer CIK on the transaction and searches only that
issuer's recent and archived submission pages. Other reporting owners may
list the same accession with different timestamps. Conflicting records
within the eligible issuer pages remain excluded and logged.

The equity panel loads `data/processed/insider_acceptances.csv` if present.
Promote an issuer-matched sidecar to that path to enable the fs-2.5
availability rules. The source filing date remains available for reporting.
Feature windows and prior insider history use the New York acceptance date
where verified and the filing date otherwise. The standalone exact-timestamp
study still excludes date-only rows from its primary run.

The fs-2.6 panel leaves insider features missing if a required quarterly
file is absent from the 90-, 180-, or 365-day window. File presence is a
minimum check and does not certify issuer coverage. Plan purchases are
excluded from the discretionary aggregates, and purchase intensity requires
positive reported holdings.
