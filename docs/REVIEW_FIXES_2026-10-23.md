# Engineering fixes following the d38c1c2 review

## Data identity

`feature_dependencies.py` is the authoritative registry for feature inputs,
archives and transformation sources. `provenance.dataset_version()` hashes their
contents, not only metadata. It includes text-similarity Parquet, filing-text
archives, `data_text.py`, macro surprises and original macro vintages. Register
future sources here. Creation/deletion and same-length edits with restored mtimes
invalidate old audits. Fingerprinting streams files in bounded memory; it reads
the complete registered data, so large archives increase runtime.

All new IDs invalidate older audit evidence. An input rewritten during its own
hashing raises an error. Scheduled jobs use the shared lock; source collectors
must finish before audits and signal generation.

## Independent profile evidence

`audit.py large` and `audit.py broad` write separate
`reports/data_audit_<profile>.json` files. Nightly and research wrappers run both
as appropriate. Consumers verify profile, dataset ID, timestamp and blocking
checks. The legacy `data_audit.json` remains a large-profile display artifact and
cannot approve broad. Insider events conservatively require all profile audits.
Intraday cross-sectional signals are checked individually; a missing broad audit
does not invalidate an otherwise matching large signal. Signals from an old
dataset are blocked pending regeneration, even if a new audit has passed.

## Per-security risk freshness

The Trading 212 entry-sizing path now calls the pure `risk_freshness` validator.
It checks the generated timestamp AND each name's source `as_of` session, requires
that session to have closed, rejects forecasts predating the signal, and rejects
missing/non-finite/non-positive VaR. Republished old forecasts cannot become fresh
merely because the enclosing JSON was regenerated. This applies to research
entries; exits remain separate. The bot file is changed in `deploy/` only until an
explicit deployment copies it into the running bot installation.

## Holdings, fees and validation

Full research now reconstructs asset total returns by adding the exact same
SPY next-open label return to each excess label. `holdings_simulation.py` maintains
drifting holdings and a cash ledger, solves post-fee allocations, charges all
value changes including retained names, and liquidates at cash gaps/sample ends.
Overlapping holding periods are rejected. Missing selected outcomes stop the NAV
chain and prevent research approval.

Ensemble runs write holdings and modeled fill ledgers for each 5/20-session
walk-forward/holdout book. Full comparison runs write the equivalent Mosaic/VQM
ledgers, rerun cost scenarios, and add a fixed buffered-holding diagnostic. New
matched-random paths use the same self-financing cost equation. These are modeled
fills, not actual broker executions; actual execution reconciliation still needs
the timestamped arrival quotes and fill files described in
[INSTITUTIONAL_RESEARCH.md](INSTITUTIONAL_RESEARCH.md).

SPA and Reality Check must both have valid computed p-values on at least 20 common
periods. The existing SPA threshold is retained; a separate Reality Check
significance threshold is not invented. Shared calendar alignment remains active.

Future entry prices in rebuilt feature panels are named `entry_adj_open` and are
simulation metadata only. They are not in `FEATURES`, model inputs or quality
calculations. Older panels may omit them; NAV accounting still works, but some
modeled fill quantities are then unavailable.

## What this does not establish

The archived historical backtest is unchanged. A new market-data run is required
to report new returns and SPA/Reality Check values. Historical membership,
delisted histories, security identities and terminal values still need verified
coverage; partial cached prices are not complete coverage. No model is promoted,
recipe frozen, or prospective history invented by these changes. Existing frozen
recipes fail source checks until a new validated version is deliberately frozen.

Local verification uses synthetic data and temporary files. It includes a small
full research run with real ridge fits, the maintained joint-test implementation,
VQM/drift reports, source identity tests, independent-profile publication, stale
risk rejection and independent NAV/fee reconciliation. The synthetic run is
software evidence only, not investment performance evidence.
