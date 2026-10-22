# Research validity changes and remaining evidence

This change adds enforcement and data-import paths. It does not certify the
existing historical research, supply missing vendor datasets or establish alpha.

| Requirement | Implemented | Still required |
| --- | --- | --- |
| Constituent-aware universe | Permanent IDs, dated membership, archived imports, verified terminal outcomes and blocking audit | Complete source exports, identifier review and a fresh audit/backtest |
| Historical feature availability | Initial-release FRED API; observed-only EIA/NFCI/mortgage snapshots; legacy sampled vintages excluded; questionable clinical feature excluded | FRED key and successful ingestion; historical clinical record reconstruction if reinstated |
| Insider incremental value | Actual matched-company covariates, value/quality controls, timestamped same-period estimate revisions | Valid estimate history, full complete-case regression and resolved primary-security mapping for multi-class issuers |
| Deployment chain | Cumulative historical, forward, data and frozen-recipe checks; broker rejects research-only targets; live-review IC condition | Fresh forward observations and consistent L/S strategy validation |
| Portfolio constraints | Explicit external constraint contract including beta and volume participation; no invented optimizer defaults | Operator-supplied constraint values and actual portfolio validation |

## Reproducible forward research

After imports, audits, research and model review, freeze the intended recipe with
`python recipe.py broad`, `python recipe.py large` or `python recipe.py insider-event`.
This records code, model specification, timestamps and content-addressed weight
snapshots. Subsequent permitted refits record new weight manifests. Changed source
or an unrecorded weight change blocks deployment. Freeze only after the final code
and intended research specification are in place; changing source invalidates it.
Old forecasts and records predating the freeze cannot satisfy a new forward gate.

The insider forward scorer uses first-recorded signals and conservative subsequent
session entries, rather than reconstructing a favorable historical signal log.
It is a shadow research record, not evidence of broker fills. No elapsed observation
period can be replaced by synthetic tests.

The existing L/S historical simulation rebalances every five sessions, whereas the
bot uses twenty. It also lacks a separately validated strategy forward book. The
new policy therefore withholds executable L/S targets. Cross-sectional IC alone
does not validate that constructed portfolio. Completing that strategy's research
and forward validation remains outstanding.

## Migration and verification limits

Existing deployments will fail closed until the new data, recipe and forward checks
pass. No services were restarted and no broker orders were submitted during this
implementation. Provider adapters were tested with simulated responses; actual VPS
credentials and subscriptions were not accessed from the local development process.
The supplied free-tier earnings/recommendation data are not an archived estimate
revision database. Historical completeness depends on source evidence, not just code.

Run `python -m unittest discover -s tests -q`, `python -m compileall -q .` and
`git diff --check` before packaging. Passing synthetic and integration checks
establishes tested mechanics, not investment performance or complete vendor coverage.
