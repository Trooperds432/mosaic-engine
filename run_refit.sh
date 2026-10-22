#!/bin/bash
# Weekly refit (learning only): rebuild panels with the week's data and retrain every model with the recipe frozen at the
# last validation. Never evaluates or touches the holdout; trading eligibility comes from the forward record.
set -eo pipefail
cd /opt/mosaic-engine
exec 9>/opt/mosaic-engine/.lock
flock 9   # wait for the nightly job if it is still running
export MOSAIC_LOCK_HELD=1
/opt/mosaic-engine/venv/bin/python -u refit.py >> ledger/refit.log 2>&1
echo "$(date -u +%FT%TZ) refit ok" >> ledger/refit.log
