#!/bin/bash
# Full re-validation for both universe profiles: PIT panels -> factor risk models -> research suite -> insider event
# study -> commodity research. Each research run is appended to ledger/research_runs.jsonl; re-running after changing
# features/config re-uses the holdout, so record why you re-ran. The pre-registered insider model is NOT re-run.
set -eo pipefail
cd /opt/mosaic-engine
exec 9>/opt/mosaic-engine/.lock
flock 9
PY=/opt/mosaic-engine/venv/bin/python
export MOSAIC_LOCK_HELD=1  # this wrapper holds .lock
for P in ${PROFILES:-large broad}; do
  $PY -u features.py $P        2>&1 | tee ledger/panel_$P.log
  $PY -u risk_model.py $P      2>&1 | tee ledger/risk_model_$P.log
  $PY -u research.py $P        2>&1 | tee ledger/research_$P.log
  $PY -u regimes.py $P         2>&1 | tee ledger/regimes_$P.log   # Markov-switching, change points, regime factor returns
  $PY -u screening.py $P       2>&1 | tee ledger/mi_$P.log        # mutual-information screen
  $PY -u baselines.py $P       2>&1 | tee ledger/baselines_$P.log # matched random portfolios + block bootstrap
  $PY -u kalman_beta.py $P     2>&1 | tee ledger/kalman_$P.log    # Kalman vs rolling vs Blume beta forecasts
done
$PY -u var_backtest.py         2>&1 | tee ledger/var_backtest.log   # VaR/ES model choice (writes processed/vol_model.json)
$PY -u event_study.py          2>&1 | tee ledger/event_study.log
$PY -u commodity_research.py   2>&1 | tee ledger/commodity_research.log
