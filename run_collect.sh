#!/bin/bash
# Nightly collectors (after the US close): Alpha Vantage earnings backfill every day (25-call free limit),
# Cboe options + Nasdaq estimates snapshots on weekdays (in parallel: different hosts), Finnhub recommendations Fridays.
cd /opt/mosaic-engine
PY=/opt/mosaic-engine/venv/bin/python
export MOSAIC_LOCK_HELD=1  # this wrapper holds .lock
L=ledger
$PY -u data_collect.py av                 >> $L/collect_av.log 2>&1
$PY -u data_collect.py options            >> $L/collect_options.log 2>&1 &
$PY -u data_collect.py estimates recs     >> $L/collect_estimates.log 2>&1 &
wait
echo "$(date -u +%FT%TZ) collect ok" >> $L/collect.log
