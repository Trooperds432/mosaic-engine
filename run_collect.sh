#!/bin/bash
# Nightly collectors (after the US close): Alpha Vantage earnings backfill every day (25-call free limit),
# Cboe options + Nasdaq estimates snapshots on weekdays (in parallel: different hosts), Finnhub recommendations Fridays.
cd /opt/mosaic-engine
PY=/opt/mosaic-engine/venv/bin/python
# Own lock (not the engine's .lock): collectors run for hours and must not make the nightly job skip, but two
# collector runs must never overlap. They only write data/raw/collect/*, which signals.py reads, so their files are
# written to a temp name and renamed (data_collect.py).
exec 8>/opt/mosaic-engine/.collect.lock
flock -n 8 || { echo "$(date -u +%FT%TZ) collectors already running" >> ledger/collect.log; exit 0; }
L=ledger
$PY -u data_collect.py av                 >> $L/collect_av.log 2>&1
$PY -u data_collect.py options            >> $L/collect_options.log 2>&1 &
$PY -u data_collect.py estimates recs     >> $L/collect_estimates.log 2>&1 &
wait
echo "$(date -u +%FT%TZ) collect ok" >> $L/collect.log
