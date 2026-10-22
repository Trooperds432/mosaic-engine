#!/bin/bash
# Daily refresh after the US close: data -> audit -> signals -> catalysts/risk -> report. Never retrains models and
# never touches the trading bot; the bot reads signals/latest.json only if it is separately configured to.
set -eo pipefail
cd /opt/mosaic-engine
exec 9>/opt/mosaic-engine/.lock
flock -n 9 || { echo "another run in progress"; exit 0; }
PY=/opt/mosaic-engine/venv/bin/python
export MOSAIC_LOCK_HELD=1  # this wrapper holds .lock
L=ledger
FRI=$([ "$(date +%u)" = "5" ] || [ "$FORCE_WEEKLY" = "1" ] && echo 1 || echo 0)
$PY -u data_prices.py            >  $L/ingest_prices.log 2>&1
$PY -u data_insider.py 2006      >  $L/ingest_insider.log 2>&1   # cached quarters skipped, latest two refreshed
$PY -u data_form4_recent.py      >  $L/ingest_form4.log 2>&1
$PY -u data_macro.py             >  $L/ingest_macro.log 2>&1     # FRED/EIA refetch; ALFRED only new vintages
$PY -u macro_surprise.py         >  $L/macro_surprise.log 2>&1   # ARIMA "expected" vs first release (spec 4D)
$PY -u data_alt.py futures cot   >  $L/ingest_alt.log 2>&1
if [ "$FRI" = "1" ]; then
  $PY -u data_fundamentals.py --refresh > $L/ingest_facts.log 2>&1
  $PY -u data_8k.py --refresh           > $L/ingest_8k.log 2>&1
  $PY -u data_13f.py                    > $L/ingest_13f.log 2>&1   # new quarterly data sets + re-aggregate
  $PY -u data_alt.py short trials fda   >> $L/ingest_alt.log 2>&1
  rm -f data/processed/cusip_cik.parquet                          # rebuild CUSIP map with new names
else
  $PY -u data_fundamentals.py     > $L/ingest_facts.log 2>&1      # only new CIKs
  $PY -u data_updates.py 5        > $L/ingest_updates.log 2>&1    # companies that filed a 10-Q/10-K/8-K in the last 5 days
fi
$PY -u audit.py large            >  $L/audit.log 2>&1 || true    # a failed audit blocks LONG signals inside signals.py
$PY -u var_backtest.py --forecast >  $L/vol_forecast.log 2>&1   # GARCH-t (VaR backtest winner) per-name risk
$PY -u signals.py                >> $L/signals.log 2>&1
$PY -u paper_trading.py          >  $L/paper.log 2>&1 || true   # forward record, shadow book, demo fills
$PY -u insider_forward.py       >  $L/insider_forward.log 2>&1 || true  # frozen-rule prospective events; missing recipe stays unvalidated
$PY -u ls_signals.py             >  $L/ls_signals.log 2>&1 || true  # long/short inputs for the Alpaca paper bot
$PY -u catalysts.py              >  $L/catalysts.log 2>&1 || true
[ "$FRI" = "1" ] && { $PY -u biotech.py > $L/biotech.log 2>&1 || true; }
$PY -u portfolio.py              >  $L/portfolio.log 2>&1 || true
$PY -u report.py                 >> $L/signals.log 2>&1
echo "$(date -u +%FT%TZ) daily ok" >> $L/daily.log
