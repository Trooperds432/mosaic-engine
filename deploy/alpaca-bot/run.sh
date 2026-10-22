#!/bin/sh
# keys are read by bot.py from /root/.hermes/.env; the bot refuses any endpoint other than paper-api.alpaca.markets
exec /opt/mosaic-engine/venv/bin/python -u /opt/alpaca-bot/bot.py "$@"
