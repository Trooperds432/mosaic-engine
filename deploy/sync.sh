#!/bin/sh
# Snapshot the bot code and systemd units into deploy/ before committing. Launchers that hold keys are not copied.
D=/opt/mosaic-engine/deploy
cp /opt/t212-bot/bot.py $D/t212-bot/bot.py
cp /opt/alpaca-bot/bot.py /opt/alpaca-bot/run.sh $D/alpaca-bot/
for u in mosaic-engine.service mosaic-engine.timer mosaic-collect.service mosaic-collect.timer mosaic-refit.service mosaic-refit.timer mosaic-intraday.service t212-bot.service alpaca-bot.service; do cp /etc/systemd/system/$u $D/systemd/; done
