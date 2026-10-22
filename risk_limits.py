"""Read-only import of the existing bot's risk configuration (spec section 0: RISK_LIMITS are imported, never
invented). Reads the bot's defaults from its source, any BOT_* overrides in its systemd environment, and
the account value from the Trading 212 demo API (GET endpoints only). The engine uses these numbers to
size its cost/impact model and optimiser constraints; it never writes to the bot."""
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import httpx

import config as C

BOT = Path("/opt/t212-bot/bot.py")
KEYS = {"BOT_POSITION_PCT": ("position_pct", float), "BOT_STOP_LOSS": ("stop_loss", float),
        "BOT_CHECK_MINUTES": ("check_minutes", int), "BOT_UNIVERSE": ("universe", lambda s: s.split(",")),
        # portfolio constraints (spec 10): imported when the bot defines them, otherwise reported as not supplied
        "BOT_SECTOR_CAP": ("sector_cap", float), "BOT_TURNOVER_CAP": ("turnover_cap", float), "BOT_RISK_AVERSION": ("risk_aversion", float),
        "BOT_MAX_BETA": ("max_beta", float), "BOT_MAX_ADV_PARTICIPATION": ("max_adv_participation", float)}


def _env_file(path, key):
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith(key + "="):
                return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return None


def run():
    src = BOT.read_text()
    lim = {"source": str(BOT), "imported_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for env, (name, cast) in KEYS.items():
        m = re.search(rf'os\.environ\.get\("{env}",\s*"([^"]*)"\)', src)
        lim[name] = cast(m.group(1)) if m else None
    unit_env = subprocess.run(["systemctl", "show", "t212-bot", "-p", "Environment", "--value"], capture_output=True, text=True).stdout
    for tok in unit_env.split():
        k, _, v = tok.partition("=")
        if k in KEYS:
            lim[KEYS[k][0]] = KEYS[k][1](v)
            lim.setdefault("overridden_by_systemd", []).append(k)
    lim["base_url"] = re.search(r'BASE = "([^"]+)"', src).group(1)
    # Trading 212 Invest accounts cannot short; the bot only sells shares it holds
    lim["long_only"] = True
    lim["leverage"] = False
    lim["derivatives_permitted"] = False
    lim["free_cash_only"] = "free * 0.98" in src
    # account value (read-only)
    kid, sec = os.environ.get("T212_API_KEY_ID") or _env_file("/root/.hermes/.env", "T212_API_KEY_ID"), \
        os.environ.get("T212_API_SECRET") or _env_file("/root/.hermes/.env", "T212_API_SECRET")
    try:
        with httpx.Client(timeout=30, auth=(kid, sec)) as c:
            cash = c.get(lim["base_url"] + "/equity/account/cash").json()
            info = c.get(lim["base_url"] + "/equity/account/info").json()
        lim["account_currency"] = info.get("currencyCode")
        lim["account_total"] = float(cash["total"])
        fx = httpx.get("https://query1.finance.yahoo.com/v8/finance/chart/GBPUSD=X", params={"range": "5d", "interval": "1d"},
                       headers={"User-Agent": "Mozilla/5.0"}, timeout=30).json()["chart"]["result"][0]["meta"]["regularMarketPrice"]
        lim["account_total_usd"] = lim["account_total"] * (fx if lim["account_currency"] == "GBP" else 1.0)
        lim["order_size_usd"] = lim["account_total_usd"] * lim["position_pct"]
        lim["max_names_at_full_size"] = int(1 / lim["position_pct"]) if lim["position_pct"] else None
    except Exception as e:
        lim["account_error"] = str(e)[:200]
    (C.PROC / "bot_risk_limits.json").write_text(json.dumps(lim, indent=1))
    print(json.dumps({k: v for k, v in lim.items() if k != "universe"}, indent=1))
    return lim


def load():
    f = C.PROC / "bot_risk_limits.json"
    return json.loads(f.read_text()) if f.exists() else run()


if __name__ == "__main__":
    run()
