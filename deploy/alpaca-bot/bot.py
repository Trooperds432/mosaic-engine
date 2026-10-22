"""Alpaca PAPER long/short bot: trades the Mosaic engine's market-neutral book (signals/ls_latest.json).

  rebalance : monthly (an engine as-of date >= 20 sessions after the last rebalance), from 15 minutes after the US open. Targets come from
              ls_rules.select() using the bot's current holdings (turnover buffer) and Alpaca's tradable /
              shortable / easy-to-borrow flags. Closing trades run first, then opening trades.
  sizing    : weights x account equity; longs in fractional shares, shorts in whole shares; trades smaller than
              MIN_TRADE_USD or 20% of the target are skipped. Gross <= 1.5x, per name <= 2% of equity.
  pre-trade : new positions need a fresh IEX trade within GAP_LIMIT (3%) of the previous close and price >= $5.
  stops     : every POLL seconds in market hours a position whose loss reaches STOP_LOSS (8%; for shorts a rise of
              8%) is closed and not re-entered until the next as-of date.
  kill      : equity more than KILL_DD (15%) below its peak -> flatten everything and create the STOP file.
  safety    : hard-wired to paper-api.alpaca.markets; stale (> 4 days) or unreadable targets -> no new trades
              (stops still run); touch /opt/alpaca-bot/STOP to pause everything.
"""
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, "/opt/mosaic-engine")
import ls_rules as R  # noqa: E402

HERE = Path(__file__).resolve().parent
PAPER = "https://paper-api.alpaca.markets"
DATA = "https://data.alpaca.markets"
TARGETS = Path("/opt/mosaic-engine/signals/ls_latest.json")
STATE = HERE / "state.json"
TRADES = HERE / "trades.jsonl"
STOP_FILE = HERE / "STOP"
NY = ZoneInfo("America/New_York")
POLL = 30
STOP_LOSS = float(os.environ.get("ALPACA_STOP_LOSS", "0.08"))
KILL_DD = float(os.environ.get("ALPACA_KILL_DD", "0.15"))
GAP_LIMIT = 0.03
MIN_PRICE = 5.0
MIN_TRADE_USD = 50.0
MAX_AGE_DAYS = 4
REBALANCE_SESSIONS = 20   # monthly: cut costs roughly in half vs weekly in the walk-forward test

log = logging.getLogger("alpaca-bot")
logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per poll otherwise


def env():
    vals = {}
    for line in Path("/root/.hermes/.env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip().strip('"').strip("'")
    base = vals.get("ALPACA_BASE_URL", PAPER).rstrip("/").removesuffix("/v2")
    if base != PAPER:
        raise SystemExit(f"refusing to run: ALPACA_BASE_URL is {base}, this bot only trades {PAPER}")
    return vals["ALPACA_API_KEY_ID"], vals["ALPACA_API_SECRET_KEY"]


KEY, SECRET = env()
HDR = {"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SECRET}
api = httpx.Client(base_url=PAPER, headers=HDR, timeout=30)
data = httpx.Client(base_url=DATA, headers=HDR, timeout=30)


def call(method, path, client=None, **kw):
    c = client or api
    for i in range(4):
        try:
            r = c.request(method, path, **kw)
            if r.status_code == 429 or r.status_code >= 500:
                raise httpx.HTTPError(f"{r.status_code} {r.text[:200]}")
            return r
        except httpx.HTTPError as e:
            log.warning("%s %s failed (%s), retry %d", method, path, e, i + 1)
            time.sleep(2 ** (i + 1))
    raise RuntimeError(f"{method} {path} failed")


def load_state():
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {"last_rebalanced_asof": None, "peak_equity": None, "stopped": {}}


def save_state(s):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, indent=1))
    tmp.replace(STATE)


def record(ev):
    ev["at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with TRADES.open("a") as f:
        f.write(json.dumps(ev) + "\n")


def positions():
    return {p["symbol"]: p for p in call("GET", "/v2/positions").json()}


_assets = {"at": None, "map": {}}


def assets():
    if _assets["at"] is None or time.time() - _assets["at"] > 6 * 3600:
        rows = call("GET", "/v2/assets", params={"status": "active", "asset_class": "us_equity"}).json()
        _assets["map"] = {a["symbol"]: a for a in rows}
        _assets["at"] = time.time()
    return _assets["map"]


def snapshots(symbols):
    out = {}
    syms = sorted(symbols)
    for i in range(0, len(syms), 100):
        r = call("GET", "/v2/stocks/snapshots", client=data, params={"symbols": ",".join(syms[i:i + 100]), "feed": "iex"})
        for s, v in (r.json() or {}).items():
            if not v:
                continue
            lt, pb = v.get("latestTrade") or {}, v.get("prevDailyBar") or {}
            out[s] = {"price": lt.get("p"), "t": lt.get("t"), "prev_close": pb.get("c")}
    return out


DRY = "--dry" in sys.argv


def submit(symbol, side, qty):
    if DRY:
        log.info("DRY %s %s %s", side, qty, symbol)
        return True
    body = {"symbol": symbol, "side": side, "type": "market", "time_in_force": "day", "qty": str(qty)}
    r = call("POST", "/v2/orders", json=body)
    ok = r.status_code in (200, 201)
    record({"event": "order", "symbol": symbol, "side": side, "qty": qty, "ok": ok, "status": r.status_code,
            "id": r.json().get("id") if ok else None, "error": None if ok else r.text[:300]})
    if not ok:
        log.warning("order %s %s %s rejected: %s", side, qty, symbol, r.text[:200])
    return ok


def wait_for_fills(timeout=90):
    if DRY:
        return True
    end = time.time() + timeout
    while time.time() < end:
        if not call("GET", "/v2/orders", params={"status": "open"}).json():
            return True
        time.sleep(3)
    log.warning("open orders still pending after %ss", timeout)
    return False


def check_stops(state, pos, asof):
    for s, p in pos.items():
        plpc = float(p["unrealized_plpc"])  # Alpaca reports short losses as negative too
        if plpc <= -STOP_LOSS:
            log.info("STOP %s %s: unrealised %.1f%% <= -%.0f%%", p["side"], s, plpc * 100, STOP_LOSS * 100)
            r = call("DELETE", f"/v2/positions/{s}")
            record({"event": "stop", "symbol": s, "side": p["side"], "qty": p["qty"], "unrealized_plpc": plpc, "ok": r.status_code in (200, 207)})
            state["stopped"][s] = asof


def kill_switch(state, equity):
    peak = max(state.get("peak_equity") or equity, equity)
    state["peak_equity"] = peak
    if equity < peak * (1 - KILL_DD):
        log.error("KILL: equity %.0f is %.1f%% below peak %.0f -> flattening and pausing", equity, (1 - equity / peak) * 100, peak)
        call("DELETE", "/v2/positions", params={"cancel_orders": "true"})
        record({"event": "kill_switch", "equity": equity, "peak": peak})
        STOP_FILE.write_text(f"kill switch {datetime.now(timezone.utc).isoformat()}: equity {equity:.0f} vs peak {peak:.0f}\n")
        return True
    return False


def rebalance(state, doc):
    asof = doc["as_of"]
    acct = call("GET", "/v2/account").json()
    equity = float(acct["equity"])
    if not acct.get("shorting_enabled"):
        log.warning("shorting disabled on the account (equity < $2,000?) - skipping rebalance")
        return
    pos = positions()
    amap = assets()
    stopped = {s for s, a in state["stopped"].items() if a == asof}
    g = pd.DataFrame(doc["names"]).rename(columns={"ticker": "symbol"})
    g = g[g["symbol"].map(lambda s: s in amap and amap[s]["tradable"] and s not in stopped)]
    held_l = {s for s, p in pos.items() if p["side"] == "long"}
    held_s = {s for s, p in pos.items() if p["side"] == "short"}
    can_short = lambda s: bool(amap[s].get("shortable") and amap[s].get("easy_to_borrow"))
    w = R.select(g, held_l, held_s, can_short=can_short, gross=doc["params"]["gross"])
    if not w:
        log.warning("selection empty for %s - no rebalance", asof)
        return
    snap = snapshots(set(w) | set(pos))
    tgt = {}
    for s, wt in w.items():
        q = snap.get(s) or {}
        px, prev = q.get("price"), q.get("prev_close")
        new = s not in pos
        if not px or px < MIN_PRICE:
            continue
        if new and prev and abs(px / prev - 1) > GAP_LIMIT:
            log.info("pre-trade: skip new %s, price %.2f is %.1f%% from previous close %.2f", s, px, (px / prev - 1) * 100, prev)
            continue
        n = wt * equity / px
        tgt[s] = round(n, 4) if wt > 0 and amap[s].get("fractionable") else float(math.floor(abs(n)) * (1 if wt > 0 else -1))
    cur = {s: float(p["qty"]) for s, p in pos.items()}  # negative for shorts
    closes, opens = [], []
    for s in set(cur) | set(tgt):
        c, t = cur.get(s, 0.0), tgt.get(s, 0.0)
        px = (snap.get(s) or {}).get("price") or float(pos[s]["current_price"]) if s in pos else (snap.get(s) or {}).get("price")
        if not px:
            continue
        if c != 0 and (t == 0 or (c > 0) != (t > 0)):
            closes.append((s, "sell" if c > 0 else "buy", abs(c)))  # exit or side flip: close first
            if t != 0:
                opens.append((s, "buy" if t > 0 else "sell", abs(t)))
            continue
        d = t - c
        if abs(d) * px < MIN_TRADE_USD or (t != 0 and abs(d) < 0.2 * abs(t)):
            continue
        reduce = abs(t) < abs(c)
        (closes if reduce else opens).append((s, ("buy" if d > 0 else "sell"), round(abs(d), 4) if (t > 0 or c > 0) else float(math.floor(abs(d)))))
    log.info("rebalance %s: equity %.0f, targets %d long / %d short, %d closing and %d opening orders",
             asof, equity, sum(v > 0 for v in tgt.values()), sum(v < 0 for v in tgt.values()), len(closes), len(opens))
    for s, side, q in closes:
        if q > 0:
            submit(s, side, q)
    wait_for_fills()
    for s, side, q in opens:
        if q > 0:
            submit(s, side, q)
    wait_for_fills()
    if DRY:
        return
    state["last_rebalanced_asof"] = asof
    acct = call("GET", "/v2/account").json()
    pos = positions()
    lv = sum(float(p["market_value"]) for p in pos.values() if p["side"] == "long")
    sv = -sum(float(p["market_value"]) for p in pos.values() if p["side"] == "short")
    eq = float(acct["equity"])
    record({"event": "rebalanced", "as_of": asof, "equity": eq, "long_value": lv, "short_value": sv, "gross": (lv + sv) / eq, "net": (lv - sv) / eq,
            "n_long": sum(p["side"] == "long" for p in pos.values()), "n_short": sum(p["side"] == "short" for p in pos.values())})
    log.info("after rebalance: gross %.2fx net %+.2f%%, %d positions", (lv + sv) / eq, (lv - sv) / eq * 100, len(pos))


def targets():
    try:
        doc = json.loads(TARGETS.read_text())
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(doc["generated_at_utc"])).total_seconds() / 86400
        return doc if age <= MAX_AGE_DAYS else None
    except Exception as e:
        log.warning("cannot read targets: %s", e)
        return None


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    acct = call("GET", "/v2/account").json()
    log.info("Alpaca PAPER bot: account %s equity %s shorting %s; stop %.0f%%, kill drawdown %.0f%%, gross %.1fx",
             acct.get("account_number"), acct.get("equity"), acct.get("shorting_enabled"), STOP_LOSS * 100, KILL_DD * 100, R.LS_GROSS)
    state = load_state()
    while True:
        try:
            if STOP_FILE.exists():
                time.sleep(60)
                continue
            clock = call("GET", "/v2/clock").json()
            if not clock["is_open"]:
                time.sleep(300)
                continue
            doc = targets()
            asof = doc["as_of"] if doc else None
            equity = float(call("GET", "/v2/account").json()["equity"])
            if kill_switch(state, equity):
                save_state(state)
                continue
            check_stops(state, positions(), asof or state.get("last_rebalanced_asof"))
            opened = datetime.fromisoformat(clock["timestamp"]).astimezone(NY)
            after_open = opened.hour * 60 + opened.minute >= 9 * 60 + 45
            last = state.get("last_rebalanced_asof")
            due = doc and (last is None or int(np.busday_count(last, asof)) >= REBALANCE_SESSIONS)
            if due and after_open:
                rebalance(state, doc)
            save_state(state)
        except Exception as e:
            log.exception("loop error: %s", e)
        time.sleep(POLL)


if __name__ == "__main__":
    if DRY:  # one rebalance computation against the live paper account, printing orders instead of sending them
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        d = targets()
        if d is None:
            raise SystemExit("no fresh signals/ls_latest.json")
        rebalance(load_state(), d)
    else:
        main()
