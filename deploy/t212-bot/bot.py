#!/usr/bin/env python3
"""Trading 212 DEMO bot driven by the Mosaic research engine's signals.

Runs 24/5: it reads /opt/mosaic-engine/signals/latest.json on every check, Monday to Friday, and places orders
only during US regular hours (9:35-15:50 ET), which matches the research engine's validation (entry at the next
session open, exit after the signal's horizon).

  entry  : a signal with direction LONG, deployment eligibility PAPER_TEST_ELIGIBLE or LIVE_REVIEW_ELIGIBLE,
           validation status of the same class, unexpired, liquidity and data quality approved and expected net
           excess return > 0. Several entries for one ticker -> the highest net return wins.
  exit   : after the signal's horizon (trading sessions since entry), or price <= avg cost * (1 - STOP_LOSS)
  sizing : POSITION_PCT of total account value per name, reduced so the position's 1-day 99% VaR (GARCH-t forecast in
           signals/risk_latest.json, chosen by the engine's VaR backtest) is at most BOT_POSITION_VAR_LIMIT (1%) of the
           account; free cash only, at most floor(1 / POSITION_PCT) names
  safety : stale or unreadable signal file -> no new entries (exits still run); STOP file pauses everything
  instant stop : a Finnhub WebSocket streams trades for held names; a print at/below the stop is confirmed with a
           live REST quote and the position is sold within seconds (the 15-minute cycle remains the fallback)
  pre-trade check : before each buy the live Finnhub quote must be < 10 minutes old and within BOT_MAX_GAP (3%)
           of the close on the signal's as-of date; otherwise the entry is skipped and logged as SKIP
Positions the old SMA trend strategy opened keep their old exit rule (below SMA50 or stop-loss); no new SMA
entries are made unless BOT_STRATEGY=sma.

Hard-wired to demo.trading212.com. Touch /opt/t212-bot/STOP to pause trading without stopping the service.
"""
import asyncio
import json
import logging
import math
import os
import threading
import time
from datetime import date, datetime, timedelta, timezone, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

BASE = "https://demo.trading212.com/api/v0"  # never change to live
HERE = Path(__file__).resolve().parent
STATE = HERE / "state.json"
TRADES = HERE / "trades.jsonl"
STOP_FILE = HERE / "STOP"
NY = ZoneInfo("America/New_York")

STRATEGY = os.environ.get("BOT_STRATEGY", "signals")  # "signals" (research engine) or "sma" (legacy trend rule)
SIGNALS = Path(os.environ.get("BOT_SIGNALS_FILE", "/opt/mosaic-engine/signals/latest.json"))
SIGNAL_MAX_AGE_DAYS = float(os.environ.get("BOT_SIGNAL_MAX_AGE_DAYS", "4"))
ELIGIBLE = {"PAPER_TEST_ELIGIBLE", "LIVE_REVIEW_ELIGIBLE"}
UNIVERSE = os.environ.get("BOT_UNIVERSE", "AAPL,MSFT,NVDA,AMZN,GOOGL,META,JPM,V,LLY,COST").split(",")
POSITION_PCT = float(os.environ.get("BOT_POSITION_PCT", "0.09"))
STOP_LOSS = float(os.environ.get("BOT_STOP_LOSS", "0.08"))
CHECK_MINUTES = int(os.environ.get("BOT_CHECK_MINUTES", "15"))
DRY_RUN = os.environ.get("BOT_DRY_RUN") == "1"
MAX_POSITIONS = int(1 / POSITION_PCT)
FINNHUB = os.environ.get("FINNHUB_API_KEY")
MAX_GAP = float(os.environ.get("BOT_MAX_GAP", "0.03"))
VAR_LIMIT = float(os.environ.get("BOT_POSITION_VAR_LIMIT", "0.01"))
RISK_FILE = SIGNALS.parent / "risk_latest.json"
QUOTE_MAX_AGE_S = 600
LOCK = threading.RLock()  # serialises orders and state changes between the cycle and the stop watcher
SELLING = set()  # tickers with a sell submitted today (either path), so a position is never sold twice

log = logging.getLogger("t212-bot")
http = httpx.Client(timeout=30, auth=(os.environ["T212_API_KEY_ID"], os.environ["T212_API_SECRET"]))


def t212(method, path, body=None, tries=4):
    for i in range(tries):
        r = http.request(method, BASE + path, json=body)
        if r.status_code == 429:
            time.sleep(6 * (i + 1))
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"{method} {path} -> HTTP {r.status_code}: {r.text[:300]}")
        return r.json() if r.text else None
    raise RuntimeError(f"{method} {path} rate limited")


def closes(symbol):
    r = httpx.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                  params={"range": "6mo", "interval": "1d"}, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    c = [x for x in res["indicators"]["quote"][0]["close"] if x is not None]
    price = res["meta"].get("regularMarketPrice") or c[-1]
    return c, price


def fh_symbol(ticker):
    """Trading 212 ticker -> exchange symbol via the instrument list's short name (FB_US_EQ -> META, BRK_B_US_EQ -> BRK.B)."""
    if not ticker.endswith("_US_EQ"):
        return None
    t212_ticker("AAPL")  # loads / refreshes the instrument cache
    return INSTR["by_ticker"].get(ticker) or ticker[: -len("_US_EQ")].replace("_", ".")


def fh_quote(sym):
    """Live Finnhub quote: (price, previous close, age in seconds); None if unavailable."""
    if not FINNHUB:
        return None
    r = httpx.get("https://finnhub.io/api/v1/quote", params={"symbol": sym, "token": FINNHUB}, timeout=10)
    if r.status_code != 200:
        return None
    q = r.json()
    if not q.get("c") or not q.get("t"):
        return None
    return float(q["c"]), float(q.get("pc") or 0), time.time() - float(q["t"])


def ref_close(sym, day):
    """Daily close on the signal's as-of date (Yahoo, split-adjusted like the quote)."""
    r = httpx.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym.replace('.', '-')}",
                  params={"range": "1mo", "interval": "1d"}, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    for ts, c in zip(res["timestamp"], res["indicators"]["quote"][0]["close"]):
        if c is not None and datetime.fromtimestamp(ts, NY).date() == day:
            return float(c)
    return None


def pretrade(sym, signal):
    """(ok, live price, reason). Entry only when the live quote is fresh and within MAX_GAP of the as-of close."""
    q = fh_quote(sym)
    if q is None:
        return False, None, "no live quote"
    price, _, age = q
    if age > QUOTE_MAX_AGE_S:
        return False, price, f"live quote is {age / 60:.0f} min old"
    ref = ref_close(sym, date.fromisoformat(signal["as_of_utc"][:10]))
    if not ref:
        return False, price, "no reference close for the as-of date"
    gap = price / ref - 1
    if abs(gap) > MAX_GAP:
        return False, price, f"price moved {gap:+.1%} since the signal's close {ref:.2f} (limit {MAX_GAP:.0%})"
    return True, price, f"gap {gap:+.2%} vs as-of close {ref:.2f}"


def position_weight(sym):
    """(weight, note): POSITION_PCT capped so that weight x 1-day 99% VaR <= VAR_LIMIT.
    A missing, stale, unreadable or non-positive VaR forecast returns weight 0: the entry is skipped, never sized blind."""
    try:
        doc = json.loads(RISK_FILE.read_text())
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(doc["generated_at_utc"].replace("Z", "+00:00"))).days
        r = doc["names"].get(sym)
        if r is None or age > SIGNAL_MAX_AGE_DAYS:
            return 0.0, "no fresh VaR forecast: entry blocked"
        v = float(r["var99_1d"])
        if not v > 0:
            return 0.0, f"invalid VaR forecast {v}: entry blocked"
        w = min(POSITION_PCT, VAR_LIMIT / v)
        return w, f"1d VaR99 {v:.2%} ({r['model']}) -> weight {w:.1%}"
    except (OSError, ValueError, KeyError, TypeError) as e:
        return 0.0, f"risk file unavailable ({e.__class__.__name__}): entry blocked"


def sma(xs, n):
    return sum(xs[-n:]) / n


def market_open(now):
    return now.weekday() < 5 and dtime(9, 35) <= now.time() <= dtime(15, 50)


def load_state():
    try:
        s = json.loads(STATE.read_text())
    except (OSError, ValueError):
        s = {}
    s.setdefault("sold_on", {})
    s.setdefault("positions", {})  # ticker -> {"strategy", "entered", "horizon", "signal_id", "model_id"}
    return s


def record(kind, ticker, qty, **extra):
    row = {"ts": datetime.now(NY).isoformat(timespec="seconds"), "kind": kind, "ticker": ticker, "qty": qty,
           "dry_run": DRY_RUN, **extra}
    with TRADES.open("a") as f:
        f.write(json.dumps(row) + "\n")
    log.info("%s %s qty=%s %s", kind, ticker, qty, extra)


def order(ticker, qty, **why):
    if DRY_RUN:
        record("DRY", ticker, qty, **why)
        return
    res = t212("POST", "/equity/orders/market", {"ticker": ticker, "quantity": qty})
    record("BUY" if qty > 0 else "SELL", ticker, qty, order_id=res.get("id"), **why)


# ---------------------------------------------------------------- instruments
INSTR = {"t": 0, "by_short": {}, "by_ticker": {}}
INSTR_LOCK = threading.Lock()


def t212_ticker(sym):
    """US-listed Trading 212 ticker for a symbol such as BRK.B (refreshed daily from the instrument list)."""
    with INSTR_LOCK:  # the instrument endpoint is heavily rate-limited: one fetch at a time, cached for a day
        if time.time() - INSTR["t"] > 86400:
            by, rev = {}, {}
            for i in t212("GET", "/equity/metadata/instruments"):
                if i.get("type") == "STOCK" and i.get("currencyCode") == "USD" and i["ticker"].endswith("_US_EQ"):
                    by.setdefault(i.get("shortName", "").upper(), []).append(i["ticker"])
                    rev[i["ticker"]] = i.get("shortName", "").upper()
            INSTR.update(t=time.time(), by_short=by, by_ticker=rev)
    for key in (sym.upper(), sym.upper().replace(".", "-"), sym.upper().replace("-", ".")):
        hits = INSTR["by_short"].get(key)
        if hits:
            pref = f"{key.replace('.', '_').replace('-', '_')}_US_EQ"
            return pref if pref in hits else hits[0]
    return None


# ---------------------------------------------------------------- signals
def read_signals():
    """Eligible LONG signals as {symbol: signal}; empty (with a reason) when the file is missing, stale or invalid."""
    try:
        doc = json.loads(SIGNALS.read_text())
    except (OSError, ValueError) as e:
        return {}, f"signal file unreadable: {e}"
    if doc.get("schema_version") != "2.0":
        return {}, f"unexpected schema_version {doc.get('schema_version')}"
    gen = datetime.fromisoformat(doc["generated_at_utc"].replace("Z", "+00:00"))
    age = (datetime.now(timezone.utc) - gen).total_seconds() / 86400
    if age > SIGNAL_MAX_AGE_DAYS:
        return {}, f"signal file is {age:.1f} days old (> {SIGNAL_MAX_AGE_DAYS})"
    now = datetime.now(timezone.utc)
    best = {}
    for s in doc.get("signals", []):
        sig, f, r, v = s["signal"], s["forecast"], s["risk"], s["validation"]
        if sig["direction"] != "LONG" or sig["deployment_eligibility"] not in ELIGIBLE or v["status"] not in ELIGIBLE:
            continue
        if not sig.get("expires_at_utc") or datetime.fromisoformat(sig["expires_at_utc"].replace("Z", "+00:00")) <= now:
            continue
        if not (r.get("liquidity_approved") and r.get("data_quality_approved")):
            continue
        net = f.get("expected_net_excess_return")
        if net is None or net <= 0 or not f.get("horizon_sessions"):
            continue
        sym = s["asset"]["ticker"]
        if sym not in best or net > best[sym]["forecast"]["expected_net_excess_return"]:
            best[sym] = s
    return best, f"{len(doc.get('signals', []))} entries, {len(best)} eligible LONG"


def sessions_since(d):
    """Weekdays elapsed since entry (exchange holidays are not excluded: a hold can end a day early around one)."""
    start, end = date.fromisoformat(d), datetime.now(NY).date()
    return sum(1 for k in range((end - start).days) if (start + timedelta(days=k)).weekday() < 5)


# ---------------------------------------------------------------- cycle
def cycle(state):
    today = datetime.now(NY).date().isoformat()
    cash = t212("GET", "/equity/account/cash")
    time.sleep(2)
    held = {p["ticker"]: p for p in t212("GET", "/equity/portfolio")}
    time.sleep(5)
    pending = {o["ticker"] for o in t212("GET", "/equity/orders")}
    free, total = float(cash["free"]), float(cash["total"])
    signals, why = read_signals() if STRATEGY == "signals" else ({}, "signals disabled")
    log.info("cycle: total=%.2f free=%.2f held=%s | %s", total, free, sorted(held), why)
    tracked = state["positions"]
    for t in list(tracked):  # forget positions closed outside the bot (e.g. manually)
        if t not in held and t not in pending:
            tracked.pop(t)

    # exits
    for ticker, pos in held.items():
        if ticker in pending:
            continue
        avg, qty, price = float(pos["averagePrice"]), float(pos["quantity"]), float(pos["currentPrice"])
        meta = tracked.get(ticker)
        reason = None
        if price <= avg * (1 - STOP_LOSS):
            reason = "stop_loss"
        elif meta and meta["strategy"] == "signals":
            if sessions_since(meta["entered"]) >= meta["horizon"]:
                reason = f"horizon_{meta['horizon']}d_reached"
        else:  # legacy SMA position: keep its original exit rule
            sym = (fh_symbol(ticker) or "").replace(".", "-")  # Yahoo symbol (FB_US_EQ -> META)
            try:
                c, px = closes(sym)
                if len(c) >= 50 and px < sma(c, 50):
                    reason = "below_sma50"
            except Exception as e:
                log.warning("price fetch failed for %s: %s", sym, e)
            time.sleep(1)
        if reason:
            with LOCK:
                if ticker in SELLING:
                    continue
                order(ticker, -qty, reason=reason, price=price, avg=avg, signal_id=(meta or {}).get("signal_id"))
                SELLING.add(ticker)
                state["sold_on"][ticker] = today
                tracked.pop(ticker, None)

    # entries
    slots = MAX_POSITIONS - len(held)
    if STRATEGY == "signals":
        ranked = sorted(signals.values(), key=lambda s: -s["forecast"]["expected_net_excess_return"])
        for s in ranked:
            if slots <= 0:
                break
            sym = s["asset"]["ticker"]
            ticker = t212_ticker(sym)
            if not ticker:
                log.warning("signal %s: no Trading 212 instrument for %s", s["signal_id"], sym)
                continue
            if ticker in held or ticker in pending or state["sold_on"].get(ticker) == today:
                continue
            try:
                ok, price, why = pretrade(sym, s)
            except Exception as e:
                ok, price, why = False, None, f"pre-trade check failed: {e}"
            if not ok:
                record("SKIP", ticker, 0, reason=why, signal_id=s["signal_id"], model_id=s["model_id"], price=price)
                state["sold_on"][ticker] = today  # do not retry the same name today after a failed check
                continue
            weight, sizing = position_weight(sym)
            if weight <= 0:
                log.info("skip %s: %s", sym, sizing)
                continue
            budget = min(total * weight, free * 0.98)
            qty = math.floor(budget * usd_per_account_ccy() / price * 100) / 100
            if qty < 0.01 or budget <= 50:
                log.info("not enough free cash for %s (budget %.2f)", sym, budget)
                break
            order(ticker, qty, reason="research_signal", price=price, pretrade=why, sizing=sizing, signal_id=s["signal_id"], model_id=s["model_id"],
                  horizon=s["forecast"]["horizon_sessions"], expected_net=s["forecast"]["expected_net_excess_return"])
            tracked[ticker] = {"strategy": "signals", "entered": today, "horizon": int(s["forecast"]["horizon_sessions"]),
                               "signal_id": s["signal_id"], "model_id": s["model_id"]}
            free -= budget
            slots -= 1
            time.sleep(1)
    else:  # legacy SMA trend entries
        for sym in UNIVERSE:
            ticker = t212_ticker(sym)
            if not ticker or ticker in held or ticker in pending or state["sold_on"].get(ticker) == today or slots <= 0:
                continue
            try:
                c, price = closes(sym)
            except Exception as e:
                log.warning("price fetch failed for %s: %s", sym, e)
                continue
            if len(c) >= 50 and price > sma(c, 50) and sma(c, 20) > sma(c, 50):
                budget = min(total * POSITION_PCT, free * 0.98)
                qty = math.floor(budget * usd_per_account_ccy() / price * 100) / 100
                if qty >= 0.01 and budget > 50:
                    order(ticker, qty, reason="trend_entry", price=price)
                    tracked[ticker] = {"strategy": "sma", "entered": today}
                    free -= budget
                    slots -= 1
            time.sleep(1)
    with LOCK:  # the stop watcher may modify the same state
        STATE.write_text(json.dumps(state))


class StopWatcher(threading.Thread):
    """Real-time stop-loss on held positions using Finnhub's trade stream (falls back to the 15-minute cycle)."""

    def __init__(self, state):
        super().__init__(daemon=True)
        self.state, self.pos, self.subs = state, {}, set()

    def refresh(self):
        held = t212("GET", "/equity/portfolio")
        self.pos = {fh_symbol(p["ticker"]): {"ticker": p["ticker"], "avg": float(p["averagePrice"]), "qty": float(p["quantity"])}
                    for p in held if fh_symbol(p["ticker"])}

    def trigger(self, sym, px):
        p = self.pos.get(sym)
        if not p or not market_open(datetime.now(NY)) or STOP_FILE.exists():
            return
        stop = p["avg"] * (1 - STOP_LOSS)
        if px > stop:
            return
        q = fh_quote(sym)  # confirm with a second, independent live price so one bad print cannot trigger a sale
        if not q or q[0] > stop or q[2] > 120:
            return
        with LOCK:
            if p["ticker"] in SELLING:
                return
            order(p["ticker"], -p["qty"], reason="stop_loss_realtime", price=q[0], avg=p["avg"], trigger_print=px,
                  signal_id=self.state["positions"].get(p["ticker"], {}).get("signal_id"))
            SELLING.add(p["ticker"])
            self.state["sold_on"][p["ticker"]] = datetime.now(NY).date().isoformat()
            self.state["positions"].pop(p["ticker"], None)
            STATE.write_text(json.dumps(self.state))
        self.pos.pop(sym, None)

    async def stream(self):
        import websockets
        async with websockets.connect(f"wss://ws.finnhub.io?token={FINNHUB}", ping_interval=20) as ws:
            last = 0
            while True:
                if time.time() - last > 60:  # follow position changes made by the cycle
                    await asyncio.to_thread(self.refresh)
                    want = set(self.pos)
                    for s in want - self.subs:
                        await ws.send(json.dumps({"type": "subscribe", "symbol": s}))
                    for s in self.subs - want:
                        await ws.send(json.dumps({"type": "unsubscribe", "symbol": s}))
                    self.subs, last = want, time.time()
                try:
                    m = json.loads(await asyncio.wait_for(ws.recv(), 30))
                except asyncio.TimeoutError:
                    continue
                if m.get("type") == "trade":
                    low = {}
                    for tr in m["data"]:
                        low[tr["s"]] = min(low.get(tr["s"], 1e18), float(tr["p"]))
                    for s, px in low.items():
                        await asyncio.to_thread(self.trigger, s, px)

    def run(self):
        if not FINNHUB:
            log.warning("FINNHUB_API_KEY missing: real-time stop disabled, 15-minute stop checks only")
            return
        while True:
            if market_open(datetime.now(NY)) or datetime.now(NY).time() < dtime(9, 35):
                try:
                    asyncio.run(self.stream())
                except Exception as e:
                    log.warning("real-time stop stream dropped (%s); reconnecting in 30s", e)
            time.sleep(30)


_fx = {"t": 0, "v": 1.0}


def usd_per_account_ccy():
    if time.time() - _fx["t"] > 3600:
        ccy = t212("GET", "/equity/account/info")["currencyCode"]
        if ccy == "USD":
            _fx.update(t=time.time(), v=1.0)
        else:
            _, rate = closes(f"{ccy}USD=X")
            _fx.update(t=time.time(), v=float(rate))
    return _fx["v"]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    log.info("starting: demo only, strategy=%s signals=%s pos=%.0f%% max_names=%d stop=%.0f%% (real-time %s) max_gap=%.0f%% every %dm dry_run=%s",
             STRATEGY, SIGNALS, POSITION_PCT * 100, MAX_POSITIONS, STOP_LOSS * 100, "on" if FINNHUB else "off",
             MAX_GAP * 100, CHECK_MINUTES, DRY_RUN)
    log.info("sizing: %.0f%% per name, capped so 1-day 99%% VaR <= %.1f%% of the account (%s)", POSITION_PCT * 100, VAR_LIMIT * 100, RISK_FILE)
    state = load_state()
    StopWatcher(state).start()
    last_day = None
    last_idle = 0
    while True:
        now = datetime.now(NY)
        if now.date() != last_day:  # a new day clears the same-day sell guard
            SELLING.clear()
            last_day = now.date()
        if STOP_FILE.exists():
            if time.time() - last_idle > 3600:
                log.info("STOP file present, trading paused")
                last_idle = time.time()
        elif market_open(now):
            try:
                cycle(state)
            except Exception:
                log.exception("cycle failed")
        elif time.time() - last_idle > 3600:
            if now.weekday() < 5 and STRATEGY == "signals":
                _, why = read_signals()
                log.info("market closed (%s ET); signals: %s; orders wait for 9:35 ET", now.strftime("%a %H:%M"), why)
            else:
                log.info("market closed (%s ET), waiting", now.strftime("%a %H:%M"))
            last_idle = time.time()
        time.sleep(CHECK_MINUTES * 60 if market_open(datetime.now(NY)) else 60)


if __name__ == "__main__":
    main()
