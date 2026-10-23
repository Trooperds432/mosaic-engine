"""Forward (prospective) record of the market-neutral long/short book: the evidence that can set
`strategy_forward_validated`, which the Alpaca paper bot needs in addition to the historical gate.

Each night, from the candidate list ls_signals.py saves (data/processed/ls_candidates_<profile>.parquet, written
whatever the deployment verdict), a shadow book is run with exactly the bot's rules: ls_rules.select with the turnover
buffer and analyst vetoes, a rebalance every REBALANCE_SESSIONS sessions, entry at the next session's open.
Positions are marked open-to-open every session with price drift; a rebalance charges max(half spread, 3bp) + impact on
every dollar traded and shorts pay the borrow-fee tier daily. Only sessions recorded under the current frozen recipe
count: a new recipe restarts the record.

Validated when, under a frozen recipe: >= MIN_SESSIONS scored sessions, mean daily net return > 0 with HAC t >= MIN_T,
and at least MIN_REBALANCES rebalances executed. Output: reports/ls_forward.json (+ data/processed/ls_forward_state.json).
"""
import json
import math
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config as C
import ls_rules as R

REBALANCE_SESSIONS = 20
MIN_SESSIONS, MIN_T, MIN_REBALANCES = 63, 2.0, 3
STATE = C.PROC / "ls_forward_state.json"
OUT = C.REPORTS / "ls_forward.json"


def _adj_open(sym, start):
    f = C.RAW / "prices" / f"{sym}.parquet"
    if not f.exists():
        return pd.Series(dtype=float)
    d = pd.read_parquet(f, columns=["date", "adj_open"]).drop_duplicates("date").set_index("date")["adj_open"].dropna()
    return d[d.index >= pd.Timestamp(start)]


def _new_state(recipe_id, model_run_id):
    return {"recipe_id": recipe_id, "model_run_id": model_run_id, "equity": 1.0, "positions": {}, "pending": None,
            "last_marked": None, "last_rebalance_asof": None, "sessions": [], "rebalances": 0}


def mark(state, cal, until=None):
    """Mark held positions open-to-open for every session after last_marked up to `until` (default: latest open).
    A pending rebalance cost is charged in the first marked session."""
    if not state["positions"] or state["last_marked"] is None:
        return
    days = cal[cal > pd.Timestamp(state["last_marked"])]
    if until is not None:
        days = days[days <= pd.Timestamp(until)]
    if not len(days):
        return
    prices = {s: _adj_open(s, state["last_marked"]) for s in state["positions"]}
    prev = pd.Timestamp(state["last_marked"])
    for d in days:
        pnl, fee = 0.0, 0.0
        for s, v in list(state["positions"].items()):
            p = prices[s]
            if prev in p.index and d in p.index:
                r = p[d] / p[prev] - 1
            else:
                r = 0.0  # no quote (halt/delisting): carried flat; a sharp gap appears when quotes resume
            pnl += v * r
            state["positions"][s] = v * (1 + r)
            if v < 0:
                fee += -v * float(R.borrow_fee([state.get("si", {}).get(s, 0.0)])[0]) / 252
        fee += state.pop("cost_due", 0.0)
        ret = (pnl - fee) / state["equity"]
        state["equity"] += pnl - fee
        state["sessions"].append({"date": str(d.date()), "net": ret})
        prev = d
    state["last_marked"] = str(prev.date())


def rebalance(state, cand, asof, cal, costs):
    """Set target weights from tonight's candidates; they are traded at the next session's open (marked from then)."""
    w = R.select(cand, {s for s, v in state["positions"].items() if v > 0}, {s for s, v in state["positions"].items() if v < 0})
    if not w:
        return False
    nxt = cal[cal > pd.Timestamp(asof)]
    if not len(nxt):
        state["pending"] = {"asof": asof, "weights": w}  # entry open not yet available
        return True
    mark(state, cal, until=nxt[0])
    _enter(state, w, nxt[0], costs)
    state["last_rebalance_asof"] = asof
    return True


def _enter(state, w, day, costs):
    eq = state["equity"]
    target = {s: x * eq for s, x in w.items()}
    traded = sum(abs(target.get(s, 0.0) - state["positions"].get(s, 0.0)) * costs.get(s, 0.003) for s in set(target) | set(state["positions"]))
    state["positions"] = {s: x * eq for s, x in w.items()}
    state["cost_due"] = state.get("cost_due", 0.0) + traded  # charged in the first session marked after entry
    state["last_marked"] = str(pd.Timestamp(day).date())
    state["rebalances"] += 1


def evaluate(state, frozen):
    from research import hac_mean
    r = pd.Series([x["net"] for x in state["sessions"]], dtype=float)
    hac = hac_mean(r, 5) if len(r) > 5 else {"mean": None, "t": None}
    fails = []
    if not frozen:
        fails.append("recipe not frozen: the forward record does not count yet")
    if len(r) < MIN_SESSIONS:
        fails.append(f"forward record {len(r)}/{MIN_SESSIONS} sessions")
    if state["rebalances"] < MIN_REBALANCES:
        fails.append(f"{state['rebalances']}/{MIN_REBALANCES} rebalances executed")
    if not ((hac.get("mean") or 0) > 0 and (hac.get("t") or 0) >= MIN_T):
        fails.append(f"mean daily net not > 0 with HAC t >= {MIN_T}")
    eq = (1 + r).cumprod()
    return {"sessions_scored": int(len(r)), "rebalances": state["rebalances"], "mean_daily_net": hac.get("mean"), "hac_t": hac.get("t"),
            "cumulative_return": float(eq.iloc[-1] - 1) if len(r) else None,
            "sharpe_ann": float(r.mean() / r.std() * math.sqrt(252)) if len(r) > 2 and r.std() > 0 else None,
            "max_drawdown": float((eq / eq.cummax() - 1).min()) if len(r) else None, "failed_checks": fails,
            "strategy_forward_validated": not fails}


def run(profile="broad"):
    import recipe
    from research import name_costs
    model = json.loads((C.PROC / f"model_{profile}.json").read_text())
    frozen, _ = recipe.check(model)
    rid, mrid = model.get("recipe_id"), model["run_id"]
    state = json.loads(STATE.read_text()) if STATE.exists() else _new_state(rid, mrid)
    if state.get("recipe_id") != rid or state.get("model_run_id") != mrid:
        state = _new_state(rid, mrid)  # a new recipe or model restarts the forward record
    cal = pd.DatetimeIndex(_adj_open(C.BENCHMARK, "2000-01-01").index)
    cand = pd.read_parquet(C.PROC / f"ls_candidates_{profile}.parquet")
    asof = str(cand["as_of"].iloc[0])
    panel = pd.read_parquet(C.PROC / f"asof_panel_{profile}.parquet")
    costs = dict(zip(panel["symbol"], name_costs(panel, 100_000 * R.LS_GROSS / max(1, len(cand) * 2 * R.LS_Q)) - C.FX_FEE))
    state["si"] = {s: (0.0 if pd.isna(x) else float(x)) for s, x in zip(cand["symbol"], cand["si"])}
    if state.get("pending"):
        nxt = cal[cal > pd.Timestamp(state["pending"]["asof"])]
        if len(nxt):
            mark(state, cal, until=nxt[0])  # the old book is held until the new one is entered
            _enter(state, state["pending"]["weights"], nxt[0], costs)
            state["last_rebalance_asof"], state["pending"] = state["pending"]["asof"], None
    mark(state, cal)
    last = state.get("last_rebalance_asof")
    from provenance import sessions_between
    if last is None or sessions_between(last, asof) >= REBALANCE_SESSIONS:
        rebalance(state, cand, asof, cal, costs)
        mark(state, cal)
    ev = evaluate(state, frozen)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, default=float))
    tmp.replace(STATE)
    out = {"generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "profile": profile, "recipe_id": rid,
           "model_run_id": mrid, "rebalance_sessions": REBALANCE_SESSIONS, "as_of": asof, "equity": state["equity"],
           "positions": len(state["positions"]), "rule": __doc__.strip().split("\n\n")[-1], **ev}
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1, default=float))
    tmp.replace(OUT)
    print(json.dumps({k: v for k, v in out.items() if k != "rule"}, default=float))
    return out


if __name__ == "__main__":
    _lock = __import__("config").hold_lock()
    run(sys.argv[1] if len(sys.argv) > 1 else "broad")
