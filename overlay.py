"""Analyst gatekeeper overlay (quantamental process) and research notebook.

How institutional quantamental desks combine the two halves: the model ranks the universe, a fundamental analyst reviews
the names it wants to trade and may VETO a name (remove it) but never ADD one, and every override is recorded and later
scored to learn whether the human adds value. Research evidence is documented as it is gathered (CFA Standard II(A):
a mosaic conclusion must be traceable to public or non-material information).

  overlay.py review                     write reports/overlay_review_latest.{md,csv}: top/bottom names with evidence
  overlay.py veto TICKER --reason CODE --note TEXT [--side long|short|any] [--days 20] [--analyst NAME]
  overlay.py lift VETO_ID [--note TEXT]
  overlay.py note TICKER TEXT [--source URL]       append to the research notebook
  overlay.py list                       active vetoes
  overlay.py score                      reports/overlay_scorecard.json: did the vetoes help?

Vetoes only ever turn a LONG into NO_SIGNAL (signals.py, intraday_watch.py) or drop a name from one side of the
long/short book (ls_rules.select); they never create or enlarge a position. Every action is appended to
ledger/overlay_actions.jsonl; overlay/vetoes.csv is the current register.
"""
import argparse
import json
import math
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config as C
from provenance import holidays

DIR = C.ROOT / "overlay"
VETOES = DIR / "vetoes.csv"
NOTES = C.LEDGER / "research_notes.jsonl"
ACTIONS = C.LEDGER / "overlay_actions.jsonl"
REASONS = ("accounting", "pending_event", "data_error", "governance", "thesis_broken", "liquidity", "legal", "other")
SIDES = ("long", "short", "any")
COLS = ["veto_id", "created_utc", "ticker", "side", "reason", "note", "analyst", "start", "expires", "lifted_utc", "lift_note"]


def norm(t):
    return str(t).strip().upper().replace("-", ".")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log(event):
    C.LEDGER.mkdir(exist_ok=True)
    with ACTIONS.open("a") as f:
        f.write(json.dumps({"at_utc": _now(), **event}) + "\n")


def load():
    if not VETOES.exists():
        return pd.DataFrame(columns=COLS)
    return pd.read_csv(VETOES, dtype=str, keep_default_na=False).reindex(columns=COLS, fill_value="")


def _save(df):
    DIR.mkdir(exist_ok=True)
    tmp = VETOES.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(VETOES)


def add_sessions(day, n):
    d = np.datetime64(pd.Timestamp(day).date(), "D")
    hol = [np.datetime64(x) for x in holidays(pd.Timestamp(day).date(), (pd.Timestamp(day) + pd.Timedelta(days=3 * n + 30)).date())]
    return str(np.busday_offset(d, n, roll="forward", holidays=hol))


def veto(ticker, reason, note, side="long", days=20, analyst="owner"):
    if reason not in REASONS:
        raise ValueError(f"reason must be one of {REASONS}")
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}")
    if not note or not note.strip():
        raise ValueError("a veto needs a written reason (note)")
    start = str(pd.Timestamp.now(tz="America/New_York").date())
    row = {"veto_id": uuid.uuid4().hex[:10], "created_utc": _now(), "ticker": norm(ticker), "side": side, "reason": reason,
           "note": note.strip(), "analyst": analyst, "start": start, "expires": add_sessions(start, int(days)), "lifted_utc": "", "lift_note": ""}
    _save(pd.concat([load(), pd.DataFrame([row])], ignore_index=True))
    _log({"event": "veto", **row})
    return row


def lift(veto_id, note=""):
    df = load()
    m = df["veto_id"] == veto_id
    if not m.any():
        raise KeyError(veto_id)
    df.loc[m, ["lifted_utc", "lift_note"]] = [_now(), note]
    _save(df)
    _log({"event": "lift", "veto_id": veto_id, "note": note})


def note(ticker, text, source=""):
    rec = {"at_utc": _now(), "ticker": norm(ticker), "note": text, "source": source}
    C.LEDGER.mkdir(exist_ok=True)
    with NOTES.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def active_vetoes(day=None):
    """{ticker: {"side", "reason", "veto_id", "note"}} for vetoes in force on `day` (New York date)."""
    day = str(pd.Timestamp(day).date()) if day is not None else str(pd.Timestamp.now(tz="America/New_York").date())
    out = {}
    for r in load().itertuples():
        if not r.lifted_utc and r.start <= day < r.expires:
            out[r.ticker] = {"side": r.side, "reason": r.reason, "veto_id": r.veto_id, "note": r.note}
    return out


def apply_to_signals(signals, vetoes):
    """LONG -> NO_SIGNAL for long/any vetoes. Never changes anything else. Returns the number of signals vetoed."""
    n = 0
    for s in signals:
        v = vetoes.get(norm(s["asset"]["ticker"]))
        if v and v["side"] in ("long", "any") and s["signal"]["direction"] == "LONG":
            s["signal"]["direction"] = "NO_SIGNAL"
            s["signal"]["reason"] = f"analyst veto {v['veto_id']} ({v['reason']}): {v['note']}"
            n += 1
    return n


def veto_flags(tickers, vetoes):
    """(veto_long, veto_short) boolean lists for the long/short book."""
    vl = [bool(vetoes.get(norm(t)) and vetoes[norm(t)]["side"] in ("long", "any")) for t in tickers]
    vs = [bool(vetoes.get(norm(t)) and vetoes[norm(t)]["side"] in ("short", "any")) for t in tickers]
    return vl, vs


# ---------------------------------------------------------------- review sheet
def review(n=25, profile="broad", horizon=C.PRIMARY_HORIZON):
    fc = pd.read_parquet(C.REPORTS / "forecast_latest.parquet")
    fc = fc[(fc["profile"] == profile) & (fc["horizon"] == horizon)].set_index("symbol")
    ap = pd.read_parquet(C.PROC / f"asof_panel_{profile}.parquet").set_index("symbol")
    sig = json.loads((C.SIGNALS / "latest.json").read_text())
    by_t = {norm(s["asset"]["ticker"]): s for s in sig["signals"] if str(s.get("model_id", "")).endswith(f"-{profile}")}
    liquid = ap[(ap["adv_dollar"] >= C.PROFILES[profile]["min_dollar_adv"]) & (ap["mcap"] >= C.PROFILES[profile]["min_mcap"])].index
    fc = fc[fc.index.isin(liquid)]
    vet = active_vetoes()
    notes = pd.read_json(NOTES, lines=True) if NOTES.exists() and NOTES.stat().st_size else pd.DataFrame(columns=["ticker", "at_utc", "note"])
    rows = []
    for side, part in [("LONG candidate", fc.nlargest(n, "ensemble_rank")), ("SHORT candidate", fc.nsmallest(n, "ensemble_rank"))]:
        for sym, p in part.iterrows():
            t = norm(sym)
            a, s = ap.loc[sym], by_t.get(t, {})
            nn = notes[notes["ticker"] == t].tail(1)
            g = lambda k: (None if k not in a or pd.isna(a[k]) else float(a[k]))
            rows.append({"side": side, "ticker": t, "sector": a.get("sector"), "rank": round(float(p["ensemble_rank"]), 3),
                         "model_disagreement": None if pd.isna(p.get("model_disagreement")) else round(float(p["model_disagreement"]), 2),
                         "drivers": "; ".join(s.get("signal", {}).get("key_drivers", [])[:3]),
                         "against": "; ".join(s.get("signal", {}).get("contradictory_evidence", [])[:3]),
                         "short_interest": g("si_ratio"), "insider_net_90d_mcap": g("ins_net_90"), "filing_similarity": g("text_sim_cos"),
                         "risk_factor_similarity": g("text_sim_rf"), "earnings_8k_last_5d": g("earn_recent"),
                         "active_veto": (vet.get(t) or {}).get("veto_id", ""), "last_note": nn["note"].iloc[0] if len(nn) else "",
                         "edgar": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={int(a['cik'])}&type=10-&count=20"})
    df = pd.DataFrame(rows)
    df.to_csv(C.REPORTS / "overlay_review_latest.csv", index=False)
    asof = sig.get("as_of_utc", "")[:10]
    md = [f"# Analyst review sheet - {profile}, {horizon}-session ranks, as of {asof}", "",
          "Veto a name the evidence does not support: `overlay.py veto TICKER --side long|short --reason CODE --note \"why\"`.",
          f"Reason codes: {', '.join(REASONS)}. A veto removes a name for 20 sessions by default; it can never add one.",
          "Check: recent 8-K/10-Q events, accounting or restatement flags, pending deals or litigation, data errors.", ""]
    for side in ("LONG candidate", "SHORT candidate"):
        md += [f"## {side}s", "", "| ticker | sector | rank | drivers | against | SI | filing sim | veto | last note |", "|---|---|---|---|---|---|---|---|---|"]
        for r in df[df["side"] == side].itertuples():
            f = lambda x, fmt: "" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, fmt)
            md.append(f"| [{r.ticker}]({r.edgar}) | {r.sector} | {r.rank:.3f} | {r.drivers} | {r.against} | {f(r.short_interest, '.1%')} | "
                      f"{f(r.filing_similarity, '.3f')} | {r.active_veto} | {str(r.last_note)[:60]} |")
        md.append("")
    (C.REPORTS / "overlay_review_latest.md").write_text("\n".join(md))
    return df


# ---------------------------------------------------------------- scorecard
def _excess(ticker, start, end):
    """Open-to-open return from the session after `start` to `end` (or latest), minus SPY over the same window."""
    def series(sym):
        f = C.RAW / "prices" / f"{sym}.parquet"
        if not f.exists():
            return None
        return pd.read_parquet(f, columns=["date", "adj_open"]).drop_duplicates("date").set_index("date")["adj_open"].dropna()
    px, spy = series(ticker.replace(".", "-")), series(C.BENCHMARK)
    if px is None or spy is None:
        return None
    px, spy = px[px.index > pd.Timestamp(start)], spy[spy.index > pd.Timestamp(start)]
    px, spy = px[px.index <= pd.Timestamp(end)], spy[spy.index <= pd.Timestamp(end)]
    common = px.index.intersection(spy.index)
    if len(common) < 2:
        return None
    a, b = common[0], common[-1]
    return float((px[b] / px[a] - 1) - (spy[b] / spy[a] - 1)), int(len(common))


def score():
    """Value added by each veto: for a long veto, the excess return the model would have earned and the analyst avoided
    (negative excess = the veto helped); for a short veto, the reverse. Summary with a t-statistic."""
    today = str(pd.Timestamp.now(tz="America/New_York").date())
    rows = []
    for r in load().itertuples():
        end = min(r.expires, r.lifted_utc[:10] or r.expires, today)
        ex = _excess(r.ticker, r.start, end)
        if ex is None:
            continue
        sign = -1.0 if r.side == "long" else (1.0 if r.side == "short" else 0.0)
        rows.append({"veto_id": r.veto_id, "ticker": r.ticker, "side": r.side, "reason": r.reason, "start": r.start, "end": end,
                     "sessions": ex[1], "excess_return": ex[0], "value_added": sign * ex[0] if sign else None, "complete": end == r.expires})
    df = pd.DataFrame(rows)
    va = df["value_added"].dropna() if len(df) else pd.Series(dtype=float)
    summary = {"vetoes_scored": int(len(df)), "complete": int(df["complete"].sum()) if len(df) else 0,
               "mean_value_added": float(va.mean()) if len(va) else None, "hit_rate": float((va > 0).mean()) if len(va) else None,
               "t": float(va.mean() / va.std() * math.sqrt(len(va))) if len(va) > 2 and va.std() > 0 else None,
               "by_reason": df.groupby("reason")["value_added"].agg(["count", "mean"]).to_dict(orient="index") if len(df) else {},
               "rule": "the overlay earns a place in the process only if completed vetoes add value with t >= 2 over >= 20 vetoes"}
    out = {"generated_at_utc": _now(), "summary": summary, "vetoes": df.to_dict(orient="records")}
    (C.REPORTS / "overlay_scorecard.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("veto")
    v.add_argument("ticker")
    v.add_argument("--reason", required=True, choices=REASONS)
    v.add_argument("--note", required=True)
    v.add_argument("--side", default="long", choices=SIDES)
    v.add_argument("--days", type=int, default=20)
    v.add_argument("--analyst", default="owner")
    li = sub.add_parser("lift")
    li.add_argument("veto_id")
    li.add_argument("--note", default="")
    no = sub.add_parser("note")
    no.add_argument("ticker")
    no.add_argument("text")
    no.add_argument("--source", default="")
    sub.add_parser("list")
    sub.add_parser("review")
    sub.add_parser("score")
    a = ap.parse_args()
    if a.cmd == "veto":
        print(json.dumps(veto(a.ticker, a.reason, a.note, a.side, a.days, a.analyst), indent=1))
    elif a.cmd == "lift":
        lift(a.veto_id, a.note)
    elif a.cmd == "note":
        print(json.dumps(note(a.ticker, a.text, a.source)))
    elif a.cmd == "list":
        print(json.dumps(active_vetoes(), indent=1))
    elif a.cmd == "review":
        df = review()
        print(f"review sheet: {len(df)} names -> reports/overlay_review_latest.md")
    elif a.cmd == "score":
        print(json.dumps(score()["summary"], indent=1, default=float))


if __name__ == "__main__":
    main()
