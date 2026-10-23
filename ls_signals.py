"""Live market-neutral long/short inputs for the Alpaca paper bot: signals/ls_latest.json.

Per name of the chosen profile: sector, ensemble score, Blume beta, short interest, borrow-fee tier. The bot applies
ls_rules.select() with its own holdings (turnover buffer) and Alpaca's shortable / easy-to-borrow flags.
Score = mean of the 5- and 20-session ensemble ranks (LS_SCORE), chosen on walk-forward results only.
Broker targets require cumulative research and forward validation. Research-only
output contains no executable names. It never feeds the Trading 212 bot.
"""
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config as C
import ls_rules as R

LS_PROFILE = "broad"
LS_SCORE = "20"    # 20-session ensemble rank: best walk-forward variant (ls_grid 2026-10-22)


def strategy_preview(weights):
    """Selection can produce numpy scalars; exported counts must be JSON numbers."""
    return {"n_long": int(sum(w > 0 for w in weights.values())),
            "n_short": int(sum(w < 0 for w in weights.values())),
            "gross": float(sum(abs(w) for w in weights.values())),
            "net": float(sum(weights.values()))}


def run(profile=LS_PROFILE):
    ap = C.PROC / f"asof_panel_{profile}.parquet"
    panel = pd.read_parquet(ap)
    fc = pd.read_parquet(C.REPORTS / "forecast_latest.parquet")
    fc = fc[fc["profile"] == profile].pivot_table(index="symbol", columns="horizon", values="ensemble_rank")
    score = fc[[5, 20]].mean(axis=1) if LS_SCORE == "blend" else fc[int(LS_SCORE)]
    p = panel.set_index("symbol")
    liquid = (p["adv_dollar"] >= C.PROFILES[profile]["min_dollar_adv"]) & (p["mcap"] >= C.PROFILES[profile]["min_mcap"]) & (p["quality_score"] >= 0.7)
    p = p[liquid]
    df = pd.DataFrame({"sector": p["sector"].fillna("Unknown"), "score": score.reindex(p.index), "beta": p["beta_252"], "si": p["si_ratio"]}).dropna(subset=["score"])
    df["sector_rank"] = df.groupby("sector")["score"].rank(pct=True)
    df["borrow_fee_ann"] = R.borrow_fee(df["si"])
    asof = str(pd.Timestamp(panel["date"].max()).date())
    research = C.REPORTS / f"ls_research_{profile}.json"
    verdict = json.loads(research.read_text())["results"].get(f"ls_rank_{LS_SCORE}") if research.exists() else None
    import recipe
    from signals import forward_gate, audit_ok, dataset_version
    from deployment_policy import cumulative
    model = json.loads((C.PROC / f"model_{profile}.json").read_text())
    frozen, reason = recipe.check(model)
    cross_gate = forward_gate(profile, C.PRIMARY_HORIZON)[0]
    audit = audit_ok(dataset_version(), profile)
    # The L/S construction has a different payoff/cost structure. A validated
    # cross-sectional ranking alone cannot authorise this strategy.
    ls_artifact = json.loads(research.read_text()) if research.exists() else {}
    ls_gate = ls_artifact.get("gates", {}).get(f"ls_rank_{LS_SCORE}") if ls_artifact.get("recipe_run_id") == model["run_id"] else None
    # Historical weekly simulations cannot validate the bot's 20-session book.
    # A cross-sectional IC record also does not validate a constructed L/S book.
    if ls_artifact.get("params", {}).get("rebalance_sessions") != 20 or not ls_artifact.get("strategy_forward_validated", False):
        ls_gate = None
    deployment = cumulative(ls_gate, cross_gate, data_ok=audit[0], recipe_ok=frozen)
    import overlay  # analyst vetoes remove a name from the vetoed side of the book, never add one
    _tk = [__import__("historical_data").ticker_for_security(x, asof) for x in df.index]
    df["veto_long"], df["veto_short"] = overlay.veto_flags(_tk, overlay.active_vetoes())
    doc = {"deployment": deployment, "recipe_id": model.get("recipe_id"), "schema": "ls-1.0", "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "as_of": asof, "profile": profile,
           "score": LS_SCORE, "mode": "VALIDATED_PAPER" if not deployment["failed_checks"] else "RESEARCH_ONLY", "params": {"q": R.LS_Q, "buffer": R.LS_BUFFER, "gross": R.LS_GROSS, "si_max_short": R.SI_MAX_SHORT,
                                                                       "max_name_weight": R.MAX_NAME_WEIGHT},
           "research": verdict,
           "names": [{"ticker": __import__("historical_data").ticker_for_security(s, asof), "security_id": s, "sector": r.sector, "score": float(r.score), "sector_rank": float(r.sector_rank),
                      "beta": None if pd.isna(r.beta) else float(r.beta), "si": None if pd.isna(r.si) else float(r.si), "borrow_fee_ann": float(r.borrow_fee_ann),
                      "veto_long": bool(r.veto_long), "veto_short": bool(r.veto_short)}
                     for s, r in df.iterrows()]}
    preview = R.select(df.reset_index().rename(columns={"index": "symbol"}))
    doc["preview_without_holdings"] = strategy_preview(preview)
    if deployment["failed_checks"]:
        doc["names"] = []  # no broker targets; offline L/S backtests remain diagnostics
    C.SIGNALS.mkdir(exist_ok=True)
    tmp = C.SIGNALS / "ls_latest.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(C.SIGNALS / "ls_latest.json")
    print(f"ls_signals {profile} as_of={asof} names={len(df)} preview={doc['preview_without_holdings']}")
    return doc


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else LS_PROFILE)
