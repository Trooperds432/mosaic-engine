"""Live market-neutral long/short inputs for the Alpaca paper bot: signals/ls_latest.json.

Per name of the chosen profile: sector, ensemble score, Blume beta, short interest, borrow-fee tier. The bot applies
ls_rules.select() with its own holdings (turnover buffer) and Alpaca's shortable / easy-to-borrow flags.
Score = mean of the 5- and 20-session ensemble ranks (LS_SCORE), chosen on walk-forward results only.
This is a paper forward test: the file carries the research verdict but the paper bot trades it regardless, so the
executed Alpaca record becomes the out-of-sample evidence. It never feeds the Trading 212 bot.
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
    doc = {"schema": "ls-1.0", "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "as_of": asof, "profile": profile,
           "score": LS_SCORE, "mode": "PAPER_FORWARD_TEST", "params": {"q": R.LS_Q, "buffer": R.LS_BUFFER, "gross": R.LS_GROSS, "si_max_short": R.SI_MAX_SHORT,
                                                                       "max_name_weight": R.MAX_NAME_WEIGHT},
           "research": verdict,
           "names": [{"ticker": s.replace("-", "."), "sector": r.sector, "score": float(r.score), "sector_rank": float(r.sector_rank),
                      "beta": None if pd.isna(r.beta) else float(r.beta), "si": None if pd.isna(r.si) else float(r.si), "borrow_fee_ann": float(r.borrow_fee_ann)}
                     for s, r in df.iterrows()]}
    preview = R.select(df.reset_index().rename(columns={"index": "symbol"}))
    doc["preview_without_holdings"] = {"n_long": sum(w > 0 for w in preview.values()), "n_short": sum(w < 0 for w in preview.values()),
                                       "gross": float(sum(abs(w) for w in preview.values())), "net": float(sum(preview.values()))}
    C.SIGNALS.mkdir(exist_ok=True)
    tmp = C.SIGNALS / "ls_latest.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(C.SIGNALS / "ls_latest.json")
    print(f"ls_signals {profile} as_of={asof} names={len(df)} preview={doc['preview_without_holdings']}")
    return doc


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else LS_PROFILE)
