"""Healthcare / biotechnology module (spec 4B).

Per pharma/biotech issuer (SIC 2833-2836, 8731), from data public at the as-of date:
  cash, annual operating cash burn, runway (years), share-count growth (dilution history), R&D intensity,
  dilution needed to fund 24 more months at the current burn and price, registered trial pipeline (phase mix,
  concentration in the top condition), phase 2/3 primary completions in the next 180 days, FDA approvals.
Cohort statistics: share of registered phase 2/3 trials that ended TERMINATED/WITHDRAWN vs COMPLETED, by phase.
These are registry outcomes, not efficacy results: they are NOT clinical success probabilities.

Scenario model (spec): E[event return] = sum_j P(outcome j) x R(outcome j). R for an approval is estimated from
openFDA approval-day reactions (catalysts.py); P(outcome) and R(failure) need licensed phase-transition and
readout-outcome data (e.g. Citeline / BIO), so expected_event_return() returns None unless both are supplied.
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config as C
from features import FUND_DEFS, fundamentals_asof, load_facts, load_prices


def expected_event_return(scenarios):
    """scenarios: [{"p": prob, "ret": conditional return}] summing to 1; None when any input is missing."""
    if not scenarios or any(s.get("p") is None or s.get("ret") is None for s in scenarios) or abs(sum(s["p"] for s in scenarios) - 1) > 1e-6:
        return None
    return float(sum(s["p"] * s["ret"] for s in scenarios))


def run():
    sic = pd.read_parquet(C.PROC / "sic.parquet").set_index("cik")["sic"]
    uni = pd.read_parquet(C.PROC / "universe.parquet")
    s = pd.to_numeric(uni["cik"].map(sic), errors="coerce")
    bio = uni[s.between(2833, 2836) | (s == 8731)]
    P, sym2cik = load_prices(C.PROFILES["broad"]["min_dollar_adv"], symbols=set(bio["yahoo"]))
    syms = list(P["adjclose"].columns)
    ciks = [sym2cik[x] for x in syms]
    today = P["adjclose"].index[-1]
    f = fundamentals_asof(pd.DatetimeIndex([today]), ciks, load_facts(ciks))
    trials = pd.read_parquet(C.PROC / "clinical_trials.parquet") if (C.PROC / "clinical_trials.parquet").exists() else pd.DataFrame()
    fda = pd.read_parquet(C.PROC / "fda_approvals.parquet") if (C.PROC / "fda_approvals.parquet").exists() else pd.DataFrame()
    rows = []
    for sym, cik in zip(syms, ciks):
        cash, cfo, sh, shp = (f[k][cik].iat[0] for k in ["cash", "cfo", "shares", "shares_prev"])
        rnd, rev = f["rnd"][cik].iat[0], f["rev"][cik].iat[0]
        px = P["close"][sym].dropna()
        mcap = sh * px.iat[-1] if len(px) and pd.notna(sh) else np.nan
        burn = -cfo if pd.notna(cfo) and cfo < 0 else 0.0
        runway = min(cash / burn, 10.0) if burn > 0 and pd.notna(cash) else (10.0 if pd.notna(cfo) else np.nan)
        need = max(2 * burn - (cash if pd.notna(cash) else 0), 0) if burn > 0 else 0.0
        t = trials[trials["cik"] == cik] if len(trials) else pd.DataFrame()
        ph = t["phase"].fillna("") if len(t) else pd.Series(dtype=str)
        active = t[t["status"].isin(["RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION", "NOT_YET_RECRUITING"])] if len(t) else t
        cond = active["conditions"].fillna("").str.split(";").str[0] if len(active) else pd.Series(dtype=str)
        upcoming = active[active["phase"].fillna("").str.contains("PHASE2|PHASE3") & active["primary_completion"].between(today, today + pd.Timedelta(days=180))] if len(active) else active
        rows.append({"symbol": sym, "cik": cik, "mcap": mcap, "cash": cash, "annual_burn": burn, "runway_years": runway,
                     "share_growth_yoy": sh / shp - 1 if pd.notna(sh) and pd.notna(shp) and shp > 0 else np.nan,
                     "rnd_to_revenue": rnd / rev if pd.notna(rnd) and pd.notna(rev) and rev > 0 else np.nan,
                     "dilution_to_fund_24m_pct_mcap": need / mcap if mcap and mcap > 0 else np.nan,
                     "active_trials": int(len(active)), "active_phase3": int(active["phase"].fillna("").str.contains("PHASE3").sum()) if len(active) else 0,
                     "top_condition_share": float(cond.value_counts(normalize=True).iloc[0]) if len(cond) else np.nan,
                     "ph2_3_completions_next_180d": int(len(upcoming)), "fda_approvals_total": int((fda["cik"] == cik).sum()) if len(fda) else 0})
    df = pd.DataFrame(rows)
    cohort = {}
    if len(trials):
        done = trials[trials["status"].isin(["COMPLETED", "TERMINATED", "WITHDRAWN"])]
        for phase in ["PHASE1", "PHASE2", "PHASE3"]:
            x = done[done["phase"].fillna("").str.contains(phase)]
            if len(x) >= 30:
                cohort[phase] = {"n": int(len(x)), "share_terminated_or_withdrawn": float(x["status"].isin(["TERMINATED", "WITHDRAWN"]).mean())}
    out = {"generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "as_of": str(today.date()), "issuers": int(len(df)),
           "short_runway_lt_1y": df[df["runway_years"] < 1][["symbol", "runway_years", "dilution_to_fund_24m_pct_mcap"]].sort_values("runway_years").head(25).to_dict(orient="records"),
           "upcoming_ph2_3_completions": df[df["ph2_3_completions_next_180d"] > 0][["symbol", "ph2_3_completions_next_180d", "active_phase3"]].sort_values("ph2_3_completions_next_180d", ascending=False).head(25).to_dict(orient="records"),
           "registry_cohorts": cohort,
           "scenario_model": {"formula": "E[R] = sum_j P(outcome j) x R(outcome j)", "probabilities": "NOT_COMPUTED: needs licensed phase-transition / readout-outcome data",
                              "approval_reaction": "see catalysts.json historical_responses.fda_approval", "failure_reaction": "NOT_COMPUTED: rejections not in openFDA"}}
    df.to_parquet(C.REPORTS / "biotech_issuers.parquet", index=False)
    (C.REPORTS / "biotech.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k not in ("short_runway_lt_1y", "upcoming_ph2_3_completions")}, indent=1, default=float))
    return out


if __name__ == "__main__":
    _lock = __import__("config").hold_lock()
    run()
