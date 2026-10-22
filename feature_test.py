"""Discovery-period tests for newly added features, without touching the holdout.

Runs the same univariate protocol as research.py (raw, sector-neutral and style-neutral ICs with HAC t-stats,
quantile spreads, BH-FDR) on pre-holdout dates only, for the named features. The FDR correction is applied jointly
with every univariate test already in reports/research_results_<profile>.json, so adding features raises the bar.
Output: reports/feature_test_<profile>.json. Models only use the features after the next full re-validation.
"""
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

import config as C
from features import FEATURES
from research import StyleNeutraliser, bh_fdr, ic_series, ic_summary, preprocess, quantile_stats, sector_neutral


def run(profile, feats):
    panel = pd.read_parquet(C.PROC / f"panel_{profile}.parquet")
    spy = pd.read_parquet(C.RAW / "prices" / f"{C.BENCHMARK}.parquet").drop_duplicates("date").set_index("date")
    cal = spy.index
    hold = cal[cal.searchsorted(cal[-1] - pd.DateOffset(years=C.HOLDOUT_YEARS))]
    panel = panel[panel["date"] < hold].reset_index(drop=True)  # discovery sample only
    base = ["log_mcap", "beta_252"]
    X = preprocess(panel, list(dict.fromkeys(feats + base)))
    Xn = sector_neutral(X, panel, feats)
    style = StyleNeutraliser(X, panel)
    dates, sectors = panel["date"].to_numpy(), panel["sector"].fillna("Unknown").to_numpy()
    recs, pv = [], []
    for f in feats:
        miss = panel[f].isna().to_numpy()
        for h in C.HORIZONS:
            yy = np.where(~miss, panel[f"y_{h}"].to_numpy(), np.nan)
            for vname, xs in [("raw", X[f]), ("sector_neutral", Xn[f]), ("style_neutral", style(X[f]))]:
                ic = ic_series(xs.to_numpy(), yy, dates)
                s = ic_summary(ic, h, boot=(h == C.PRIMARY_HORIZON))
                rec = {"feature": f, "family": FEATURES[f][0], "hyp_sign": FEATURES[f][1], "horizon": h, "variant": vname, **s}
                if vname == "raw" and h == C.PRIMARY_HORIZON:
                    rec.update(quantile_stats(xs.to_numpy(), yy, dates, sectors) or {})
                p = s["p"]
                if p is not None and rec["hyp_sign"] != 0 and s["t"] is not None:
                    p = float(stats.norm.sf(s["t"] * rec["hyp_sign"]))
                rec["p_test"] = p if p is not None else 1.0
                recs.append(rec)
        print("tested", f, flush=True)
    old = json.loads((C.REPORTS / f"research_results_{profile}.json").read_text())["univariate"]
    old_p = [u.get("p_test", 1.0) if u.get("p_test") is not None else 1.0 for u in old]
    rej, padj = bh_fdr(old_p + [r["p_test"] for r in recs], C.FDR_Q)
    for r, rj, pa in zip(recs, rej[len(old_p):], padj[len(old_p):]):
        r["fdr_reject_joint"], r["p_bh_joint"] = bool(rj), float(pa)
        r["sign_matches_hypothesis"] = r["mean"] is not None and (r["hyp_sign"] == 0 or np.sign(r["mean"]) == r["hyp_sign"])
    out = {"profile": profile, "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sample_end_exclusive": str(hold.date()),
           "tests_new": len(recs), "tests_in_joint_fdr": len(old_p) + len(recs), "results": recs}
    (C.REPORTS / f"feature_test_{profile}.json").write_text(json.dumps(out, indent=1, default=float))
    passed = [r for r in recs if r["fdr_reject_joint"] and r["sign_matches_hypothesis"]]
    print(f"{profile}: {len(passed)} of {len(recs)} new tests pass joint BH-FDR with the hypothesised sign")
    for r in sorted(passed, key=lambda r: -abs(r["t"]))[:15]:
        print(f"  {r['variant']:15s} {r['feature']:22s} h={r['horizon']:3d} IC {r['mean']:+.4f} t {r['t']:+.2f}")
    return out


if __name__ == "__main__":
    _lock = __import__("config").hold_lock()
    new = ["ivol_60", "resid_mom_12_1", "max_ret_21", "skew_252", "r2_mkt_252", "vol_ratio_20_60", "vol_of_vol_60", "ar1_60",
           "variance_ratio_120", "tail_ratio_60", "beta_instability_252"]
    run(sys.argv[1] if len(sys.argv) > 1 else "large", new)
