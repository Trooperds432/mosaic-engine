"""Mutual-information screening (adapted from the "Mutual information vs correlation" notebook).

For every feature, on the discovery sample only (pre-holdout) at the primary horizon: k-NN mutual information
(sklearn, k=3) between the feature's sector-neutral rank and the forward excess return demeaned per date and divided by the stock's
volatility known at t (so dependence through return dispersion alone is not counted), on a fixed
random subsample. Significance: z-score against a permutation null (returns shuffled within dates), BH-FDR across features.
Features whose MI is significant while their rank correlation (IC) is not are flagged "nonlinear only": dependence
that the linear models and IC tests would miss. Output reports/mi_screen_<profile>.json.
"""
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

import config as C
from features import FEATURES
from research import bh_fdr, preprocess, sector_neutral

N_ROWS, N_PERM = 60_000, 10


def run(profile="large", h=None):
    h = h or C.PRIMARY_HORIZON
    panel = pd.read_parquet(C.PROC / f"panel_{profile}.parquet")
    model = json.loads((C.PROC / f"model_{profile}.json").read_text())
    feats = model["feats"]
    spy = pd.read_parquet(C.RAW / "prices" / f"{C.BENCHMARK}.parquet").drop_duplicates("date").set_index("date")
    cal = spy.index
    hold = cal[cal.searchsorted(cal[-1] - pd.DateOffset(years=C.HOLDOUT_YEARS))]
    panel = panel[(panel["date"] < hold) & panel[f"y_{h}"].notna()].reset_index(drop=True)
    X = sector_neutral(preprocess(panel, feats), panel, feats)
    y = panel[f"y_{h}"] - panel.groupby("date")[f"y_{h}"].transform("mean")
    # divide by each stock's volatility known at t: otherwise MI mostly detects that volatile stocks have more dispersed
    # returns (risk), not predictability of the return itself
    vol_h = (panel["vol_60"] * np.sqrt(h / 252)).clip(lower=0.005)
    y = (y / vol_h).where(panel["vol_60"].notna())
    keep = y.notna().to_numpy()
    panel, X, y = panel[keep].reset_index(drop=True), X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
    rng = np.random.default_rng(11)
    idx = np.sort(rng.choice(len(panel), size=min(N_ROWS, len(panel)), replace=False))
    Xs, ys, ds = X.iloc[idx], y.iloc[idx].to_numpy(), panel["date"].iloc[idx].to_numpy()
    mi = mutual_info_regression(Xs[feats].to_numpy(), ys, n_neighbors=3, random_state=0)
    null = []
    for k in range(N_PERM):
        yp = pd.Series(ys).groupby(ds).transform(lambda v: v.sample(frac=1, random_state=k).to_numpy()).to_numpy()
        null.append(mutual_info_regression(Xs[feats].to_numpy(), yp, n_neighbors=3, random_state=0))
        print("permutation", k + 1, flush=True)
    null = np.array(null)
    # z-score against the permutation null (an exact permutation p-value would need thousands of shuffles)
    from scipy import stats
    z = (mi - null.mean(axis=0)) / np.maximum(null.std(axis=0, ddof=1), 1e-9)
    p = 1 - stats.norm.cdf(z)
    rej, padj = bh_fdr(p, C.FDR_Q)
    res = json.loads((C.REPORTS / f"research_results_{profile}.json").read_text())
    ic = {u["feature"]: u for u in res["univariate"] if u["horizon"] == h and u["variant"] == "sector_neutral"}
    rows = []
    for j, f in enumerate(feats):
        u = ic.get(f, {})
        rows.append({"feature": f, "family": FEATURES[f][0], "mi": float(mi[j]), "mi_null_mean": float(null[:, j].mean()),
                     "mi_excess": float(mi[j] - null[:, j].mean()), "z_vs_null": float(z[j]), "p_perm": float(p[j]), "p_bh": float(padj[j]), "mi_significant": bool(rej[j]),
                     "ic": u.get("mean"), "ic_t": u.get("t"), "ic_fdr": u.get("fdr_reject"),
                     "nonlinear_only": bool(rej[j]) and not u.get("fdr_reject", False)})
    rows.sort(key=lambda r: -r["mi_excess"])
    out = {"profile": profile, "horizon": h, "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "rows_sampled": int(len(idx)), "permutations": N_PERM, "p_method": "normal approximation to the permutation null", "features": rows,
           "nonlinear_only": [r["feature"] for r in rows if r["nonlinear_only"]],
           "interpretation": "MI measures any dependence (including in tails/dispersion beyond the volatility scaling), not predictability of "
                             "the mean; 'nonlinear_only' lists candidates for non-linear models, which (LightGBM, GAM) are already in the suite."}
    (C.REPORTS / f"mi_screen_{profile}.json").write_text(json.dumps(out, indent=1, default=float))
    print("top MI:", [(r["feature"], round(r["mi_excess"], 4), r["p_perm"]) for r in rows[:8]], "| nonlinear only:", out["nonlinear_only"], flush=True)
    return out


if __name__ == "__main__":
    _lock = __import__("config").hold_lock()
    run(sys.argv[1] if len(sys.argv) > 1 else "large")
