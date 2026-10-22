"""Weekly production refit with a frozen recipe (learning, not judging).

Rebuilds each profile's point-in-time panel with the latest data and retrains every model of every horizon on all
labelled rows, using exactly the recipe fixed at the last full validation (model_<profile>.json): the same feature
list, model types, ensemble members, hyper-parameters and cost settings. It never evaluates anything and never reads
or changes the holdout or the gate; whether signals may trade is decided by the forward record (paper_trading.py).
Previous weights are archived under models/archive/<timestamp>/ and every refit is appended to ledger/refits.jsonl.
"""
import json
import shutil
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

import config as C
from features import build_panel_chunked
from research import MISSING_GROUPS, MODELS_DIR, StyleNeutraliser, fit_predict, name_costs, preprocess, sector_neutral


def refit(profile):
    model = json.loads((C.PROC / f"model_{profile}.json").read_text())
    import recipe
    frozen, why = recipe.check(model)
    if not frozen:
        raise ValueError(why)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    panel, meta = build_panel_chunked(profile=profile)
    panel.to_parquet(C.PROC / f"panel_{profile}.parquet", index=False)
    (C.PROC / f"panel_meta_{profile}.json").write_text(json.dumps(meta, indent=1))
    feats = model["feats"]  # frozen at validation: new features enter only through a full re-validation
    X = preprocess(panel, feats)
    Xn = sector_neutral(X, panel, feats)
    Xm = pd.concat([Xn[feats], X[[k for k in MISSING_GROUPS if k in X]]], axis=1).reindex(columns=model["features"], fill_value=0.0)
    sectors = panel["sector"].fillna("Unknown").to_numpy()
    cost = name_costs(panel, model.get("order_size_usd") or 600.0)
    rt_cost = float(cost.mean() * 2)
    style = StyleNeutraliser(X, panel) if any("factor_resid" in model["horizons"][str(h)]["names"] for h in C.HORIZONS) else None
    archive = MODELS_DIR / "archive" / stamp
    archive.mkdir(parents=True, exist_ok=True)
    rows = {}
    for h in C.HORIZONS:
        y = panel[f"y_{h}"].to_numpy().astype("float64")
        for n in model["horizons"][str(h)]["names"]:
            yt = y
            if n == "factor_resid":
                yt = style(pd.Series(np.nan_to_num(y), index=X.index)).to_numpy().astype("float64")
                yt[np.isnan(y)] = np.nan
            ok = ~np.isnan(yt)
            dest = MODELS_DIR / f"{profile}_h{h}_{n}.joblib"
            if dest.exists():
                shutil.copy2(dest, archive / dest.name)
            m, _ = fit_predict(n, Xm[ok], yt[ok], Xm.iloc[:2], sectors[ok], sectors[:2], rt_cost)
            joblib.dump(m, dest)
            rows[f"h{h}_{n}"] = int(ok.sum())
        print(profile, "h", h, "refit", model["horizons"][str(h)]["names"], flush=True)
    recipe.snapshot_weights(model)
    model["weights_refit_utc"] = stamp
    model["weights_trained_through"] = str(pd.Timestamp(panel["date"].max()).date())
    (C.PROC / f"model_{profile}.json").write_text(json.dumps(model, indent=1, default=float))
    with (C.LEDGER / "refits.jsonl").open("a") as f:
        f.write(json.dumps({"at": stamp, "profile": profile, "recipe_run_id": model["run_id"], "trained_through": model["weights_trained_through"],
                            "rows_per_model": rows, "archive": str(archive)}) + "\n")
    return model["weights_trained_through"]


if __name__ == "__main__":
    _lock = __import__("config").hold_lock()
    for p in (sys.argv[1:] or [p for p in C.PROFILES if (C.PROC / f"model_{p}.json").exists()]):
        print(p, "trained through", refit(p), flush=True)
