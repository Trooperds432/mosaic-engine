"""Pre-registered spread check (prereg/spread_check_v1.json).

Compares the half-spread used by research.name_costs (clip(Corwin-Schultz/2, 3 bp, 2%)) with the
trailing 60-session mean of LSEG end-of-day quoted half-spreads, on panel_large rows. Measurement only:
it does not change the cost model.

usage: python spread_check.py BIDASK_CSV [OUT_JSON]
"""
import json
import sys

import numpy as np
import pandas as pd

import config as C
from research import name_costs

BUNDLE = "data/raw/historical/8433bd236228ba351c9a/security_master.parquet"
WINDOW, MIN_DAYS, MAX_REL = 60, 40, 0.20


def quoted_half_spread(bidask):
    q = bidask.dropna(subset=["date", "bid", "ask"]).copy()
    q["date"] = pd.to_datetime(q["date"])
    q = q[(q.bid > 0) & (q.ask > q.bid)]
    mid = (q.bid + q.ask) / 2
    q["q"] = (q.ask - q.bid) / (2 * mid)
    q = q[(q.ask - q.bid) / mid < MAX_REL]
    q = q.sort_values(["ric", "date"]).drop_duplicates(["ric", "date"])
    q["hs_quoted"] = q.groupby("ric")["q"].transform(lambda s: s.rolling(WINDOW, min_periods=MIN_DAYS).mean())
    return q[["ric", "date", "hs_quoted", "src"]]


def summarise(d):
    r = d.hs_model / d.hs_quoted
    return {"rows": int(len(d)), "hs_model_median": float(d.hs_model.median()), "hs_model_mean": float(d.hs_model.mean()),
            "hs_quoted_median": float(d.hs_quoted.median()), "hs_quoted_mean": float(d.hs_quoted.mean()),
            "ratio_median": float(r.median()), "share_model_gt_quoted": float((d.hs_model > d.hs_quoted).mean())}


def main(bidask_csv, out_json=None):
    sm = pd.read_parquet(BUNDLE)[["security_id", "ric"]]
    panel = pd.read_parquet("data/processed/panel_large.parquet",
                            columns=["date", "symbol", "cs_spread", "vol_60", "adv_dollar", "mcap"])
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.merge(sm, left_on="symbol", right_on="security_id", how="left")
    q = quoted_half_spread(pd.read_csv(bidask_csv))
    # quote on the decision date or the last quote day before it (the window ends on the decision date)
    panel["date"] = panel["date"].astype("datetime64[ns]")
    panel = panel.sort_values("date")
    q = q.dropna(subset=["hs_quoted"]).astype({"date": "datetime64[ns]"}).sort_values("date")
    panel = pd.merge_asof(panel, q, on="date", by="ric", direction="backward", tolerance=pd.Timedelta(days=5))
    panel["hs_model"] = (panel.cs_spread / 2).clip(lower=C.HALF_SPREAD, upper=0.02).fillna(C.HALF_SPREAD)
    have = panel.dropna(subset=["hs_quoted"])
    coverage = len(have) / len(panel)

    order = 600.0
    cost_now = name_costs(panel, order)
    hs_v2 = panel.hs_quoted.clip(lower=C.HALF_SPREAD).fillna(panel.hs_model)
    cost_v2 = cost_now - panel.hs_model + hs_v2

    have = have.assign(year=have.date.dt.year,
                       size=pd.qcut(have.groupby("date").mcap.rank(pct=True), 3, labels=["small", "mid", "large"]))
    rank_corr = have.groupby("date").apply(lambda g: g.hs_model.corr(g.hs_quoted, method="spearman") if len(g) > 10 else np.nan)
    out = {"prereg": "prereg/spread_check_v1.json", "panel_rows": int(len(panel)), "rows_with_quote": int(len(have)),
           "coverage": coverage, "adopt_v2_by_rule": bool(coverage >= 0.80),
           "overall": summarise(have),
           "raw_cs_half_median": float((have.cs_spread / 2).median()),
           "by_year": {str(k): summarise(g) for k, g in have.groupby("year")},
           "by_size": {str(k): summarise(g) for k, g in have.groupby("size", observed=True)},
           "spearman_across_names_median": float(rank_corr.median()),
           "quote_source_rows": {str(k): int(v) for k, v in have.src.value_counts().items()},
           "one_way_cost_at_600": {"current_mean": float(cost_now.mean()), "current_median": float(cost_now.median()),
                                   "v2_mean": float(cost_v2.mean()), "v2_median": float(cost_v2.median()),
                                   "fx": C.FX_FEE, "half_spread_current_mean": float(panel.hs_model.mean()),
                                   "half_spread_v2_mean": float(hs_v2.mean())}}
    text = json.dumps(out, indent=2)
    print(text)
    if out_json:
        with open(out_json, "w") as f:
            f.write(text)
    return out


if __name__ == "__main__":
    main(*sys.argv[1:])
