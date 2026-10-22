"""Turn the latest validated model into schema-v2.0 signals for the external trading bot.

The engine never places orders. It writes:
  signals/latest.json                 every security evaluated today (LONG or NO_SIGNAL, with reasons)
  signals/log/<research_run_id>.jsonl immutable per-run log (never overwritten)
  reports/forecast_latest.parquet     all-horizon forecasts for the report

Forecast numbers are not raw model outputs: a security's prediction is ranked into today's
cross-sectional decile and the forecast fields are that decile's realised out-of-sample excess-return
distribution from research.py. If that mapping was not monotonic out-of-sample, fields stay null.
"""
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import jsonschema
import numpy as np
import pandas as pd

import config as C
from features import FEATURES, FUND_FEATURES, build_panel, load_insider
from research import FEATS, preprocess, sector_neutral

SCHEMA = json.loads((C.ROOT / "schema" / "signal_v2.schema.json").read_text())
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA, format_checker=jsonschema.FormatChecker())
NY = ZoneInfo("America/New_York")


def iso(ts):
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def session_close_utc(d):
    return datetime(d.year, d.month, d.day, 16, 0, tzinfo=NY)


def dataset_version():
    parts = []
    for p in [C.RAW / "prices" / f"{C.BENCHMARK}.parquet", C.RAW / "facts" / "companyfacts.parquet",
              C.RAW / "insider_recent" / "form4_recent.parquet", *sorted((C.RAW / "insider").glob("*.parquet"))[-2:]]:
        if p.exists():
            st = p.stat()
            parts.append(f"{p.name}:{st.st_size}:{int(st.st_mtime)}")
    return "ds-" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def decile_lookup(cal_table, buckets):
    tab = pd.DataFrame(cal_table).set_index("b")
    return tab.reindex(buckets)


def run(dry=False):
    model = json.loads((C.PROC / "model.json").read_text())
    gen = datetime.now(timezone.utc)
    panel, meta = build_panel(include_labels=False, last_n_dates=1)
    asof_day = pd.Timestamp(panel["date"].max())
    panel = panel[panel["date"] == asof_day].reset_index(drop=True)
    asof = session_close_utc(asof_day)
    X = preprocess(panel)
    Xn = sector_neutral(X, panel)
    Xm = pd.concat([Xn[FEATS], X[["fund_missing"]]], axis=1)[model["features"]]

    sic = pd.read_parquet(C.PROC / "sic.parquet").set_index("cik")
    sessions_stale = int(np.busday_count(asof_day.date(), gen.astimezone(NY).date()))
    market_fresh = sessions_stale <= C.PRICE_MAX_STALENESS_SESSIONS
    # insider features read "no activity" when coverage is missing, so coverage itself must be current
    ins_cov = pd.Timestamp(load_insider(list(panel["cik"].astype(int)))["filing_date"].max())
    insider_fresh = (asof_day - ins_cov).days <= C.INSIDER_MAX_COVERAGE_GAP_DAYS
    dsv = dataset_version()
    h = C.PRIMARY_HORIZON
    rows, out, fc_rows = [], [], []

    preds = {}
    for hh in C.HORIZONS:
        coef = pd.Series(model["coefficients"][str(hh)])
        preds[hh] = Xm.to_numpy() @ coef[Xm.columns].to_numpy() + model["intercepts"][str(hh)]
        bucket = pd.Series(preds[hh]).rank(method="first").pipe(
            lambda s: pd.qcut(s, C.CAL_BUCKETS, labels=False)).to_numpy()
        cal = model["calibration"][str(hh)]
        tab = decile_lookup(cal["table"], bucket)
        ok_cal = cal["status"] == "HISTORICAL_DECILE_CALIBRATED"
        for i, sym in enumerate(panel["symbol"]):
            fc_rows.append({"symbol": sym, "horizon": hh, "raw_model_score": float(preds[hh][i]), "decile": int(bucket[i]),
                            "calibrated": ok_cal, "exp_excess": tab["mean"].iat[i] if ok_cal else None,
                            "std": tab["std"].iat[i] if ok_cal else None, "q025": tab["q025"].iat[i] if ok_cal else None,
                            "q975": tab["q975"].iat[i] if ok_cal else None, "q05": tab["q05"].iat[i] if ok_cal else None,
                            "p_pos": tab["p_pos"].iat[i] if ok_cal else None, "es05": tab.get("es05", pd.Series([None] * len(tab))).iat[i] if ok_cal else None,
                            "gate": model["gate"][str(hh)]["status"]})
    fc = pd.DataFrame(fc_rows)
    prim = fc[fc["horizon"] == h].set_index("symbol")
    coef_h = pd.Series(model["coefficients"][str(h)])
    contrib = Xm[FEATS] * coef_h[FEATS].to_numpy()
    gate_h = model["gate"][str(h)]
    val_h = model["validation"][str(h)]
    top_cut = 1 - C.SIGNAL_TOP_QUANTILE
    pct = pd.Series(preds[h]).rank(pct=True).to_numpy()
    cost = C.ROUND_TRIP_COST

    def clean(x):
        return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else float(x)

    for i, r in panel.iterrows():
        sym, cik = r["symbol"], int(r["cik"])
        p = prim.loc[sym]
        reasons = []
        data_ok = market_fresh and insider_fresh and not any(pd.isna(r[f]) for f in FUND_FEATURES[:3]) and \
            (pd.notna(r["fund_age_days"]) and r["fund_age_days"] <= C.FUNDAMENTAL_MAX_AGE_DAYS)
        if not market_fresh:
            reasons.append(f"market data {sessions_stale} sessions old (> {C.PRICE_MAX_STALENESS_SESSIONS})")
        if not insider_fresh:
            reasons.append(f"insider filings only ingested through {ins_cov.date()}")
        if not data_ok and market_fresh and insider_fresh:
            reasons.append("fundamentals missing or older than limit")
        liq_ok = bool(r["adv_shares"] >= C.MIN_ADV_SHARES and r["mcap"] >= C.MIN_MCAP_USD)
        elig = gate_h["status"] if data_ok else "INSUFFICIENT_DATA"
        if elig not in ("PAPER_TEST_ELIGIBLE", "LIVE_REVIEW_ELIGIBLE"):
            reasons.append(f"model gate at {h}d is {gate_h['status']}: " + "; ".join(gate_h["failed_checks"]) if gate_h["failed_checks"] else f"model gate {elig}")
        exp = clean(p["exp_excess"])
        net = exp - cost if exp is not None else None
        if pct[i] < top_cut:
            reasons.append(f"forecast rank {pct[i]:.0%} below top-{C.SIGNAL_TOP_QUANTILE:.0%} cut")
        if net is not None and net <= 0:
            reasons.append("expected excess return does not cover estimated round-trip cost")
        long_ok = not reasons and liq_ok and net is not None and net > 0
        c = contrib.iloc[i].sort_values()
        drivers = [f"{k} ({FEATURES[k][0]}): {v:+.4f}" for k, v in c[::-1].items() if v > 0][:4]
        contra = [f"{k} ({FEATURES[k][0]}): {v:+.4f}" for k, v in c.items() if v < 0][:4]
        vol = clean(r["vol_60"])
        sid = hashlib.sha256(f"{C.MODEL_ID}|{C.MODEL_VERSION}|{model['run_id']}|{cik}|{asof_day.date()}|{h}".encode()).hexdigest()[:20]
        sig = {
            "schema_version": "2.0", "model_id": C.MODEL_ID, "model_version": C.MODEL_VERSION, "signal_id": sid,
            "generated_at_utc": iso(gen), "as_of_utc": iso(asof),
            "asset": {"ticker": sym.replace("-", "."), "security_id": f"CIK{cik:010d}",
                      "exchange": (str(sic["exchanges"].get(cik)) or None) if cik in sic.index else None, "asset_class": "Equity"},
            "forecast": {"horizon_sessions": h, "expected_excess_return": exp, "expected_net_excess_return": net,
                         "predictive_std": clean(p["std"]),
                         "prediction_interval_95": [clean(p["q025"]), clean(p["q975"])] if clean(p["q025"]) is not None else None,
                         "probability_positive_excess_return": clean(p["p_pos"]), "expected_cost_return": cost},
            "validation": {"status": gate_h["status"], "out_of_sample_ic": clean(val_h["oos_ic"]),
                           "ic_uncertainty": clean(val_h["oos_ic_se"]), "net_sharpe": clean(val_h["net_sharpe"]),
                           "deflated_sharpe_ratio": clean(model["deflated_sharpe_ratio"]), "tested_hypotheses": int(model["tested_hypotheses"])},
            "risk": {"forecast_volatility": vol * math.sqrt(h / 252) if vol is not None else None,
                     "estimated_beta": clean(r["beta_252"]), "expected_shortfall": clean(p["es05"]),
                     "liquidity_approved": liq_ok, "data_quality_approved": bool(data_ok)},
            "signal": {"direction": "LONG" if long_ok else "NO_SIGNAL",
                       "deployment_eligibility": elig,
                       "expires_at_utc": iso(session_close_utc(asof_day + pd.offsets.BDay(C.DECISION_STEP))),
                       "reason": None if long_ok else "; ".join(reasons) or "not selected",
                       "key_drivers": drivers, "contradictory_evidence": contra},
            "provenance": {"feature_set_version": C.FEATURE_SET_VERSION, "dataset_version": dsv,
                           "source_ids": ["yahoo_chart_v8_daily", "sec_xbrl_companyfacts", "sec_form345_datasets", "sec_edgar_form4_xml"],
                           "latest_required_data_utc": iso(asof), "research_run_id": model["run_id"]},
        }
        errs = sorted(VALIDATOR.iter_errors(sig), key=str)
        if errs:
            raise ValueError(f"{sym}: schema violation: {errs[0].message} at {list(errs[0].path)}")
        out.append(sig)
        rows.append({"symbol": sym, "pct_rank": float(pct[i]), "direction": sig["signal"]["direction"], "exp": exp, "net": net})

    # insider-event model (pre-registered, prereg/insider_event_v1.json): today's qualifying Form 4 filings
    n_xs = len(out)
    if (C.PROC / "insider_model.json").exists():
        import insider_model
        for sig in insider_model.live_signals(asof_day, gen, dsv, market_fresh, insider_fresh):
            errs = sorted(VALIDATOR.iter_errors(sig), key=str)
            if errs:
                raise ValueError(f"insider {sig['asset']['ticker']}: schema violation: {errs[0].message} at {list(errs[0].path)}")
            out.append(sig)
    longs = [s for s in out if s["signal"]["direction"] == "LONG"]
    doc = {"schema_version": "2.0", "generated_at_utc": iso(gen), "as_of_utc": iso(asof), "research_run_id": model["run_id"],
           "model_gate_primary": gate_h, "market_data_sessions_stale": sessions_stale,
           "insider_coverage_through": str(ins_cov.date()), "n_evaluated": len(out),
           "n_long": len(longs), "n_insider_event_signals": len(out) - n_xs, "signals": sorted(out, key=lambda s: -(s["forecast"]["expected_net_excess_return"] or -9))}
    if dry:
        print(json.dumps(doc["signals"][0], indent=1))
    else:
        C.SIGNALS.mkdir(exist_ok=True)
        (C.SIGNALS / "log").mkdir(exist_ok=True)
        log = C.SIGNALS / "log" / f"{asof_day.date()}_{gen.strftime('%H%M%S')}_{model['run_id']}.jsonl"
        if log.exists():
            raise FileExistsError(log)
        log.write_text("".join(json.dumps(s) + "\n" for s in out))
        tmp = C.SIGNALS / "latest.json.tmp"
        tmp.write_text(json.dumps(doc, indent=1))
        tmp.replace(C.SIGNALS / "latest.json")  # atomic swap so the bot never reads a half-written file
        fc.to_parquet(C.REPORTS / "forecast_latest.parquet", index=False)
    print(f"as_of={asof_day.date()} evaluated={len(out)} insider_events={len(out) - n_xs} LONG={len(longs)} gate[{h}d]={gate_h['status']} stale_sessions={sessions_stale}")
    return doc


if __name__ == "__main__":
    run(dry="--dry" in sys.argv)
