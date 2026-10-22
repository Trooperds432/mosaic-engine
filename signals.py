"""Turn the latest validated models into schema-v2.0 signals for the external trading bot.

The engine never places orders. It writes:
  signals/latest.json                 every security evaluated today (LONG or NO_SIGNAL, with reasons)
  signals/log/<as_of>_<time>.jsonl    immutable per-run log (never overwritten)
  reports/forecast_latest.parquet     all-horizon forecasts for the report

For each universe profile with a trained model (model_<profile>.json) the ensemble members' predictions are
ranked within today's cross-section; the forecast fields are the realised out-of-sample excess-return
distribution of that ensemble decile from research.py. If the decile mapping was not monotonic out of sample
the fields stay null. Costs are per name: FX fee + Corwin-Schultz half-spread + square-root impact at the bot's
order size, each way. A failed data audit (look-ahead / duplicate / timestamp tests) blocks every LONG.
"""
import hashlib
from functools import lru_cache
import json
import math
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import joblib
from publication import VALIDATOR, publish
import numpy as np
import pandas as pd

import config as C
import gates
from provenance import dataset_version, sessions_between
import regime
from features import FEATURES, FUND_FEATURES, build_panel_chunked, load_insider

NY = ZoneInfo("America/New_York")
LIVE = ("PAPER_TEST_ELIGIBLE", "LIVE_REVIEW_ELIGIBLE")


def iso(ts):
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def session_close_utc(d):
    return datetime(d.year, d.month, d.day, 16, 0, tzinfo=NY)


def clean(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def audit_ok(dsv):
    f = C.REPORTS / "data_audit.json"
    try:
        doc = json.loads(f.read_text())
    except (OSError, ValueError):
        doc = None
    return gates.check_audit(doc, dsv)


@lru_cache(maxsize=None)
def forward_gate(pname, h):
    """Gate and out-of-sample statistics from the forward record (paper_trading.py), never the reused holdout.
    Only a fresh gate produced for this profile's current model recipe counts (gates.check_forward_gate)."""
    try:
        doc = json.loads((C.PROC / "forward_gate.json").read_text())
    except (OSError, ValueError):
        doc = None
    run_id = json.loads((C.PROC / f"model_{pname}.json").read_text())["run_id"]
    status, failed, st = gates.check_forward_gate(doc, pname, h, run_id)
    return {"status": status, "failed_checks": failed}, {"oos_ic": st["forward_ic"], "oos_ic_se": st["forward_ic_se"], "net_sharpe": None}


def profile_signals(pname, gen, dsv, reg, audit):
    import __main__
    import research
    from research import MISSING_GROUPS, name_costs, preprocess, sector_neutral
    for cls in ("GAM", "Hierarchical", "CalibratedLogit"):  # models pickled while research.py ran as __main__
        setattr(__main__, cls, getattr(research, cls))
    model = json.loads((C.PROC / f"model_{pname}.json").read_text())
    panel, _ = build_panel_chunked(include_labels=False, last_n_dates=1, profile=pname, chunk=1200)  # bounded memory
    asof_day = pd.Timestamp(panel["date"].max())
    panel = panel[panel["date"] == asof_day].reset_index(drop=True)
    panel.to_parquet(C.PROC / f"asof_panel_{pname}.parquet", index=False)  # as-of cross-section for ls_signals.py
    asof = session_close_utc(asof_day)
    feats = model["feats"]
    X = preprocess(panel, feats)
    Xn = sector_neutral(X, panel, feats)
    miss = [k for k in MISSING_GROUPS if k in X]
    Xm = pd.concat([Xn[feats], X[miss]], axis=1).reindex(columns=model["features"], fill_value=0.0)
    sectors = panel["sector"].fillna("Unknown").to_numpy()
    cost_rt = (name_costs(panel, model.get("order_size_usd") or 600.0) * 2).to_numpy()

    sic = pd.read_parquet(C.PROC / "sic.parquet").set_index("cik")
    sessions_stale = sessions_between(asof_day.date(), gen.astimezone(NY).date())
    market_fresh = sessions_stale <= C.PRICE_MAX_STALENESS_SESSIONS
    ins_cov = pd.Timestamp(load_insider(list(panel["cik"].astype(int)))["filing_date"].max())
    insider_fresh = (asof_day - ins_cov).days <= C.INSIDER_MAX_COVERAGE_GAP_DAYS

    preds, ens_rank, disagree, fc_rows = {}, {}, {}, []
    for hh in C.HORIZONS:
        hz = model["horizons"][str(hh)]
        ranks = {}
        for n in hz["names"]:
            m = joblib.load(C.ROOT / "models" / f"{pname}_h{hh}_{n}.joblib")
            if n == "hier":
                p = m.predict(Xm.to_numpy(dtype="float32"), sectors)
            elif n in ("gam", "lgbm"):
                p = m.predict(Xm)
            else:
                p = m.predict(Xm.to_numpy(dtype="float32"))
            ranks[n] = pd.Series(p).rank(pct=True).to_numpy() - 0.5
        ens = np.mean(np.stack([ranks[n] for n in hz["members"]]), axis=0)
        ens_rank[hh] = pd.Series(ens).rank(pct=True).to_numpy()
        disagree[hh] = np.std(np.stack(list(ranks.values())), axis=0) if len(ranks) > 1 else np.full(len(ens), np.nan)
        bucket = pd.qcut(pd.Series(ens).rank(method="first"), C.CAL_BUCKETS, labels=False).to_numpy()
        cal = hz["calibration"] or {}
        ok_cal = cal.get("status") == "HISTORICAL_DECILE_CALIBRATED"
        tab = pd.DataFrame(cal.get("table") or []).set_index("b").reindex(bucket) if cal.get("table") else None
        for i, sym in enumerate(panel["symbol"]):
            g = (lambda k: clean(tab[k].iat[i]) if ok_cal and tab is not None and k in tab else None)
            fc_rows.append({"profile": pname, "model_run_id": model["run_id"], "symbol": sym, "horizon": hh, "ensemble_rank": float(ens_rank[hh][i]), "decile": int(bucket[i]),
                            "model_disagreement": clean(disagree[hh][i]), "calibrated": ok_cal, "exp_excess": g("mean"), "std": g("std"),
                            "q025": g("q025"), "q975": g("q975"), "q05": g("q05"), "p_pos": g("p_pos"), "es05": g("es05"),
                            "cost_round_trip": float(cost_rt[i]), "gate": forward_gate(pname, hh)[0]["status"]})
    fc = pd.DataFrame(fc_rows)
    h = C.PRIMARY_HORIZON
    prim = fc[fc["horizon"] == h].set_index("symbol")
    hz = model["horizons"][str(h)]
    gate_h, val_h = forward_gate(pname, h)
    # drivers: sector-neutral feature ranks x ridge coefficients at the primary horizon (transparent attribution)
    ridge = joblib.load(C.ROOT / "models" / f"{pname}_h{h}_ridge.joblib")
    contrib = Xm.mul(ridge.coef_, axis=1)[feats]
    rm_vol = {}
    rmf = C.PROC / f"risk_model_{pname}.json"
    if rmf.exists():
        rm_vol = json.loads(rmf.read_text()).get("ewma_vol_annual", {})
    # 1-day volatility from the model that won the VaR/ES backtest (var_backtest.py: GARCH-t), EWMA as fallback
    vf = pd.read_parquet(C.PROC / "vol_forecasts.parquet").set_index("symbol") if (C.PROC / "vol_forecasts.parquet").exists() else pd.DataFrame()

    out = []
    for i, r in panel.iterrows():
        sym, cik = r["symbol"], int(r["cik"])
        p = prim.loc[sym]
        reasons = []
        if "bio_ph3_trials" in feats and model.get("feature_set_version") != C.FEATURE_SET_VERSION:
            reasons.append("clinical feature semantics changed: new recipe validation required")
        fund_ok = not any(pd.isna(r[f]) for f in FUND_FEATURES[:3]) and pd.notna(r["fund_age_days"]) and r["fund_age_days"] <= C.FUNDAMENTAL_MAX_AGE_DAYS
        data_ok = market_fresh and insider_fresh and fund_ok and r["quality_score"] >= 0.7 and audit[0]
        if not market_fresh:
            reasons.append(f"market data {sessions_stale} sessions old (> {C.PRICE_MAX_STALENESS_SESSIONS})")
        if not insider_fresh:
            reasons.append(f"insider filings only ingested through {ins_cov.date()}")
        if not fund_ok:
            reasons.append("fundamentals missing or older than limit")
        if r["quality_score"] < 0.7:
            reasons.append(f"feature quality score {r['quality_score']:.2f} < 0.70")
        if not audit[0]:
            reasons.append(audit[1])
        order_pct_adv = (model.get("order_size_usd") or 600.0) / max(float(r["adv_dollar"]), 1.0)
        liq_ok = bool(r["adv_shares"] >= C.PROFILES[pname]["min_adv_shares"] and r["mcap"] >= C.PROFILES[pname]["min_mcap"] and order_pct_adv <= 0.01)
        elig = gate_h["status"] if data_ok else "INSUFFICIENT_DATA"
        if elig not in LIVE:
            reasons.append(f"model gate at {h}d is {gate_h['status']}" + (": " + "; ".join(gate_h["failed_checks"]) if gate_h["failed_checks"] else ""))
        exp = clean(p["exp_excess"])
        cost = float(p["cost_round_trip"])
        net = exp - cost if exp is not None else None
        if p["ensemble_rank"] < 1 - C.SIGNAL_TOP_QUANTILE:
            reasons.append(f"ensemble rank {p['ensemble_rank']:.0%} below top-{C.SIGNAL_TOP_QUANTILE:.0%} cut")
        if net is not None and net <= 0:
            reasons.append("expected excess return does not cover estimated round-trip cost")
        long_ok = not reasons and liq_ok and net is not None and net > 0
        c = contrib.iloc[i].sort_values()
        drivers = [f"{k} ({FEATURES[k][0]}): {v:+.4f}" for k, v in c[::-1].items() if v > 0][:4]
        contra = [f"{k} ({FEATURES[k][0]}): {v:+.4f}" for k, v in c.items() if v < 0][:4]
        if clean(p["model_disagreement"]) is not None and p["model_disagreement"] > 0.25:
            contra.append(f"model disagreement: rank std {p['model_disagreement']:.2f} across {len(hz['names'])} models")
        vol_h = None
        if sym in vf.index and pd.notna(vf.at[sym, "sigma_1d"]):
            vol_h = float(vf.at[sym, "sigma_1d"]) * math.sqrt(h)  # daily sigma scaled to the horizon
        else:
            ann = rm_vol.get(sym) or clean(r["vol_60"])
            vol_h = ann * math.sqrt(h / 252) if ann else None
        sid = hashlib.sha256(f"{model['model_id']}|{C.MODEL_VERSION}|{model['run_id']}|{cik}|{asof_day.date()}|{h}".encode()).hexdigest()[:20]
        ex = sic["exchanges"].get(cik) if cik in sic.index else None
        sig = {
            "schema_version": "2.0", "model_id": model["model_id"], "model_version": C.MODEL_VERSION, "signal_id": sid,
            "generated_at_utc": iso(gen), "as_of_utc": iso(asof),
            "asset": {"ticker": sym.replace("-", "."), "security_id": f"CIK{cik:010d}", "exchange": str(ex) if ex else None, "asset_class": "Equity"},
            "forecast": {"horizon_sessions": h, "expected_excess_return": exp, "expected_net_excess_return": net, "predictive_std": clean(p["std"]),
                         "prediction_interval_95": [clean(p["q025"]), clean(p["q975"])] if clean(p["q025"]) is not None else None,
                         "probability_positive_excess_return": clean(p["p_pos"]), "expected_cost_return": cost},
            "validation": {"status": gate_h["status"], "out_of_sample_ic": clean(val_h["oos_ic"]), "ic_uncertainty": clean(val_h["oos_ic_se"]),
                           "net_sharpe": clean(val_h["net_sharpe"]), "deflated_sharpe_ratio": clean(model["deflated_sharpe_ratio"]),
                           "tested_hypotheses": int(model["tested_hypotheses"])},
            # Blume-adjusted beta: beat the raw 252-day and Kalman betas out of sample (reports/kalman_beta_large.json)
            "risk": {"forecast_volatility": vol_h, "estimated_beta": (0.67 * clean(r["beta_252"]) + 0.33) if clean(r["beta_252"]) is not None else None,
                     "expected_shortfall": clean(p["es05"]), "liquidity_approved": liq_ok, "data_quality_approved": bool(data_ok)},
            "signal": {"direction": "LONG" if long_ok else "NO_SIGNAL", "deployment_eligibility": elig,
                       "expires_at_utc": iso(session_close_utc(asof_day + pd.offsets.BDay(C.DECISION_STEP))),
                       "reason": None if long_ok else "; ".join(reasons) or "not selected", "key_drivers": drivers, "contradictory_evidence": contra},
            "provenance": {"feature_set_version": C.FEATURE_SET_VERSION, "dataset_version": dsv,
                           "source_ids": ["yahoo_chart_v8_daily", "sec_xbrl_companyfacts", "sec_form345_datasets", "sec_edgar_form4_xml", "sec_form13f_datasets",
                                          "finra_short_interest", "fred_alfred", "eia_wpsr", "cftc_cot", "clinicaltrials_gov", "openfda"],
                           "latest_required_data_utc": iso(asof), "research_run_id": model["run_id"]},
            "regime": reg,
        }
        errs = sorted(VALIDATOR.iter_errors(sig), key=str)
        if errs:
            raise ValueError(f"{sym}: schema violation: {errs[0].message} at {list(errs[0].path)}")
        out.append(sig)
    meta = {"profile": pname, "as_of": str(asof_day.date()), "model_run_id": model["run_id"], "model_gate_primary": gate_h, "market_data_sessions_stale": sessions_stale,
            "insider_coverage_through": str(ins_cov.date()), "evaluated": len(out)}
    return out, fc, meta, asof_day, market_fresh, insider_fresh


def run(dry=False, profiles=None):
    gen = datetime.now(timezone.utc)
    dsv = dataset_version()
    reg = regime.field()
    audit = audit_ok(dsv)
    profiles = profiles or [p for p in C.PROFILES if (C.PROC / f"model_{p}.json").exists()]
    out, fcs, metas = [], [], []
    asof_day, market_fresh, insider_fresh = None, False, False
    for p in profiles:
        o, fc, meta, asof_day, market_fresh, insider_fresh = profile_signals(p, gen, dsv, reg, audit)
        out += o
        fcs.append(fc)
        metas.append(meta)
        print(f"profile={p} as_of={meta['as_of']} evaluated={meta['evaluated']} LONG={sum(s['signal']['direction'] == 'LONG' for s in o)} gate={meta['model_gate_primary']['status']}", flush=True)
    n_xs = len(out)
    if asof_day is None:  # no trained cross-sectional model yet: still publish insider-event signals for the last session
        spy = pd.read_parquet(C.RAW / "prices" / f"{C.BENCHMARK}.parquet", columns=["date"])
        asof_day = pd.Timestamp(spy["date"].max())
        market_fresh = sessions_between(asof_day.date(), gen.astimezone(NY).date()) <= C.PRICE_MAX_STALENESS_SESSIONS
        insider_fresh = True
    if (C.PROC / "insider_model.json").exists():  # pre-registered insider-event model
        import insider_model
        for sig in insider_model.live_signals(asof_day, gen, dsv, market_fresh, insider_fresh and audit[0]):
            sig["regime"] = reg
            errs = sorted(VALIDATOR.iter_errors(sig), key=str)
            if errs:
                raise ValueError(f"insider {sig['asset']['ticker']}: schema violation: {errs[0].message} at {list(errs[0].path)}")
            out.append(sig)
    longs = [s for s in out if s["signal"]["direction"] == "LONG"]
    doc = {"schema_version": "2.0", "generated_at_utc": iso(gen), "as_of_utc": iso(session_close_utc(asof_day)) if asof_day is not None else None,
           "regime": reg, "data_audit_passed": audit[0], "profiles": metas, "n_evaluated": len(out), "n_long": len(longs),
           "n_insider_event_signals": len(out) - n_xs, "signals": sorted(out, key=lambda s: -(s["forecast"]["expected_net_excess_return"] or -9))}
    if dry:
        print(json.dumps(doc["signals"][0], indent=1))
        return doc
    publish(doc, C.SIGNALS, f"{asof_day.date()}_{gen.strftime('%H%M%S%f')}")
    vfp = C.PROC / "vol_forecasts.parquet"
    if vfp.exists():  # per-name 1-day risk for the bot's sizing (ticker -> sigma, 99% VaR, 97.5% ES)
        vfd = pd.read_parquet(vfp)
        side = {r.symbol.replace("-", "."): {"sigma_1d": float(r.sigma_1d), "var99_1d": float(r.var99_1d), "es975_1d": float(r.es975_1d),
                                             "model": r.model, "as_of": r.as_of} for r in vfd.itertuples()}
        tmp = C.SIGNALS / "risk_latest.json.tmp"
        tmp.write_text(json.dumps({"generated_at_utc": iso(gen), "source": "var_backtest.forecast_names", "names": side}))
        tmp.replace(C.SIGNALS / "risk_latest.json")
    (pd.concat(fcs, ignore_index=True) if fcs else pd.DataFrame()).to_parquet(C.REPORTS / "forecast_latest.parquet", index=False)
    print(f"total evaluated={len(out)} insider_events={len(out) - n_xs} LONG={len(longs)} audit={'pass' if audit[0] else audit[1]}")
    return doc


if __name__ == "__main__":
    _lock = __import__("config").hold_lock()
    run(dry="--dry" in sys.argv, profiles=[a for a in sys.argv[1:] if not a.startswith("--")] or None)
