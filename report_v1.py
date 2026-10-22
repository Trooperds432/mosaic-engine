"""Institutional report (spec section 13, A-J) assembled only from computed artifacts:
reports/research_results.json, reports/insider_event_study.json, signals/latest.json,
reports/forecast_latest.parquet. Anything not computed is printed as NOT_COMPUTED."""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config as C
from features import FEATURES, load_insider

NC = "NOT_COMPUTED"


def f(x, nd=4, pct=False):
    if isinstance(x, str):  # numpy scalars were serialised with default=str
        try:
            x = float(x)
        except ValueError:
            return x
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return NC
    return f"{x * 100:.2f}%" if pct else f"{x:.{nd}f}"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def run():
    R = json.loads((C.REPORTS / "research_results.json").read_text())
    ES = json.loads((C.REPORTS / "insider_event_study.json").read_text()) if (C.REPORTS / "insider_event_study.json").exists() else None
    S = json.loads((C.SIGNALS / "latest.json").read_text())
    fc = pd.read_parquet(C.REPORTS / "forecast_latest.parquet")
    h = C.PRIMARY_HORIZON
    M = {int(k): v for k, v in R["models"].items()}
    mp = M[h]
    longs = [s for s in S["signals"] if s["signal"]["direction"] == "LONG"]
    L = []
    L.append(f"# Mosaic Quantamental Research Report — {S['as_of_utc'][:10]}\n")
    L.append(f"Run `{R['run_id']}` · model `{C.MODEL_ID}` v{C.MODEL_VERSION} · features `{C.FEATURE_SET_VERSION}` · "
             f"generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")

    # A
    L.append("## A. Executive summary\n")
    L.append(f"- **Universe:** historical S&P 500 (constituent-aware) ∪ current NASDAQ-100, screened point-in-time to "
             f"market cap ≥ ${C.MIN_MCAP_USD/1e9:.0f}bn and {C.ADV_WINDOW}-day ADV ≥ {C.MIN_ADV_SHARES/1e6:.0f}m shares; "
             f"{S['n_evaluated']} securities eligible on the as-of date.")
    L.append(f"- **Sample:** {R['sample']['first_date']} → {R['sample']['last_date']}, {R['sample']['rows']:,} security-dates; "
             f"untouched holdout from {R['sample']['holdout_start']}.")
    cr = R["current_regime"]
    L.append(f"- **Regime (observable, filtered):** {cr['state']} — SPY above 200d: {cr['spy_above_200d']}, VIX {cr['vix']:.1f} "
             f"vs 5y median {cr['vix_5y_median']:.1f}.")
    L.append(f"- **Primary horizon {h} sessions:** walk-forward IC {f(mp['walk_forward']['mean'])} (t={f(mp['walk_forward']['t'], 2)}), "
             f"holdout IC {f(mp['holdout']['mean'])} (t={f(mp['holdout']['t'], 2)}).")
    L.append(f"- **Validation gate ({h}d):** **{mp['gate']['status']}**" +
             (f" — failed: {'; '.join(mp['gate']['failed_checks'])}" if mp["gate"]["failed_checks"] else ""))
    L.append(f"- **Signals to bot:** {len(longs)} LONG, {S['n_evaluated'] - len(longs)} NO_SIGNAL. "
             "A research thesis is not execution approval; the bot's own risk checks always apply.\n")

    # B
    L.append("## B. Evidence and source audit\n")
    L.append(table(["Dataset", "Coverage", "Timestamp rule", "Source", "Quality", "PIT-eligible"], [
        ["Daily OHLCV (split/div-adjusted)", f"{R['sample']['symbols']} symbols", "close of t; entry at t+1 open", "Yahoo chart API (unofficial)", "vendor; no delisting returns (0% assumed)", "yes"],
        ["Fundamentals (as-reported XBRL)", "10-K/10-Q since ~2009", "SEC `filed` date ≤ t−1", "SEC companyfacts", "original + restated kept; earliest filing used", "yes"],
        ["Insider Forms 3/4/5", "2006 → latest SEC quarterly set", "filing date ≤ t−1", "SEC Insider Transactions Data Sets", "10b5-1 flag only from 2023", "yes"],
        ["Recent Form 4", "after last quarterly set", "filing date ≤ t−1", "EDGAR Form 4 XML", "parsed; 4/A deduped", "yes"],
        ["Index membership", "S&P 500 since 2005", "membership not used as a feature", "fja05680/sp500, Wikipedia", "321 removed names lack a current SEC ticker → excluded (survivorship risk)", "n/a"],
        ["13F holdings / options / short interest / estimates", "—", "—", "not ingested", NC, "no"],
    ]))
    L.append("")

    # C
    L.append("## C. Factor dashboard (discovery sample, sector-neutral, primary horizon)\n")
    uni = [u for u in R["univariate"] if u["horizon"] == h and u["variant"] == "sector_neutral"]
    raw = {u["feature"]: u for u in R["univariate"] if u["horizon"] == h and u["variant"] == "raw"}
    coef = mp["holdout_coefficients"]
    rows = []
    for u in sorted(uni, key=lambda u: -abs(u["t"] or 0)):
        rows.append([FEATURES[u["feature"]][0], u["feature"], "+" if u["hyp_sign"] > 0 else "−",
                     f(u["mean"]), f(u["t"], 2), f(u["p_bh"], 3), "✓" if u["fdr_reject"] and u["sign_matches_hypothesis"] else ("✗ sign" if u["fdr_reject"] else "—"),
                     f(raw[u["feature"]].get("top_minus_bottom"), pct=True), f(coef.get(u["feature"]))])
    L.append(table(["Family", "Feature", "Hyp.", "Mean IC", "HAC t", "BH p", "FDR pass", "Q5−Q1 (raw)", "Ridge coef"], rows))
    L.append("\nIC = per-date Spearman rank correlation with forward SPY-excess return; HAC (Newey-West) lags = horizon/step. "
             f"BH-FDR at q={C.FDR_Q} across all {len(R['univariate'])} univariate tests.\n")

    # D
    L.append("## D. Insider activity\n")
    if ES:
        rows = []
        for kind, res in ES["results"].items():
            for samp in ["discovery", "holdout"]:
                r = res[samp]
                rows.append([kind, samp, r["1d"]["n"]] + [f"{f(r[w]['mean'], pct=True)} (t={f(r[w]['t'], 2)})" for w in ["1d", "5d", "20d", "60d"]])
        L.append(table(["Event", "Sample", "N", "CAR 1d", "CAR 5d", "CAR 20d", "CAR 60d"], rows))
        L.append("\nBeta-adjusted abnormal returns from first open after the filing date; one event per issuer per 60 sessions; "
                 "SEs clustered by month. Association, not causation.\n")
    IM = C.REPORTS / "insider_model_results.json"
    if IM.exists():
        im = json.loads(IM.read_text())
        P = im["primary"]
        L.append(f"**Pre-registered insider event model** (`prereg/insider_event_v1.json`, sha256 `{im['prereg_sha256'][:12]}`): "
                 f"CEO/CFO or cluster open-market purchase, entry next open, hold 5 sessions, beta-adjusted, cost {C.ROUND_TRIP_COST:.2%}. "
                 f"Gate: **{im['gate']['status']}**" + (f" — failed: {'; '.join(im['gate']['failed_checks'])}" if im["gate"]["failed_checks"] else "") + "\n")
        rows = [[k, v["events"], f(v["discovery_net"]["mean"] + C.ROUND_TRIP_COST if v["discovery_net"]["mean"] is not None else None, pct=True),
                 f(v["discovery_net"]["mean"], pct=True), f(v["discovery_net"]["t"], 2), f(v["holdout_net"]["mean"], pct=True), f(v["holdout_net"]["t"], 2),
                 f(v["discovery_daily_sharpe_ann"], 2)] for k, v in im["variants"].items()]
        L.append(table(["Variant", "Events", "Disc. gross", "Disc. net", "t", "Holdout net", "t", "Daily Sharpe"], rows))
        es = P["earnings_split_discovery"]
        L.append(f"\nMatched-control difference {f(P['discovery_event_minus_control']['mean'], pct=True)} (t={f(P['discovery_event_minus_control']['t'], 2)}); "
                 f"incremental regression coefficient {f(P['incremental_regression_discovery']['event_coef'], pct=True)} (t={f(P['incremental_regression_discovery']['t'], 2)}); "
                 f"deflated Sharpe {f(P['discovery_daily_portfolio']['deflated_sharpe_ratio'], 3)} over {P['discovery_daily_portfolio']['dsr_trials']} trials. "
                 + (f"Net excluding windows containing an earnings 8-K: {f(es['without_earnings_in_window']['mean'], pct=True)} (n={es['without_earnings_in_window']['n']})." if isinstance(es, dict) else "") + "\n")
    ins = load_insider(list(pd.read_parquet(C.PROC / "universe.parquet")["cik"].astype(int)))
    cut = pd.Timestamp(S["as_of_utc"][:10]) - pd.Timedelta(days=30)
    recent = ins[(ins["filing_date"] > cut) & (ins["trans_code"] == "P")].sort_values("value", ascending=False).head(15)
    if len(recent):
        L.append("Open-market purchases filed in the last 30 days (largest 15):\n")
        L.append(table(["Filed", "Issuer CIK", "Insider", "Title", "Shares", "Price", "Value $"],
                       [[str(r.filing_date.date()), r.issuercik, r.rptownername, r.rptowner_title or "", f"{r.trans_shares:,.0f}",
                         f"{r.trans_pricepershare:.2f}", f"{r.value:,.0f}"] for r in recent.itertuples()]))
    else:
        L.append("No open-market purchases by universe insiders filed in the last 30 days in the ingested data.")
    L.append("\n13F institutional holdings: NOT_COMPUTED (not ingested; 13F is quarterly, 45-day lagged and omits shorts).\n")

    # E
    L.append("## E. Model performance and validation\n")
    rows = []
    for hh, m in sorted(M.items()):
        wf, ho, bt = m["walk_forward"], m["holdout"], m["backtest_holdout"] or {}
        rows.append([hh, f"{m['wf_first_test_year']}–", f(wf["mean"]), f(wf["t"], 2), f(ho["mean"]), f(ho["t"], 2),
                     ho.get("effective_n", NC), f(m["wf_positive_year_frac"], 2), f(bt.get("mean_net_excess"), pct=True),
                     f(bt.get("net_sharpe_ann"), 2), m["calibration"]["status"], m["gate"]["status"]])
    L.append(table(["h", "WF test", "WF IC", "WF t", "Holdout IC", "HO t", "HO eff. N", "WF +yrs", "HO net excess/period",
                    "HO net Sharpe", "Calibration", "Gate"], rows))
    o = R["overfitting"]
    L.append(f"\nOverfitting ({o['primary_horizon']}d): PSR vs 0 = {f(o['psr_vs_zero'], 3)}, deflated Sharpe = {f(o['deflated_sharpe_ratio'], 3)} "
             f"over {o['n_trials_for_dsr']} trials, PBO (CSCV, univariate strategies) = {f(o['pbo_cscv_univariate_strategies'], 3)}, "
             f"WF→holdout IC degradation = {f(o['train_vs_test_ic_degradation'])}. Tested hypotheses: {R['tested_hypotheses']}.")
    L.append(f"Baselines at {h}d: 12-1 momentum OOS IC {f(mp['baseline_mom_12_1_oos_ic'])}, equal-weight hypothesis-signed composite "
             f"holdout IC {f(mp['baseline_equal_weight_holdout_ic'])}.")
    if mp.get("regime_ic"):
        L.append("\nRegime-conditioned walk-forward IC: " + ", ".join(f"{k}: {v['mean']:.4f} (n={v['n']}, t={f(v['t'], 2)})" for k, v in mp["regime_ic"].items()))
    L.append("")

    # F
    L.append("## F. Forecast distribution (top-decile forecast bucket, out-of-sample history)\n")
    rows = []
    for hh, m in sorted(M.items()):
        tab = pd.DataFrame(m["calibration"]["table"]).set_index("b")
        t9 = tab.loc[tab.index.max()]
        ok = m["calibration"]["status"] == "HISTORICAL_DECILE_CALIBRATED"
        rows.append([hh, f(t9["mean"], pct=True) if ok else NC, f"[{f(t9['q025'], pct=True)}, {f(t9['q975'], pct=True)}]" if ok else NC,
                     f(t9["p_pos"], 3) if ok else NC, f(t9["q05"], pct=True) if ok else NC, m["calibration"]["status"]])
    L.append(table(["Horizon", "Expected excess", "95% interval", "P(excess>0)", "5% quantile", "Calibration"], rows))
    L.append("")

    # G
    L.append("## G. Risk and implementation\n")
    L.append(f"- Cost model (GBP account buying US shares): FX {C.FX_FEE:.2%} + half-spread {C.HALF_SPREAD:.2%} + slippage {C.SLIPPAGE:.2%} per side "
             f"→ **{C.ROUND_TRIP_COST:.2%} round trip**. Turnover-adjusted in all net figures.")
    bt = mp["backtest_holdout"] or {}
    L.append(f"- Holdout top-quintile long-only at {h}d: mean turnover {f(bt.get('mean_turnover'), 2)}, avg names {f(bt.get('avg_names'), 0)}, "
             f"max drawdown of cumulative excess {f(bt.get('max_drawdown_excess'), pct=True)}.")
    L.append("- Factor-covariance risk model (Σ = BFB′ + D), GARCH, stress tests and optimiser: NOT_COMPUTED — sizing stays with the bot.")
    L.append("- Per-signal volatility (60d realised, horizon-scaled), beta (252d) and decile expected shortfall are in the JSON.\n")

    # H
    L.append("## H. Trading-bot signal decision\n")
    if longs:
        L.append(table(["Ticker", "Exp. excess", "Net of cost", "95% PI", "P(>0)", "Vol (h)", "Beta", "Expires"],
                       [[s["asset"]["ticker"], f(s["forecast"]["expected_excess_return"], pct=True), f(s["forecast"]["expected_net_excess_return"], pct=True),
                         str([round(x, 4) for x in s["forecast"]["prediction_interval_95"]]), f(s["forecast"]["probability_positive_excess_return"], 3),
                         f(s["risk"]["forecast_volatility"], pct=True), f(s["risk"]["estimated_beta"], 2), s["signal"]["expires_at_utc"]] for s in longs]))
    else:
        reason = S["signals"][0]["signal"]["reason"] if S["signals"] else "no eligible securities"
        L.append(f"**NO_SIGNAL for all {S['n_evaluated']} securities.** Primary reason: {mp['gate']['status']} at the model level"
                 f" ({'; '.join(mp['gate']['failed_checks']) or reason}).")
    L.append("")

    # I
    L.append("## I. Screener — top candidates\n")
    if longs:
        L.append(f"{len(longs)} securities passed all checks (see H). Fewer than 10 are shown when fewer pass.")
    else:
        prim = fc[fc["horizon"] == h].sort_values("raw_model_score", ascending=False).head(10)
        L.append("No security is eligible, so no ranked candidate list is issued. For research transparency only, the model's top "
                 "10 by raw score (NOT signals, not validated):\n")
        L.append(table(["Ticker", "Raw score", "Decile"], [[r.symbol, f(r.raw_model_score, 5), r.decile] for r in prim.itertuples()]))
    L.append("")

    # J
    L.append("## J. Limitations and reproduction\n")
    L += [f"- Model: cross-sectional ridge (α={C.RIDGE_ALPHA}×n/1000) on per-date winsorised ({C.WINSOR_Q:.0%}) percentile ranks, sector-neutralised (SIC); "
          f"one model per horizon; yearly expanding walk-forward refits with purge (h+1 sessions) and {5}-session embargo.",
          f"- Train/test: walk-forward from {mp['wf_first_test_year']}; holdout {R['sample']['holdout_start']} → {R['sample']['last_date']} evaluated once per run "
          "(re-running after changing features or config spends the holdout — log in ledger/research_runs.jsonl).",
          "- Known biases: Yahoo has no delisting returns (0% assumed); removed index members without a current SEC ticker are excluded; "
          "10b5-1 flag absent before 2023 so older 'discretionary' sales include plan sales; index membership is current+historical union, not a PIT index replay.",
          "- Not built: 13F, options/IV, short interest, estimate revisions (no PIT vendor), sector Mosaic modules 4A–4E, HMM regimes, factor risk model.",
          f"- Config hash `{R['config_hash']}`; JSON schema `schema/signal_v2.schema.json`; deterministic signal ids = sha256(model|run|CIK|date|h).",
          "- Next tests: paper-trade the bot on signals for ≥3 months; add point-in-time earnings-revision data; short-horizon (1–5d) insider event model, which the event study supports better than the 20d cross-section."]
    out = C.REPORTS / f"report_{S['as_of_utc'][:10]}.md"
    out.write_text("\n".join(L) + "\n")
    (C.REPORTS / "report_latest.md").write_text("\n".join(L) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    run()
