"""Institutional report (spec section 13, A-J) assembled only from computed artifacts. Every number is tagged
with its evidence class (spec section 1):
  [OBS]  VERIFIED_OBSERVATION   value read from a source record
  [CALC] CALCULATED_STATISTIC   computed from observations (ICs, t-stats, backtests)
  [MODEL] MODEL_ESTIMATE        model output (forecasts, risk-model volatility)
  [HYP]  HYPOTHESIS             pre-specified expectation, not evidence
  [UNVER] UNVERIFIED_INFORMATION rule-based or sponsor-estimated (e.g. EIA dates, trial completion estimates)
Anything not computed prints as NOT_COMPUTED with the reason.
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config as C
from features import FEATURES, load_insider

NC = "NOT_COMPUTED"


def f(x, nd=4, pct=False):
    if isinstance(x, str):
        try:
            x = float(x)
        except ValueError:
            return x
    if x is None or isinstance(x, (dict, list)) or (isinstance(x, float) and not np.isfinite(x)):
        return NC
    return f"{x * 100:.2f}%" if pct else f"{x:.{nd}f}"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def load(name):
    p = C.REPORTS / name
    return json.loads(p.read_text()) if p.exists() else None


def run():
    S = json.loads((C.SIGNALS / "latest.json").read_text())
    profiles = [m["profile"] for m in S["profiles"]]
    RR = {p: load(f"research_results_{p}.json") for p in profiles}
    prim = profiles[0] if profiles else None
    R = RR.get(prim)
    ES, IM, AU, CA, CR, BI, PF = (load(n) for n in ["insider_event_study.json", "insider_model_results.json", "data_audit.json", "catalysts.json",
                                                    "commodity_research.json", "biotech.json", "portfolio_risk.json"])
    fc = pd.read_parquet(C.REPORTS / "forecast_latest.parquet") if (C.REPORTS / "forecast_latest.parquet").exists() else pd.DataFrame()
    h = C.PRIMARY_HORIZON
    longs = [s for s in S["signals"] if s["signal"]["direction"] == "LONG"]
    L = [f"# Mosaic Quantamental Research Report — {S['as_of_utc'][:10]}\n",
         f"Feature set `{C.FEATURE_SET_VERSION}` · model `{C.MODEL_ID}` v{C.MODEL_VERSION} · generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · "
         "evidence tags: [OBS] observation, [CALC] calculated statistic, [MODEL] model estimate, [HYP] hypothesis, [UNVER] unverified\n"]

    # ---------------- A
    L.append("## A. Executive summary\n")
    reg = S["regime"]
    L.append(f"- **Regime [CALC]:** {reg['state']}; filtered HMM P(high-vol) = {f(reg['p_high_vol'], 2)}.")
    L.append(f"- **Data audit [CALC]:** {'passed' if S['data_audit_passed'] else 'FAILED — all LONG signals blocked'} (look-ahead truncation, label alignment, duplicates, timestamps).")
    for p in profiles:
        r = RR[p]
        if not r:
            continue
        e = r["models"][str(h)]["models"]["ensemble"]
        g = e["gate"]
        prof = C.PROFILES[p]
        L.append(f"- **Universe `{p}` [OBS]:** mcap ≥ ${prof['min_mcap'] / 1e9:g}bn, ADV ≥ {prof['min_adv_shares'] / 1e3:g}k shares and ≥ ${prof['min_dollar_adv'] / 1e6:g}m, "
                 f"point-in-time; {r['sample']['symbols']} names ever eligible, {r['sample']['avg_names_per_date']:.0f} per date; {r['sample']['first_date']} → {r['sample']['last_date']}, holdout from {r['sample']['holdout_start']}.")
        L.append(f"  - **Ensemble at {h}d [CALC]:** walk-forward IC {f(e['walk_forward']['mean'])} (t={f(e['walk_forward']['t'], 2)}), holdout IC {f(e['holdout']['mean'])} "
                 f"(t={f(e['holdout']['t'], 2)}); holdout net excess/period {f((e['backtest_holdout'] or {}).get('mean_net_excess'), pct=True)}. "
                 f"Gate: **{g['status']}**" + (f" — {'; '.join(g['failed_checks'])}" if g["failed_checks"] else ""))
    L.append(f"- **Signals:** {len(longs)} LONG of {S['n_evaluated']} evaluated ({S['n_insider_event_signals']} insider-event). "
             "A research thesis is not execution approval; the bot's own risk checks always apply.\n")

    # ---------------- B
    L.append("## B. Evidence and source audit\n")
    rows = [["Daily OHLCV, split/dividend adjusted", "close t; entry t+1 open", "Yahoo chart API", "vendor survivorship: no delisted prices (0% delisting return)", "yes"],
            ["Fundamentals (XBRL, as-reported)", "filed ≤ t−1", "SEC companyfacts", "first-filed value used; restated series kept separately", "yes"],
            ["Insider Forms 3/4/5 + footnotes + derivative table", "filed ≤ t−1", "SEC insider data sets, EDGAR XML", "10b5-1 from checkbox (2023+) or footnote", "yes"],
            ["13F holdings", "period end + 46d, on-time originals only", "SEC 13F data sets", "CUSIP→issuer by name match; omits shorts", "yes"],
            ["Short interest", "settlement + 12d", "FINRA consolidated (2020+)", "short history", "yes (2020+)"],
            ["Macro (rates, credit, dollar, oil products)", "obs + 1d", "FRED", "market series, unrevised", "yes"],
            ["Macro (UNRATE, PAYEMS, CPI, INDPRO, semis IP, orders)", "first ALFRED vintage containing the value", "ALFRED", "twice-monthly vintages (conservative lag)", "yes"],
            ["EIA petroleum status", "week end + 5d", "EIA history files", "", "yes"],
            ["CFTC COT (disaggregated)", "report + 3d", "CFTC Socrata", "", "yes"],
            ["Commodity futures", "close + 1d", "Yahoo continuous front month", "roll gaps; no term structure", "yes"],
            ["Earnings dates", "8-K item 2.02 acceptance time", "SEC submissions", "", "yes"],
            ["Clinical trials", "first-posted date", "ClinicalTrials.gov", "records edited in place: only first-posted/results-posted PIT-safe", "partial"],
            ["FDA approvals", "action date", "openFDA drugs@FDA", "approvals only", "yes"],
            ["Options / implied volatility / estimate revisions history", "—", "not available free", NC, "no"]]
    L.append(table(["Dataset", "Timestamp rule", "Source", "Quality notes", "PIT-eligible"], rows))
    if AU:
        d = AU["diagnostics"]
        L.append(f"\nAudit [CALC]: truncation invariance worst relative difference {f(AU['truncation_invariance']['worst'], 8)} over {len(AU['truncation_invariance']['dates'])} dates; "
                 f"label max error {f(AU['label_alignment']['max_abs_error'], 8)}; duplicate insider rows {AU['duplicates_timestamps']['duplicate_insider_rows']}; "
                 f"restated periods {f(d['restatement']['share_restated'], pct=True)}; series ending early {d['survivorship']['series_ending_before_last_session']}; "
                 f"rows with stale fundamentals {f(d['stale']['share_rows_fundamentals_older_than_limit'], pct=True)}. Missing-not-at-random IC: "
                 + ", ".join(f"{k} {f(v['ic_missing_vs_fwd_return'])}" for k, v in d["missing_not_at_random"].items()) + ".")
    coll = C.RAW / "collect"
    n_opt = len(list((coll / "options").glob("*.parquet"))) if (coll / "options").exists() else 0
    n_est = len(list((coll / "estimates").glob("*.parquet"))) if (coll / "estimates").exists() else 0
    n_av = len(list((coll / "av_earnings").glob("*.json"))) if (coll / "av_earnings").exists() else 0
    L.append(f"\nHistory being collected [OBS]: Cboe options snapshots {n_opt} days, Nasdaq estimate snapshots {n_est} days "
             f"(per-stock options/estimate features enter models after 3 years); Alpha Vantage consensus-surprise backfill "
             f"{n_av} companies (24/day, largest first). VIX term structure and CBOE SKEW: full history in use.")
    if len(fc) and longs:
        L.append("\nObserved values for LONG candidates are in `signals/latest.json` (key_drivers).")
    L.append("")

    # ---------------- C
    L.append("## C. Factor and sector Mosaic dashboard (discovery sample, sector-neutral, primary horizon) [CALC]\n")
    if R:
        uni = [u for u in R["univariate"] if u["horizon"] == h and u["variant"] == "sector_neutral"]
        sty = {u["feature"]: u for u in R["univariate"] if u["horizon"] == h and u["variant"] == "style_neutral"}
        raw = {u["feature"]: u for u in R["univariate"] if u["horizon"] == h and u["variant"] == "raw"}
        rows = []
        for u in sorted(uni, key=lambda u: (u["family"], -abs(u["t"] or 0))):
            rows.append([u["family"], u["feature"], {1: "+", -1: "−", 0: "±"}[u["hyp_sign"]] + " [HYP]", f(u["mean"]), f(u["t"], 2),
                         f((sty.get(u["feature"]) or {}).get("mean")), f(u["p_bh"], 3),
                         "✓" if u["fdr_reject"] and u["sign_matches_hypothesis"] else ("✗ sign" if u["fdr_reject"] else "—"),
                         f(raw.get(u["feature"], {}).get("top_minus_bottom"), pct=True)])
        L.append(table(["Family", "Feature", "Hyp.", "IC", "HAC t", "Style-neutral IC", "BH p", "FDR", "Q5−Q1"], rows))
        L.append(f"\nIC = per-date Spearman correlation with forward SPY-excess return; Newey-West lags = h/step; BH-FDR q={C.FDR_Q} over {len(R['univariate'])} tests "
                 "(one-sided where a sign was pre-specified). Sector features are NaN outside their sector. Contributions to forecasts are not reported where the gate failed.")
        for pp in profiles:
            FT = load(f"feature_test_{pp}.json")
            if FT:
                ok = [r for r in FT["results"] if r["fdr_reject_joint"] and r["sign_matches_hypothesis"]]
                best = {}
                for r in sorted(ok, key=lambda r: -abs(r["t"])):
                    best.setdefault(r["feature"], r)
                L.append(f"New idiosyncratic / lottery / stationarity features, `{pp}`, discovery period only [CALC]: {len(ok)} of {FT['tests_new']} tests pass BH-FDR jointly with all "
                         f"{FT['tests_in_joint_fdr']} tests: " + (", ".join(f"{k} ({r['variant']}, {r['horizon']}d, IC {r['mean']:+.3f}, t={r['t']:.1f})" for k, r in best.items()) or "none") + ".")
        MI = load(f"mi_screen_{prim}.json")
        if MI:
            top = [r for r in MI["features"] if r["mi_significant"]][:8]
            L.append(f"Mutual information (k-NN, permutation null, BH-FDR) [CALC]: {len([r for r in MI['features'] if r['mi_significant']])} of {len(MI['features'])} features "
                     f"show significant dependence; strongest: " + (", ".join(f"{r['feature']} (z={r['z_vs_null']:.1f})" for r in top) or "none")
                     + f". Non-linear only (MI significant, IC not): {', '.join(MI['nonlinear_only']) or 'none'}.")
        mc = R["multicollinearity"]
        L.append(f"Multicollinearity: {len(mc['pairs_abs_corr_gt_0_7'])} feature pairs with |ρ| > 0.7; max VIF {f(max(mc['vif'].values()) if mc['vif'] else None, 1)}.\n")

    # ---------------- D
    L.append("## D. Insider activity and ownership\n")
    if ES:
        rows = []
        for kind, res in ES["results"].items():
            for samp in ["discovery", "holdout"]:
                r = res[samp]
                rows.append([kind, samp, r["1d"]["n"]] + [f"{f(r[w]['mean'], pct=True)} (t={f(r[w]['t'], 2)})" for w in ["1d", "5d", "20d", "60d"]])
        L.append("Event study [CALC] (beta-adjusted CAR from first open after filing; SEs clustered by month):\n")
        L.append(table(["Event", "Sample", "N", "1d", "5d", "20d", "60d"], rows))
    if IM:
        P = IM["primary"]
        L.append(f"\nPre-registered insider event model (`{IM['prereg_id']}`) [CALC]: gate **{IM['gate']['status']}**; discovery net {f(P['discovery_net']['mean'], pct=True)} "
                 f"(t={f(P['discovery_net']['t'], 2)}), vs matched controls {f(P['discovery_event_minus_control']['mean'], pct=True)} (t={f(P['discovery_event_minus_control']['t'], 2)}), "
                 f"incremental regression t={f(P['incremental_regression_discovery']['t'], 2)}.")
    uni_ciks = list(pd.read_parquet(C.PROC / "universe.parquet")["cik"].astype(int))
    ins = load_insider(uni_ciks)
    cut = pd.Timestamp(S["as_of_utc"][:10]) - pd.Timedelta(days=30)
    rec = ins[ins["filing_date"] > cut]
    if len(rec):
        b = rec[rec["trans_code"] == "P"].sort_values("value", ascending=False).head(15)
        tick = pd.read_parquet(C.PROC / "universe.parquet").set_index("cik")["ticker"]
        L.append("\nLargest open-market purchases filed in the last 30 days [OBS]:\n")
        L.append(table(["Filed", "Ticker", "Insider", "Role", "Class", "Value $", "Plan"],
                       [[str(r.filing_date.date()), tick.get(r.issuercik, r.issuercik), r.rptownername, (r.rptowner_title or r.rptowner_relationship or ""),
                         r.insider_class, f"{r.value:,.0f}", "10b5-1" if r.is_10b5_1 else ""] for r in b.itertuples()]))
        net = rec.assign(sv=np.where(rec["trans_code"] == "P", rec["value"], np.where(~rec["is_10b5_1"], -rec["value"], 0))).groupby("insider_class")["sv"].sum()
        L.append(f"\nNet discretionary $ (buys − non-plan sales), last 30d, by insider class [CALC]: " + ", ".join(f"{k} ${v / 1e6:,.1f}m" for k, v in net.items())
                 + ". Routine/opportunistic per Cohen-Malloy-Pomorski using only each insider's earlier trades.")
        clus = rec[rec["trans_code"] == "P"].groupby("issuercik")["owner"].nunique()
        L.append(f"Cluster indicator [CALC]: {int((clus >= 2).sum())} issuers with ≥2 distinct buyers in the last 30 days.")
    io = C.PROC / "inst_ownership_cusip.parquet"
    if io.exists():
        L.append("\n13F [CALC]: ownership ratio, breadth change (Chen-Hong-Stein), holder-count change and HHI enter the factor tests in section C. "
                 "13F is filed up to 45 days after quarter end, omits short positions and non-13(f) securities, and holding-value changes are not flows.")
    L.append("")

    # ---------------- E
    L.append("## E. Model performance and validation [CALC]\n")
    for p in profiles:
        r = RR[p]
        if not r:
            continue
        L.append(f"**Profile `{p}`** — {r['sample']['rows']:,} security-dates; costs per name (mean one-way {f(r['costs']['mean_one_way'], pct=True)} at ${r['costs']['order_size_usd']:,.0f} orders).\n")
        rows = []
        for hh, res in sorted(r["models"].items(), key=lambda x: int(x[0])):
            for n, m in res["models"].items():
                if hh != str(h) and n != "ensemble":
                    continue
                bt = m["backtest_holdout"] or {}
                rows.append([hh, n, f(m["walk_forward"]["mean"]), f(m["walk_forward"]["t"], 2), f(m["holdout"]["mean"]), f(m["holdout"]["t"], 2),
                             str([round(x, 3) for x in (m["holdout"].get("ci95_block_bootstrap") or [])]) or NC, m["holdout"].get("effective_n", NC),
                             f(m["wf_positive_year_frac"], 2), f(bt.get("mean_net_excess"), pct=True), f(bt.get("net_sharpe_ann"), 2), m.get("gate", {}).get("status", "")])
        L.append(table(["h", "Model", "WF IC", "WF t", "HO IC", "HO t", "HO IC 95% (block bootstrap)", "HO eff. N", "WF +yrs", "HO net/period", "HO net SR", "Gate"], rows))
        e = r["models"][str(h)]["models"]["ensemble"]
        o = r["overfitting"]
        spa = o.get("reality_check_spa") or {}
        L.append(f"\nOverfitting: {o['strategies_tested']} strategies; deflated Sharpe {f(o['deflated_sharpe_ratio'], 3)}, PSR {f(o['psr_vs_zero'], 3)}, PBO {f(o['pbo_cscv'], 3)}, "
                 f"White RC p {f(spa.get('reality_check_p'), 3)}, Hansen SPA p {f(spa.get('spa_p'), 3)}, MinTRL {f(o['min_track_record_periods'], 0)} periods, "
                 f"WF→holdout IC degradation {f(o['train_vs_test_ic_degradation'])}.")
        fa = e.get("factor_adjusted_alpha_holdout") or e.get("factor_adjusted_alpha_wf")
        if fa:
            L.append(f"Factor-adjusted alpha (holdout long-only vs risk-model factors): {f(fa['alpha_per_period'], pct=True)}/period (HAC t={f(fa['alpha_t_hac'], 2)}), R² {f(fa['r2'], 2)}; "
                     f"diagnostics: Durbin-Watson {f(fa['durbin_watson'], 2)}, Jarque-Bera p {f(fa['jarque_bera_p'], 3)}, Breusch-Pagan p {f(fa['breusch_pagan_p'], 3)}, max VIF {f(max(fa['vif'].values()), 1)}.")
        cal = e["calibration"]
        conf = cal.get("conformal") or {}
        L.append(f"Calibration: decile monotonicity WF {f(cal['wf_monotonicity'], 2)}, holdout {f(cal['holdout_monotonicity'], 2)} → {cal['status']}; "
                 f"Mondrian conformal 95% interval holdout coverage {f(conf.get('holdout_coverage'), pct=True)}.")
        pm = e.get("probability_model")
        if pm:
            L.append(f"Probability model {pm['target']}: holdout Brier {f(pm['holdout_brier'], 4)} vs base-rate {f(pm['holdout_base_rate_brier'], 4)}.")
        L.append(f"Model disagreement (mean cross-model rank SD): {f(e.get('model_disagreement_mean_rank_std'), 3)}.")
        rc = r["models"][str(h)].get("regime_conditioning")
        if rc:
            L.append(f"Regime conditioning (filtered HMM interactions) vs unconditional: ΔIC {f(rc['wf_ic_difference']['mean'])} (t={f(rc['wf_ic_difference']['t'], 2)}) → "
                     f"{'improves' if rc['improves'] else 'no reliable improvement; unconditional model kept'}.")
        RG = load(f"regime_research_{p}.json")
        if RG:
            ms = RG["markov_switching"]
            L.append(f"Markov-switching regression: current filtered P(high-vol) {f(ms['current_p_high_vol'], 2)}; regime weight sets correlate {f(ms['weight_vector_correlation_high_vs_low'], 2)}; "
                     f"walk-forward ΔIC {f(ms['walk_forward']['difference']['mean'])} (t={f(ms['walk_forward']['difference']['t'], 2)}), holdout ΔIC {f(ms['holdout']['difference']['mean'])} "
                     f"(t={f(ms['holdout']['difference']['t'], 2)}) → {'adopt' if ms['improves'] else 'not adopted'}.")
            br = [(c["series"], b) for c in RG["change_points"] for b in c["breaks"] if b["p"] < 0.001]
            L.append("Structural breaks (p<0.001): " + ("; ".join(f"{s_} {b['date']} ({b['mean_before']:+.3f}→{b['mean_after']:+.3f})" for s_, b in br[:8]) or "none") + ".")
            h3 = RG.get("hmm_2_vs_3_states")
            if h3:
                L.append(f"HMM 2 vs 3 states: 3-state better out-of-sample likelihood in {h3['years_3_state_better']}/{h3['years']} years (t={f(h3['t'], 2)}) → live regime uses 3 states; "
                         f"ranking improvement t={f(h3['ranking_test_3_state_switching_ridge']['wf_ic_difference']['t'], 2)} (not adopted for forecasts).")
            fr = {k: v for k, v in RG["factor_regimes"].items() if abs(v["difference_t_hac"]) >= 2}
            L.append("Factor returns that differ by regime (|t|≥2): " + (", ".join(f"{k} calm {f(v['calm_ann_mean'], pct=True)} vs stress {f(v['stress_ann_mean'], pct=True)} (t={f(v['difference_t_hac'], 2)})" for k, v in fr.items()) or "none") + ".")
        BL = load(f"baselines_{p}.json")
        if BL:
            rows = [[k.replace("_", " "), f(v["model_mean_net_per_period"], pct=True), f(v["random_mean_net_median"], pct=True),
                     f"[{f(v['random_mean_net_q05'], pct=True)}, {f(v['random_mean_net_q95'], pct=True)}]", f(v["model_percentile"], pct=True)]
                    for k, v in BL["results"].items() if "model_percentile" in v]
            L.append(f"\nMatched random portfolios ({BL['paths']} paths; same names count, rebalancing, turnover and per-name costs):\n")
            L.append(table(["Test", "Model net/period", "Random median", "Random 5–95%", "Model percentile"], rows))
            for k, v in BL["results"].items():
                if k.endswith("bootstrap") and v:
                    for lab, b in v.items():
                        if b:
                            L.append(f"Block bootstrap {k.split('_')[0]} {lab}: mean net 5/50/95% {', '.join(f(x, pct=True) for x in b['mean_net_q05_q50_q95'])}; "
                                     f"max drawdown 5/50/95% {', '.join(f(x, pct=True) for x in b['max_drawdown_q05_q50_q95'])}; P(mean < 0) {f(b['prob_mean_below_zero'], 2)}.")
        if RG and RG.get("observable_markov"):
            om = RG["observable_markov"]
            L.append(f"Observable Markov chain on SPY (±{om['threshold']:.0%} over {om['window']} days): long/cash Sharpe {f(om['long_cash_discovery']['sharpe_ann'], 2)} vs SPY "
                     f"{f(om['long_cash_discovery']['spy_sharpe_ann'], 2)} (discovery), {f(om['long_cash_holdout']['sharpe_ann'], 2)} vs {f(om['long_cash_holdout']['spy_sharpe_ann'], 2)} (holdout); "
                     f"signal vs next 20d return t={f(om['signal_vs_next_20d_spy_return_discovery']['t'], 2)} → not adopted.")
        rob = r["models"][str(h)].get("robustness", {})
        if rob:
            L.append("\nRobustness (walk-forward ensemble IC by split):\n")
            rows = [[k, sk, f(v["mean_ic"]), f(v["t"], 2), v["n_dates"]] for k in ["cap_tercile", "liquidity_tercile", "regime", "earnings_window", "sector"] for sk, v in rob.get(k, {}).items()]
            L.append(table(["Split", "Group", "IC", "t", "Dates"], rows))
            L.append(f"\nCost sensitivity (WF net/period at cost multiples): " + ", ".join(f"{k} {f(v, pct=True)}" for k, v in rob["cost_sensitivity_wf_net_per_period"].items())
                     + f". Top-quantile sensitivity: " + ", ".join(f"{k} {f(v, pct=True)}" for k, v in rob["top_quantile_sensitivity_wf_net"].items())
                     + f". Ridge α sensitivity (WF IC): " + ", ".join(f"{k} {f(v)}" for k, v in rob["ridge_alpha_sensitivity_wf_ic"].items())
                     + f". Equal-weight-universe benchmark WF net/period {f((rob.get('benchmark_equal_weight_universe_wf') or {}).get('mean_net_excess'), pct=True)}.")
        L.append("")

    # ---------------- F
    L.append("## F. Forecast distribution (top ensemble decile, out-of-sample history)\n")
    for p in profiles:
        r = RR[p]
        if not r:
            continue
        rows = []
        for hh, res in sorted(r["models"].items(), key=lambda x: int(x[0])):
            cal = res["models"]["ensemble"]["calibration"]
            tab = pd.DataFrame(cal["table"]).set_index("b") if cal["table"] else None
            ok = cal["status"] == "HISTORICAL_DECILE_CALIBRATED" and tab is not None
            t9 = tab.loc[tab.index.max()] if tab is not None else None
            rows.append([p, hh, f(t9["mean"], pct=True) if ok else NC, f"[{f(t9['q025'], pct=True)}, {f(t9['q975'], pct=True)}]" if ok else NC,
                         f(t9["p_pos"], 3) if ok else NC, f(t9["q05"], pct=True) if ok else NC, cal["status"]])
        L.append(table(["Profile", "Horizon", "Expected excess [MODEL]", "95% interval", "P(excess>0)", "5% quantile", "Calibration"], rows))
    L.append("\nFields stay NOT_COMPUTED unless the ensemble's deciles were monotonic both in walk-forward and holdout.\n")

    # ---------------- G
    L.append("## G. Risk and implementation\n")
    lim = json.loads((C.PROC / "bot_risk_limits.json").read_text()) if (C.PROC / "bot_risk_limits.json").exists() else {}
    L.append(f"- Bot limits imported read-only [OBS]: position {f(lim.get('position_pct'), pct=True)} of account (≈ ${f(lim.get('order_size_usd'), 0)}), stop-loss "
             f"{f(lim.get('stop_loss'), pct=True)}, long-only {lim.get('long_only')}, leverage {lim.get('leverage')}, account ≈ ${f(lim.get('account_total_usd'), 0)}.")
    L.append(f"- Cost model [MODEL]: FX {C.FX_FEE:.2%} per side + Corwin-Schultz half-spread + 0.7·σ·√(order/ADV) impact at the bot's order size.")
    if PF:
        cr = PF.get("current_risk")
        if isinstance(cr, dict):
            L.append(f"- Bot's current demo holdings [MODEL]: gross {f(cr['gross_exposure'], pct=True)}, factor-model vol {f(cr['vol_annual'], pct=True)}/yr, "
                     f"factor share of variance {f(cr['factor_share_of_var'], pct=True)}; exposures " + ", ".join(f"{k} {v:+.2f}" for k, v in cr["exposures"].items()) + ".")
            tl = PF.get("tail_liquidity") or {}
            L.append(f"- Tail [CALC]: 97.5% daily ES {f(tl.get('es_97_5_daily'), pct=True)}, VaR {f(tl.get('var_97_5_daily'), pct=True)}, 2y max drawdown {f(tl.get('max_drawdown_2y'), pct=True)}.")
            st = PF.get("stress") or {}
            L.append("- Stress [CALC/MODEL]: " + "; ".join(f"{k} {f(v['portfolio_return'], pct=True)}" + (f" (SPY {f(v['spy_return'], pct=True)}, corr {f(v.get('avg_pairwise_corr_window'), 2)} vs {f(v.get('avg_pairwise_corr_prior_year'), 2)})" if 'spy_return' in v else "") for k, v in st.items()))
        else:
            L.append(f"- Bot's current demo holdings: {cr}.")
        L.append(f"- Optimiser proposal: {PF['proposal']['status']}" + (f" — {PF['proposal'].get('reason')}" if PF["proposal"].get("reason") else ""))
    VB = load("var_backtest.json")
    if VB:
        rows = []
        for ser, v in VB["series"].items():
            for m_, r_ in v.items():
                rows.append([ser, m_, f(r_["0.95"]["kupiec"]["rate"], pct=True), f(r_["0.95"]["kupiec"]["p"], 3), f(r_["0.99"]["kupiec"]["rate"], pct=True),
                             f(r_["0.99"]["kupiec"]["p"], 3), f(r_["0.99"]["christoffersen"]["p"], 3), f((r_.get("es_975") or {}).get("t"), 2), "✓" if r_["passes"] else "—"])
        L.append("\nVaR / ES backtest since 2010 [CALC] (Kupiec = right number of breaches; Christoffersen = breaches not clustered; ES t > 2 = shortfall underestimated):\n")
        L.append(table(["Series", "Model", "95% breaches", "Kupiec p", "99% breaches", "Kupiec p", "Christoffersen p", "ES t", "Passes"], rows))
        ss = VB["stock_sample"]
        L.append(f"\n40-stock sample pass rate: " + ", ".join(f"{k} {f(v, pct=True)}" for k, v in ss["pass_rate"].items())
                 + f". **Winner: {VB['winner']['single_name']}** → per-name volatility in signals and the bot's VaR-capped sizing (1-day 99% VaR ≤ 1% of the account).")
    KB = load(f"kalman_beta_{prim}.json")
    if KB:
        L.append(f"- Beta forecasts 2015+ [CALC]: MSE rolling-252 {f(KB['mse']['rolling_252'], 3)}, Blume {f(KB['mse']['blume'], 3)}, Kalman {f(KB['mse']['kalman'], 3)} "
                 f"(Kalman vs rolling t={f(KB['kalman_vs_rolling']['t'], 2)}) → **{KB['winner']}** used for signal betas.")
    if PF and PF.get("tail_dependence"):
        td = PF["tail_dependence"]
        L.append(f"- Tail dependence, t-copula [CALC] ({td['basket']}): mean lower-tail dependence with SPY {f(td['mean_tail_dependence_vs_spy'], 3)}, "
                 f"mean pairwise {f(td['mean_pairwise_tail_dependence'], 3)}; t-copula preferred to Gaussian for {f(td['share_t_preferred_over_gaussian'], pct=True)} of names.")
    rm = C.PROC / f"risk_model_{prim}.json"
    if rm.exists():
        m = json.loads(rm.read_text())
        L.append(f"- Factor risk model [MODEL]: Σ = BFB′ + D with {len(m['factors'])} factors (market, 7 styles, sectors); EWMA F (126d half-life) shrunk {f(m['shrinkage_intensity'], 2)} toward diagonal.")
    L.append("- GARCH(1,1) volatility is computed per name on request (risk_model.garch_vol); signals use EWMA (λ≈0.94) volatility.\n")

    # ---------------- H
    L.append("## H. Trading-bot signal decision\n")
    if longs:
        L.append(table(["Ticker", "Model", "Exp. excess", "Net", "95% PI", "P(>0)", "Vol (h)", "Beta", "Expires"],
                       [[s["asset"]["ticker"], s["model_id"], f(s["forecast"]["expected_excess_return"], pct=True), f(s["forecast"]["expected_net_excess_return"], pct=True),
                         str([round(x, 4) for x in s["forecast"]["prediction_interval_95"]]) if s["forecast"]["prediction_interval_95"] else NC,
                         f(s["forecast"]["probability_positive_excess_return"], 3), f(s["risk"]["forecast_volatility"], pct=True),
                         f(s["risk"]["estimated_beta"], 2), s["signal"]["expires_at_utc"]] for s in longs]))
    else:
        reasons = "; ".join(f"`{m['profile']}`: {m['model_gate_primary']['status']}" for m in S["profiles"])
        L.append(f"**NO_SIGNAL for all {S['n_evaluated']} entries.** Model gates — {reasons}. Each entry's `signal.reason` lists every failed check.")
    L.append("")

    PT = load("paper_trading.json")
    if PT:
        ex = PT["executed_demo_book"]
        L.append(f"**Paper trading (Trading 212 demo) [CALC]:** {ex.get('closed_trades', 0)} closed signal trades, open {ex.get('open_signal_positions') or 'none'}, "
                 f"{ex.get('skipped_by_pretrade_check', 0)} entries skipped by the pre-trade price check; forward record {PT['forecast_dates_recorded']} nightly forecast dates. "
                 f"Live-review recommendation: **{PT['live_review_recommendation']}** ({PT['rule']}).")
        if ex.get("closed_trades"):
            L.append(f"Realised: mean net excess {f(ex.get('mean_net_excess'), pct=True)} (t={f(ex.get('t'), 2)}), hit rate {f(ex.get('hit_rate'), 2)}, fees {f(ex.get('mean_fees'), pct=True)}, "
                     f"slippage vs pre-trade quote {f(ex.get('mean_slippage'), pct=True)}, vs expected {f(ex.get('mean_expected_net'), pct=True)}.")
        for k, v in PT["shadow_book"].items():
            L.append(f"- Shadow `{k}`: {v['scored_dates']} scored dates, forward IC {f(v['forward_ic_mean'])} (t={f(v['forward_ic_t'], 2)}), "
                     f"top-quintile net {f(v['top_quintile_net_mean'], pct=True)} over {v['non_overlapping_cohorts']} non-overlapping cohorts — {PT['shadow_status'].get(k)}")
        L.append("")

    # ---------------- I
    L.append("## I. Screener — top candidates\n")
    if longs:
        L.append(f"{len(longs)} securities passed every check (section H); fewer than 10 are shown when fewer pass.")
    else:
        L.append("No security passed validation, so no ranked list is issued. Showing unsupported rankings would violate the spec.")
    L.append("")

    # ---------------- 4 / 11 modules
    L.append("## Sector modules and catalysts\n")
    if CR:
        co = CR["crude_products_cointegration"]["full"]
        inv = CR["eia_inventory_surprise"]
        L.append(f"- Energy/commodities [CALC]: crude–gasoline Engle-Granger p {f(co['engle_granger_p']['gasoline'], 4)}, Johansen rank {co['johansen_rank_95']}, crack half-life "
                 f"{f(co['half_life_months'], 1)} months; EIA inventory-surprise effect on next-week WTI t={f(inv['discovery']['t'], 2)} (discovery) / {f(inv['holdout']['t'], 2)} (holdout). "
                 + "; ".join(f"{k}: equity beta {f(v['median_equity_beta'], 2)}, {v['commodity']} loading {f(v['median_commodity_loading'], 2)}" for k, v in CR["equity_vs_commodity_exposure"].items()) + ".")
        sig_seas = [c for c, r in CR["commodities"].items() if r["seasonality"]["wald_p_discovery"] < 0.05 and (r["seasonality"]["holdout_sign_agreement"] or 0) >= 0.66]
        L.append(f"- Commodity seasonality significant in discovery and ≥2/3 sign agreement in holdout: {', '.join(sig_seas) or 'none'} (RBOB continuous series includes seasonal-grade rolls).")
    MS = load("macro_surprise.json")
    if MS:
        L.append("- Macro surprises vs ARIMA forecast from earlier first releases (spec 4D) [MODEL]: " + "; ".join(
            f"{k} z={f(v['latest']['z'], 2)} (ARIMA median error {f(v['mae_arima'], 3)} vs naive {f(v['mae_random_walk'], 3)})" for k, v in MS["series"].items()) + ".")
    if BI:
        L.append(f"- Biotech [CALC]: {BI['issuers']} issuers; {len(BI['short_runway_lt_1y'])} with < 1 year cash runway; registry cohorts " +
                 ", ".join(f"{k}: {f(v['share_terminated_or_withdrawn'], pct=True)} terminated/withdrawn (n={v['n']})" for k, v in BI["registry_cohorts"].items()) +
                 ". Clinical success probabilities: NOT_COMPUTED (licensed outcome data required).")
    if CA:
        L.append(f"- Catalyst calendar next 90 days [OBS/UNVER]: {CA['n_events']} events — " + ", ".join(f"{k} {v}" for k, v in CA["by_type"].items()) + ". "
                 "Not computed: " + "; ".join(CA["not_computed"].values()) + ".")
        hr = CA["historical_responses"]
        e = hr.get("earnings", {})
        L.append(f"- Historical responses [CALC]: earnings median |abnormal| {f(e.get('pooled_abs_reaction_median'), pct=True)} (p90 {f(e.get('pooled_abs_reaction_p90'), pct=True)}), "
                 f"drift d2–20 after up/down reactions {f(e.get('drift_2_20_after_positive_reaction'), pct=True)} / {f(e.get('drift_2_20_after_negative_reaction'), pct=True)}; "
                 f"FOMC days |SPY| ×{f(hr.get('fomc', {}).get('ratio'), 2)} normal; FDA approvals 2-day abnormal mean {f(hr.get('fda_approval', {}).get('mean_abnormal_2d'), pct=True)} "
                 f"(n={hr.get('fda_approval', {}).get('events')}).")
    L.append("")

    # ---------------- J
    L.append("## J. Limitations and reproduction\n")
    L += ["- Models: ridge, elastic net, Huber, LightGBM, GAM (splines on 10 pre-specified features), hierarchical sector-pooled ridge, isotonic-calibrated logistic "
          "(P(excess > cost)), factor-residual ridge; ensemble members chosen each year from earlier walk-forward ICs only. Hyper-parameters fixed, not tuned on the holdout.",
          "- Validation: yearly expanding walk-forward with purge (h+1) and 5-session embargo; 2-year holdout. The large-cap holdout was already evaluated by research v1; "
          "every run is appended to `ledger/research_runs.jsonl`, and paper trading is the only untouched test left.",
          "- Known biases/gaps: no delisted-company prices (vendor survivorship), 321 removed S&P names without current tickers excluded; no point-in-time analyst "
          "estimate history (only forward consensus in the calendar); no options/IV data; 13F CUSIP mapping by issuer name; ALFRED vintages twice-monthly; "
          "ClinicalTrials records are edited in place; continuous futures include roll gaps.",
          "- Section 13 B's per-feature observed-value table is held in the JSON (`key_drivers`) rather than printed for every security.",
          f"- Versions: feature set {C.FEATURE_SET_VERSION}; run ids " + ", ".join(f"`{RR[p]['run_id']}`" for p in profiles if RR[p]) + "; schema `schema/signal_v2.schema.json`.",
          "- Next tests: 3+ months of paper trading with the bot reading LONG signals only; licensed estimate-revision and options data; delisted-price data (CRSP/Norgate)."]
    txt = "\n".join(L) + "\n"
    (C.REPORTS / f"report_{S['as_of_utc'][:10]}.md").write_text(txt)
    (C.REPORTS / "report_latest.md").write_text(txt)
    print("wrote report", len(txt), "chars")


if __name__ == "__main__":
    run()
