"""Insider purchase event model (pre-registered: prereg/insider_event_v1.json).

Evaluates the frozen primary rule and its 11 counted secondary variants, runs the section 3.3
controls (matched control firms, earnings-window split, incremental-value regression, subgroups),
applies the pre-registered gate, and stores the historical conditional return distribution that
live signals quote. `live_signals()` turns today's qualifying Form 4 filings into schema-v2 signals.
"""
import hashlib
import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import statsmodels.api as sm

import config as C
from features import load_insider, load_prices, sic_sector
from research import deflated_sr, psr, sharpe_stats

PREREG = C.ROOT / "prereg" / "insider_event_v1.json"
REG = json.loads(PREREG.read_text())
REG_HASH = hashlib.sha256(PREREG.read_bytes()).hexdigest()
MODEL_ID, MODEL_VERSION = "insider-event", "1.1.0"
COST = REG["primary"]["cost_round_trip"]
DEDUP = 20
MIN_DV, MIN_PX = 20e6, 5.0
HOLDOUT = pd.Timestamp("2024-10-16")
VARIANTS = [("primary", 5)] + [(v["trigger"], h) for v in REG["secondary_variants_counted_as_trials"] for h in v["h"]]


# ------------------------------------------------------------- data
class Market:
    def __init__(self):
        P, sym2cik = load_prices()
        self.P, self.cal = P, P["adjclose"].index
        self.sym_of = {c: s for s, c in sym2cik.items()}
        self.syms = list(P["adjclose"].columns)
        op = P["adj_open"]
        self.op = op.to_numpy()
        self.spy_op = P["spy_open"].to_numpy()
        r = P["adjclose"].pct_change(fill_method=None)
        rs = P["spy_adjclose"].pct_change(fill_method=None)
        self.beta = (r.rolling(252, min_periods=200).cov(rs).div(rs.rolling(252, min_periods=200).var(), axis=0)).to_numpy()
        dv = (P["close"] * P["volume"]).rolling(60, min_periods=40).mean()
        self.dv = dv.to_numpy()
        self.close = P["close"].to_numpy()
        ac = P["adjclose"]
        self.mom = (ac.shift(21) / ac.shift(252) - 1).to_numpy()
        self.rev = (ac / ac.shift(21) - 1).to_numpy()
        sic = pd.read_parquet(C.PROC / "sic.parquet").set_index("cik")["sic"]
        self.sector = np.array([sic_sector(sic.get(sym2cik[s])) for s in self.syms])
        self.col = {s: j for j, s in enumerate(self.syms)}
        vix = pd.read_parquet(C.RAW / "prices" / "^VIX.parquet").set_index("date")["close"].reindex(self.cal).ffill()
        spy_ac = P["spy_adjclose"]
        up = (spy_ac > spy_ac.rolling(200).mean()).to_numpy()
        hv = (vix > vix.rolling(1260, min_periods=252).median()).to_numpy()
        self.regime = np.where(up, "up", "down").astype(object) + "/" + np.where(hv, "hivol", "lovol").astype(object)

    def ok_liquid(self, j, i):
        return self.dv[i, j] >= MIN_DV and self.close[i, j] >= MIN_PX

    def ar(self, j, e, h):
        """beta-adjusted open-to-open abnormal return from entry session e to e+h (NaN if unavailable)"""
        x = e + h
        if x >= len(self.cal) or np.isnan(self.op[e, j]) or np.isnan(self.op[x, j]) or np.isnan(self.beta[e - 1, j]):
            return np.nan
        return (self.op[x, j] / self.op[e, j] - 1) - self.beta[e - 1, j] * (self.spy_op[x] / self.spy_op[e] - 1)


def buys(ins):
    b = ins[(ins["trans_code"] == "P") & ~ins["is_10b5_1"]].copy()
    b["owner"] = b["rptownercik"].fillna(b["rptownername"])
    return b.sort_values("filing_date")


def triggers(b, kind):
    """(cik, filing_date) rows for each trigger definition; cluster = filing completing >=2 distinct buyers in 30d"""
    out = []
    for cik, g in b.groupby("issuercik"):
        fd, ow, ceo = g["filing_date"].to_numpy(), g["owner"].to_numpy(), g["is_ceo_cfo"].to_numpy()
        for i in range(len(g)):
            lo = np.searchsorted(fd, fd[i] - np.timedelta64(30, "D"))
            clus = len(set(ow[lo:i + 1])) >= 2
            hit = {"any_purchase": True, "ceo_cfo_only": bool(ceo[i]), "cluster_only": clus,
                   "primary": bool(ceo[i]) or clus}[kind]
            if hit:
                out.append((int(cik), fd[i], bool(ceo[i]), clus))
    return pd.DataFrame(out, columns=["cik", "fdate", "ceo_cfo", "cluster"])


def build_events(m, b, kind, h):
    trig = triggers(b, kind)
    rows, last = [], {}
    for cik, fd, ceo, clus in trig.sort_values("fdate").itertuples(index=False):
        s = m.sym_of.get(cik)
        if s is None:
            continue
        j = m.col[s]
        e = int(m.cal.searchsorted(pd.Timestamp(fd), side="right"))
        if e < 253 or e >= len(m.cal):
            continue
        if cik in last and e - last[cik] < DEDUP:
            continue
        if not m.ok_liquid(j, e - 1):
            continue
        last[cik] = e
        rows.append({"cik": cik, "symbol": s, "j": j, "fdate": pd.Timestamp(fd), "e": e, "entry_date": m.cal[e],
                     "ceo_cfo": ceo, "cluster": clus, "ar": m.ar(j, e, h), "sector": m.sector[j], "regime": m.regime[e - 1],
                     "mom": m.mom[e - 1, j], "rev": m.rev[e - 1, j], "ldv": math.log(m.dv[e - 1, j]), "beta": m.beta[e - 1, j]})
    ev = pd.DataFrame(rows)
    if len(ev):
        ev["net"] = ev["ar"] - COST
        ev["sample"] = np.where(ev["entry_date"] < HOLDOUT, "discovery", "holdout")
    return ev


# ------------------------------------------------------------- statistics
def clustered_mean(x, dates):
    s = pd.DataFrame({"x": x, "d": pd.DatetimeIndex(dates).to_period("M").astype(str)}).dropna()
    if len(s) < 20:
        return {"n": int(len(s)), "mean": None, "se": None, "t": None, "ci95": None}
    res = sm.OLS(s["x"].to_numpy(), np.ones(len(s))).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s["d"])[0]})
    mu, se = float(res.params[0]), float(res.bse[0])
    return {"n": int(len(s)), "mean": mu, "se": se, "t": mu / se, "ci95": [mu - 1.96 * se, mu + 1.96 * se],
            "median": float(s["x"].median()), "hit_rate": float((s["x"] > 0).mean())}


def daily_portfolio(m, ev, h):
    """Equal-weight book of open event positions; daily open-to-open abnormal return, entry cost charged
    on the entry day. Days with no positions earn 0 (excess cash)."""
    n = len(m.cal)
    tot, cnt, cst = np.zeros(n), np.zeros(n), np.zeros(n)
    for j, e in zip(ev["j"], ev["e"]):
        if e + h >= n:
            continue
        for k in range(e + 1, e + h + 1):
            a, b = m.op[k - 1, j], m.op[k, j]
            if np.isnan(a) or np.isnan(b):
                continue
            tot[k] += (b / a - 1) - m.beta[e - 1, j] * (m.spy_op[k] / m.spy_op[k - 1] - 1)
            cnt[k] += 1
        cst[e + 1] += COST
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(cnt > 0, (tot - cst) / np.maximum(cnt, 1), 0.0)
    first = ev["e"].min() if len(ev) else 0
    return pd.Series(r[first:], index=m.cal[first:])


def matched_controls(m, ev, b, h, details=False):
    """Same sector, liquid, no code-P filing within +/-90d; nearest on (log dollar volume, 12-1 momentum) ranks."""
    bdates = {c: g["filing_date"].to_numpy() for c, g in b.groupby("issuercik")}
    cik_of = np.array([int(next(c for c, s in m.sym_of.items() if s == sym)) for sym in m.syms])
    out = []
    for r in ev.itertuples():
        i = r.e - 1
        ldv = np.log(np.where(m.dv[i] > 0, m.dv[i], np.nan))
        cand = (m.sector == r.sector) & (m.dv[i] >= MIN_DV) & (m.close[i] >= MIN_PX) & ~np.isnan(m.mom[i]) & ~np.isnan(ldv)
        cand[r.j] = False
        for jj in np.nonzero(cand)[0]:
            d = bdates.get(cik_of[jj])
            if d is not None:
                k = np.searchsorted(d, np.datetime64(r.fdate) - np.timedelta64(90, "D"))
                if k < len(d) and d[k] <= np.datetime64(r.fdate) + np.timedelta64(90, "D"):
                    cand[jj] = False
        if not cand.any() or np.isnan(m.mom[i, r.j]):
            out.append({})
            continue
        live = ~np.isnan(m.mom[i]) & ~np.isnan(ldv)
        rk_m = pd.Series(m.mom[i]).rank(pct=True).to_numpy()
        rk_d = pd.Series(ldv).rank(pct=True).to_numpy()
        dist = np.where(cand & live, (rk_m - rk_m[r.j]) ** 2 + (rk_d - rk_d[r.j]) ** 2, np.inf)
        jj = int(np.argmin(dist))
        out.append({"ar": m.ar(jj, r.e, h), "symbol": m.syms[jj], "mom": m.mom[i, jj],
                    "rev": m.rev[i, jj], "ldv": ldv[jj], "beta": m.beta[i, jj], "sector": m.sector[jj],
                    "entry_date": r.entry_date})
    frame = pd.DataFrame(out, index=ev.index).reindex(columns=["ar", "symbol", "mom", "rev", "ldv", "beta", "sector", "entry_date"])
    return frame if details else frame["ar"].to_numpy()


def incremental_regression(ev, controls):
    cols = ["ar", "mom", "rev", "ldv", "beta", "sector", "entry_date"]
    a = ev[cols]
    c = controls.reindex(ev.index)[cols]
    paired = a.notna().all(axis=1) & c.notna().all(axis=1)
    d = pd.concat([a[paired].assign(event=1.0), c[paired].assign(event=0.0)], ignore_index=True)
    if len(d) < 20 or pd.to_datetime(d["entry_date"]).dt.to_period("M").nunique() < 2:
        return {"event_coef": None, "t": None, "n": int(len(d)), "status": "NOT_COMPUTED: insufficient matched observations/month clusters"}
    X = pd.get_dummies(d[["event", "mom", "rev", "ldv", "beta", "sector"]], columns=["sector"], drop_first=True, dtype=float)
    X = sm.add_constant(X)
    g = pd.factorize(pd.DatetimeIndex(d["entry_date"]).to_period("M").astype(str))[0]
    res = sm.OLS(d["ar"].to_numpy(), X.to_numpy()).fit(cov_type="cluster", cov_kwds={"groups": g})
    k = list(X.columns).index("event")
    return {"event_coef": float(res.params[k]), "t": float(res.tvalues[k]), "n": int(len(d)),
            "covariates": "12-1 momentum, 1m reversal, log $ADV, beta, sector FE", "note": "earnings-revision control NOT_COMPUTED (no PIT estimates data)"}


def earnings_overlap(m, ev, h):
    f = C.PROC / "8k_items.parquet"
    if not f.exists():
        return None
    k8 = pd.read_parquet(f)
    k8 = k8[k8["items"].fillna("").str.contains(r"\b2\.02\b")]
    sess = {c: np.sort(m.cal.searchsorted(g["filingDate"].to_numpy(), side="left")) for c, g in k8.groupby("cik")}
    flag = []
    for cik, e in zip(ev["cik"], ev["e"]):
        s = sess.get(cik)
        flag.append(bool(s is not None and ((s >= e) & (s <= e + h)).any()))
    return np.array(flag)


# ------------------------------------------------------------- main
def run():
    m = Market()
    ins = load_insider(list(m.sym_of))
    b = buys(ins)
    R = {"prereg_id": REG["id"], "prereg_sha256": REG_HASH, "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "holdout_start": str(HOLDOUT.date()), "insider_coverage_through": str(ins["filing_date"].max().date()), "variants": {}}
    trial_srs, series = [], {}
    for kind, h in VARIANTS:
        ev = build_events(m, b, kind, h)
        disc, hold = ev[ev["sample"] == "discovery"], ev[ev["sample"] == "holdout"]
        port = daily_portfolio(m, disc, h)
        st = sharpe_stats(port)
        series[(kind, h)] = (ev, port)
        if st:
            trial_srs.append(st["sr"])
        R["variants"][f"{kind}_h{h}"] = {"events": int(len(ev)), "discovery_net": clustered_mean(disc["net"], disc["entry_date"]),
                                         "holdout_net": clustered_mean(hold["net"], hold["entry_date"]),
                                         "discovery_daily_sharpe_ann": st["sr"] * math.sqrt(252) if st else None,
                                         "discovery_days_invested": float((port != 0).mean()) if len(port) else None}
        v = R["variants"][f"{kind}_h{h}"]
        print(kind, h, len(ev), "disc", v["discovery_net"]["mean"], v["discovery_net"]["t"], "hold", v["holdout_net"]["mean"], v["holdout_net"]["t"], flush=True)

    # ---- primary: section 3.3 controls
    ev, port = series[("primary", 5)]
    h = 5
    ev = ev.copy()
    controls = matched_controls(m, ev, b, h, details=True)
    ev["ctrl_ar"] = controls["ar"]
    ev["diff"] = ev["ar"] - ev["ctrl_ar"]
    disc = ev[ev["sample"] == "discovery"]
    P = {"events": int(len(ev)), "discovery_n": int(len(disc)), "holdout_n": int((ev["sample"] == "holdout").sum())}
    P["discovery_gross"] = clustered_mean(disc["ar"], disc["entry_date"])
    P["discovery_net"] = clustered_mean(disc["net"], disc["entry_date"])
    P["holdout_net"] = clustered_mean(ev.loc[ev["sample"] == "holdout", "net"], ev.loc[ev["sample"] == "holdout", "entry_date"])
    P["discovery_control_car"] = clustered_mean(disc["ctrl_ar"], disc["entry_date"])
    P["discovery_event_minus_control"] = clustered_mean(disc["diff"], disc["entry_date"])
    P["incremental_regression_discovery"] = incremental_regression(disc, controls.reindex(disc.index))
    eo = earnings_overlap(m, ev, h)
    if eo is not None:
        ev["earn_in_window"] = eo
        d2 = ev[ev["sample"] == "discovery"]
        P["earnings_split_discovery"] = {"with_earnings_in_window": clustered_mean(d2.loc[d2["earn_in_window"], "net"], d2.loc[d2["earn_in_window"], "entry_date"]),
                                         "without_earnings_in_window": clustered_mean(d2.loc[~d2["earn_in_window"], "net"], d2.loc[~d2["earn_in_window"], "entry_date"])}
    else:
        P["earnings_split_discovery"] = "NOT_COMPUTED (8-K items not ingested)"
    ev["dv_tercile"] = pd.qcut(ev["ldv"].rank(method="first"), 3, labels=["low", "mid", "high"]).astype(str)
    P["subgroups_all_samples"] = {g: {str(k): clustered_mean(x["net"], x["entry_date"]) for k, x in ev.groupby(g) if len(x) >= 20}
                                  for g in ["sector", "dv_tercile", "regime", "ceo_cfo", "cluster"]}
    st = sharpe_stats(port)
    dsr, sr0 = deflated_sr(st, trial_srs)
    P["discovery_daily_portfolio"] = {"days": int(len(port)), "sharpe_ann": st["sr"] * math.sqrt(252) if st else None,
                                      "psr_vs_zero": psr(st["sr"], 0, st["n"], st["skew"], st["kurt"]) if st else None,
                                      "deflated_sharpe_ratio": dsr, "dsr_trials": len(trial_srs)}
    R["primary"] = P

    # ---- pre-registered gate
    checks = [
        ("discovery net mean > 0, t >= 3.0", (P["discovery_net"]["t"] or 0) >= 3.0 and (P["discovery_net"]["mean"] or 0) > 0),
        ("event minus matched control t >= 2.0", (P["discovery_event_minus_control"]["t"] or 0) >= 2.0),
        ("incremental regression event t >= 2.0", (P["incremental_regression_discovery"]["t"] or 0) >= 2.0),
        ("deflated Sharpe >= 0.90", dsr is not None and dsr >= 0.90),
        ("holdout net mean > 0", (P["holdout_net"]["mean"] or -1) > 0),
    ]
    failed = [name for name, ok in checks if not ok]
    status = "PAPER_TEST_ELIGIBLE" if not failed else ("RESEARCH_CANDIDATE" if (P["discovery_net"]["t"] or 0) >= 2 else "UNVALIDATED")
    R["gate"] = {"status": status, "checks": {n: bool(ok) for n, ok in checks}, "failed_checks": failed}

    # historical conditional distribution of the primary rule's net abnormal return (all events)
    x = ev["net"].dropna()
    dist = {"n": int(len(x)), "mean": float(x.mean()), "std": float(x.std()), "q025": float(x.quantile(0.025)),
            "q05": float(x.quantile(0.05)), "q975": float(x.quantile(0.975)), "p_pos_gross_excess": float((ev["ar"].dropna() > 0).mean()),
            "es05": float(x[x <= x.quantile(0.05)].mean())}
    R["primary_distribution_net"] = dist
    model = {"model_id": MODEL_ID, "model_version": MODEL_VERSION, "prereg_sha256": REG_HASH, "horizon": h, "cost": COST,
             "gate": R["gate"], "distribution_net": dist, "run_at_utc": R["run_at_utc"],
             "validation": {"oos_mean_net": P["holdout_net"]["mean"], "oos_se": P["holdout_net"]["se"],
                            "net_sharpe": P["discovery_daily_portfolio"]["sharpe_ann"], "dsr": dsr, "trials": len(trial_srs)}}
    (C.PROC / "insider_model.json").write_text(json.dumps(model, indent=1, default=float))
    (C.REPORTS / "insider_model_results.json").write_text(json.dumps(R, indent=1, default=str))
    ev.drop(columns=["j"]).to_parquet(C.REPORTS / "insider_model_events.parquet", index=False)
    with (C.LEDGER / "research_runs.jsonl").open("a") as f:
        f.write(json.dumps({"run_id": f"{MODEL_ID}-{R['run_at_utc']}", "prereg_sha256": REG_HASH, "gate": R["gate"]["status"],
                            "failed": failed, "discovery_net": P["discovery_net"]["mean"], "holdout_net": P["holdout_net"]["mean"]}) + "\n")
    print(json.dumps({k: P[k] for k in ["discovery_net", "holdout_net", "discovery_event_minus_control", "incremental_regression_discovery", "discovery_daily_portfolio"]}, indent=1, default=str))
    print("earnings split:", json.dumps(P["earnings_split_discovery"], default=str))
    print("GATE:", R["gate"])
    return R


# ------------------------------------------------------------- live signals
def live_signals(asof_day, gen, dataset_version, market_fresh, insider_cov):
    """Schema-v2 signals for primary-rule triggers filed on/after the last completed session (entry at the next open)."""
    model = json.loads((C.PROC / "insider_model.json").read_text())
    if model["prereg_sha256"] != REG_HASH:
        raise RuntimeError("pre-registration changed since the model was evaluated; register a new version")
    m = Market()
    ins = load_insider(list(m.sym_of))
    b = buys(ins)
    trig = triggers(b, "primary")
    i_last = len(m.cal) - 1
    new = trig[trig["fdate"] >= np.datetime64(asof_day)]
    # dedup against triggers that would already be in an open window
    out = []
    from signals import NY, iso, session_close_utc  # local import avoids a cycle at module load
    nxt = asof_day + pd.offsets.BDay(1)
    for cik, g in new.groupby("cik"):
        s = m.sym_of.get(int(cik))
        if s is None:
            continue
        j = m.col[s]
        prior = trig[(trig["cik"] == cik) & (trig["fdate"] < np.datetime64(asof_day))]["fdate"]
        recent_entry = len(prior) and (i_last + 1 - m.cal.searchsorted(prior.max(), side="right")) < DEDUP
        r = g.iloc[-1]
        liq = bool(m.ok_liquid(j, i_last))
        data_ok = market_fresh and insider_cov
        reasons = []
        gate = model["gate"]
        if model.get("model_version") != MODEL_VERSION:
            reasons.append("control methodology changed: model revalidation required")
        if gate["status"] not in ("PAPER_TEST_ELIGIBLE", "LIVE_REVIEW_ELIGIBLE"):
            reasons.append(f"model gate {gate['status']}: " + "; ".join(gate["failed_checks"]))
        if recent_entry:
            reasons.append(f"issuer already triggered within {DEDUP} sessions")
        if not liq:
            reasons.append("liquidity screen failed")
        if not data_ok:
            reasons.append("market or insider data stale")
        d = model["distribution_net"]
        net = d["mean"]
        if net <= 0:
            reasons.append("historical net expected return <= 0")
        long_ok = not reasons
        why = ("CEO/CFO open-market purchase" if r["ceo_cfo"] else "") + (" + " if r["ceo_cfo"] and r["cluster"] else "") + \
              ("cluster: >=2 insiders buying within 30d" if r["cluster"] else "")
        filings = ins[(ins["issuercik"] == cik) & (ins["trans_code"] == "P") & (ins["filing_date"] >= pd.Timestamp(asof_day))]
        drivers = [why, f"filed {pd.Timestamp(r['fdate']).date()}; ${filings['value'].sum():,.0f} purchased by {filings['rptownercik'].nunique()} insider(s)"]
        vol = None
        b_ = m.beta[i_last, j]
        sid = hashlib.sha256(f"{MODEL_ID}|{MODEL_VERSION}|{REG_HASH[:12]}|{cik}|{pd.Timestamp(r['fdate']).date()}".encode()).hexdigest()[:20]
        sic = pd.read_parquet(C.PROC / "sic.parquet").set_index("cik")
        sig = {
            "schema_version": "2.0", "model_id": MODEL_ID, "model_version": MODEL_VERSION, "signal_id": sid,
            "generated_at_utc": iso(gen), "as_of_utc": iso(session_close_utc(asof_day)),
            "asset": {"ticker": s.replace("-", "."), "security_id": f"CIK{int(cik):010d}",
                      "exchange": str(sic["exchanges"].get(int(cik))) if int(cik) in sic.index else None, "asset_class": "Equity"},
            "forecast": {"horizon_sessions": model["horizon"], "expected_excess_return": d["mean"] + COST, "expected_net_excess_return": d["mean"],
                         "predictive_std": d["std"], "prediction_interval_95": [d["q025"] + COST, d["q975"] + COST],
                         "probability_positive_excess_return": d["p_pos_gross_excess"], "expected_cost_return": COST},
            "validation": {"status": gate["status"], "out_of_sample_ic": None, "ic_uncertainty": None,
                           "net_sharpe": model["validation"]["net_sharpe"], "deflated_sharpe_ratio": model["validation"]["dsr"],
                           "tested_hypotheses": model["validation"]["trials"]},
            "risk": {"forecast_volatility": vol, "estimated_beta": None if np.isnan(b_) else float(b_), "expected_shortfall": d["es05"] + COST,
                     "liquidity_approved": liq, "data_quality_approved": bool(data_ok)},
            "signal": {"direction": "LONG" if long_ok else "NO_SIGNAL", "deployment_eligibility": gate["status"] if data_ok else "INSUFFICIENT_DATA",
                       "expires_at_utc": iso(session_close_utc(nxt)), "reason": None if long_ok else "; ".join(reasons),
                       "key_drivers": drivers, "contradictory_evidence": []},
            "provenance": {"feature_set_version": f"prereg-{REG_HASH[:12]}", "dataset_version": dataset_version,
                           "source_ids": ["sec_form345_datasets", "sec_edgar_form4_xml", "yahoo_chart_v8_daily"],
                           "latest_required_data_utc": iso(session_close_utc(asof_day)), "research_run_id": f"{MODEL_ID}-{model['run_at_utc']}"},
        }
        r60 = m.P["adjclose"][s].pct_change(fill_method=None).iloc[-60:].std()
        if pd.notna(r60):
            sig["risk"]["forecast_volatility"] = float(r60 * math.sqrt(model["horizon"]))
        out.append(sig)
    return out


if __name__ == "__main__":
    run()
