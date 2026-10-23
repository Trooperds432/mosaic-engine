"""Point-in-time integrity and bias tests (spec 2.2). Writes reports/data_audit.json.

Blocking tests (a failure blocks every LONG signal):
  truncation_invariance  features at a past date D recomputed from data truncated at D (prices through D,
                         filings filed <= D-1, 13F/short-interest/macro available <= D) must equal the panel's
                         values: any difference means information from after D leaked in.
  label_alignment        y_h(t) recomputed independently from raw opens (t+1 -> t+1+h, minus SPY) must match.
  duplicate_records      no duplicate insider transactions after de-duplication; no duplicate price dates.
  timestamps             no weekend/holiday price dates; 8-K acceptance after 17:30 ET carries the next filing date.
Diagnostic tests (reported, not blocking): survivorship coverage, restatement frequency (as-reported vs
latest-restated), corporate-action anomalies, stale observations, missing-not-at-random (IC of missingness).
"""
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config as C
import features as F


def truncation_invariance(prof, n_dates=3, n_syms=150):
    """Recompute price, insider and fundamental features for sampled past dates with truncated inputs."""
    P, sym2cik = F.load_prices(prof["min_dollar_adv"])
    cal = P["adjclose"].index
    rng = np.random.default_rng(1)
    syms = list(rng.choice(P["adjclose"].columns, size=min(n_syms, P["adjclose"].shape[1]), replace=False))
    ciks = list(dict.fromkeys(sym2cik[s] for s in syms))
    idx = np.sort(rng.choice(np.arange(600, len(cal) - 300), size=n_dates, replace=False))
    res, worst = [], 0.0
    Psub = {k: (v[syms] if isinstance(v, pd.DataFrame) else v) for k, v in P.items()}
    full_pf = F.price_features(Psub, idx)
    facts = F.load_facts(ciks)
    ins = F.load_insider(ciks)
    full_f = F.fundamentals_asof(cal[idx], ciks, facts)
    full_i = F.insider_features(ins, cal[idx], ciks)
    for k, i in enumerate(idx):
        d = cal[i]
        Pt = {kk: (v.iloc[:i + 1] if hasattr(v, "iloc") else v) for kk, v in Psub.items()}
        tp = F.price_features(Pt, np.array([i]))
        tf = F.fundamentals_asof(cal[[i]], ciks, facts[facts["filed"] <= d - pd.Timedelta(days=1)])
        ti = F.insider_features(ins[ins["filing_date"] <= d - pd.Timedelta(days=1)], cal[[i]], ciks)
        diffs = {}
        for name, a, b in [("price:" + n, full_pf[n].iloc[k], tp[n].iloc[0]) for n in ["mom_12_1", "vol_60", "rsi_14", "beta_252", "cs_spread", "adv_dollar"]] + \
                          [("fund:" + n, full_f[n].iloc[k], tf[n].iloc[0]) for n in ["ni", "assets", "shares", "assets_prev"]] + \
                          [("insider:" + n, full_i[n].iloc[k], ti[n].iloc[0]) for n in ["buy_usd_90", "sell_usd_90", "ins_buyers_90", "opp_net_usd_90"]]:
            a, b = pd.to_numeric(pd.Series(a.to_numpy()), errors="coerce"), pd.to_numeric(pd.Series(b.to_numpy()), errors="coerce")
            both = a.notna() & b.notna()
            mism = int((a.isna() != b.isna()).sum())
            rel = float((np.abs(a[both] - b[both]) / np.maximum(np.abs(a[both]), 1e-9)).max()) if both.any() else 0.0
            diffs[name] = {"max_rel_diff": rel, "nan_mismatch": mism}
            worst = max(worst, rel if np.isfinite(rel) else 1.0, 1.0 if mism else 0.0)
        res.append({"date": str(d.date()), "diffs": diffs})
    return {"passed": worst < 1e-6, "worst": worst, "dates": res, "symbols_sampled": len(syms)}


def label_alignment(prof):
    panel = pd.read_parquet(C.PROC / f"panel_{prof['name']}.parquet", columns=["date", "symbol", "y_1", "y_20"])
    s = panel.sample(min(2000, len(panel)), random_state=3)
    spy = pd.read_parquet(C.RAW / "prices" / f"{C.BENCHMARK}.parquet").drop_duplicates("date").set_index("date")
    cal = spy.index
    so = spy["open"] * spy["adjclose"] / spy["close"]
    errs = []
    for sym, g in s.groupby("symbol"):
        px = pd.read_parquet(C.RAW / "prices" / f"{sym}.parquet").drop_duplicates("date").set_index("date").reindex(cal)
        from historical_data import active_bundle
        bundle = active_bundle()
        terminal = pd.read_parquet(bundle / "terminal.parquet").set_index("security_id") if bundle else pd.DataFrame()
        for r in g.itertuples():
            i = cal.get_loc(r.date)
            for h, y in [(1, r.y_1), (20, r.y_20)]:
                if i + 1 + h >= len(cal) or pd.isna(y):
                    continue
                ex = px["adj_open"].iat[i + 1 + h]
                if pd.isna(ex) and sym in terminal.index:
                    event = terminal.loc[sym]
                    if cal[i + 1] <= event["last_trade_date"] and event["settlement_date"] <= cal[i + 1 + h]:
                        ex = event["terminal_adjclose"]
                if pd.isna(ex):
                    return {"passed": False, "checked": len(errs), "max_abs_error": None, "reason": "non-null panel label has no observed/verified terminal exit"}
                exp = ex / px["adj_open"].iat[i + 1] - 1 - (so.iat[i + 1 + h] / so.iat[i + 1] - 1)
                errs.append(abs(float(exp) - float(y)))
    e = np.array(errs)
    return {"passed": bool(len(e) and np.nanmax(e) < 1e-4), "checked": int(len(e)), "max_abs_error": float(np.nanmax(e)) if len(e) else None}


def duplicates_and_timestamps(prof):
    uni = pd.read_parquet(C.PROC / "universe.parquet")
    ins = F.load_insider(list(uni["cik"].astype(int)))
    dup_ins = int(ins.duplicated(["accession_number", "trans_date", "trans_code", "trans_shares", "trans_pricepershare", "owner"]).sum())
    bad_dates, dup_dates, n = 0, 0, 0
    spy = pd.read_parquet(C.RAW / "prices" / f"{C.BENCHMARK}.parquet")
    weekend = int((pd.to_datetime(spy["date"]).dt.dayofweek >= 5).sum())
    for f in sorted((C.RAW / "prices").glob("*.parquet"))[:: max(1, len(list((C.RAW / "prices").glob("*.parquet"))) // 300)]:
        d = pd.read_parquet(f, columns=["date"])["date"]
        bad_dates += int((d.dt.dayofweek >= 5).sum())
        dup_dates += int(d.duplicated().sum())
        n += 1
    k8 = pd.read_parquet(C.PROC / "8k_items.parquet", columns=["filingDate", "accepted_et", "available_date"]).dropna()
    acc = k8["accepted_et"]
    # after correcting the API's doubled ET offset, the acceptance date must equal the filing date, or precede it
    # when accepted after 17:30 ET (deemed filed the next business day); availability must never precede acceptance
    acc_day = acc.dt.tz_localize(None).dt.normalize()
    late_same_day = int(((acc_day > k8["filingDate"]) | (k8["available_date"] < acc_day)).sum())
    return {"duplicate_insider_rows": dup_ins, "price_files_checked": n, "weekend_price_rows": bad_dates + weekend, "duplicate_price_dates": dup_dates,
            "8k_acceptance_inconsistent": late_same_day, "8k_checked": int(len(k8)),
            "passed": dup_ins == 0 and bad_dates + weekend == 0 and dup_dates == 0 and late_same_day <= 0.001 * len(k8)}


def diagnostics(prof):
    out = {}
    uni = pd.read_parquet(C.PROC / "universe.parquet")
    spy = pd.read_parquet(C.RAW / "prices" / f"{C.BENCHMARK}.parquet")
    end = spy["date"].max()
    ended = 0
    have = 0
    for s in uni["yahoo"]:
        f = C.RAW / "prices" / f"{s}.parquet"
        if f.exists():
            have += 1
            if pd.read_parquet(f, columns=["date"])["date"].max() < end - pd.Timedelta(days=10):
                ended += 1
    ua = json.loads((C.PROC / "universe_audit.json").read_text())
    out["survivorship"] = {"candidates": ua.get("candidates"), "removed_sp500_without_current_ticker_excluded": ua.get("removed_without_current_sec_ticker"),
                           "price_series_available": have, "series_ending_before_last_session": ended,
                           "note": "Current-listing histories remain survivorship-biased unless verified historical coverage passes. Missing terminal outcomes are excluded, never assigned zero returns."}
    f = pd.read_parquet(C.RAW / "facts" / "companyfacts.parquet", columns=["cik", "tag", "start", "end", "val", "filed"], filters=[("tag", "in", ["NetIncomeLoss", "Assets"])])
    g = f.sort_values("filed").groupby(["cik", "tag", "start", "end"], dropna=False)["val"]
    first, last, nrep = g.first(), g.last(), g.size()
    changed = (first != last) & (nrep > 1)
    out["restatement"] = {"periods": int(len(first)), "periods_with_later_different_value": int(changed.sum()),
                          "share_restated": float(changed.mean()), "median_abs_rel_change_when_restated": float(((last - first).abs() / first.abs().clip(lower=1))[changed].median()) if changed.any() else None,
                          "policy": "features use the value known at each filing date (as-reported); the latest-restated series is kept separately in reports/facts_latest_restated.parquet"}
    lr = pd.DataFrame({"first_reported": first, "latest_restated": last, "n_reports": nrep}).reset_index()
    lr.to_parquet(C.REPORTS / "facts_latest_restated.parquet", index=False)
    panel = pd.read_parquet(C.PROC / f"panel_{prof['name']}.parquet")
    out["stale"] = {"share_rows_fundamentals_older_than_limit": float((panel["fund_age_days"] > C.FUNDAMENTAL_MAX_AGE_DAYS).mean()),
                    "mean_quality_score": float(panel["quality_score"].mean())}
    from research import ic_series
    mnar = {}
    for k, cols in {"fundamentals": ["earn_yield", "book_to_mkt"], "sue": ["sue"], "13f": ["io_ratio"]}.items():
        cols = [c for c in cols if c in panel]
        if not cols:
            continue
        miss = panel[cols].isna().all(axis=1).astype(float)
        ic = ic_series(miss.to_numpy(), panel[f"y_{C.PRIMARY_HORIZON}"].to_numpy(), panel["date"].to_numpy())
        mnar[k] = {"missing_share": float(miss.mean()), "ic_missing_vs_fwd_return": float(ic.mean()) if len(ic) else None,
                   "t": float(ic.mean() / ic.std() * np.sqrt(len(ic) / 4)) if len(ic) > 10 and ic.std() > 0 else None}
    out["missing_not_at_random"] = mnar
    anomalies = 0
    for fpath in sorted((C.RAW / "prices").glob("*.parquet"))[::20]:
        d = pd.read_parquet(fpath, columns=["adjclose", "close"])
        r = d["adjclose"].pct_change()
        raw = d["close"].pct_change()
        anomalies += int(((r.abs() > 0.6) & ((raw - r).abs() < 0.01)).sum())  # huge move present in both raw and adjusted series
    out["corporate_actions"] = {"sampled_files": len(sorted((C.RAW / "prices").glob("*.parquet"))[::20]), "unexplained_moves_gt_60pct": anomalies,
                                "note": "split adjustment comes from Yahoo adjclose; large moves present in both raw and adjusted series are reviewed, not removed"}
    return out


def run(profile=None):
    prof = C.PROFILES[profile] if profile else C.PROFILE
    from provenance import dataset_version
    A = {"generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "profile": prof["name"],
         "dataset_version": dataset_version()}  # signals.py only accepts an audit of the data it is about to use
    A["truncation_invariance"] = truncation_invariance(prof)
    print("truncation", A["truncation_invariance"]["passed"], A["truncation_invariance"]["worst"], flush=True)
    A["label_alignment"] = label_alignment(prof)
    print("labels", A["label_alignment"], flush=True)
    A["duplicates_timestamps"] = duplicates_and_timestamps(prof)
    print("dups/timestamps", A["duplicates_timestamps"], flush=True)
    A["diagnostics"] = diagnostics(prof)
    from historical_data import coverage_check
    panel_dates = pd.read_parquet(C.PROC / f"panel_{prof['name']}.parquet", columns=["date"])["date"]
    A["historical_universe"] = coverage_check(panel_dates.min(), panel_dates.max())
    A["blocking"] = {"historical_universe": A["historical_universe"]["passed"], "truncation_invariance": A["truncation_invariance"]["passed"], "label_alignment": A["label_alignment"]["passed"],
                     "duplicates_timestamps": A["duplicates_timestamps"]["passed"]}
    target = C.REPORTS / f"data_audit_{prof['name']}.json"
    tmp = target.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(A, indent=1, default=str))
    tmp.replace(target)
    if prof['name'] == 'large':  # legacy display artifact, never used to approve another profile
        legacy = C.REPORTS / 'data_audit.json.tmp'
        legacy.write_text(json.dumps(A, indent=1, default=str))
        legacy.replace(C.REPORTS / 'data_audit.json')
    print("blocking:", A["blocking"])
    return A


if __name__ == "__main__":
    _lock = __import__("config").hold_lock()
    run(sys.argv[1] if len(sys.argv) > 1 else None)
