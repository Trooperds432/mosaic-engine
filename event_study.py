"""Insider purchase event study (spec 3.3).

Events (all from filings, keyed on filing date, never transaction date):
  cluster_buy   >= 2 distinct insiders with open-market purchases (code P) filed within 30 days;
                the event date is the filing that completes the cluster
  ceo_cfo_buy   CEO/CFO open-market purchase
  any_buy       any open-market purchase
  discr_sell    open-market sale not flagged as a 10b5-1 plan trade (control comparison)
Entry is the open of the first session strictly after the filing date (Form 4s are often filed
after the close). Abnormal return = stock return - beta * SPY return, beta from the 252 sessions
before the filing. One event per issuer per 60 sessions per type so windows do not overlap within
a firm; standard errors are clustered by calendar month to absorb cross-firm event clustering.
Results are split at the research holdout boundary so the later sample is a genuine check.
"""
import json
import math

import numpy as np
import pandas as pd
import statsmodels.api as sm

import config as C
from features import load_insider, load_prices

WINDOWS = [1, 5, 20, 60]


def events(ins, kind):
    d = ins.sort_values("filing_date")
    if kind == "cluster_buy":
        b = d[d["trans_code"] == "P"].copy()
        b["owner"] = b["rptownercik"].fillna(b["rptownername"])
        out = []
        for cik, g in b.groupby("issuercik"):
            fd, ow = g["filing_date"].to_numpy(), g["owner"].to_numpy()
            for i in range(len(g)):
                lo = np.searchsorted(fd, fd[i] - np.timedelta64(30, "D"))
                if len(set(ow[lo:i + 1])) >= 2:
                    out.append((cik, fd[i]))
        return pd.DataFrame(out, columns=["cik", "fdate"])
    if kind == "ceo_cfo_buy":
        m = (d["trans_code"] == "P") & d["is_ceo_cfo"]
    elif kind == "any_buy":
        m = d["trans_code"] == "P"
    else:
        m = (d["trans_code"] == "S") & ~d["is_10b5_1"]
    return d.loc[m, ["issuercik", "filing_date"]].rename(columns={"issuercik": "cik", "filing_date": "fdate"})


def car_table(ev, P, sym_of, cal):
    op = P["adj_open"]
    spy_op = P["spy_open"]
    r = P["adjclose"].pct_change(fill_method=None)
    rs = P["spy_adjclose"].pct_change(fill_method=None)
    rows, last = [], {}
    ev = ev.sort_values("fdate")
    for cik, fdate in zip(ev["cik"], ev["fdate"]):
        sym = sym_of.get(int(cik))
        if sym is None:
            continue
        e = cal.searchsorted(pd.Timestamp(fdate), side="right")  # first session after filing date
        if e < 253 or e >= len(cal) - 1:
            continue
        if cik in last and e - last[cik] < 60:
            continue
        rr, mm = r[sym].iloc[e - 252:e], rs.iloc[e - 252:e]
        ok = rr.notna() & mm.notna()
        if ok.sum() < 200 or pd.isna(op[sym].iat[e]):
            continue
        beta = float(np.cov(rr[ok], mm[ok])[0, 1] / mm[ok].var())
        rec = {"cik": int(cik), "symbol": sym, "event_date": cal[e], "beta": beta}
        for w in WINDOWS:
            x = e + w
            if x >= len(cal) or pd.isna(op[sym].iat[x]):
                rec[f"car_{w}"] = np.nan
                continue
            rec[f"car_{w}"] = (op[sym].iat[x] / op[sym].iat[e] - 1) - beta * (spy_op.iat[x] / spy_op.iat[e] - 1)
        rows.append(rec)
        last[cik] = e
    return pd.DataFrame(rows)


def summarize(df, w):
    s = df[["event_date", f"car_{w}"]].dropna()
    if len(s) < 20:
        return {"n": int(len(s)), "mean": None, "se": None, "t": None, "ci95": None}
    s = s.copy()
    s["win"] = s[f"car_{w}"].clip(*s[f"car_{w}"].quantile([0.005, 0.995]))
    groups = pd.DatetimeIndex(s["event_date"]).to_period("M").astype(str)
    res = sm.OLS(s["win"].to_numpy(), np.ones(len(s))).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(groups)[0]})
    m, se = float(res.params[0]), float(res.bse[0])
    return {"n": int(len(s)), "mean": m, "se": se, "t": m / se if se else None, "ci95": [m - 1.96 * se, m + 1.96 * se],
            "median": float(s[f"car_{w}"].median()), "hit_rate": float((s[f"car_{w}"] > 0).mean())}


def run():
    P, sym2cik = load_prices()
    cal = P["adjclose"].index
    sym_of = {c: s for s, c in sym2cik.items()}
    ins = load_insider(list(sym2cik.values()))
    holdout = cal[-1] - pd.DateOffset(years=C.HOLDOUT_YEARS)
    out = {"method": __doc__.strip(), "holdout_start": str(holdout.date()), "results": {}}
    for kind in ["cluster_buy", "ceo_cfo_buy", "any_buy", "discr_sell"]:
        df = car_table(events(ins, kind), P, sym_of, cal)
        res = {}
        for label, sub in [("discovery", df[df["event_date"] < holdout]), ("holdout", df[df["event_date"] >= holdout])]:
            res[label] = {f"{w}d": summarize(sub, w) for w in WINDOWS}
        out["results"][kind] = res
        print(kind, {k: {w: (v[w]["n"], v[w]["mean"] and round(v[w]["mean"], 4), v[w]["t"] and round(v[w]["t"], 2))
                         for w in v} for k, v in res.items()}, flush=True)
    (C.REPORTS / "insider_event_study.json").write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    _lock = __import__("config").hold_lock()
    run()
