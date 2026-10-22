"""Forward 90-day catalyst calendar with historical conditional responses (spec section 11).

Sources (each event carries its source; nothing is inferred without one):
  earnings          Finnhub earnings calendar (date, before/after market, consensus EPS/revenue)
  fda_adcomm        Finnhub FDA advisory-committee calendar (FDA meeting announcements)
  fomc              federalreserve.gov FOMC calendar page
  bea               bea.gov release schedule (GDP, personal income/PCE, trade)
  eia_wpsr          EIA Weekly Petroleum Status Report: Wednesdays 10:30 ET by rule (holiday shifts NOT verified)
  trial_completion  ClinicalTrials.gov primary-completion dates of phase 2/3 trials (sponsor estimates, not readouts)
NOT_COMPUTED (no verifiable free source reachable from this host): BLS CPI / employment dates (bls.gov blocks
scripted access), OPEC+ meetings, USDA WASDE (usda.gov blocks scripted access), shareholder votes.

Historical conditional responses (computed, not assumed):
  earnings  per stock and pooled: abnormal (beta-adjusted) close-to-close return over the reaction day
            (|move| distribution), and post-announcement drift days 2-20 by sign of the reaction (decay test)
  fomc      SPY absolute return on FOMC decision days vs all other days
  eia       energy-stock absolute abnormal return on Wednesdays vs other weekdays
  fda       pharma/biotech abnormal return on openFDA approval dates (positive outcomes only: rejections are
            not in the open data, so the downside scenario is NOT_COMPUTED)
"""
import html
import json
import os
import re
from datetime import date, datetime, timedelta, timezone

import httpx
import numpy as np
import pandas as pd

import config as C
from features import load_prices, sector_flags

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/130 Safari/537.36"}
MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August", "September",
                                      "October", "November", "December"], 1)}


def _key(name):
    v = os.environ.get(name)
    if v:
        return v
    for line in open("/root/.hermes/.env"):
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip()
    return None


def fomc_dates(client):
    t = client.get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm", headers=UA).text
    out = []
    for m in re.finditer(r"(\d{4}) FOMC Meetings(.*?)(?=\d{4} FOMC Meetings|$)", t, re.S):
        yr, body = int(m.group(1)), m.group(2)
        for mon, days in re.findall(r'fomc-meeting__month[^>]*><strong>([A-Za-z/]+)</strong>.*?fomc-meeting__date[^>]*>([^<]+)<', body, re.S):
            mon = mon.split("/")[-1]
            last = re.findall(r"\d+", days)
            if mon in MONTHS and last:
                d = date(yr, MONTHS[mon], int(last[-1]))  # decision is announced on the meeting's last day
                out.append({"date": d, "event": "FOMC rate decision", "tentative": "*" in days})
    return pd.DataFrame(out).drop_duplicates("date").sort_values("date")


def bea_releases(client):
    t = client.get("https://www.bea.gov/news/schedule", headers=UA).text
    out, year = [], date.today().year
    for r in re.findall(r"<tr[^>]*>(.*?)</tr>", t, re.S):
        cells = [c.strip() for c in html.unescape(re.sub(r"<[^>]+>", "|", r)).split("|") if c.strip()]
        y = [c for c in cells if re.fullmatch(r"Year \d{4}", c)]
        if y:
            year = int(y[0][5:])
            continue
        md = [c for c in cells if re.fullmatch(r"[A-Z][a-z]+ \d{1,2}", c)]
        if md and cells:
            mon, dd = md[0].split()
            if mon in MONTHS:
                out.append({"date": date(year, MONTHS[mon], int(dd)), "event": cells[-1], "time": next((c for c in cells if "AM" in c or "PM" in c), None)})
    return pd.DataFrame(out)


def historical_responses(P, sym2cik):
    cal = P["adjclose"].index
    r = P["adjclose"].pct_change(fill_method=None)
    rs = P["spy_adjclose"].pct_change(fill_method=None)
    beta = r.rolling(252, min_periods=120).cov(rs).div(rs.rolling(252, min_periods=120).var(), axis=0).shift(1)
    ar = r - beta.mul(rs, axis=0)
    out = {}
    # earnings: reaction day = filing date of the item 2.02 8-K if accepted before 16:00 ET, else next session
    k8 = pd.read_parquet(C.PROC / "8k_items.parquet")
    k8 = k8[k8["items"].fillna("").str.contains(r"\b2\.02\b")].dropna(subset=["available_date"])
    k8["react"] = k8["available_date"]  # corrected acceptance time: same session if before 16:00 ET, else next
    col = {c: s for s, c in sym2cik.items()}
    rows = []
    arv = ar.to_numpy()
    for cik, rd in zip(k8["cik"], k8["react"]):
        s = col.get(cik)
        if s is None:
            continue
        i = cal.searchsorted(rd)
        if i >= len(cal) - 21 or i < 1:
            continue
        j = ar.columns.get_loc(s)
        a0 = arv[i, j]
        if np.isnan(a0):
            continue
        rows.append({"symbol": s, "date": cal[i], "react": a0, "drift_2_20": np.nansum(arv[i + 1:i + 20, j])})
    e = pd.DataFrame(rows)
    by = e.groupby("symbol")["react"].agg(lambda x: float(x.abs().median()))
    pos, neg = e[e["react"] > 0], e[e["react"] < 0]
    out["earnings"] = {"events": int(len(e)), "pooled_abs_reaction_median": float(e["react"].abs().median()), "pooled_abs_reaction_p90": float(e["react"].abs().quantile(0.9)),
                       "drift_2_20_after_positive_reaction": float(pos["drift_2_20"].mean()), "drift_2_20_after_negative_reaction": float(neg["drift_2_20"].mean()),
                       "drift_t_positive": float(pos["drift_2_20"].mean() / pos["drift_2_20"].std() * np.sqrt(len(pos))),
                       "drift_t_negative": float(neg["drift_2_20"].mean() / neg["drift_2_20"].std() * np.sqrt(len(neg))),
                       "note": "drift t-stats ignore cross-event correlation (overstated); decay = drift relative to the reaction"}
    out["_earn_by_symbol"] = by.to_dict()
    return out, ar, rs, cal


def run():
    gen = datetime.now(timezone.utc)
    today = pd.Timestamp(date.today())
    end = today + pd.Timedelta(days=90)
    uni = pd.read_parquet(C.PROC / "universe.parquet")
    P, sym2cik = load_prices(C.PROFILES["broad"]["min_dollar_adv"])
    for k in ["high", "low", "volume", "adj_open", "close"]:
        P.pop(k, None)
    hist, ar, rs, cal = historical_responses(P, sym2cik)
    tick = set(P["adjclose"].columns)
    events, notes = [], {}
    fh = _key("FINNHUB_API_KEY")
    with httpx.Client(timeout=60) as c:
        if fh:
            j = c.get("https://finnhub.io/api/v1/calendar/earnings", params={"from": str(today.date()), "to": str(end.date()), "token": fh}).json()
            for e in j.get("earningsCalendar", []):
                s = e["symbol"].replace(".", "-")
                if s in tick:
                    events.append({"date": e["date"], "type": "earnings", "symbol": s, "detail": f"Q{e.get('quarter')} {e.get('year')} ({e.get('hour') or 'time n/a'})",
                                   "consensus_eps": e.get("epsEstimate"), "consensus_revenue": e.get("revenueEstimate"), "source": "finnhub_earnings_calendar",
                                   "historical_abs_reaction_median": hist["_earn_by_symbol"].get(s)})
            fda = c.get(f"https://finnhub.io/api/v1/fda-advisory-committee-calendar?token={fh}").json()
            for e in fda if isinstance(fda, list) else []:
                d = pd.Timestamp(e["fromDate"][:10])
                if today <= d <= end:
                    events.append({"date": str(d.date()), "type": "fda_adcomm", "symbol": None, "detail": e["eventDescription"][:200], "source": "finnhub_fda_calendar", "url": e.get("url")})
        else:
            notes["earnings"] = "NOT_COMPUTED: FINNHUB_API_KEY missing"
        fo = fomc_dates(c)
        for r in fo.itertuples():
            if today <= pd.Timestamp(r.date) <= end:
                events.append({"date": str(r.date), "type": "fomc", "symbol": None, "detail": r.event + (" (tentative)" if r.tentative else ""), "source": "federalreserve.gov"})
        try:
            for r in bea_releases(c).itertuples():
                if today <= pd.Timestamp(r.date) <= end:
                    events.append({"date": str(r.date), "type": "macro_release", "symbol": None, "detail": f"{r.event} {r.time or ''}".strip(), "source": "bea.gov"})
        except Exception as ex:
            notes["bea"] = f"NOT_COMPUTED: {ex!r}"[:200]
    for d in pd.date_range(today, end, freq="W-WED"):
        events.append({"date": str(d.date()), "type": "eia_wpsr", "symbol": None, "detail": "EIA Weekly Petroleum Status Report 10:30 ET (rule-based; holiday shifts not verified)", "source": "eia_rule"})
    tr = C.PROC / "clinical_trials.parquet"
    if tr.exists():
        t = pd.read_parquet(tr)
        t = t[t["phase"].fillna("").str.contains("PHASE2|PHASE3") & t["primary_completion"].between(today, end) & t["status"].isin(["ACTIVE_NOT_RECRUITING", "RECRUITING", "ENROLLING_BY_INVITATION"])]
        c2s = {c: s for s, c in sym2cik.items()}
        for r in t.itertuples():
            s = c2s.get(r.cik)
            if s:
                events.append({"date": str(r.primary_completion.date()), "type": "trial_primary_completion", "symbol": s, "detail": f"{r.nct} {r.phase} {r.conditions[:80]}",
                               "source": "clinicaltrials.gov (sponsor-estimated completion; readout timing unknown)"})
    notes.update({"cpi_employment": "NOT_COMPUTED: bls.gov blocks scripted access; dates not verified", "opec": "NOT_COMPUTED: no machine-readable OPEC+ schedule",
                  "wasde": "NOT_COMPUTED: usda.gov blocks scripted access", "shareholder_votes": "NOT_COMPUTED: DEF 14A meeting dates not parsed"})

    # historical conditional responses for macro catalysts
    absr = rs.abs()
    fo_hist = [pd.Timestamp(d) for d in fo["date"] if pd.Timestamp(d) < today]
    fmask = absr.index.isin(fo_hist)
    hist["fomc"] = {"events": int(fmask.sum()), "spy_abs_return_fomc_days": float(absr[fmask].mean()), "spy_abs_return_other_days": float(absr[~fmask & absr.notna()].mean()),
                    "ratio": float(absr[fmask].mean() / absr[~fmask & absr.notna()].mean())}
    sic = pd.read_parquet(C.PROC / "sic.parquet").set_index("cik")["sic"]
    energy = [s for s, c in sym2cik.items() if sector_flags(sic.get(c))["energy"]]
    if energy:
        ea = ar[energy].abs().mean(axis=1).iloc[-1260:]
        wed = ea.index.dayofweek == 2
        hist["eia_wednesday"] = {"energy_names": len(energy), "abs_abnormal_wed": float(ea[wed].mean()), "abs_abnormal_other": float(ea[~wed].mean())}
    fa = C.PROC / "fda_approvals.parquet"
    if fa.exists():
        a = pd.read_parquet(fa)
        a = a[a["class"].isin(["TYPE 1", "TYPE 2", "TYPE 3", "TYPE 4", "TYPE 5"]) | (a["type"] == "ORIG")]
        rows = []
        c2s = {c: s for s, c in sym2cik.items()}
        for r in a.itertuples():
            s = c2s.get(r.cik)
            if s is None or pd.isna(r.action_date):
                continue
            i = cal.searchsorted(r.action_date)
            if 1 <= i < len(cal) - 2:
                rows.append(ar[s].iloc[i:i + 2].sum())
        x = pd.Series(rows).dropna()
        hist["fda_approval"] = {"events": int(len(x)), "mean_abnormal_2d": float(x.mean()) if len(x) else None, "median": float(x.median()) if len(x) else None,
                                "p10": float(x.quantile(0.1)) if len(x) else None, "p90": float(x.quantile(0.9)) if len(x) else None,
                                "downside_scenario": "NOT_COMPUTED: complete-response letters / rejections not in openFDA history"}
    hist.pop("_earn_by_symbol", None)
    ev = pd.DataFrame(events).sort_values(["date", "type"]) if events else pd.DataFrame()
    doc = {"generated_at_utc": gen.isoformat(timespec="seconds"), "window": [str(today.date()), str(end.date())], "n_events": int(len(ev)),
           "by_type": ev["type"].value_counts().to_dict() if len(ev) else {}, "not_computed": notes, "historical_responses": hist,
           "events": json.loads(ev.to_json(orient="records")) if len(ev) else []}
    (C.REPORTS / "catalysts.json").write_text(json.dumps(doc, indent=1, default=str))
    print(json.dumps({k: doc[k] for k in ["n_events", "by_type", "historical_responses"]}, indent=1, default=str))
    return doc


if __name__ == "__main__":
    _lock = __import__("config").hold_lock()
    run()
