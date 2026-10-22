"""Point-in-time feature panel.

Decision time t = close of session t. Price features use data through close t. Filings are
eligible only if filed on or before session t-1 (one-session ingestion latency, since Form 4s and
10-Ks are routinely filed after the close). Labels: entry at session t+1 open, exit at session
t+1+h open, excess vs SPY over the identical window.
"""
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config import (ADV_WINDOW, BENCHMARK, DECISION_STEP, HORIZONS, MIN_ADV_SHARES, MIN_MCAP_USD, PROC, RAW)

# Feature registry: name -> (family, hypothesised sign of relation to forward excess return, description)
FEATURES = {
    "mom_12_1": ("momentum", +1, "12-1 month total return (Jegadeesh-Titman)"),
    "mom_6_1": ("momentum", +1, "6-1 month total return"),
    "ret_1m": ("reversal", -1, "last 21-session return (short-term reversal)"),
    "dist_52w_high": ("momentum", +1, "price / 252-session high (George-Hwang)"),
    "px_sma200": ("technical", +1, "price / 200-session SMA - 1"),
    "vol_60": ("risk", -1, "60-session realised volatility (low-vol anomaly)"),
    "beta_252": ("risk", -1, "252-session beta vs SPY (betting-against-beta)"),
    "log_mcap": ("size", -1, "log market cap"),
    "amihud": ("liquidity", +1, "Amihud illiquidity, 60 sessions"),
    "earn_yield": ("value", +1, "annual net income / market cap"),
    "book_to_mkt": ("value", +1, "stockholders' equity / market cap"),
    "fcf_yield": ("value", +1, "(operating cash flow - capex) / market cap"),
    "gross_prof": ("quality", +1, "gross profit / assets (Novy-Marx)"),
    "roa": ("quality", +1, "net income / assets"),
    "accruals": ("quality", -1, "(net income - operating cash flow) / assets (Sloan)"),
    "asset_growth": ("investment", -1, "annual asset growth (Cooper-Gulen-Schill)"),
    "ins_net_90": ("insider", +1, "(open-mkt buy $ - non-10b5-1 sell $) / mcap, filings in last 90d"),
    "ins_buyers_90": ("insider", +1, "distinct insiders with open-market purchases, last 90d"),
    "ins_sellers_90": ("insider", -1, "distinct insiders with discretionary open-market sales, last 90d"),
    "ins_officer_buy_180": ("insider", +1, "CEO/CFO open-market purchase in last 180d (0/1)"),
    "ins_surprise": ("insider", +1, "90d net insider $ minus 90d-scaled trailing-365d net, / mcap"),
    "ins_buy_x_drawdown": ("insider", +1, "distinct buyers 90d x drawdown from 252d high"),
}
FUND_FEATURES = ["earn_yield", "book_to_mkt", "fcf_yield", "gross_prof", "roa", "accruals", "asset_growth"]


def sic_sector(sic):
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return "Unknown"
    if s in range(2830, 2837) or s in range(3841, 3852) or s in range(8000, 8100) or s == 5122:
        return "Health Care"
    if s in range(3570, 3580) or s in range(3660, 3700) or s in range(7370, 7380) or s == 3825:
        return "Technology"
    if s in range(1300, 1400) or s in range(2900, 3000) or s == 4922 or s == 4923:
        return "Energy"
    if s in range(4900, 5000):
        return "Utilities"
    if s in range(4800, 4900) or s in range(2710, 2750) or s in range(7810, 7850):
        return "Communication"
    if s in range(6000, 6800):
        return "Real Estate" if s in range(6500, 6600) or s == 6798 else "Financials"
    if s in range(5000, 6000) or s in range(7000, 7100) or s in range(5800, 5900):
        return "Consumer"
    if s in range(2000, 2200):
        return "Consumer Staples"
    if s in range(1000, 1500) or s in range(2600, 2700) or s in range(2800, 2830) or s in range(2840, 2900) or s in range(3300, 3400):
        return "Materials"
    if s in range(1500, 1800) or s in range(3400, 4800):
        return "Industrials"
    return "Other"


# ---------------------------------------------------------------- prices
def load_prices():
    uni = pd.read_parquet(PROC / "universe.parquet")
    spy = pd.read_parquet(RAW / "prices" / f"{BENCHMARK}.parquet").set_index("date")
    cal = spy.index
    cols = {k: {} for k in ["adjclose", "adj_open", "close", "volume", "high", "low"]}
    sym2cik = {}
    for sym, cik in zip(uni["yahoo"], uni["cik"]):
        f = RAW / "prices" / f"{sym}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f, columns=["date", "adjclose", "adj_open", "close", "volume", "high", "low"]).set_index("date")
        d = d[~d.index.duplicated()].reindex(cal)
        for k in cols:
            cols[k][sym] = d[k].astype("float32")
        sym2cik[sym] = int(cik)
    P = {k: pd.DataFrame(v, index=cal) for k, v in cols.items()}
    P["spy_adjclose"] = spy["adjclose"].astype("float32")
    P["spy_open"] = (spy["open"] * spy["adjclose"] / spy["close"]).astype("float32")
    return P, sym2cik


def price_features(P):
    ac, vol = P["adjclose"], P["volume"]
    r = ac.pct_change(fill_method=None)
    rs = P["spy_adjclose"].pct_change(fill_method=None)
    out = {
        "mom_12_1": ac.shift(21) / ac.shift(252) - 1,
        "mom_6_1": ac.shift(21) / ac.shift(126) - 1,
        "ret_1m": ac / ac.shift(21) - 1,
        "dist_52w_high": ac / ac.rolling(252, min_periods=200).max(),
        "px_sma200": ac / ac.rolling(200, min_periods=180).mean() - 1,
        "vol_60": r.rolling(60, min_periods=50).std() * np.sqrt(252),
    }
    cov = r.rolling(252, min_periods=200).cov(rs)
    out["beta_252"] = cov.div(rs.rolling(252, min_periods=200).var(), axis=0)
    dv = P["close"] * vol
    out["amihud"] = (r.abs() / dv.replace(0, np.nan)).rolling(60, min_periods=50).mean() * 1e9
    out["adv_shares"] = vol.rolling(ADV_WINDOW, min_periods=40).mean()
    out["adv_dollar"] = dv.rolling(ADV_WINDOW, min_periods=40).mean()
    out["drawdown_252"] = 1 - out["dist_52w_high"]
    return {k: v.astype("float32") for k, v in out.items()}


def labels(P, dec_idx):
    """Forward excess returns: open(t+1) -> open(t+1+h). A series that ends inside the window
    (delisting/acquisition) is closed at its last available adjusted close (0% delisting return
    assumption)."""
    op = P["adj_open"].to_numpy()
    last = P["adjclose"].ffill().to_numpy()
    spy = P["spy_open"].to_numpy()
    n = len(op)
    out = {}
    for h in HORIZONS:
        e, x = dec_idx + 1, dec_idx + 1 + h
        ok = x < n
        ent = np.full((len(dec_idx), op.shape[1]), np.nan, dtype="float32")
        ext = ent.copy()
        bench = np.full(len(dec_idx), np.nan, dtype="float32")
        ent[ok] = op[e[ok]]
        ext_open = op[x[ok]]
        ext[ok] = np.where(np.isnan(ext_open), last[x[ok]], ext_open)
        bench[ok] = spy[x[ok]] / spy[e[ok]] - 1
        out[h] = (ext / ent - 1) - bench[:, None]
    return out


# ---------------------------------------------------------------- fundamentals
def _step_latest(df):
    """For rows sorted by filed date, the latest-period record known at each filing date."""
    df = df.sort_values(["filed", "end"])
    best_end, rows = None, []
    for f, e, v in zip(df["filed"].to_numpy(), df["end"].to_numpy(), df["val"].to_numpy()):
        if best_end is None or e >= best_end:
            best_end = e
            rows.append((f, e, v))
    return pd.DataFrame(rows, columns=["filed", "end", "val"]).drop_duplicates("filed", keep="last")


def fundamentals_asof(dec_dates, ciks):
    """Return dict tag -> DataFrame[dec_date x cik] of latest as-reported values (filed < t),
    plus end-date frame for staleness, and prior-year assets for growth."""
    f = pd.read_parquet(RAW / "facts" / "companyfacts.parquet")
    f = f[f["cik"].isin(ciks)]
    dur = (f["end"] - f["start"]).dt.days
    annual_flow = (f["form"].str.startswith(("10-K", "20-F", "40-F"))) & dur.between(340, 390)
    instant = f["start"].isna()
    defs = {
        "ni": (["NetIncomeLoss"], annual_flow),
        "cfo": (["NetCashProvidedByUsedInOperatingActivities"], annual_flow),
        "capex": (["PaymentsToAcquirePropertyPlantAndEquipment"], annual_flow),
        "gp": (["GrossProfit"], annual_flow),
        "rev": (["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"], annual_flow),
        "cogs": (["CostOfRevenue", "CostOfGoodsAndServicesSold"], annual_flow),
        "equity": (["StockholdersEquity"], instant),
        "assets": (["Assets"], instant),
        "shares": (["EntityCommonStockSharesOutstanding"], instant),
    }
    t_avail = pd.DatetimeIndex(dec_dates) - pd.Timedelta(days=1)  # filed on or before t-1
    res = {}
    for key, (tags, mask) in defs.items():
        sub = f[mask & f["tag"].isin(tags)][["cik", "tag", "end", "filed", "val", "accn"]]
        if key == "shares":  # multi-class issuers report one cover-page fact per class
            sub = sub.groupby(["cik", "accn", "end", "filed"], as_index=False)["val"].sum()
        else:  # tag preference order, then dedupe
            sub["pri"] = sub["tag"].map({t: i for i, t in enumerate(tags)})
            sub = sub.sort_values("pri").drop_duplicates(["cik", "end", "filed"], keep="first")
        vals = pd.DataFrame(np.nan, index=dec_dates, columns=ciks, dtype="float64")
        ends = pd.DataFrame(pd.NaT, index=dec_dates, columns=ciks)
        prev = vals.copy() if key == "assets" else None
        for cik, g in sub.groupby("cik"):
            st = _step_latest(g)
            pos = np.searchsorted(st["filed"].to_numpy(), t_avail.to_numpy(), side="right") - 1
            okp = pos >= 0
            vals.loc[okp, cik] = st["val"].to_numpy()[pos[okp]]
            ends.loc[okp, cik] = st["end"].to_numpy()[pos[okp]]
            if key == "assets":  # value for the period ~1y before the latest one, as known when that latest was filed
                g2 = g.sort_values("filed")
                gf, ge, gv = g2["filed"].to_numpy(), g2["end"].to_numpy(), g2["val"].to_numpy()
                pv = np.full(len(st), np.nan)
                for k, (sf, se) in enumerate(zip(st["filed"].to_numpy(), st["end"].to_numpy())):
                    m = (gf <= sf) & (np.abs(ge - (se - np.timedelta64(365, "D"))) <= np.timedelta64(20, "D"))
                    if m.any():
                        pv[k] = gv[np.nonzero(m)[0][-1]]
                prev.loc[okp, cik] = pv[pos[okp]]
        res[key] = vals
        res[key + "_end"] = ends
        if prev is not None:
            res["assets_prev"] = prev
    return res


# ---------------------------------------------------------------- insider
def load_insider(ciks):
    files = sorted((RAW / "insider").glob("*.parquet"))
    cols = ["issuercik", "accession_number", "filing_date", "trans_date", "trans_code", "trans_shares",
            "trans_pricepershare", "trans_acquired_disp_cd", "rptownercik", "rptownername",
            "rptowner_relationship", "rptowner_title", "aff10b5one", "document_type", "dataset"]
    # older SEC quarterly files predate some columns (e.g. AFF10B5ONE, added 2023): read what exists
    def rd(f):
        have = set(pq.read_schema(f).names)
        return pd.read_parquet(f, columns=[c for c in cols if c in have]).reindex(columns=cols)
    parts = [rd(f) for f in files]
    recent = RAW / "insider_recent" / "form4_recent.parquet"
    if recent.exists():
        parts.append(rd(recent))
    d = pd.concat(parts, ignore_index=True)
    d = d[d["issuercik"].isin(ciks) & d["trans_code"].isin(["P", "S"])]
    # quarterly SEC files and the recent-filings parser can both carry the same filing
    d = d.drop_duplicates(["accession_number", "trans_date", "trans_code", "trans_shares", "trans_pricepershare"])
    # a 4/A restates transactions already reported on the original 4: keep the earliest report
    d = d.sort_values("filing_date").drop_duplicates(
        ["issuercik", "rptownercik", "trans_date", "trans_code", "trans_shares", "trans_pricepershare"], keep="first")
    d = d[(d["trans_shares"] > 0) & (d["trans_pricepershare"] > 0)]
    d["value"] = d["trans_shares"] * d["trans_pricepershare"]
    d = d[d["value"] < 5e9]  # guard against unit/keying errors in individual filings
    flag = d["aff10b5one"].astype(str).str.lower().isin(["1", "true"])
    d["is_10b5_1"] = flag
    t = d["rptowner_title"].fillna("").str.upper()
    d["is_ceo_cfo"] = t.str.contains(r"\bCEO\b|CHIEF EXECUTIVE|\bCFO\b|CHIEF FINANCIAL", regex=True)
    d["filing_date"] = pd.to_datetime(d["filing_date"])
    return d


def insider_features(ins, dec_dates, ciks):
    t_avail = pd.DatetimeIndex(dec_dates) - pd.Timedelta(days=1)
    shape = (len(dec_dates), len(ciks))
    out = {k: np.zeros(shape, dtype="float32") for k in
           ["buy_usd_90", "sell_usd_90", "net_usd_365", "ins_buyers_90", "ins_sellers_90", "ins_officer_buy_180"]}
    col = {c: i for i, c in enumerate(ciks)}
    tv = t_avail.to_numpy()
    for cik, g in ins.groupby("issuercik"):
        if cik not in col:
            continue
        j = col[cik]
        g = g.sort_values("filing_date")
        fd = g["filing_date"].to_numpy()
        buy = (g["trans_code"] == "P").to_numpy()
        sell = ((g["trans_code"] == "S") & ~g["is_10b5_1"]).to_numpy()
        val = g["value"].to_numpy()
        owner = g["rptownercik"].fillna(g["rptownername"]).to_numpy()
        ceo = g["is_ceo_cfo"].to_numpy()
        cb = np.concatenate([[0], np.cumsum(np.where(buy, val, 0))])
        cs = np.concatenate([[0], np.cumsum(np.where(sell, val, 0))])
        hi = np.searchsorted(fd, tv, side="right")
        lo90 = np.searchsorted(fd, tv - np.timedelta64(90, "D"), side="right")
        lo180 = np.searchsorted(fd, tv - np.timedelta64(180, "D"), side="right")
        lo365 = np.searchsorted(fd, tv - np.timedelta64(365, "D"), side="right")
        out["buy_usd_90"][:, j] = cb[hi] - cb[lo90]
        out["sell_usd_90"][:, j] = cs[hi] - cs[lo90]
        out["net_usd_365"][:, j] = (cb[hi] - cb[lo365]) - (cs[hi] - cs[lo365])
        for i in np.nonzero(hi > lo180)[0]:
            a, b, c = lo90[i], hi[i], lo180[i]
            if b > a:
                out["ins_buyers_90"][i, j] = len(set(owner[a:b][buy[a:b]]))
                out["ins_sellers_90"][i, j] = len(set(owner[a:b][sell[a:b]]))
            out["ins_officer_buy_180"][i, j] = float((buy[c:b] & ceo[c:b]).any())
    return {k: pd.DataFrame(v, index=dec_dates, columns=ciks) for k, v in out.items()}


# ---------------------------------------------------------------- panel
def build_panel(include_labels=True, last_n_dates=None):
    P, sym2cik = load_prices()
    cal = P["adjclose"].index
    syms = list(P["adjclose"].columns)
    ciks = [sym2cik[s] for s in syms]
    start = 260  # need a year of history for 12-1 momentum and beta
    dec_idx = np.arange(start, len(cal), DECISION_STEP)
    if dec_idx[-1] != len(cal) - 1:  # always include the latest session for live signals
        dec_idx = np.append(dec_idx, len(cal) - 1)
    if last_n_dates:
        dec_idx = dec_idx[-last_n_dates:]
    dec_dates = cal[dec_idx]

    pf = {k: v.iloc[dec_idx] for k, v in price_features(P).items()}
    fund = fundamentals_asof(dec_dates, ciks)
    ins = insider_features(load_insider(ciks), dec_dates, ciks)

    def C(df):  # cik-columned -> symbol-columned
        x = df.copy()
        x.columns = syms
        return x

    ac = P["adjclose"].iloc[dec_idx]
    # market cap: cover-page shares x close at the shares' as-of date, rolled forward with total return
    shares = C(fund["shares"])
    sh_end = C(fund["shares_end"])
    close_full = P["close"]
    adj_full = P["adjclose"]
    mcap = pd.DataFrame(np.nan, index=dec_dates, columns=syms)
    for s in syms:
        e = sh_end[s]
        okm = e.notna() & shares[s].notna()
        if not okm.any():
            continue
        pos = cal.searchsorted(pd.DatetimeIndex(e[okm]), side="right") - 1
        pos = np.clip(pos, 0, len(cal) - 1)
        c0 = close_full[s].to_numpy()[pos]
        a0 = adj_full[s].to_numpy()[pos]
        mcap.loc[okm, s] = shares[s][okm].to_numpy() * c0 * (ac[s][okm].to_numpy() / a0)
    gp = C(fund["gp"]).fillna(C(fund["rev"]) - C(fund["cogs"]))
    assets = C(fund["assets"]).where(C(fund["assets"]) > 0)
    feats = {**{k: pf[k] for k in ["mom_12_1", "mom_6_1", "ret_1m", "dist_52w_high", "px_sma200", "vol_60", "beta_252", "amihud"]}}
    feats["log_mcap"] = np.log(mcap.where(mcap > 0))
    feats["earn_yield"] = C(fund["ni"]) / mcap
    feats["book_to_mkt"] = C(fund["equity"]) / mcap
    feats["fcf_yield"] = (C(fund["cfo"]) - C(fund["capex"]).fillna(0)) / mcap
    feats["gross_prof"] = gp / assets
    feats["roa"] = C(fund["ni"]) / assets
    feats["accruals"] = (C(fund["ni"]) - C(fund["cfo"])) / assets
    feats["asset_growth"] = assets / C(fund["assets_prev"]).where(C(fund["assets_prev"]) > 0) - 1
    insC = {k: C(v) for k, v in ins.items()}
    feats["ins_net_90"] = (insC["buy_usd_90"] - insC["sell_usd_90"]) / mcap
    feats["ins_buyers_90"] = insC["ins_buyers_90"]
    feats["ins_sellers_90"] = insC["ins_sellers_90"]
    feats["ins_officer_buy_180"] = insC["ins_officer_buy_180"]
    feats["ins_surprise"] = ((insC["buy_usd_90"] - insC["sell_usd_90"]) - insC["net_usd_365"] * 90 / 365) / mcap
    feats["ins_buy_x_drawdown"] = insC["ins_buyers_90"] * pf["drawdown_252"]

    fund_end = C(fund["ni_end"])
    fund_age = pd.DataFrame(dec_dates.values[:, None] - fund_end.to_numpy(), index=dec_dates, columns=syms) / np.timedelta64(1, "D")
    elig = (mcap >= MIN_MCAP_USD) & (pf["adv_shares"] >= MIN_ADV_SHARES) & ac.notna()

    sic = pd.read_parquet(PROC / "sic.parquet").set_index("cik")["sic"]
    sector = {s: sic_sector(sic.get(c)) for s, c in sym2cik.items()}

    long = []
    lab = labels(P, dec_idx) if include_labels else {}
    for i, d in enumerate(dec_dates):
        m = elig.iloc[i].to_numpy()
        if not m.any():
            continue
        row = {"date": d, "symbol": np.array(syms)[m]}
        for k, v in feats.items():
            row[k] = v.iloc[i].to_numpy()[m].astype("float32")
        row["mcap"] = mcap.iloc[i].to_numpy()[m]
        row["adv_shares"] = pf["adv_shares"].iloc[i].to_numpy()[m]
        row["fund_age_days"] = fund_age.iloc[i].to_numpy()[m].astype("float32")
        for h in lab:
            row[f"y_{h}"] = lab[h][i][m]
        long.append(pd.DataFrame(row))
    panel = pd.concat(long, ignore_index=True)
    panel["sector"] = panel["symbol"].map(sector)
    panel["cik"] = panel["symbol"].map(sym2cik)
    meta = {"decision_dates": len(dec_dates), "first": str(dec_dates[0].date()), "last": str(dec_dates[-1].date()),
            "symbols_with_prices": len(syms), "rows": len(panel)}
    return panel, meta


if __name__ == "__main__":
    panel, meta = build_panel()
    panel.to_parquet(PROC / "panel.parquet", index=False)
    (PROC / "panel_meta.json").write_text(json.dumps(meta, indent=1))
    print(meta)
    print(panel.groupby(panel["date"].dt.year)["symbol"].count().to_string())
