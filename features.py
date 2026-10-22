"""Point-in-time feature panel (feature set fs-2.0).

Decision time t = close of session t. Price features use data through close t. Filings are eligible
only if filed on or before t-1 (one-session ingestion latency). Other sources use the availability
dates stored by their ingesters (macro/ALFRED vintages, EIA +5d, CFTC +3d, FINRA short interest +12d,
13F period end +46d). Labels: entry at session t+1 open, exit at t+1+h open, excess vs SPY.

Memory: prices for several thousand symbols are held as float32 arrays and price features are computed
in column chunks, keeping only decision-date rows.
"""
import json
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config as C
from config import ADV_WINDOW, BENCHMARK, DECISION_STEP, HORIZONS, PROC, RAW

# name -> (family, hypothesised sign (+1/-1, 0 = no directional prior, tested two-sided), description)
FEATURES = {
    # momentum / reversal / technical
    "mom_12_1": ("momentum", +1, "12-1 month total return (Jegadeesh-Titman)"),
    "mom_6_1": ("momentum", +1, "6-1 month total return"),
    "ret_1m": ("reversal", -1, "last 21-session return (short-term reversal)"),
    "dist_52w_high": ("momentum", +1, "price / 252-session high (George-Hwang)"),
    "px_sma200": ("technical", +1, "price / 200-session SMA - 1"),
    "rsi_14": ("technical", -1, "14-session RSI (overbought -> lower returns)"),
    "macd_hist": ("technical", +1, "MACD(12,26,9) histogram / price"),
    "bb_pctb": ("technical", -1, "Bollinger %b (20, 2sd)"),
    # risk / size / liquidity
    "vol_60": ("risk", -1, "60-session realised volatility (low-vol anomaly)"),
    # idiosyncratic / lottery / stationarity features (from markov_hmm_vs_code_project research_public_features)
    "ivol_60": ("idiosyncratic", -1, "60-session std of market-model residual returns (Ang-Hodrick-Xing-Zhang)"),
    "resid_mom_12_1": ("idiosyncratic", +1, "12-1 month sum of market-model residuals / their std (Blitz-Huij-Martens)"),
    "max_ret_21": ("lottery", -1, "largest daily return in the last 21 sessions (Bali-Cakici-Whitelaw MAX)"),
    "skew_252": ("lottery", -1, "252-session skewness of daily returns"),
    "r2_mkt_252": ("idiosyncratic", 0, "market-model R-squared over 252 sessions (price synchronicity)"),
    "vol_ratio_20_60": ("risk", 0, "EWMA vol 20d / EWMA vol 60d (volatility shock)"),
    "vol_of_vol_60": ("risk", -1, "60-session std of the 20-session realised volatility"),
    "ar1_60": ("stationarity", 0, "60-session first-order autocorrelation of daily returns"),
    "variance_ratio_120": ("stationarity", 0, "Lo-MacKinlay variance ratio VR(5) over 120 sessions (trend >1, mean reversion <1)"),
    "tail_ratio_60": ("lottery", 0, "(99th - 1st percentile of daily returns) / normal-equivalent range, 60 sessions"),
    "beta_instability_252": ("stationarity", 0, "252-session std of the rolling 60-session beta (parameter instability)"),
    "beta_252": ("risk", -1, "252-session beta vs SPY (betting-against-beta)"),
    "log_mcap": ("size", -1, "log market cap"),
    "amihud": ("liquidity", +1, "Amihud illiquidity, 60 sessions"),
    "cs_spread": ("liquidity", +1, "Corwin-Schultz high-low bid-ask spread estimate, 60-session mean"),
    "turnover": ("liquidity", -1, "60-session average volume / shares outstanding"),
    # value
    "earn_yield": ("value", +1, "annual net income / market cap"),
    "book_to_mkt": ("value", +1, "stockholders' equity / market cap"),
    "fcf_yield": ("value", +1, "(operating cash flow - capex) / market cap"),
    # quality / profitability
    "gross_prof": ("quality", +1, "gross profit / assets (Novy-Marx)"),
    "roa": ("quality", +1, "net income / assets"),
    "roic": ("quality", +1, "operating income x (1-21%) / (equity + long-term debt)"),
    "accruals": ("quality", -1, "(net income - operating cash flow) / assets (Sloan)"),
    "leverage": ("quality", -1, "total liabilities / assets"),
    # growth / investment / earnings
    "asset_growth": ("investment", -1, "annual asset growth (Cooper-Gulen-Schill)"),
    "capex_intensity": ("investment", -1, "capex / revenue"),
    "share_issuance": ("investment", -1, "year-on-year change in shares outstanding (dilution)"),
    "sue": ("earnings", +1, "standardised unexpected EPS: seasonal change / std of last 8 seasonal changes (filed 10-Q/10-K)"),
    "eps_growth": ("earnings", +1, "year-on-year change in quarterly diluted EPS / |prior EPS|"),
    "gm_change": ("earnings", +1, "annual gross-margin change"),
    "op_margin_change": ("earnings", +1, "annual operating-margin change"),
    "inv_to_sales_chg": ("earnings", -1, "change in inventory / revenue (inventory build-up)"),
    "earn_recent": ("event", 0, "earnings 8-K (item 2.02) in the last 5 sessions (0/1)"),
    # insider (section 3.2)
    "ins_net_90": ("insider", +1, "(open-mkt buy $ - discretionary sell $) / mcap, last 90d"),
    "ins_net_180": ("insider", +1, "same, last 180d"),
    "ins_buyers_90": ("insider", +1, "distinct insiders with open-market purchases, last 90d"),
    "ins_sellers_90": ("insider", -1, "distinct insiders with discretionary sales, last 90d"),
    "ins_officer_buy_180": ("insider", +1, "CEO/CFO open-market purchase in last 180d (0/1)"),
    "ins_dir_buy_90": ("insider", +1, "non-officer director purchase $ / mcap, 90d"),
    "ins_sponsor_sell_90": ("insider", -1, "10%-owner (non-officer) sale $ / mcap, 90d (sponsor exits)"),
    "ins_purchase_intensity": ("insider", +1, "max over 90d of shares bought / shares held after the trade"),
    "ins_conviction": ("insider", +1, "max over 90d of log(purchase $ / that insider's prior median trade $)"),
    "ins_opp_net_90": ("insider", +1, "net $ / mcap from opportunistic insiders (Cohen-Malloy-Pomorski), 90d"),
    "ins_routine_net_90": ("insider", 0, "net $ / mcap from routine insiders, 90d"),
    "ins_surprise": ("insider", +1, "90d net insider $ minus 90d-scaled trailing-365d net, / mcap"),
    "ins_buy_x_drawdown": ("insider", +1, "distinct buyers 90d x drawdown from 252d high"),
    "ins_plan_sale_share": ("insider", 0, "share of sale $ under 10b5-1 plans, 180d"),
    # institutional ownership (13F)
    "io_ratio": ("institutional", 0, "13F institutional shares / shares outstanding"),
    "io_breadth_chg": ("institutional", +1, "change in fraction of 13F filers holding (Chen-Hong-Stein)"),
    "io_holders_chg": ("institutional", +1, "log change in number of 13F holders"),
    "io_hhi": ("institutional", -1, "concentration (HHI) of 13F holdings"),
    # positioning (FINRA short interest, 2020+)
    "si_ratio": ("positioning", -1, "short interest / shares outstanding"),
    "si_days_to_cover": ("positioning", -1, "FINRA days to cover"),
    "si_change": ("positioning", -1, "change in short interest / shares outstanding vs prior settlement"),
    # macro exposures and tilts (section 4D / 5.10)
    "beta_rates": ("macro", 0, "252-session beta to daily change in 10y Treasury yield"),
    "beta_credit": ("macro", 0, "252-session beta to daily change in BAA-10y spread"),
    "beta_dollar": ("macro", 0, "252-session beta to broad dollar index return"),
    "rates_tilt": ("macro", +1, "beta_rates x 63-session change in 10y yield"),
    "credit_tilt": ("macro", +1, "beta_credit x 63-session change in BAA-10y"),
    # sector Mosaic modules (section 4; NaN outside the sector)
    "semi_cycle": ("sector_tech", +1, "semis/equipment: first-release semiconductor IP y/y growth"),
    "semi_orders": ("sector_tech", +1, "semis/equipment: computers & electronics new orders 3m log change (first release)"),
    "bio_cash_runway": ("sector_health", +1, "pharma/biotech: cash / annual operating cash burn (years, capped 10)"),
    "bio_rnd_intensity": ("sector_health", 0, "pharma/biotech: R&D / revenue"),
    "bio_ph3_trials": ("sector_health", +1, "pharma/biotech: registered phase-3 trials first posted, not past primary completion"),
    "bio_fda_appr_365": ("sector_health", +1, "pharma/biotech: FDA approvals (openFDA) in last 365d"),
    "energy_oil_tilt": ("sector_energy", +1, "energy: beta to WTI x 63-session WTI return"),
    "energy_inv_surprise": ("sector_energy", +1, "energy: -(EIA crude stocks vs 5y same-week mean, z) x beta to WTI"),
    "energy_crack_chg": ("sector_energy", +1, "refiners: 63-session change in 3-2-1 crack spread (z)"),
    "energy_cot_z": ("sector_energy", -1, "energy: WTI managed-money net % OI 3y z-score x beta to WTI (crowding)"),
    "fin_deposit_growth": ("sector_fin", +1, "banks: deposit growth y/y"),
    "fin_nim_proxy": ("sector_fin", +1, "banks: net interest income / assets"),
    "fin_provision_ratio": ("sector_fin", -1, "banks: loan-loss provisions / loans"),
    "fin_slope_tilt": ("sector_fin", +1, "banks: 63-session change in 10y-3m slope"),
    "mat_commodity_tilt": ("sector_materials", +1, "materials/ag/energy: sum over commodities of beta x 63-session return"),
    # market-level options information (VIX term structure, CBOE SKEW: 10y+ history)
    "opt_mkt_term_tilt": ("options_market", 0, "beta x VIX term slope (VIX3M / VIX - 1): high-beta names when the curve is in contango/backwardation"),
    "opt_mkt_skew_tilt": ("options_market", 0, "beta x CBOE SKEW 3y z-score (priced tail risk)"),
    # macro releases vs an ARIMA forecast from earlier first releases (macro_surprise.py; spec 4D released-vs-expected)
    "growth_surprise_tilt": ("macro", 0, "beta x growth surprise index (payrolls, -unemployment, IP, -claims), last 31 days"),
    "inflation_surprise_tilt": ("macro", 0, "beta to 10y yield changes x CPI surprise z, last 31 days"),
    # earnings surprise vs consensus at report time (Alpha Vantage, backfilled 24 companies/day)
    "sue_consensus": ("earnings", +1, "latest quarterly EPS surprise vs consensus (%), clipped +-100, NaN if > 200 days old"),
    # per-stock snapshots collected nightly from 2026-10-20 (excluded from models until >= 3 years of history)
    "opt_iv_rv": ("options", -1, "30d ATM implied vol / 60-session realised vol - 1 (Bali-Hovakimyan)"),
    "opt_term_slope": ("options", +1, "90d / 30d ATM implied vol - 1 (inverted term structure = stress)"),
    "opt_skew_rr": ("options", -1, "25-delta put IV - call IV (Xing-Zhang-Zhao smirk)"),
    "opt_pc_ratio": ("options", -1, "put / call option volume"),
    "est_rev_1m": ("estimates", +1, "1-month change in current-year consensus EPS"),
    "est_dispersion": ("estimates", -1, "(high - low) quarterly EPS estimate / |consensus| (Diether-Malloy-Scherbina)"),
}
SNAPSHOT_FEATURES = {"opt_iv_rv", "opt_term_slope", "opt_skew_rr", "opt_pc_ratio", "est_rev_1m", "est_dispersion"}
SNAPSHOT_MIN_YEARS = 3  # a snapshot feature enters models only once its history spans this long
FUND_FEATURES = ["earn_yield", "book_to_mkt", "fcf_yield", "gross_prof", "roa", "accruals", "asset_growth"]
PRICE_FEATURES = ["mom_12_1", "mom_6_1", "ret_1m", "dist_52w_high", "px_sma200", "rsi_14", "macd_hist", "bb_pctb",
                  "vol_60", "beta_252", "amihud", "cs_spread", "beta_rates", "beta_credit", "beta_dollar",
                  "ivol_60", "resid_mom_12_1", "max_ret_21", "skew_252", "r2_mkt_252", "vol_ratio_20_60", "vol_of_vol_60", "ar1_60",
                  "variance_ratio_120", "tail_ratio_60", "beta_instability_252"]
CHUNK = 400


def sic_sector(sic):
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return "Unknown"
    if s in range(2830, 2837) or s in range(3841, 3852) or s in range(8000, 8100) or s == 5122 or s == 8731:
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


def sector_flags(sic):
    try:
        s = int(sic)
    except (TypeError, ValueError):
        s = -1
    return {"semis": s in (3674, 3559, 3825, 3827), "pharma": s in range(2833, 2837) or s == 8731,
            "energy": s in range(1300, 1400) or s in (2911, 4922, 4923, 4924, 5171, 5172), "refiner": s == 2911,
            "bank": s in range(6020, 6037) or s in range(6110, 6170),
            "commodity": s in range(100, 1000) or s in range(1000, 1500) or s in range(2000, 2100) or s in range(2800, 2830)
            or s in range(2870, 2880) or s in range(3300, 3400) or s in range(1300, 1400) or s == 2911}


# ---------------------------------------------------------------- prices
def load_prices(min_dollar_adv=None, symbols=None):
    """Adjusted daily prices as float32 frames on the SPY calendar. Symbols whose 60-session dollar volume
    never reaches `min_dollar_adv` are skipped (they could never pass any liquidity screen)."""
    min_dollar_adv = C.PROFILE["min_dollar_adv"] if min_dollar_adv is None else min_dollar_adv
    uni = pd.read_parquet(PROC / "universe.parquet")
    spy = pd.read_parquet(RAW / "prices" / f"{BENCHMARK}.parquet").set_index("date")
    spy = spy[~spy.index.duplicated()]
    cal = spy.index
    keys = ["adjclose", "adj_open", "close", "volume", "high", "low"]
    arrs = {k: [] for k in keys}
    syms, sym2cik = [], {}
    for sym, cik in zip(uni["yahoo"], uni["cik"]):
        if symbols is not None and sym not in symbols:
            continue
        f = RAW / "prices" / f"{sym}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f, columns=["date"] + keys).drop_duplicates("date").set_index("date").reindex(cal)
        dv = (d["close"] * d["volume"]).rolling(ADV_WINDOW, min_periods=40).mean()
        if not (dv.max() >= min_dollar_adv):
            continue
        for k in keys:
            arrs[k].append(d[k].to_numpy(dtype="float32"))
        syms.append(sym)
        sym2cik[sym] = int(cik)
    P = {k: pd.DataFrame(np.column_stack(v) if v else np.empty((len(cal), 0), "float32"), index=cal, columns=syms) for k, v in arrs.items()}
    P["spy_adjclose"] = spy["adjclose"].astype("float32")
    P["spy_open"] = (spy["open"] * spy["adjclose"] / spy["close"]).astype("float32")
    return P, sym2cik


def _macro_returns(cal):
    """Daily factor moves used for exposure estimation: dy10 (pp), dbaa (pp), dollar/commodity log returns."""
    out = pd.DataFrame(index=cal)
    try:
        from data_macro import daily
        m = daily(cal)
        out["dy10"] = m["DGS10"].diff()
        out["dbaa"] = m["BAA10Y"].diff()
        out["dollar"] = np.log(m["DTWEXBGS"]).diff()
    except Exception:
        pass
    fut = RAW / "alt" / "futures.parquet"
    if fut.exists():
        f = pd.read_parquet(fut)
        for c in ["wti", "natgas", "copper", "gold", "corn"]:
            s = f[f["commodity"] == c].set_index("date")["close"]
            s = s[~s.index.duplicated()].reindex(cal).ffill(limit=3)
            out[c] = np.log(s.where(s > 0)).diff()
    return out


def _cs_spread(high, low):
    """Corwin-Schultz (2012) two-day high-low spread estimator, negatives set to 0."""
    hl = np.log(high / low) ** 2
    beta = hl + hl.shift(1)
    h2 = np.maximum(high, high.shift(1))
    l2 = np.minimum(low, low.shift(1))
    gamma = np.log(h2 / l2) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return s.clip(lower=0)


def price_features(P, dec_idx, macro_ret=None):
    """Price-derived features at decision rows only, computed in column chunks."""
    cal = P["adjclose"].index
    rs = P["spy_adjclose"].astype("float64").pct_change(fill_method=None)
    out = {k: [] for k in PRICE_FEATURES + ["adv_shares", "adv_dollar", "drawdown_252"] + [f"beta_{c}" for c in ["wti", "natgas", "copper", "gold", "corn"]]}
    for a in range(0, P["adjclose"].shape[1], CHUNK):
        cols = P["adjclose"].columns[a:a + CHUNK]
        ac = P["adjclose"][cols].astype("float64")
        vol = P["volume"][cols].astype("float64")
        r = ac.pct_change(fill_method=None)
        o = {}
        o["mom_12_1"] = ac.shift(21) / ac.shift(252) - 1
        o["mom_6_1"] = ac.shift(21) / ac.shift(126) - 1
        o["ret_1m"] = ac / ac.shift(21) - 1
        hi252 = ac.rolling(252, min_periods=200).max()
        o["dist_52w_high"] = ac / hi252
        o["drawdown_252"] = 1 - o["dist_52w_high"]
        o["px_sma200"] = ac / ac.rolling(200, min_periods=180).mean() - 1
        d = ac.diff()
        up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        o["rsi_14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
        macd = ac.ewm(span=12, adjust=False).mean() - ac.ewm(span=26, adjust=False).mean()
        o["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / ac
        m20, s20 = ac.rolling(20).mean(), ac.rolling(20).std()
        o["bb_pctb"] = (ac - (m20 - 2 * s20)) / (4 * s20)
        o["vol_60"] = r.rolling(60, min_periods=50).std() * np.sqrt(252)
        o["beta_252"] = r.rolling(252, min_periods=200).cov(rs).div(rs.rolling(252, min_periods=200).var(), axis=0)
        dv = P["close"][cols].astype("float64") * vol
        o["amihud"] = (r.abs() / dv.replace(0, np.nan)).rolling(60, min_periods=50).mean() * 1e9
        o["adv_shares"] = vol.rolling(ADV_WINDOW, min_periods=40).mean()
        o["adv_dollar"] = dv.rolling(ADV_WINDOW, min_periods=40).mean()
        hi, lo = P["high"][cols].astype("float64"), P["low"][cols].astype("float64")
        o["cs_spread"] = _cs_spread(hi.where(hi >= lo), lo.where(hi >= lo)).rolling(60, min_periods=40).mean()
        o.update(stat_features(r, rs, o["beta_252"]))
        if macro_ret is not None:
            for name, col in [("beta_rates", "dy10"), ("beta_credit", "dbaa"), ("beta_dollar", "dollar"),
                              ("beta_wti", "wti"), ("beta_natgas", "natgas"), ("beta_copper", "copper"), ("beta_gold", "gold"), ("beta_corn", "corn")]:
                if col in macro_ret:
                    x = macro_ret[col]
                    o[name] = r.rolling(252, min_periods=200).cov(x).div(x.rolling(252, min_periods=200).var(), axis=0)
        for k in out:
            if k in o:
                out[k].append(o[k].iloc[dec_idx].astype("float32"))
            else:
                out[k].append(pd.DataFrame(np.nan, index=cal[dec_idx], columns=cols, dtype="float32"))
        del o, ac, vol, r, dv
    return {k: pd.concat(v, axis=1) for k, v in out.items()}


def stat_features(r, rs, beta_252):
    """Idiosyncratic, lottery and stationarity features from daily returns (r: dates x names, rs: SPY returns).
    Every window ends at t; residuals use betas estimated through t (a rolling-window, not look-ahead, regression)."""
    o = {}
    var60 = rs.rolling(60, min_periods=50).var()
    beta60 = r.rolling(60, min_periods=50).cov(rs).div(var60, axis=0)
    resid60 = r - beta60.mul(rs, axis=0)
    o["ivol_60"] = resid60.rolling(60, min_periods=50).std() * np.sqrt(252)
    resid252 = r - beta_252.mul(rs, axis=0)
    s_ = resid252.shift(21).rolling(231, min_periods=180)
    o["resid_mom_12_1"] = s_.sum() / s_.std()
    o["max_ret_21"] = r.rolling(21, min_periods=18).max()
    o["skew_252"] = r.rolling(252, min_periods=200).skew()
    var_r = r.rolling(252, min_periods=200).var()
    o["r2_mkt_252"] = (beta_252 ** 2).mul(rs.rolling(252, min_periods=200).var(), axis=0) / var_r
    o["vol_ratio_20_60"] = np.sqrt((r ** 2).ewm(span=20, adjust=False).mean() / (r ** 2).ewm(span=60, adjust=False).mean())
    o["vol_of_vol_60"] = r.rolling(20, min_periods=15).std().rolling(60, min_periods=50).std() * np.sqrt(252)
    o["ar1_60"] = r.rolling(60, min_periods=50).corr(r.shift(1))
    r5 = r.rolling(5).sum()
    o["variance_ratio_120"] = r5.rolling(120, min_periods=100).var() / (5 * r.rolling(120, min_periods=100).var())
    rng = r.rolling(60, min_periods=50).quantile(0.99) - r.rolling(60, min_periods=50).quantile(0.01)
    o["tail_ratio_60"] = rng / (2 * 2.326 * r.rolling(60, min_periods=50).std())
    o["beta_instability_252"] = beta60.rolling(252, min_periods=200).std()
    return o


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


FUND_DEFS = {  # key: (tags in preference order, kind) ; kind: annual flow / instant
    "ni": (["NetIncomeLoss"], "annual"), "cfo": (["NetCashProvidedByUsedInOperatingActivities"], "annual"),
    "capex": (["PaymentsToAcquirePropertyPlantAndEquipment"], "annual"), "gp": (["GrossProfit"], "annual"),
    "rev": (["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"], "annual"),
    "cogs": (["CostOfRevenue", "CostOfGoodsAndServicesSold"], "annual"), "opinc": (["OperatingIncomeLoss"], "annual"),
    "rnd": (["ResearchAndDevelopmentExpense"], "annual"), "nii": (["InterestIncomeExpenseNet"], "annual"),
    "prov": (["ProvisionForLoanLeaseAndOtherLosses", "ProvisionForLoanAndLeaseLosses", "ProvisionForCreditLosses"], "annual"),
    "equity": (["StockholdersEquity"], "instant"), "assets": (["Assets"], "instant"), "liab": (["Liabilities"], "instant"),
    "ltd": (["LongTermDebt", "LongTermDebtNoncurrent"], "instant"), "cash": (["CashAndCashEquivalentsAtCarryingValue"], "instant"),
    "inv": (["InventoryNet"], "instant"), "deposits": (["Deposits"], "instant"),
    "loans": (["LoansAndLeasesReceivableNetReportedAmount"], "instant"), "shares": (["EntityCommonStockSharesOutstanding"], "instant"),
    "shares_dil": (["WeightedAverageNumberOfDilutedSharesOutstanding"], "annual"),
}
YOY = {"assets", "deposits", "shares", "rev", "gp", "opinc", "inv"}  # also keep the value ~1y earlier, as known then


def load_facts(ciks):
    """Only the tags the features use, strings as categoricals (the full file is ~10M rows)."""
    tags = sorted({t for tl, _ in FUND_DEFS.values() for t in tl} | {"EarningsPerShareDiluted"})  # includes diluted share count
    t = pq.read_table(RAW / "facts" / "companyfacts.parquet", columns=["cik", "tag", "start", "end", "val", "filed", "form", "accn"],
                      filters=[("tag", "in", tags), ("cik", "in", [int(c) for c in ciks])])
    f = t.to_pandas(strings_to_categorical=True)
    for c in ["start", "end", "filed"]:
        v = f[c]  # filings occasionally carry keying errors (e.g. year 202): treat out-of-range dates as missing
        v = v.where((v >= pd.Timestamp("1990-01-01")) & (v <= pd.Timestamp("2100-01-01")))
        f[c] = v.astype("datetime64[ns]")
    f["form"] = f["form"].astype(str)
    f["tag"] = f["tag"].astype(str)
    return f


def fundamentals_asof(dec_dates, ciks, f=None):
    """dict key -> DataFrame[dec_date x cik] of latest as-reported values filed <= t-1 (plus *_end and *_prev)."""
    f = load_facts(ciks) if f is None else f
    dur = (f["end"] - f["start"]).dt.days
    masks = {"annual": f["form"].str.startswith(("10-K", "20-F", "40-F")) & dur.between(340, 390), "instant": f["start"].isna()}
    t_avail = pd.DatetimeIndex(dec_dates) - pd.Timedelta(days=1)
    tv = t_avail.to_numpy()
    res = {}
    col = {c: j for j, c in enumerate(ciks)}
    for key, (tags, kind) in FUND_DEFS.items():
        sub = f[masks[kind] & f["tag"].isin(tags)][["cik", "tag", "end", "filed", "val", "accn"]]
        if key == "shares":  # multi-class issuers report one cover-page fact per class
            sub = sub.groupby(["cik", "accn", "end", "filed"], as_index=False)["val"].sum()
        else:
            sub = sub.assign(pri=sub["tag"].map({t: i for i, t in enumerate(tags)})).sort_values("pri")
            sub = sub.drop_duplicates(["cik", "end", "filed"], keep="first")
        vals = np.full((len(dec_dates), len(ciks)), np.nan, dtype="float64")
        ends = np.full((len(dec_dates), len(ciks)), np.datetime64("NaT"), dtype="datetime64[ns]")
        prev = np.full_like(vals, np.nan) if key in YOY else None
        for cik, g in sub.groupby("cik"):
            j = col.get(cik)
            if j is None:
                continue
            st = _step_latest(g)
            pos = np.searchsorted(st["filed"].to_numpy(), tv, side="right") - 1
            okp = pos >= 0
            vals[okp, j] = st["val"].to_numpy()[pos[okp]]
            ends[okp, j] = st["end"].to_numpy()[pos[okp]]
            if prev is not None:
                g2 = g.sort_values("filed")
                gf, ge, gv = g2["filed"].to_numpy(), g2["end"].to_numpy(), g2["val"].to_numpy()
                pv = np.full(len(st), np.nan)
                for k, (sf, se) in enumerate(zip(st["filed"].to_numpy(), st["end"].to_numpy())):
                    m = (gf <= sf) & (np.abs(ge - (se - np.timedelta64(365, "D"))) <= np.timedelta64(20, "D"))
                    if m.any():
                        pv[k] = gv[np.nonzero(m)[0][-1]]
                prev[okp, j] = pv[pos[okp]]
        res[key] = pd.DataFrame(vals, index=dec_dates, columns=ciks)
        if key in ("ni", "shares"):  # only these as-of period ends are used downstream
            res[key + "_end"] = pd.DataFrame(ends, index=dec_dates, columns=ciks)
        if prev is not None:
            res[key + "_prev"] = pd.DataFrame(prev, index=dec_dates, columns=ciks)
    return res


def eps_surprise(dec_dates, ciks, f=None):
    """SUE and EPS growth from quarterly diluted EPS (10-Q quarters; Q4 = fiscal year minus Q1-Q3), each
    usable once the filing carrying it was filed (<= t-1)."""
    f = load_facts(ciks) if f is None else f
    e = f[f["tag"] == "EarningsPerShareDiluted"].dropna(subset=["start"]).copy()
    e["dur"] = (e["end"] - e["start"]).dt.days
    e = e.sort_values("filed").drop_duplicates(["cik", "start", "end"], keep="first")  # as first reported
    q = e[e["dur"].between(80, 100)][["cik", "end", "val", "filed"]]
    fy = e[e["dur"].between(350, 380)]
    q4 = []
    qg = {c: g for c, g in q.groupby("cik")}
    for r in fy.itertuples():
        g = qg.get(r.cik)
        if g is None:
            continue
        qs = g[(g["end"] > r.start) & (g["end"] < r.end - pd.Timedelta(days=60))]
        if len(qs) == 3:
            q4.append({"cik": r.cik, "end": r.end, "val": r.val - qs["val"].sum(), "filed": r.filed})
    q = pd.concat([q, pd.DataFrame(q4)], ignore_index=True).sort_values(["cik", "end"])
    t_avail = (pd.DatetimeIndex(dec_dates) - pd.Timedelta(days=1)).to_numpy()
    sue = np.full((len(dec_dates), len(ciks)), np.nan)
    gro = sue.copy()
    col = {c: j for j, c in enumerate(ciks)}
    for cik, g in q.groupby("cik"):
        j = col.get(cik)
        if j is None or len(g) < 6:
            continue
        g = g.drop_duplicates("end").reset_index(drop=True)
        ends, vals = g["end"].to_numpy(), g["val"].to_numpy()
        rows = []
        for i in range(len(g)):
            k = np.nonzero(np.abs(ends[:i] - (ends[i] - np.timedelta64(364, "D"))) <= np.timedelta64(20, "D"))[0]
            if not len(k):
                continue
            ch = vals[i] - vals[k[-1]]
            hist = []
            for m in range(max(0, i - 8), i):
                kk = np.nonzero(np.abs(ends[:m] - (ends[m] - np.timedelta64(364, "D"))) <= np.timedelta64(20, "D"))[0]
                if len(kk):
                    hist.append(vals[m] - vals[kk[-1]])
            sd = np.std(hist, ddof=1) if len(hist) >= 4 else np.nan
            avail = max(g["filed"].iloc[i], g["filed"].iloc[k[-1]])
            rows.append((avail, ch / sd if sd and sd > 0 else np.nan, ch / abs(vals[k[-1]]) if vals[k[-1]] else np.nan))
        if not rows:
            continue
        rr = pd.DataFrame(rows, columns=["avail", "sue", "gro"]).sort_values("avail")
        pos = np.searchsorted(rr["avail"].to_numpy(), t_avail, side="right") - 1
        ok = pos >= 0
        sue[ok, j] = rr["sue"].to_numpy()[pos[ok]]
        gro[ok, j] = rr["gro"].to_numpy()[pos[ok]]
    return pd.DataFrame(sue, index=dec_dates, columns=ciks), pd.DataFrame(gro, index=dec_dates, columns=ciks)


# ---------------------------------------------------------------- insider
INS_COLS = ["issuercik", "accession_number", "filing_date", "trans_date", "trans_code", "trans_shares",
            "trans_pricepershare", "shrs_ownd_folwng_trans", "rptownercik", "rptownername",
            "rptowner_relationship", "rptowner_title", "aff10b5one", "fn_10b5_1", "document_type"]


def load_insider(ciks, codes=("P", "S")):
    files = sorted((RAW / "insider").glob("*.parquet"))

    want = set(int(c) for c in ciks)

    def rd(f):  # older files predate some columns: read what exists; filter rows at read time to save memory
        have = set(pq.read_schema(f).names)
        t = pq.read_table(f, columns=[c for c in INS_COLS if c in have], filters=[("trans_code", "in", list(codes))])
        d = t.to_pandas(strings_to_categorical=True)
        d = d[d["issuercik"].isin(want)]
        return d.reindex(columns=INS_COLS)
    parts = [rd(f) for f in files]
    recent = RAW / "insider_recent" / "form4_recent.parquet"
    if recent.exists():
        parts.append(rd(recent))
    for x in parts:  # align categorical dtypes across files before concatenating
        for c in x.columns:
            if isinstance(x[c].dtype, pd.CategoricalDtype):
                x[c] = x[c].astype(str).where(x[c].notna(), None)
    d = pd.concat(parts, ignore_index=True)
    d = d[d["issuercik"].isin(ciks) & d["trans_code"].isin(codes)]
    for c in ["accession_number", "rptownercik", "rptownername", "rptowner_relationship", "rptowner_title", "document_type", "trans_code"]:
        d[c] = d[c].astype("category")
    # quarterly SEC files and the recent-filings parser can both carry the same filing
    d = d.drop_duplicates(["accession_number", "trans_date", "trans_code", "trans_shares", "trans_pricepershare"])
    # a 4/A restates transactions already reported on the original 4: keep the earliest report
    d = d.sort_values("filing_date").drop_duplicates(
        ["issuercik", "rptownercik", "trans_date", "trans_code", "trans_shares", "trans_pricepershare"], keep="first")
    d = d[(d["trans_shares"] > 0) & (d["trans_pricepershare"] > 0)]
    d["value"] = d["trans_shares"] * d["trans_pricepershare"]
    d = d[d["value"] < 5e9]  # guard against unit/keying errors in individual filings
    d["is_10b5_1"] = d["aff10b5one"].astype(str).str.lower().isin(["1", "true"]) | d["fn_10b5_1"].fillna(False).astype(bool)
    t = d["rptowner_title"].astype(str).replace("nan", "").str.upper()
    rel = d["rptowner_relationship"].astype(str).replace("nan", "")
    d["is_ceo_cfo"] = t.str.contains(r"\bCEO\b|CHIEF EXECUTIVE|\bCFO\b|CHIEF FINANCIAL", regex=True)
    d["is_officer"] = rel.str.contains("Officer")
    d["is_director"] = rel.str.contains("Director") & ~d["is_officer"]
    d["is_10pct"] = rel.str.contains("TenPercentOwner") & ~d["is_officer"] & ~rel.str.contains("Director")
    d["filing_date"] = pd.to_datetime(d["filing_date"])
    d["owner"] = d["rptownercik"].astype(str).where(d["rptownercik"].notna(), d["rptownername"].astype(str)).astype("category")
    return classify_insiders(d)


def classify_insiders(d):
    """Cohen-Malloy-Pomorski (2012): an insider is 'routine' for a trade in year y if they traded in the same
    calendar month in each of y-1, y-2, y-3; 'opportunistic' if they have trades in all three prior years
    but not this pattern; otherwise unclassified. Uses only the insider's earlier trades. Also the insider's
    prior median trade value (for conviction)."""
    d = d.sort_values(["owner", "trans_date", "filing_date"]).copy()
    td = pd.to_datetime(d["trans_date"]).fillna(d["filing_date"])
    d["_y"], d["_m"] = td.dt.year.to_numpy(), td.dt.month.to_numpy()
    cls = np.full(len(d), "unclassified", dtype=object)
    prior_med = np.full(len(d), np.nan)
    ys, ms, vals, own = d["_y"].to_numpy(), d["_m"].to_numpy(), d["value"].to_numpy(), d["owner"].to_numpy()
    start = 0
    for i in range(1, len(d) + 1):
        if i == len(d) or own[i] != own[start]:
            yrs, ym, vv = ys[start:i], set(), []
            for k in range(start, i):
                y, m = ys[k], ms[k]
                prev_years = {yy for yy, _ in ym}
                if {y - 1, y - 2, y - 3} <= prev_years:
                    cls[k] = "routine" if all((y - z, m) in ym for z in (1, 2, 3)) else "opportunistic"
                if vv:
                    prior_med[k] = np.median(vv[-50:])
                # make this trade visible only to later-year trades / later trades
                ym.add((y, m))
                vv.append(vals[k])
            start = i
    d["insider_class"] = cls
    d["prior_median_value"] = prior_med
    return d.drop(columns=["_y", "_m"])


def insider_features(ins, dec_dates, ciks):
    t_avail = (pd.DatetimeIndex(dec_dates) - pd.Timedelta(days=1)).to_numpy()
    shape = (len(dec_dates), len(ciks))
    names = ["buy_usd_90", "sell_usd_90", "buy_usd_180", "sell_usd_180", "net_usd_365", "ins_buyers_90", "ins_sellers_90",
             "ins_officer_buy_180", "dir_buy_usd_90", "sponsor_sell_usd_90", "ins_purchase_intensity", "ins_conviction",
             "opp_net_usd_90", "routine_net_usd_90", "plan_sell_usd_180", "all_sell_usd_180"]
    out = {k: np.zeros(shape, dtype="float32") for k in names}
    for k in ["ins_purchase_intensity", "ins_conviction"]:
        out[k][:] = np.nan
    col = {c: i for i, c in enumerate(ciks)}
    for cik, g in ins.groupby("issuercik"):
        j = col.get(cik)
        if j is None:
            continue
        g = g.sort_values("filing_date")
        fd = g["filing_date"].to_numpy()
        buy = (g["trans_code"] == "P").to_numpy()
        sale = (g["trans_code"] == "S").to_numpy()
        sell = sale & ~g["is_10b5_1"].to_numpy()
        val = g["value"].to_numpy()
        owner = g["owner"].to_numpy()
        ceo, dire, ten = g["is_ceo_cfo"].to_numpy(), g["is_director"].to_numpy(), g["is_10pct"].to_numpy()
        opp, rout = (g["insider_class"] == "opportunistic").to_numpy(), (g["insider_class"] == "routine").to_numpy()
        inten = np.where(buy, g["trans_shares"].to_numpy() / np.maximum(g["shrs_ownd_folwng_trans"].fillna(0).to_numpy(), 1.0), np.nan)
        conv = np.where(buy & (g["prior_median_value"].to_numpy() > 0), np.log(val / g["prior_median_value"].to_numpy()), np.nan)

        def cs(x):
            return np.concatenate([[0.0], np.cumsum(x)])
        c = {"b": cs(np.where(buy, val, 0)), "s": cs(np.where(sell, val, 0)), "db": cs(np.where(buy & dire, val, 0)),
             "ss": cs(np.where(sale & ten, val, 0)), "on": cs(np.where(opp & buy, val, 0) - np.where(opp & sell, val, 0)),
             "rn": cs(np.where(rout & buy, val, 0) - np.where(rout & sell, val, 0)),
             "ps": cs(np.where(sale & ~sell, val, 0)), "as": cs(np.where(sale, val, 0))}
        hi = np.searchsorted(fd, t_avail, side="right")
        lo = {d: np.searchsorted(fd, t_avail - np.timedelta64(d, "D"), side="right") for d in (90, 180, 365)}
        out["buy_usd_90"][:, j] = c["b"][hi] - c["b"][lo[90]]
        out["sell_usd_90"][:, j] = c["s"][hi] - c["s"][lo[90]]
        out["buy_usd_180"][:, j] = c["b"][hi] - c["b"][lo[180]]
        out["sell_usd_180"][:, j] = c["s"][hi] - c["s"][lo[180]]
        out["net_usd_365"][:, j] = (c["b"][hi] - c["b"][lo[365]]) - (c["s"][hi] - c["s"][lo[365]])
        out["dir_buy_usd_90"][:, j] = c["db"][hi] - c["db"][lo[90]]
        out["sponsor_sell_usd_90"][:, j] = c["ss"][hi] - c["ss"][lo[90]]
        out["opp_net_usd_90"][:, j] = c["on"][hi] - c["on"][lo[90]]
        out["routine_net_usd_90"][:, j] = c["rn"][hi] - c["rn"][lo[90]]
        out["plan_sell_usd_180"][:, j] = c["ps"][hi] - c["ps"][lo[180]]
        out["all_sell_usd_180"][:, j] = c["as"][hi] - c["as"][lo[180]]
        for i in np.nonzero(hi > lo[180])[0]:
            a, b, s180 = lo[90][i], hi[i], lo[180][i]
            if b > a:
                ob = buy[a:b]
                out["ins_buyers_90"][i, j] = len(set(owner[a:b][ob]))
                out["ins_sellers_90"][i, j] = len(set(owner[a:b][sell[a:b]]))
                if ob.any():
                    x = inten[a:b][ob]
                    out["ins_purchase_intensity"][i, j] = np.nanmax(x) if np.isfinite(x).any() else np.nan
                    y = conv[a:b][ob]
                    out["ins_conviction"][i, j] = np.nanmax(y) if np.isfinite(y).any() else np.nan
            out["ins_officer_buy_180"][i, j] = float((buy[s180:b] & ceo[s180:b]).any())
    return {k: pd.DataFrame(v, index=dec_dates, columns=ciks) for k, v in out.items()}


# ---------------------------------------------------------------- 13F, short interest, 8-K
_SUFFIX = re.compile(r"\b(INC|CORP|CORPORATION|CO|COMPANY|LTD|LIMITED|PLC|HOLDINGS?|HLDGS?|GROUP|GRP|N ?V|S ?A|AG|SE|LLC|LP|THE|DEL|NEW|CL [A-Z]|COM)\b")


def _norm(n):
    n = re.sub(r"[^A-Z0-9 ]", " ", str(n).upper().replace("&", " AND ").lstrip("*"))
    return re.sub(r"\s+", " ", _SUFFIX.sub(" ", n)).strip()


def cusip_map():
    """Map 13F CUSIPs to universe CIKs by normalised issuer name (common-stock classes only)."""
    dest = PROC / "cusip_cik.parquet"
    if dest.exists():
        return pd.read_parquet(dest)
    names = pd.concat([pd.read_parquet(f) for f in sorted((RAW / "13f" / "names").glob("*.parquet"))])
    names["CUSIP"] = names["CUSIP"].astype(str).str.zfill(9).str.upper()
    names = names.drop_duplicates("CUSIP")
    cls = names["TITLEOFCLASS"].fillna("").str.upper()
    names = names[cls.str.contains(r"\bCOM|CL [A-Z]|ORD|SHS|COMMON|CAP STK|STOCK", regex=True) & ~cls.str.contains("PFD|PREF|WT|WARR|NOTE|DEB|RIGHT|UNIT|PUT|CALL|ETF|FUND")]
    names["key"] = names["NAMEOFISSUER"].map(_norm)
    uni = pd.read_parquet(PROC / "universe.parquet")
    uni["key"] = uni["sec_name"].map(_norm)
    m = names.merge(uni[["key", "cik"]].drop_duplicates("key"), on="key")
    m = m[["CUSIP", "cik"]].drop_duplicates()
    m.to_parquet(dest, index=False)
    return m


def inst_features(dec_dates, ciks, shares):
    f = PROC / "inst_ownership_cusip.parquet"
    out = {k: pd.DataFrame(np.nan, index=dec_dates, columns=ciks) for k in ["io_ratio", "io_breadth_chg", "io_holders_chg", "io_hhi"]}
    if not f.exists():
        return out
    agg = pd.read_parquet(f)
    agg["CUSIP"] = agg["CUSIP"].astype(str).str.zfill(9).str.upper()
    agg = agg.merge(cusip_map(), on="CUSIP")
    # keep only CUSIPs sharing the issuer's dominant 6-character issuer code (drops filer typos and name collisions)
    agg["prefix"] = agg["CUSIP"].str[:6]
    dom = agg.groupby(["cik", "prefix"])["n_holders"].sum().reset_index().sort_values("n_holders").drop_duplicates("cik", keep="last")
    agg = agg.merge(dom[["cik", "prefix"]], on=["cik", "prefix"])
    # one row per issuer-period: the most widely held class; shares summed across classes
    agg = agg.sort_values("n_holders", ascending=False)
    g = agg.groupby(["cik", "period"])
    per = g.first()[["n_holders", "hhi", "d_breadth", "available_date"]].join(g["inst_shares"].sum()).reset_index().sort_values(["cik", "period"])
    per["holders_chg"] = np.log(per["n_holders"]) - np.log(per.groupby("cik")["n_holders"].shift(1))
    t = pd.DatetimeIndex(dec_dates).to_numpy()
    col = {c: j for j, c in enumerate(ciks)}
    for cik, gg in per.groupby("cik"):
        j = col.get(cik)
        if j is None:
            continue
        pos = np.searchsorted(gg["available_date"].to_numpy(), t, side="right") - 1
        ok = pos >= 0
        for k, src in [("io_breadth_chg", "d_breadth"), ("io_holders_chg", "holders_chg"), ("io_hhi", "hhi"), ("io_ratio", "inst_shares")]:
            out[k].iloc[ok, j] = gg[src].to_numpy()[pos[ok]]
    out["io_ratio"] = out["io_ratio"] / shares.where(shares > 0)
    out["io_ratio"] = out["io_ratio"].where(out["io_ratio"] < 2)  # >200% implies a mapping/units error
    return out


def short_features(dec_dates, syms, shares):
    f = PROC / "short_interest.parquet"
    out = {k: pd.DataFrame(np.nan, index=dec_dates, columns=syms) for k in ["si_ratio", "si_days_to_cover", "si_change"]}
    if not f.exists():
        return out
    si = pd.read_parquet(f)
    si["sym"] = si["symbolCode"].str.replace(".", "-", regex=False)
    t = pd.DatetimeIndex(dec_dates).to_numpy()
    col = {s: j for j, s in enumerate(syms)}
    for s, g in si.sort_values("settlement_date").groupby("sym"):
        j = col.get(s)
        if j is None:
            continue
        g = g.drop_duplicates("settlement_date", keep="last")
        pos = np.searchsorted(g["available_date"].to_numpy(), t, side="right") - 1
        ok = pos >= 0
        cur = g["currentShortPositionQuantity"].astype(float).to_numpy()
        out["si_ratio"].iloc[ok, j] = cur[pos[ok]]
        out["si_days_to_cover"].iloc[ok, j] = g["daysToCoverQuantity"].astype(float).to_numpy()[pos[ok]]
        chg = np.concatenate([[np.nan], np.diff(cur)])
        out["si_change"].iloc[ok, j] = chg[pos[ok]]
    sh = shares.where(shares > 0)
    out["si_ratio"] = out["si_ratio"] / sh
    out["si_change"] = out["si_change"] / sh
    return out


def earnings_recent(dec_idx, cal, ciks):
    f = PROC / "8k_items.parquet"
    out = np.zeros((len(dec_idx), len(ciks)), dtype="float32")
    if not f.exists():
        return pd.DataFrame(out, index=cal[dec_idx], columns=ciks)
    k8 = pd.read_parquet(f, columns=["cik", "available_date", "items"])
    k8 = k8[k8["items"].fillna("").str.contains(r"\b2\.02\b")].dropna(subset=["available_date"])
    col = {c: j for j, c in enumerate(ciks)}
    for cik, g in k8.groupby("cik"):
        j = col.get(cik)
        if j is None:
            continue
        s = np.sort(cal.searchsorted(g["available_date"].to_numpy(), side="left"))  # first session at/after availability
        k = np.searchsorted(s, dec_idx, side="right")  # releases filed on/before t
        last = np.where(k > 0, s[np.maximum(k - 1, 0)], -10 ** 9)
        out[:, j] = ((dec_idx - last) <= 5).astype("float32")
    return pd.DataFrame(out, index=cal[dec_idx], columns=ciks)


# ---------------------------------------------------------------- sector/macro (date-level series)
def _series(sym, cal, col="close"):
    f = RAW / "prices" / f"{sym}.parquet"
    if not f.exists():
        return pd.Series(np.nan, index=cal)
    d = pd.read_parquet(f, columns=["date", col]).drop_duplicates("date").set_index("date")[col]
    return d.reindex(cal).ffill(limit=3)


def macro_state(cal):
    """Date-level macro and sector series on the trading calendar (available-date aligned)."""
    from data_macro import daily
    m = daily(cal)
    s = pd.DataFrame(index=cal)
    s["dy10_63"] = m["DGS10"] - m["DGS10"].shift(63)
    s["dbaa_63"] = m["BAA10Y"] - m["BAA10Y"].shift(63)
    s["dslope_63"] = m["T10Y3M"] - m["T10Y3M"].shift(63)
    s["semi_ip_yoy"] = np.log(m["IPG3344S"]) - np.log(m["IPG3344S"].shift(252))
    s["semi_orders_3m"] = np.log(m["A34SNO"]) - np.log(m["A34SNO"].shift(63))
    crack = (2 * m["DGASNYH"] * 42 + m["DHOILNYH"] * 42) / 3 - m["DCOILWTICO"]
    ch = crack - crack.shift(63)
    s["crack_chg_z"] = ch / ch.rolling(756, min_periods=252).std()
    # EIA crude stocks vs 5-year same-week mean, z-scored by the 5-year same-week std (all data known at t)
    mp = pd.read_parquet(PROC / "macro_pit.parquet")
    cr = mp[mp["series"] == "crude_stocks_ex_spr"].sort_values("obs_date")
    cr["woy"] = cr["obs_date"].dt.isocalendar().week.astype(int)
    z = []
    for r in cr.itertuples():
        h = cr[(cr["woy"] == r.woy) & (cr["obs_date"] < r.obs_date) & (cr["obs_date"] >= r.obs_date - pd.Timedelta(days=5 * 366))]["value"]
        z.append((r.value - h.mean()) / h.std() if len(h) >= 4 and h.std() > 0 else np.nan)
    cr["z"] = z
    zs = cr.set_index("available_date")["z"]
    zs = zs[~zs.index.duplicated(keep="last")]
    s["crude_inv_z"] = zs.reindex(zs.index.union(cal)).ffill().reindex(cal)
    cot = RAW / "alt" / "cot.parquet"
    if cot.exists():
        c = pd.read_parquet(cot)
        w = c[c["commodity"] == "wti"].sort_values("report_date")
        w["z"] = (w["mm_net_pct_oi"] - w["mm_net_pct_oi"].rolling(156, min_periods=52).mean()) / w["mm_net_pct_oi"].rolling(156, min_periods=52).std()
        wz = w.set_index("available_date")["z"]
        wz = wz[~wz.index.duplicated(keep="last")]
        s["wti_cot_z"] = wz.reindex(wz.index.union(cal)).ffill().reindex(cal)
    fut = RAW / "alt" / "futures.parquet"
    if fut.exists():
        f = pd.read_parquet(fut)
        for cmd in ["wti", "natgas", "copper", "gold", "corn"]:
            x = f[f["commodity"] == cmd].set_index("date")["close"]
            x = x[~x.index.duplicated()].reindex(cal).ffill(limit=3)
            s[f"{cmd}_ret63"] = np.log(x / x.shift(63))
    vix, vix3m, skew = _series("^VIX", cal), _series("^VIX3M", cal), _series("^SKEW", cal)
    s["vix_term_slope"] = vix3m / vix - 1
    from macro_surprise import daily_z
    s = s.join(daily_z(cal))
    s["skew_z"] = (skew - skew.rolling(756, min_periods=252).mean()) / skew.rolling(756, min_periods=252).std()
    return s


def consensus_surprise(dec_dates, syms):
    """Latest Alpha Vantage surprise vs consensus available at t (pre-market: same day; post-market: next day)."""
    f = PROC / "av_earnings.parquet"
    out = pd.DataFrame(np.nan, index=dec_dates, columns=syms)
    if not f.exists():
        return out
    a = pd.read_parquet(f, columns=["symbol", "available_date", "surprisePercentage"]).dropna()
    t = pd.DatetimeIndex(dec_dates).to_numpy()
    for sym, g in a.sort_values("available_date").groupby("symbol"):
        if sym not in out.columns:
            continue
        av = g["available_date"].to_numpy()
        pos = np.searchsorted(av, t, side="right") - 1
        ok = (pos >= 0)
        v = np.full(len(t), np.nan)
        v[ok] = g["surprisePercentage"].clip(-100, 100).to_numpy()[pos[ok]]
        age = np.full(len(t), np.inf)
        age[ok] = (t[ok] - av[pos[ok]]) / np.timedelta64(1, "D")
        v[age > 200] = np.nan
        out[sym] = v
    return out


def snapshot_features(dec_dates, syms, vol60):
    """Per-stock options / estimate snapshots (collected nightly), latest snapshot no older than 5 days."""
    out = {k: pd.DataFrame(np.nan, index=dec_dates, columns=syms) for k in SNAPSHOT_FEATURES}
    t = pd.DatetimeIndex(dec_dates)
    for kind, cols in [("options", {"opt_term_slope": "term_slope", "opt_skew_rr": "risk_reversal_25d", "opt_pc_ratio": "pc_volume_ratio", "_iv30": "atm_iv_30"}),
                       ("estimates", {"est_rev_1m": "yr_rev_1m", "est_dispersion": "dispersion"})]:
        files = sorted((RAW / "collect" / kind).glob("*.parquet"))
        if not files:
            continue
        snap = pd.concat([pd.read_parquet(f, columns=["symbol", "snapshot_date"] + list(cols.values())) for f in files], ignore_index=True)
        for feat, col in cols.items():
            w = snap.pivot_table(index="snapshot_date", columns="symbol", values=col, aggfunc="last").reindex(columns=syms)
            w = w.reindex(w.index.union(t)).ffill(limit=5).reindex(t)
            if feat == "_iv30":
                out["opt_iv_rv"] = w / vol60.where(vol60 > 0) - 1
            else:
                out[feat] = w
    return out


def health_features(dec_dates, ciks, flags):
    out = {k: pd.DataFrame(np.nan, index=dec_dates, columns=ciks) for k in ["bio_ph3_trials", "bio_fda_appr_365"]}
    t = pd.DatetimeIndex(dec_dates)
    pharma = [c for c in ciks if flags.get(c, {}).get("pharma")]
    tr = PROC / "clinical_trials.parquet"
    if tr.exists():
        d = pd.read_parquet(tr)
        d = d[d["phase"].fillna("").str.contains("PHASE3")]
        for cik, g in d.groupby("cik"):
            if cik not in out["bio_ph3_trials"].columns:
                continue
            fp, pc = g["first_posted"].to_numpy(), g["primary_completion"].fillna(pd.Timestamp("2100-01-01")).to_numpy()
            out["bio_ph3_trials"][cik] = [(fp <= x).sum() - ((pc < x) & (fp <= x)).sum() for x in t.to_numpy()]
    fd = PROC / "fda_approvals.parquet"
    if fd.exists():
        a = pd.read_parquet(fd)
        for cik, g in a.groupby("cik"):
            if cik not in out["bio_fda_appr_365"].columns:
                continue
            ad = np.sort(g["action_date"].dropna().to_numpy())
            out["bio_fda_appr_365"][cik] = np.searchsorted(ad, t.to_numpy(), side="right") - np.searchsorted(ad, (t - pd.Timedelta(days=365)).to_numpy(), side="right")
    non = [c for c in ciks if c not in set(pharma)]
    for k in out:
        out[k][non] = np.nan
        out[k][pharma] = out[k][pharma].fillna(0)
    return out


# ---------------------------------------------------------------- panel
def active_symbols(min_dollar_adv):
    """Symbols whose 60-session dollar volume ever reaches the profile minimum (cheap pre-scan)."""
    uni = pd.read_parquet(PROC / "universe.parquet")
    out = []
    for sym in uni["yahoo"]:
        f = RAW / "prices" / f"{sym}.parquet"
        if f.exists():
            d = pd.read_parquet(f, columns=["close", "volume"])
            if (d["close"] * d["volume"]).rolling(ADV_WINDOW, min_periods=40).mean().max() >= min_dollar_adv:
                out.append(sym)
    return out


def build_panel_chunked(include_labels=True, last_n_dates=None, profile=None, chunk=900):
    """Every feature here is a per-security time series (cross-sectional transforms happen in research.py), so the
    panel can be built in symbol chunks and concatenated without changing any value; this bounds memory."""
    prof = C.PROFILES[profile] if profile else C.PROFILE
    syms = active_symbols(prof["min_dollar_adv"])
    parts, metas = [], []
    for a in range(0, len(syms), chunk):
        p, m = build_panel(include_labels, last_n_dates, prof["name"], symbols=set(syms[a:a + chunk]))
        parts.append(p)
        metas.append(m)
        print(f"chunk {a // chunk + 1}/{-(-len(syms) // chunk)}: {m['rows']} rows", flush=True)
    panel = pd.concat(parts, ignore_index=True).sort_values(["date", "symbol"], ignore_index=True)
    meta = metas[0] | {"symbols_with_prices": sum(m["symbols_with_prices"] for m in metas), "rows": len(panel), "chunks": len(metas)}
    return panel, meta


def build_panel(include_labels=True, last_n_dates=None, profile=None, symbols=None):
    prof = C.PROFILES[profile] if profile else C.PROFILE
    P, sym2cik = load_prices(prof["min_dollar_adv"], symbols=symbols)
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
    lab = labels(P, dec_idx) if include_labels else {}
    mret = _macro_returns(cal)
    pf = price_features(P, dec_idx, mret)
    ac = P["adjclose"].iloc[dec_idx]
    close_full, adj_full = P["close"].to_numpy(), P["adjclose"].to_numpy()
    for k in ["high", "low", "volume", "adj_open"]:
        del P[k]

    fdf = load_facts(ciks)
    # foreign private issuers (20-F/40-F): XBRL counts ordinary shares while US prices are ADR prices with unknown
    # ratios, so market cap (and every per-share/market-cap feature) is unreliable -> excluded from eligibility
    last_annual = fdf[fdf["form"].str.match(r"^(10-K|20-F|40-F)")].sort_values("filed").drop_duplicates("cik", keep="last")
    foreign = set(last_annual.loc[last_annual["form"].str.startswith(("20-F", "40-F")), "cik"].astype(int))
    fund = fundamentals_asof(dec_dates, ciks, fdf)
    sue, eps_g = eps_surprise(dec_dates, ciks, fdf)
    del fdf
    ins = insider_features(load_insider(ciks), dec_dates, ciks)

    def S(df):  # cik-columned -> symbol-columned (relabels in place: the cik-keyed frames are not reused)
        df.columns = syms
        return df.astype("float32") if df.dtypes.iloc[0] == "float64" else df

    # cover-page shares go stale for some multi-class issuers (dimensioned facts are not in companyfacts): fall back to
    # annual diluted weighted shares when the cover-page count is more than 400 days old
    now = np.repeat(dec_dates.values[:, None], len(ciks), axis=1)
    stale = (pd.DataFrame(now - fund["shares_end"].to_numpy(), index=dec_dates, columns=ciks) > pd.Timedelta(days=400)) | fund["shares"].isna()
    fund["shares"] = fund["shares"].where(~stale | fund["shares_dil"].isna(), fund["shares_dil"])
    fund["shares"] = fund["shares"].where(~(stale & fund["shares_dil"].isna()))  # stale with no fallback -> unknown, not wrong
    fund["shares_end"] = fund["shares_end"].where(~stale | fund["shares_dil"].isna(), pd.DataFrame(now - np.timedelta64(1, "D"), index=dec_dates, columns=ciks))
    shares = S(fund["shares"])
    sh_end = S(fund["shares_end"])
    mcap = pd.DataFrame(np.nan, index=dec_dates, columns=syms)
    for j, s in enumerate(syms):  # cover-page shares x close on the shares' as-of date, rolled forward with total return
        e = sh_end[s]
        okm = e.notna() & shares[s].notna()
        if not okm.any():
            continue
        pos = np.clip(cal.searchsorted(pd.DatetimeIndex(e[okm]), side="right") - 1, 0, len(cal) - 1)
        mcap.loc[okm, s] = shares[s][okm].to_numpy() * close_full[pos, j] * (ac[s][okm].to_numpy() / adj_full[pos, j])
    del close_full, adj_full
    F = {k: S(v) for k, v in fund.items() if not k.endswith("_end") or k == "ni_end"}
    gp = F["gp"].fillna(F["rev"] - F["cogs"])
    assets = F["assets"].where(F["assets"] > 0)
    rev = F["rev"].where(F["rev"] > 0)
    feats = {k: pf[k] for k in PRICE_FEATURES}
    feats["log_mcap"] = np.log(mcap.where(mcap > 0))
    feats["turnover"] = pf["adv_shares"] / shares.where(shares > 0)
    feats["earn_yield"] = F["ni"] / mcap
    feats["book_to_mkt"] = F["equity"] / mcap
    feats["fcf_yield"] = (F["cfo"] - F["capex"].fillna(0)) / mcap
    feats["gross_prof"] = gp / assets
    feats["roa"] = F["ni"] / assets
    cap = (F["equity"] + F["ltd"].fillna(0)).where(lambda x: x > 0)
    feats["roic"] = F["opinc"] * 0.79 / cap
    feats["accruals"] = (F["ni"] - F["cfo"]) / assets
    feats["leverage"] = F["liab"] / assets
    feats["asset_growth"] = assets / F["assets_prev"].where(F["assets_prev"] > 0) - 1
    feats["capex_intensity"] = F["capex"] / rev
    feats["share_issuance"] = shares / F["shares_prev"].where(F["shares_prev"] > 0) - 1
    feats["sue"] = S(sue)
    feats["eps_growth"] = S(eps_g)
    rev_p = F["rev_prev"].where(F["rev_prev"] > 0)
    gp_prev = F["gp_prev"]
    feats["gm_change"] = gp / rev - gp_prev / rev_p
    feats["op_margin_change"] = F["opinc"] / rev - F["opinc_prev"] / rev_p
    feats["inv_to_sales_chg"] = F["inv"] / rev - F["inv_prev"] / rev_p
    feats["earn_recent"] = S(earnings_recent(dec_idx, cal, ciks))

    I = {k: S(v) for k, v in ins.items()}
    feats["ins_net_90"] = (I["buy_usd_90"] - I["sell_usd_90"]) / mcap
    feats["ins_net_180"] = (I["buy_usd_180"] - I["sell_usd_180"]) / mcap
    feats["ins_buyers_90"] = I["ins_buyers_90"]
    feats["ins_sellers_90"] = I["ins_sellers_90"]
    feats["ins_officer_buy_180"] = I["ins_officer_buy_180"]
    feats["ins_dir_buy_90"] = I["dir_buy_usd_90"] / mcap
    feats["ins_sponsor_sell_90"] = I["sponsor_sell_usd_90"] / mcap
    feats["ins_purchase_intensity"] = I["ins_purchase_intensity"]
    feats["ins_conviction"] = I["ins_conviction"]
    feats["ins_opp_net_90"] = I["opp_net_usd_90"] / mcap
    feats["ins_routine_net_90"] = I["routine_net_usd_90"] / mcap
    feats["ins_surprise"] = ((I["buy_usd_90"] - I["sell_usd_90"]) - I["net_usd_365"] * 90 / 365) / mcap
    feats["ins_buy_x_drawdown"] = I["ins_buyers_90"] * pf["drawdown_252"]
    feats["ins_plan_sale_share"] = (I["plan_sell_usd_180"] / I["all_sell_usd_180"].where(I["all_sell_usd_180"] > 0))

    io = inst_features(dec_dates, ciks, fund["shares"].set_axis(ciks, axis=1))  # cik-keyed for the 13F join
    for k, v in io.items():
        feats[k] = S(v)
    for k, v in short_features(dec_dates, syms, shares).items():
        feats[k] = v

    sic = pd.read_parquet(PROC / "sic.parquet").set_index("cik")["sic"]
    flags = {c: sector_flags(sic.get(c)) for c in ciks}
    fl = {k: pd.Series({s: flags[sym2cik[s]][k] for s in syms}) for k in ["semis", "pharma", "energy", "refiner", "bank", "commodity"]}

    def only(x, flag):  # sector feature: NaN outside its sector
        return x.where(pd.DataFrame(np.broadcast_to(fl[flag].to_numpy(), x.shape), index=x.index, columns=x.columns))

    ms = macro_state(cal).iloc[dec_idx]

    def date_series(v):
        return pd.DataFrame(np.broadcast_to(v.to_numpy()[:, None], (len(dec_idx), len(syms))).copy(), index=dec_dates, columns=syms)
    feats["rates_tilt"] = pf["beta_rates"] * ms["dy10_63"].to_numpy()[:, None]
    feats["credit_tilt"] = pf["beta_credit"] * ms["dbaa_63"].to_numpy()[:, None]
    feats["semi_cycle"] = only(date_series(ms["semi_ip_yoy"]), "semis")
    feats["semi_orders"] = only(date_series(ms["semi_orders_3m"]), "semis")
    cfo = F["cfo"]
    burn = (-cfo).where(cfo < 0)
    feats["bio_cash_runway"] = only((F["cash"] / burn).clip(upper=10).where(cfo < 0, 10.0), "pharma")
    feats["bio_rnd_intensity"] = only(F["rnd"] / rev, "pharma")
    hf = health_features(dec_dates, ciks, flags)
    for k, v in hf.items():
        feats[k] = S(v)
    feats["energy_oil_tilt"] = only(pf["beta_wti"] * ms["wti_ret63"].to_numpy()[:, None], "energy") if "wti_ret63" in ms else None
    feats["energy_inv_surprise"] = only(-pf["beta_wti"] * ms["crude_inv_z"].to_numpy()[:, None], "energy")
    feats["energy_crack_chg"] = only(date_series(ms["crack_chg_z"]), "refiner")
    feats["energy_cot_z"] = only(pf["beta_wti"] * ms["wti_cot_z"].to_numpy()[:, None], "energy") if "wti_cot_z" in ms else None
    loans = F["loans"].where(F["loans"] > 0)
    feats["fin_deposit_growth"] = only(F["deposits"] / F["deposits_prev"].where(F["deposits_prev"] > 0) - 1, "bank")
    feats["fin_nim_proxy"] = only(F["nii"] / assets, "bank")
    feats["fin_provision_ratio"] = only(F["prov"] / loans, "bank")
    feats["fin_slope_tilt"] = only(date_series(ms["dslope_63"]), "bank")
    tilt = sum(pf[f"beta_{c}"].fillna(0) * ms[f"{c}_ret63"].fillna(0).to_numpy()[:, None] for c in ["wti", "natgas", "copper", "gold", "corn"] if f"{c}_ret63" in ms)
    feats["mat_commodity_tilt"] = only(tilt, "commodity")
    if "growth_surprise_z" in ms:
        feats["growth_surprise_tilt"] = pf["beta_252"] * ms["growth_surprise_z"].to_numpy()[:, None]
        feats["inflation_surprise_tilt"] = pf["beta_rates"] * ms["inflation_surprise_z"].to_numpy()[:, None]
    feats["opt_mkt_term_tilt"] = pf["beta_252"] * ms["vix_term_slope"].to_numpy()[:, None]
    feats["opt_mkt_skew_tilt"] = pf["beta_252"] * ms["skew_z"].to_numpy()[:, None]
    feats["sue_consensus"] = consensus_surprise(dec_dates, syms)
    feats.update(snapshot_features(dec_dates, syms, pf["vol_60"]))
    feats = {k: v for k, v in feats.items() if v is not None}

    fund_end = F["ni_end"]
    fund_age = pd.DataFrame(dec_dates.values[:, None] - fund_end.to_numpy(), index=dec_dates, columns=syms) / np.timedelta64(1, "D")
    is_dom = pd.Series({s: sym2cik[s] not in foreign for s in syms})
    elig = is_dom.to_numpy()[None, :] & (mcap >= prof["min_mcap"]) & (pf["adv_shares"] >= prof["min_adv_shares"]) & (pf["adv_dollar"] >= prof["min_dollar_adv"]) & ac.notna()
    sector = {s: sic_sector(sic.get(c)) for s, c in sym2cik.items()}

    long = []
    for i, d in enumerate(dec_dates):
        m = elig.iloc[i].to_numpy()
        if not m.any():
            continue
        row = {"date": d, "symbol": np.array(syms)[m]}
        for k, v in feats.items():
            row[k] = v.iloc[i].to_numpy()[m].astype("float32")
        row["mcap"] = mcap.iloc[i].to_numpy()[m]
        row["adv_shares"] = pf["adv_shares"].iloc[i].to_numpy()[m].astype("float32")
        row["adv_dollar"] = pf["adv_dollar"].iloc[i].to_numpy()[m].astype("float32")
        row["fund_age_days"] = fund_age.iloc[i].to_numpy()[m].astype("float32")
        for h in lab:
            row[f"y_{h}"] = lab[h][i][m]
        long.append(pd.DataFrame(row))
    panel = pd.concat(long, ignore_index=True)
    panel["sector"] = panel["symbol"].map(sector)
    panel["cik"] = panel["symbol"].map(sym2cik)
    # observation quality: share of the core features that are present (missing-data status, spec 2.2)
    core = [k for k in FEATURES if k in panel and FEATURES[k][0] not in ("sector_tech", "sector_health", "sector_energy", "sector_fin", "sector_materials", "positioning", "institutional")]
    panel["quality_score"] = panel[core].notna().mean(axis=1).astype("float32")
    meta = {"profile": prof["name"], "feature_set_version": C.FEATURE_SET_VERSION, "decision_dates": len(dec_dates), "first": str(dec_dates[0].date()),
            "last": str(dec_dates[-1].date()), "symbols_with_prices": len(syms), "rows": len(panel),
            "features": [k for k in FEATURES if k in panel]}
    return panel, meta


if __name__ == "__main__":
    _lock = __import__("config").hold_lock()
    import sys
    prof = sys.argv[1] if len(sys.argv) > 1 else C.PROFILE["name"]
    panel, meta = build_panel_chunked(profile=prof)
    panel.to_parquet(PROC / f"panel_{prof}.parquet", index=False)
    (PROC / f"panel_meta_{prof}.json").write_text(json.dumps(meta, indent=1))
    print(meta)
    print(panel.groupby(panel["date"].dt.year)["symbol"].count().to_string())
