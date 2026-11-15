"""Convert a CodeBook monthly FY1 consensus export into the pit_estimates.py import format.

Input:  mosaic_est_hist.csv from CodeBook with columns Instrument, Calc Date, Earnings Per Share - Mean,
        Period End Date, Earnings Per Share - Standard Deviation, EPS Number of Estimates
        (TR.EPSMean with .calcdate and .periodenddate, Period=FY1, Frq=M).
Output: CSV with security_id, fiscal_period, consensus_eps, available_at_utc, observed_at_utc,
        timestamp_basis, source, plus eps_std and n_estimates for the dispersion feature.

Rules (prereg/lseg_analyst_flow_v1.json):
  * security_id comes from the active historical bundle's RIC map; unmapped RICs are dropped and counted.
  * A value is available from the end of its LSEG calc date in New York (23:59:59 America/New_York),
    so a decision on the calc date itself never sees it. Rows without a calc date or period end are dropped.
  * timestamp_basis = vendor_archive. LSEG restates history for later splits, so levels are on today's
    share basis; revisions within one fiscal period are unaffected, levels must not be compared with
    as-reported SEC figures.
Usage: python lseg_estimates.py mosaic_est_hist.csv OUTPUT.csv [OBSERVED_AT_UTC]
       python lseg_estimates.py analyst mosaic_pt_hist.csv mosaic_rec_hist.csv   -> processed/lseg_analyst.parquet
       python lseg_estimates.py etf mosaic_etf_flows.csv                          -> processed/etf_flows.parquet
"""
import sys
from datetime import datetime, timezone

import pandas as pd

from historical_data import active_bundle

SOURCE = "lseg_codebook:TR.EPSMean FY1 monthly"


def convert(path, observed_at=None):
    bundle = active_bundle()
    if bundle is None:
        raise ValueError("an active historical bundle is required for the RIC to security_id map")
    master = pd.read_parquet(bundle / "security_master.parquet", columns=["security_id", "ric"])
    ric_map = dict(zip(master["ric"], master["security_id"]))
    raw = pd.read_csv(path)
    d = raw.rename(columns={"Instrument": "ric", "Calc Date": "calc", "Earnings Per Share - Mean": "consensus_eps",
                            "Period End Date": "period", "Earnings Per Share - Standard Deviation": "eps_std",
                            "EPS Number of Estimates": "n_estimates"})
    stats = {"rows_in": len(d)}
    d["security_id"] = d["ric"].map(ric_map)
    stats["rows_unmapped_ric"] = int(d["security_id"].isna().sum())
    d["calc"] = pd.to_datetime(d["calc"], errors="coerce")
    d["period"] = pd.to_datetime(d["period"], errors="coerce")
    d["consensus_eps"] = pd.to_numeric(d["consensus_eps"], errors="coerce")
    keep = d["security_id"].notna() & d["calc"].notna() & d["period"].notna() & d["consensus_eps"].notna()
    stats["rows_missing_value_or_dates"] = int((~keep & d["security_id"].notna()).sum())
    d = d[keep].copy()
    end_of_day = (d["calc"] + pd.Timedelta(hours=23, minutes=59, seconds=59)).dt.tz_localize("America/New_York")
    d["available_at_utc"] = end_of_day.dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    d["observed_at_utc"] = observed_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    d["fiscal_period"] = d["period"].dt.strftime("FY%Y-%m-%d")
    d["timestamp_basis"] = "vendor_archive"
    d["source"] = SOURCE
    before = len(d)
    d = d.drop_duplicates(["security_id", "fiscal_period", "available_at_utc"], keep="last")
    stats["rows_duplicate"] = before - len(d)
    stats["rows_out"] = len(d)
    stats["securities"] = int(d["security_id"].nunique())
    cols = ["security_id", "fiscal_period", "consensus_eps", "available_at_utc", "observed_at_utc", "timestamp_basis",
            "source", "eps_std", "n_estimates"]
    return d[cols], stats


def _available_at(calc):
    """End of the LSEG calc date in New York, as a UTC timestamp."""
    end_of_day = (pd.to_datetime(calc) + pd.Timedelta(hours=23, minutes=59, seconds=59)).dt.tz_localize("America/New_York")
    return end_of_day.dt.tz_convert("UTC")


def convert_analyst(pt_csv, rec_csv):
    """Price-target and recommendation means -> long table (security_id, measure, value, available_at_utc)."""
    bundle = active_bundle()
    if bundle is None:
        raise ValueError("an active historical bundle is required for the RIC to security_id map")
    master = pd.read_parquet(bundle / "security_master.parquet", columns=["security_id", "ric"])
    ric_map = dict(zip(master["ric"], master["security_id"]))
    parts, stats = [], {}
    for measure, path, column in [("pt", pt_csv, "Price Target - Mean"), ("rec", rec_csv, "Recommendation - Mean (1-5)")]:
        d = pd.read_csv(path).rename(columns={"Instrument": "ric", "Calc Date": "calc", column: "value"})
        d["security_id"] = d["ric"].map(ric_map)
        d["value"] = pd.to_numeric(d["value"], errors="coerce")
        d["calc"] = pd.to_datetime(d["calc"], errors="coerce")
        n = len(d)
        d = d.dropna(subset=["security_id", "calc", "value"])
        d = d[d["value"] > 0] if measure == "pt" else d[d["value"].between(1, 5)]
        d = d.drop_duplicates(["security_id", "calc"], keep="last")
        d["available_at_utc"] = _available_at(d["calc"])
        d["measure"] = measure
        stats[measure] = {"rows_in": n, "rows_out": len(d), "securities": int(d["security_id"].nunique())}
        parts.append(d[["security_id", "measure", "value", "calc", "available_at_utc"]])
    out = pd.concat(parts, ignore_index=True)
    out["source"] = "lseg_codebook:TR.PriceTargetMean/TR.RecMean monthly"
    return out, stats


def convert_etf(etf_csv):
    """ETF shares outstanding and NAV -> (etf, date, shares, nav). Ticker suffixes (.O/.K) are removed."""
    d = pd.read_csv(etf_csv)
    cols = list(d.columns)
    # columns: Instrument, Date, Outstanding Shares, Date, NAV, Price Close (pandas suffixes the repeated Date)
    d = d.rename(columns={cols[0]: "ric", cols[1]: "date", cols[2]: "shares", cols[3]: "nav_date", cols[4]: "nav"})
    d["etf"] = d["ric"].astype(str).str.split(".").str[0]
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["nav_date"] = pd.to_datetime(d["nav_date"], errors="coerce")
    d["shares"] = pd.to_numeric(d["shares"], errors="coerce")
    d["nav"] = pd.to_numeric(d["nav"], errors="coerce")
    n = len(d)
    # LSEG dates shares and NAV separately; attach to each share count the latest NAV dated on or before it
    shares = d[d["date"].notna() & (d["shares"] > 0)][["etf", "date", "shares"]].drop_duplicates(["etf", "date"], keep="last")
    navs = d[d["nav_date"].notna() & (d["nav"] > 0)][["etf", "nav_date", "nav"]].drop_duplicates(["etf", "nav_date"], keep="last")
    out = pd.merge_asof(shares.sort_values("date"), navs.sort_values("nav_date"), left_on="date", right_on="nav_date",
                        by="etf", direction="backward", tolerance=pd.Timedelta(days=7))
    out = out.dropna(subset=["nav"]).sort_values(["etf", "date"])
    stats = {"rows_in": n, "rows_out": len(out), "etfs": int(out["etf"].nunique()),
             "nav_lag_days_max": int((out["date"] - out["nav_date"]).dt.days.max())}
    return out[["etf", "date", "shares", "nav"]].reset_index(drop=True), stats


if __name__ == "__main__":
    import config as C
    if sys.argv[1] == "analyst":
        frame, stats = convert_analyst(sys.argv[2], sys.argv[3])
        frame.to_parquet(C.PROC / "lseg_analyst.parquet", index=False)
    elif sys.argv[1] == "etf":
        frame, stats = convert_etf(sys.argv[2])
        frame.to_parquet(C.PROC / "etf_flows.parquet", index=False)
    else:
        frame, stats = convert(sys.argv[1], sys.argv[3] if len(sys.argv) > 3 else None)
        frame.to_csv(sys.argv[2], index=False)
    print(stats)
