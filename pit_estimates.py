"""Timestamped consensus observations; earnings history is not an estimate revision archive."""
import numpy as np
import pandas as pd


def validate(frame):
    fields = {"security_id", "fiscal_period", "consensus_eps", "available_at_utc", "observed_at_utc", "timestamp_basis", "source"}
    if not fields <= set(frame) or frame[list(fields)].isna().any().any():
        raise ValueError("estimate vintages require security, fiscal period, value, provenance and availability timestamps")
    d = frame.copy()
    for field in ["security_id", "fiscal_period", "source"]:
        d[field] = d[field].astype("string").str.strip()
        if d[field].eq("").any():
            raise ValueError(f"estimate {field} cannot be blank")
    for field in ["available_at_utc", "observed_at_utc"]:
        raw = d[field].astype("string").str.strip()
        if not raw.str.contains(r"(?:Z|[+-][0-9]{2}:?[0-9]{2})$", regex=True).all():
            raise ValueError(f"estimate {field} requires an explicit timezone")
        d[field] = pd.to_datetime(raw, utc=True, format="mixed", errors="raise")
    if (d["observed_at_utc"] > pd.Timestamp.now(tz="UTC")).any():
        raise ValueError("estimate observation timestamp is in the future")
    if not d["timestamp_basis"].isin(["vendor_archive", "observed_snapshot"]).all():
        raise ValueError("estimate timestamp basis must be a vendor archive or an observed snapshot")
    snapshot = d["timestamp_basis"].eq("observed_snapshot")
    if (snapshot & (d["available_at_utc"] < d["observed_at_utc"])).any():
        raise ValueError("a current estimate snapshot cannot be backdated")
    d["consensus_eps"] = pd.to_numeric(d["consensus_eps"], errors="raise")
    if not np.isfinite(d["consensus_eps"]).all() or d.duplicated(["security_id", "fiscal_period", "available_at_utc"]).any():
        raise ValueError("invalid or duplicate consensus observations")
    return d


def revision_series(observations, decisions, days=31):
    """Same-fiscal-period revisions on timezone-aware decision timestamps."""
    decisions = pd.DatetimeIndex(decisions)
    if decisions.tz is None or decisions.hasnans:
        raise ValueError("estimate decisions require valid timezone-aware timestamps")
    if days <= 0:
        raise ValueError("revision window must be positive")
    result = np.full(len(decisions), np.nan)
    if observations.empty:
        return result
    d = observations.sort_values("available_at_utc", kind="stable").reset_index(drop=True)
    times = pd.DatetimeIndex(d["available_at_utc"])
    lookback = pd.Timedelta(days=days)
    latest = times.searchsorted(decisions, side="right") - 1
    # If the most recent timestamp covers several periods, the feed needs
    # an explicit target selection rule before a unique revision can be used.
    unique = ~times.duplicated(keep=False)
    for _, period in d.groupby("fiscal_period", sort=False):
        ids = np.flatnonzero(np.isin(latest, period.index))
        if not len(ids):
            continue
        ids = ids[unique[latest[ids]] & (decisions[ids] - times[latest[ids]] <= lookback)]
        prior_times = pd.DatetimeIndex(period["available_at_utc"])
        prior = prior_times.searchsorted(decisions[ids] - lookback, side="right") - 1
        have = prior >= 0
        ids, prior = ids[have], prior[have]
        old = period["consensus_eps"].to_numpy()[prior]
        valid = (decisions[ids] - prior_times[prior] <= lookback * 3) & (np.abs(old) >= 1e-8)
        ids, old = ids[valid], old[valid]
        result[ids] = (d["consensus_eps"].to_numpy()[latest[ids]] - old) / np.abs(old)
    return result


def revision(frame, security_id, decision, days=31):
    decision = pd.Timestamp(decision)
    decision = decision.tz_localize("UTC") if decision.tzinfo is None else decision
    return float(revision_series(frame[frame["security_id"].eq(security_id)], [decision], days)[0])


def revision_panel(frame, dates, security_ids):
    """Use only estimates visible by midnight New York time on each panel date."""
    dates = pd.DatetimeIndex(dates)
    decisions = dates.tz_localize("America/New_York")
    out = pd.DataFrame(np.nan, index=dates, columns=security_ids)
    for sid, observations in frame[frame["security_id"].isin(security_ids)].groupby("security_id", sort=False):
        out[sid] = revision_series(observations, decisions)
    return out


def import_csv(path):
    """Import an explicit timestamped export; never infer historical availability."""
    import hashlib
    from pathlib import Path
    import config as C
    payload = Path(path).read_bytes()
    import io
    frame = validate(pd.read_csv(io.BytesIO(payload), dtype={"security_id": "string", "fiscal_period": "string"}))
    if frame.empty:
        raise ValueError("estimate archive has no observations")
    archive = C.RAW / "estimate_vintages"
    archive.mkdir(parents=True, exist_ok=True)
    target = archive / (hashlib.sha256(payload).hexdigest() + ".csv")
    if not target.exists():
        with target.open("xb") as stream:
            stream.write(payload)
    C.PROC.mkdir(parents=True, exist_ok=True)
    output = C.PROC / "estimate_vintages.parquet"
    temporary = output.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(output)
    return output


if __name__ == "__main__":
    import argparse
    import config as C
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="complete timestamped consensus export, not current estimates")
    args = parser.parse_args()
    _lock = C.hold_lock()
    print(import_csv(args.csv))
