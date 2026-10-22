"""Timestamped consensus observations; earnings history is not an estimate revision archive."""
import numpy as np
import pandas as pd


def validate(frame):
    fields = {"security_id", "fiscal_period", "consensus_eps", "available_at_utc", "observed_at_utc", "timestamp_basis", "source"}
    if not fields <= set(frame) or frame[list(fields)].isna().any().any():
        raise ValueError("estimate vintages require security, fiscal period, value, provenance and availability timestamps")
    d = frame.copy()
    for field in ["available_at_utc", "observed_at_utc"]:
        d[field] = pd.to_datetime(d[field], utc=True, errors="raise")
    if not d["timestamp_basis"].isin(["vendor_archive", "observed_snapshot"]).all():
        raise ValueError("estimate timestamp basis must be a vendor archive or an observed snapshot")
    snapshot = d["timestamp_basis"].eq("observed_snapshot")
    if (snapshot & (d["available_at_utc"] < d["observed_at_utc"])).any():
        raise ValueError("a current estimate snapshot cannot be backdated")
    d["consensus_eps"] = pd.to_numeric(d["consensus_eps"], errors="raise")
    if not np.isfinite(d["consensus_eps"]).all() or d.duplicated(["security_id", "fiscal_period", "available_at_utc"]).any():
        raise ValueError("invalid or duplicate consensus observations")
    return d


def revision(frame, security_id, decision, days=31):
    decision = pd.Timestamp(decision)
    decision = decision.tz_localize("UTC") if decision.tzinfo is None else decision.tz_convert("UTC")
    d = frame[(frame["security_id"] == security_id) & (frame["available_at_utc"] <= decision)]
    if d.empty:
        return np.nan
    latest = d.sort_values("available_at_utc").iloc[-1]
    same = d[(d["fiscal_period"] == latest["fiscal_period"]) & (d["available_at_utc"] <= decision - pd.Timedelta(days=days))]
    if same.empty or decision - latest["available_at_utc"] > pd.Timedelta(days=days):
        return np.nan
    prior = same.sort_values("available_at_utc").iloc[-1]
    if decision - prior["available_at_utc"] > pd.Timedelta(days=days * 3) or abs(prior["consensus_eps"]) < 1e-8:
        return np.nan
    return float((latest["consensus_eps"] - prior["consensus_eps"]) / abs(prior["consensus_eps"]))


def import_csv(path):
    """Import an explicit timestamped export; never infer historical availability."""
    import hashlib
    from pathlib import Path
    import config as C
    payload = Path(path).read_bytes()
    import io
    frame = validate(pd.read_csv(io.BytesIO(payload)))
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
