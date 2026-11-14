"""Point-in-time futures curves from individual contract settlements."""
import argparse
import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

FIELDS = ["commodity", "contract_id", "session", "expiry", "settlement",
          "available_at_utc", "currency", "price_unit", "source"]


def validate(frame):
    missing = set(FIELDS) - set(frame)
    if missing or frame[FIELDS].isna().any().any():
        raise ValueError(f"contract settlements require all fields, missing columns: {sorted(missing)}")
    d = frame[FIELDS].copy()
    for column in ["commodity", "contract_id", "currency", "price_unit", "source"]:
        d[column] = d[column].astype(str).str.strip()
        if d[column].eq("").any():
            raise ValueError(f"blank {column}")
    for column in ["session", "expiry"]:
        d[column] = pd.to_datetime(d[column], format="%Y-%m-%d", errors="raise")
        if not d[column].eq(d[column].dt.normalize()).all() or d[column].dt.tz is not None:
            raise ValueError(f"{column} requires a local calendar date")
    raw = d["available_at_utc"].astype(str)
    if not raw.str.contains(r"(?:Z|[+-]\d\d:\d\d)$").all():
        raise ValueError("availability requires an explicit timezone")
    d["available_at_utc"] = pd.to_datetime(raw, utc=True, format="mixed", errors="raise")
    available_day = d["available_at_utc"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    if available_day.lt(d["session"]).any():
        raise ValueError("settlement availability precedes its session")
    d["settlement"] = pd.to_numeric(d["settlement"], errors="raise")
    if not np.isfinite(d["settlement"]).all() or d["expiry"].lt(d["session"]).any():
        raise ValueError("invalid settlement or expiry")
    keys = ["commodity", "contract_id", "session", "available_at_utc"]
    if d.duplicated(keys).any():
        raise ValueError("duplicate contract settlement vintage")
    definitions = d.groupby(["commodity", "contract_id"])[["expiry", "currency", "price_unit"]].nunique()
    if definitions.gt(1).any().any():
        raise ValueError("contract expiry, currency and price unit must be stable")
    return d


def curve_at(frame, decision, session):
    """Select the two nearest unexpired contracts from one visible session."""
    decision, session = pd.Timestamp(decision), pd.Timestamp(session)
    if decision.tzinfo is None or session.tzinfo is not None or session != session.normalize():
        raise ValueError("expected timezone-aware decision and local session date")
    if session.date() > decision.tz_convert("America/New_York").date():
        raise ValueError("session cannot be after the decision date")
    visible = frame[frame["available_at_utc"].le(decision) & frame["session"].eq(session)
                    & frame["expiry"].gt(session)]
    result = {}
    for commodity, group in visible.groupby("commodity"):
        latest = group.sort_values("available_at_utc").drop_duplicates("contract_id", keep="last")
        expiries = latest["expiry"].drop_duplicates().sort_values().iloc[:2]
        pair = latest[latest["expiry"].isin(expiries)].sort_values("expiry")
        if len(expiries) < 2 or len(pair) != 2 or pair[["currency", "price_unit"]].nunique().gt(1).any():
            result[commodity] = {"status": "NOT_COMPUTED", "reason": "need two unambiguous comparable expiries"}
            continue
        front, second = tuple(pair.itertuples(index=False))
        days = (second.expiry - front.expiry).days
        spread = front.settlement - second.settlement
        carry = float(np.log(front.settlement / second.settlement) * 365 / days) if min(front.settlement, second.settlement) > 0 else None
        result[commodity] = {"status": "COMPUTED", "front": front.contract_id, "second": second.contract_id,
                             "spread_front_minus_second": float(spread), "annualized_log_carry": carry,
                             "shape": "backwardation" if spread > 0 else "contango" if spread < 0 else "flat",
                             "currency": front.currency, "price_unit": front.price_unit,
                             "source_ids": sorted({front.source, second.source}),
                             "latest_available_at_utc": max(front.available_at_utc, second.available_at_utc).isoformat()}
    return result


def import_csv(path):
    payload = Path(path).read_bytes()
    frame = validate(pd.read_csv(io.BytesIO(payload), dtype={"contract_id": str}))
    if frame.empty:
        raise ValueError("no settlement observations")
    if frame["available_at_utc"].gt(pd.Timestamp.now(tz="UTC")).any():
        raise ValueError("settlement availability is in the future")
    digest = hashlib.sha256(payload).hexdigest()
    raw, processed = C.RAW / "futures_contracts", C.PROC / "futures_contracts"
    for directory in [raw, processed]:
        directory.mkdir(parents=True, exist_ok=True)
    source = raw / f"{digest}.csv"
    if not source.exists():
        with source.open("xb") as stream:
            stream.write(payload)
    output = processed / f"{digest}.parquet"
    if not output.exists():
        temporary = output.with_suffix(".tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    lock = C.hold_lock()
    print(import_csv(args.csv))
