"""Clinical registry features use archived observations, never current fields backfilled to first-posted dates."""
import numpy as np
import pandas as pd


def phase3_asof(snapshots, dates, ciks):
    """Latest observed issuer snapshot, available from the next UTC date.

    Missing archive/issuer coverage stays NaN. Revisions only affect subsequent
    decisions; legacy mutable registry exports cannot establish historical phase.
    """
    out = pd.DataFrame(np.nan, index=dates, columns=ciks)
    required = {"observed_at_utc", "snapshot_id", "cik", "nct", "phase", "primary_completion"}
    if not required <= set(snapshots.columns):
        return out
    d = snapshots.copy()
    d["observed"] = pd.to_datetime(d["observed_at_utc"], utc=True, errors="coerce")
    d["available"] = d["observed"].dt.normalize() + pd.Timedelta(days=1)
    d["completion"] = pd.to_datetime(d["primary_completion"], utc=True, errors="coerce")
    d = d.dropna(subset=["observed", "snapshot_id", "cik", "nct"])
    for cik, issuer in d.groupby("cik"):
        if cik not in out.columns:
            continue
        for day in dates:
            cutoff = pd.Timestamp(day)
            cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
            known = issuer[issuer["available"] <= cutoff]
            if known.empty:
                continue
            newest = known.sort_values(["observed", "snapshot_id"]).iloc[-1]["snapshot_id"]
            current = known[known["snapshot_id"] == newest].drop_duplicates("nct", keep="last")
            active = current["completion"].isna() | (current["completion"] >= cutoff)
            out.loc[day, cik] = int((current["phase"].fillna("").str.contains("PHASE3") & active).sum())
    return out
