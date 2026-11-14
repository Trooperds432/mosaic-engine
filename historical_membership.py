"""Reconstruct index membership from a verified starting roster and dated changes."""
import argparse
import json
from pathlib import Path

import pandas as pd

from historical_data import INDICES, require


def reconstruct(seed, changes, start, end):
    require(seed, ["security_id", "index", "known_at", "source"], "starting roster")
    require(changes, ["security_id", "index", "effective_date", "known_at", "action", "source"], "changes")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(start) or pd.isna(end) or start >= end:
        raise ValueError("invalid membership reconstruction window")
    seed, changes = seed.copy(), changes.copy()
    for frame in [seed, changes]:
        if not frame["security_id"].str.fullmatch(r"[A-Za-z0-9_]{1,64}").all() or not set(frame["index"]) <= INDICES:
            raise ValueError("unsupported index or permanent security identifier")
        frame["known_at"] = pd.to_datetime(frame["known_at"], format="%Y-%m-%d", errors="raise")
        if frame["known_at"].isna().any():
            raise ValueError("missing membership availability date")
    if seed.empty or seed.duplicated(["index", "security_id"]).any() or seed["known_at"].gt(start).any():
        raise ValueError("starting roster must be unique and known at the start")
    changes["effective_date"] = pd.to_datetime(changes["effective_date"], format="%Y-%m-%d", errors="raise")
    if not changes["effective_date"].between(start, end, inclusive="right").all():
        raise ValueError("changes must fall after the starting roster and within the window")
    if not changes["action"].isin(["join", "leave"]).all() or changes.duplicated(["index", "security_id", "effective_date"]).any():
        raise ValueError("invalid or ambiguous membership actions")

    def opening(row, effective):
        return {"security_id": row.security_id, "index": row.index, "start": effective,
                "end": pd.NaT, "known_at": row.known_at, "source": row.source}

    active = {(row.index, row.security_id): opening(row, start) for row in seed.itertuples()}
    closed = []
    for row in changes.sort_values("effective_date").itertuples():
        key = (row.index, row.security_id)
        if row.action == "join":
            if key in active:
                raise ValueError(f"join for an existing member: {key}")
            active[key] = opening(row, row.effective_date)
        else:
            if key not in active:
                raise ValueError(f"leave for a security absent from the roster: {key}")
            if row.known_at > row.effective_date:
                raise ValueError(f"removal was not known by its effective date: {key}")
            interval = active.pop(key)
            if interval["known_at"] < row.effective_date:
                closed.append({**interval, "end": row.effective_date,
                               "source": f"{interval['source']} | removal: {row.source}"})
    return pd.DataFrame(closed + list(active.values()), columns=[
        "security_id", "index", "start", "end", "known_at", "source"
    ]).sort_values(["index", "security_id", "start"])


def reconcile_snapshots(intervals, snapshots):
    """Compare reconstructed membership against independently dated rosters."""
    require(snapshots, ["index", "date", "security_id"], "validation snapshots")
    snapshots = snapshots.copy()
    snapshots["date"] = pd.to_datetime(snapshots["date"], format="%Y-%m-%d", errors="raise")
    if snapshots.empty or not set(snapshots["index"]) <= INDICES:
        raise ValueError("validation requires dated rosters for supported indices")
    if snapshots.duplicated(["index", "date", "security_id"]).any():
        raise ValueError("duplicate security in validation snapshot")
    rows = []
    for (index, date), roster in snapshots.groupby(["index", "date"]):
        actual = set(roster["security_id"])
        selected = (intervals["index"].eq(index) & intervals["start"].le(date) &
                    intervals["known_at"].le(date) &
                    (intervals["end"].isna() | intervals["end"].gt(date)))
        expected = set(intervals.loc[selected, "security_id"])
        rows.append({"index": index, "date": date.date().isoformat(),
                     "observed": len(actual), "reconstructed": len(expected),
                     "missing": sorted(actual - expected), "unexpected": sorted(expected - actual),
                     "matches": actual == expected})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=Path)
    parser.add_argument("changes", type=Path)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--snapshots", type=Path, help="independently sourced index,date,security_id rosters")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = reconstruct(pd.read_csv(args.seed, dtype="string"),
                         pd.read_csv(args.changes, dtype="string"), args.start, args.end)
    if args.snapshots:
        checks = reconcile_snapshots(result, pd.read_csv(args.snapshots, dtype="string"))
        print(json.dumps(checks, indent=2))
        if any(not check["matches"] for check in checks):
            raise SystemExit("reconstructed membership disagrees with validation rosters")
    result.to_csv(args.out, mode="x", index=False)
    print(f"{len(result)} membership intervals reconstructed")


if __name__ == "__main__":
    main()
