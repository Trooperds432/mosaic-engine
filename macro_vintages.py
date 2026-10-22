"""FRED initial releases via documented output_type=4; no sampled-vintage substitution."""
from datetime import datetime, timezone

import pandas as pd

ENDPOINT = "https://api.stlouisfed.org/fred/series/observations"


def initial_releases(client, series, key, start="2005-01-01"):
    if not key:
        raise ValueError("FRED_API_KEY required for verified initial releases")
    rows, offset = [], 0
    while True:
        response = client.get(ENDPOINT, params={"series_id": series, "api_key": key, "file_type": "json", "output_type": 4,
                             "realtime_start": "1776-07-04", "realtime_end": "9999-12-31", "observation_start": start,
                             "limit": 100000, "offset": offset})
        if response.status_code != 200:
            raise ValueError(f"FRED initial-release request failed: HTTP {response.status_code}")
        payload = response.json()
        batch = payload.get("observations")
        if not isinstance(batch, list) or "count" not in payload:
            raise ValueError("FRED did not return an observation page")
        rows.extend(batch)
        offset += len(batch)
        if offset >= int(payload["count"]):
            break
        if not batch:
            raise ValueError("FRED pagination incomplete")
    d = pd.DataFrame(rows)
    if d.empty:
        raise ValueError("empty initial-release history")
    d = d.rename(columns={"date": "obs_date", "realtime_start": "release_date"})
    d["obs_date"] = pd.to_datetime(d["obs_date"], errors="raise")
    d["release_date"] = pd.to_datetime(d["release_date"], errors="raise")
    d["value"] = pd.to_numeric(d["value"].replace(".", None), errors="raise")
    d = d.dropna(subset=["value"])
    if d["obs_date"].duplicated().any() or (d["release_date"] < d["obs_date"]).any():
        raise ValueError("invalid/duplicate first-release observations")
    # Date-level releases have no intraday timestamp. Next-day eligibility avoids
    # assuming a publication time and includes weekends via subsequent reindexing.
    d["available_date"] = d["release_date"] + pd.Timedelta(days=1)
    d["vintage"] = d["release_date"].dt.strftime("%Y-%m-%d")
    d["series"], d["source"], d["pit_eligible"] = series, "fred_initial_release_api", True
    d["ingested_utc"] = datetime.now(timezone.utc).isoformat()
    return d[["series", "obs_date", "value", "release_date", "available_date", "vintage", "source", "pit_eligible", "ingested_utc"]]
