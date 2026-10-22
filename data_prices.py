"""Daily OHLCV with split/dividend adjustment from Yahoo Finance chart API.
Stores raw and adjusted series; adjusted open is derived from the adjclose/close factor so that
entries can be modelled at the next session's open."""
import json
import sys
import time
from datetime import datetime, timezone

import httpx
import numpy as np
import pandas as pd

from config import BENCHMARK, HISTORY_START, PROC, RAW, REGIME_SERIES

URL = "https://query1.finance.yahoo.com/v8/finance/chart/{}"


def fetch(symbol: str, client: httpx.Client) -> pd.DataFrame | None:
    p1 = int(pd.Timestamp(HISTORY_START, tz="UTC").timestamp())
    r = None
    for i in range(5):
        try:
            r = client.get(URL.format(symbol), params={"period1": p1, "period2": int(time.time()), "interval": "1d",
                                                       "events": "div,split"})
        except httpx.HTTPError:
            time.sleep(5 * (i + 1))
            continue
        if r.status_code == 429:
            time.sleep(10 * (i + 1))
            continue
        break
    if r is None or r.status_code != 200:
        return None
    res = (r.json().get("chart") or {}).get("result")
    if not res or not res[0].get("timestamp"):
        return None
    res = res[0]
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
    tz = res["meta"].get("exchangeTimezoneName", "America/New_York")
    df = pd.DataFrame({"date": pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(tz).normalize().tz_localize(None),
                       "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
                       "volume": q["volume"], "adjclose": adj if adj else q["close"]})
    df = df.dropna(subset=["close"]).drop_duplicates("date", keep="last")
    f = df["adjclose"] / df["close"]
    df["adj_open"] = df["open"] * f
    df["symbol"] = symbol
    df["ingested_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return df


def run(symbols=None, resume=False):
    uni = pd.read_parquet(PROC / "universe.parquet")
    symbols = symbols or (list(uni["yahoo"]) + [BENCHMARK] + REGIME_SERIES)
    out = RAW / "prices"
    out.mkdir(parents=True, exist_ok=True)
    missing, ok = [], 0
    with httpx.Client(timeout=30, headers={"User-Agent": "Mozilla/5.0"}) as c:
        for s in symbols:
            if resume and (out / f"{s}.parquet").exists():
                ok += 1
                continue
            df = fetch(s, c)
            if df is None or len(df) < 30:
                missing.append(s)
            else:
                for col in ["open", "high", "low", "close", "adjclose", "adj_open"]:
                    df[col] = df[col].astype(np.float32)
                df.to_parquet(out / f"{s}.parquet", index=False)
                ok += 1
            time.sleep(0.25)
    (PROC / "prices_audit.json").write_text(json.dumps({"ok": ok, "missing": missing}, indent=1))
    print(f"prices ok={ok} missing={len(missing)}: {missing[:30]}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--resume"]
    run(args or None, resume="--resume" in sys.argv)
