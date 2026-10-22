"""Security-level historical bundle, without current-ticker joins or invented delisting returns.

The importer accepts documented vendor exports. It does not claim that a vendor
catalog is complete or manufacture identifiers for ambiguous historical tickers.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

INDICES = {"SP500", "NASDAQ100", "RUSSELL2000"}
PRICE_FIELDS = ["open", "high", "low", "close", "volume", "adjclose", "adj_open"]


def require(frame, columns, name):
    if not set(columns) <= set(frame):
        raise ValueError(f"{name}: missing {sorted(set(columns) - set(frame))}")
    if frame[list(columns)].isna().any().any():
        raise ValueError(f"{name}: required values are missing")


def validate_bundle(master, membership, prices, terminal):
    require(master, ["security_id", "cik", "ticker", "exchange", "source"], "master")
    if master["security_id"].duplicated().any() or not master["security_id"].str.fullmatch(r"[A-Za-z0-9_]{1,64}").all():
        raise ValueError("master requires unique filesystem-safe permanent security IDs")
    ids = set(master["security_id"])
    cik = pd.to_numeric(master["cik"], errors="raise")
    if not np.isfinite(cik).all() or (cik <= 0).any() or (cik % 1 != 0).any():
        raise ValueError("issuer CIKs must be verified positive integers")
    master["cik"] = cik.astype("int64")
    if ids & {C.BENCHMARK, *C.REGIME_SERIES}:
        raise ValueError("security IDs cannot shadow benchmark/regime symbols")
    for name, frame in [("membership", membership), ("prices", prices), ("terminal", terminal)]:
        if not set(frame.get("security_id", [])) <= ids:
            raise ValueError(f"{name}: unknown security IDs")
    require(membership, ["security_id", "index", "start", "known_at", "source"], "membership")
    if not set(membership["index"]) <= INDICES:
        raise ValueError("unsupported index identifier")
    for col in ["start", "end", "known_at"]:
        membership[col] = pd.to_datetime(membership[col], errors="raise")
    if (membership["end"].notna() & (membership["end"] <= membership["start"])).any():
        raise ValueError("membership end must be exclusive and later than start")
    for _, g in membership.groupby(["security_id", "index"]):
        g = g.sort_values("start")
        ends = g["end"].fillna(pd.Timestamp.max)
        if len(g) > 1 and (g["start"].iloc[1:].to_numpy() < ends.iloc[:-1].to_numpy()).any():
            raise ValueError("overlapping membership intervals")
    require(prices, ["security_id", "date", "ticker", *PRICE_FIELDS, "market_cap", "source"], "prices")
    prices["date"] = pd.to_datetime(prices["date"], errors="raise")
    if prices.duplicated(["security_id", "date"]).any():
        raise ValueError("duplicate security/date prices")
    if prices.duplicated(["ticker", "date"]).any():
        raise ValueError("ticker maps to multiple securities on the same date")
    numeric = prices[PRICE_FIELDS].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric).all().all() or (numeric.drop(columns="volume") <= 0).any().any() or (numeric["volume"] < 0).any():
        raise ValueError("invalid prices/volume")
    if (numeric["high"] < numeric["low"]).any():
        raise ValueError("high below low")
    prices[PRICE_FIELDS] = numeric
    cap = pd.to_numeric(prices["market_cap"], errors="raise")
    if not np.isfinite(cap).all() or (cap <= 0).any():
        raise ValueError("positive security-level market_cap required")
    prices["market_cap"] = cap
    # A class-specific market cap is mandatory when issuer fundamentals would
    # otherwise allocate the same total shares to several classes.
    multiple = set(master.loc[master["cik"].duplicated(keep=False), "security_id"])
    if multiple and ("market_cap" not in prices or prices.loc[prices["security_id"].isin(multiple), "market_cap"].isna().any()):
        raise ValueError("multi-class issuers require security-level market_cap")
    require(terminal, ["security_id", "last_trade_date", "settlement_date", "terminal_adjclose", "source"], "terminal")
    terminal["last_trade_date"] = pd.to_datetime(terminal["last_trade_date"], errors="raise")
    terminal["settlement_date"] = pd.to_datetime(terminal["settlement_date"], errors="raise")
    if terminal["security_id"].duplicated().any() or (terminal["settlement_date"] < terminal["last_trade_date"]).any():
        raise ValueError("invalid terminal event chronology or duplicates")
    value = pd.to_numeric(terminal["terminal_adjclose"], errors="raise")
    if not np.isfinite(value).all() or (value < 0).any():
        raise ValueError("terminal price must be nonnegative and finite; zero means verified total loss")
    terminal["terminal_adjclose"] = value
    last_price = prices.groupby("security_id")["date"].max()
    for t in terminal.itertuples():
        if t.security_id not in last_price or t.last_trade_date != last_price[t.security_id]:
            raise ValueError("terminal last trade must match the last observed security price")
    return master, membership, prices, terminal


def membership_mask(membership, dates, security_ids):
    """As-of constituent-aware union; future additions and removed members are excluded."""
    out = pd.DataFrame(False, index=dates, columns=security_ids)
    for r in membership.itertuples():
        if r.security_id not in out:
            continue
        known = pd.Timestamp(r.known_at)
        eligible = (dates >= max(pd.Timestamp(r.start), known))
        if pd.notna(r.end):
            eligible &= dates < pd.Timestamp(r.end)
        out.loc[eligible, r.security_id] = True
    return out


def forward_returns(opens, terminal, entries, exits):
    """Security returns to exit open; terminal cash is held after known settlement.

    Future terminal events are outcomes, never inputs to a decision feature.
    Missing opens without a verified settled terminal event stay unknown.
    """
    entries, exits = np.asarray(entries, dtype=int), np.asarray(exits, dtype=int)
    result = np.full((len(entries), len(opens.columns)), np.nan)
    calendar = opens.index
    valid = (entries >= 0) & (exits >= entries) & (entries < len(calendar)) & (exits < len(calendar))
    if not valid.any():
        return result
    data = opens.to_numpy(dtype=float)
    initial = data[entries[valid]].copy()
    final = data[exits[valid]].copy()
    terms = terminal.set_index("security_id") if len(terminal) else pd.DataFrame()
    for j, sid in enumerate(opens.columns):
        if sid in terms.index:
            t = terms.loc[sid]
            settled = ((calendar[entries[valid]] <= t["last_trade_date"]) & (t["settlement_date"] <= calendar[exits[valid]]))
            fill = np.isnan(final[:, j]) & settled
            final[fill, j] = t["terminal_adjclose"]
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = final / initial - 1
    returns[(initial <= 0) | ~np.isfinite(initial) | ~np.isfinite(returns)] = np.nan
    result[valid] = returns
    return result


def import_bundle(directory):
    directory = Path(directory)
    frames = [pd.read_csv(directory / f"{name}.csv", dtype={"security_id": str}) for name in ["security_master", "membership", "prices", "terminal"]]
    master, members, prices, terminal = validate_bundle(*frames)
    manifest = json.loads((directory / "coverage.json").read_text())
    # Attestation is preserved as a claim, then checked against actual bundle rows
    # by coverage_check. No import turns an assertion into proof of completeness.
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob("*.csv")}
    destination = C.RAW / "historical" / hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:20]
    destination.mkdir(parents=True, exist_ok=False)
    for name, frame in zip(["security_master", "membership", "prices", "terminal"], [master, members, prices, terminal]):
        frame.to_parquet(destination / f"{name}.parquet", index=False)
    (destination / "coverage.json").write_text(json.dumps({**manifest, "input_sha256": hashes}, indent=2))
    C.PROC.mkdir(parents=True, exist_ok=True)
    (C.RAW / "prices").mkdir(parents=True, exist_ok=True)
    if set(master["security_id"]) & {C.BENCHMARK, *C.REGIME_SERIES}:
        raise ValueError("security IDs cannot shadow benchmark/regime symbols")
    for sid, frame in prices.groupby("security_id"):
        frame.sort_values("date").to_parquet(C.RAW / "prices" / f"{sid}.parquet", index=False)
    master = master.assign(yahoo=master["security_id"], sec_name=master.get("name", master["ticker"]), sources="historical_bundle")
    master.to_parquet(C.PROC / "universe.parquet", index=False)
    (C.PROC / "universe_audit.json").write_text(json.dumps({"candidates": len(master), "kept_with_cik": len(master),
                "removed_without_current_sec_ticker": 0, "source": "historical_bundle", "completeness": "see coverage_check"}, indent=2))
    (C.PROC / "historical_bundle.json").write_text(json.dumps({"path": str(destination.relative_to(C.ROOT))}, indent=2))
    return destination


def active_bundle():
    pointer = C.PROC / "historical_bundle.json"
    if not pointer.exists():
        return None
    path = (C.ROOT / json.loads(pointer.read_text())["path"]).resolve()
    if not path.is_relative_to((C.RAW / "historical").resolve()):
        raise ValueError("historical bundle must be within the historical archive")
    return path


def ticker_for_security(symbol, day=None):
    if active_bundle() is None:
        return symbol.replace("-", ".")
    data = pd.read_parquet(C.RAW / "prices" / f"{symbol}.parquet", columns=["date", "ticker"])
    if day is not None:
        data = data[data["date"] <= pd.Timestamp(day)]
    if data.empty:
        raise ValueError(f"no known ticker for {symbol} at decision date")
    return str(data.sort_values("date").iloc[-1]["ticker"])


def security_for_ticker(ticker, day=None):
    if active_bundle() is None:
        return ticker.replace(".", "-")
    master = pd.read_parquet(active_bundle() / "security_master.parquet")
    matches = master[master["ticker"].eq(ticker)]
    day = pd.Timestamp(day) if day is not None else pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    terminal = pd.read_parquet(active_bundle() / "terminal.parquet")
    ended = set(terminal.loc[terminal["last_trade_date"] < day, "security_id"])
    matches = matches[~matches["security_id"].isin(ended)]
    if len(matches) != 1:
        raise ValueError(f"ambiguous or unmapped current ticker {ticker}")
    return str(matches.iloc[0]["security_id"])


def coverage_check(start, end):
    bundle = active_bundle()
    failures = []
    if bundle is None:
        return {"passed": False, "failed_checks": ["verified historical security bundle missing"]}
    claim = json.loads((bundle / "coverage.json").read_text())
    for key in ["constituents_complete", "delisted_prices_complete", "terminal_events_complete", "identifiers_verified"]:
        if claim.get(key) is not True:
            failures.append(f"{key} not attested by the source")
    if set(claim.get("indices", [])) != INDICES:
        failures.append("coverage must include SP500, NASDAQ100 and RUSSELL2000")
    try:
        if pd.Timestamp(claim["start"]) > pd.Timestamp(start) or pd.Timestamp(claim["end"]) < pd.Timestamp(end):
            failures.append("source coverage does not span the research window")
    except (KeyError, ValueError, TypeError):
        failures.append("invalid source coverage dates")
    members = pd.read_parquet(bundle / "membership.parquet")
    prices = pd.read_parquet(bundle / "prices.parquet")
    terminal = pd.read_parquet(bundle / "terminal.parquet")
    in_window = members[(members["start"] <= pd.Timestamp(end)) & (members["end"].isna() | (members["end"] > pd.Timestamp(start)))]
    if set(in_window["index"]) != INDICES:
        failures.append("actual membership records do not cover all three required indices")
    missing = set(in_window["security_id"]) - set(prices["security_id"])
    if missing:
        failures.append(f"{len(missing)} member securities lack prices")
    last = prices.groupby("security_id")["date"].max()
    expected_end = min(pd.Timestamp(end), prices["date"].max())
    early = set(last[last < expected_end].index) & set(in_window["security_id"])
    unresolved = early - set(terminal["security_id"])
    if unresolved:
        failures.append(f"{len(unresolved)} ended series lack verified terminal outcomes")
    return {"passed": not failures, "failed_checks": failures, "source_claim": claim,
            "members_without_prices": sorted(missing), "ended_without_terminal": sorted(unresolved)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="vendor export directory; see docs/HISTORICAL_DATA.md")
    args = parser.parse_args()
    _lock = C.hold_lock()
    print(import_bundle(args.directory))
