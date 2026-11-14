"""Security-level historical bundle, without current-ticker joins or invented delisting returns.

The importer accepts documented vendor exports. It does not claim that a vendor
catalog is complete or manufacture identifiers for ambiguous historical tickers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

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
    final[(final <= 0) | ~np.isfinite(final)] = np.nan
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


def read_sessions(path):
    """Read a provider's independently sourced regular-session calendar."""
    frame = pd.read_csv(path, dtype="string")
    require(frame, ["date", "source"], "sessions")
    if not frame["date"].str.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}").all():
        raise ValueError("calendar requires ISO session dates")
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise"))
    if dates.empty or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("calendar requires ordered unique session dates")
    return dates


def import_bundle(directory):
    directory = Path(directory)
    frames = [pd.read_csv(directory / f"{name}.csv", dtype={"security_id": str}) for name in ["security_master", "membership", "prices", "terminal"]]
    master, members, prices, terminal = validate_bundle(*frames)
    calendar = directory / "sessions.csv"
    if calendar.is_file():
        read_sessions(calendar)
    manifest = json.loads((directory / "coverage.json").read_text())
    # Attestation is preserved as a claim, then checked against actual bundle rows
    # by coverage_check. No import turns an assertion into proof of completeness.
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob("*.csv")}
    destination = C.RAW / "historical" / hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:20]
    if destination.exists():
        raise FileExistsError(destination)
    master = master.assign(yahoo=master["security_id"], sec_name=master.get("name", master["ticker"]),
                           sources="historical_bundle")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as scratch:
        stage = Path(scratch)
        archive = stage / "archive"
        archive.mkdir()
        if calendar.is_file():
            shutil.copyfile(calendar, archive / calendar.name)
        for name, frame in zip(["security_master", "membership", "prices", "terminal"],
                               [master, members, prices, terminal]):
            frame.to_parquet(archive / f"{name}.parquet", index=False)
        replacements = []
        price_hashes = {}
        for sid, frame in prices.groupby("security_id"):
            staged = stage / f"price-{sid}.parquet"
            frame.sort_values("date").to_parquet(staged, index=False)
            with staged.open("rb") as stream:
                price_hashes[sid] = hashlib.file_digest(stream, "sha256").hexdigest()
            replacements.append((staged, C.RAW / "prices" / f"{sid}.parquet"))
        (archive / "coverage.json").write_text(json.dumps(
            {**manifest, "input_sha256": hashes, "materialized_prices_sha256": price_hashes}, indent=2))
        master.to_parquet(stage / "universe.parquet", index=False)
        (stage / "universe_audit.json").write_text(json.dumps({
            "candidates": len(master), "kept_with_cik": len(master),
            "removed_without_current_sec_ticker": 0, "source": "historical_bundle",
            "completeness": "see coverage_check"}, indent=2))
        pointer = "historical_bundle.json"
        (stage / pointer).write_text(json.dumps({"path": str(destination.relative_to(C.ROOT))}, indent=2))
        for name in ["universe.parquet", "universe_audit.json", pointer]:
            replacements.append((stage / name, C.PROC / name))

        # Back up the active files before changing any of them. Publish the
        # pointer last so a failed write cannot select an incomplete archive.
        backups = {}
        for i, (_, target) in enumerate(replacements):
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                backup = stage / f"backup-{i}"
                shutil.copyfile(target, backup)
                backups[target] = backup
        archive.rename(destination)
        installed = []
        try:
            for staged, target in replacements:
                staged.replace(target)
                installed.append(target)
        except BaseException:
            for target in reversed(installed):
                if target in backups:
                    backups[target].replace(target)
                else:
                    target.unlink()
            destination.rename(archive)
            raise
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
    files = ["coverage.json", "security_master.parquet", "membership.parquet", "prices.parquet", "terminal.parquet"]
    absent = [name for name in files if not (bundle / name).is_file()]
    if absent:
        return {"passed": False, "failed_checks": [f"historical bundle missing {name}" for name in absent]}

    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(start) or pd.isna(end) or start > end:
        return {"passed": False, "failed_checks": ["invalid research window"]}
    claim = json.loads((bundle / "coverage.json").read_text())
    for key in ["constituents_complete", "delisted_prices_complete", "terminal_events_complete", "identifiers_verified"]:
        if claim.get(key) is not True:
            failures.append(f"{key} not attested by the source")
    if set(claim.get("indices", [])) != INDICES:
        failures.append("coverage must include SP500, NASDAQ100 and RUSSELL2000")
    try:
        first, last = pd.Timestamp(claim["start"]), pd.Timestamp(claim["end"])
        if pd.isna(first) or pd.isna(last) or first > start or last < end:
            failures.append("source coverage does not span the research window")
    except (KeyError, ValueError, TypeError):
        failures.append("invalid source coverage dates")

    master = pd.read_parquet(bundle / "security_master.parquet", columns=["security_id", "cik"])
    members = pd.read_parquet(bundle / "membership.parquet")
    prices = pd.read_parquet(bundle / "prices.parquet", columns=["security_id", "date"])
    terminal = pd.read_parquet(bundle / "terminal.parquet",
                               columns=["security_id", "last_trade_date", "settlement_date"])
    effective_start = members[["start", "known_at"]].max(axis=1)
    in_window = members[effective_start.le(end) & (members["end"].isna() | members["end"].gt(start))].copy()
    in_window["effective_start"] = effective_start.loc[in_window.index].clip(lower=start)
    in_window = in_window[in_window["end"].isna() | in_window["end"].gt(in_window["effective_start"])]
    if set(in_window["index"]) != INDICES:
        failures.append("actual membership records do not cover all three required indices")
    universe_path = C.PROC / "universe.parquet"
    if universe_path.is_file():
        universe = pd.read_parquet(universe_path, columns=["yahoo", "cik"])
        expected_ids = master.rename(columns={"security_id": "yahoo"})[["yahoo", "cik"]]
        if (universe["yahoo"].duplicated().any()
                or set(universe.itertuples(index=False, name=None)) != set(expected_ids.itertuples(index=False, name=None))):
            failures.append("materialized universe differs from the historical security master")
    else:
        failures.append("materialized universe missing")
    identifiers = set(master["security_id"])
    unknown = (set(members["security_id"]) | set(prices["security_id"]) | set(terminal["security_id"])) - identifiers
    if unknown:
        failures.append(f"{len(unknown)} security identifiers absent from the security master")

    # Use the requested window, not the last date in the supplied price file.
    # Otherwise an entire export can end early and still pass its own audit.
    observed = prices[prices["date"].between(start, end)]
    if observed.empty or observed["date"].max() < end:
        failures.append("actual prices do not reach the research window end")
    calendar_path = C.RAW / "prices" / f"{C.BENCHMARK}.parquet"
    calendar_source = "observed security price dates"
    if calendar_path.is_file():
        calendar_source = "benchmark"
        dates = pd.read_parquet(calendar_path, columns=["date"])["date"]
        sessions = pd.DatetimeIndex(dates[dates.between(start, end)].unique()).sort_values()
        if sessions.empty or sessions[0] != start or sessions[-1] != end:
            failures.append("benchmark sessions do not span the research window")
    else:
        sessions = pd.DatetimeIndex(observed["date"].unique()).sort_values()

    missing_benchmark, extra_benchmark = [], []
    exchange_path = bundle / "sessions.csv"
    if exchange_path.is_file():
        expected = read_sessions(exchange_path)
        expected = expected[(expected >= start) & (expected <= end)]
        if expected.empty or expected[0] != start or expected[-1] != end:
            failures.append("exchange calendar does not span the research window")
        if not calendar_path.is_file():
            failures.append("benchmark prices missing for exchange calendar reconciliation")
        else:
            missing_benchmark = expected.difference(sessions).strftime("%Y-%m-%d").tolist()
            extra_benchmark = sessions.difference(expected).strftime("%Y-%m-%d").tolist()
            if missing_benchmark or extra_benchmark:
                failures.append("benchmark dates disagree with the exchange calendar")
        sessions = expected
        calendar_source = "supplied exchange calendar"

    price_dates = {sid: pd.DatetimeIndex(group["date"].unique()) for sid, group in observed.groupby("security_id")}
    missing, gaps, unresolved = set(), set(), set()
    terms = terminal.set_index("security_id")
    last_prices = prices.groupby("security_id")["date"].max()
    ended = set(last_prices[last_prices.lt(end)].index) & set(in_window["security_id"])
    unresolved.update(ended - set(terms.index))
    for row in in_window.itertuples():
        lo = sessions.searchsorted(row.effective_start)
        hi = len(sessions) if pd.isna(row.end) else sessions.searchsorted(row.end)
        expected = sessions[lo:hi]
        if expected.empty:
            continue
        sid = row.security_id
        if sid in terms.index:
            term = terms.loc[sid]
            expected = expected[expected <= term["last_trade_date"]]
            if term["settlement_date"] > end:
                unresolved.add(sid)
        actual = price_dates.get(sid, pd.DatetimeIndex([]))
        if len(expected) and not len(actual.intersection(expected)):
            missing.add(sid)
        elif len(expected.difference(actual)):
            gaps.add(sid)
        if sid not in terms.index and len(expected) and len(actual) and actual.max() < expected[-1]:
            unresolved.add(sid)
    if missing:
        failures.append(f"{len(missing)} member securities lack prices during membership")
    if gaps:
        failures.append(f"{len(gaps)} member securities have gaps during membership")
    if unresolved:
        failures.append(f"{len(unresolved)} member securities have unresolved terminal outcomes")
    # The equity panel reads the materialized files, not the archived table.
    # Older bundles lack digests, so compare the price fields directly.
    ids = set(in_window["security_id"]) & set(prices["security_id"])
    expected_hashes = claim.get("materialized_prices_sha256")
    mismatches = []
    fields = ["date", *PRICE_FIELDS, "market_cap"]
    if expected_hashes is None and ids:
        archived = pd.read_parquet(bundle / "prices.parquet", columns=["security_id", *fields])
        by_security = archived.groupby("security_id")
    for sid in sorted(ids):
        path = C.RAW / "prices" / f"{sid}.parquet"
        if not path.is_file():
            mismatches.append(sid)
            continue
        if expected_hashes is not None:
            with path.open("rb") as stream:
                same = hashlib.file_digest(stream, "sha256").hexdigest() == expected_hashes.get(sid)
        else:
            actual = pd.read_parquet(path, columns=fields).sort_values("date").reset_index(drop=True)
            expected = by_security.get_group(sid)[fields].sort_values("date").reset_index(drop=True)
            same = actual.equals(expected)
        if not same:
            mismatches.append(sid)
    if mismatches:
        failures.append(f"{len(mismatches)} materialized price files missing or different from the historical import")
    return {"passed": not failures, "failed_checks": failures, "source_claim": claim,
            "materialized_price_mismatches": mismatches,
            "calendar_source": calendar_source,
            "missing_benchmark_sessions": missing_benchmark, "extra_benchmark_sessions": extra_benchmark,
            "members_without_prices": sorted(missing), "members_with_price_gaps": sorted(gaps),
            "ended_without_terminal": sorted(unresolved)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="vendor export directory; see docs/HISTORICAL_DATA.md")
    args = parser.parse_args()
    _lock = C.hold_lock()
    print(import_bundle(args.directory))
