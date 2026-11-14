import pandas as pd
import pytest

from lseg_bundle import dated_returns, membership, prices, reference_by_ric


def test_dated_returns_rejects_stale_and_conflicting_rows():
    rows = pd.DataFrame([
        ("A.N", "2024-01-02", "2024-01-02", 1.0),
        ("A.N", "2024-01-02", "2024-01-02", 1.0),
        ("A.N", "2024-01-03", "2024-01-02", 1.0),
        ("A.N", "2024-01-04", "2024-01-04", 2.0),
        ("A.N", "2024-01-04", "2024-01-04", 3.0),
    ], columns=["Instrument", "CalcDate", "ReturnDate", "Total Return"])
    out = dated_returns(rows)
    assert out["tr"].tolist() == [1.0]
    assert out["date"].tolist() == [pd.Timestamp("2024-01-02")]


@pytest.mark.parametrize("missing_ohlcv", [False, True])
def test_prices_use_dated_returns_and_never_backfill_cap(missing_ohlcv):
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    raw = pd.DataFrame({
        "Instrument": "A.N", "Date": dates, "Price Close": [10, 11, 12, 13, 14],
        "Price Open": 9, "Price High": 15, "Price Low": 8, "Volume": 100,
        "Total Return": -90, "Company Market Cap": [None, 100, 100, 100, 100],
        "Outstanding Shares": 10,
    })
    if missing_ohlcv:
        raw.loc[3, "Price Open"] = None
    matched_dates = dates if missing_ohlcv else dates[[0, 1, 2, 4]]
    returns = pd.DataFrame({
        "Instrument": "A.N", "CalcDate": matched_dates, "ReturnDate": matched_dates,
        "Total Return": [0, 10, 20, 30, 30] if missing_ohlcv else [0, 10, 20, 30],
    })
    ids = pd.DataFrame({"security_id": ["A"], "ticker": ["A"]}, index=["A.N"])
    master = pd.DataFrame({"security_id": ["A"], "cik": [1]})
    out, coverage = prices(raw, ids, master, returns)
    assert out["date"].tolist() == list(dates[[1, 2, 4]] if missing_ohlcv else dates[1:3])
    assert out["adjclose"].iloc[1] / out["adjclose"].iloc[0] == pytest.approx(1.2)
    assert coverage["rows_dropped_no_market_cap"] == 1
    assert coverage["dated_returns_without_ohlcv"] == int(missing_ohlcv)
    assert coverage["rows_excluded_by_return_coverage"] == (0 if missing_ohlcv else 2)
    if missing_ohlcv:
        assert out["adjclose"].iloc[2] / out["adjclose"].iloc[1] == pytest.approx(1.3 ** 2)

def test_ambiguous_issuer_reference_is_excluded_instead_of_taking_first():
    raw = pd.DataFrame({
        "Instrument": ["HST.N", "HST.N", "SAFE.N", "SAFE.N", "INVALID.N"],
        "CIK Number": [314733, 1070750, None, 42, 1.5],
    })
    reference, ambiguous = reference_by_ric(raw)
    assert ambiguous == ["HST.N"]
    assert "HST.N" not in reference.index
    assert reference.loc["SAFE.N", "CIK Number"] == 42
    assert pd.isna(reference.loc["INVALID.N", "CIK Number"])

def test_membership_preserves_original_snapshot_dates_and_bounds_final_roster():
    rows = pd.DataFrame({
        "ric": ["A.N", "UNMAPPED.N", "A.N"],
        "date": pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-28"]),
    })
    ids = pd.DataFrame({"security_id": ["A"]}, index=["A.N"])
    out = membership(rows, ids)
    assert out["start"].tolist() == list(pd.to_datetime(["2024-01-31", "2024-03-28"]))
    assert out["end"].tolist() == list(pd.to_datetime(["2024-02-29", "2024-03-29"]))
