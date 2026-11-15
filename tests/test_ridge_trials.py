import json
from pathlib import Path

import numpy as np
import pandas as pd

from recipe import digest
from research import long_only_backtest, settled_before
from ridge_controls import decision, shuffled_training, signed_composite
from ridge_paper import NY, evaluate, review_gate


def test_historical_ledger_known_answer():
    fixture = json.loads((Path(__file__).parent / "fixtures/ridge_ledger_known_answer.json").read_text())
    rows = pd.DataFrame(fixture["positions"] + fixture["lower_ranked"])
    dates = pd.DatetimeIndex([fixture["source_period"]] * len(rows))

    def run(multiplier):
        fills = []
        periods = long_only_backtest(
            rows.score, rows.realized_excess, dates, rows.symbol, rows.one_way_cost, 20, top=.2,
            benchmark_returns=np.repeat(fixture["benchmark"], len(rows)),
            trades=fills, cost_mult=multiplier,
        )
        canonical = {"fills": sorted([
            ["entry" if f["phase"] == "rebalance" else "exit", f["symbol"],
             format(f["notional_nav_units"], ".10f"), format(f["fee_nav_units"], ".10f")]
            for f in fills]), "nav": format(periods.nav_end.iloc[-1], ".10f")}
        return digest(canonical), periods.nav_end.iloc[-1]

    actual, base = run(1)
    changed, costly = run(2)
    assert actual == fixture["expected_sha256"]
    assert changed != actual and costly < base


def test_signed_composite_omits_missing_and_unsigned():
    x = pd.DataFrame({"a": [.2, .3], "b": [.4, -.4], "c": [99, 99]})
    raw = x.copy()
    raw.loc[1, "b"] = np.inf
    np.testing.assert_allclose(signed_composite(x, raw, {"a": 1, "b": -1, "c": 0}), [-.1, .3])


def test_whole_block_shuffle_and_purge():
    dates = np.repeat(pd.date_range("2024-01-01", periods=8), 2)
    symbols = np.tile(["A", "B"], 8)
    values = pd.DataFrame({"input": np.arange(16)})
    original = values.copy()
    first, keep = shuffled_training(values, dates, symbols, 7)
    second, _ = shuffled_training(values, dates, symbols, 7)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(values, original)
    assert keep.all() and not first.equals(values)
    np.testing.assert_array_equal(first["input"].to_numpy().reshape(-1, 2)[:, 1] -
                                  first["input"].to_numpy().reshape(-1, 2)[:, 0], np.ones(8))
    assert settled_before([2, 3], 20, 29).tolist() == [True, False]


def test_forward_rejects_overlap_unsettled_and_old_recipe():
    cal = pd.bdate_range("2024-01-01", periods=369)
    symbols = ["S" + str(n) for n in range(20)]
    prices = pd.DataFrame(100. + np.arange(len(cal))[:, None] * np.arange(1, 21)[None, :] / 100,
                          index=cal, columns=symbols)
    spy = pd.DataFrame(100., index=cal, columns=["open", "close", "adjclose"])
    reg = dict(horizon=120, entry_fraction=.2, retention_fraction=.4, cost_sensitivity_multiplier=2,
               min_scored_dates=63, min_settled_cohorts=3, min_hac_t=2)
    spec = dict(recipe_id="current", frozen_at="2024-01-01T00:00:00Z", registration=reg)
    rows = [dict(symbol=s, score=n, cost=.001) for n, s in enumerate(symbols)]
    records = [
        dict(recipe_id="old" if day == cal[0] else "current", decision=str(day.date()),
             recorded_at=day.replace(hour=17).tz_localize(NY).isoformat(), forecasts=rows)
        for day in cal[:363]
    ]
    before, _, _ = evaluate(spec, records, spy.iloc[:360], prices, pd.DataFrame())
    after, _, _ = evaluate(spec, records, spy, prices, pd.DataFrame())
    assert before["settled_cohorts"] == 2 and before["status"] == "INCONCLUSIVE"
    assert after["settled_cohorts"] == 3
    assert after["forward_ic"]["n"] >= 63
    assert after["double_cost_mean_net_excess"] < after["mean_net_excess"]
    assert review_gate({"n": 63, "mean": .1, "t": 2.1}, .01, 3, reg) == "PAPER_TEST_ELIGIBLE"
    assert review_gate({"n": 63, "mean": .1, "t": 2.1}, -.01, 3, reg) == "FAILED"
    assert decision(False, True, -.01, -.01, -.01, False) == "UNVALIDATED"
    assert decision(True, True, 0., -.01, -.01, False) == "FAILED"

def test_staged_coverage_reports_unpriced_member_dates(tmp_path):
    from ridge_sp500_study import audit_bundle

    cal = pd.bdate_range("2024-01-02", periods=4)
    (tmp_path / "coverage.json").write_text(json.dumps({
        "start": str(cal[0].date()), "end": str(cal[-1].date()),
        "constituents_complete": True, "identifiers_verified": True,
        "delisted_prices_complete": True, "terminal_events_complete": True,
        "corporate_actions_verified": True,
    }))
    pd.DataFrame({"security_id": ["A"], "ric": ["A.N"]}).to_csv(
        tmp_path / "security_master.csv", index=False)
    pd.DataFrame({"security_id": ["A"], "index": ["SP500"], "start": [cal[0]],
                  "end": [None], "known_at": [cal[0]]}).to_csv(tmp_path / "membership.csv", index=False)
    pd.DataFrame({"security_id": ["A"] * 2, "date": cal[:2]}).to_csv(tmp_path / "prices.csv", index=False)
    pd.DataFrame(columns=["security_id", "last_trade_date", "settlement_date"]).to_csv(
        tmp_path / "terminal.csv", index=False)
    pd.DataFrame({"date": cal, "source": "fixture exchange"}).to_csv(tmp_path / "sessions.csv", index=False)
    snapshots = tmp_path / "snapshot.csv"
    pd.DataFrame({"index": ["SP500"], "date": [cal[0]], "security_id": ["A"]}).to_csv(snapshots, index=False)
    result, _ = audit_bundle(tmp_path, cal, cal[0], cal[-1], validation_rosters=snapshots)
    assert not result["passed"]
    assert result["missing_prices"] == {"A": cal[2:].strftime("%Y-%m-%d").tolist()}
    assert result["unresolved_terminals"] == {"A": str(cal[1].date())}
    assert result["roster_reconciliation"][0]["matches"]
