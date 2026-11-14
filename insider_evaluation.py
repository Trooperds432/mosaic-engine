"""Matched controls and portfolio accounting for the preregistered Form 4 study."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from features import fundamentals_asof, load_facts
from pit_estimates import revision
from research import bh_fdr, deflated_sr, hac_mean, pbo_cscv, sharpe_stats


def covariates(market, sid, e, facts, vintages):
    i = e - 1
    cik = market.cik[sid]
    day = market.cal[e]
    cap = market.cap[sid].iloc[i]
    ni, equity, assets = (facts[key].at[day, cik] for key in ("ni", "equity", "assets"))
    cutoff = day.tz_localize("America/New_York").tz_convert("UTC")
    return {"mom": market.mom[sid].iloc[i], "rev": market.rev[sid].iloc[i],
            "log_adv": np.log(market.adv[sid].iloc[i]), "beta": market.beta[sid].iloc[i],
            "earn_yield": ni / cap if cap > 0 else np.nan,
            "book_to_mkt": equity / cap if cap > 0 else np.nan,
            "roa": ni / assets if assets > 0 else np.nan,
            "earnings_revision": revision(vintages, sid, cutoff) if vintages is not None else np.nan}


def matched_evaluation(market, events, purchases, vintages, controls, size):
    ciks = sorted(set(market.cik.values()))
    days = pd.DatetimeIndex(events["entry_date"].unique()).sort_values()
    facts = fundamentals_asof(days, ciks, load_facts(ciks))
    last_purchase = purchases.groupby("issuercik")["e"].apply(lambda s: np.sort(s.unique())).to_dict()
    rows = []
    for row in events.itertuples():
        i, sid = row.e - 1, row.symbol
        e = row.e
        # Check only purchases visible to this session. The control's later
        # returns and future purchases play no role in selecting it.
        eligible = market.eligible.iloc[i]
        cutoff = market.cal.searchsorted(market.cal[e] - pd.Timedelta(days=90), side="right")
        candidates = []
        for other in eligible.index[eligible]:
            cik = market.cik[other]
            if cik == row.cik or market.sector[other] != market.sector[sid]:
                continue
            prior = last_purchase.get(cik, np.array([], dtype=int))
            end = prior.searchsorted(e, side="right")
            if end and prior[end - 1] >= cutoff:
                continue
            candidates.append(other)
        if not candidates:
            continue
        pool = [sid, *candidates]
        ranks = pd.DataFrame({"adv": market.adv.iloc[i].reindex(pool),
                              "mom": market.mom.iloc[i].reindex(pool)}).dropna().rank(pct=True)
        if sid not in ranks or len(ranks) < 2:
            continue
        distance = (ranks - ranks.loc[sid]).abs().sum(axis=1).drop(sid).sort_index()
        other = distance.idxmin()
        own = covariates(market, sid, e, facts, vintages)
        matched = covariates(market, other, e, facts, vintages)
        outcome = market.outcome(other, e)
        control_net = np.nan
        if outcome is not None:
            stock, gross, _, settlement, _ = outcome
            cost = market.costs[size][other].iloc[i]
            control_net = gross - cost * (1 + (0 if settlement else 1 + stock))
        rows.append({"entry_date": row.entry_date, "event": sid, "control": other,
                     "score": row.score, "event_net": row.net, "control_net": control_net,
                     **{f"event_{key}": own[key] for key in controls},
                     **{f"control_{key}": matched[key] for key in controls}})
    pairs = pd.DataFrame(rows)
    result = {"events": len(events), "matched": len(pairs), "complete_cases": 0, "p": None}
    if pairs.empty:
        return result, pairs
    x = pd.DataFrame({key: pairs[f"event_{key}"] - pairs[f"control_{key}"] for key in controls})
    result["paired_control_coverage"] = x.notna().sum().to_dict()
    x["score"] = pairs["score"]
    x["outcome"] = pairs["event_net"] - pairs["control_net"]
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    months = pairs.loc[x.index, "entry_date"].dt.to_period("M")
    result.update(complete_cases=len(x), months=int(months.nunique()))
    if len(x) < 30 or months.nunique() < 12:
        return result, pairs
    y = x.pop("outcome")
    design = sm.add_constant(x, has_constant="add")
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return result, pairs
    reg = sm.OLS(y, design).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(months)[0]}, use_t=True)
    result.update(score_coef=float(reg.params["score"]), p=float(reg.pvalues["score"]),
                  event_intercept=float(reg.params["const"]))
    return result, pairs


def portfolio(events, outcomes, spy, positions, size, capacity, horizon, cost_col="cost", multiplier=1):
    """Fixed-dollar entry sleeves. Unknown prices on selected positions invalidate returns."""
    daily = pd.DataFrame(0., index=spy.index, columns=["gross_pnl", "cost_usd"])
    active, ledger = [], []
    for e, group in events.groupby("e", sort=True):
        active = [x for x in active if x > e]
        available = capacity - len(active)
        for idx, row in group.sort_values(["rank", "symbol"], ascending=[False, True]).head(available).iterrows():
            outcome = outcomes.get(idx)
            cost = row[cost_col] * multiplier
            if outcome is None and bool(row.get("entry_unfilled", False)):
                ledger.append({"entry_date": spy.index[e], "exit_date": spy.index[e + horizon],
                               "symbol": row.symbol, "order_usd": size, "status": "UNFILLED_ENTRY"})
                continue
            active.append(e + horizon)
            entry_fee = size * cost
            record = {"entry_date": spy.index[e], "exit_date": spy.index[e + horizon],
                      "symbol": row.symbol, "order_usd": size, "entry_cost_usd": entry_fee}
            daily.iloc[e, 1] += entry_fee
            if outcome is None:
                daily.iloc[e + 1:e + horizon + 1, 0] = np.nan
                record.update(status="UNKNOWN_OUTCOME")
            else:
                stock, gross, price, settlement, marks = outcome
                exit_fee = 0. if settlement else size * (1 + stock) * cost
                hedge_marks = spy.iloc[e:e + horizon + 1].to_numpy()
                pnl = size * (np.diff(marks) / price - row.beta * np.diff(hedge_marks) / hedge_marks[0])
                daily.iloc[e + 1:e + horizon + 1, 0] += pnl
                daily.iloc[e + horizon, 1] += exit_fee
                record.update(status="SETTLED", exit_kind="terminal_cash" if settlement else "open",
                              entry_adj_open=price, exit_adj_value=marks[-1], gross_pnl_usd=size * gross,
                              exit_cost_usd=exit_fee, net_pnl_usd=size * gross - entry_fee - exit_fee)
            ledger.append(record)
    daily = daily.iloc[positions]
    capital = size * capacity
    daily["gross"] = daily["gross_pnl"] / capital
    daily["net"] = (daily["gross_pnl"] - daily["cost_usd"]) / capital
    return daily, pd.DataFrame(ledger)


def diagnostics(matrix, horizon, secondary, q):
    # Common, nonoverlapping blocks. A block with an unresolved selected exit
    # is excluded for all strategies; a legitimate no-position day is cash.
    n = len(matrix) // horizon * horizon
    b = matrix.iloc[:n].groupby(np.arange(n) // horizon).sum(min_count=horizon).dropna()
    stats = [sharpe_stats(b[col]) for col in b]
    dsr, sr0 = deflated_sr(stats[0], [s["sr"] for s in stats]) if all(stats) else (None, None)
    # CSCV needs a defined ranking in every half-sample.
    blocks = np.array_split(b.to_numpy(), 10)
    nonconstant = len(b) >= 40 and all((block.std(axis=0) > 0).all() for block in blocks)
    pbo = pbo_cscv(b) if all(stats) and nonconstant else None
    tests = {}
    for name, col in zip(["ridge_minus_primary", "ridge_minus_any"], b.columns[1:]):
        tests[name] = hac_mean(b.iloc[:, 0] - b[col], lags=2)
    tests["incremental_ridge"] = {"p": secondary}
    pvals = [v["p"] if v["p"] is not None and np.isfinite(v["p"]) else 1. for v in tests.values()]
    rejected, adjusted = bh_fdr(pvals, q)
    for rec, flag, p in zip(tests.values(), rejected, adjusted):
        rec.update(p_bh=float(p), rejected=bool(flag))
    return {"common_blocks": len(b), "excluded_blocks": n // horizon - len(b), "dsr": dsr,
            "dsr_benchmark_sr": sr0, "pbo": pbo, "secondary_tests": tests}
