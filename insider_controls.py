"""As-of value, quality and estimate-revision controls, independently for event and matched firms."""
import numpy as np
import pandas as pd

import config as C
from features import fundamentals_asof, load_facts
from pit_estimates import validate, revision

FACTORS = ["earn_yield", "book_to_mkt", "roa", "earnings_revision"]


def attach(market, events, controls):
    events, controls = events.copy(), controls.copy()
    dates = pd.DatetimeIndex(sorted(set(events["entry_date"] - pd.Timedelta(days=1))))
    ciks = list(market.sym_of)
    facts = fundamentals_asof(dates, ciks, load_facts(ciks))
    cik_of = {sym: cik for cik, sym in market.sym_of.items()}
    path = C.PROC / "estimate_vintages.parquet"
    estimates = validate(pd.read_parquet(path)) if path.exists() else None
    for frame in [events, controls]:
        for factor in FACTORS:
            frame[factor] = np.nan
        for idx, row in frame.iterrows():
            sym = row.get("symbol")
            if sym not in market.col:
                continue
            event = events.loc[idx]
            i = int(event["e"]) - 1
            day = event["entry_date"] - pd.Timedelta(days=1)
            cik = cik_of[sym]
            ni, equity, assets, shares = (facts[k].loc[day, cik] for k in ["ni", "equity", "assets", "shares"])
            cap = market.close[i, market.col[sym]] * shares
            from historical_data import active_bundle
            if active_bundle():
                vendor = pd.read_parquet(C.RAW / "prices" / f"{sym}.parquet", columns=["date", "market_cap"]).set_index("date")
                cap = vendor["market_cap"].reindex([market.cal[i]]).iloc[0]
            frame.loc[idx, "earn_yield"] = ni / cap if cap > 0 else np.nan
            frame.loc[idx, "book_to_mkt"] = equity / cap if cap > 0 else np.nan
            frame.loc[idx, "roa"] = ni / assets if assets > 0 else np.nan
            if estimates is not None:
                decision = pd.Timestamp(market.cal[i]).tz_localize("America/New_York") + pd.Timedelta(hours=16)
                frame.loc[idx, "earnings_revision"] = revision(estimates, sym, decision)
    return events, controls
