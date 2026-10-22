"""Risk report on the bot's current holdings and a research portfolio proposal (spec section 10).

Reads the demo account's positions with GET requests only. Risk: factor-model volatility and exposures,
marginal/component risk, sector weights, 97.5% expected shortfall from 2 years of history, historical and factor
stress scenarios, overnight-gap and liquidity per name. Proposal: cvxpy mean-variance with costs under the bot's
imported limits, built only from LONG signals of validated models; otherwise NO_SIGNAL. The proposal is research
output for review and is never sent to the bot.
"""
import json
import math
from datetime import datetime, timezone

import httpx
import pandas as pd

import config as C
import risk_limits
from features import load_prices
from risk_model import RiskModel, optimise, stress, tail_and_liquidity


def holdings(lim):
    kid, sec = risk_limits._env_file("/root/.hermes/.env", "T212_API_KEY_ID"), risk_limits._env_file("/root/.hermes/.env", "T212_API_SECRET")
    with httpx.Client(timeout=30, auth=(kid, sec)) as c:
        pos = c.get(lim["base_url"] + "/equity/portfolio").json()
    rows = []
    for p in pos:
        t = p["ticker"]
        sym = t[:-len("_US_EQ")].replace("_", "-") if t.endswith("_US_EQ") else None  # T212 suffixes US shares with _US_EQ
        rows.append({"t212": t, "symbol": sym, "qty": float(p["quantity"]), "price": float(p["currentPrice"]), "avg": float(p["averagePrice"])})
    return pd.DataFrame(rows)


def run(profile="broad"):
    lim = risk_limits.run()
    rm = RiskModel(profile)
    out = {"generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "profile": profile, "limits_used": {k: lim.get(k) for k in
           ["position_pct", "stop_loss", "account_total_usd", "order_size_usd", "long_only", "leverage"]}}
    h = holdings(lim)
    out["holdings"] = h.to_dict(orient="records")
    P, _ = load_prices(0, symbols=set(h["symbol"].dropna()) if len(h) else set())
    if len(h):
        val_usd = h["qty"] * h["price"]  # T212 quotes US shares in USD
        w = pd.Series((val_usd / lim["account_total_usd"]).to_numpy(), index=h["symbol"]).dropna()
        w = w[w.index.isin(P["adjclose"].columns)]
        if len(w):
            out["current_risk"] = rm.decompose(w)
            out["current_risk"]["gross_exposure"] = float(w.sum())
            out["stress"] = stress(w, P, rm)
            out["tail_liquidity"] = tail_and_liquidity(w, P, lim.get("order_size_usd") or 600)
    else:
        out["current_risk"] = "no open positions"
    # tail dependence (t-copula): bot holdings, or while in cash the top-20 names of tonight's broad 20-day forecasts
    from risk_model import tail_dependence_report
    basket = [x for x in h["symbol"].dropna()] if len(h) else []
    label = "bot holdings"
    if not basket:
        fc = pd.read_parquet(C.REPORTS / "forecast_latest.parquet")
        top = fc[(fc["profile"] == profile) & (fc["horizon"] == C.PRIMARY_HORIZON)].nlargest(20, "ensemble_rank")
        basket, label = list(top["symbol"]), "reference basket: top 20 of tonight's forecasts (not holdings)"
    Pt, _ = load_prices(0, symbols=set(basket))
    basket = [b for b in basket if b in Pt["adjclose"].columns]
    if basket:
        out["tail_dependence"] = {"basket": label, "symbols": basket, **tail_dependence_report(basket, Pt)}
    sig = json.loads((C.SIGNALS / "latest.json").read_text())
    longs = [s for s in sig["signals"] if s["signal"]["direction"] == "LONG"]
    if not longs:
        out["proposal"] = {"status": "NO_SIGNAL", "reason": "no validated LONG signals today; nothing to optimise"}
    else:
        mu = pd.Series({s["asset"]["ticker"].replace(".", "-"): s["forecast"]["expected_net_excess_return"] * 252 / s["forecast"]["horizon_sessions"] for s in longs})
        cost = pd.Series({s["asset"]["ticker"].replace(".", "-"): s["forecast"]["expected_cost_return"] / 2 for s in longs})
        w0 = pd.Series((h["qty"] * h["price"] / lim["account_total_usd"]).to_numpy(), index=h["symbol"]) if len(h) else pd.Series(dtype=float)
        wopt = optimise(mu, rm, w0, lim, cost=cost)
        out["proposal"] = {"status": "RESEARCH_PROPOSAL_NOT_AN_ORDER", "weights": wopt.to_dict() if wopt is not None else None,
                           "risk": rm.decompose(wopt) if wopt is not None and len(wopt) else None,
                           "objective": "max w'mu - 2.5 w'Sigma w - costs; long-only, per-name cap = bot position_pct, sector cap 30%, turnover cap 50%"}
    (C.REPORTS / "portfolio_risk.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k not in ("stress", "tail_liquidity")}, indent=1, default=float)[:3000])
    return out


if __name__ == "__main__":
    _lock = __import__("config").hold_lock()
    run()
