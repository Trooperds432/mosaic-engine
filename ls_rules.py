"""Market-neutral long/short selection rules shared by the backtest (ls_research.py) and the Alpaca paper bot.

Sector-neutral: within each sector long the top LS_Q of the score and short the bottom LS_Q; a held name stays until it
leaves the top / bottom LS_BUFFER. Equal weight inside each leg, legs sized so long beta = short beta and long + short
gross = LS_GROSS. Shorts need short interest / shares <= SI_MAX_SHORT (and, live, an easy-to-borrow flag).
"""
import numpy as np
import pandas as pd

LS_Q = 0.10
LS_BUFFER = 0.40   # chosen on walk-forward only (2014-2024): widest buffer cut costs most
LS_GROSS = 1.5
SI_MAX_SHORT = 0.15
MAX_NAME_WEIGHT = 0.02   # per-name cap as a fraction of equity


def blume(beta):
    return 0.67 * np.nan_to_num(np.asarray(beta, dtype="float64"), nan=1.0) + 0.33


def borrow_fee(si):
    si = np.nan_to_num(np.asarray(si, dtype="float64"), nan=0.0)
    return np.where(si < 0.05, 0.003, np.where(si < 0.10, 0.01, 0.03))


def select(g, held_long=(), held_short=(), can_short=None, gross=LS_GROSS):
    """g: one date, columns symbol, sector, score, beta, si. Returns {symbol: weight} (negative = short) or {}."""
    g = g.dropna(subset=["score"])
    if len(g) < 50:
        return {}
    r = g.groupby(g["sector"].fillna("Unknown"))["score"].rank(pct=True).to_numpy()
    sym = g["symbol"].to_numpy()
    long_m = (r >= 1 - LS_Q) | (np.isin(sym, list(held_long)) & (r >= 1 - LS_BUFFER))
    short_ok = np.nan_to_num(g["si"].to_numpy(dtype="float64"), nan=0.0) <= SI_MAX_SHORT
    if can_short is not None:
        short_ok &= np.array([can_short(s) for s in sym], dtype=bool)
    short_m = ((r <= LS_Q) | (np.isin(sym, list(held_short)) & (r <= LS_BUFFER))) & short_ok & ~long_m
    L, S = g[long_m], g[short_m]
    if len(L) < 5 or len(S) < 5:
        return {}
    bl, bs = blume(L["beta"]).mean(), blume(S["beta"]).mean()
    gl = gross * bs / (bl + bs)
    gs = gross - gl
    wl, ws = gl / len(L), gs / len(S)
    k = min(1.0, MAX_NAME_WEIGHT / max(wl, ws))  # per-name cap scales both legs together, so the book stays beta-neutral
    wl, ws = wl * k, ws * k
    return {**{s: wl for s in L["symbol"]}, **{s: -ws for s in S["symbol"]}}
