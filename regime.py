"""Current market regime for live signals (spec 8 / 12.1).

Observable rule: SPY vs its 200-session average x VIX vs its 5-year median. Latent: 3-state Gaussian HMM on
(SPY daily return, VIX log change) fitted on all data before the current calendar year, then run forward
through today; the reported probability is the FILTERED P(highest-variance state) at the last session (no smoothing).
3 states (calm / normal / stress) beat 2 on out-of-sample likelihood in 16 of 19 years (regimes.py, t=3.6).
"""
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats

import config as C

N_STATES = 3


@lru_cache(maxsize=1)
def current():
    spy = pd.read_parquet(C.RAW / "prices" / f"{C.BENCHMARK}.parquet").drop_duplicates("date").set_index("date")
    cal = spy.index
    vix = pd.read_parquet(C.RAW / "prices" / "^VIX.parquet").drop_duplicates("date").set_index("date")["close"].reindex(cal).ffill()
    ac = spy["adjclose"]
    up = bool(ac.iloc[-1] > ac.rolling(200).mean().iloc[-1])
    hv = bool(vix.iloc[-1] > vix.rolling(1260, min_periods=252).median().iloc[-1])
    state = ("uptrend" if up else "downtrend") + "/" + ("high_vol" if hv else "low_vol")
    p = None
    try:
        from hmmlearn.hmm import GaussianHMM
        obs = pd.DataFrame({"r": ac.pct_change(), "v": np.log(vix).diff()}, index=cal).dropna()
        yr = obs.index[-1].year
        m = GaussianHMM(n_components=N_STATES, covariance_type="full", n_iter=300, random_state=7).fit(obs[obs.index.year < yr].to_numpy())
        k = int(np.argmax([m.covars_[i][0, 0] for i in range(N_STATES)]))
        X = obs.to_numpy()[-756:]
        logB = np.column_stack([stats.multivariate_normal(m.means_[i], m.covars_[i], allow_singular=True).logpdf(X) for i in range(N_STATES)])
        a = m.startprob_
        for t in range(len(X)):
            a = (a @ m.transmat_ if t else m.startprob_) * np.exp(logB[t] - logB[t].max())
            a = a / a.sum()
        p = float(a[k])
    except Exception:
        p = None
    return {"state": state, "p_high_vol": p, "method": "observable SPY>200d x VIX>5y median; 3-state Gaussian HMM (P of stress state), filtered",
            "as_of": str(cal[-1].date()), "vix": float(vix.iloc[-1])}


def field():
    c = current()
    return {"state": c["state"], "p_high_vol": c["p_high_vol"], "method": c["method"]}
