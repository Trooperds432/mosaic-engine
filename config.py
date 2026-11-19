"""Mosaic research engine configuration. Every threshold here is a configurable research
parameter, not an empirically established optimum."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
SIGNALS = ROOT / "signals"
REPORTS = ROOT / "reports"
LEDGER = ROOT / "ledger"

SEC_UA = "mosaic-research-engine jonwuogle@gmail.com"  # SEC fair-access policy requires a contact UA
HISTORY_START = "2005-01-01"
BENCHMARK = "SPY"
REGIME_SERIES = ["^VIX", "GBPUSD=X", "^VIX9D", "^VIX3M", "^VIX6M", "^VVIX", "^SKEW"]  # VIX term structure + options-implied tail risk

# Screening (spec section 0): configurable, applied point-in-time. "large" is the spec's initial screen;
# "broad" extends to the Russell 2000 size range. Select with MOSAIC_PROFILE.
import os
PROFILES = {
    "large": {"name": "large", "min_mcap": 10e9, "min_adv_shares": 1_000_000, "min_dollar_adv": 5e6},
    "broad": {"name": "broad", "min_mcap": 0.3e9, "min_adv_shares": 200_000, "min_dollar_adv": 2e6},
}
PROFILE = PROFILES[os.environ.get("MOSAIC_PROFILE", "large")]
MIN_MCAP_USD = PROFILE["min_mcap"]
MIN_ADV_SHARES = PROFILE["min_adv_shares"]
ADV_WINDOW = 60

# Research design
HORIZONS = [1, 5, 20, 60, 120]
PRIMARY_HORIZON = 20
DECISION_STEP = 5            # one cross-section every 5 sessions
HOLDOUT_YEARS = 2            # final untouched out-of-sample block
WF_MIN_TRAIN_YEARS = 5       # first walk-forward fit needs this much history
WF_REFIT_SESSIONS = 252      # refit once per year
RIDGE_ALPHA = 10.0
WINSOR_Q = 0.01
N_QUANTILES = 5
CAL_BUCKETS = 10

# Costs for a GBP Trading 212 account buying US shares (per side, in return units)
FX_FEE = 0.0015              # T212 FX conversion fee, 0.15% each way
HALF_SPREAD = 0.0003         # large-cap assumption, 3 bp
SLIPPAGE = 0.0005            # 5 bp market-order slippage / impact assumption
ROUND_TRIP_COST = 2 * (FX_FEE + HALF_SPREAD + SLIPPAGE)
# Cost model v2 (prereg/spread_check_v1.json): half-spread = max(quoted 60-session mean half-spread, HALF_SPREAD),
# Corwin-Schultz where no quote exists. Set MOSAIC_COST_MODEL=v1_cs to reproduce earlier runs.
COST_MODEL = os.environ.get("MOSAIC_COST_MODEL", "v2_quoted")
QUOTED_SPREAD_FILE = PROC / "quoted_spread.parquet"
QUOTE_MAX_STALENESS_DAYS = 5

# Validation gate (model level)
GATE_MIN_HOLDOUT_IC_T = 2.0
GATE_MIN_WF_POSITIVE_YEAR_FRAC = 0.6
GATE_MIN_DSR = 0.90
GATE_MAX_PBO = 0.50          # CSCV probability of backtest overfitting (Bailey et al. 2014); fails closed if not computed
SPA_MIN_COVERAGE = 0.90      # strategy admission to SPA/Reality Check (prereg/spa_coverage_v1.json); fixed before results
FDR_Q = 0.10

# Signal level
SIGNAL_TOP_QUANTILE = 0.2    # LONG candidates come from the top quintile of predicted alpha
PRICE_MAX_STALENESS_SESSIONS = 3
FUNDAMENTAL_MAX_AGE_DAYS = 460
INSIDER_MAX_COVERAGE_GAP_DAYS = 7

MODEL_ID = "mosaic-xs-ridge"
MODEL_VERSION = "2.0.0"
FEATURE_SET_VERSION = "fs-2.8"


def hold_lock():
    """Hold the shared research lock until the calling process exits."""
    if os.environ.get("MOSAIC_LOCK_HELD"):
        return None
    stream = (ROOT / ".lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            # Windows can lock a byte beyond the end of an empty lock file.
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX)
    except BaseException:
        stream.close()
        raise
    os.environ["MOSAIC_LOCK_HELD"] = "1"
    return stream
