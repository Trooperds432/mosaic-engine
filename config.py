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

# Validation gate (model level)
GATE_MIN_HOLDOUT_IC_T = 2.0
GATE_MIN_WF_POSITIVE_YEAR_FRAC = 0.6
GATE_MIN_DSR = 0.90
FDR_Q = 0.10

# Signal level
SIGNAL_TOP_QUANTILE = 0.2    # LONG candidates come from the top quintile of predicted alpha
PRICE_MAX_STALENESS_SESSIONS = 3
FUNDAMENTAL_MAX_AGE_DAYS = 460
INSIDER_MAX_COVERAGE_GAP_DAYS = 7

MODEL_ID = "mosaic-xs-ridge"
MODEL_VERSION = "2.0.0"
FEATURE_SET_VERSION = "fs-2.1"


def hold_lock():
    """Block until the engine's shared lock (/opt/mosaic-engine/.lock, also taken by run_daily/run_research/run_refit)
    is free, then hold it for the life of the process. Heavy scripts call this when run directly so a manual run can
    never overlap the nightly job and exhaust memory. Skipped when a wrapper script already holds the lock."""
    import fcntl
    import os
    if os.environ.get("MOSAIC_LOCK_HELD"):
        return None
    f = open(ROOT / ".lock", "a")
    fcntl.flock(f, fcntl.LOCK_EX)
    os.environ["MOSAIC_LOCK_HELD"] = "1"
    return f
