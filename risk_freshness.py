"""Per-security forecast freshness and finite VaR checks, without broker access."""
import math
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from provenance import sessions_between

NY = ZoneInfo('America/New_York')


def position_weight(doc, symbol, position_pct, var_limit, *, now=None, max_age_days=4,
                    max_stale_sessions=3, signal_asof=None):
    now = now or datetime.now(timezone.utc)
    try:
        generated = datetime.fromisoformat(doc['generated_at_utc'].replace('Z', '+00:00'))
        if generated.tzinfo is None or now.tzinfo is None:
            raise ValueError('risk timestamps must be timezone-aware')
        age = (now - generated).total_seconds()
        if age < -300 or age > max_age_days * 86400:
            return 0., 'risk file is stale or future-dated: entry blocked'
        r = doc['names'][symbol]
        asof = datetime.strptime(r['as_of'], '%Y-%m-%d').date()
        if sessions_between(asof, asof + timedelta(days=1)) != 1:
            return 0., 'risk as-of is not a trading session: entry blocked'
        close = datetime.combine(asof, time(16), NY)
        stale = sessions_between(asof, now.astimezone(NY).date())
        if close > now or stale > max_stale_sessions:
            return 0., 'per-security risk forecast is stale or future-dated: entry blocked'
        if signal_asof is not None:
            signal = datetime.fromisoformat(signal_asof.replace('Z', '+00:00'))
            if signal.tzinfo is None or asof < signal.astimezone(NY).date():
                return 0., 'risk forecast predates the signal: entry blocked'
        v = float(r['var99_1d'])
        if not all(math.isfinite(x) and x > 0 for x in [v, position_pct, var_limit]):
            return 0., 'invalid VaR or sizing limits: entry blocked'
        weight = min(position_pct, var_limit / v)
        return weight, f"1d VaR99 {v:.2%} ({r['model']}, as of {asof}) -> weight {weight:.1%}"
    except (KeyError, ValueError, TypeError, AttributeError, OverflowError):
        return 0., 'missing or invalid per-security risk forecast: entry blocked'
