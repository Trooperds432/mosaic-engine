"""Read-only readiness report for the Mosaic V2 research specification."""
import argparse
from datetime import datetime, timezone
import json

import pandas as pd

import config as C
from historical_data import active_bundle, coverage_check
from data_macro import REVISED
from pit_estimates import read_primary_quarterly


def inspect(profile):
    if profile not in C.PROFILES:
        raise ValueError(f"unknown profile: {profile}")
    paths = {
        "benchmark_prices": C.RAW / "prices" / f"{C.BENCHMARK}.parquet",
        "universe": C.PROC / "universe.parquet",
        "sector_mapping": C.PROC / "sic.parquet",
        "company_facts": C.RAW / "facts" / "companyfacts.parquet",
        "estimate_vintages": C.PROC / "estimate_vintages.parquet",
        "macro_vintages": C.PROC / "macro_pit.parquet",
        "feature_panel": C.PROC / f"panel_{profile}.parquet",
    }
    present = {name: path.is_file() for name, path in paths.items()}
    present["estimate_vintages"] = read_primary_quarterly(paths["estimate_vintages"]) is not None
    prices = sum(1 for p in (C.RAW / "prices").glob("*.parquet") if p.is_file())
    filing_files = list((C.RAW / "insider").glob("*.parquet"))
    recent = C.RAW / "insider_recent" / "form4_recent.parquet"
    present["form4_inputs"] = bool(filing_files) or recent.is_file()
    form4_paths = sorted(filing_files) + ([recent] if recent.is_file() else [])
    sidecar = C.PROC / "insider_acceptances.csv"
    present["form4_acceptance_sidecar"] = sidecar.is_file()
    timestamps = None
    if form4_paths and sidecar.is_file():
        from insider_acceptances import coverage as acceptance_coverage
        timestamps = acceptance_coverage(form4_paths, sidecar)

    bundle = active_bundle()
    present["historical_bundle"] = bundle is not None and bundle.is_dir()
    missing = [name for name, exists in present.items() if not exists]
    if timestamps is not None and timestamps["unmatched"]:
        missing.append("complete_form4_acceptance_coverage")
    first_release_series = set()
    if present["macro_vintages"]:
        macro = pd.read_parquet(paths["macro_vintages"], columns=["source", "series"])
        first_release_series = set(macro.loc[macro["source"].eq("fred_initial_release_api"), "series"])
    missing_initial_releases = sorted(set(REVISED) - first_release_series)
    if missing_initial_releases:
        missing.append("verified_macro_initial_releases")
    window = None
    coverage = {"passed": False, "failed_checks": ["Benchmark or historical security bundle unavailable"]}
    if present["benchmark_prices"]:
        dates = pd.to_datetime(pd.read_parquet(paths["benchmark_prices"], columns=["date"])["date"], errors="raise")
        if dates.empty or dates.isna().any() or dates.duplicated().any():
            coverage = {"passed": False, "failed_checks": ["Benchmark dates are empty, missing or duplicated"]}
        else:
            window = {"start": dates.min().date().isoformat(), "end": dates.max().date().isoformat()}
            if present["historical_bundle"]:
                coverage = coverage_check(window["start"], window["end"])
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": C.PROFILES[profile],
        "benchmark": C.BENCHMARK,
        "forecast_horizons_sessions": C.HORIZONS,
        "primary_horizon_sessions": C.PRIMARY_HORIZON,
        "price_files": prices,
        "form4_quarter_files": len(filing_files),
        "form4_acceptance_coverage": timestamps,
        "company_facts_cache_files": sum(1 for p in (C.RAW / "facts" / "by_cik").glob("*.parquet") if p.is_file()),
        "missing_macro_initial_releases": missing_initial_releases,
        "benchmark_window": window,
        "inputs_present": present,
        "missing_inputs": missing,
        "historical_coverage": coverage,
        "readiness": "INPUT_CHECKS_PASSED" if not missing and coverage["passed"] else "NOT_COMPUTED",
        "validation_status": "UNVALIDATED",
        "note": "Presence and coverage checks only. Model performance and deployment eligibility require separate validation.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=C.PROFILES, default=C.PROFILE["name"])
    args = parser.parse_args()
    print(json.dumps(inspect(args.profile), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
