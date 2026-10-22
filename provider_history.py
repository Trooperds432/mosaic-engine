"""Bounded, opt-in provider collection. Raw archives are discovery evidence, never automatic PIT certification."""
import argparse
import io
import json
import os
from datetime import date, datetime, timezone
from uuid import uuid4

import httpx
import pandas as pd

import config as C
from provider_access import key as credential, reserve_alpha_request


def collect(provider, *, day=None, state="delisted", symbol=None, query=None, client=None):
    client = client or httpx.Client(timeout=45)
    key_names = {"alphavantage": ["ALPHAVANTAGE_API_KEY", "ALPHA_VANTAGE_API_KEY"], "finnhub": ["FINNHUB_API_KEY"], "exa": ["EXA_API_KEY"]}
    key = next((value for k in key_names[provider] if (value := credential(k))), None)
    if not key:
        raise ValueError(f"{provider}: credential not configured in environment")
    if provider == "alphavantage":
        day = date.fromisoformat(day or date.today().isoformat())
        if day <= date(2010, 1, 1):
            raise ValueError("Alpha Vantage listing history is documented only after 2010-01-01")
        if state not in ("active", "delisted"):
            raise ValueError("state must be active or delisted")
        reserve_alpha_request()
        response = client.get("https://www.alphavantage.co/query", params={"function": "LISTING_STATUS", "date": day.isoformat(), "state": state, "apikey": key})
        source = "https://www.alphavantage.co/documentation/#listing-status"
        if response.status_code != 200 or response.text.lstrip().startswith("{"):
            raise ValueError("Alpha Vantage denied/throttled listing request; no catalog accepted")
        catalog = pd.read_csv(io.StringIO(response.text))
        if not {"symbol", "ipoDate", "delistingDate", "exchange", "assetType"} <= set(catalog):
            raise ValueError("invalid listing catalog")
        payload = {"requested_date": day.isoformat(), "state": state, "csv": response.text}
    elif provider == "finnhub":
        if not symbol:
            raise ValueError("Finnhub historical constituents require an index symbol")
        response = client.get("https://finnhub.io/api/v1/index/historical-constituents", params={"symbol": symbol}, headers={"X-Finnhub-Token": key})
        source = "https://finnhub.io/docs/api/indices-historical-constituents"
        if response.status_code != 200:
            raise ValueError(f"Finnhub history unavailable: HTTP {response.status_code}; check subscription entitlement")
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("error") or not isinstance(payload.get("historicalConstituents"), list):
            raise ValueError("Finnhub did not return historical constituent records")
    else:
        if not query:
            raise ValueError("Exa requires a research query")
        response = client.post("https://api.exa.ai/search", headers={"x-api-key": key}, json={"query": query, "numResults": 5, "contents": {"highlights": True}})
        source = "https://exa.ai/docs/reference/search"
        if response.status_code != 200:
            raise ValueError(f"Exa unavailable: HTTP {response.status_code}")
        payload = response.json()
        if not isinstance(payload.get("results"), list):
            raise ValueError("invalid Exa response")
    now = datetime.now(timezone.utc)
    archive = C.RAW / "provider_history" / provider
    archive.mkdir(parents=True, exist_ok=True)
    path = archive / (now.strftime("%Y%m%dT%H%M%S%f") + "_" + uuid4().hex[:8] + ".json")
    # Never store request URLs/headers, which may contain provider credentials.
    doc = {"provider": provider, "source_documentation": source, "observed_at_utc": now.isoformat(),
           "status": "REQUIRES_IDENTITY_AND_COVERAGE_REVIEW", "response": payload}
    with path.open("x", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, allow_nan=False)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One provider request; uses existing entitlement, never purchases a subscription")
    parser.add_argument("provider", choices=["alphavantage", "finnhub", "exa"])
    parser.add_argument("--day")
    parser.add_argument("--state", default="delisted")
    parser.add_argument("--symbol")
    parser.add_argument("--query")
    args = parser.parse_args()
    with httpx.Client(timeout=45) as client:
        print(collect(args.provider, day=args.day, state=args.state, symbol=args.symbol, query=args.query, client=client))
