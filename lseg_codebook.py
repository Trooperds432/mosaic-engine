"""Run in signed-in LSEG CodeBook; export bounded, auditable research data.

No credentials, purchases, orders, inferred release timestamps or PIT certification.
The default universe is an entitlement sample, not a constituent-complete backtest.
"""
import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

VERSION = "1.0"
EQUITIES = ["AAPL.O", "MSFT.O", "NVDA.O", "JPM.N", "XOM.N", "PFE.N"]


def default_requests():
    """Explicit public-instrument sample. Expand only after reviewing coverage."""
    jobs = [
        ("identifiers", "data", EQUITIES,
         ["TR.TickerSymbol", "TR.ISIN", "TR.OrganizationID", "TR.CommonName"], {}),
        ("consensus_snapshot", "data", EQUITIES,
         ["TR.EPSMean", "TR.EPSMean.periodenddate", "TR.EPSHigh", "TR.EPSLow",
          "TR.EPSStdDev", "TR.EPSNumOfEst", "TR.RevenueMean", "TR.CapexMean"], {"Period": "FY1"}),
        ("company_guidance", "data", ["NVDA.O", "MSFT.O", "PFE.N"],
         ["TR.GuidanceDate", "TR.GuidanceMeasure", "TR.GuidancePeriodYear",
          "TR.GuidanceText", "TR.GuidanceDocType"], {"Period": "FY1"}),
        ("banking_snapshot", "data", ["JPM.N", "BAC.N", "C.N"],
         ["TR.F.DeposCustTotASR", "TR.F.DeposCustTotASR.date"], {}),
        ("dividends", "data", EQUITIES,
         ["TR.DivAnnouncementDate", "TR.DivExDate", "TR.DivPayDate", "TR.DivUnadjustedGross", "TR.DivCurr", "TR.DivType"],
         {"SDate": "2020-01-01", "EDate": "2026-10-23", "DateType": "ED"}),
    ]
    # One instrument per time-series request avoids cross-instrument/date ambiguity.
    for ric in EQUITIES:
        jobs.append(("consensus_history_" + re.sub(r"\W", "_", ric), "history", ric,
                     ["TR.EPSMean", "TR.EPSHigh", "TR.EPSLow", "TR.RevenueMean"],
                     {"start": "2020-01-01", "end": "2026-10-23", "interval": "1M", "parameters": {"Period": "FY1"}}))
    for index in [".SPX", ".NDX", ".RUT"]:
        for direction in ["J", "L"]:
            jobs.append(("index_" + index[1:] + "_" + direction, "data", index,
                         ["TR.IndexJLConstituentChangeDate", "TR.IndexJLConstituentRIC", "TR.IndexJLConstituentName"],
                         {"SDate": "2020-01-01", "EDate": "2026-10-23", "IC": direction}))
    for ric in ["CLZ6", "CLF7", "NGZ6", "NGF7", "GCZ6", "GCG7"]:
        jobs.append(("contract_" + ric, "history", ric, ["SETTLE", "TRDPRC_1", "ACVOL_UNS", "OPINT_1"],
                     {"start": "2026-04-01", "end": "2026-10-23", "interval": "1D"}))
    return jobs


def _batches(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def full_requests(universe, start="2012-01-01", end=None, snapshot_batch=200, history_batch=10, membership_from=2009):
    """Universe-wide capture for building a point-in-time archive (run on a schedule; each run is one snapshot).

    universe: list of RICs, or the dict from lseg_universe.json. Batches keep each request bounded; a batch LSEG rejects
    is recorded as UNAVAILABLE and can be re-run with a smaller size. Datasets:
      identifiers / consensus_snapshot          current reference data and FY1 consensus (batches of snapshot_batch)
      consensus_hist_NNN                        monthly FY1 EPS/revenue consensus history with its calculation date
      index_SPX/NDX_J/L                         dated joiners / leavers since membership_from
      members_SPX/NDX_YYYYQq                    quarter-end constituent lists (historical membership snapshots)
    """
    rics = universe["rics"] if isinstance(universe, dict) else list(universe)
    end = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    jobs = []
    for i, part in enumerate(_batches(rics, snapshot_batch)):
        jobs.append((f"identifiers_{i:03d}", "data", part, ["TR.TickerSymbol", "TR.ISIN", "TR.OrganizationID", "TR.CommonName"], {}))
        jobs.append((f"consensus_snapshot_{i:03d}", "data", part,
                     ["TR.EPSMean", "TR.EPSMean.periodenddate", "TR.EPSHigh", "TR.EPSLow", "TR.EPSStdDev", "TR.EPSNumOfEst",
                      "TR.RevenueMean", "TR.RevenueNumOfEst"], {"Period": "FY1"}))
    for i, part in enumerate(_batches(rics, history_batch)):
        jobs.append((f"consensus_hist_{i:03d}", "data", part,
                     ["TR.EPSMean.calcdate", "TR.EPSMean", "TR.EPSMean.periodenddate", "TR.EPSStdDev", "TR.EPSNumOfEst", "TR.RevenueMean"],
                     {"SDate": start, "EDate": end, "Frq": "M", "Period": "FY1"}))
    for index in [".SPX", ".NDX"]:
        for direction in ["J", "L"]:
            jobs.append((f"index_{index[1:]}_{direction}", "data", index,
                         ["TR.IndexJLConstituentChangeDate", "TR.IndexJLConstituentRIC", "TR.IndexJLConstituentName"],
                         {"SDate": f"{membership_from}-01-01", "EDate": end, "IC": direction}))
        for year in range(membership_from, int(end[:4]) + 1):
            for q, md in enumerate(["03-31", "06-30", "09-30", "12-31"], 1):
                day = f"{year}-{md}"
                if day <= end:
                    jobs.append((f"members_{index[1:]}_{year}Q{q}", "data", index,
                                 ["TR.IndexConstituentRIC", "TR.IndexConstituentName"], {"SDate": day}))
    return jobs


def export(destination=None, requests=None):
    import lseg.data as ld
    ld.open_session()  # CodeBook supplies the session; never paste an API key here.
    started = datetime.now(timezone.utc)
    root = Path(destination or ("mosaic_lseg_" + started.strftime("%Y%m%dT%H%M%SZ")))
    root.mkdir(exist_ok=False)
    manifest = {"schema_version": VERSION, "provider": "LSEG CodeBook", "generated_at_utc": started.isoformat(),
                "universe_scope": ("six-equity entitlement sample; index changes and selected futures contracts" if requests is None
                                   else f"explicit request list: {len(requests)} requests"),
                "historical_pit_certified": False, "datasets": []}
    for name, method, universe, fields, parameters in (requests if requests is not None else default_requests()):
        record = {"name": name, "method": method, "universe": universe, "requested_fields": fields,
                  "parameters": parameters, "historical_pit_certified": False,
                  "source_id": "LSEG/" + name, "availability_basis": "observed_snapshot"}
        try:
            if method == "history":
                frame = ld.get_history(universe=universe, fields=fields, **parameters)
                frame = frame.rename_axis("observation_date").reset_index()
                frame.insert(0, "Instrument", universe)
            else:
                frame = ld.get_data(universe=universe, fields=fields, parameters=parameters)
            if name.startswith("index_"):
                import pandas as pd
                # Some index endpoints return a current name with no change date.
                # Such a placeholder is not a historical membership event.
                if "Date" not in frame or pd.to_datetime(frame["Date"], errors="coerce").notna().sum() == 0:
                    raise ValueError("no dated membership changes returned")
                frame = frame[pd.to_datetime(frame["Date"], errors="coerce").notna()].copy()
            observed = datetime.now(timezone.utc).isoformat()
            record.update(observed_at_utc=observed, available_at_utc=observed,
                          rows=len(frame), columns=[str(c) for c in frame.columns])
            value_columns = [c for c in frame if str(c) not in ("Instrument", "observation_date")]
            record["non_null"] = {str(c): int(frame[c].notna().sum()) for c in value_columns}
            record["status"] = "OBSERVED" if any(record["non_null"].values()) else "EMPTY"
            frame.columns = [str(c) for c in frame.columns]
            payload = frame.to_csv(index=False).encode("utf-8")
            filename = name + ".csv"
            (root / filename).write_bytes(payload)
            record.update(file=filename, sha256=hashlib.sha256(payload).hexdigest())
            print(name, record["status"], len(frame), record["non_null"])
        except Exception as exc:
            # Avoid serializing URLs, headers, session objects or arbitrary exceptions.
            error_text = str(exc).lower()
            category = "ACCESS_DENIED" if any(s in error_text for s in ["notentitled", "access denied", "permission"]) else "REQUEST_OR_CONTENT_UNAVAILABLE"
            record.update(status="UNAVAILABLE", error_type=type(exc).__name__, error_category=category)
            print(name, "UNAVAILABLE", type(exc).__name__)
        manifest["datasets"].append(record)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    bundle = root.with_suffix(".zip")
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.iterdir()):
            archive.write(path, path.name)
    print("BUNDLE", bundle)
    return bundle


if __name__ == "__main__":
    export()
