"""Import CodeBook research bundles without certifying revised history as PIT.

The research store is separate from production prices/facts and model inputs.
Use observations_asof to retrieve only information actually archived by a decision.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import pandas as pd
import config as C

MAX_BYTES = 100_000_000


def read_bundle(path):
    payload = Path(path).read_bytes()
    if len(payload) > MAX_BYTES:
        raise ValueError("LSEG archive exceeds bounded import size")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(Path(n).name != n or "\\" in n for n in names):
            raise ValueError("duplicate or unsafe archive paths")
        if sum(i.file_size for i in archive.infolist()) > MAX_BYTES:
            raise ValueError("expanded archive exceeds bounded import size")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("schema_version") != "1.0" or manifest.get("provider") != "LSEG CodeBook":
            raise ValueError("unsupported LSEG manifest")
        if manifest.get("historical_pit_certified") is not False:
            raise ValueError("CodeBook exports cannot self-certify PIT history")
        datasets = manifest["datasets"]
        if len({d["name"] for d in datasets}) != len(datasets):
            raise ValueError("duplicate dataset names")
        frames = []
        for item in datasets:
            if item["status"] == "UNAVAILABLE":
                continue
            if item["status"] not in {"OBSERVED", "EMPTY"}:
                raise ValueError("unknown dataset status")
            filename = item["file"]
            raw = archive.read(filename)
            if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise ValueError("LSEG file checksum mismatch")
            observed = pd.Timestamp(item["observed_at_utc"])
            available = pd.Timestamp(item["available_at_utc"])
            if observed.tzinfo is None or available.tzinfo is None:
                raise ValueError("availability timestamps require timezone")
            if available < observed or observed > pd.Timestamp.now(tz="UTC"):
                raise ValueError("backdated snapshot or future observation")
            if item.get("availability_basis") != "observed_snapshot" or item.get("historical_pit_certified") is not False:
                raise ValueError("historical availability is unverified")
            frame = pd.read_csv(io.BytesIO(raw), dtype=str)
            if len(frame) != item["rows"] or list(frame) != item["columns"]:
                raise ValueError("LSEG manifest/frame mismatch")
            if "Instrument" not in frame:
                raise ValueError("LSEG rows require an explicit RIC")
            if item["name"].startswith("index_") and ("Date" not in frame or pd.to_datetime(frame["Date"], errors="coerce").isna().any()):
                raise ValueError("index changes require actual event dates")
            requested = item["universe"]
            requested = requested if isinstance(requested, list) else [requested]
            if frame["Instrument"].isna().any() or not frame["Instrument"].isin(requested).all():
                raise ValueError("unexpected or missing instrument identity")
            frame["source_row"] = range(len(frame))
            ids = ["Instrument", "source_row"]
            if "observation_date" in frame:
                ids.append("observation_date")
            long = frame.melt(id_vars=ids, var_name="field", value_name="value").dropna(subset=["value"])
            long["dataset"] = item["name"]
            long["source_id"] = item["source_id"]
            long["observed_at_utc"] = observed.tz_convert("UTC")
            long["available_at_utc"] = available.tz_convert("UTC")
            long["availability_basis"] = "observed_snapshot"
            long["historical_pit_certified"] = False
            long["bundle_sha256"] = hashlib.sha256(payload).hexdigest()
            frames.append(long)
        data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return payload, manifest, data


def import_bundle(path):
    payload, manifest, frame = read_bundle(path)
    digest = hashlib.sha256(payload).hexdigest()
    archive = C.RAW / "lseg"
    processed = C.PROC / "lseg"
    archive.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    target = archive / (digest + ".zip")
    if not target.exists():
        with target.open("xb") as stream:
            stream.write(payload)
    output = processed / (digest + ".parquet")
    if not output.exists():
        temporary = output.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(output)
    report = C.REPORTS / "lseg_coverage.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    coverage = {"bundle_sha256": digest, "historical_pit_certified": False,
                "scope": manifest["universe_scope"], "observations": len(frame),
                "datasets": manifest["datasets"],
                "limitation": "Historical values are vendor retrievals; first-release timestamps and identity mappings remain unverified."}
    temporary = report.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(coverage, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(report)
    return output


def observations_asof(decision, dataset=None):
    """Research API: all snapshots known by decision; no backfill or zero imputation.

    A historical observation_date never overrides the actual archive availability.
    This does not certify the release time, source units or permanent identifier.
    """
    decision = pd.Timestamp(decision)
    if decision.tzinfo is None:
        raise ValueError("decision must be timezone-aware")
    frames = [pd.read_parquet(p) for p in sorted((C.PROC / "lseg").glob("*.parquet"))]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    result = frame[frame["available_at_utc"] <= decision]
    if dataset is not None:
        result = result[result["dataset"] == dataset]
    return result.reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", help="ZIP produced by lseg_codebook.py")
    args = parser.parse_args()
    import os
    if os.name == "nt":
        # The engine's general lock uses Linux fcntl. Use the same lock file
        # with Windows byte locking for this standalone import command.
        import msvcrt
        _lock = (C.ROOT / ".lock").open("a+b")
        if _lock.tell() == 0:
            _lock.write(b"0")
            _lock.flush()
        _lock.seek(0)
        msvcrt.locking(_lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        _lock = C.hold_lock()
    print(import_bundle(args.bundle))
