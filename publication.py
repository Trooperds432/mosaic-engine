"""One strict JSON/schema boundary for nightly and intraday signal publication."""
import json
import os
import tempfile
from pathlib import Path

import jsonschema

SCHEMA = json.loads((Path(__file__).parent / "schema" / "signal_v2.schema.json").read_text())
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA, format_checker=jsonschema.FormatChecker())


def validate_signal(signal):
    try:
        json.dumps(signal, allow_nan=False)
        VALIDATOR.validate(signal)
    except (ValueError, TypeError, jsonschema.ValidationError) as exc:
        raise ValueError(f"invalid signal: {exc}") from exc


def publish(doc, directory, log_id):
    """Validate *all* entries before writing; exclusive history and atomic latest.

    Caller holds the engine lock. Failed validation leaves both outputs untouched.
    A failed atomic replace can leave an audit log but never a partial latest file.
    """
    if not isinstance(doc.get("signals"), list):
        raise ValueError("signals must be a list")
    for signal in doc["signals"]:
        validate_signal(signal)
    payload = json.dumps(doc, indent=1, allow_nan=False)
    history = "".join(json.dumps(s, allow_nan=False) + "\n" for s in doc["signals"])
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    (root / "log").mkdir(exist_ok=True)
    if not log_id or Path(log_id).name != log_id or any(c in log_id for c in "\\/:"):
        raise ValueError("log_id must be a plain filename stem")
    with (root / "log" / f"{log_id}.jsonl").open("x", encoding="utf-8") as f:
        f.write(history)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, suffix=".tmp", delete=False) as f:
            name = f.name
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, root / "latest.json")
    finally:
        if name and Path(name).exists():
            Path(name).unlink()
