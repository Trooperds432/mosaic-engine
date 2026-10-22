"""Immutable, content-addressed research recipes and separately versioned weights."""
import argparse
import hashlib
import json
from datetime import datetime, timezone

import config as C


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(C.ROOT.glob("*.py"))}


def specification(model):
    if model.get("model_id") == "insider-event":
        spec = {k: model.get(k) for k in ["model_id", "model_version", "run_id", "prereg_sha256", "horizon", "cost", "gate", "distribution_net"]}
        spec["sources"] = source_hashes()
        spec["requirements_sha256"] = hashlib.sha256((C.ROOT / "requirements.txt").read_bytes()).hexdigest()
        return spec
    keys = ["model_id", "model_version", "feature_set_version", "run_id", "profile", "features", "feats", "order_size_usd"]
    spec = {k: model.get(k) for k in keys}
    spec["horizons"] = {h: {k: doc.get(k) for k in ["members", "names", "calibration", "gate"]} for h, doc in model["horizons"].items()}
    spec["sources"] = source_hashes()
    spec["requirements_sha256"] = hashlib.sha256((C.ROOT / "requirements.txt").read_bytes()).hexdigest()
    return spec


def write_exclusive(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, allow_nan=False)


def snapshot_weights(model):
    hashes = {}
    for h, cfg in model.get("horizons", {}).items():
        for name in cfg["names"]:
            path = C.ROOT / "models" / f"{model['profile']}_h{h}_{name}.joblib"
            content = path.read_bytes()
            weight_hash = hashlib.sha256(content).hexdigest()
            hashes[str(path.relative_to(C.ROOT)).replace("\\", "/")] = weight_hash
            archived = C.ROOT / "models" / "snapshots" / f"{weight_hash}.joblib"
            archived.parent.mkdir(parents=True, exist_ok=True)
            if not archived.exists():
                with archived.open("xb") as f:
                    f.write(content)
    doc = {"recipe_id": model["recipe_id"], "files": hashes}
    stamp = digest(doc)
    path = C.ROOT / "prereg" / "weights" / f"{stamp}.json"
    if not path.exists():
        write_exclusive(path, doc)
    model["weights_id"] = stamp


def freeze(profile):
    path = C.PROC / ("insider_model.json" if profile == "insider-event" else f"model_{profile}.json")
    model = json.loads(path.read_text())
    if model.get("recipe_id"):
        raise ValueError("already frozen: create a new validation recipe/run before changing the contract")
    spec = specification(model)
    stamp = datetime.now(timezone.utc).isoformat()
    doc = {"specification": spec, "frozen_at_utc": stamp}
    identity = digest(doc)
    model.update(recipe_id=identity, forward_start_utc=stamp)
    snapshot_weights(model)
    write_exclusive(C.ROOT / "prereg" / "recipes" / f"{identity}.json", doc)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(model, indent=2, allow_nan=False))
    temporary.replace(path)
    return identity


def check(model, weights=True):
    try:
        identity = model["recipe_id"]
        if not isinstance(identity, str) or len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
            raise ValueError("invalid recipe ID")
        doc = json.loads((C.ROOT / "prereg" / "recipes" / f"{identity}.json").read_text())
        if digest(doc) != identity or doc["specification"] != specification(model):
            return False, "frozen recipe/source differs: revalidation and a new forward record required"
        if model["forward_start_utc"] != doc["frozen_at_utc"]:
            return False, "forward start differs from frozen recipe"
        if weights:
            wid = model["weights_id"]
            if not isinstance(wid, str) or len(wid) != 64 or any(c not in "0123456789abcdef" for c in wid):
                raise ValueError("invalid weights ID")
            wdoc = json.loads((C.ROOT / "prereg" / "weights" / f"{wid}.json").read_text())
            if digest(wdoc) != wid or wdoc["recipe_id"] != identity:
                return False, "weights not bound to frozen recipe"
            for name, expected in wdoc["files"].items():
                file = (C.ROOT / name).resolve()
                if not file.is_relative_to((C.ROOT / "models").resolve()) or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
                    return False, "weights changed without a recorded frozen-recipe refit"
        return True, None
    except (OSError, ValueError, KeyError, TypeError):
        return False, "frozen recipe or weights manifest missing/invalid"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=list(C.PROFILES) + ["insider-event"])
    args = parser.parse_args()
    _lock = C.hold_lock()
    print(freeze(args.profile))
