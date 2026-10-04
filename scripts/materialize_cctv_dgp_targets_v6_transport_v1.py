"""Reconstruct the unchanged V6 package from a smaller, checksum-bound transport.

No model imports, updates or changes to the frozen training recipe. Cache data
is read only. A partially assembled package can reuse only files with exact pins.
"""
import argparse
from collections import Counter
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from cctv_dgp_targets_v6 import read, reduced_target, safe_path, sha, verify, write
from cctv_camera_stress import degrade

PROTOCOL_SHA = "0fe7d036bf8567bf0860790df553afd627ac50bd5096cfda29cc7d09d71d320c"


def check_file(path, pin):
    if sha(path) != pin:
        raise ValueError("Asset fingerprint differs: " + str(path))


def materialize(root, cache, manifest_pin):
    import shutil
    start = time.monotonic()
    root, cache = Path(root), Path(cache)
    manifest_file = root / "transport_manifest_v1.json"
    check_file(manifest_file, manifest_pin)
    manifest = read(manifest_file)
    if manifest["format"] != "cctv-dgp-v6-lossless-transport-v1" or manifest["protocol_sha256"] != PROTOCOL_SHA:
        raise ValueError("Wrong transport recipe")
    check_file(root / "targets_protocol_v6.json", PROTOCOL_SHA)
    check_file(Path(__file__), manifest["materializer_sha256"])
    p = read(root / "targets_protocol_v6.json")
    if set(manifest["methods"]) != set(p["assets_sha256"]):
        raise ValueError("Transport inventory does not cover all frozen assets")
    if cache.resolve() == root.resolve():
        raise ValueError("Cache must be separate and read only")
    refs = {r["id"]: r for r in p["references"]}
    cases = {c["input"]: c for c in sum(p["training_epochs"].values(), []) + p["validation_cases"]}
    controls = {r["reduced_target"]: r for r in p["references"] if r["role"] == "train"}
    counts = Counter()
    # Canonical targets/observed masks are installed before generated derivatives.
    order = sorted(manifest["methods"], key=lambda n: manifest["methods"][n]["method"] in ("camera", "reduce"))
    for name in order:
        if time.monotonic() - start > 180:
            raise TimeoutError("Three-minute data materialization cap exceeded")
        item = manifest["methods"][name]
        target, pin = safe_path(root, name), p["assets_sha256"][name]
        method = item["method"]
        if target.exists():
            check_file(target, pin)
            counts["already_verified"] += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if method == "ship":
            raise ValueError("Missing transmitted asset: " + name)
        if method == "cache":
            source = safe_path(cache, item["source"])
            check_file(source, pin)
            shutil.copyfile(source, target)
        elif method == "reduce":
            if name not in controls:
                raise ValueError("Undeclared reduced target")
            with Image.open(root / controls[name]["target"]) as canonical:
                rgb = np.asarray(reduced_target(canonical))
            Image.fromarray(rgb).save(target)
        elif method == "camera":
            if name not in cases:
                raise ValueError("Undeclared camera input")
            case = cases[name]
            ref = refs[case["reference_id"]]
            with Image.open(root / ref["target"]) as canonical:
                rgb, details = degrade(np.asarray(canonical.convert("RGB")), ref["bounds"], case["camera"], case["proxy_details"]["seed"])
            if details != case["proxy_details"]:
                raise ValueError("Camera provenance differs")
            Image.fromarray(rgb).save(target)
        else:
            raise ValueError("Unknown transport method")
        check_file(target, pin)
        counts[method] += 1
    verify(root)
    receipt = {"complete": True, "protocol_sha256": PROTOCOL_SHA,
        "transport_manifest_sha256": manifest_pin, "materializer_sha256": sha(Path(__file__)),
        "all_frozen_asset_sha256_verified": len(p["assets_sha256"]), "methods": dict(counts),
        "seconds": time.monotonic() - start, "model_forwards": 0, "backward_calls": 0,
        "optimizer_updates": 0, "cache_read_only": True, "training_recipe_changed": False}
    write(root / "transport_materialization_v1.json", receipt)
    print(receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    materialize(args.root, args.cache_root, args.manifest_sha256)
