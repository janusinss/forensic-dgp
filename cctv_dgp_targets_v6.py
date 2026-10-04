"""Frozen target-bandwidth experiment; verification never performs model updates."""
import copy
import hashlib
import json
import math
import platform
from pathlib import Path, PurePosixPath
import sys

import numpy as np
from PIL import Image

FORMAT = "cctv-dgp-target-bandwidth-v6"
SEED = 20261004
ARMS = [
    {"id": "reduced_target", "target_key": "reduced_target", "lambda_identity": .1},
    {"id": "hq_target", "target_key": "target", "lambda_identity": .1},
]
START_SHA = "b30aeabecc60dff2fbd289575ce8691be9e670c653cacb6721bc154b618c3916"
START_STATE = "d89f4e52777b5260e94a3e533aa21a0204cbd79b578137881fc6a4c67f1bb0c3"
LINEAGE = {
    "lineage/parent_protocol.json": "b652914335e33375f8108ba427e4b5be99fd86e811f455b307ab72ae7cf152f6",
    "lineage/eligible_references.json": "cadeedd2906fe169b7cc9741aac9fc29db8aedc1ba21d4438f97443490c1281e",
    "lineage/source_review_audit.json": "d3fb788eb775d2c178179e283b7a353653473ed088b2a19141ec02e6467078d4",
    "lineage/catalog_audit.json": "d4989390d2c69bcd3c3abb8169aa1c02b4ad6a50aed3c5575b544aa93e96c0c0",
    "lineage/catalog_results.json": "7388a67b655d8259db8a30cefcdfa5b520597e80fd71bfd1d5952051fac98370",
    "lineage/source_plan.json": "ad65e35c70104b72a38dfe3cdac1f1027029445313f9d5530dbb18af7cde9501",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def safe_path(root, name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError("Unsafe asset path")
    item = PurePosixPath(name)
    if item.is_absolute() or ".." in item.parts:
        raise ValueError("Unsafe asset path")
    path = Path(root) / name
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Asset escapes bundle")
    return path


def reduced_target(image):
    """Hold canonical RGB/framing fixed; remove bandwidth with declared LANCZOS."""
    if image.mode != "RGB" or image.size != (256, 256):
        raise ValueError("Reduced control requires canonical RGB256")
    return image.resize((128, 128), Image.Resampling.LANCZOS).resize(
        (256, 256), Image.Resampling.LANCZOS)


def arm_protocol(protocol, arm):
    """Only reconstruction targets change; common identity targets stay canonical."""
    if arm not in ARMS:
        raise ValueError("Unknown target arm")
    result = copy.deepcopy(protocol)
    for ref in result["references"]:
        if ref["role"] == "train":
            ref["target"] = ref[arm["target_key"]]
    return result


def require_vm(root):
    # This guard must precede every model/backward/training entry point.
    if sys.platform != "linux" or platform.node().split(".")[0] != "forensic-dgp-thesis":
        raise RuntimeError("Actual training/backward is permitted only on forensic-dgp-thesis Linux VM")
    if not Path(root).resolve().is_relative_to((Path.home() / "forensic-dgp").resolve()):
        raise RuntimeError("Training bundle must be under ~/forensic-dgp")
    import torch
    if not torch.cuda.is_available() or "L4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Existing CUDA NVIDIA L4 required")
    if torch.cuda.get_device_properties(0).total_memory < 8 * 1024**3:
        raise RuntimeError("At least 8 GiB device VRAM required")


def verify(root, rebuild_controls=True):
    root = Path(root)
    if sha(root / "targets_protocol_v6.json") != (root / "targets_protocol_v6.sha256").read_text().strip():
        raise ValueError("V6 protocol fingerprint differs")
    p = read(root / "targets_protocol_v6.json")
    if p["format"] != FORMAT or p["arms"] != ARMS:
        raise ValueError("Frozen target factor differs")
    if (p["seed"], p["epochs"], p["batch_size"], p["runtime_cap_seconds"],
        p["updates_per_arm"], p["expected_total_updates"]) != (SEED, 2, 8, 1200, 98, 196):
        raise ValueError("Finite V6 budget differs")
    if p["starting_checkpoint_sha256"] != START_SHA or p["starting_state_hash"] != START_STATE:
        raise ValueError("Starting baseline differs")
    if p["selection"] != "unchanged_strict_source_profile_png_guard_plus_0.1dB":
        raise ValueError("Selection guard differs")
    if p["native_reserved_used"] or p["production_promotion_permitted"]:
        raise ValueError("Development pilot cannot use reserved cases/promote production")
    for name, pin in p["assets_sha256"].items():
        if sha(safe_path(root, name)) != pin:
            raise ValueError("Changed asset: " + name)
    for name, pin in LINEAGE.items():
        if p["assets_sha256"].get(name) != pin:
            raise ValueError("Lineage fingerprint differs: " + name)
    if p["assets_sha256"].get(p["weights"]["start"]) != START_SHA:
        raise ValueError("Starting checkpoint asset differs")
    parent = read(root / "lineage/parent_protocol.json")
    if p["optimizer"] != parent["optimizer"] or p["loss"] != parent["loss"]:
        raise ValueError("Matched optimizer/reconstruction objective changed")
    for key in ("arcface", "vgg_trunk"):
        if p["weights"][key] != parent["weights"][key] or p["assets_sha256"][p["weights"][key]] != parent["assets_sha256"][parent["weights"][key]]:
            raise ValueError("Frozen teacher asset differs")
    old = {r["id"]: r for r in parent["references"]}
    eligible = read(root / "lineage/eligible_references.json")
    approved = {r["reference_id"]: r for r in eligible["references"]}
    sentinels = {r["id"] for r in parent["references"]
                 if r["role"] == "validation" and r["source"] == "dataset/asian_faces"}
    refs = {r["id"]: r for r in p["references"]}
    if len(refs) != len(p["references"]) or set(refs) != set(approved) | sentinels:
        raise ValueError("Reviewed reference IDs/sentinel cohort differ")
    train = {r["id"] for r in p["references"] if r["role"] == "train"}
    val = set(refs) - train
    if len(train) != 391 or len(val) != 104 or len(sentinels) != 51 or train & val:
        raise ValueError("Split integrity differs")
    for ref in refs.values():
        historical = old[ref["id"]]
        for key in ("role", "source", "matrix112", "bounds"):
            if ref[key] != historical[key]:
                raise ValueError("Historical role/framing changed: " + ref["id"])
        if ref["identity_target"] != ref["target"]:
            raise ValueError("Identity target must be common canonical target")
        if ref["id"] in approved:
            if sha(root / ref["target"]) != approved[ref["id"]]["target_sha256"]:
                raise ValueError("Reviewed HQ target binding differs")
            if rebuild_controls and ref["role"] == "train":
                with Image.open(root / ref["target"]) as image:
                    expected = np.asarray(reduced_target(image))
                with Image.open(root / ref["reduced_target"]) as image:
                    np.testing.assert_array_equal(np.asarray(image), expected)
        else:
            for key in ("target", "observed"):
                if sha(root / ref[key]) != parent["assets_sha256"][historical[key]]:
                    raise ValueError("Asian sentinel asset changed")
    profile_ids = {x["id"] for x in p["evaluation_profiles"]}
    if p["evaluation_profiles"] != parent["validation_camera_profiles"]:
        raise ValueError("Historical evaluation camera profiles changed")
    cases = p["validation_cases"]
    pairs = {(c["reference_id"], c["profile"]) for c in cases}
    if len(cases) != 520 or pairs != {(r, x) for r in val for x in profile_ids}:
        raise ValueError("Common validation case grid differs")
    if len({c["id"] for c in cases}) != 520:
        raise ValueError("Duplicate validation ID")
    old_cases = {c["id"]: c for c in parent["validation_cases"]}
    for case in cases:
        if case["reference_id"] in sentinels:
            previous = old_cases[case["id"]]
            for key in ("id", "reference_id", "source", "profile", "camera", "proxy_details"):
                if case[key] != previous[key]:
                    raise ValueError("Asian sentinel case changed")
            if sha(root / case["input"]) != parent["assets_sha256"][previous["input"]]:
                raise ValueError("Asian sentinel input changed")
    for epoch in (1, 2):
        cases = p["training_epochs"][str(epoch)]
        if len(cases) != 391 or {c["reference_id"] for c in cases} != train:
            raise ValueError("Training epoch includes held-out/missing reference")
        if len({c["id"] for c in cases}) != 391:
            raise ValueError("Duplicate training case")
        if math.ceil(len(cases) / p["batch_size"]) != 49:
            raise ValueError("Training tail/budget differs")
    if len(p["preview_case_ids"]) != 10 or len(set(p["preview_case_ids"])) != 10:
        raise ValueError("Fixed 10-row preview differs")
    if not set(p["preview_case_ids"]) <= {c["id"] for c in p["validation_cases"]}:
        raise ValueError("Preview is not validation-only")
    return p
