"""Prepare target-only matched V6 package. No model forwards or training."""
import ast
import copy
from pathlib import Path
import shutil
import sys
import tarfile
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from cctv_dgp_targets_v6 import (ARMS, FORMAT, LINEAGE, SEED, START_SHA, START_STATE,
                                  read, reduced_target, sha, verify, write)
from prepare_cctv_dgp_vm import jitter
from cctv_camera_stress import degrade

BASE = ROOT / "outputs/cctv_dgp_vm_bundle_v1"
DEST = ROOT / "outputs/cctv_dgp_targets_vm_v6"
ARCHIVE = ROOT / "outputs/cctv-dgp-targets-v6.tar.gz"
PREFIX = "cctv_dgp_targets_vm_v6"
SOURCE_FILES = ["cctv_dgp_targets_v6.py", "cctv_dgp_pilot.py", "cctv_dgp_frozen_norm.py",
                "scripts/prepare_cctv_dgp_targets_v6.py", "scripts/run_cctv_dgp_targets_vm_v6.py",
                "scripts/audit_cctv_dgp_targets_v6.py", "scripts/run_cctv_dgp_targets_vm_v6.sh"]


def copy_file(source, name):
    target = DEST / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError("Refuse overwriting prepared asset: " + name)
    shutil.copyfile(source, target)
    return name


def save_rgb(rgb, name):
    target = DEST / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError("Refuse overwriting prepared image")
    Image.fromarray(rgb).save(target)
    return name


def main():
    start = time.monotonic()
    if DEST.exists() or ARCHIVE.exists() or Path(str(ARCHIVE) + ".sha256").exists():
        raise ValueError("Preserve existing V6 preparation; no automatic overwrite")
    paths = {
        "lineage/parent_protocol.json": BASE / "protocol.json",
        "lineage/eligible_references.json": ROOT / "outputs/cctv_dgp_hq_source_review_v3/eligible_references.json",
        "lineage/source_review_audit.json": ROOT / "outputs/cctv_dgp_hq_source_review_v3/source_review_audit.json",
        "lineage/catalog_audit.json": ROOT / "outputs/cctv_dgp_hq_cohort_v2/local_independent_audit.json",
        "lineage/catalog_results.json": ROOT / "outputs/cctv_dgp_hq_cohort_v2/results.json",
        "lineage/source_plan.json": ROOT / "outputs/cctv_dgp_hq_cohort_plan_v2/manifest.json",
    }
    for name, source in paths.items():
        if sha(source) != LINEAGE[name]:
            raise ValueError("Source lineage changed: " + name)
    for name in SOURCE_FILES:
        if name.endswith(".py"):
            ast.parse((ROOT / name).read_text(encoding="utf-8"), feature_version=(3, 10))
        elif b"\r" in (ROOT / name).read_bytes():
            raise ValueError("Launcher requires LF")
    parent = read(paths["lineage/parent_protocol.json"])
    eligible = read(paths["lineage/eligible_references.json"])
    old = {r["id"]: r for r in parent["references"]}
    if eligible["eligible_by_role"] != {"train": 391, "validation": 53}:
        raise ValueError("Source-reviewed cohort differs")
    DEST.mkdir()
    for name, path in paths.items():
        copy_file(path, name)
    for name in SOURCE_FILES:
        copy_file(ROOT / name, name)
    for path in sorted((BASE / "models").glob("*.py")):
        copy_file(path, "models/" + path.name)
    copy_file(BASE / "scripts/cctv_camera_stress.py", "scripts/cctv_camera_stress.py")
    copy_file(ROOT / "scripts/prepare_cctv_dgp_vm.py", "scripts/prepare_cctv_dgp_vm.py")
    start_path = ROOT / "outputs/cctv_dgp_normfix_return_v2/outputs/cctv_dgp_pilot/camera_identity/best.pth"
    if sha(start_path) != START_SHA:
        raise ValueError("Audited starting weights differ")
    weights = {"start": copy_file(start_path, "weights/start_v2.pth")}
    for key in ("arcface", "vgg_trunk"):
        file = parent["weights"][key]
        if sha(BASE / file) != parent["assets_sha256"][file]:
            raise ValueError("Teacher changed")
        weights[key] = copy_file(BASE / file, file)
    references = []
    checked_sources = []
    for approved in eligible["references"]:
        ref = copy.deepcopy(old[approved["reference_id"]])
        source = ROOT / approved["source_path_from_workspace"]
        target = ROOT / approved["target_path_from_workspace"]
        if sha(source) != approved["source_sha256"] or sha(target) != approved["target_sha256"]:
            raise ValueError("HQ source/target changed")
        with Image.open(source) as image:
            rebuilt = image.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
        with Image.open(target) as image:
            np.testing.assert_array_equal(np.asarray(image), np.asarray(rebuilt))
        ref["target"] = copy_file(target, "data/targets/" + ref["id"] + ".png")
        ref["observed"] = copy_file(BASE / ref["observed"], ref["observed"])
        ref["identity_target"] = ref["target"]
        ref["evaluation_source_kind"] = "HQ_FFHQ_counterpart"
        ref["hq_source_sha256"] = approved["source_sha256"]
        if ref["role"] == "train":
            ref["reduced_target"] = save_rgb(np.asarray(reduced_target(rebuilt)), "data/reduced_targets/" + ref["id"] + ".png")
        references.append(ref)
        checked_sources.append({"id": ref["id"], "source_sha256": sha(source), "target_sha256": sha(target),
                                "canonical_target_rebuilt_from_1024": True})
    sentinels = [copy.deepcopy(r) for r in parent["references"]
                 if r["role"] == "validation" and r["source"] == "dataset/asian_faces"]
    for ref in sentinels:
        for key in ("target", "observed"):
            copy_file(BASE / ref[key], ref[key])
        ref["identity_target"] = ref["target"]
        ref["evaluation_source_kind"] = "unchanged_Asian_regression_sentinel"
        references.append(ref)
    refs = {r["id"]: r for r in references}
    train = [r for r in references if r["role"] == "train"]
    profiles = parent["validation_camera_profiles"]
    epochs = {}
    for epoch in (1, 2):
        rng = np.random.default_rng(SEED + epoch)
        cases = []
        for slot, index in enumerate(rng.permutation(len(train))):
            ref = train[index]
            profile = jitter(profiles[(slot + epoch - 1) % 5], rng)
            noise_seed = int(rng.integers(0, 2**32))
            target = np.asarray(Image.open(DEST / ref["target"]))
            rgb, details = degrade(target, ref["bounds"], profile, noise_seed)
            case_id = f"e{epoch}_{ref['id']}"
            cases.append({"id": case_id, "reference_id": ref["id"], "source": ref["source"],
                          "profile": profile["id"], "input": save_rgb(rgb, f"data/inputs/{case_id}.png"),
                          "camera": profile, "proxy_details": details})
        epochs[str(epoch)] = cases
    validation = []
    for original in parent["validation_cases"]:
        if original["reference_id"] not in refs or refs[original["reference_id"]]["role"] != "validation":
            continue
        case = copy.deepcopy(original)
        if case["reference_id"] in {r["id"] for r in sentinels}:
            copy_file(BASE / case["input"], case["input"])
        else:
            ref = refs[case["reference_id"]]
            rgb, details = degrade(np.asarray(Image.open(DEST / ref["target"])), ref["bounds"],
                                   case["camera"], case["proxy_details"]["seed"])
            case["input"] = save_rgb(rgb, f"data/validation/{case['id']}.png")
            case["proxy_details"] = details
        validation.append(case)
    preview = []
    for source in ("dataset/thumbnails128x128", "dataset/asian_faces"):
        first = next(r for r in references if r["role"] == "validation" and r["source"] == source)
        preview.extend([c["id"] for c in validation if c["reference_id"] == first["id"]])
    write(DEST / "hq_source_derivation_receipt.json", {"complete": True, "references": checked_sources,
          "model_forwards": 0, "backward_calls": 0, "optimizer_updates": 0,
          "native_reserved_used": False, "original_roles_preserved": True})
    pins = {p.relative_to(DEST).as_posix(): sha(p) for p in sorted(DEST.rglob("*")) if p.is_file()}
    protocol = {"format": FORMAT, "date": "2026-10-04", "seed": SEED, "arms": ARMS,
        "references": references, "training_epochs": epochs, "validation_cases": validation,
        "evaluation_profiles": profiles, "preview_case_ids": preview, "epochs": 2, "batch_size": 8,
        "updates_per_arm": 98, "expected_total_updates": 196, "runtime_cap_seconds": 1200,
        "weights": weights, "starting_checkpoint_sha256": START_SHA, "starting_state_hash": START_STATE,
        "optimizer": parent["optimizer"], "loss": parent["loss"], "assets_sha256": pins,
        "identity_supervision": "same canonical HQ256 target embeddings in both arms; historical fixed affine",
        "factor": "reconstruction target bandwidth only; HQ256 vs HQ256->128->256 with PIL LANCZOS",
        "control_is_exact_legacy_replication": False,
        "selection": "unchanged_strict_source_profile_png_guard_plus_0.1dB",
        "historical_split_sha256": parent["historical_split_sha256"],
        "native_reserved_used": False, "production_promotion_permitted": False,
        "limitations": ["Development cohort; earlier exposure and identity overlap remain unestablished",
            "FFHQ-only training diagnostic, not Asian/Zamboanga CCTV generalization",
            "Asian validation targets remain low resolution; source-specific metrics required",
            "Embedding similarity is a frozen research proxy, not identity identification",
            "Metric eligibility still requires visual/native and independent final review"]}
    write(DEST / "targets_protocol_v6.json", protocol)
    (DEST / "targets_protocol_v6.sha256").write_text(sha(DEST / "targets_protocol_v6.json") + "\n", encoding="ascii")
    verify(DEST)
    with tarfile.open(ARCHIVE, "x:gz", compresslevel=5) as archive:
        for path in sorted(DEST.rglob("*")):
            if path.is_file():
                info = tarfile.TarInfo(PREFIX + "/" + path.relative_to(DEST).as_posix())
                info.size, info.mtime, info.mode = path.stat().st_size, 0, 0o644
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
    with Path(str(ARCHIVE) + ".sha256").open("x", encoding="ascii", newline="\n") as stream:
        stream.write(sha(ARCHIVE) + "  " + ARCHIVE.name + "\n")
    write(ROOT / "outputs/cctv_dgp_targets_v6_preparation.json", {"complete": True,
        "protocol_sha256": sha(DEST / "targets_protocol_v6.json"), "archive_sha256": sha(ARCHIVE),
        "archive_bytes": ARCHIVE.stat().st_size, "assets": len(pins),
        "training_references": 391, "validation_references": 104, "validation_cases": 520,
        "expected_optimizer_updates": 196, "seconds": time.monotonic() - start,
        "local_model_forwards": 0, "local_backward_calls": 0, "local_optimizer_updates": 0,
        "vm_training_pending": True, "native_reserved_used": False})
    print(read(ROOT / "outputs/cctv_dgp_targets_v6_preparation.json"))


if __name__ == "__main__":
    main()
