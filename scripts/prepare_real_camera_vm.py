"""Package verified real input pairs and a VM-only diagnostic; no model construction."""
import json
from pathlib import Path
import tarfile

import torch

from native_expert import MODEL_SHA, validate_source, validate_source_optimizer
from real_camera_experiment import fixed_schedules
from supported_real_data import SupportedMasks
from scripts.prepare_native_expert_vm import (ASSETS, BOUND, SPLIT, code_closure, tensor_parameter_groups)
from scripts.train_native_expert_vm import sha, read_json, verify_fixture

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "outputs/real_camera_protocol_v1"
ARCHIVE = ROOT / "outputs/real-camera-vm-bundle.tar.gz"
PREFIX = "real_camera_vm_bundle"
REGISTRY = "dataset/detector_supported_review_v2/manifest.json"
PAIRS = "outputs/real_camera_pairs_v1/manifest.json"
PINS = {
    REGISTRY: "3054c864bf7612fdcd5d0f55d47130b70f2895b6a0590f1341a60bbb011e2c7e",
    PAIRS: "872b1e0d4cb75835892d4df2b6a17fd7dafde954a076724e10426ea0e71a047c",
    "outputs/mendeley_supported_dataset_validation_v1/verification.json": "67adbbcafc34002345ea1fc2995a5291c83b797cd3123873dca1c6eb12461c41",
    "outputs/real_camera_pairs_v1/independent_verification.json": "7fd28b5c9d91322ac762f659046a56b8a44b52cd34fcf3f138a353333ac3a408",
    "dataset/detector_supported_review_v1/manifest.json": "860ea1dfba8fa442cab19bf3ab41f6a678109dcdb067c38ec0bd79ce43b51ace",
}


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def main():
    if DEST.exists() or ARCHIVE.exists() or ARCHIVE.with_name(ARCHIVE.name + ".sha256").exists():
        raise ValueError("Preserve prepared/partial package; use a new version for changes")
    torch.set_num_threads(4)
    for name, expected in PINS.items():
        assert sha(ROOT / name) == expected
    assert sha(ROOT / SPLIT) == BOUND[SPLIT]
    for role, (local, vm, expected) in ASSETS.items():
        assert sha(ROOT / local) == expected
    real = SupportedMasks(ROOT / REGISTRY, split="train")
    assert len(real) == 91
    previous = read_json(ROOT / "dataset/detector_supported_review_v1/manifest.json")
    assert real.metadata["supported_records"][:115] == previous["supported_records"]
    fixtures = read_json(ROOT / ASSETS["fixture_manifest"][0])["cases"]
    schedules = fixed_schedules(real.rows, fixtures)
    split = read_json(ROOT / SPLIT)
    training = {path.replace("\\", "/") for path in split["train"]}
    assert not training.intersection(path.replace("\\", "/") for path in split["validation"])
    assert all(row["source"].replace("\\", "/") in training for row in real.rows[73:83] + fixtures)
    cache = torch.load(ROOT / ASSETS["fixture_pixels"][0], map_location="cpu", weights_only=True)
    assert cache["format"] == "dgp-reflection-coverage-pixels-v1" and set(cache["pixels"]) == set(range(280))
    for i, row in enumerate(fixtures):
        verify_fixture(cache["pixels"][i], row)
    payload = torch.load(ROOT / ASSETS["model"][0], map_location="cpu", weights_only=True)
    validate_source(payload)
    moments = torch.load(ROOT / ASSETS["optimizer"][0], map_location="cpu", weights_only=True)
    moment_check = validate_source_optimizer(moments, tensor_parameter_groups(payload), MODEL_SHA)
    pairs = read_json(ROOT / PAIRS)
    assert len(pairs["cases"]) == 182 and pairs["held_out_inputs_in_cache"] == 0
    for item in pairs["cases"]:
        assert item["split"] == "train" and sha(ROOT / item["input"]) == item["input_sha256"]
        for field in item["target"].values():
            assert sha(ROOT / field["path"]) == field["sha256"]
    code = code_closure(["scripts/train_real_camera_vm.py", "scripts/prepare_real_camera_vm.py",
                         "real_camera_experiment.py", "real_camera_pairs.py",
                         "tests/test_real_camera_experiment.py", "tests/test_real_camera_pairs.py"])
    protocol = {
        "format": "dgp-real-camera-pilot-v1", "date": "2026-10-02", "frozen_before_training": True,
        "stage": "Training-cohort diagnosis only", "arms": ["native83", "camera83", "camera91"],
        "epochs_per_arm": 2, "steps_per_epoch": 56, "updates_per_arm": 112, "total_update_budget": 336,
        "batch_size": 8, "real_batch_size": 3, "fixture_batch_size": 5, "seed": 42,
        "schedules": schedules, "registry_sha256": PINS[REGISTRY], "pairs_sha256": PINS[PAIRS],
        "phase4_split_sha256": BOUND[SPLIT], "local_audited_inputs": PINS,
        "existing_assets": {role: {"vm_path": vm, "sha256": digest} for role, (_, vm, digest) in ASSETS.items()},
        "source_counters": {"epoch": 42, "model_updates": 882, "moment_step": 672}, "moment_check": moment_check,
        "setup_sha256": "6af41c0662a0a91d55871c4faadb6d0810d3e366a964a4da23c5b52183ef210a",
        "measurement_images_per_checkpoint": 462, "measurement_checkpoints": 4,
        "actual_training_forward_image_budget": 2688, "training_and_measurement_forward_image_budget": 4536,
        "one_batch_preflight_forward_images_separate": 1, "wall_budget_seconds_after_loading": 1800,
        "encoder_lr": 1e-5, "decoder_head_lr": 1e-4, "weight_decay": 1e-4, "clip": 1.,
        "domain_weights": {"real": .5, "reflection": .5}, "background_weight": .25, "hard_fraction": .1,
        "threshold": .5, "optimizer_reset": False, "frozen_reference_head_and_bn_stats": True,
        "preview_case_ids": [168, 169, 170, 171, 174, 175, 166, 167, 149, 146],
        "held_out_forward_images": 0, "original_425_case_gates_evaluated": False,
        "historical_synthetic_failure_unchanged": True, "promoted": False,
        "pilot_ready_for_vm_preflight": True, "actual_cuda_execution_verified": False,
        "code_sha256": {name: sha(ROOT / name) for name in code},
        "fit_criteria": ["camera83 old degraded IoU > native83", "camera83 old native/replay fit retained vs native83",
                         "camera91 new native/degraded IoU > camera83", "camera91 old native/degraded/replay fit retained vs camera83",
                         "Every final clear control empty"],
        "limitations": ["Only eight related-capture additions, not subject-diverse broad-family training",
                        "Approximate assistant labels with explicit uncertainty; no expert or hidden-face target",
                        "No actual local model/optimizer construction or training; CUDA is still pending",
                        "Training fit does not select best.pth, qualify original gates or prove generated output improvement"],
    }
    DEST.mkdir()
    write(DEST / "protocol.json", protocol)
    members = {name: ROOT / name for name in code}
    for path in (ROOT / Path(REGISTRY).parent).rglob("*"):
        if path.is_file():
            members[path.relative_to(ROOT).as_posix()] = path
    for path in (ROOT / Path(PAIRS).parent).rglob("*"):
        if path.is_file():
            members[path.relative_to(ROOT).as_posix()] = path
    members["inputs/real_camera_protocol.json"] = DEST / "protocol.json"
    members["inputs/phase4_split.json"] = ROOT / SPLIT
    members["inputs/parent_supported_manifest.json"] = ROOT / "dataset/detector_supported_review_v1/manifest.json"
    for name in ("REAL_CAMERA_VM.md", "MENDELEY_OCCLUSION_DATA.md", "PRACTICAL_DIRECT_DETECTOR_RESULTS.md"):
        members[name] = ROOT / name
    for name in ("outputs/mendeley_supported_dataset_validation_v1/verification.json",
                 "outputs/mendeley_mask_proposals_v3/annotation_decisions.json",
                 "outputs/mendeley_mask_proposals_v3/independent_verification.json",
                 "outputs/mendeley_occlusion_candidate_v1/uploaded_source_review/independent_verification.json"):
        members[name] = ROOT / name
    assert all(path.resolve().is_relative_to(ROOT.resolve()) for path in members.values())
    inventory = {name: sha(path) for name, path in sorted(members.items())}
    write(DEST / "inventory.json", inventory)
    with tarfile.open(ARCHIVE, "x:gz") as tar:
        for name, path in sorted(members.items()):
            tar.add(path, arcname=PREFIX + "/" + name, recursive=False)
        tar.add(DEST / "inventory.json", arcname=PREFIX + "/inventory.json", recursive=False)
    with ARCHIVE.with_name(ARCHIVE.name + ".sha256").open("x", encoding="ascii", newline="\n") as stream:
        stream.write(sha(ARCHIVE) + "  " + ARCHIVE.name + "\n")
    report = {"complete": True, "archive_sha256": sha(ARCHIVE), "archive_bytes": ARCHIVE.stat().st_size,
              "protocol_sha256": sha(DEST / "protocol.json"), "inventory_sha256": sha(DEST / "inventory.json"),
              "files_in_archive": len(members) + 1, "code_files": len(code), "dataset_records": 123,
              "real_training_sources": 91, "camera_inputs": 182, "fixture_cases_verified": 280,
              "source_moments_verified": moment_check, "local_model_constructed": False,
              "local_optimizer_constructed": False, "local_model_forward_images": 0, "local_optimizer_updates": 0,
              "actual_cuda_execution_verified": False, "promoted": False,
              "vm_reuse_bytes_not_reuploaded": sum((ROOT / ASSETS[role][0]).stat().st_size for role in ("model", "optimizer", "fixture_pixels"))}
    write(DEST / "build.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
