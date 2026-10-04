"""Verify tensor/data lineage and package a finite VM pilot; never construct models."""
import json
from pathlib import Path
import sys
import tarfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from varied_covering_experiment import ARMS, SOURCE_SHA, validate_source, fixed_schedules, require, preflight_case
from supported_real_data import SupportedMasks
from scripts.prepare_native_expert_vm import code_closure, tensor_parameter_groups
from scripts.train_native_expert_vm import sha, read_json, write_json, verify_fixture

DEST = ROOT / "outputs/varied_covering_protocol_v2"
ARCHIVE = ROOT / "outputs/varied-covering-vm-bundle-v2.tar.gz"
PREFIX = "varied_covering_vm_bundle"
REGISTRY = "dataset/detector_supported_review_v3/manifest.json"
PAIRS = "outputs/cofw_camera_pairs_v2/manifest.json"
SPLIT = "outputs/downloaded_phase4/outputs/phase4_with_progress/split.json"
ASSETS = {
    "model": ("outputs/downloaded_real_camera_v1/outputs/real_camera_vm/camera91/last.pth", "camera",
              "outputs/real_camera_vm/camera91/last.pth", SOURCE_SHA),
    "fixture_manifest": ("outputs/reflection_coverage_data_v1/manifest.json", "coverage",
                         "outputs/reflection_coverage_data_v1/manifest.json",
                         "6696cee3a3acf48049f2f121a2a6328625c4351f558deabb5475da7fe948baf9"),
    "fixture_pixels": ("outputs/reflection_coverage_data_v1/pixels.pth", "coverage",
                       "outputs/reflection_coverage_data_v1/pixels.pth",
                       "ab47a88d79fe8d300ca92d572a2c1a05b92fa273b9577cb209df99e692009cc1"),
}
PINS = {
    REGISTRY: "abab07e152941c4ae24d8aea3755fa7aaedb18965b25f6d0d1b6a7e00094aadd",
    PAIRS: "b6f4a800427e96a52468a0ec10eec478ec8f5875e1302b35ffd6c072dfec041c",
    SPLIT: "5c71bc358a351d50e3c0a7d76abe749cb4412aca80d2c7eb4de5fb33dbd29071",
    "dataset/detector_supported_review_v2/manifest.json": "3054c864bf7612fdcd5d0f55d47130b70f2895b6a0590f1341a60bbb011e2c7e",
    "outputs/cofw_covering_data_validation_v3/verification.json": "e16625fa2f66d51c33a8ee5481dec821648d531522a93d4f54236e187938faf1",
    "outputs/cofw_supported_dataset_validation_v1/verification.json": "a403f14457d7ba850ff2dbea4236b1db88276cd9e5d4f8192a65e0a0159261cc",
    "outputs/cofw_camera_pairs_v2/independent_verification.json": "c7fb19ff5d936c5e43f8a5c3a56b8ca59db40986d04f3be4c156660258652471",
    "outputs/cofw_camera_pairs_v2/visual_review.json": "a28c17f3bdf8bfe599bf7ae67a7b626231c228ca85543adb6f0e89b6831f4836",
    "outputs/cofw_covering_proposals_v3/manifest.json": "e180481c516a6025a162d5c3e135ff41212f9a120a628e7ccf8ec5b6029a8733",
    "outputs/cofw_source_acquisition_v3/current_record.json": "3c0c1b317b7b14272452a0161347cc3fd5c3950d9b1aed27a917fa0a293c256c",
    "outputs/real_camera_results_validation_v1/verification.json": "fbe47739fa7ec05f88b7ee22861d81b64b945c320109e4b7aee5a05c8af74245",
}


def main():
    require(not DEST.exists() and not ARCHIVE.exists() and not ARCHIVE.with_name(ARCHIVE.name + ".sha256").exists(),
            "Preserve prepared/partial package; create a new version for any change")
    torch.set_num_threads(4)
    for name, digest in PINS.items():
        require(sha(ROOT / name) == digest, "Audited input changed: " + name)
    for name, _, _, digest in ASSETS.values():
        require(sha(ROOT / name) == digest, "Read-only source asset changed: " + name)
    payload = torch.load(ROOT / ASSETS["model"][0], map_location="cpu", weights_only=True)
    validate_source(payload)
    groups = tensor_parameter_groups(payload)  # Tensor inspection only; no network/optimizer construction.
    require(sum(len(group["params"]) for group in groups) == 92, "Source parameter-state layout differs")
    real = SupportedMasks(ROOT / REGISTRY, split="train")
    parent = read_json(ROOT / "dataset/detector_supported_review_v2/manifest.json")
    require(real.metadata["supported_records"][:123] == parent["supported_records"], "Previous registry lineage changed")
    fixture_rows = read_json(ROOT / ASSETS["fixture_manifest"][0])["cases"]
    schedules = fixed_schedules(real.rows, fixture_rows)
    split = read_json(ROOT / SPLIT)
    training = {p.replace("\\", "/") for p in split["train"]}
    require(not training.intersection(p.replace("\\", "/") for p in split["validation"]), "Original split overlap")
    require(all(row["source"].replace("\\", "/") in training for row in real.rows[73:83] + fixture_rows),
            "Previous native/fixture source outside original training split")
    cache = torch.load(ROOT / ASSETS["fixture_pixels"][0], map_location="cpu", weights_only=True)
    require(cache["format"] == "dgp-reflection-coverage-pixels-v1" and set(cache["pixels"]) == set(range(280)), "Fixture cache differs")
    for i, row in enumerate(fixture_rows):
        verify_fixture(cache["pixels"][i], row)
    pairs = read_json(ROOT / PAIRS)
    require(len(pairs["cases"]) == 266 and pairs["registry_sha256"] == sha(ROOT / REGISTRY), "Paired registry differs")
    for case in pairs["cases"]:
        require(case["split"] == "train" and sha(ROOT / case["input"]) == case["input_sha256"], "Training pair changed")
        for field in case["target"].values():
            require(sha(ROOT / field["path"]) == field["sha256"], "Pair target/support changed")
    code = code_closure(["scripts/train_varied_covering_vm.py", "scripts/prepare_varied_covering_vm.py",
                         "varied_covering_experiment.py", "tests/test_varied_covering_experiment.py"])
    preview_specs = [("cofw_train_0935", "native"), ("cofw_train_0935", "degraded"),
                     ("cofw_train_0868", "native"), ("cofw_train_0868", "degraded"),
                     ("cofw_train_1057", "degraded"), ("cofw_train_1212", "degraded"),
                     ("cofw_train_1143", "degraded"), ("cofw_train_0930", "native"),
                     ("cofw_train_0246", "native"), ("cofw_train_0246", "degraded")]
    preview_ids = [next(c["case_id"] for c in pairs["cases"] if (c["source_id"], c["condition"]) == spec) for spec in preview_specs]
    require(len(set(preview_ids)) == 10, "Fixed preview differs")
    protocol = {"format": "dgp-varied-covering-pilot-v1", "date": "2026-10-03", "frozen_before_training": True,
                "stage": "Finite training-cohort data/family-sampling comparison", "arms": list(ARMS),
                "epochs_per_arm": 2, "steps_per_epoch": 64, "updates_per_arm": 128, "total_update_budget": 256,
                "batch_size": 8, "real_batch_size": 4, "fixture_batch_size": 4,
                "real_covered_clear_per_batch": [2, 2], "fixture_covered_clear_per_batch": [2, 2], "seed": 42,
                "schedules": schedules, "registry_sha256": PINS[REGISTRY], "pairs_sha256": PINS[PAIRS],
                "phase4_split_sha256": PINS[SPLIT], "local_audited_inputs": PINS,
                "existing_assets": {role: {"root": root, "vm_path": vm, "sha256": digest}
                                    for role, (_, root, vm, digest) in ASSETS.items()},
                "source_counters": {"epoch": 44, "additional_epoch": 34, "model_updates": 994,
                                    "previous_moment_step_metadata": 784, "source_experiment_updates": 112},
                "optimizer_reset": True, "initial_optimizer_state_step": 0, "final_optimizer_state_step": 128,
                "final_cumulative_model_updates": 1122, "encoder_lr": 1e-5, "decoder_head_lr": 1e-4,
                "weight_decay": 1e-4, "clip": 1., "threshold": .5,
                "domain_weights": {"real": .5, "reflection": .5}, "background_weight": .25, "hard_fraction": .1,
                "frozen_reference_head_and_bn_stats": True,
                "setup_sha256": "6af41c0662a0a91d55871c4faadb6d0810d3e366a964a4da23c5b52183ef210a",
                "measurement_images_per_checkpoint": 546, "measurement_checkpoints": 3,
                "actual_training_forward_image_budget": 2048, "training_and_measurement_forward_image_budget": 3686,
                "one_batch_preflight_forward_images_separate": 1, "wall_budget_seconds_after_loading": 1800,
                "preflight_case_id": preflight_case(pairs["cases"])["case_id"],
                "preview_case_ids": preview_ids, "preview_specs": preview_specs,
                "held_out_forward_images": 0, "original_425_case_gates_evaluated": False,
                "historical_gate_failure_unchanged": True, "promoted": False,
                "ready_for_vm_preflight": True, "actual_cuda_execution_verified": False,
                "code_sha256": {name: sha(ROOT / name) for name in code},
                "fit_criteria": ["Every new family native/degraded IoU improves versus matched control",
                                 "Old native/degraded fit retains IoU, miss, visible FP and case errors versus control",
                                 "Reflection fit retains the same metrics versus control",
                                 "Every final clear control in both branches empty",
                                 "Every new covered native/degraded training prediction nonempty"],
                "limitations": ["Assistant approximate real-covering labels, not independent expert or hidden-face targets",
                                "Original author-training cohort and exact overlap checks do not prove identity/pretraining separation",
                                "Fresh optimizer and increased clear sampling are shared changes; this compares data/sampling within that new recipe",
                                "Fit-only diagnostic; original425 gates and fixed practical output review still required before promotion"]}
    DEST.mkdir()
    write_json(DEST / "protocol.json", protocol)
    members = {name: ROOT / name for name in code}
    for folder in (Path(REGISTRY).parent, Path(PAIRS).parent, Path("outputs/cofw_covering_proposals_v3")):
        for path in (ROOT / folder).rglob("*"):
            if path.is_file():
                members[path.relative_to(ROOT).as_posix()] = path
    raw_sources = {}
    for row in real.metadata["supported_records"]:
        name = row["source"].replace("\\", "/")
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and sha(path) == row["source_sha256"], "Raw-source provenance missing/changed")
        raw_sources[name] = row["source_sha256"]
        members[name] = path
    for name in PINS:
        if name != SPLIT:
            members[name] = ROOT / name
    for name in ("VARIED_COVERING_VM.md", "COFW_DATA_PREPARATION.md", "REAL_CAMERA_RESULTS.md",
                 "outputs/cofw_covering_proposals_v2/refinement.json",
                 "outputs/cofw_covering_data_validation_v1/verification.json"):
        members[name] = ROOT / name
    members["inputs/varied_covering_protocol.json"] = DEST / "protocol.json"
    members["inputs/phase4_split.json"] = ROOT / SPLIT
    members["inputs/parent_supported_manifest.json"] = ROOT / "dataset/detector_supported_review_v2/manifest.json"
    require(all(path.resolve().is_relative_to(ROOT) for path in members.values()), "Archive member escapes workspace")
    inventory = {name: sha(path) for name, path in sorted(members.items())}
    write_json(DEST / "inventory.json", inventory)
    with tarfile.open(ARCHIVE, "x:gz") as tar:
        for name, path in sorted(members.items()):
            tar.add(path, arcname=PREFIX + "/" + name, recursive=False)
        tar.add(DEST / "inventory.json", arcname=PREFIX + "/inventory.json", recursive=False)
    with ARCHIVE.with_name(ARCHIVE.name + ".sha256").open("x", encoding="ascii", newline="\n") as stream:
        stream.write(sha(ARCHIVE) + "  " + ARCHIVE.name + "\n")
    report = {"complete": True, "archive_sha256": sha(ARCHIVE), "archive_bytes": ARCHIVE.stat().st_size,
              "protocol_sha256": sha(DEST / "protocol.json"), "inventory_sha256": sha(DEST / "inventory.json"),
              "files_in_archive": len(members) + 1, "code_files": len(code), "registry_records": 165,
              "training_sources": 133, "paired_inputs": 266, "fixture_cases_verified": 280,
              "raw_source_dependencies_verified": len(raw_sources), "source_parameter_tensors_verified": 92,
              "local_model_constructed": False, "local_optimizer_constructed": False,
              "local_model_forward_images": 0, "local_optimizer_updates": 0,
              "actual_cuda_execution_verified": False, "promoted": False,
              "vm_reused_bytes_not_uploaded": sum((ROOT / ASSETS[role][0]).stat().st_size for role in ("model", "fixture_pixels"))}
    write_json(DEST / "build.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
