"""Package matched grayscale exposure using verified V2 data; no model construction."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gray_covering_experiment import ARMS, EPOCHS, STEPS, UPDATES, FORWARD_BUDGET, fixed_schedules, validate_source, require
from supported_real_data import SupportedMasks
from scripts.prepare_native_expert_vm import code_closure, tensor_parameter_groups
from scripts.prepare_varied_covering_vm import ASSETS, PINS as PREVIOUS_PINS, REGISTRY, PAIRS
from scripts.train_native_expert_vm import sha, read_json, write_json

DEST = ROOT / "outputs/gray_covering_protocol_v1"
ARCHIVE = ROOT / "outputs/gray-covering-vm-bundle.tar.gz"
PREFIX = "gray_covering_vm_bundle"
BASE_ARCHIVE = ROOT / "outputs/varied-covering-vm-bundle-v2.tar.gz"
BASE_SHA = "780c44f2e3f724ff600eb869e99422b1bf81b18678353a59ac68afb137e32d0b"
REFERENCE = "outputs/downloaded_varied_covering_v1/outputs/varied_covering_vm/varied133/last.pth"
REFERENCE_SHA = "e3a087f55e248e64f93c96ad50aa3cb73bd08d0ea923583ac11f02aa85179b29"
PINS = {**PREVIOUS_PINS,
    "outputs/varied_covering_results_validation_v1/verification.json": "ebae5bd787f0fe0f19fa57d483c3ea29766c1ef73825e76560cd0f7f3654c95d",
    "outputs/varied_covering_footprint_diagnostic_v1/frozen_protocol.json": "1c65b7b43bca386289898f4100eba538ed7983104f9ccb15f24ded7a0fe78a2e",
    "outputs/varied_covering_footprint_diagnostic_v1/results.json": "b88e696f4cc5183c4e5b44924db9279353dd6126c6dad53fed79cefe61f93469",
    "outputs/varied_covering_footprint_diagnostic_v1/visual_review.json": "f87d360ced332819f9039ae4f0a2a0634d86d99047d96e3dec740a6279b677bb",
}

SETUP = '''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f ../feature_vm_bundle/.venv/bin/activate ]]; then
  source ../feature_vm_bundle/.venv/bin/activate
elif [[ -f ../venv/bin/activate ]]; then
  source ../venv/bin/activate
else
  echo "Missing existing CUDA venv: ../feature_vm_bundle/.venv or ../venv" >&2
  exit 1
fi
export PYTHONNOUSERSITE=1
nvidia-smi
python - <<'PY'
import hashlib, json
from pathlib import Path
import torch
from native_expert import require_vm_gpu
require_vm_gpu()
assert str(torch.__version__) == "2.9.1+cu129", "Require the existing pinned CUDA runtime"
assert torch.cuda.get_device_properties(0).total_memory >= 4 * 1024**3
assert torch.cuda.mem_get_info()[0] >= 2 * 1024**3, "Require at least2GiB free VRAM"
root = Path.cwd()
def sha(path):
    with path.open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()
inventory = json.loads((root / "inventory.json").read_text())
assert all(sha(root / n) == pin for n, pin in inventory.items()), "Bundle changed"
protocol = json.loads((root / "inputs/gray_covering_protocol.json").read_text())
roots = {"camera": root.parent / "real_camera_vm_bundle", "coverage": root.parent / "coverage_vm_bundle",
         "varied": root.parent / "varied_covering_vm_bundle"}
for role, item in protocol["existing_assets"].items():
    assert sha(roots[item["root"]] / item["vm_path"]) == item["sha256"], "Missing/changed source asset: " + role
assert all((root.parent / "dataset" / n).is_dir() for n in ("asian_faces", "thumbnails128x128"))
assert sha(root.parent / "outputs/phase4_with_progress/split.json") == protocol["phase4_split_sha256"]
split = json.loads((root / "inputs/phase4_split.json").read_text())
assert not {p.replace("\\\\", "/") for p in split["train"]}.intersection(p.replace("\\\\", "/") for p in split["validation"])
print("Setup verified: CUDA, VRAM, source models, cached data and split; zero model forwards/updates.")
PY
'''

RUN = '''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/setup_gray_covering_vm.sh
if [[ -f ../feature_vm_bundle/.venv/bin/activate ]]; then
  source ../feature_vm_bundle/.venv/bin/activate
else
  source ../venv/bin/activate
fi
export PYTHONNOUSERSITE=1
python -u scripts/train_gray_covering_vm.py --dry_run --batch_size 1
python -u scripts/train_gray_covering_vm.py
'''


def original_members():
    require(sha(BASE_ARCHIVE) == BASE_SHA, "Frozen V2 archive changed")
    require(BASE_ARCHIVE.with_name(BASE_ARCHIVE.name + ".sha256").read_bytes() ==
            (BASE_SHA + "  " + BASE_ARCHIVE.name + "\n").encode("ascii"), "V2 LF checksum changed")
    blobs = {}
    with tarfile.open(BASE_ARCHIVE, "r:gz") as tar:
        for member in tar.getmembers():
            prefix = "varied_covering_vm_bundle/"
            require(member.isfile() and member.name.startswith(prefix) and member.size <= 32 * 1024**2,
                    "Unsafe frozen V2 member")
            name = member.name[len(prefix):]
            require(name and "\\" not in name and ":" not in name and not name.startswith("/")
                    and PurePosixPath(name).as_posix() == name and ".." not in name.split("/")
                    and name.casefold() not in {n.casefold() for n in blobs}, "Unsafe duplicate V2 member")
            blobs[name] = tar.extractfile(member).read()
    require(len(blobs) == 1010, "V2 member count changed")
    inventory = json.loads(blobs.pop("inventory.json"))
    require(set(inventory) == set(blobs) and all(hashlib.sha256(blobs[n]).hexdigest() == pin
            for n, pin in inventory.items()), "V2 inventory differs")
    return blobs


def main():
    require(not DEST.exists() and not ARCHIVE.exists() and not ARCHIVE.with_name(ARCHIVE.name + ".sha256").exists(),
            "Preserve prepared/partial pilot; create a new version for changes")
    torch.set_num_threads(4)
    for name, pin in PINS.items():
        require(sha(ROOT / name) == pin, "Audited input changed: " + name)
    for name, _, _, pin in ASSETS.values():
        require(sha(ROOT / name) == pin, "Read-only source asset changed")
    require(sha(ROOT / REFERENCE) == REFERENCE_SHA, "Previous RGB128 checkpoint changed")
    payload = torch.load(ROOT / ASSETS["model"][0], map_location="cpu", weights_only=True)
    validate_source(payload)
    require(sum(len(group["params"]) for group in tensor_parameter_groups(payload)) == 92, "Source tensors differ")
    reference = torch.load(ROOT / REFERENCE, map_location="cpu", weights_only=True)
    require(reference["varied_covering_arm"] == "varied133" and reference["optimizer_updates"] == 1122
            and reference["fresh_optimizer_updates"] == 128, "RGB128 checkpoint lineage differs")
    tensor_parameter_groups(reference)
    real = SupportedMasks(ROOT / REGISTRY, split="train")
    fixtures = read_json(ROOT / ASSETS["fixture_manifest"][0])["cases"]
    schedules = fixed_schedules(real.rows, fixtures)
    previous = read_json(ROOT / "outputs/varied_covering_protocol_v2/protocol.json")
    first128 = [{"real": [{k: r[k] for k in ("index", "degraded")} for r in b["real"]], "fixture": b["fixture"]}
                for epoch in schedules["rgb133"][:2] for b in epoch]
    require(first128 == [b for epoch in previous["schedules"]["varied133"] for b in epoch], "Previous128-step trace differs")
    code = code_closure(["scripts/train_gray_covering_vm.py", "scripts/prepare_gray_covering_vm.py",
                         "gray_covering_experiment.py", "tests/test_gray_covering_experiment.py"])
    protocol = {**previous, "format": "dgp-gray-covering-pilot-v1", "date": "2026-10-03",
        "stage": "Matched RGB/grayscale exposure; previous128-step reproduction and longer RGB control",
        "arms": list(ARMS), "epochs_per_arm": EPOCHS, "steps_per_epoch": STEPS,
        "updates_per_arm": UPDATES, "total_update_budget": 2 * UPDATES, "schedules": schedules,
        "local_audited_inputs": PINS, "initial_optimizer_state_step": 0, "final_optimizer_state_step": UPDATES,
        "final_cumulative_model_updates": 994 + UPDATES, "final_epoch": 44 + EPOCHS, "final_additional_epoch": 34 + EPOCHS,
        "measurement_images_per_checkpoint": 812, "measurement_checkpoints": 4,
        "actual_training_forward_image_budget": 2 * UPDATES * 8,
        "training_and_measurement_forward_image_budget": FORWARD_BUDGET,
        "rgb128_exact_model_reproduction_required": True, "rgb128_checkpoint_fresh_updates": 128,
        "grayscale": {"coefficients": [.299, .587, .114], "changes": "Input RGB channels only",
                      "visit_pattern": [False, False, True, True], "condition_pattern": [False, True, False, True],
                      "roles": "All four real training slots; cached reflection fixtures stay unchanged",
                      "targets_support_padding_geometry_changed": False,
                      "all133_sources_seen_native_degraded_and_rgb_gray": True,
                      "cause_claim": "A grayscale transfer hypothesis, not an established cause of the hair failure"},
        "existing_assets": {**previous["existing_assets"], "rgb128_reference": {"root": "varied",
            "vm_path": "outputs/varied_covering_vm/varied133/last.pth", "sha256": REFERENCE_SHA}},
        "code_sha256": {name: sha(ROOT / name) for name in code},
        "fit_criteria": ["Grayscale native/degraded hair IoU improves over matched RGB512 control",
                         "Other new grayscale family IoUs do not decline versus control",
                         "RGB old/new cohorts and reflection retain IoU/misses/FP/case-error measures",
                         "Final supervised clear controls are empty in both branches",
                         "New covered predictions are nonempty across RGB/grayscale and native/degraded conditions"],
        "limitations": ["Existing approximate author-train detector labels; no new hidden-face supervision",
                        "Both branches extend the prior family-cycling schedule; only grayscale input exposure differs between them",
                        "Previously inspected hair/near-hidden sources remain evaluation-only; they are not added to training",
                        "No generator/restoration/identity training or app selection; original425 gates and practical output review remain required",
                        "Exact RGB128 model reproduction is a stopping requirement; preserve any partial run instead of retrying in place"]}
    members = original_members()
    for name in code:
        members[name] = (ROOT / name).read_bytes()
    for name in PINS:
        if name != "outputs/downloaded_phase4/outputs/phase4_with_progress/split.json":
            members[name] = (ROOT / name).read_bytes()
    members["GRAY_COVERING_VM.md"] = (ROOT / "GRAY_COVERING_VM.md").read_bytes()
    members["scripts/setup_gray_covering_vm.sh"] = SETUP.encode("utf-8")
    members["scripts/run_gray_covering_vm.sh"] = RUN.encode("utf-8")
    DEST.mkdir()
    write_json(DEST / "protocol.json", protocol)
    members["inputs/gray_covering_protocol.json"] = (DEST / "protocol.json").read_bytes()
    inventory = {name: hashlib.sha256(blob).hexdigest() for name, blob in sorted(members.items())}
    write_json(DEST / "inventory.json", inventory)
    members["inventory.json"] = (DEST / "inventory.json").read_bytes()
    with tarfile.open(ARCHIVE, "x:gz") as tar:
        for name, blob in sorted(members.items()):
            info = tarfile.TarInfo(PREFIX + "/" + name)
            info.size, info.mode, info.mtime = len(blob), 0o644, 0
            tar.addfile(info, io.BytesIO(blob))
    with ARCHIVE.with_name(ARCHIVE.name + ".sha256").open("x", encoding="ascii", newline="\n") as stream:
        stream.write(sha(ARCHIVE) + "  " + ARCHIVE.name + "\n")
    write_json(DEST / "build.json", {"complete": True, "archive_sha256": sha(ARCHIVE), "archive_bytes": ARCHIVE.stat().st_size,
        "protocol_sha256": sha(DEST / "protocol.json"), "inventory_sha256": sha(DEST / "inventory.json"),
        "files_in_archive": len(members), "code_files": len(code), "first128_rgb_schedule_matches_executed_v2": True,
        "registry_records": 165, "training_sources": 133, "paired_inputs": 266, "fixture_cases": 280,
        "reused_data_archive_sha256": BASE_SHA, "updates_per_arm": UPDATES, "total_update_budget": 2 * UPDATES,
        "local_models_constructed": 0, "local_optimizers_constructed": 0, "local_model_forwards": 0,
        "local_optimizer_updates": 0, "actual_cuda_execution_verified": False, "promoted": False})
    print(json.dumps({"prepared": True, "archive_bytes": ARCHIVE.stat().st_size, "archive_sha256": sha(ARCHIVE),
                      "updates_per_arm": UPDATES, "local_training_updates": 0}))


if __name__ == "__main__":
    main()
