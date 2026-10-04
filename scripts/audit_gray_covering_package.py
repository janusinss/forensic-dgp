"""Independent archive, schedule, source/label and budget checks; no model imports."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import tarfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "outputs/gray-covering-vm-bundle.tar.gz"
ARCHIVE_SHA = "212b91935e548ad1423f92f71014681be25abd052bad1d7bf2785e8b2b702d18"
BASE = ROOT / "outputs/varied-covering-vm-bundle-v2.tar.gz"
BASE_SHA = "780c44f2e3f724ff600eb869e99422b1bf81b18678353a59ac68afb137e32d0b"
OUT = ROOT / "outputs/gray_covering_package_validation_v1"


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def members(path, prefix):
    blobs, folded = {}, set()
    with tarfile.open(path, "r:gz") as tar:
        for entry in tar.getmembers():
            require(entry.isfile() and entry.name.startswith(prefix) and entry.size <= 32 * 1024**2,
                    "Unsafe/oversize member")
            name = entry.name[len(prefix):]
            require(name and "\\" not in name and ":" not in name and not name.startswith("/")
                    and PurePosixPath(name).as_posix() == name and ".." not in name.split("/")
                    and name.casefold() not in folded, "Unsafe/duplicate path")
            folded.add(name.casefold())
            blobs[name] = tar.extractfile(entry).read()
    require(sum(map(len, blobs.values())) <= 256 * 1024**2, "Total archive budget exceeded")
    return blobs


def main():
    require(not OUT.exists(), "Preserve existing package audit")
    require(sha(ARCHIVE.read_bytes()) == ARCHIVE_SHA and sha(BASE.read_bytes()) == BASE_SHA, "Archive hash differs")
    require(ARCHIVE.with_name(ARCHIVE.name + ".sha256").read_bytes() ==
            (ARCHIVE_SHA + "  " + ARCHIVE.name + "\n").encode("ascii"), "Require exact LF checksum")
    blobs = members(ARCHIVE, "gray_covering_vm_bundle/")
    base = members(BASE, "varied_covering_vm_bundle/")
    original = json.loads(base.pop("inventory.json"))
    inventory = json.loads(blobs["inventory.json"])
    require(set(inventory) == set(blobs) - {"inventory.json"}, "Inventory membership differs")
    require(all(sha(blobs[n]) == pin for n, pin in inventory.items()), "Inventory digest differs")
    require(set(original) == set(base) and all(sha(base[n]) == pin for n, pin in original.items()), "V2 inventory differs")
    require(all(blobs[n] == data for n, data in base.items()), "Any original V2 code/data/doc changed in new bundle")
    p = json.loads(blobs["inputs/gray_covering_protocol.json"])
    old = json.loads(base["inputs/varied_covering_protocol.json"])
    require(p["format"] == "dgp-gray-covering-pilot-v1" and p["arms"] == ["rgb133", "gray133"], "Experiment differs")
    expected = {"epochs_per_arm": 8, "steps_per_epoch": 64, "updates_per_arm": 512, "total_update_budget": 1024,
                "measurement_images_per_checkpoint": 812, "measurement_checkpoints": 4,
                "actual_training_forward_image_budget": 8192, "training_and_measurement_forward_image_budget": 11440,
                "final_optimizer_state_step": 512, "final_cumulative_model_updates": 1506,
                "final_epoch": 52, "final_additional_epoch": 42, "wall_budget_seconds_after_loading": 1800,
                "held_out_forward_images": 0, "one_batch_preflight_forward_images_separate": 1}
    require(all(type(p[k]) is int and p[k] == value for k, value in expected.items()), "Finite budget/counters differ")
    require(p["optimizer_reset"] is True and p["promoted"] is False and p["original_425_case_gates_evaluated"] is False
            and p["actual_cuda_execution_verified"] is False and p["rgb128_exact_model_reproduction_required"] is True,
            "Preparation cannot claim execution or promotion")
    registry = json.loads(blobs["dataset/detector_supported_review_v3/manifest.json"])
    require(sha(blobs["dataset/detector_supported_review_v3/manifest.json"]) ==
            "abab07e152941c4ae24d8aea3755fa7aaedb18965b25f6d0d1b6a7e00094aadd", "Reviewed registry changed")
    rows = [r for r in registry["supported_records"] if r["split"] == "train"]
    require(len(rows) == 133 and len(registry["supported_records"]) == 165, "Split scope changed")
    old_pos = [i for i, r in enumerate(rows[:91]) if r["kind"] == "covered"]
    old_clear = [i for i, r in enumerate(rows[:91]) if r["kind"] == "uncovered"]
    new_clear = [i for i in range(91, 133) if rows[i]["kind"] == "uncovered"]
    names = ("hand", "hair", "cloth", "object", "eyewear_mask")
    mapping = {"hand": "hand", "obstructing_hair": "hair", "cloth_or_scarf": "cloth",
               "other_cloth_or_object": "cloth", "object": "object", "sunglasses": "eyewear_mask",
               "costume_mask": "eyewear_mask", "face_mask_preserve_clear_goggles": "eyewear_mask"}
    families = {n: [i for i in range(91, 133) if rows[i]["kind"] == "covered"
                    and mapping[rows[i]["occlusion_stratum"]] == n] for n in names}
    fixture_path = ROOT / "outputs/reflection_coverage_data_v1/manifest.json"
    fixture_blob = fixture_path.read_bytes()
    require(sha(fixture_blob) == p["existing_assets"]["fixture_manifest"]["sha256"], "External fixture definition differs")
    fixture = json.loads(fixture_blob)["cases"]
    fp = [r["case_id"] for r in fixture if r["style"] != "clear"]
    fc = [r["case_id"] for r in fixture if r["style"] == "clear"]
    rng = random.Random(42)
    for pool in (old_pos, old_clear, new_clear, fp, fc, *(families[n] for n in names)): rng.shuffle(pool)
    visits = [Counter() for _ in range(4)]
    exposure = {arm: set() for arm in p["arms"]}
    grayscale_occurrences = 0
    for epoch in range(8):
        a, b = p["schedules"]["rgb133"][epoch], p["schedules"]["gray133"][epoch]
        require(len(a) == len(b) == 64, "Epoch step count differs")
        for step in range(64):
            i = epoch * 64 + step
            family = names[i % 5]
            selected = [old_pos[i % 58], families[family][(i // 5) % len(families[family])],
                        old_clear[i % 33], new_clear[i % 11]]
            fixtures = [fp[(2 * i) % 224], fp[(2 * i + 1) % 224], fc[(2 * i) % 56], fc[(2 * i + 1) % 56]]
            require(a[step]["fixture"] == b[step]["fixture"] == fixtures, "Replay slots differ")
            require([rows[j]["kind"] for j in selected] == ["covered", "covered", "uncovered", "uncovered"], "Real strata differ")
            for slot, source in enumerate(selected):
                visit = visits[slot][source]
                visits[slot][source] += 1
                for arm, batch in (("rgb133", a[step]), ("gray133", b[step])):
                    row = batch["real"][slot]
                    wanted = {"index": source, "degraded": bool(visit % 2),
                              "grayscale": bool((visit // 2) % 2) if arm == "gray133" else False}
                    require(row == wanted and type(row["index"]) is int
                            and type(row["degraded"]) is type(row["grayscale"]) is bool, "Matched condition cadence differs")
                    exposure[arm].add((source, row["degraded"], row["grayscale"]))
                    grayscale_occurrences += int(arm == "gray133" and row["grayscale"])
            if i < 128:
                stripped = {"real": [{k: r[k] for k in ("index", "degraded")} for r in a[step]["real"]], "fixture": fixtures}
                require(stripped == old["schedules"]["varied133"][epoch][step], "Original executed128-step trace differs")
    require(exposure["rgb133"] == {(i, d, False) for i in range(133) for d in (False, True)}
            and exposure["gray133"] == {(i, d, g) for i in range(133) for d in (False, True) for g in (False, True)},
            "Missing source/camera/color condition")
    for name, pin in p["code_sha256"].items():
        require(sha(blobs[name]) == pin, "Consumed code changed")
        compile(blobs[name].decode("utf-8-sig"), name, "exec")
    trainer = blobs["scripts/train_gray_covering_vm.py"].decode("utf-8")
    tree = ast.parse(trainer)
    main_body = next(n.body for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    require(main_body[0].value.func.id == "require_vm_gpu", "VM/CUDA guard must precede all main operations")
    require("optimizer.load_state_dict" not in trainer and "RGB128 failed exact previous-run reproduction" in trainer
            and 'model.detect(torch.cat((r[0], f[0])))' in trainer, "Fresh optimizer/reproduction/real-forward contract differs")
    for name in ("scripts/setup_gray_covering_vm.sh", "scripts/run_gray_covering_vm.sh"):
        require(b"\r" not in blobs[name] and blobs[name].startswith(b"#!/usr/bin/env bash\nset -euo pipefail\n"), "Shell file must have LF/strict mode")
    setup = blobs["scripts/setup_gray_covering_vm.sh"].decode()
    compile(setup.split("python - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0], "setup-inline", "exec")
    require(b'--dry_run --batch_size 1' in blobs["scripts/run_gray_covering_vm.sh"], "One-batch preflight absent")
    split = json.loads(blobs["inputs/phase4_split.json"])
    require(not {n.replace("\\", "/") for n in split["train"]}.intersection(n.replace("\\", "/") for n in split["validation"]),
            "Original Phase4 split overlap")
    require(p["existing_assets"]["rgb128_reference"]["sha256"] ==
            "e3a087f55e248e64f93c96ad50aa3cb73bd08d0ea923583ac11f02aa85179b29", "RGB128 external model differs")
    OUT.mkdir()
    report = {"format": "dgp-gray-covering-package-independent-verification-v1", "date": "2026-10-03", "complete": True,
        "archive_sha256": ARCHIVE_SHA, "archive_bytes": ARCHIVE.stat().st_size, "protocol_sha256": sha(blobs["inputs/gray_covering_protocol.json"]),
        "inventory_sha256": sha(blobs["inventory.json"]), "auditor_sha256": sha(Path(__file__).read_bytes()),
        "regular_members_verified": len(blobs), "original_v2_members_byte_preserved": len(base),
        "code_files_compiled": len(p["code_sha256"]), "schedule_independently_reconstructed": True,
        "first128_rgb_schedule_matches_executed_v2": True, "all133_sources_have_all_declared_conditions": True,
        "gray_training_image_occurrences": grayscale_occurrences, "gray_branch_real_training_occurrences": 2048,
        "updates_per_arm": 512, "total_update_budget": 1024, "actual_image_forward_budget": 11440,
        "saved_measurement_masks_expected": 3248, "vm_guard_first": True, "old_optimizer_moment_restore_absent": True,
        "local_model_forwards": 0, "local_optimizer_updates": 0, "actual_cuda_execution_verified": False,
        "promoted": False, "limitation": "Archive preparation verification, not an executed CUDA preflight or training/quality result"}
    with (OUT / "verification.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("complete", "archive_bytes", "regular_members_verified", "updates_per_arm", "local_optimizer_updates")}))


if __name__ == "__main__":
    main()
