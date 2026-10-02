"""Independent transfer/inventory/sampling audit; never extracts or runs models."""
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
import tarfile
import ast

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "outputs/real-camera-vm-bundle.tar.gz"
OUT = ROOT / "outputs/real_camera_package_validation_v1"
ARCHIVE_SHA = "2b237b2e121951203bde131be5d7c649910392df47bff1bbc953f81a8c8c0e2b"
PROTOCOL_SHA = "a8adb21910c5aed249f5f2a75e17eef92db35717bd428326c360a1aecf3b99e1"
PREFIX = "real_camera_vm_bundle/"


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def main():
    if OUT.exists():
        raise ValueError("Preserve previous independent package audit")
    assert digest(ARCHIVE.read_bytes()) == ARCHIVE_SHA
    assert ARCHIVE.with_name(ARCHIVE.name + ".sha256").read_bytes() == (ARCHIVE_SHA + "  " + ARCHIVE.name + "\n").encode("ascii")
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        members = {}
        folded = set()
        for member in tar.getmembers():
            assert member.isfile() and member.name.startswith(PREFIX) and member.size <= 32 * 1024**2
            relative = member.name[len(PREFIX):]
            assert relative and "\\" not in relative and ":" not in relative and not relative.startswith("/")
            assert PurePosixPath(relative).as_posix() == relative and ".." not in relative.split("/")
            assert relative.casefold() not in folded
            folded.add(relative.casefold())
            members[relative] = member
        assert len(members) == 419
        def raw(name):
            assert name in members
            blob = tar.extractfile(members[name]).read()
            assert len(blob) == members[name].size
            return blob
        inventory = json.loads(raw("inventory.json"))
        assert set(members) == set(inventory) | {"inventory.json"}
        for name, expected in inventory.items():
            assert digest(raw(name)) == expected == digest((ROOT / name).read_bytes()) if not name.startswith("inputs/") else digest(raw(name)) == expected
        spec_bytes = raw("inputs/real_camera_protocol.json")
        assert digest(spec_bytes) == PROTOCOL_SHA
        p = json.loads(spec_bytes)
        assert p["format"] == "dgp-real-camera-pilot-v1" and p["source_counters"] == {"epoch": 42, "model_updates": 882, "moment_step": 672}
        assert p["epochs_per_arm"] == 2 and p["steps_per_epoch"] == 56 and p["updates_per_arm"] == 112 and p["total_update_budget"] == 336
        assert p["measurement_images_per_checkpoint"] == 462 and p["training_and_measurement_forward_image_budget"] == 4536
        assert p["held_out_forward_images"] == 0 and p["promoted"] is False and p["actual_cuda_execution_verified"] is False
        assert p["optimizer_reset"] is False and p["original_425_case_gates_evaluated"] is False
        for name, expected in p["code_sha256"].items():
            assert digest(raw(name)) == expected
            compile(raw(name), name, "exec")
        tree = ast.parse(raw("scripts/train_real_camera_vm.py"))
        main_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        assert isinstance(main_node.body[0], ast.Expr) and isinstance(main_node.body[0].value, ast.Call)
        assert main_node.body[0].value.func.id == "require_vm_gpu"
        droot = "dataset/detector_supported_review_v2/"
        registry = json.loads(raw(droot + "manifest.json"))
        assert digest(raw(droot + "manifest.json")) == p["registry_sha256"]
        parent = json.loads(raw("inputs/parent_supported_manifest.json"))
        assert registry["supported_records"][:115] == parent["supported_records"]
        real = [r for r in registry["supported_records"] if r["split"] == "train"]
        assert len(real) == 91 and len(registry["supported_records"]) == 123
        for split in ("validation", "test"):
            assert [r for r in registry["supported_records"] if r["split"] == split] == [r for r in parent["supported_records"] if r["split"] == split]
        def image(name):
            with Image.open(io.BytesIO(raw(name))) as source:
                return np.asarray(source).copy()
        for row in registry["supported_records"]:
            arrays = {}
            for key in ("image", "mask", "valid", "source_valid"):
                name = droot + row[key]
                assert digest(raw(name)) == row[key + "_sha256"]
                arrays[key] = image(name)
            rgb, mask, valid, source = (arrays[k] for k in ("image", "mask", "valid", "source_valid"))
            assert rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8
            assert all(a.shape == (256, 256) and np.isin(a, [0, 255]).all() for a in (mask, valid, source))
            assert not (mask.astype(bool) & ~valid.astype(bool)).any() and not (valid.astype(bool) & ~source.astype(bool)).any()
            assert np.all(rgb[source == 0] == 96) and bool(mask.any()) == (row["kind"] == "covered")
        pairs = json.loads(raw("outputs/real_camera_pairs_v1/manifest.json"))
        assert digest(raw("outputs/real_camera_pairs_v1/manifest.json")) == p["pairs_sha256"]
        assert len(pairs["cases"]) == 182 and pairs["held_out_inputs_in_cache"] == 0
        for i, row in enumerate(real):
            native, degraded = pairs["cases"][2 * i:2 * i + 2]
            assert native["real_train_index"] == degraded["real_train_index"] == i
            assert native["target"] == degraded["target"] and native["split"] == degraded["split"] == "train"
            for case in (native, degraded):
                assert digest(raw(case["input"])) == case["input_sha256"]
                for field in case["target"].values():
                    assert digest(raw(field["path"])) == field["sha256"]
        fixtures = json.loads((ROOT / "outputs/reflection_coverage_data_v1/manifest.json").read_text())["cases"]
        rng = random.Random(42)
        for epoch in range(2):
            pos = [i for i, r in enumerate(real[:83]) if r["kind"] == "covered"]
            clear = [i for i, r in enumerate(real[:83]) if r["kind"] == "uncovered"]
            covered_fixture = [r["case_id"] for r in fixtures if r["style"] != "clear"]
            clear_fixture = [r["case_id"] for r in fixtures if r["style"] == "clear"]
            for pool in (pos, clear, covered_fixture, clear_fixture):
                rng.shuffle(pool)
            for arm in p["arms"]:
                batches = p["schedules"][arm][epoch]
                assert len(batches) == 56
                assert Counter(i for b in batches for i in b["fixture"]) == Counter(range(280))
                for step, batch in enumerate(batches):
                    expected_indices = [pos[2 * step % 51], pos[(2 * step + 1) % 51], clear[step % 32]]
                    expected_conditions = [False] * 3 if arm == "native83" else [bool((epoch + step + j) % 2) for j in range(3)]
                    if arm == "camera91":
                        if step % 4 == 0:
                            expected_indices[0] = 84 + (step // 4) % 7
                            expected_conditions[0] = bool((step // 4 + epoch) % 2)
                        if step in (0, 8, 16):
                            expected_indices[2] = 83
                            expected_conditions[2] = bool((step // 8 + epoch) % 2)
                    assert batch["real"] == [{"index": i, "degraded": d} for i, d in zip(expected_indices, expected_conditions)]
                    assert batch["fixture"] == covered_fixture[4 * step:4 * step + 4] + [clear_fixture[step]]
                    assert Counter(real[i]["kind"] for i in expected_indices) == {"covered": 2, "uncovered": 1}
                assert {r["index"] for b in batches for r in b["real"]} == set(range(91 if arm == "camera91" else 83))
        split = json.loads(raw("inputs/phase4_split.json"))
        assert digest(raw("inputs/phase4_split.json")) == p["phase4_split_sha256"]
        normalize = lambda values: {v.replace("\\", "/") for v in values}
        assert not normalize(split["train"]).intersection(normalize(split["validation"]))
        assert all(r["source"].replace("\\", "/") in normalize(split["train"]) for r in real[73:83] + fixtures)
    record = {"format": "dgp-real-camera-package-independent-verification-v1", "date": "2026-10-02",
              "complete": True, "archive_sha256": ARCHIVE_SHA, "protocol_sha256": PROTOCOL_SHA,
              "auditor_sha256": digest(Path(__file__).read_bytes()), "files_verified": 419,
              "safe_regular_archive_members_and_lf_checksum": True, "code_files_compiled": len(p["code_sha256"]),
              "windows_cuda_guard_is_first": True, "dataset_records": 123, "camera_cases": 182,
              "schedule_independently_reconstructed": True, "original_split_and_data_preserved": True,
              "new_subject_cohort_training_slots": 34, "optimizer_updates_per_arm": 112,
              "total_vm_optimizer_update_budget": 336, "actual_cuda_execution_verified": False,
              "local_model_forwards": 0, "local_optimizer_updates": 0, "promoted": False}
    OUT.mkdir()
    destination = OUT / "verification.json"
    destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": True, "files_verified": 419, "verification_sha256": digest(destination.read_bytes())}))


if __name__ == "__main__":
    main()
