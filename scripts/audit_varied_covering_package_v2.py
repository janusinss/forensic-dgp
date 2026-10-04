"""Independently verify finite VM bundle/inventory/sampling; never execute models."""
import ast
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
import tarfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "outputs/varied-covering-vm-bundle-v2.tar.gz"
OUT = ROOT / "outputs/varied_covering_package_validation_v2"
ARCHIVE_SHA = "780c44f2e3f724ff600eb869e99422b1bf81b18678353a59ac68afb137e32d0b"
PROTOCOL_SHA = "a6ff0780c86a9dc207477b5b40b6ccf6da5d88cd10fc5bb3624fb9e177d13928"
PREFIX = "varied_covering_vm_bundle/"


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    require(not OUT.exists(), "Preserve previous independent package audit")
    require(digest(ARCHIVE.read_bytes()) == ARCHIVE_SHA, "Frozen archive differs")
    require(ARCHIVE.with_name(ARCHIVE.name + ".sha256").read_bytes() ==
            (ARCHIVE_SHA + "  " + ARCHIVE.name + "\n").encode("ascii"), "LF checksum sidecar differs")
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        members, folded = {}, set()
        for member in tar.getmembers():
            require(member.isfile() and member.name.startswith(PREFIX) and member.size <= 32 * 1024**2,
                    "Non-regular, oversize or non-prefix archive member")
            name = member.name[len(PREFIX):]
            require(name and "\\" not in name and ":" not in name and not name.startswith("/")
                    and PurePosixPath(name).as_posix() == name and ".." not in name.split("/")
                    and name.casefold() not in folded, "Unsafe/duplicate member")
            folded.add(name.casefold())
            members[name] = member
        require(len(members) == 1010, "Frozen archive inventory count differs")

        def raw(name):
            require(name in members, "Missing archive member: " + name)
            blob = tar.extractfile(members[name]).read()
            require(len(blob) == members[name].size, "Incomplete archive member")
            return blob

        inventory = json.loads(raw("inventory.json"))
        require(set(members) == set(inventory) | {"inventory.json"}, "Archive inventory coverage differs")
        aliases = {"inputs/varied_covering_protocol.json": "outputs/varied_covering_protocol_v2/protocol.json",
                   "inputs/phase4_split.json": "outputs/downloaded_phase4/outputs/phase4_with_progress/split.json",
                   "inputs/parent_supported_manifest.json": "dataset/detector_supported_review_v2/manifest.json"}
        for name, expected in inventory.items():
            blob = raw(name)
            require(digest(blob) == expected == digest((ROOT / aliases.get(name, name)).read_bytes()),
                    "Archived/current source hash differs: " + name)
        spec = raw("inputs/varied_covering_protocol.json")
        require(digest(spec) == PROTOCOL_SHA, "Frozen protocol differs")
        p = json.loads(spec)
        require(p["format"] == "dgp-varied-covering-pilot-v1" and p["arms"] == ["existing91", "varied133"], "Wrong experiment")
        require((p["epochs_per_arm"], p["steps_per_epoch"], p["updates_per_arm"], p["total_update_budget"],
                 p["batch_size"], p["real_batch_size"], p["fixture_batch_size"]) == (2, 64, 128, 256, 8, 4, 4), "Budget differs")
        require(p["optimizer_reset"] is True and p["initial_optimizer_state_step"] == 0
                and p["final_optimizer_state_step"] == 128 and p["final_cumulative_model_updates"] == 1122, "Fresh optimizer counters differ")
        require(p["source_counters"] == {"epoch": 44, "additional_epoch": 34, "model_updates": 994,
                "previous_moment_step_metadata": 784, "source_experiment_updates": 112}, "Source lineage differs")
        require(p["measurement_images_per_checkpoint"] == 546 and p["measurement_checkpoints"] == 3
                and p["actual_training_forward_image_budget"] == 256 * 8
                and p["training_and_measurement_forward_image_budget"] == 546 * 3 + 256 * 8 == 3686,
                "Forward-image budget differs")
        require(p["held_out_forward_images"] == 0 and p["original_425_case_gates_evaluated"] is False
                and p["historical_gate_failure_unchanged"] is True and p["actual_cuda_execution_verified"] is False
                and p["promoted"] is False and p["wall_budget_seconds_after_loading"] == 1800, "Selection/scope differs")
        for name, expected in p["code_sha256"].items():
            require(digest(raw(name)) == expected, "Bound code differs")
            compile(raw(name), name, "exec")
        tree = ast.parse(raw("scripts/train_varied_covering_vm.py"))
        main_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        require(isinstance(main_node.body[0], ast.Expr) and isinstance(main_node.body[0].value, ast.Call)
                and main_node.body[0].value.func.id == "require_vm_gpu", "VM guard must be first")
        require(not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "load_state_dict"
                        for n in ast.walk(main_node)), "Runner must not restore inherited optimizer state")
        for name, expected in p["local_audited_inputs"].items():
            require(digest((ROOT / name).read_bytes()) == expected, "Preceding audited input changed")
        require(p["existing_assets"]["model"]["root"] == "camera"
                and p["existing_assets"]["model"]["vm_path"] == "outputs/real_camera_vm/camera91/last.pth"
                and p["existing_assets"]["model"]["sha256"] == "e4b16da0ccb92b2a3a6b29f10b863320ccf3ead1fb9e1464f753a667f76c1702",
                "Model source binding differs")
        require(not any(name.endswith(("last.pth", "final_optimizer.pth", "pixels.pth")) for name in members),
                "Read-only large model/optimizer assets must not be uploaded")
        droot = "dataset/detector_supported_review_v3/"
        registry = json.loads(raw(droot + "manifest.json"))
        parent = json.loads(raw("inputs/parent_supported_manifest.json"))
        require(digest(raw(droot + "manifest.json")) == p["registry_sha256"] and
                registry["supported_records"][:123] == parent["supported_records"], "Preserved record lineage differs")
        all_rows = registry["supported_records"]
        rows = [r for r in all_rows if r["split"] == "train"]
        require(len(all_rows) == 165 and len(rows) == 133, "Fixed registry membership differs")
        for split in ("validation", "test"):
            require([r for r in all_rows if r["split"] == split] == [r for r in parent["supported_records"] if r["split"] == split],
                    "Held-out records/support changed")

        def image(name):
            with Image.open(io.BytesIO(raw(name))) as source:
                return np.asarray(source).copy()

        raw_sources = set()
        for row in all_rows:
            arrays = {}
            for key in ("image", "mask", "valid", "source_valid"):
                name = droot + row[key]
                require(digest(raw(name)) == row[key + "_sha256"], "Registered PNG changed")
                arrays[key] = image(name)
            rgb, mask, valid, source = (arrays[k] for k in ("image", "mask", "valid", "source_valid"))
            require(rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8
                    and all(a.shape == (256, 256) and np.isin(a, [0, 255]).all() for a in (mask, valid, source)), "PNG shape/type differs")
            require(not ((mask > 0) & (valid == 0)).any() and not ((valid > 0) & (source == 0)).any()
                    and np.all(rgb[source == 0] == 96) and bool(mask.any()) == (row["kind"] == "covered"), "Support/padding/kind differs")
            name = row["source"].replace("\\", "/")
            require(digest(raw(name)) == row["source_sha256"], "Raw-source closure missing/changed")
            raw_sources.add(name)
        pair_blob = raw("outputs/cofw_camera_pairs_v2/manifest.json")
        pairs = json.loads(pair_blob)
        require(digest(pair_blob) == p["pairs_sha256"] and len(pairs["cases"]) == 266 and pairs["held_out_inputs_in_cache"] == 0, "Pair cache differs")
        require(p["preflight_case_id"] == 202 and pairs["cases"][202]["source_id"] == "cofw_train_0868"
                and pairs["cases"][202]["condition"] == "native" and pairs["cases"][202]["family"] == "obstructing_hair",
                "Reviewed native hair preflight binding differs")
        for index, row in enumerate(rows):
            native, degraded = pairs["cases"][2 * index:2 * index + 2]
            require(native["real_train_index"] == degraded["real_train_index"] == index
                    and native["target"] == degraded["target"] and native["split"] == degraded["split"] == "train", "Pair membership differs")
            for case in (native, degraded):
                require(digest(raw(case["input"])) == case["input_sha256"], "Paired input changed")
                for field in case["target"].values():
                    require(digest(raw(field["path"])) == field["sha256"], "Pair label/support changed")
        fixture_path = ROOT / "outputs/reflection_coverage_data_v1/manifest.json"
        require(digest(fixture_path.read_bytes()) == p["existing_assets"]["fixture_manifest"]["sha256"], "Read-only fixture manifest differs")
        fixtures = json.loads(fixture_path.read_text())["cases"]
        old_pos = [i for i, r in enumerate(rows[:91]) if r["kind"] == "covered"]
        old_neg = [i for i, r in enumerate(rows[:91]) if r["kind"] == "uncovered"]
        new_neg = [i for i in range(91, 133) if rows[i]["kind"] == "uncovered"]
        mapping = {"hand": "hand", "obstructing_hair": "hair", "cloth_or_scarf": "cloth", "other_cloth_or_object": "cloth",
                   "object": "object", "sunglasses": "eyewear_mask", "costume_mask": "eyewear_mask", "face_mask_preserve_clear_goggles": "eyewear_mask"}
        names = ("hand", "hair", "cloth", "object", "eyewear_mask")
        pools = {name: [i for i in range(91, 133) if rows[i]["kind"] == "covered" and mapping[rows[i]["occlusion_stratum"]] == name] for name in names}
        fp = [r["case_id"] for r in fixtures if r["style"] != "clear"]
        fc = [r["case_id"] for r in fixtures if r["style"] == "clear"]
        rng = random.Random(42)
        for pool in (old_pos, old_neg, new_neg, fp, fc, *(pools[name] for name in names)):
            rng.shuffle(pool)
        occurrence = [Counter() for _ in range(4)]
        seen = {arm: set() for arm in p["arms"]}
        fixture_seen = set()
        new_families = Counter()
        for number in range(128):
            epoch, step = divmod(number, 64)
            group = names[number % 5]
            selected = [old_pos[number % 58], pools[group][(number // 5) % len(pools[group])], old_neg[number % 33], new_neg[number % 11]]
            conditions = [bool(occurrence[slot][index] % 2) for slot, index in enumerate(selected)]
            for slot, index in enumerate(selected):
                occurrence[slot][index] += 1
            control = [selected[0], old_pos[(number + 29) % 58], selected[2], old_neg[(number + 16) % 33]]
            replay = [fp[2 * number % 224], fp[(2 * number + 1) % 224], fc[2 * number % 56], fc[(2 * number + 1) % 56]]
            for arm, indices in zip(p["arms"], (control, selected)):
                batch = p["schedules"][arm][epoch][step]
                require(batch == {"real": [{"index": i, "degraded": d} for i, d in zip(indices, conditions)], "fixture": replay},
                        "Independently reconstructed sampling differs")
                seen[arm].update(zip(indices, conditions))
            fixture_seen.update(replay)
            new_families[group] += 1
        for arm, size in zip(p["arms"], (91, 133)):
            require(seen[arm] == {(i, d) for i in range(size) for d in (False, True)}, "Native/degraded source exposure missing")
        require(fixture_seen == set(range(280)) and max(new_families.values()) - min(new_families.values()) == 1,
                "Fixture/family exposure differs")
        require(len(p["preview_case_ids"]) == len(set(p["preview_case_ids"])) == 10, "Preview budget differs")
        require([(pairs["cases"][i]["source_id"], pairs["cases"][i]["condition"]) for i in p["preview_case_ids"]] ==
                [tuple(spec) for spec in p["preview_specs"]], "Preview binding differs")
        split = json.loads(raw("inputs/phase4_split.json"))
        require(digest(raw("inputs/phase4_split.json")) == p["phase4_split_sha256"], "Original split changed")
        train = {path.replace("\\", "/") for path in split["train"]}
        require(not train.intersection(path.replace("\\", "/") for path in split["validation"])
                and all(row["source"].replace("\\", "/") in train for row in rows[73:83] + fixtures), "Original split leakage")
    report = {"format": "dgp-varied-covering-package-independent-verification-v1", "date": "2026-10-03", "complete": True,
              "archive_sha256": ARCHIVE_SHA, "archive_bytes": ARCHIVE.stat().st_size, "protocol_sha256": PROTOCOL_SHA,
              "auditor_sha256": digest(Path(__file__).read_bytes()), "files_verified": 1010,
              "safe_regular_members_and_lf_checksum": True, "code_files_compiled": len(p["code_sha256"]),
              "vm_guard_first": True, "inherited_optimizer_restore_absent": True,
              "preflight_hair_source_verified": "cofw_train_0868/native/202",
              "registry_records": 165, "paired_inputs": 266, "raw_sources_verified": len(raw_sources),
              "all_source_views_and_fixtures_exposed": True, "schedule_independently_reconstructed": True,
              "new_positive_family_slots": dict(new_families), "old_held_out_records_preserved": True,
              "updates_per_arm": 128, "total_optimizer_update_budget": 256, "fresh_optimizer_start": 0,
              "actual_cuda_execution_verified": False, "local_model_forwards": 0, "local_optimizer_updates": 0, "promoted": False}
    OUT.mkdir()
    destination = OUT / "verification.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": True, "files_verified": 1010, "verification_sha256": digest(destination.read_bytes())}))


if __name__ == "__main__":
    main()
