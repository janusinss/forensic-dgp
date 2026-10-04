"""Independent finite-pilot return verification; no models, optimizers or training."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tarfile

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_real_camera_results import (require, sha, read, write, safe, exact,
                                              binary, recount, aggregate)

ARCHIVE = ROOT / "outputs/varied-covering-results.tar.gz"
EXTRACT = ROOT / "outputs/downloaded_varied_covering_v1"
OUT = ROOT / "outputs/varied_covering_results_validation_v1"
PREFIX = "varied_covering_vm_results/"
RESULT = "outputs/varied_covering_vm/"
ARMS = ("existing91", "varied133")
FAMILIES = ("hand", "hair", "cloth", "object", "eyewear_mask")
PROTOCOL_SHA = "a6ff0780c86a9dc207477b5b40b6ccf6da5d88cd10fc5bb3624fb9e177d13928"
INVENTORY_SHA = "d7d4e50ace391a86771ae2bd6b59a7d2e2f6a5fcdbca67155c717bf8738c3e93"
PACKAGE_SHA = "780c44f2e3f724ff600eb869e99422b1bf81b18678353a59ac68afb137e32d0b"
PACKAGE_AUDIT_SHA = "2c82066ea9a608e5e75907f4b5245e955bacb8aab2a2f85aebe95bff7580c66a"
MODEL_PATH = "outputs/downloaded_real_camera_v1/outputs/real_camera_vm/camera91/last.pth"
ASSET_PATHS = {"model": MODEL_PATH, "fixture_manifest": "outputs/reflection_coverage_data_v1/manifest.json",
               "fixture_pixels": "outputs/reflection_coverage_data_v1/pixels.pth"}


def expected_files():
    names = {"bundle_inventory.json", "inputs/varied_covering_protocol.json"}
    names.update(RESULT + name for name in ("baseline.json", "complete.json", "execution.json", "fit_decision.json",
                                          "preflight.json", "return_inventory.json", "preview.png"))
    for arm in ARMS:
        names.update(RESULT + arm + "/" + name for name in ("last.pth", "metrics.json", "steps.jsonl"))
    for folder in ("initial_masks", *(arm + "/final_masks" for arm in ARMS)):
        for domain, size in (("real", 266), ("fixture", 280)):
            names.update(RESULT + folder + f"/{domain}/{i:03d}.png" for i in range(size))
    require(len(names) == 1653, "Return auditor file budget differs")
    return names


def validate_member(member, expected, folded, extraction_root=EXTRACT):
    require(member.isfile() and member.name.startswith(PREFIX), "Unexpected archive type/root")
    name = member.name[len(PREFIX):]
    safe(extraction_root, name)
    require(name in expected and name.casefold() not in folded, "Extra or duplicate archive member")
    limit = 64 * 1024**2 if name.endswith(".pth") else 2 * 1024**2
    require(0 <= member.size <= limit, "Returned member exceeds fixed size budget")
    return name


def extract_return():
    require(ARCHIVE.is_file() and ARCHIVE.with_name(ARCHIVE.name + ".sha256").is_file(),
            "Download varied-covering-results.tar.gz and its .sha256 into outputs first")
    require(not EXTRACT.exists() and not OUT.exists(), "Preserve earlier extraction/audit evidence")
    digest = sha(ARCHIVE)
    require(ARCHIVE.with_name(ARCHIVE.name + ".sha256").read_bytes() ==
            (digest + "  " + ARCHIVE.name + "\n").encode("ascii"), "Transfer/LF checksum differs")
    expected = expected_files()
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        members, folded, total = {}, set(), 0
        for member in tar.getmembers():
            name = validate_member(member, expected, folded)
            folded.add(name.casefold())
            members[name] = member
            total += member.size
        require(set(members) == expected and total <= 192 * 1024**2, "Return membership/total budget differs")
        inventory = json.load(tar.extractfile(members[RESULT + "return_inventory.json"]))
        require(set(inventory) == expected - {RESULT + "return_inventory.json"}, "Return inventory coverage differs")
        require(all(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) for value in inventory.values()),
                "Invalid returned digest")
        EXTRACT.mkdir()
        hashes = {}
        for name, member in members.items():
            path = safe(EXTRACT, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            with tar.extractfile(member) as stream, path.open("xb") as output:
                for chunk in iter(lambda: stream.read(1024**2), b""):
                    h.update(chunk)
                    output.write(chunk)
            hashes[name] = h.hexdigest()
            if name in inventory:
                require(hashes[name] == inventory[name], "Returned inventory hash differs: " + name)
    return digest, hashes


def audit_steps(records, schedule, arm):
    require(len(records) == 128 and len(schedule) == 2 and all(len(epoch) == 64 for epoch in schedule),
            "Incomplete/extra step log or schedule")
    for number, row in enumerate(records):
        fixed = {"arm": arm, "experiment_epoch": number // 64 + 1, "step": number % 64 + 1,
                 "experiment_updates": number + 1, "optimizer_state_step": number + 1,
                 "cumulative_model_updates": 995 + number, **schedule[number // 64][number % 64]}
        require(set(row) == set(fixed) | {"loss", "real_loss", "fixture_loss", "pre_clip_norm"}, "Step fields differ")
        for key, value in fixed.items():
            exact(row[key], value, arm + "/step/" + key)
        require(all(type(row[key]) is float and math.isfinite(row[key]) and row[key] >= 0
                    for key in ("loss", "real_loss", "fixture_loss", "pre_clip_norm")), "Invalid logged loss/gradient")
        require(math.isclose(row["loss"], .5 * (row["real_loss"] + row["fixture_loss"]), rel_tol=1e-6, abs_tol=1e-7),
                "Weighted logged loss differs from fp32 calculation")
    return {"logged_updates": 128, "first_fresh_moment_step": 1, "last_fresh_moment_step": 128,
            "cumulative_model_updates": 1122, "finite_losses_and_gradients": True, "frozen_schedule_verified": True}


def fit_checks(final):
    control, treatment = (final[arm]["groups"] for arm in ARMS)

    def retained(a, b):
        return a["iou"] >= b["iou"] and all(a[key] <= b[key] for key in
                   ("missed_fraction", "visible_false_positive", "empty_mask_cases", "negative_false_positive_cases"))

    checks = {
        "all_new_families_native_and_degraded_iou_gain": all(treatment[f"cofw_{family}_{condition}"]["iou"] >
             control[f"cofw_{family}_{condition}"]["iou"] for family in FAMILIES for condition in ("native", "degraded")),
        "old_native_and_degraded_fit_retained": all(retained(treatment[group], control[group])
                                                    for group in ("old_native", "old_degraded")),
        "reflection_fit_retained": retained(treatment["reflection"], control["reflection"]),
        "all_final_clear_controls_empty": all(group["negative_false_positive_cases"] == 0
                                               for value in final.values() for group in value["groups"].values()),
        "new_covered_masks_nonempty_native_and_degraded": all(treatment[f"cofw_{family}_{condition}"]["empty_mask_cases"] == 0
             for family in FAMILIES for condition in ("native", "degraded")),
    }
    return {"checks": checks, "all_training_fit_checks_pass": all(checks.values()), "training_fit_only": True,
            "original_425_case_gates_evaluated": False, "historical_gate_failure_unchanged": True,
            "selected_checkpoint": None, "promoted": False,
            "next": "Audit returned states/masks and full preview, then decide original-gate and practical-output evaluation; do not promote from training fit"}


def checkpoint_audit(payload, source, protocol, arm):
    import torch
    expected = {key: value for key, value in source.items() if key != "model"}
    expected.update(epoch=46, additional_epoch=36, optimizer_updates=1122, fresh_optimizer_updates=128,
                    experiment_updates=128, source_varied_checkpoint_counters=protocol["source_counters"], optimizer_reset=True,
                    varied_covering_arm=arm, varied_covering_protocol_sha256=PROTOCOL_SHA, detector_only=True,
                    source_varied_selection=source["selection"],
                    selection={"selected": False, "training_fit_only": True, "historical_gates_not_evaluated": True})
    exact({key: value for key, value in payload.items() if key != "model"}, expected, arm + "/checkpoint")
    state, before = payload["model"], source["model"]
    require(set(state) == set(before), "Checkpoint tensor membership differs")
    require(all(isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.shape == before[key].shape
                and value.dtype == before[key].dtype and torch.isfinite(value).all() for key, value in state.items()),
            "Invalid detector state tensor")
    frozen = [key for key in state if key.startswith("reference_visible_head.")
              or key.endswith(("running_mean", "running_var", "num_batches_tracked"))]
    require(len(frozen) == 92 and all(torch.equal(state[key], before[key]) for key in frozen), "Frozen head/BN tensors changed")
    changed = {prefix: sum(not torch.equal(value, before[key]) for key, value in state.items() if key.startswith(prefix))
               for prefix in ("network.encoder.", "network.decoder.", "network.segmentation_head.")}
    require(all(changed.values()), "Declared trainable component unchanged")
    return {"tensor_states": len(state), "frozen_tensors_verified": len(frozen), "changed_tensors": changed,
            "metadata_verified": True, "final_optimizer_tensors_returned": False}


def recreate_preview(cases, ids):
    canvas = Image.new("RGB", (960, 42 + 218 * len(ids)), "#16181c")
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(("input", "target core", "camera91 source", *ARMS)):
        draw.text((column * 192 + 3, 7), title, fill="white")
    draw.text((3, 24), "TRAIN-COHORT MASKS: green TP, red FP, yellow FN, purple ignored; no generated faces.", fill="white")
    for row, case_id in enumerate(ids):
        case = cases[case_id]
        with Image.open(ROOT / case["input"]) as image:
            rgb = np.asarray(image).copy()
        truth, support = (binary(ROOT / case["target"][key]["path"]) for key in ("mask", "valid"))

        def overlay(prediction):
            color = rgb.copy()
            color[~support] = (165, 85, 215)
            color[truth & prediction & support] = (16, 185, 129)
            color[~truth & prediction & support] = (245, 50, 50)
            color[truth & ~prediction & support] = (255, 220, 40)
            marked = truth | prediction | ~support
            result = rgb.copy()
            result[marked] = (.55 * rgb[marked] + .45 * color[marked]).round().astype(np.uint8)
            return result

        panels = [rgb, overlay(truth)]
        for folder in ("initial_masks", *(arm + "/final_masks" for arm in ARMS)):
            panels.append(overlay(binary(EXTRACT / RESULT / folder / "real" / f"{case_id:03d}.png")))
        y = 42 + row * 218
        for column, panel in enumerate(panels):
            canvas.paste(Image.fromarray(panel).resize((192, 192)), (column * 192, y))
        draw.text((3, y + 195), f"{case['source_id']}/{case['condition']}", fill="white")
    with Image.open(EXTRACT / RESULT / "preview.png") as returned:
        require(returned.mode == "RGB" and np.array_equal(np.asarray(returned), np.asarray(canvas)), "Preview pixels differ")
    return canvas


def main():
    digest, hashes = extract_return()  # Missing or malformed returns cannot construct an extraction/quality report.
    import torch
    torch.set_num_threads(4)
    require(sha(ROOT / "outputs/varied-covering-vm-bundle-v2.tar.gz") == PACKAGE_SHA, "Sent V2 package changed")
    require(sha(ROOT / "outputs/varied_covering_package_validation_v2/verification.json") == PACKAGE_AUDIT_SHA,
            "Independent sent-package audit changed")
    require(hashes["inputs/varied_covering_protocol.json"] == sha(ROOT / "outputs/varied_covering_protocol_v2/protocol.json") == PROTOCOL_SHA,
            "Returned frozen protocol differs")
    require(hashes["bundle_inventory.json"] == sha(ROOT / "outputs/varied_covering_protocol_v2/inventory.json") == INVENTORY_SHA,
            "Returned sent-inventory differs")
    protocol = read(EXTRACT / "inputs/varied_covering_protocol.json")
    for name, pin in protocol["code_sha256"].items():
        require(sha(safe(ROOT, name)) == pin, "Frozen runner dependency changed")
    for name, pin in protocol["local_audited_inputs"].items():
        require(sha(safe(ROOT, name)) == pin, "Frozen audited input changed")
    for role, name in ASSET_PATHS.items():
        require(sha(ROOT / name) == protocol["existing_assets"][role]["sha256"], "Source/fixture asset changed")
    pairs = read(ROOT / "outputs/cofw_camera_pairs_v2/manifest.json")
    cases = pairs["cases"]
    registry = read(ROOT / "dataset/detector_supported_review_v3/manifest.json")
    rows = [row for row in registry["supported_records"] if row["split"] == "train"]
    require(len(cases) == 266 and len(rows) == 133 and pairs["held_out_inputs_in_cache"] == 0, "Training case budget differs")
    fixture_rows = read(ROOT / ASSET_PATHS["fixture_manifest"])["cases"]
    cache = torch.load(ROOT / ASSET_PATHS["fixture_pixels"], map_location="cpu", weights_only=True)
    require(cache["format"] == "dgp-reflection-coverage-pixels-v1" and set(cache["pixels"]) == set(range(280)), "Fixture cache differs")
    mapping = {"hand": "hand", "obstructing_hair": "hair", "cloth_or_scarf": "cloth", "other_cloth_or_object": "cloth",
               "object": "object", "sunglasses": "eyewear_mask", "costume_mask": "eyewear_mask", "face_mask_preserve_clear_goggles": "eyewear_mask"}
    targets = {}
    for case_id, case in enumerate(cases):
        index = case_id // 2
        require(case["case_id"] == case_id and case["real_train_index"] == index and case["split"] == "train"
                and case["condition"] == ("degraded" if case_id % 2 else "native"), "Case ordering/membership differs")
        require(sha(ROOT / case["input"]) == case["input_sha256"], "Cached RGB differs")
        for asset in case["target"].values():
            require(sha(ROOT / asset["path"]) == asset["sha256"], "Target/support asset changed")
        target, valid = (binary(ROOT / case["target"][key]["path"]) for key in ("mask", "valid"))
        group = "old" if index < 91 else ("cofw_clear" if rows[index]["kind"] == "uncovered"
                                          else "cofw_" + mapping[rows[index]["occlusion_stratum"]])
        targets[f"real/{case_id:03d}"] = (target, valid, group + "_" + case["condition"])
    for index, item in cache["pixels"].items():
        require(set(item) == {"input", "mask", "valid", "geometry"}, "Fixture tensor fields differ")
        for name, tensor in item.items():
            require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu" and tensor.dtype == torch.uint8
                    and tensor.shape == ((3 if name == "input" else 1), 256, 256), "Fixture tensor format differs")
            require(hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest() == fixture_rows[index]["pixel_sha256"][name],
                    "Fixture raw pixels differ")
            if name != "input":
                require(((tensor == 0) | (tensor == 1)).all(), "Nonbinary fixture map")
        targets[f"fixture/{index:03d}"] = (item["mask"][0].numpy().astype(bool), item["valid"][0].numpy().astype(bool), "reflection")

    def measure(folder, logged):
        groups, records, mask_hashes, errors = defaultdict(list), [], {}, []
        for name, (target, valid, group) in targets.items():
            path = EXTRACT / RESULT / folder / (name + ".png")
            counted = recount(binary(path), target, valid)
            groups[group].append(counted)
            records.append({"id": name, "group": group, "counts": counted})
            mask_hashes[name + ".png"] = sha(path)
            if counted["negative_false_positive_cases"] or counted["empty_mask_cases"]:
                errors.append({"id": name, "group": group, **counted})
        calculated = {"groups": {key: aggregate(value) for key, value in groups.items()}, "rows": records,
                      "mask_sha256": mask_hashes, "forward_images": 546, "held_out_forward_images": 0,
                      "scope": "Exposed training-cohort diagnosis only; added sources are not an unseen holdout"}
        exact(logged, calculated, folder + "/metrics")
        return calculated, errors

    baseline, baseline_errors = measure("initial_masks", read(EXTRACT / RESULT / "baseline.json"))
    source = torch.load(ROOT / MODEL_PATH, map_location="cpu", weights_only=True)
    final, errors, states, steps = {}, {"camera91_source": baseline_errors}, {}, {}
    for arm in ARMS:
        final[arm], errors[arm] = measure(arm + "/final_masks", read(EXTRACT / RESULT / arm / "metrics.json"))
        records = [json.loads(line) for line in (EXTRACT / RESULT / arm / "steps.jsonl").read_text().splitlines()]
        steps[arm] = audit_steps(records, protocol["schedules"][arm], arm)
        payload = torch.load(EXTRACT / RESULT / arm / "last.pth", map_location="cpu", weights_only=True)
        states[arm] = checkpoint_audit(payload, source, protocol, arm)
        del payload
    decision = fit_checks(final)
    exact(read(EXTRACT / RESULT / "fit_decision.json"), decision, "fit decision")
    complete = read(EXTRACT / RESULT / "complete.json")
    fixed = {"complete": True, "protocol_sha256": PROTOCOL_SHA, "updates_per_arm": 128, "total_optimizer_updates": 256,
             "moment_steps_per_arm": 128, "cumulative_model_updates_per_arm": 1122, "optimizer_reset": True,
             "actual_model_forward_images": 3686, "held_out_forward_images": 0, "fit_decision": decision,
             "original_425_case_gates_evaluated": False, "promoted": False, "selected_checkpoint": None,
             "preview_sha256": hashes[RESULT + "preview.png"], "original_assets_unchanged": True}
    require(set(complete) == set(fixed) | {"seconds", "checkpoints"}, "Completion fields differ")
    for key, value in fixed.items():
        exact(complete[key], value, "complete/" + key)
    require(type(complete["seconds"]) is float and 0 < complete["seconds"] < 1800, "Execution time outside finite cap")
    require(set(complete["checkpoints"]) == set(ARMS), "Final checkpoint registry differs")
    for arm, binding in complete["checkpoints"].items():
        require(set(binding) == {"model_sha256", "optimizer_sha256", "optimizer_retained_on_vm_not_in_return_archive"}
                and re.fullmatch("[0-9a-f]{64}", binding["optimizer_sha256"])
                and binding["optimizer_retained_on_vm_not_in_return_archive"] is True, "Optimizer retention binding differs")
        exact(binding["model_sha256"], hashes[RESULT + arm + "/last.pth"], "Checkpoint file digest")
    preflight, execution = (read(EXTRACT / RESULT / name) for name in ("preflight.json", "execution.json"))
    fixed_preflight = {"complete": True, "batch_size": 1, "protocol_sha256": PROTOCOL_SHA,
                       "gpu": "NVIDIA L4", "cuda_available": True, "fresh_optimizer_planned": True,
                       "optimizer_constructed": False, "optimizer_updates": 0,
                       "actual_model_forward_images": 1, "held_out_forward_images": 0}
    require(set(preflight) == set(fixed_preflight) | {"loss", "dependencies"}, "Preflight fields differ")
    for key, value in fixed_preflight.items():
        exact(preflight[key], value, "preflight/" + key)
    require(type(preflight["loss"]) is float and math.isfinite(preflight["loss"]) and preflight["loss"] >= 0, "Nonfinite preflight loss")
    dependencies = {"segmentation-models-pytorch": "0.5.0", "timm": "1.0.15", "huggingface-hub": "0.29.3", "safetensors": "0.5.3", "PyYAML": "6.0.2"}
    require(set(preflight["dependencies"]) == set(dependencies), "Pinned dependency membership differs")
    for distribution, version in dependencies.items():
        row = preflight["dependencies"][distribution]
        require(set(row) == {"version", "module_path"} and row["version"] == version
                and "/outputs/face_occlusion_dependencies/" in row["module_path"], "Pinned runtime dependency differs")
    exact(execution, {"protocol_sha256": PROTOCOL_SHA, "inventory_sha256": INVENTORY_SHA, "gpu": "NVIDIA L4",
                     "torch": "2.9.1+cu129", "arms": list(ARMS), "updates_per_arm": 128, "total_update_budget": 256,
                     "optimizer_reset": True, "source_model_updates": 994, "fresh_moment_start_step": 0,
                     "held_out_forward_images": 0, "promoted": False}, "execution")
    grid = recreate_preview(cases, protocol["preview_case_ids"])
    OUT.mkdir()
    grid.save(OUT / "verified_preview.png")
    report = {"format": "dgp-varied-covering-return-independent-verification-v1", "date": "2026-10-03", "complete": True,
              "archive_sha256": digest, "archive_bytes": ARCHIVE.stat().st_size, "members_verified": len(hashes),
              "auditor_sha256": sha(__file__), "protocol_sha256": PROTOCOL_SHA, "saved_masks_recounted": 1638,
              "logged_steps_verified": 256, "local_optimizer_constructed": False, "local_optimizer_updates": 0,
              "local_model_forwards": 0, "checkpoint_checks": states, "step_checks": steps,
              "baseline": baseline["groups"], "final": {arm: final[arm]["groups"] for arm in ARMS},
              "empty_or_clear_error_cases": errors, "fit_decision": decision, "seconds": complete["seconds"],
              "checkpoint_sha256": {arm: complete["checkpoints"][arm]["model_sha256"] for arm in ARMS},
              "preview_pixels_reproduced_exactly": True, "visual_review_pending": True, "promoted": False,
              "limitations": ["Final optimizer tensors remain on VM; moment counters are verified log claims, not locally inspected moments",
                              "Saved-mask recount and state inspection do not reproduce model execution; CPU prediction checks remain separate",
                              "Only exposed training-cohort evidence; original425 gates and fixed practical output review remain unevaluated",
                              "Approximate assistant labels; identity/pretraining overlap and true hidden facial appearance remain unknown",
                              "Remote preservation/forward counters are frozen-code/log claims, not independently observed live VM state"]}
    write(OUT / "members.json", hashes)
    write(OUT / "verification.json", report)
    print(json.dumps({"complete": True, "members": len(hashes), "masks": 1638, "steps": 256,
                      "fit_checks": decision["checks"], "verification_sha256": sha(OUT / "verification.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()
