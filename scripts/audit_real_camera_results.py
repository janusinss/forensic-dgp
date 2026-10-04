"""Independent return audit: pixels, schedules and checkpoint tensors; no training."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import re
import tarfile

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "outputs/real-camera-results.tar.gz"
EXTRACT = ROOT / "outputs/downloaded_real_camera_v1"
OUT = ROOT / "outputs/real_camera_results_validation_v1"
PREFIX = "real_camera_vm_results/"
RESULT = "outputs/real_camera_vm/"
ARMS = ("native83", "camera83", "camera91")
PROTOCOL_SHA = "a8adb21910c5aed249f5f2a75e17eef92db35717bd428326c360a1aecf3b99e1"
INVENTORY_SHA = "488c2fd610887baee4c1aec924b34296dd3875ee76ff5e16fbe3f7cc043ab868"
PACKAGE_SHA = "2b237b2e121951203bde131be5d7c649910392df47bff1bbc953f81a8c8c0e2b"
MODEL_PATH = "outputs/downloaded_reflection_coverage/outputs/reflection_coverage_vm/reflective/epoch_42.pth"
ASSET_PATHS = {
    "model": MODEL_PATH,
    "optimizer": "outputs/downloaded_reflection_coverage/outputs/reflection_coverage_vm/reflective/final_optimizer.pth",
    "parent": "outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth",
    "fixture_manifest": "outputs/reflection_coverage_data_v1/manifest.json",
    "fixture_pixels": "outputs/reflection_coverage_data_v1/pixels.pth",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def safe(root, name):
    require(isinstance(name, str) and name and "\\" not in name and ":" not in name
            and not name.startswith("/") and PurePosixPath(name).as_posix() == name
            and ".." not in name.split("/"), "Noncanonical archive/reference path")
    destination = (root / name).resolve()
    require(destination.is_relative_to(root.resolve()), "Path escapes root")
    return destination


def exact(actual, expected, context):
    """Type-aware structure comparison; bool/int substitution is not accepted."""
    require(type(actual) is type(expected), context + ": type differs")
    if isinstance(expected, dict):
        require(set(actual) == set(expected), context + ": fields differ")
        for key in expected:
            exact(actual[key], expected[key], context + "/" + str(key))
    elif isinstance(expected, list):
        require(len(actual) == len(expected), context + ": length differs")
        for i, (a, b) in enumerate(zip(actual, expected)):
            exact(a, b, context + "/" + str(i))
    elif isinstance(expected, float):
        require(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12),
                context + ": numeric value differs")
    else:
        require(actual == expected, context + ": value differs")


def expected_files():
    names = {"bundle_inventory.json", "inputs/real_camera_protocol.json"}
    names.update(RESULT + n for n in ("baseline.json", "complete.json", "execution.json",
                                     "fit_decision.json", "preflight.json", "return_inventory.json", "preview.png"))
    for arm in ARMS:
        names.update(RESULT + arm + "/" + n for n in ("last.pth", "metrics.json", "steps.jsonl"))
    for folder in ("initial_masks", *(a + "/final_masks" for a in ARMS)):
        for domain, size in (("real", 182), ("fixture", 280)):
            names.update(RESULT + folder + f"/{domain}/{i:03d}.png" for i in range(size))
    require(len(names) == 1866, "Audit file budget differs")
    return names


def extract_return():
    digest = sha(ARCHIVE)
    require(ARCHIVE.with_name(ARCHIVE.name + ".sha256").read_bytes()
            == (digest + "  " + ARCHIVE.name + "\n").encode("ascii"), "Transfer/LF checksum differs")
    require(not EXTRACT.exists() and not OUT.exists(), "Preserve existing extraction/audit evidence")
    expected = expected_files()
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        members, folded, total = {}, set(), 0
        for member in tar.getmembers():
            require(member.isfile() and member.name.startswith(PREFIX), "Unexpected archive type/root")
            name = member.name[len(PREFIX):]
            safe(EXTRACT, name)
            require(name in expected and name.casefold() not in folded, "Extra or duplicate archive member")
            limit = 64 * 1024**2 if name.endswith(".pth") else 2 * 1024**2
            require(0 <= member.size <= limit, "Returned member exceeds budget")
            folded.add(name.casefold()); members[name] = member; total += member.size
        require(set(members) == expected and total <= 256 * 1024**2, "Return membership/total budget differs")
        inventory = json.load(tar.extractfile(members[RESULT + "return_inventory.json"]))
        require(set(inventory) == expected - {RESULT + "return_inventory.json"}, "Return inventory membership differs")
        require(all(isinstance(v, str) and re.fullmatch("[0-9a-f]{64}", v) for v in inventory.values()),
                "Invalid returned digest")
        EXTRACT.mkdir()
        hashes = {}
        for name, member in members.items():
            destination = safe(EXTRACT, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            with tar.extractfile(member) as stream, destination.open("xb") as output:
                for chunk in iter(lambda: stream.read(1024**2), b""):
                    h.update(chunk); output.write(chunk)
            hashes[name] = h.hexdigest()
            if name in inventory:
                require(hashes[name] == inventory[name], "Return inventory hash differs: " + name)
    return digest, hashes


def binary(path):
    with Image.open(path) as image:
        a = np.asarray(image).copy()
        require(image.mode == "L" and image.format == "PNG" and a.shape == (256, 256)
                and a.dtype == np.uint8 and np.isin(a, [0, 255]).all(), "Invalid binary mask/support")
    return a == 255


def recount(prediction, target, valid):
    require(all(a.dtype == np.bool_ and a.shape == (256, 256) for a in (prediction, target, valid))
            and valid.any() and not (target & ~valid).any(), "Invalid supported prediction/target")
    covered, present = bool(target.any()), bool((prediction & valid).any())
    return {"tp": int(np.count_nonzero(prediction & target & valid)),
            "fp": int(np.count_nonzero(prediction & ~target & valid)),
            "fn": int(np.count_nonzero(~prediction & target & valid)),
            "visible": int(np.count_nonzero(~target & valid)),
            "covered_cases": int(covered), "empty_mask_cases": int(covered and not present),
            "negative_cases": int(not covered), "negative_false_positive_cases": int(not covered and present),
            "ignored_positive_pixels": int(np.count_nonzero(prediction & ~valid))}


def aggregate(rows):
    total = {key: sum(row[key] for row in rows) for key in rows[0]}
    return {**total, "cases": len(rows), "iou": total["tp"] / max(1, total["tp"] + total["fp"] + total["fn"]),
            "missed_fraction": total["fn"] / max(1, total["tp"] + total["fn"]),
            "visible_false_positive": total["fp"] / max(1, total["visible"])}


def independent_schedules(rows, fixtures):
    require(len(rows) == 91 and all(r["split"] == "train" for r in rows), "Training membership differs")
    schedules = {a: [] for a in ARMS}; rng = random.Random(42)
    for epoch in range(2):
        pools = [[i for i, r in enumerate(rows[:83]) if r["kind"] == kind] for kind in ("covered", "uncovered")]
        pools += [[r["case_id"] for r in fixtures if (r["style"] == "clear") == clear] for clear in (False, True)]
        require([len(p) for p in pools] == [51, 32, 224, 56], "Training strata differ")
        for p in pools:
            rng.shuffle(p)
        positive, clear, replay, empty = pools
        for arm in ARMS:
            batches = []
            for step in range(56):
                indices = [positive[2 * step % 51], positive[(2 * step + 1) % 51], clear[step % 32]]
                degraded = [False] * 3 if arm == "native83" else [bool((epoch + step + j) % 2) for j in range(3)]
                if arm == "camera91":
                    if step % 4 == 0:
                        indices[0] = 84 + (step // 4) % 7
                        degraded[0] = bool((step // 4 + epoch) % 2)
                    if step in (0, 8, 16):
                        indices[2] = 83; degraded[2] = bool((step // 8 + epoch) % 2)
                batches.append({"real": [{"index": i, "degraded": d} for i, d in zip(indices, degraded)],
                                "fixture": replay[4 * step:4 * step + 4] + [empty[step]]})
            schedules[arm].append(batches)
    return schedules


def audit_steps(records, schedule, arm):
    require(len(records) == 112, "Incomplete/extra optimizer step log")
    for i, row in enumerate(records):
        fixed = {"arm": arm, "experiment_epoch": i // 56 + 1, "step": i % 56 + 1,
                 "experiment_updates": i + 1, "optimizer_state_step": 673 + i,
                 "cumulative_model_updates": 883 + i, **schedule[i // 56][i % 56]}
        require(set(row) == set(fixed) | {"loss", "real_loss", "fixture_loss", "pre_clip_norm"}, "Step fields differ")
        for k, value in fixed.items():
            exact(row[k], value, arm + "/step/" + k)
        require(all(type(row[k]) is float and math.isfinite(row[k]) and row[k] >= 0
                    for k in ("loss", "real_loss", "fixture_loss", "pre_clip_norm")), "Invalid loss/gradient log")
        require(math.isclose(row["loss"], .5 * (row["real_loss"] + row["fixture_loss"]),
                             rel_tol=1e-6, abs_tol=1e-7), "Logged weighted loss differs from fp32 calculation")
    return {"logged_updates": 112, "last_moment_step": 784, "cumulative_model_updates": 994,
            "finite_losses_and_gradients": True, "schedule_verified": True}


def fit_checks(base, final):
    def retained(a, b):
        return a["iou"] >= b["iou"] and all(a[k] <= b[k] for k in
               ("missed_fraction", "visible_false_positive", "empty_mask_cases", "negative_false_positive_cases"))
    a, b, c = (final[x]["groups"] for x in ARMS)
    checks = {"camera83_degraded_fit_beats_native83": b["old_degraded"]["iou"] > a["old_degraded"]["iou"],
              "camera83_native_and_replay_fit_retained_vs_native83": all(retained(b[g], a[g]) for g in ("old_native", "reflection")),
              "camera91_new_native_and_degraded_fit_gain_vs_camera83": all(c[g]["iou"] > b[g]["iou"] for g in ("new_native", "new_degraded")),
              "camera91_old_native_degraded_and_replay_fit_retained": all(retained(c[g], b[g]) for g in ("old_native", "old_degraded", "reflection")),
              "all_final_clear_controls_empty": all(g["negative_false_positive_cases"] == 0 for m in final.values() for g in m["groups"].values())}
    return {"checks": checks, "all_training_fit_checks_pass": all(checks.values()),
            "source42_baseline_groups": base["groups"], "training_fit_only": True,
            "original_425_case_gates_evaluated": False, "historical_gate_failure_unchanged": True,
            "promoted": False, "selected_checkpoint": None,
            "next": "Independently audit returned states/masks and review the fixed grid before any original-gate or end-to-end evaluation"}


def checkpoint_audit(payload, source, protocol, arm):
    import torch
    expected = {k: v for k, v in source.items() if k != "model"}
    expected.update(epoch=44, additional_epoch=34, optimizer_updates=994, fresh_optimizer_updates=784,
                    experiment_updates=112, source_camera_checkpoint_counters=protocol["source_counters"],
                    real_camera_arm=arm, real_camera_protocol_sha256=PROTOCOL_SHA, detector_only=True,
                    source_camera_selection=source["selection"],
                    selection={"selected": False, "training_fit_only": True, "historical_gates_not_evaluated": True})
    exact({k: v for k, v in payload.items() if k != "model"}, expected, arm + "/checkpoint")
    state, before = payload["model"], source["model"]
    require(set(state) == set(before), "Checkpoint tensor membership differs")
    require(all(isinstance(v, torch.Tensor) and v.device.type == "cpu" and v.shape == before[k].shape
                and v.dtype == before[k].dtype and torch.isfinite(v).all() for k, v in state.items()), "Invalid detector tensor")
    frozen = [k for k in state if k.startswith("reference_visible_head.")
              or k.endswith(("running_mean", "running_var", "num_batches_tracked"))]
    require(frozen and all(torch.equal(state[k], before[k]) for k in frozen), "Frozen head/BN state changed")
    changes = {p: sum(not torch.equal(v, before[k]) for k, v in state.items() if k.startswith(p))
               for p in ("network.encoder.", "network.decoder.", "network.segmentation_head.")}
    require(all(changes.values()), "Declared trainable component unchanged")
    return {"tensor_states": len(state), "frozen_tensors_verified": len(frozen), "changed_tensors": changes,
            "metadata_verified": True, "final_optimizer_tensors_returned": False}


def recreate_preview(cases, ids):
    canvas = Image.new("RGB", (1152, 42 + 219 * len(ids)), "#16181c"); draw = ImageDraw.Draw(canvas)
    for col, text in enumerate(("input", "supervised core", "source42", *ARMS)):
        draw.text((col * 192 + 3, 7), text, fill="white")
    draw.text((3, 24), "TRAIN-COHORT MASKS ONLY: green TP, red FP, yellow FN, purple unsupervised. No generated faces.", fill="white")
    for row, i in enumerate(ids):
        case = cases[i]
        with Image.open(ROOT / case["input"]) as image:
            rgb = np.asarray(image).copy()
        truth, valid = (binary(ROOT / case["target"][k]["path"]) for k in ("mask", "valid"))
        def overlay(pred):
            colored = rgb.copy(); marked = truth | pred | ~valid
            colored[~valid] = (165, 85, 215); colored[truth & pred & valid] = (16, 185, 129)
            colored[~truth & pred & valid] = (245, 50, 50); colored[truth & ~pred & valid] = (255, 220, 40)
            result = rgb.copy(); result[marked] = (.55 * rgb[marked] + .45 * colored[marked]).round().astype(np.uint8)
            return result
        panels = [rgb, overlay(truth)]
        for folder in ("initial_masks", *(a + "/final_masks" for a in ARMS)):
            panels.append(overlay(binary(EXTRACT / RESULT / folder / "real" / f"{i:03d}.png")))
        y = 42 + row * 219
        for col, array in enumerate(panels):
            canvas.paste(Image.fromarray(array).resize((192, 192)), (col * 192, y))
            draw.text((col * 192 + 3, y + 195), f"real{case['real_train_index']:03d}/{case['condition']}", fill="white")
    with Image.open(EXTRACT / RESULT / "preview.png") as returned:
        require(returned.mode == "RGB" and np.array_equal(np.asarray(returned), np.asarray(canvas)), "Preview pixels differ")
    return canvas


def main():
    import torch
    torch.set_num_threads(4)
    digest, hashes = extract_return()
    require(sha(ROOT / "outputs/real-camera-vm-bundle.tar.gz") == PACKAGE_SHA, "Sent package changed")
    require(hashes["inputs/real_camera_protocol.json"] == sha(ROOT / "outputs/real_camera_protocol_v1/protocol.json") == PROTOCOL_SHA,
            "Returned protocol differs")
    require(hashes["bundle_inventory.json"] == sha(ROOT / "outputs/real_camera_protocol_v1/inventory.json") == INVENTORY_SHA,
            "Returned input inventory differs")
    p = read(EXTRACT / "inputs/real_camera_protocol.json")
    for name, pin in p["code_sha256"].items():
        require(sha(safe(ROOT, name)) == pin, "Frozen runner dependency changed")
    for name, pin in p["local_audited_inputs"].items():
        require(sha(safe(ROOT, name)) == pin, "Frozen audited input changed")
    for role, name in ASSET_PATHS.items():
        require(sha(ROOT / name) == p["existing_assets"][role]["sha256"], "Source/fixture asset changed")
    pairs = read(ROOT / "outputs/real_camera_pairs_v1/manifest.json"); cases = pairs["cases"]
    registry = read(ROOT / "dataset/detector_supported_review_v2/manifest.json")
    real = [r for r in registry["supported_records"] if r["split"] == "train"]
    require(len(cases) == 182 and len(real) == 91 and pairs["held_out_inputs_in_cache"] == 0, "Source/case budget differs")
    fixtures = read(ROOT / ASSET_PATHS["fixture_manifest"])["cases"]
    schedule = independent_schedules(real, fixtures); exact(p["schedules"], schedule, "protocol schedules")
    cache = torch.load(ROOT / ASSET_PATHS["fixture_pixels"], map_location="cpu", weights_only=True)
    require(cache["format"] == "dgp-reflection-coverage-pixels-v1" and set(cache["pixels"]) == set(range(280)), "Fixture cache differs")
    targets = {}
    for i, case in enumerate(cases):
        require(case["case_id"] == i and case["real_train_index"] == i // 2 and case["split"] == "train"
                and case["condition"] == ("degraded" if i % 2 else "native"), "Camera case ordering/membership differs")
        require(sha(ROOT / case["input"]) == case["input_sha256"], "Camera RGB hash differs")
        for field in case["target"].values():
            require(sha(ROOT / field["path"]) == field["sha256"], "Real target/support hash differs")
        target, valid = (binary(ROOT / case["target"][k]["path"]) for k in ("mask", "valid"))
        targets[f"real/{i:03d}"] = (target, valid, ("old" if i < 166 else "new") + "_" + case["condition"])
    for i, item in cache["pixels"].items():
        require(set(item) == {"input", "mask", "valid", "geometry"}, "Fixture tensor keys differ")
        for name, tensor in item.items():
            require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu" and tensor.dtype == torch.uint8
                    and tensor.shape == ((3 if name == "input" else 1), 256, 256), "Fixture tensor format differs")
            require(hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest() == fixtures[i]["pixel_sha256"][name],
                    "Fixture pixels differ")
            if name != "input":
                require(((tensor == 0) | (tensor == 1)).all(), "Nonbinary fixture map")
        targets[f"fixture/{i:03d}"] = (item["mask"][0].numpy().astype(bool), item["valid"][0].numpy().astype(bool), "reflection")
    def measure(folder, logged):
        groups, rows, mask_hashes, errors = defaultdict(list), [], {}, []
        for name, (target, valid, group) in targets.items():
            path = EXTRACT / RESULT / folder / (name + ".png")
            counted = recount(binary(path), target, valid)
            groups[group].append(counted); rows.append({"id": name, "group": group, "counts": counted})
            mask_hashes[name + ".png"] = sha(path)
            if counted["negative_false_positive_cases"] or counted["empty_mask_cases"]:
                errors.append({"id": name, "group": group, **counted})
        calculated = {"groups": {k: aggregate(v) for k, v in groups.items()}, "rows": rows,
                      "mask_sha256": mask_hashes, "forward_images": 462, "held_out_forward_images": 0,
                      "scope": "Training-cohort diagnosis only; new sources are trained by camera91 and are not a holdout"}
        exact(logged, calculated, folder + "/metrics")
        return calculated, errors
    baseline, base_errors = measure("initial_masks", read(EXTRACT / RESULT / "baseline.json"))
    source = torch.load(ROOT / MODEL_PATH, map_location="cpu", weights_only=True)
    final, errors, states, logs = {}, {"source42": base_errors}, {}, {}
    for arm in ARMS:
        final[arm], errors[arm] = measure(arm + "/final_masks", read(EXTRACT / RESULT / arm / "metrics.json"))
        records = [json.loads(line) for line in (EXTRACT / RESULT / arm / "steps.jsonl").read_text().splitlines()]
        logs[arm] = audit_steps(records, schedule[arm], arm)
        payload = torch.load(EXTRACT / RESULT / arm / "last.pth", map_location="cpu", weights_only=True)
        states[arm] = checkpoint_audit(payload, source, p, arm)
        del payload
    decision = fit_checks(baseline, final)
    exact(read(EXTRACT / RESULT / "fit_decision.json"), decision, "fit decision")
    complete = read(EXTRACT / RESULT / "complete.json")
    fixed = {"complete": True, "protocol_sha256": PROTOCOL_SHA, "updates_per_arm": 112,
             "total_optimizer_updates": 336, "moment_steps_per_arm": 784, "cumulative_model_updates_per_arm": 994,
             "actual_model_forward_images": 4536, "held_out_forward_images": 0,
             "fit_decision": decision, "original_425_case_gates_evaluated": False, "promoted": False,
             "selected_checkpoint": None, "preview_sha256": hashes[RESULT + "preview.png"], "original_assets_unchanged": True}
    require(set(complete) == set(fixed) | {"seconds", "checkpoints"}, "Completion fields differ")
    for key, value in fixed.items():
        exact(complete[key], value, "complete/" + key)
    require(type(complete["seconds"]) is float and 0 < complete["seconds"] < 1800, "Execution time outside budget")
    require(set(complete["checkpoints"]) == set(ARMS), "Checkpoint registry differs")
    for arm, binding in complete["checkpoints"].items():
        exact(binding["model_sha256"], hashes[RESULT + arm + "/last.pth"], "checkpoint file digest")
        require(set(binding) == {"model_sha256", "optimizer_sha256", "optimizer_retained_on_vm_not_in_return_archive"}
                and re.fullmatch("[0-9a-f]{64}", binding["optimizer_sha256"])
                and binding["optimizer_retained_on_vm_not_in_return_archive"] is True, "Optimizer retention registry differs")
    preflight, execution = (read(EXTRACT / RESULT / name) for name in ("preflight.json", "execution.json"))
    moment = {"parameter_states": 92, "parameter_elements": 14328209, "restored_step": 672, "cumulative_model_updates": 882}
    expected_preflight = {"complete": True, "batch_size": 1, "protocol_sha256": PROTOCOL_SHA,
        "gpu": "NVIDIA L4", "cuda_available": True, "moment_check": moment, "optimizer_constructed": False,
        "optimizer_updates": 0, "actual_model_forward_images": 1, "held_out_forward_images": 0, "training_quality_verified": False}
    require(set(preflight) == set(expected_preflight) | {"loss", "dependencies"}, "Preflight fields differ")
    for key, value in expected_preflight.items():
        exact(preflight[key], value, "preflight/" + key)
    require(type(preflight["loss"]) is float and math.isfinite(preflight["loss"]) and preflight["loss"] >= 0, "Nonfinite preflight")
    dependencies = {"segmentation-models-pytorch": "0.5.0", "timm": "1.0.15", "huggingface-hub": "0.29.3", "safetensors": "0.5.3", "PyYAML": "6.0.2"}
    require(set(preflight["dependencies"]) == set(dependencies), "Runtime dependency membership differs")
    for distribution, version in dependencies.items():
        row = preflight["dependencies"][distribution]
        require(set(row) == {"version", "module_path"} and row["version"] == version
                and "/coverage_vm_bundle/outputs/face_occlusion_dependencies/" in row["module_path"], "Runtime dependency differs")
    exact(execution, {"protocol_sha256": PROTOCOL_SHA, "inventory_sha256": INVENTORY_SHA,
        "gpu": "NVIDIA L4", "torch": "2.9.1+cu129", "moment_check": moment, "arms": list(ARMS),
        "updates_per_arm": 112, "total_update_budget": 336, "held_out_forward_images": 0,
        "source_gate_failure_unchanged": True, "promoted": False}, "execution")
    grid = recreate_preview(cases, p["preview_case_ids"])
    OUT.mkdir(); grid.save(OUT / "verified_preview.png")
    record = {"format": "dgp-real-camera-return-independent-verification-v1", "date": "2026-10-03", "complete": True,
        "archive_sha256": digest, "archive_bytes": ARCHIVE.stat().st_size, "members_verified": len(hashes),
        "auditor_sha256": sha(__file__), "protocol_sha256": PROTOCOL_SHA, "saved_masks_recounted": 1848,
        "logged_steps_verified": 336, "local_optimizer_constructed": False, "local_optimizer_updates": 0,
        "local_model_forwards": 0, "checkpoint_checks": states, "step_checks": logs,
        "baseline": baseline["groups"], "final": {a: final[a]["groups"] for a in ARMS},
        "empty_or_clear_error_cases": errors, "fit_decision": decision, "seconds": complete["seconds"],
        "checkpoint_sha256": {a: complete["checkpoints"][a]["model_sha256"] for a in ARMS},
        "preview_pixels_reproduced_exactly": True, "visual_review_pending": True, "promoted": False,
        "limitations": ["Final optimizer tensors remain on VM; final moment steps are verified log claims, not locally inspected moments",
            "Checkpoint-to-prediction CPU reproduction pending; recount verifies saved masks, not model execution",
            "Only exposed training-cohort fit; original425 gates and external practical transfer not evaluated",
            "Approximate assistant labels; one related eight-source cohort; no hidden facial ground truth",
            "Remote asset preservation/forward counters are code/log claims; no independent live VM observation"]}
    write(OUT / "members.json", hashes); write(OUT / "verification.json", record)
    print(json.dumps({"complete": True, "members": len(hashes), "masks": 1848, "steps": 336,
                      "fit_checks": decision["checks"], "verification_sha256": sha(OUT / "verification.json")}))


if __name__ == "__main__":
    main()
