"""Independently verify RGB/grayscale VM returns; no detector construction or training."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import tarfile

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_real_camera_results import require, sha, read, write, safe, exact, binary, recount, aggregate

ARCHIVE = ROOT / "outputs/gray-covering-results.tar.gz"
EXTRACT = ROOT / "outputs/downloaded_gray_covering_v1"
OUT = ROOT / "outputs/gray_covering_results_validation_v1"
PREFIX = "gray_covering_vm_results/"
RESULT = "outputs/gray_covering_vm/"
ARMS = ("rgb133", "gray133")
FAMILIES = ("hand", "hair", "cloth", "object", "eyewear_mask")
MAPPING = {"hand": "hand", "obstructing_hair": "hair", "cloth_or_scarf": "cloth",
           "other_cloth_or_object": "cloth", "object": "object", "sunglasses": "eyewear_mask",
           "costume_mask": "eyewear_mask", "face_mask_preserve_clear_goggles": "eyewear_mask"}
PROTOCOL_SHA = "c29eabcc23b07e2f9353116068c623b35258a56ed5ab259126b13b9c781be47a"
INVENTORY_SHA = "652c473fb15c6af622d23dc99cf123f821e6eb3181ce2a43a12ef1f8c931e420"
PACKAGE_SHA = "212b91935e548ad1423f92f71014681be25abd052bad1d7bf2785e8b2b702d18"
PACKAGE_AUDIT_SHA = "4a341c260c4ed39df3af562136c4da41fd2bdbbfa58bd6dbda05048b86288899"
SHELL_AUDIT_SHA = "ec5056207ea8eeda196846228b73c0d9a7750effa2f0a0d6135b110bb9ff298d"
ASSET_PATHS = {
    "model": "outputs/downloaded_real_camera_v1/outputs/real_camera_vm/camera91/last.pth",
    "fixture_manifest": "outputs/reflection_coverage_data_v1/manifest.json",
    "fixture_pixels": "outputs/reflection_coverage_data_v1/pixels.pth",
    "rgb128_reference": "outputs/downloaded_varied_covering_v1/outputs/varied_covering_vm/varied133/last.pth",
}
PREVIOUS = ROOT / "outputs/downloaded_varied_covering_v1/outputs/varied_covering_vm"
STATES = ("initial_masks", "rgb133/rgb128_masks", "rgb133/final_masks", "gray133/final_masks")


def expected_files():
    names = {"bundle_inventory.json", "inputs/gray_covering_protocol.json"}
    names.update(RESULT + n for n in ("baseline.json", "complete.json", "execution.json", "fit_decision.json",
                                     "preflight.json", "return_inventory.json", "preview.png", "preview_grayscale.png"))
    for arm in ARMS:
        names.update(RESULT + arm + "/" + n for n in ("last.pth", "metrics.json", "steps.jsonl"))
    names.update(RESULT + "rgb133/" + n for n in ("epoch_2.pth", "rgb128_reproduction.json", "rgb128_metrics.json"))
    for folder in STATES:
        for domain, size in (("real", 266), ("fixture", 280), ("gray", 266)):
            names.update(RESULT + folder + f"/{domain}/{i:03d}.png" for i in range(size))
    require(len(names) == 3267, "Auditor return membership budget differs")
    return names


def validate_member(member, expected, folded, extraction_root=None):
    require(member.isfile() and member.name.startswith(PREFIX), "Unexpected archive type/root")
    name = member.name[len(PREFIX):]
    safe(EXTRACT if extraction_root is None else extraction_root, name)
    require(name in expected and name.casefold() not in folded, "Extra or duplicate archive member")
    limit = 64 * 1024**2 if name.endswith(".pth") else 2 * 1024**2
    require(0 <= member.size <= limit, "Returned member exceeds fixed size budget")
    return name


def extract_return():
    require(ARCHIVE.is_file() and ARCHIVE.with_name(ARCHIVE.name + ".sha256").is_file(),
            "Download gray-covering-results.tar.gz and its .sha256 into outputs first")
    require(not EXTRACT.exists() and not OUT.exists(), "Preserve earlier extraction/audit evidence")
    digest = sha(ARCHIVE)
    require(ARCHIVE.with_name(ARCHIVE.name + ".sha256").read_bytes() ==
            (digest + "  " + ARCHIVE.name + "\n").encode("ascii"), "Transfer/LF checksum differs")
    expected = expected_files()
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        members, folded, total = {}, set(), 0
        for member in tar:
            name = validate_member(member, expected, folded)
            folded.add(name.casefold())
            members[name] = member
            total += member.size
            require(total <= 256 * 1024**2, "Return total size budget exceeded")
        require(set(members) == expected, "Return membership differs")
        inventory = json.load(tar.extractfile(members[RESULT + "return_inventory.json"]))
        require(type(inventory) is dict and set(inventory) == expected - {RESULT + "return_inventory.json"},
                "Return inventory coverage differs")
        require(all(type(v) is str and re.fullmatch("[0-9a-f]{64}", v) for v in inventory.values()),
                "Invalid returned digest")
        # Validate every byte before creating any extraction or quality evidence.
        hashes = {}
        for name, member in members.items():
            with tar.extractfile(member) as stream:
                hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
            if name in inventory:
                require(hashes[name] == inventory[name], "Returned inventory hash differs: " + name)
        require(hashes["inputs/gray_covering_protocol.json"] == PROTOCOL_SHA
                and hashes["bundle_inventory.json"] == INVENTORY_SHA, "Returned protocol/inventory differs")
        EXTRACT.mkdir()
        for name, member in members.items():
            path = safe(EXTRACT, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as stream, path.open("xb") as destination:
                for chunk in iter(lambda: stream.read(1024**2), b""):
                    destination.write(chunk)
            require(sha(path) == hashes[name], "Extracted bytes differ: " + name)
    return digest, hashes


def independent_schedules(rows, fixtures):
    require(len(rows) == 133 and all(r["split"] == "train" for r in rows), "Training membership differs")
    old_positive = [i for i, r in enumerate(rows[:91]) if r["kind"] == "covered"]
    old_clear = [i for i, r in enumerate(rows[:91]) if r["kind"] == "uncovered"]
    new_clear = [i for i in range(91, 133) if rows[i]["kind"] == "uncovered"]
    families = {name: [i for i in range(91, 133) if rows[i]["kind"] == "covered"
                      and MAPPING[rows[i]["occlusion_stratum"]] == name] for name in FAMILIES}
    require([len(old_positive), len(old_clear), len(new_clear)] == [58, 33, 11]
            and [len(families[n]) for n in FAMILIES] == [9, 7, 6, 4, 5], "Training strata differ")
    require(all(r["data_origin"] == "cofw_supported_extension" and r["publisher_split"] == "train"
                for r in rows[91:]), "Added sources outside author training")
    require(len(fixtures) == 280 and [r["case_id"] for r in fixtures] == list(range(280)), "Fixture membership differs")
    positive = [r["case_id"] for r in fixtures if r["style"] != "clear"]
    clear = [r["case_id"] for r in fixtures if r["style"] == "clear"]
    require([len(positive), len(clear)] == [224, 56], "Replay strata differ")
    rng = random.Random(42)
    for pool in (old_positive, old_clear, new_clear, positive, clear, *(families[n] for n in FAMILIES)):
        rng.shuffle(pool)
    visits = [Counter() for _ in range(4)]
    schedules = {arm: [] for arm in ARMS}
    for epoch in range(8):
        batches = {arm: [] for arm in ARMS}
        for step in range(64):
            number = epoch * 64 + step
            name = FAMILIES[number % 5]
            indices = [old_positive[number % 58], families[name][(number // 5) % len(families[name])],
                       old_clear[number % 33], new_clear[number % 11]]
            replay = [positive[(2 * number) % 224], positive[(2 * number + 1) % 224],
                      clear[(2 * number) % 56], clear[(2 * number + 1) % 56]]
            views = []
            for slot, index in enumerate(indices):
                visit = visits[slot][index]
                visits[slot][index] += 1
                views.append({"index": index, "degraded": bool(visit % 2), "grayscale": bool((visit // 2) % 2)})
            batches["gray133"].append({"real": views, "fixture": replay})
            batches["rgb133"].append({"real": [{**v, "grayscale": False} for v in views], "fixture": list(replay)})
        for arm in ARMS:
            schedules[arm].append(batches[arm])
    return schedules


def audit_steps(records, schedule, arm):
    require(arm in ARMS and len(records) == 512 and len(schedule) == 8 and all(len(e) == 64 for e in schedule),
            "Incomplete/extra step log or schedule")
    for number, row in enumerate(records):
        fixed = {"arm": arm, "experiment_epoch": number // 64 + 1, "step": number % 64 + 1,
                 "experiment_updates": number + 1, "optimizer_state_step": number + 1,
                 "cumulative_model_updates": 995 + number, **schedule[number // 64][number % 64]}
        require(set(row) == set(fixed) | {"loss", "real_loss", "fixture_loss", "pre_clip_norm"}, "Step fields differ")
        for key, value in fixed.items():
            exact(row[key], value, arm + "/step/" + key)
        require(all(type(row[k]) is float and math.isfinite(row[k]) and row[k] >= 0
                    for k in ("loss", "real_loss", "fixture_loss", "pre_clip_norm")), "Invalid logged loss/gradient")
        require(math.isclose(row["loss"], .5 * (row["real_loss"] + row["fixture_loss"]), rel_tol=1e-6, abs_tol=1e-7),
                "Weighted logged loss differs from fp32 calculation")
    return {"logged_updates": 512, "first_fresh_moment_step": 1, "last_fresh_moment_step": 512,
            "cumulative_model_updates": 1506, "finite_losses_and_gradients": True, "frozen_schedule_verified": True}


def fit_checks(final):
    control, treatment = (final[arm]["groups"] for arm in ARMS)

    def retained(a, b):
        return a["iou"] >= b["iou"] and all(a[k] <= b[k] for k in
                   ("missed_fraction", "visible_false_positive", "empty_mask_cases", "negative_false_positive_cases"))

    rgb = ("old_native", "old_degraded", "reflection") + tuple(
        f"cofw_{name}_{condition}" for name in FAMILIES for condition in ("native", "degraded"))
    checks = {
        "grayscale_hair_native_and_degraded_iou_gain": all(treatment[f"cofw_hair_{c}_grayscale"]["iou"] >
             control[f"cofw_hair_{c}_grayscale"]["iou"] for c in ("native", "degraded")),
        "other_new_grayscale_family_iou_retained": all(treatment[f"cofw_{n}_{c}_grayscale"]["iou"] >=
             control[f"cofw_{n}_{c}_grayscale"]["iou"] for n in FAMILIES if n != "hair" for c in ("native", "degraded")),
        "rgb_cohorts_and_reflection_fit_retained": all(retained(treatment[g], control[g]) for g in rgb),
        "all_final_supervised_clear_controls_empty": all(g["negative_false_positive_cases"] == 0
             for value in final.values() for g in value["groups"].values()),
        "new_covered_predictions_nonempty_all_conditions": all(treatment[f"cofw_{n}_{c}{s}"]["empty_mask_cases"] == 0
             for n in FAMILIES for c in ("native", "degraded") for s in ("", "_grayscale")),
    }
    return {"checks": checks, "all_training_fit_checks_pass": all(checks.values()), "training_fit_only": True,
            "original_425_case_gates_evaluated": False, "historical_gate_failure_unchanged": True,
            "selected_checkpoint": None, "promoted": False,
            "next": "Independently audit the return and evaluate unchanged practical sources/guard and face outputs before model selection"}


def frozen_inputs():
    pins = {"outputs/gray-covering-vm-bundle.tar.gz": PACKAGE_SHA,
            "outputs/gray_covering_protocol_v1/protocol.json": PROTOCOL_SHA,
            "outputs/gray_covering_protocol_v1/inventory.json": INVENTORY_SHA,
            "outputs/gray_covering_package_validation_v1/verification.json": PACKAGE_AUDIT_SHA,
            "outputs/gray_covering_package_validation_v1/shell_verification.json": SHELL_AUDIT_SHA}
    for name, pin in pins.items():
        require(sha(safe(ROOT, name)) == pin, "Prepared package evidence changed: " + name)
    package = ROOT / "outputs/gray-covering-vm-bundle.tar.gz"
    require(package.with_name(package.name + ".sha256").read_bytes() ==
            (PACKAGE_SHA + "  " + package.name + "\n").encode("ascii"), "Prepared package LF checksum differs")
    protocol = read(ROOT / "outputs/gray_covering_protocol_v1/protocol.json")
    require(protocol["format"] == "dgp-gray-covering-pilot-v1" and protocol["actual_cuda_execution_verified"] is False,
            "Preparation protocol cannot claim execution")
    for name, pin in {**protocol["code_sha256"], **protocol["local_audited_inputs"]}.items():
        require(sha(safe(ROOT, name)) == pin, "Frozen code/evidence changed: " + name)
    for role, name in ASSET_PATHS.items():
        require(sha(safe(ROOT, name)) == protocol["existing_assets"][role]["sha256"], "Source/fixture asset changed: " + role)
    registry = read(ROOT / "dataset/detector_supported_review_v3/manifest.json")
    rows = [row for row in registry["supported_records"] if row["split"] == "train"]
    require(len(registry["supported_records"]) == 165 and len(rows) == 133, "Registry scope differs")
    parent = read(ROOT / "dataset/detector_supported_review_v2/manifest.json")
    exact(registry["supported_records"][:123], parent["supported_records"], "Parent registry")
    for row in registry["supported_records"]:
        require(sha(safe(ROOT, row["source"].replace("\\", "/"))) == row["source_sha256"], "Raw source changed")
    split = read(ROOT / "outputs/downloaded_phase4/outputs/phase4_with_progress/split.json")
    training = {p.replace("\\", "/") for p in split["train"]}
    validation = {p.replace("\\", "/") for p in split["validation"]}
    require(not training.intersection(validation), "Phase4 split overlap")
    fixtures = read(ROOT / ASSET_PATHS["fixture_manifest"])["cases"]
    require(all(r["source"].replace("\\", "/") in training for r in rows[73:83] + fixtures),
            "Original native/replay source outside Phase4 training")
    schedule = independent_schedules(rows, fixtures)
    exact(protocol["schedules"], schedule, "Independent schedule")
    previous = read(ROOT / "outputs/varied_covering_protocol_v2/protocol.json")
    first128 = [{"real": [{k: r[k] for k in ("index", "degraded")} for r in b["real"]], "fixture": b["fixture"]}
                for epoch in schedule["rgb133"][:2] for b in epoch]
    exact(first128, [b for epoch in previous["schedules"]["varied133"] for b in epoch], "Original RGB128 schedule")
    return protocol, rows, fixtures


def load_targets(rows, fixtures):
    import torch
    pairs = read(ROOT / "outputs/cofw_camera_pairs_v2/manifest.json")
    require(len(pairs["cases"]) == 266 and pairs["held_out_inputs_in_cache"] == 0, "Paired cache scope differs")
    cases, targets, gray = pairs["cases"], {}, {}
    for case_id, case in enumerate(cases):
        index, condition = case_id // 2, "degraded" if case_id % 2 else "native"
        require(case["case_id"] == case_id and case["real_train_index"] == index and case["split"] == "train"
                and case["condition"] == condition and case["source_sha256"] == rows[index]["source_sha256"]
                and case["group"] == rows[index]["group"] and case["kind"] == rows[index]["kind"],
                "Paired case ordering/provenance differs")
        require(sha(safe(ROOT, case["input"])) == case["input_sha256"], "Paired RGB changed")
        with Image.open(safe(ROOT, case["input"])) as image:
            require(image.format == "PNG" and image.mode == "RGB" and image.size == (256, 256), "Invalid cached RGB")
        for asset in case["target"].values():
            require(sha(safe(ROOT, asset["path"])) == asset["sha256"], "Target/support changed")
        target, valid = (binary(safe(ROOT, case["target"][k]["path"])) for k in ("mask", "valid"))
        require(valid.any() and not (target & ~valid).any(), "Target outside supervised support")
        if case_id % 2:
            require(np.array_equal(target, targets[f"real/{case_id-1:03d}"][0])
                    and np.array_equal(valid, targets[f"real/{case_id-1:03d}"][1]), "Camera condition changed supervision")
        prefix = "old" if index < 91 else ("cofw_clear" if rows[index]["kind"] == "uncovered"
                                           else "cofw_" + MAPPING[rows[index]["occlusion_stratum"]])
        targets[f"real/{case_id:03d}"] = (target, valid, prefix + "_" + condition)
        gray[f"gray/{case_id:03d}"] = (target, valid, prefix + "_" + condition + "_grayscale")
    cache = torch.load(ROOT / ASSET_PATHS["fixture_pixels"], map_location="cpu", weights_only=True)
    require(cache["format"] == "dgp-reflection-coverage-pixels-v1" and set(cache["pixels"]) == set(range(280)),
            "Fixture cache membership differs")
    for index in range(280):
        item = cache["pixels"][index]
        require(set(item) == {"input", "mask", "valid", "geometry"}, "Fixture tensor fields differ")
        for name, tensor in item.items():
            require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu" and tensor.dtype == torch.uint8
                    and tensor.shape == ((3 if name == "input" else 1), 256, 256), "Fixture tensor format differs")
            require(hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest() == fixtures[index]["pixel_sha256"][name],
                    "Fixture pixels changed")
            if name != "input":
                require(((tensor == 0) | (tensor == 1)).all(), "Nonbinary fixture map")
        target, valid = (item[k][0].numpy().astype(bool) for k in ("mask", "valid"))
        require(valid.any() and not (target & ~valid).any(), "Fixture target outside supervised support")
        targets[f"fixture/{index:03d}"] = (target, valid, "reflection")
    targets.update(gray)  # Match returned real, fixture, grayscale record order.
    require(len(targets) == 812, "Independent measurement target budget differs")
    return cases, targets


def audit_measurement(folder, logged, targets):
    require(len(targets) == 812, "Measurement target budget differs")
    groups, records, hashes, errors = defaultdict(list), [], {}, []
    for name, (target, valid, group) in targets.items():
        path = safe(EXTRACT, RESULT + folder + "/" + name + ".png")
        counted = recount(binary(path), target, valid)
        groups[group].append(counted)
        records.append({"id": name, "group": group, "counts": counted})
        hashes[name + ".png"] = sha(path)
        if counted["negative_false_positive_cases"] or counted["empty_mask_cases"]:
            errors.append({"id": name, "group": group, **counted})
    calculated = {"groups": {k: aggregate(v) for k, v in groups.items()}, "rows": records,
                  "mask_sha256": hashes, "forward_images": 812, "held_out_forward_images": 0,
                  "scope": "Exposed training-cohort RGB/grayscale diagnosis only; no hidden facial targets"}
    exact(logged, calculated, folder + "/metrics")
    return calculated, errors


def checkpoint_audit(payload, source, protocol, arm, intermediate=False):
    import torch
    require(arm in ARMS and (not intermediate or arm == "rgb133"), "Invalid checkpoint role")
    expected = {k: v for k, v in source.items() if k != "model"}
    expected.update(epoch=46 if intermediate else 52, additional_epoch=36 if intermediate else 42,
                    optimizer_updates=1122 if intermediate else 1506, fresh_optimizer_updates=128 if intermediate else 512,
                    experiment_updates=128 if intermediate else 512, optimizer_reset=True, detector_only=True,
                    gray_covering_arm=arm, gray_covering_protocol_sha256=PROTOCOL_SHA,
                    selection={"selected": False, "training_fit_only": True})
    if not intermediate:
        expected.update(source_varied_checkpoint_counters=protocol["source_counters"], source_varied_selection=source["selection"])
        expected["selection"]["historical_gates_not_evaluated"] = True
    exact({k: v for k, v in payload.items() if k != "model"}, expected, arm + "/checkpoint metadata")
    state, before = payload["model"], source["model"]
    require(len(state) == 184 and set(state) == set(before), "Checkpoint tensor membership differs")
    require(all(isinstance(v, torch.Tensor) and v.device.type == "cpu" and v.shape == before[k].shape
                and v.dtype == before[k].dtype and torch.isfinite(v).all() for k, v in state.items()), "Invalid detector state tensor")
    frozen = [k for k in state if k.startswith("reference_visible_head.")
              or k.endswith(("running_mean", "running_var", "num_batches_tracked"))]
    require(len(frozen) == 92 and all(torch.equal(state[k], before[k]) for k in frozen), "Frozen head/BN tensors changed")
    changed = {prefix: sum(not torch.equal(v, before[k]) for k, v in state.items() if k.startswith(prefix))
               for prefix in ("network.encoder.", "network.decoder.", "network.segmentation_head.")}
    require(all(changed.values()), "Declared trainable component unchanged")
    return {"tensor_states": len(state), "frozen_tensors_verified": len(frozen), "changed_tensors": changed,
            "metadata_verified": True, "final_optimizer_tensors_returned": False}


def audit_reproduction(payload, reference, logged):
    import torch
    require(reference["varied_covering_arm"] == "varied133" and reference["optimizer_updates"] == 1122
            and reference["fresh_optimizer_updates"] == 128, "Prior RGB128 checkpoint role differs")
    actual, before = payload["model"], reference["model"]
    require(len(actual) == 184 and set(actual) == set(before), "RGB128 reproduction tensor membership differs")
    require(all(v.dtype == before[k].dtype and v.shape == before[k].shape and torch.equal(v, before[k])
                for k, v in actual.items()), "RGB128 checkpoint failed exact prior-run reproduction")
    exact(logged, {"complete": True, "reference_model_sha256": sha(ROOT / ASSET_PATHS["rgb128_reference"]),
                   "tensors_compared": 184, "different_tensors": [], "fresh_updates": 128,
                   "cumulative_model_updates": 1122, "all_model_states_equal": True, "optimizer_reset": True},
          "RGB128 reproduction receipt")
    return {"all_model_states_equal": True, "tensors_compared": 184, "fresh_updates": 128}


def preview_input(case, grayscale=False):
    import torch
    with Image.open(safe(ROOT, case["input"])) as image:
        rgb = np.asarray(image).copy()
    x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
    if grayscale:
        # Reproduce the declared float32 arithmetic without importing the trainer.
        weights = torch.tensor([.299, .587, .114], dtype=torch.float32)[:, None, None]
        x = (x * weights).sum(0, keepdim=True).expand_as(x).contiguous()
    return np.rint(x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def recreate_preview(cases, ids, grayscale=False):
    canvas = Image.new("RGB", (960, 42 + 218 * len(ids)), "#16181c")
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(("input", "target core", "camera91 source", *ARMS)):
        draw.text((column * 192 + 3, 7), title, fill="white")
    draw.text((3, 24), "TRAIN-COHORT MASKS: green TP, red FP, yellow FN, purple ignored; no generated faces.", fill="white")
    for row, case_id in enumerate(ids):
        case = cases[case_id]
        rgb = preview_input(case, grayscale)
        truth, support = (binary(safe(ROOT, case["target"][k]["path"])) for k in ("mask", "valid"))

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
            path = EXTRACT / RESULT / folder / ("gray" if grayscale else "real") / f"{case_id:03d}.png"
            panels.append(overlay(binary(path)))
        y = 42 + row * 218
        for column, panel in enumerate(panels):
            canvas.paste(Image.fromarray(panel).resize((192, 192)), (column * 192, y))
        draw.text((3, y + 195), f"{case['source_id']}/{case['condition']}", fill="white")
    name = "preview_grayscale.png" if grayscale else "preview.png"
    with Image.open(EXTRACT / RESULT / name) as returned:
        require(returned.mode == "RGB" and returned.format == "PNG"
                and np.array_equal(np.asarray(returned), np.asarray(canvas)), "Returned preview pixels differ")
    return canvas


def audit_execution(preflight, execution):
    fixed = {"complete": True, "batch_size": 1, "protocol_sha256": PROTOCOL_SHA, "gpu": "NVIDIA L4",
             "cuda_available": True, "fresh_optimizer_planned": True, "optimizer_constructed": False,
             "optimizer_updates": 0, "actual_model_forward_images": 1, "held_out_forward_images": 0, "grayscale_input": True}
    require(set(preflight) == set(fixed) | {"loss", "dependencies"}, "Preflight fields differ")
    for key, value in fixed.items():
        exact(preflight[key], value, "preflight/" + key)
    require(type(preflight["loss"]) is float and math.isfinite(preflight["loss"]) and preflight["loss"] >= 0,
            "Invalid one-batch preflight loss")
    versions = {"segmentation-models-pytorch": "0.5.0", "timm": "1.0.15", "huggingface-hub": "0.29.3",
                "safetensors": "0.5.3", "PyYAML": "6.0.2"}
    require(set(preflight["dependencies"]) == set(versions), "Pinned dependency membership differs")
    for name, version in versions.items():
        row = preflight["dependencies"][name]
        require(set(row) == {"version", "module_path"} and row["version"] == version
                and type(row["module_path"]) is str
                and row["module_path"].startswith("/home/janusdominic0/forensic-dgp/coverage_vm_bundle/outputs/face_occlusion_dependencies/"),
                "Pinned runtime dependency differs")
    exact(execution, {"protocol_sha256": PROTOCOL_SHA, "inventory_sha256": INVENTORY_SHA,
                      "gpu": "NVIDIA L4", "torch": "2.9.1+cu129", "arms": list(ARMS), "updates_per_arm": 512,
                      "total_update_budget": 1024, "optimizer_reset": True, "source_model_updates": 994,
                      "fresh_moment_start_step": 0, "held_out_forward_images": 0, "promoted": False}, "execution")


def audit_completion(complete, decision, hashes):
    fixed = {"complete": True, "protocol_sha256": PROTOCOL_SHA, "updates_per_arm": 512,
             "total_optimizer_updates": 1024, "moment_steps_per_arm": 512, "cumulative_model_updates_per_arm": 1506,
             "optimizer_reset": True, "epochs_per_arm": 8, "actual_model_forward_images": 11440,
             "held_out_forward_images": 0, "fit_decision": decision, "original_425_case_gates_evaluated": False,
             "promoted": False, "selected_checkpoint": None, "preview_sha256": hashes[RESULT + "preview.png"],
             "preview_grayscale_sha256": hashes[RESULT + "preview_grayscale.png"],
             "rgb128_reproduction_equal": True, "original_assets_unchanged": True}
    require(set(complete) == set(fixed) | {"seconds", "checkpoints"}, "Completion fields differ")
    for key, value in fixed.items():
        exact(complete[key], value, "complete/" + key)
    require(type(complete["seconds"]) is float and 0 < complete["seconds"] < 1800, "Execution time outside finite cap")
    require(set(complete["checkpoints"]) == set(ARMS), "Final checkpoint registry differs")
    for arm, binding in complete["checkpoints"].items():
        require(set(binding) == {"model_sha256", "optimizer_sha256", "optimizer_retained_on_vm_not_in_return_archive"}
                and type(binding["optimizer_sha256"]) is str and re.fullmatch("[0-9a-f]{64}", binding["optimizer_sha256"])
                and binding["optimizer_retained_on_vm_not_in_return_archive"] is True, "Optimizer retention binding differs")
        exact(binding["model_sha256"], hashes[RESULT + arm + "/last.pth"], "Checkpoint file digest")


def main():
    # An absent/partial return cannot create a success or quality report.
    digest, hashes = extract_return()
    import torch
    torch.set_num_threads(4)
    protocol, rows, fixtures = frozen_inputs()
    cases, targets = load_targets(rows, fixtures)
    source = torch.load(ROOT / ASSET_PATHS["model"], map_location="cpu", weights_only=True)
    source_counters = {"epoch": 44, "additional_epoch": 34, "optimizer_updates": 994,
                       "fresh_optimizer_updates": 784, "experiment_updates": 112}
    for key, value in source_counters.items():
        exact(source[key], value, "Source camera91/" + key)
    baseline, baseline_errors = audit_measurement("initial_masks", read(EXTRACT / RESULT / "baseline.json"), targets)
    intermediate, intermediate_errors = audit_measurement("rgb133/rgb128_masks",
                                      read(EXTRACT / RESULT / "rgb133/rgb128_metrics.json"), targets)
    reference = torch.load(ROOT / ASSET_PATHS["rgb128_reference"], map_location="cpu", weights_only=True)
    intermediate_payload = torch.load(EXTRACT / RESULT / "rgb133/epoch_2.pth", map_location="cpu", weights_only=True)
    intermediate_state = checkpoint_audit(intermediate_payload, source, protocol, "rgb133", intermediate=True)
    reproduction = audit_reproduction(intermediate_payload, reference, read(EXTRACT / RESULT / "rgb133/rgb128_reproduction.json"))
    del reference, intermediate_payload
    # Identical source/inputs/runtime must also retain the previously returned RGB/fixture masks.
    previous_members = read(ROOT / "outputs/varied_covering_results_validation_v1/members.json")
    for new_folder, old_folder in (("initial_masks", "initial_masks"), ("rgb133/rgb128_masks", "varied133/final_masks")):
        for name in targets:
            if name.startswith("gray/"):
                continue
            old_path = PREVIOUS / old_folder / (name + ".png")
            require(sha(old_path) == previous_members["outputs/varied_covering_vm/" + old_folder + "/" + name + ".png"],
                    "Previously audited RGB mask changed")
            require(np.array_equal(binary(old_path), binary(EXTRACT / RESULT / new_folder / (name + ".png"))),
                    "Source/RGB128 saved-mask reproduction differs: " + name)
    final, states, steps = {}, {}, {}
    errors = {"camera91_source": baseline_errors, "rgb128": intermediate_errors}
    for arm in ARMS:
        final[arm], errors[arm] = audit_measurement(arm + "/final_masks", read(EXTRACT / RESULT / arm / "metrics.json"), targets)
        log = [json.loads(line) for line in (EXTRACT / RESULT / arm / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
        steps[arm] = audit_steps(log, protocol["schedules"][arm], arm)
        payload = torch.load(EXTRACT / RESULT / arm / "last.pth", map_location="cpu", weights_only=True)
        states[arm] = checkpoint_audit(payload, source, protocol, arm)
        del payload
    decision = fit_checks(final)
    exact(read(EXTRACT / RESULT / "fit_decision.json"), decision, "Fit decision")
    audit_execution(read(EXTRACT / RESULT / "preflight.json"), read(EXTRACT / RESULT / "execution.json"))
    complete = read(EXTRACT / RESULT / "complete.json")
    audit_completion(complete, decision, hashes)
    grids = [recreate_preview(cases, protocol["preview_case_ids"], g) for g in (False, True)]
    OUT.mkdir()
    for grid, name in zip(grids, ("verified_preview.png", "verified_preview_grayscale.png")):
        grid.save(OUT / name)
    report = {"format": "dgp-gray-covering-return-independent-verification-v1", "date": "2026-10-03", "complete": True,
              "archive_sha256": digest, "archive_bytes": ARCHIVE.stat().st_size, "members_verified": len(hashes),
              "auditor_sha256": sha(__file__), "protocol_sha256": PROTOCOL_SHA, "saved_masks_recounted": 3248,
              "logged_steps_verified": 1024, "local_optimizer_constructed": False, "local_optimizer_updates": 0,
              "local_model_forwards": 0, "checkpoint_checks": states, "step_checks": steps,
              "rgb128_state_check": intermediate_state, "rgb128_exact_reproduction": reproduction,
              "source_and_rgb128_prior_rgb_masks_matched": 1092,
              "baseline": baseline["groups"], "rgb128": intermediate["groups"],
              "final": {arm: final[arm]["groups"] for arm in ARMS}, "empty_or_clear_error_cases": errors,
              "fit_decision": decision, "seconds": complete["seconds"],
              "checkpoint_sha256": {arm: complete["checkpoints"][arm]["model_sha256"] for arm in ARMS},
              "both_preview_pixels_reproduced_exactly": True, "visual_review_pending": True, "promoted": False,
              "limitations": ["Final optimizer tensors remain on VM; moment counters are verified log claims rather than inspected moments",
                              "Saved-mask recount and checkpoint inspection do not independently reproduce final model execution",
                              "Only exposed training-cohort evidence; unchanged original425 gates and fixed practical/face output review remain required",
                              "Approximate detector labels; identity/pretraining overlap and true hidden facial appearance remain unknown",
                              "Remote preservation, timing and forward counts are frozen-code/log claims rather than live VM observations"]}
    write(OUT / "members.json", hashes)
    write(OUT / "verification.json", report)
    print(json.dumps({"complete": True, "members": len(hashes), "masks": 3248, "steps": 1024,
                      "fit_checks": decision["checks"], "verification_sha256": sha(OUT / "verification.json")}))


def check_contract():
    """Read-only readiness inspection. It must never emit a return success report."""
    protocol, rows, fixtures = frozen_inputs()
    import torch
    torch.set_num_threads(4)
    cases, targets = load_targets(rows, fixtures)
    print(json.dumps({"state": "awaiting_vm_return", "prepared_contract_verified": True,
                      "result_audit_completed": False, "protocol_sha256": PROTOCOL_SHA,
                      "expected_members": len(expected_files()), "measurement_targets_verified": len(targets),
                      "paired_cases_verified": len(cases), "training_sources": len(rows),
                      "expected_updates": protocol["total_update_budget"], "local_model_forwards": 0,
                      "local_optimizer_updates": 0, "actual_cuda_execution_verified": False, "promoted": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-contract", action="store_true", help="Read-only preparation check; no result audit claim")
    args = parser.parse_args()
    try:
        check_contract() if args.check_contract else main()
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, "Location: gray-covering return audit\nCause: " + str(error) +
                    "\nFix: use the unchanged bundle and complete archive/checksum; preserve partial evidence for diagnosis.\n")
