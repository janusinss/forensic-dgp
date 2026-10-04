"""Fixed practical transfer comparison of audited checkpoints; inference only."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_varied_covering_results import ARMS, require, sha, read, write, safe, exact, binary
from scripts.run_real_camera_inference_review import metrics, pixels, state_sha

PROTOCOL_DIR = ROOT / "outputs/varied_covering_practical_protocol_v1"
OUT = ROOT / "outputs/varied_covering_practical_review_v1"
RETURN = "outputs/downloaded_varied_covering_v1/outputs/varied_covering_vm/"
CAMERA = "outputs/real_camera_inference_review_v2/"
RETURN_AUDIT = "outputs/varied_covering_results_validation_v1/verification.json"
VISUAL_REVIEW = "outputs/varied_covering_results_validation_v1/visual_review.json"
PINS = {
    RETURN_AUDIT: "ebae5bd787f0fe0f19fa57d483c3ea29766c1ef73825e76560cd0f7f3654c95d",
    VISUAL_REVIEW: "7ce8b8b91819eec8603f359e3b92f1a895da3b7bd7e44715ecdb52f85e9f5816",
    CAMERA + "frozen_protocol.json": "024a3f7aa3775fbf45ca54c6cbfa87641893b594e5aec79b3b4f869b53b1f001",
    CAMERA + "results.json": "1ffef0024298e88c0a91e8d222aae67b0a0f898dfe06366305dc77ce9849706c",
    "outputs/real_camera_inference_validation_v1/verification.json": "8050c978a555ec7a56a8c6c9653f4ee03f59fd683419222ce3ffd22d2a7d6bfe",
    "outputs/xseg_mask_comparison_v2/results.json": "f844ec2740352f1ec5ce15fb32b06bc280fe72e6f41fea8d6b7e51f53da518df",
    "outputs/varied_covering_protocol_v2/protocol.json": "a6ff0780c86a9dc207477b5b40b6ccf6da5d88cd10fc5bb3624fb9e177d13928",
}
MODEL_NAMES = ("retained_app_default", "camera91_source", *ARMS)
EXPECTED_FAMILIES = {"face_mask": 6, "sunglasses": 4, "strong_lens_glare": 2,
    "sunglasses_and_glare": 2, "hand_over_mask": 2, "uncovered_control": 2,
    "clear_glasses_control": 2, "hand": 4, "obstructing_hair": 2, "scarf": 4,
    "other_object": 4, "visibility_rejection": 2}


def validate_probability(value):
    require(value.dtype == np.float32 and value.shape == (256, 256) and np.isfinite(value).all()
            and value.min() >= 0 and value.max() <= 1, "Invalid FP32 detector probability")
    return value


def validate_protocol(protocol):
    from collections import Counter
    cases = protocol["cases"]
    require(len(cases) == 36 and len({case["id"] for case in cases}) == 36, "Fixed practical membership differs")
    require(all(type(case["synthetically_degraded"]) is bool and type(case["expected_rejection"]) is bool
                for case in cases), "Case condition/rejection flags must be booleans")
    require(dict(Counter(case["family"] for case in cases)) == EXPECTED_FAMILIES
            and sum(case["synthetically_degraded"] for case in cases) == 18
            and sum(case["expected_rejection"] for case in cases) == 2, "Family/condition/control scope differs")
    require(protocol["threshold"] == .5 and protocol["proposal_margin_256_pixels"] == 3
            and protocol["device"] == "cpu" and protocol["threads"] == 4, "Frozen inference policy differs")
    exact(protocol["budget"], {"reproduction_forward_images": 34, "practical_forward_images": 72,
          "total_forward_images": 106, "cached_proposals": 72, "practical_metric_records": 144,
          "generator_forward_images": 0, "restorer_forward_images": 0, "optimizer_updates": 0,
          "wall_seconds_excluding_loading": 300}, "practical budget")
    require(len(protocol["reproduction_cases"]) == 16
            and len({row["name"] for row in protocol["reproduction_cases"]}) == 16,
            "Fixed reproduction membership differs")
    require(protocol["checkpoint_selection"] is False and protocol["promoted"] is False
            and protocol["training"] is False, "Diagnostic cannot promote or train")


def prepare():
    require(not PROTOCOL_DIR.exists(), "Preserve existing frozen practical protocol")
    require((ROOT / RETURN_AUDIT).is_file() and (ROOT / VISUAL_REVIEW).is_file(),
            "Audit VM return and inspect all ten preview rows before practical preparation")
    for name, pin in PINS.items():
        require(sha(safe(ROOT, name)) == pin, "Verified reference changed: " + name)
    returned, visual = read(ROOT / RETURN_AUDIT), read(ROOT / VISUAL_REVIEW)
    require(returned["complete"] is True and returned["promoted"] is False
            and visual["all_ten_rows_inspected"] is True and visual["promoted"] is False, "Prior audit/review incomplete")
    prior = read(ROOT / CAMERA / "frozen_protocol.json")
    previous = {row["id"]: row for row in read(ROOT / CAMERA / "results.json")["rows"]}
    assets, cases = dict(PINS), []
    def asset(name):
        assets[name] = sha(safe(ROOT, name))
        return name
    for old in prior["cases"]:
        item = {key: old[key] for key in ("id", "family", "input", "reviewed", "protected", "synthetically_degraded",
                                        "expected_rejection", "pose_scope", "exposure", "hidden_ground_truth")}
        require(item["hidden_ground_truth"] is None, "Do not fabricate hidden facial targets")
        for key in ("input", "reviewed", "protected"):
            if item[key]: asset(item[key])
        require(pixels(ROOT / item["input"]).shape == (256, 256, 3), "Fixed RGB geometry differs")
        truth = binary(ROOT / item["reviewed"])
        item["clear_control"] = not bool(truth.any())
        camera = previous[item["id"]]["models"]["camera91"]
        item["cached"] = {"retained_app_default": asset(old["baseline_cached"]),
                          "camera91_source": asset(CAMERA + camera["proposal"])}
        for name in item["cached"].values(): binary(ROOT / name)
        cases.append(item)
    require(sum(case["clear_control"] for case in cases) == 4, "Four clear controls required")
    models = {arm: asset(RETURN + arm + "/last.pth") for arm in ARMS}
    for arm, path in models.items():
        require(assets[path] == returned["checkpoint_sha256"][arm], "Audited candidate differs")
    pairs = read(ROOT / "outputs/cofw_camera_pairs_v2/manifest.json")["cases"]
    pilot = read(ROOT / "outputs/varied_covering_protocol_v2/protocol.json")
    ids = pilot["preview_case_ids"] + [0, 1, 20, 21, 244, 245]
    reproduction = []
    for index in ids:
        case = pairs[index]
        reproduction.append({"name": f"real/{index:03d}", "input": asset(case["input"])})
    for arm in ARMS:
        for row in reproduction + [{"name": "fixture/171"}]:
            asset(RETURN + arm + "/final_masks/" + row["name"] + ".png")
    for name in ("outputs/cofw_camera_pairs_v2/manifest.json", "outputs/reflection_coverage_data_v1/pixels.pth",
                 "scripts/run_varied_covering_practical_review.py", "scripts/audit_varied_covering_practical_review.py",
                 "scripts/run_real_camera_inference_review.py", "scripts/audit_real_camera_inference_review.py",
                 "scripts/audit_varied_covering_results.py", "scripts/audit_real_camera_results.py",
                 "face_occlusion_adapter.py", "face_workflow.py"):
        asset(name)
    protocol = {"format": "dgp-varied-covering-practical-review-v1", "date": "2026-10-03",
        "frozen_before_new_forwards": True, "cases": cases, "models": models, "assets_sha256": assets,
        "reproduction_cases": reproduction, "reproduction_fixture_id": 171,
        "threshold": .5, "proposal_margin_256_pixels": 3, "threads": 4, "device": "cpu",
        "budget": {"reproduction_forward_images": 34, "practical_forward_images": 72, "total_forward_images": 106,
                   "cached_proposals": 72, "practical_metric_records": 144, "generator_forward_images": 0,
                   "restorer_forward_images": 0, "optimizer_updates": 0, "wall_seconds_excluding_loading": 300},
        "cpu_cuda_reproduction": {"threshold_ambiguous_pixel_limit_per_mask": 4,
                                  "max_local_probability_distance_from_half_for_disagreement": 1e-5},
        "criteria": ["Use every frozen practical case; report covering families and native/degraded conditions separately",
                     "Require empty clear controls; retain missed near-hidden rejection and excessive visible edits as failures",
                     "Inspect all six preview sheets and per-case defects; aggregate gains cannot establish useful face output",
                     "No automatic selection; preserve failed training-fit checks and separate original425 qualification"],
        "scope": prior["scope"], "source_training_fit_decision": returned["fit_decision"],
        "historical_425_gates_evaluated": False, "checkpoint_selection": False, "promoted": False, "training": False}
    validate_protocol(protocol)
    PROTOCOL_DIR.mkdir()
    write(PROTOCOL_DIR / "protocol.json", protocol)
    write(PROTOCOL_DIR / "binding.json", {"protocol_sha256": sha(PROTOCOL_DIR / "protocol.json"),
                                         "preparation_only": True, "model_forwards": 0, "optimizer_updates": 0})
    print(json.dumps({"prepared": True, "cases": 36, "forward_budget": 106, "protocol_sha256": sha(PROTOCOL_DIR / "protocol.json")}))


def run():
    # Refuse before model/runtime construction when the independent return is missing.
    require((ROOT / RETURN_AUDIT).is_file(), "Audited VM return required; do not run another training recipe")
    protocol = read(PROTOCOL_DIR / "protocol.json")
    require(sha(PROTOCOL_DIR / "protocol.json") == read(PROTOCOL_DIR / "binding.json")["protocol_sha256"], "Frozen practical protocol changed")
    validate_protocol(protocol)
    require(not OUT.exists(), "Preserve partial/completed practical inference")
    for name, pin in protocol["assets_sha256"].items():
        require(sha(safe(ROOT, name)) == pin, "Frozen inference asset changed: " + name)
    import torch
    sys.path.insert(0, str(ROOT / "outputs/face_extraction_dependencies"))
    from scripts.train_native_expert_vm import dependency_versions
    dependencies = dependency_versions(ROOT / "outputs/face_extraction_dependencies")
    from face_occlusion_adapter import load_adapter
    torch.set_num_threads(4)
    cache = torch.load(ROOT / "outputs/reflection_coverage_data_v1/pixels.pth", map_location="cpu", weights_only=True)
    fixture = cache["pixels"][171]["input"].permute(1, 2, 0).numpy()
    OUT.mkdir()
    for name in ("raw", "proposals", "probabilities", "reproduction", "preview"): (OUT / name).mkdir()
    write(OUT / "execution.json", {"protocol_sha256": sha(PROTOCOL_DIR / "protocol.json"), "budget": protocol["budget"],
          "device": "cpu", "torch": str(torch.__version__), "dependencies": dependencies,
          "optimizer_constructed": False, "optimizer_updates": 0})
    rows = [{**{key: case[key] for key in ("id", "family", "synthetically_degraded", "expected_rejection", "pose_scope", "clear_control")},
             "models": {}} for case in protocol["cases"]]
    artifacts, reproduction, states, total, elapsed = {}, [], {}, 0, 0.
    def save(name, value, probability=False):
        if probability: np.save(OUT / name, value, allow_pickle=False)
        else: Image.fromarray(value.astype(np.uint8) * 255).save(OUT / name)
        artifacts[name] = sha(OUT / name)
        return name
    for arm, path in protocol["models"].items():
        model, _ = load_adapter(ROOT / path, "cpu")
        model.eval().requires_grad_(False)
        before, count, started = state_sha(model), [0], time.monotonic()
        def hook(module, args, output): count[0] += len(args[0])
        handle = model.network.register_forward_hook(hook)
        def predict(rgb):
            require(elapsed + time.monotonic() - started < 300, "Finite five-minute inference cap exceeded")
            x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()[None] / 255
            with torch.inference_mode(): probability = model.detect(x).sigmoid()[0, 0].numpy()
            return validate_probability(probability)
        for case in protocol["reproduction_cases"] + [{"name": "fixture/171", "input": None}]:
            probability = predict(pixels(ROOT / case["input"]) if case["input"] else fixture)
            actual = probability >= .5
            returned = binary(ROOT / RETURN / arm / "final_masks" / (case["name"] + ".png"))
            different = actual != returned
            require(int(different.sum()) <= 4 and (not different.any() or np.all(np.abs(probability[different] - .5) <= 1e-5)),
                    "CPU/CUDA prediction reproduction differs")
            stem = arm + "_" + case["name"].replace("/", "_")
            reproduction.append({"model": arm, "case": case["name"], "different_pixels": int(different.sum()),
                "probability": save("reproduction/" + stem + ".npy", probability, True),
                "raw": save("reproduction/" + stem + ".png", actual)})
        for case, row in zip(protocol["cases"], rows):
            probability = predict(pixels(ROOT / case["input"])); raw = probability >= .5
            proposal = cv2.dilate(raw.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            truth = binary(ROOT / case["reviewed"])
            protected = binary(ROOT / case["protected"]) if case["protected"] else np.zeros_like(truth)
            if arm == ARMS[0]:
                for name, cached in case["cached"].items():
                    row["models"][name] = {"proposal": cached, "cached": True, "metrics": metrics(binary(ROOT / cached), truth, protected)}
            stem = arm + "_" + case["id"]
            row["models"][arm] = {"raw": save("raw/" + stem + ".png", raw),
                "proposal": save("proposals/" + stem + ".png", proposal),
                "probability": save("probabilities/" + stem + ".npy", probability, True),
                "cached": False, "metrics": metrics(proposal, truth, protected)}
        elapsed += time.monotonic() - started
        after = state_sha(model)
        require(before == after and not model.training and not any(p.requires_grad for p in model.parameters()), "Inference changed detector state")
        require(count[0] == 53, "Branch forward count differs")
        states[arm] = {"before": before, "after": after, "forward_images": count[0]}
        total += count[0]; handle.remove(); del model; gc.collect()
        print(f"{arm}: 53 inference images, zero updates; elapsed {elapsed:.2f}s", flush=True)
    require(total == 106 and len(reproduction) == 34 and elapsed < 300, "Finite practical budget differs")
    for first in range(0, 36, 6):
        canvas = Image.new("RGB", (1152, 32 + 217 * 6), "#16181c"); draw = ImageDraw.Draw(canvas)
        for col, name in enumerate(("input", *MODEL_NAMES, "reviewed reference")): draw.text((192 * col + 3, 7), name, fill="white")
        for index, case in enumerate(protocol["cases"][first:first + 6]):
            row, rgb = rows[first + index], pixels(ROOT / case["input"])
            panels = [rgb]
            for name in (*MODEL_NAMES, "reviewed reference"):
                if name == "reviewed reference": path = ROOT / case["reviewed"]
                else:
                    item = row["models"][name]; path = (ROOT if item["cached"] else OUT) / item["proposal"]
                active = binary(path); marked = rgb.copy()
                marked[active] = (.55 * rgb[active] + .45 * np.array([16, 185, 129])).round().astype(np.uint8)
                panels.append(marked)
            y = 32 + index * 217; draw.text((3, y), case["id"], fill="white")
            for col, panel in enumerate(panels): canvas.paste(Image.fromarray(panel).resize((192, 192)), (192 * col, y + 19))
        name = f"preview/rows_{first + 1:02d}_{first + 6:02d}.png"; canvas.save(OUT / name); artifacts[name] = sha(OUT / name)
    for name, pin in protocol["assets_sha256"].items(): require(sha(ROOT / name) == pin, "Inference source file changed")
    require(len(artifacts) == 290, "Saved probability/mask/preview budget differs")
    result = {"format": "dgp-varied-covering-practical-result-v1", "date": "2026-10-03", "complete": True,
        "protocol_sha256": sha(PROTOCOL_DIR / "protocol.json"), "rows": rows, "reproduction": reproduction,
        "model_states": states, "actual_forward_images": total, "seconds_excluding_loading": elapsed,
        "artifact_sha256": artifacts, "generator_forwards": 0, "restorer_forwards": 0, "optimizer_updates": 0,
        "training": False, "promoted": False, "original425_gates_evaluated": False, "visual_review_pending": True}
    write(OUT / "results.json", result)
    print(json.dumps({"complete": True, "forward_images": total, "result_sha256": sha(OUT / "results.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else run()
