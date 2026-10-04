"""Frozen return reproduction and practical transfer review; inference only."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from face_workflow import visibility_check

OUT = ROOT / "outputs/real_camera_inference_review_v1"
RETURN = "outputs/downloaded_real_camera_v1/outputs/real_camera_vm/"
SOURCE = "outputs/downloaded_reflection_coverage/outputs/reflection_coverage_vm/reflective/epoch_42.pth"
ARMS = ("native83", "camera83", "camera91")
PINS = {
    "outputs/real_camera_results_validation_v1/verification.json": "fbe47739fa7ec05f88b7ee22861d81b64b945c320109e4b7aee5a05c8af74245",
    "outputs/practical_direct_detector_v1/frozen_protocol.json": "69c6a75e4dc1af099a3cb1a621082a870966170108e9a2d2255d614b708a8db8",
    "outputs/practical_direct_detector_v1/results.json": "4b07e9a7765486d588f025392f7cc3d2adc84e3a731a8012b4037fa7885975b0",
    "face_occlusion_adapter.py": "3991dbf2b0bcd19363cfd945aad0f73b4a323ff0c039d2d99bc951518aa76eed",
    "face_workflow.py": "ed043f9819e4655c297f092037af156cbfa9e32506f6b15cca0c5d5dfe4cacd1",
    SOURCE: "a51f20e8fa12cee7dec574195debc5ada9b8cfeeb289b12bad11a6d39a9acf25",
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def pixels(path, mode="RGB"):
    with Image.open(path) as image:
        return np.asarray(image.convert(mode)).copy()


def prepare():
    if OUT.exists():
        raise ValueError("Preserve existing protocol/evidence")
    for name, pin in PINS.items():
        if sha(ROOT / name) != pin:
            raise ValueError("Frozen reference changed: " + name)
    prior = read(ROOT / "outputs/practical_direct_detector_v1/frozen_protocol.json")
    prior_rows = {r["id"]: r for r in read(ROOT / "outputs/practical_direct_detector_v1/results.json")["rows"]}
    audit = read(ROOT / "outputs/real_camera_results_validation_v1/verification.json")
    p = read(ROOT / "outputs/real_camera_protocol_v1/protocol.json")
    pairs = read(ROOT / "outputs/real_camera_pairs_v1/manifest.json")
    assets = dict(PINS); models = {"source42": SOURCE}
    for arm in ARMS:
        name = RETURN + arm + "/last.pth"; models[arm] = name
        assets[name] = audit["checkpoint_sha256"][arm]
    cases = []
    for row in prior["cases"]:
        previous = prior_rows[row["id"]]
        case = {**row, "source42_raw": "outputs/practical_direct_detector_v1/" + previous["raw_mask"],
                "source42_proposal": "outputs/practical_direct_detector_v1/" + previous["direct_mask"]}
        cases.append(case)
        for field in ("input", "reviewed", "protected", "source42_raw", "source42_proposal"):
            if case[field]:
                assets[case[field]] = sha(ROOT / case[field])
    reproduction = []
    for i in p["preview_case_ids"]:
        case = pairs["cases"][i]
        reproduction.append({"name": f"real/{i:03d}", "input": case["input"], "input_sha256": case["input_sha256"]})
        assets[case["input"]] = case["input_sha256"]
    for model in models:
        folder = "initial_masks" if model == "source42" else model + "/final_masks"
        for row in reproduction + [{"name": "fixture/171"}]:
            name = RETURN + folder + "/" + row["name"] + ".png"; assets[name] = sha(ROOT / name)
    assets["outputs/reflection_coverage_data_v1/pixels.pth"] = "ab47a88d79fe8d300ca92d572a2c1a05b92fa273b9577cb209df99e692009cc1"
    assets["scripts/run_real_camera_inference_review.py"] = sha(__file__)
    for name, pin in assets.items():
        if sha(ROOT / name) != pin:
            raise ValueError("Review asset changed: " + name)
    protocol = {"format": "dgp-real-camera-local-inference-review-v1", "date": "2026-10-03",
        "frozen_before_new_forwards": True, "cases": cases, "models": models, "assets_sha256": assets,
        "reproduction_cases": reproduction, "reproduction_fixture_id": 171,
        "threshold": .5, "proposal_margin_256_pixels": 3, "threads": 4, "device": "cpu",
        "budget": {"reproduction_forward_images": 44, "practical_forward_images": 108,
                   "total_forward_images": 152, "generator_forward_images": 0, "optimizer_updates": 0,
                   "wall_seconds_excluding_loading": 300},
        "cpu_cuda_reproduction": {"threshold_ambiguous_pixel_limit_per_mask": 4,
                                   "max_local_probability_distance_from_half_for_disagreement": 1e-5,
                                   "selection_threshold_unchanged": True},
        "criteria": ["Reproduce ten fixed real and one clear-fixture masks for source42 and all three arms",
                     "Report all36 practical references, family/condition and clear/rejection controls",
                     "Keep ordinary clear glasses and uncovered controls; flag misses and excessive removal",
                     "Inspect six full preview sheets; do not qualify checkpoints or alter historical425 gates"],
        "scope": "Exposed development gallery, not unseen population or hidden-identity evidence",
        "training": False, "checkpoint_selection": False, "promoted": False}
    OUT.mkdir(); write(OUT / "frozen_protocol.json", protocol)
    print(json.dumps({"protocol_sha256": sha(OUT / "frozen_protocol.json"), "total_forward_budget": 152}))


def state_sha(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def metrics(prediction, truth, protected):
    p, t = prediction.astype(bool), truth.astype(bool)
    hit, union = int((p & t).sum()), int((p | t).sum())
    tolerance = cv2.dilate(t.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    return {"predicted_pixels": int(p.sum()), "reference_pixels": int(t.sum()), "intersection_pixels": hit,
            "union_pixels": union, "recall": hit / int(t.sum()) if t.any() else None,
            "precision": hit / int(p.sum()) if p.any() else None, "iou": hit / union if t.any() and union else None,
            "excess_pixels_outside_3px_tolerance": int((p & ~tolerance).sum()),
            "protected_wire_pixels_changed": int((p & protected).sum()),
            "empty_control_marked_fraction": float(p.mean()) if not t.any() else None,
            "guard": visibility_check(prediction)}


def previews(cases, rows):
    saved = {}
    for first in range(0, len(cases), 6):
        selected = cases[first:first + 6]
        canvas = Image.new("RGB", (1152, 32 + 217 * len(selected)), "#16181c"); draw = ImageDraw.Draw(canvas)
        for col, label in enumerate(("input", "source42", *ARMS, "reviewed reference")):
            draw.text((192 * col + 3, 7), label, fill="white")
        for index, case in enumerate(selected):
            row = rows[first + index]; original = pixels(ROOT / case["input"])
            panels = [original]
            paths = [ROOT / case["source42_proposal"]] + [OUT / row["models"][a]["proposal"] for a in ARMS] + [ROOT / case["reviewed"]]
            for path in paths:
                marked = original.copy(); active = pixels(path, "L") == 255
                marked[active] = (.55 * original[active] + .45 * np.array([16, 185, 129])).round().astype(np.uint8)
                panels.append(marked)
            y = 32 + index * 217; draw.text((3, y), case["id"], fill="white")
            for col, panel in enumerate(panels):
                canvas.paste(Image.fromarray(panel).resize((192, 192)), (192 * col, y + 19))
        name = f"preview/rows_{first + 1:02d}_{first + len(selected):02d}.png"
        canvas.save(OUT / name); saved[name] = sha(OUT / name)
    return saved


def run():
    if (OUT / "execution.json").exists():
        raise ValueError("Preserve completed/partial inference; never rerun in place")
    p = read(OUT / "frozen_protocol.json")
    if not p["frozen_before_new_forwards"] or len(p["cases"]) != 36:
        raise ValueError("Frozen review budget differs")
    for name, pin in p["assets_sha256"].items():
        if sha(ROOT / name) != pin:
            raise ValueError("Frozen asset differs: " + name)
    sys.path.insert(0, str(ROOT / "outputs/face_extraction_dependencies"))
    from face_occlusion_adapter import load_adapter
    torch.set_num_threads(4)
    cache = torch.load(ROOT / "outputs/reflection_coverage_data_v1/pixels.pth", map_location="cpu", weights_only=True)
    fixture = cache["pixels"][171]["input"].permute(1, 2, 0).numpy()
    for name in ("raw", "proposals", "probabilities", "preview"):
        (OUT / name).mkdir()
    write(OUT / "execution.json", {"protocol_sha256": sha(OUT / "frozen_protocol.json"), "budget": p["budget"],
          "device": "cpu", "torch": str(torch.__version__), "optimizer_constructed": False})
    rows = [{"id": c["id"], "family": c["family"], "synthetically_degraded": c["synthetically_degraded"],
             "pose_scope": c["pose_scope"], "expected_rejection": c["expected_rejection"], "models": {}}
            for c in p["cases"]]
    total, elapsed, reproduction, states, hashes = 0, 0., [], {}, {}
    for name, path in p["models"].items():
        model, _ = load_adapter(ROOT / path, "cpu"); model.eval().requires_grad_(False)
        before = state_sha(model); counter = [0]
        def hook(module, args, output):
            counter[0] += len(args[0])
        handle = model.network.register_forward_hook(hook)
        started = time.monotonic()
        def predict(image):
            if elapsed + time.monotonic() - started > 300:
                raise ValueError("Fixed inference wall budget exceeded")
            x = torch.from_numpy(image.copy()).permute(2, 0, 1).float()[None] / 255
            with torch.inference_mode():
                probability = model.detect(x).sigmoid()[0, 0].numpy()
            if probability.shape != (256, 256) or not np.isfinite(probability).all():
                raise ValueError("Invalid inference probability")
            return probability
        folder = "initial_masks" if name == "source42" else name + "/final_masks"
        for case in p["reproduction_cases"] + [{"name": "fixture/171", "input": None}]:
            probability = predict(pixels(ROOT / case["input"]) if case["input"] else fixture)
            actual = probability >= .5; returned = pixels(ROOT / RETURN / folder / (case["name"] + ".png"), "L") == 255
            different = actual != returned; count = int(different.sum())
            ambiguous = count <= 4 and (count == 0 or bool((np.abs(probability[different] - .5) <= 1e-5).all()))
            if not ambiguous:
                raise ValueError("CPU/CUDA prediction reproduction differs: " + name + "/" + case["name"])
            reproduction.append({"model": name, "case": case["name"], "different_pixels": count,
                                 "all_differences_threshold_ambiguous": ambiguous})
        for i, case in enumerate(p["cases"]):
            truth = pixels(ROOT / case["reviewed"], "L") == 255
            protected = pixels(ROOT / case["protected"], "L") == 255 if case["protected"] else np.zeros((256, 256), bool)
            if name == "source42":
                raw = pixels(ROOT / case["source42_raw"], "L") == 255
                proposal = pixels(ROOT / case["source42_proposal"], "L") == 255
                rows[i]["models"][name] = {"raw": case["source42_raw"], "proposal": case["source42_proposal"],
                                          "cached": True, "metrics": metrics(proposal, truth, protected)}
                continue
            probability = predict(pixels(ROOT / case["input"])); raw = probability >= .5
            proposal = cv2.dilate(raw.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            saved = {"raw": f"raw/{name}_{case['id']}.png", "proposal": f"proposals/{name}_{case['id']}.png",
                     "probability": f"probabilities/{name}_{case['id']}.npy"}
            Image.fromarray(raw.astype(np.uint8) * 255).save(OUT / saved["raw"])
            Image.fromarray(proposal.astype(np.uint8) * 255).save(OUT / saved["proposal"])
            np.save(OUT / saved["probability"], probability, allow_pickle=False)
            hashes.update({v: sha(OUT / v) for v in saved.values()})
            rows[i]["models"][name] = {**saved, "cached": False, "metrics": metrics(proposal, truth, protected)}
        elapsed += time.monotonic() - started
        after = state_sha(model)
        if before != after or model.training or any(t.requires_grad for t in model.parameters()):
            raise ValueError("Inference model state/frozen contract changed")
        total += counter[0]; states[name] = {"before": before, "after": after, "forward_images": counter[0]}
        handle.remove(); del model; gc.collect()
    if total != 152 or len(reproduction) != 44:
        raise ValueError("Actual inference/reproduction budget differs")
    hashes.update(previews(p["cases"], rows))
    result = {"complete": True, "date": "2026-10-03", "protocol_sha256": sha(OUT / "frozen_protocol.json"),
        "rows": rows, "reproduction": reproduction, "model_states": states, "actual_forward_images": total,
        "practical_case_predictions": 108, "source42_cached_case_predictions": 36, "seconds_excluding_loading": elapsed,
        "artifact_sha256": hashes, "generator_forwards": 0, "restorer_forwards": 0, "optimizer_updates": 0,
        "training": False, "promoted": False, "historical_gates_unchanged": True, "visual_review_pending": True}
    write(OUT / "results.json", result)
    print(json.dumps({"complete": True, "forward_images": total, "reproduction_different_pixels": sum(r["different_pixels"] for r in reproduction),
                      "seconds_excluding_loading": elapsed, "results_sha256": sha(OUT / "results.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else run()
