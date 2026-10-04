"""Nine finite CPU inference requests on four fixed assisted cases; no training."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from completion_context_margin import ContextMarginCompletion, conditioning_mask
from face_workflow import visibility_check
from face_workflow_palette import PaletteFaceWorkflow
from scripts.run_real_camera_inference_review import sha, read, write, pixels, state_sha

OUT = ROOT / "outputs/completion_context_review_v1"
IDS = ("val_18_hand_eyes_degraded", "val_362_hair_eye_degraded",
       "val_336_scarf_gloves_degraded", "val_6_flower_mouth_degraded")
BASE = "outputs/grayscale_palette_comparison_v2/"


def prepare():
    if OUT.exists():
        raise ValueError("Preserve earlier context experiment")
    source = read(ROOT / "outputs/broad_covering_gallery_v1/frozen_protocol.json")
    prior = read(ROOT / BASE / "results.json")
    proof = read(ROOT / BASE / "independent_verification.json")
    if proof["verified"] is not True or prior["complete"] is not True or proof["saved_outputs_verified"] != 28 \
            or proof["results_sha256"] != sha(ROOT / BASE / "results.json") \
            or proof["protocol_sha256"] != sha(ROOT / BASE / "frozen_protocol.json"):
        raise ValueError("Verified cached assisted baseline required")
    files = ["completion_context_margin.py", "tests/test_completion_context_margin.py",
             "scripts/run_completion_context_review.py", "scripts/audit_completion_context_review.py",
             "face_workflow.py", "face_workflow_palette.py", "face_color_policy.py", "pretrained_completion.py",
             "pretrained_face_restoration.py", "completion.py", "scripts/run_real_camera_inference_review.py",
             "outputs/broad_covering_gallery_v1/frozen_protocol.json", BASE + "results.json",
             BASE + "independent_verification.json", BASE + "frozen_protocol.json", "checkpoints/codeformer_inpainting.pth",
             "outputs/codeformer_restoration_pretrained_v1/codeformer.pth"]
    cases = []
    for name in IDS:
        case = next(c for c in source["cases"] if c["id"] == name)
        row = next(r for r in prior["rows"] if r["id"] == name and r["arm"] == "assisted")
        mask = (pixels(ROOT / case["proposal"], "L") == 255).astype(np.uint8)
        if case["expected_rejection"] or visibility_check(mask)["rejected"] or not row["restoration_applied"]:
            raise ValueError("Require non-rejected fixed degraded assisted inputs with cached Auto restoration")
        for radius in (0, 2, 6):
            context = conditioning_mask(torch.from_numpy(mask).float()[None, None], radius)
            if float(context.mean()) >= .85:
                raise ValueError("Prospective context exceeds unchanged backend safety limit")
        cases.append({"id": name, "input": case["input"], "mask": case["proposal"],
                      "reference": case["reference"], "baseline": BASE + row["output"],
                      "family": case["family"], "exposure": case["exposure"], "hidden_ground_truth": None})
        files.extend([case["input"], case["proposal"], case["reference"], BASE + row["output"]])
    p = {"format": "dgp-completion-conditioning-margin-review-v1", "date": "2026-10-03",
         "frozen_before_generation": True, "cases": cases, "radii_at256": [2, 6],
         "baseline_reproduction_case": "val_336_scarf_gloves_degraded",
         "assets_sha256": {name: sha(ROOT / name) for name in files}, "device": "cpu", "restoration": "auto",
         "mask_source": "Existing reviewed assisted proposals; no edits or added output support",
         "context_rule": "Square dilation of input suppression by 2 or 6 pixels at256; final generation restricted to original reviewed area",
         "budget": {"generation_requests": 9, "completion_forwards": 9, "restoration_forwards": 9,
                    "detector_forwards": 0, "optimizer_updates": 0, "wall_seconds_after_loading": 180},
         "criteria": ["Inspect reduced covering texture/remnants and plausible rough anatomy versus cached assisted baseline",
                      "Preserve reviewed output support, input-based Auto routing and generated pixels through visible restoration",
                      "Reject a candidate with material visible-appearance or structural regressions; numeric visible MAE is diagnostic, not hidden identity"],
         "limitations": ["Four previously inspected degraded development sources; native, clear glasses and full-gallery regressions not covered",
                         "This tests completion conditioning only; automatic hair detection and near-hidden misses remain unresolved"],
         "optimizer_constructed": False, "application_change": False, "historical_gates_unchanged": True}
    OUT.mkdir()
    write(OUT / "frozen_protocol.json", p)
    print(json.dumps({"prepared": True, "protocol_sha256": sha(OUT / "frozen_protocol.json"), "new_generation_budget": 9}))


def run():
    if (OUT / "execution.json").exists():
        raise ValueError("Preserve earlier completed/partial inference")
    p = read(OUT / "frozen_protocol.json")
    for name, pin in p["assets_sha256"].items():
        if sha(ROOT / name) != pin:
            raise ValueError("Frozen context asset changed: " + name)
    engine = PaletteFaceWorkflow(device="cpu")
    engine._runtime(); engine._generator(); engine._restorer()
    base = engine.generator
    models = {"completion": base, "restoration": engine.restorer}
    if engine.detector is not None or any(m.training or any(v.requires_grad for v in m.parameters()) for m in models.values()):
        raise ValueError("Require frozen completion/restoration with no detector")
    before = {k: state_sha(v) for k, v in models.items()}
    counts, captured = {"completion": 0, "restoration": 0}, {}

    def hook(kind):
        def capture(module, args, output):
            counts[kind] += 1
            captured[kind] = output.detach().cpu().numpy().copy()
        return capture

    handles = [base.register_forward_hook(hook("completion")), engine.restorer.register_forward_hook(hook("restoration"))]
    write(OUT / "execution.json", {"protocol_sha256": sha(OUT / "frozen_protocol.json"),
          "state_before": before, "budget": p["budget"], "device": "cpu", "optimizer_constructed": False})
    for folder in ("outputs", "stages", "masks"):
        (OUT / folder).mkdir()
    start, rows, hashes = time.monotonic(), [], {}
    requests = [(p["baseline_reproduction_case"], 0)] + [(c["id"], r) for c in p["cases"] for r in p["radii_at256"]]
    for name, radius in requests:
        if time.monotonic() - start > 180:
            raise ValueError("Fixed context comparison processing cap exceeded")
        case = next(c for c in p["cases"] if c["id"] == name)
        original = pixels(ROOT / case["input"])
        mask = (pixels(ROOT / case["mask"], "L") == 255).astype(np.uint8)
        engine.generator = wrapper = ContextMarginCompletion(base, radius).eval().requires_grad_(False)
        captured.clear()
        output, metadata = engine.generate(original, mask, "auto")
        if not metadata["restoration_applied"] or set(captured) != set(counts):
            raise ValueError("Fixed completion/Auto restoration membership differs")
        tag = f"context{radius}_{name}"
        filename = "outputs/" + tag + ".png"
        Image.fromarray(output).save(OUT / filename)
        context_name = "masks/" + tag + ".png"
        Image.fromarray(wrapper.last_context_mask[0, 0].numpy().astype(np.uint8) * 255).save(OUT / context_name)
        stages = {}
        for kind, value in captured.items():
            stage = "stages/" + tag + "_" + kind + ".npy"
            np.save(OUT / stage, value, allow_pickle=False)
            stages[kind] = stage
            hashes[stage] = sha(OUT / stage)
        for path in (filename, context_name):
            hashes[path] = sha(OUT / path)
        metadata.update(mask_source=p["mask_source"], context_suppression_radius_at256=radius,
                        conditioning_support_changed=radius > 0, final_removal_support_changed=False)
        if radius == 0 and not np.array_equal(output, pixels(ROOT / case["baseline"])):
            raise ValueError("Zero-radius output failed exact cached baseline reproduction; preserve partial evidence")
        reference = pixels(ROOT / case["reference"])
        visible = mask == 0
        difference = np.abs(output.astype(float) - reference.astype(float))[visible] / 255
        rows.append({"id": name, "radius_at256": radius, "output": filename, "context_mask": context_name,
                     "stage_arrays": stages, "metadata": metadata, "hidden_ground_truth": None,
                     "visible_mae": float(difference.mean()),
                     "baseline_visible_mae": float((np.abs(pixels(ROOT / case["baseline"]).astype(float) - reference.astype(float))[visible] / 255).mean())})
        write(OUT / f"progress_{len(rows):02d}.json", {"completed_requests": len(rows), "latest": rows[-1], "forwards": dict(counts)})
        print(tag + f": {metadata['elapsed_seconds']:.2f}s", flush=True)
    after = {k: state_sha(v) for k, v in models.items()}
    elapsed = time.monotonic() - start
    if before != after or counts != {"completion": 9, "restoration": 9} or engine.detector is not None or elapsed > 180:
        raise ValueError("Frozen inference state or budget differs")
    for handle in handles:
        handle.remove()
    canvas = Image.new("RGB", (1280, 32 + 281 * len(p["cases"])), "#16181c")
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(("input", "reviewed area", "cached assisted", "context 2px", "context 6px")):
        draw.text((256 * column + 4, 7), title, fill="white")
    for line, case in enumerate(p["cases"]):
        original = pixels(ROOT / case["input"])
        mask = pixels(ROOT / case["mask"], "L") == 255
        marked = original.copy()
        marked[mask] = (.55 * original[mask] + .45 * np.array([16, 185, 129])).round().astype(np.uint8)
        arrays = [original, marked, pixels(ROOT / case["baseline"])]
        arrays.extend(pixels(OUT / next(r["output"] for r in rows if r["id"] == case["id"] and r["radius_at256"] == radius)) for radius in (2, 6))
        y = 32 + line * 281
        draw.text((3, y), case["id"], fill="white")
        for column, array in enumerate(arrays):
            canvas.paste(Image.fromarray(array), (256 * column, y + 19))
    canvas.save(OUT / "preview.png")
    hashes["preview.png"] = sha(OUT / "preview.png")
    write(OUT / "results.json", {"complete": True, "protocol_sha256": sha(OUT / "frozen_protocol.json"), "rows": rows,
          "forwards": counts, "detector_forwards": 0, "optimizer_updates": 0, "generation_requests": 9,
          "state_before": before, "state_after": after, "seconds_excluding_loading": elapsed,
          "artifacts_sha256": hashes, "zero_radius_cached_baseline_matches": True,
          "application_changed": False, "promoted": False, "visual_review_pending": True})
    print(json.dumps({"complete": True, "forwards": counts, "seconds": elapsed, "results_sha256": sha(OUT / "results.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else run()
