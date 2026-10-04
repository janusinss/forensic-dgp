"""Twenty fixed downstream estimates from audited varied-covering masks; no training."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from face_workflow_palette import PaletteFaceWorkflow
from scripts.run_real_camera_inference_review import sha, read, write, pixels, state_sha

OUT = ROOT / "outputs/varied_covering_completion_review_v1"
MASK_ROOT = "outputs/varied_covering_practical_review_v1/"
ARMS = ("existing91", "varied133")
IDS = ("00_cloth_mask_degraded", "03_dark_sunglasses_degraded", "05_white_glare_degraded",
       "val_18_hand_eyes_degraded", "val_25_hand_mouth_degraded", "val_362_hair_eye_degraded",
       "val_244_knit_scarf_degraded", "val_336_scarf_gloves_degraded",
       "val_6_flower_mouth_degraded", "val_7_leaf_eye_degraded")


def prepare():
    if OUT.exists():
        raise ValueError("Preserve frozen output experiment")
    previous = read(ROOT / "outputs/varied_covering_practical_protocol_v1/protocol.json")
    proof = read(ROOT / "outputs/varied_covering_practical_validation_v1/verification.json")
    visual = read(ROOT / "outputs/varied_covering_practical_validation_v1/visual_review.json")
    if not proof["complete"] or not visual["all_36_cases_inspected"]:
        raise ValueError("Independent practical audit and full visual review required")
    rows = {r["id"]: r for r in read(ROOT / MASK_ROOT / "results.json")["rows"]}
    files = ["face_workflow.py", "face_workflow_palette.py", "face_color_policy.py", "pretrained_completion.py",
             "pretrained_face_restoration.py", "scripts/run_varied_covering_completion_review.py",
             "scripts/run_real_camera_inference_review.py",
             "scripts/audit_varied_covering_completion_review.py",
             "outputs/varied_covering_practical_protocol_v1/protocol.json", MASK_ROOT + "results.json",
             "outputs/varied_covering_practical_validation_v1/verification.json",
             "outputs/varied_covering_practical_validation_v1/visual_review.json",
             "checkpoints/codeformer_inpainting.pth", "outputs/codeformer_restoration_pretrained_v1/codeformer.pth"]
    cases = []
    for name in IDS:
        case = next(c for c in previous["cases"] if c["id"] == name)
        if case["expected_rejection"]:
            raise ValueError("Do not generate source-held nearly hidden inputs")
        masks = {a: MASK_ROOT + rows[name]["models"][a]["proposal"] for a in ARMS}
        cases.append({"id": name, "input": case["input"], "reviewed_reference": case["reviewed"], "masks": masks,
                      "family": case["family"], "exposure": case["exposure"], "hidden_ground_truth": None})
        files.extend([case["input"], case["reviewed"], *masks.values()])
    protocol = {"format": "dgp-varied-covering-downstream-review-v1", "date": "2026-10-03", "frozen_before_generation": True,
        "cases": cases, "arms": list(ARMS), "assets_sha256": {n: sha(ROOT / n) for n in files},
        "device": "cpu", "restoration": "auto", "mask_policy": "Previously frozen automatic proposals; no manual edits or further expansion",
        "budget": {"generation_requests": 20, "completion_forwards": 18, "maximum_restoration_forwards": 20,
                   "detector_forwards": 0, "optimizer_updates": 0, "wall_seconds_after_loading": 300},
        "criteria": ["Inspect covering remnants and plausible estimated anatomy",
                     "Inspect visible appearance and seams with the unchanged local Auto restoration policy",
                     "Record failures per family; hair remains an automatic miss, and near-hidden sources remain excluded"],
        "training": False, "application_checkpoint_selection": False, "historical_gates_unchanged": True}
    if len(cases) != 10 or any(case["family"] == "visibility_rejection" for case in cases):
        raise ValueError("Require exact ten-case non-rejection diagnostic")
    nonempty = sum(bool((pixels(ROOT / path, "L") == 255).any()) for case in cases for path in case["masks"].values())
    if nonempty != 18:
        raise ValueError("Expected eighteen completion masks and two hair bypasses")
    OUT.mkdir(); write(OUT / "frozen_protocol.json", protocol)
    print(json.dumps({"protocol_sha256": sha(OUT / "frozen_protocol.json"), "generation_budget": 20}))


def run():
    if (OUT / "execution.json").exists():
        raise ValueError("Preserve completed/partial generation")
    p = read(OUT / "frozen_protocol.json")
    for name, pin in p["assets_sha256"].items():
        if sha(ROOT / name) != pin:
            raise ValueError("Frozen downstream asset changed")
    engine = PaletteFaceWorkflow(device="cpu"); engine._runtime(); engine._generator(); engine._restorer()
    if engine.detector is not None:
        raise ValueError("Detector must remain unloaded; use cached proposals")
    before = {n: state_sha(m) for n, m in (("completion", engine.generator), ("restoration", engine.restorer))}
    counter = {"completion": 0, "restoration": 0}; captured = {}
    def hook(kind):
        def save(module, args, result):
            counter[kind] += 1; captured[kind] = result.detach().cpu().numpy().copy()
        return save
    handles = [engine.generator.register_forward_hook(hook("completion")), engine.restorer.register_forward_hook(hook("restoration"))]
    write(OUT / "execution.json", {"protocol_sha256": sha(OUT / "frozen_protocol.json"), "state_before": before,
          "device": "cpu", "optimizer_constructed": False, "budget": p["budget"]})
    for folder in ("outputs", "stages"):
        (OUT / folder).mkdir()
    started, rows, hashes = time.monotonic(), [], {}
    for case in p["cases"]:
        image = pixels(ROOT / case["input"])
        for arm in ARMS:
            if time.monotonic() - started > 300:
                raise ValueError("Frozen downstream wall budget exceeded")
            captured.clear(); mask = (pixels(ROOT / case["masks"][arm], "L") == 255).astype(np.uint8)
            output, metadata = engine.generate(image, mask, "auto")
            metadata["mask_source"] = "frozen candidate automatic proposal; no manual edits"
            filename = f"outputs/{arm}_{case['id']}.png"
            Image.fromarray(output).save(OUT / filename); hashes[filename] = sha(OUT / filename)
            stages = {}
            for kind, value in captured.items():
                name = f"stages/{arm}_{case['id']}_{kind}.npy"
                np.save(OUT / name, value, allow_pickle=False); hashes[name] = sha(OUT / name); stages[kind] = name
            rows.append({"id": case["id"], "arm": arm, "mask": case["masks"][arm], "output": filename,
                         "output_sha256": hashes[filename], "stage_arrays": stages, "metadata": metadata, "hidden_ground_truth": None})
            print(arm + "/" + case["id"] + f": {metadata['elapsed_seconds']:.2f}s", flush=True)
    elapsed = time.monotonic() - started
    after = {n: state_sha(m) for n, m in (("completion", engine.generator), ("restoration", engine.restorer))}
    if before != after or engine.detector is not None or counter["completion"] != 18 or counter["restoration"] > 20 or elapsed > 300:
        raise ValueError("Frozen inference budget/state differs")
    for handle in handles:
        handle.remove()
    canvas = Image.new("RGB", (1280, 32 + 281 * len(p["cases"])), "#16181c"); draw = ImageDraw.Draw(canvas)
    for col, label in enumerate(("input", "existing91 area", "existing91 estimate", "varied133 area", "varied133 estimate")):
        draw.text((256 * col + 4, 7), label, fill="white")
    for line, case in enumerate(p["cases"]):
        y = 32 + line * 281; draw.text((3, y), case["id"], fill="white"); original = pixels(ROOT / case["input"])
        panels = [original]
        for arm in ARMS:
            mask = pixels(ROOT / case["masks"][arm], "L") == 255; marked = original.copy()
            marked[mask] = (.55 * original[mask] + .45 * np.array([16, 185, 129])).round().astype(np.uint8)
            row = next(r for r in rows if r["id"] == case["id"] and r["arm"] == arm)
            panels.extend([marked, pixels(OUT / row["output"])])
        for col, panel in enumerate(panels):
            canvas.paste(Image.fromarray(panel), (256 * col, y + 19))
    canvas.save(OUT / "preview.png"); hashes["preview.png"] = sha(OUT / "preview.png")
    write(OUT / "results.json", {"complete": True, "date": "2026-10-03", "protocol_sha256": sha(OUT / "frozen_protocol.json"),
          "rows": rows, "forwards": counter, "generation_requests": 20, "detector_forwards": 0,
          "seconds_excluding_loading": elapsed, "state_before": before, "state_after": after,
          "artifacts_sha256": hashes, "optimizer_updates": 0, "promoted": False, "visual_review_pending": True})
    print(json.dumps({"complete": True, "forwards": counter, "seconds": elapsed, "results_sha256": sha(OUT / "results.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args(); prepare() if args.prepare else run()
