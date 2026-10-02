"""Compare a separate face restorer on frozen completed crops, without training."""
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
from pretrained_face_restoration import load_face_restorer
from scripts.run_practical_restoration_comparison import quantize, tensor, visible_metrics
from scripts.run_practical_lama_comparison import pixels, sha, state_sha, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "outputs/practical_face_restoration_v1/frozen_protocol.json")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs/practical_face_restoration_outputs_v1")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT) or out.exists():
        raise ValueError("Use a new output folder and preserve prior evidence")
    digest = sha(args.protocol)
    p = json.loads(args.protocol.read_text(encoding="utf-8"))
    if p["format"] != "dgp-practical-face-restoration-v1" or p["status"] != "frozen_before_generation":
        raise ValueError("Expected a frozen face-restoration protocol")
    for collection in (p["artifact_sha256"], p["asset_sha256"]):
        for name, expected in collection.items():
            if sha(ROOT / name) != expected:
                raise ValueError(f"Frozen source changed: {name}")
    torch.set_num_threads(p["threads"])
    out.mkdir()
    for name in (*p["arms"], "raw_restoration", "preview"):
        (out / name).mkdir()
    print("Loading separate CodeFormer restoration weights; no training...", flush=True)
    model, provenance = load_face_restorer(ROOT / p["weights"], "cpu")
    before = state_sha(model)
    write_json(out / "execution.json", {"protocol_sha256": digest, "provenance": provenance,
               "torch": str(torch.__version__), "device": "cpu", "threads": p["threads"],
               "optimizer_constructed": False, "optimizer_updates": 0, "state_before": before})
    started, rows, failures, raw, forwards = time.monotonic(), [], [], {}, 0
    for case in p["cases"]:
        tick = time.monotonic()
        try:
            baseline = pixels(ROOT / case["cached_completion"])
            original = pixels(ROOT / case["input"])
            reference = pixels(ROOT / case["reference"])
            mask_pixels = pixels(ROOT / case["removal"], "L")
            if baseline.shape != (256, 256, 3) or not np.isin(mask_pixels, (0, 255)).all():
                raise ValueError("Unexpected crop or mask")
            selected = mask_pixels == 255
            if not np.array_equal(baseline[~selected], original[~selected]):
                raise ValueError("Cached completion outside active mask differs")
            restored = {}
            for weight in p["fidelity_weights"]:
                image = quantize(model(tensor(baseline), fidelity=weight))
                forwards += 1
                restored[weight] = image
                path = out / "raw_restoration" / f"{case['id']}_w{weight}.png"
                Image.fromarray(image).save(path)
                raw[path.relative_to(out).as_posix()] = sha(path)
            for arm, spec in p["arms"].items():
                if arm == "off":
                    array = baseline.copy()
                else:
                    blend = spec["visible_blend"]
                    array = ((1-blend)*baseline.astype(np.float32)+blend*restored[spec["fidelity"]].astype(np.float32)).round().astype(np.uint8)
                    array[selected] = baseline[selected]
                if not np.array_equal(array[selected], baseline[selected]):
                    raise ValueError("Restoration changed completed facial pixels")
                path = out / arm / f"{case['id']}.png"
                Image.fromarray(array).save(path)
                rows.append({"id": case["id"], "arm": arm, "family": case["family"],
                             "synthetically_degraded": case["synthetically_degraded"],
                             "output": path.relative_to(out).as_posix(), "output_sha256": sha(path),
                             "mask_pixels": int(selected.sum()), **visible_metrics(array, reference, selected),
                             "hole_changed_pixels": 0, "hidden_face_mae": None})
            print(f"{case['id']}: {time.monotonic()-tick:.2f}s", flush=True)
        except Exception as error:
            failures.append({"id": case["id"], "error": str(error)})
            print(json.dumps(failures[-1]), flush=True)
        write_json(out / "progress.json", {"complete": False, "rows": rows, "failures": failures, "forwards": forwards})
    after = state_sha(model)
    if before != after or sha(args.protocol) != digest:
        raise ValueError("Model state or frozen protocol changed")
    aggregates = {}
    for degraded in (False, True):
        key = "degraded" if degraded else "native"
        aggregates[key] = {}
        for arm in p["arms"]:
            group = [r for r in rows if r["synthetically_degraded"] == degraded and r["arm"] == arm]
            aggregates[key][arm] = {"cases": len(group), "mean_visible_mae": float(np.mean([r["visible_mae"] for r in group])) if group else None}
    previews = {}
    if not failures:
        indexed = {(r["id"], r["arm"]): r for r in rows}
        for degraded in (False, True):
            chosen = [c for c in p["cases"] if c["synthetically_degraded"] == degraded]
            canvas = Image.new("RGB", (1792, 2830), "#16181c")
            draw = ImageDraw.Draw(canvas)
            for column, title in enumerate(("visible reference", "input", *p["arms"])):
                draw.text((column * 256 + 5, 10), title, fill="white")
            for index, case in enumerate(chosen):
                tiles = [pixels(ROOT / case["reference"]), pixels(ROOT / case["input"])]
                tiles.extend(pixels(out / indexed[(case["id"], arm)]["output"]) for arm in p["arms"])
                y = 42 + index * 276
                for column, tile in enumerate(tiles):
                    canvas.paste(Image.fromarray(tile), (column * 256, y))
                draw.text((10, y+258), case["id"], fill="white")
            name = "degraded" if degraded else "native"
            path = out / f"{name}_preview.png"
            canvas.save(path)
            previews[path.name] = sha(path)
            for start in (0, 5):
                canvas.crop((0,42+start*276,1792,42+(start+5)*276)).save(out / "preview" / f"{name}_{start+1:02d}_{start+5:02d}.png")
    report = {"complete": len(rows) == p["planned_outputs"] and not failures, "date": "2026-10-02",
              "protocol_sha256": digest, "elapsed_seconds": time.monotonic()-started,
              "new_restoration_forwards": forwards, "reused_completed_crops": len(p["cases"]),
              "new_completion_forwards": 0, "new_detector_forwards": 0,
              "optimizer_constructed": False, "optimizer_updates": 0,
              "state_before": before, "state_after": after, "state_unchanged": before == after,
              "aggregates": aggregates, "rows": rows, "failures": failures,
              "raw_restoration_sha256": raw, "preview_sha256": previews,
              "hidden_face_ground_truth": False, "promoted": False,
              "reference_scope": p["reference_scope"], "missing_families": p["missing_families"]}
    write_json(out / "results.json", report)
    print(json.dumps({"complete": report["complete"], "outputs": len(rows), "restoration_forwards": forwards,
                      "seconds": report["elapsed_seconds"], "aggregates": aggregates}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
