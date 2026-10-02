"""Frozen face-specific AOT-GAN comparison, reviewed native masks, inference only."""
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
from aot_completion import load_aot
from scripts.run_practical_lama_comparison import pixels, sha, state_sha, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "outputs/practical_aot_protocol_v1.json")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs/practical_aot_outputs_v1")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT) or out.exists():
        raise ValueError("Preserve earlier comparisons; use a fresh directory")
    protocol_digest = sha(args.protocol)
    p = json.loads(args.protocol.read_text(encoding="utf-8"))
    if p["format"] != "dgp-practical-aot-comparison-v1" or p["status"] != "frozen_before_generation":
        raise ValueError("Expected a frozen protocol")
    for name, value in p["artifact_sha256"].items():
        if sha(ROOT / name) != value:
            raise ValueError(f"Artifact changed: {name}")
    native = json.loads((ROOT / p["native_protocol"]).read_text(encoding="utf-8"))
    gallery = ROOT / native["gallery_root"]
    if [case["id"] for case in native["cases"]] != p["case_ids"]:
        raise ValueError("Cohort differs")
    for name, expected in native["assets_sha256"].items():
        if sha(gallery / name) != expected:
            raise ValueError("Frozen gallery changed")
    reused = {}
    for backend, value in p["baselines"].items():
        folder = ROOT / value["root"]
        old = json.loads((folder / "results.json").read_text(encoding="utf-8"))
        reused[backend] = {row["id"]: folder / row["output"] for row in old["rows"] if row["arm"] == value["arm"]}
        for row in old["rows"]:
            if row["arm"] == value["arm"] and sha(folder / row["output"]) != row["output_sha256"]:
                raise ValueError("Cached baseline changed")
    torch.set_num_threads(p["threads"])
    out.mkdir()
    (out / "aot512").mkdir()
    (out / "preview").mkdir()
    write_json(out / "execution.json", {"protocol_sha256": protocol_digest,
               "torch": str(torch.__version__), "device": "cpu", "threads": p["threads"],
               "artifact_sha256": p["artifact_sha256"], "optimizer_constructed": False, "optimizer_updates": 0})
    tick = time.monotonic()
    print("Loading author-provided CelebA-HQ AOT-GAN; no training...", flush=True)
    model, provenance = load_aot(ROOT / p["generator"], "cpu")
    before = state_sha(model)
    rows, failures, requests, forwards = [], [], 0, 0
    for position, case in enumerate(native["cases"]):
        rgb = pixels(gallery / case["input"])
        removal = pixels(gallery / case["removal_proposal"], "L")
        if rgb.shape != (256, 256, 3) or removal.shape != (256, 256) or not np.isin(removal, (0, 255)).all():
            raise ValueError("Invalid gallery input/mask")
        mask = removal == 255
        image = torch.from_numpy(rgb).permute(2, 0, 1).float()[None] / 255
        m = torch.from_numpy(mask.copy()).float()[None, None]
        started = time.monotonic()
        requests += 1
        try:
            result = model(image, m)
            array = (result[0].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
            if not np.array_equal(array[~mask], rgb[~mask]):
                raise ValueError("Visible pixels outside reviewed mask changed")
            filename = out / "aot512" / f"{case['id']}.png"
            Image.fromarray(array).save(filename)
            forwards += int(mask.any())
            rows.append({"id": case["id"], "family": case["family"], "arm": "aot512",
                         "output": filename.relative_to(out).as_posix(), "output_sha256": sha(filename),
                         "mask_pixels": int(mask.sum()), "empty_mask_bypass": not bool(mask.any()),
                         "exact_outside_reviewed_mask": True, "hidden_face_mae": None,
                         "seconds": time.monotonic() - started})
            print(f"{position + 1}/10 {case['id']}: {rows[-1]['seconds']:.2f}s", flush=True)
        except Exception as error:
            failures.append({"id": case["id"], "error": str(error)})
            print(json.dumps(failures[-1]), flush=True)
        write_json(out / "progress.json", {"requests": requests, "rows": rows, "failures": failures, "complete": False})
    after = state_sha(model)
    if before != after or sha(args.protocol) != protocol_digest:
        raise ValueError("Model/protocol changed during inference")
    if not failures:
        canvas = Image.new("RGB", (1280, 2830), "#16181c")
        draw = ImageDraw.Draw(canvas)
        for column, title in enumerate(("original", "reviewed removal", "CodeFormer baseline", "LaMa native256", "AOT-GAN CelebA-HQ")):
            draw.text((column * 256 + 10, 10), title, fill="white")
        for position, case in enumerate(native["cases"]):
            rgb = pixels(gallery / case["input"])
            mask = pixels(gallery / case["removal_proposal"], "L") == 255
            overlay = rgb.astype(np.float32)
            overlay[mask] = .55 * overlay[mask] + .45 * np.array([255, 120, 20])
            tiles = [rgb, overlay.round().astype(np.uint8), pixels(reused["codeformer"][case["id"]]),
                     pixels(reused["lama"][case["id"]]), pixels(out / "aot512" / f"{case['id']}.png")]
            y = 42 + position * 276
            for column, tile in enumerate(tiles):
                canvas.paste(Image.fromarray(tile), (column * 256, y))
            draw.text((10, y + 258), case["id"], fill="white")
        canvas.save(out / "preview.png")
        for start in (0, 5):
            canvas.crop((0, 42 + start * 276, 1280, 42 + (start + 5) * 276)).save(out / "preview" / f"rows_{start + 1:02d}_{start + 5:02d}.png")
    report = {"complete": requests == 10 and not failures, "date": "2026-10-02",
              "protocol_sha256": protocol_digest, "requests": requests, "nonempty_generator_forwards": forwards,
              "reused_baseline_rows": 20, "new_detector_forwards": 0, "optimizer_constructed": False, "optimizer_updates": 0,
              "generator_state_before": before, "generator_state_after": after, "state_unchanged": before == after,
              "elapsed_seconds": time.monotonic() - tick, "provenance": provenance,
              "rows": rows, "failures": failures, "hidden_face_ground_truth": False, "hidden_face_mae": None,
              "promoted": False, "scope": native["scope"], "missing_families": native["missing_families"],
              "preview_sha256": sha(out / "preview.png") if not failures else None}
    write_json(out / "results.json", report)
    print(json.dumps({"complete": report["complete"], "requests": requests, "nonempty_forwards": forwards,
                      "failures": len(failures), "seconds": report["elapsed_seconds"]}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
