"""Frozen LaMa resolution comparison on reviewed native masks; inference only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lama_completion import load_lama


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def pixels(path, mode="RGB"):
    with Image.open(path) as image:
        return np.asarray(image.convert(mode)).copy()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def state_sha(net):
    result = hashlib.sha256()
    for name, tensor in sorted(net.state_dict().items()):
        result.update(name.encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "outputs/practical_lama_protocol_v1.json")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs/practical_lama_outputs_v1")
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if not destination.is_relative_to(ROOT) or destination.exists():
        raise ValueError("Preserve prior comparisons; use a new output directory")
    protocol_digest = sha(args.protocol)
    p = json.loads(args.protocol.read_text(encoding="utf-8"))
    if p["status"] != "frozen_before_generation" or p["format"] != "dgp-practical-lama-comparison-v1":
        raise ValueError("Expected a frozen comparison protocol")
    for name, value in p["artifact_sha256"].items():
        if sha(ROOT / name) != value:
            raise ValueError(f"Frozen artifact changed: {name}")
    native_path = ROOT / p["native_protocol"]
    native = json.loads(native_path.read_text(encoding="utf-8"))
    gallery = ROOT / native["gallery_root"]
    if [case["id"] for case in native["cases"]] != p["case_ids"]:
        raise ValueError("Frozen cohort differs")
    for name, value in native["assets_sha256"].items():
        if sha(gallery / name) != value:
            raise ValueError(f"Frozen gallery changed: {name}")
    baseline = ROOT / p["baseline_root"]
    old = json.loads((baseline / "results.json").read_text(encoding="utf-8"))
    reused = {row["id"]: row for row in old["rows"] if row["arm"] == "reviewed"}
    for row in reused.values():
        if sha(baseline / row["output"]) != row["output_sha256"]:
            raise ValueError("Previously verified CodeFormer output changed")
    torch.set_num_threads(p["threads"])
    destination.mkdir()
    for arm in p["arms"]:
        (destination / arm).mkdir()
    (destination / "preview").mkdir()
    write_json(destination / "execution.json", {
        "protocol_sha256": protocol_digest, "torch": str(torch.__version__),
        "device": "cpu", "threads": p["threads"], "cuda_available": torch.cuda.is_available(),
        "artifact_sha256": p["artifact_sha256"], "optimizer_constructed": False, "optimizer_updates": 0,
    })
    tick = time.monotonic()
    print("Loading prepared Big-LaMa; reviewed masks only; no training...", flush=True)
    model, provenance = load_lama(ROOT / p["generator"], "cpu")
    before = state_sha(model)
    rows, failures, requests, forwards = [], [], 0, 0
    for number, case in enumerate(native["cases"]):
        image = pixels(gallery / case["input"])
        removal = pixels(gallery / case["removal_proposal"], "L")
        if image.shape != (256, 256, 3) or removal.shape != (256, 256) or not np.isin(removal, (0, 255)).all():
            raise ValueError("Unexpected input/mask format")
        selected = removal == 255
        x = torch.from_numpy(image).permute(2, 0, 1).float()[None] / 255
        mask = torch.from_numpy(selected.copy()).float()[None, None]
        for arm in p["arms"]:
            requests += 1
            started = time.monotonic()
            try:
                if arm == "lama_native256":
                    result = model(x, mask)
                elif arm == "lama_visible_resize512":
                    support = F.interpolate(1 - mask, (512, 512), mode="bilinear", align_corners=False)
                    resized = F.interpolate(x * (1 - mask), (512, 512), mode="bilinear", align_corners=False)
                    resized = (resized / support.clamp_min(1e-8)).clamp(0, 1)
                    larger_mask = F.interpolate(mask, (512, 512), mode="nearest")
                    generated = model(resized, larger_mask)
                    generated = F.interpolate(generated, (256, 256), mode="bilinear", align_corners=False)
                    result = torch.where(mask.bool(), generated, x)
                else:
                    raise ValueError("Unknown frozen arm")
                array = (result[0].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
                if not np.array_equal(array[~selected], image[~selected]):
                    raise ValueError("Visible pixels changed outside reviewed mask")
                filename = destination / arm / f"{case['id']}.png"
                Image.fromarray(array).save(filename)
                forwards += int(selected.any())
                rows.append({"id": case["id"], "family": case["family"], "arm": arm,
                             "output": filename.relative_to(destination).as_posix(), "output_sha256": sha(filename),
                             "mask_pixels": int(selected.sum()), "empty_mask_bypass": not bool(selected.any()),
                             "exact_outside_reviewed_mask": True, "hidden_face_mae": None,
                             "seconds": time.monotonic() - started})
                print(f"{number + 1}/10 {arm}: {rows[-1]['seconds']:.2f}s", flush=True)
            except Exception as error:
                failures.append({"id": case["id"], "arm": arm, "error": str(error)})
                print(json.dumps(failures[-1]), flush=True)
        write_json(destination / "progress.json", {"requests": requests, "rows": rows, "failures": failures, "complete": False})
    after = state_sha(model)
    if before != after or sha(args.protocol) != protocol_digest:
        raise ValueError("Generator/protocol changed during inference")
    if not failures:
        canvas = Image.new("RGB", (1280, 2830), "#16181c")
        draw = ImageDraw.Draw(canvas)
        for column, text in enumerate(("original", "reviewed removal", "CodeFormer baseline", "LaMa native256", "LaMa resize512")):
            draw.text((column * 256 + 10, 10), text, fill="white")
        for number, case in enumerate(native["cases"]):
            rgb = pixels(gallery / case["input"])
            selected = pixels(gallery / case["removal_proposal"], "L") == 255
            overlay = rgb.astype(np.float32)
            overlay[selected] = .55 * overlay[selected] + .45 * np.array([255, 120, 20])
            tiles = [rgb, overlay.round().astype(np.uint8), pixels(baseline / reused[case["id"]]["output"])]
            tiles.extend(pixels(destination / arm / f"{case['id']}.png") for arm in p["arms"])
            y = 42 + number * 276
            for column, tile in enumerate(tiles):
                canvas.paste(Image.fromarray(tile), (column * 256, y))
            draw.text((10, y + 258), case["id"], fill="white")
        canvas.save(destination / "preview.png")
        for start in (0, 5):
            canvas.crop((0, 42 + start * 276, 1280, 42 + (start + 5) * 276)).save(destination / "preview" / f"rows_{start + 1:02d}_{start + 5:02d}.png")
    report = {
        "complete": requests == 20 and not failures, "date": "2026-10-02",
        "protocol_sha256": protocol_digest, "requests": requests, "nonempty_generator_forwards": forwards,
        "optimizer_constructed": False, "optimizer_updates": 0, "new_detector_forwards": 0,
        "reused_codeformer_rows": 10, "generator_state_before": before, "generator_state_after": after,
        "state_unchanged": before == after, "provenance": provenance,
        "elapsed_seconds": time.monotonic() - tick, "rows": rows, "failures": failures,
        "hidden_face_ground_truth": False, "hidden_face_mae": None, "promoted": False,
        "scope": native["scope"], "missing_families": native["missing_families"],
        "preview_sha256": sha(destination / "preview.png") if not failures else None,
    }
    write_json(destination / "results.json", report)
    print(json.dumps({"complete": report["complete"], "requests": requests, "nonempty_forwards": forwards,
                      "failures": len(failures), "seconds": report["elapsed_seconds"]}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
