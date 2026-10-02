"""Frozen native gallery: pretrained completion only, no local training.

Compare two previously audited automatic detector masks with reviewed removal
masks on exactly the same inputs. No hidden-face target or oracle routing is used.
"""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pretrained_completion import load_codeformer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def image(path):
    with Image.open(path) as value:
        return np.asarray(value.convert("RGB")).copy()


def mask(path):
    with Image.open(path) as value:
        array = np.asarray(value).copy()
    if array.shape != (256, 256) or not np.isin(array, [0, 255]).all():
        raise ValueError(f"Invalid mask: {path}")
    return array == 255


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "outputs/practical_gallery_v2/frozen_native_protocol_v1.json")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs/practical_native_outputs_v1")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError("Preserve existing runs; use a new output directory")
    protocol_sha = sha(args.protocol)
    p = json.loads(args.protocol.read_text(encoding="utf-8"))
    if p["format"] != "dgp-practical-native-output-protocol-v1" or not p["gallery_frozen"] or len(p["cases"]) != 10:
        raise ValueError("Expected a frozen ten-source native protocol")
    folder = ROOT / p["gallery_root"]
    for path, digest in p["assets_sha256"].items():
        if sha(folder / path) != digest:
            raise ValueError(f"Gallery artifact changed: {path}")
    for value in p["models"].values():
        if sha(ROOT / value["path"]) != value["sha256"]:
            raise ValueError("Frozen checkpoint changed")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA is unavailable")
    torch.set_num_threads(args.threads)
    output.mkdir(parents=True)
    for arm in p["arms"]:
        (output / arm).mkdir()
    (output / "preview").mkdir()
    code_paths = ("scripts/run_practical_native_gallery.py", "pretrained_completion.py",
                  "completion.py", "third_party/codeformer/codeformer_arch.py")
    write_json(output / "execution.json", {
        "protocol": str(args.protocol), "protocol_sha256": protocol_sha,
        "code_sha256": {name: sha(ROOT / name) for name in code_paths},
        "torch": str(torch.__version__), "device": args.device,
        "cuda_available": torch.cuda.is_available(), "threads": args.threads,
        "optimizer_constructed": False, "optimizer_updates": 0,
        "scope": p["scope"], "new_detector_forwards": 0,
    })
    started = time.monotonic()
    print("Loading verified CodeFormer inpainting weights...", flush=True)
    generator, provenance = load_codeformer(ROOT / p["models"]["generator"]["path"], args.device)
    before = state_digest(generator)
    rows, requests, forwards, failures = [], 0, 0, []
    for position, case in enumerate(p["cases"]):
        rgb = image(folder / case["input"])
        allowed = mask(folder / case["removal_proposal"])
        target = mask(folder / case["label_mask"])
        if rgb.shape != (256, 256, 3) or case["synthetically_degraded"]:
            raise ValueError("Native cached-mask run requires exact undegraded crops")
        x = torch.from_numpy(rgb).permute(2, 0, 1).float()[None].to(args.device) / 255
        completed = {}
        for arm in p["arms"]:
            selected = allowed if arm == "reviewed" else mask(folder / case["cached_original_masks"][arm])
            m = torch.from_numpy(selected.copy()).float()[None, None].to(args.device)
            tick = time.monotonic()
            requests += 1
            try:
                with torch.inference_mode():
                    result = generator(x, m)
                pixels = (result[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                if not np.array_equal(pixels[~selected], rgb[~selected]):
                    raise ValueError("Observed bytes outside the active removal mask changed")
                elapsed = time.monotonic() - tick
                forwards += int(selected.any())
                path = output / arm / f"{case['id']}.png"
                Image.fromarray(pixels).save(path)
                difference = np.abs(pixels.astype(np.float32) - rgb.astype(np.float32)) / 255
                row = {
                    "id": case["id"], "family": case["family"], "arm": arm,
                    "output": path.relative_to(output).as_posix(), "output_sha256": sha(path),
                    "seconds": elapsed, "mask_pixels": int(selected.sum()),
                    "empty_mask_bypass": not bool(selected.any()),
                    "covered_input_with_empty_predicted_mask": bool(target.any() and not selected.any()),
                    "exact_outside_own_mask": True,
                    "visible_change_outside_reviewed_area": float(difference[~allowed].mean()),
                    "label_coverage_fraction": float((selected & target).sum() / target.sum()) if target.any() else None,
                    "pixels_outside_original_label": int((selected & ~target).sum()),
                    "hidden_face_mae": None, "inference_failed": False,
                }
                completed[arm] = pixels
                rows.append(row)
                print(f"{position + 1}/10 {arm} {case['id']}: {elapsed:.2f}s, mask={row['mask_pixels']}px", flush=True)
            except Exception as error:
                failure = {"id": case["id"], "arm": arm, "error": str(error)}
                failures.append(failure)
                print(json.dumps({"failure": failure}), flush=True)
        write_json(output / "progress.json", {"finished_sources": position + 1,
                   "requests": requests, "nonempty_generator_forwards": forwards,
                   "rows": rows, "failures": failures, "complete": False})
    after = state_digest(generator)
    if before != after or sha(args.protocol) != protocol_sha:
        raise ValueError("Frozen source state or protocol changed during inference")
    if not failures:
        columns = ("original", "parent output", "candidate output", "reviewed removal", "reviewed output")
        canvas = Image.new("RGB", (1280, 2830), "#16181c")
        draw = ImageDraw.Draw(canvas)
        for column, title in enumerate(columns):
            draw.text((column * 256 + 10, 10), title, fill="white")
        for position, case in enumerate(p["cases"]):
            rgb = image(folder / case["input"])
            allowed = mask(folder / case["removal_proposal"])
            colored = rgb.copy().astype(np.float32)
            colored[allowed] = .55 * colored[allowed] + .45 * np.array([255, 120, 20])
            tiles = [rgb, image(output / "parent" / f"{case['id']}.png"),
                     image(output / "candidate" / f"{case['id']}.png"),
                     colored.round().astype(np.uint8), image(output / "reviewed" / f"{case['id']}.png")]
            y = 42 + position * 276
            for column, pixels in enumerate(tiles):
                canvas.paste(Image.fromarray(pixels), (column * 256, y))
            draw.text((10, y + 258), case["id"], fill="white")
        canvas.save(output / "preview.png")
        for start in (0, 5):
            canvas.crop((0, 42 + start * 276, 1280, 42 + (start + 5) * 276)).save(output / "preview" / f"rows_{start + 1:02d}_{start + 5:02d}.png")
    report = {
        "complete": requests == 30 and not failures, "protocol_sha256": protocol_sha,
        "date": "2026-10-02", "sources": 10, "generator_requests": requests,
        "nonempty_generator_forwards": forwards, "new_detector_forwards": 0,
        "optimizer_constructed": False, "optimizer_updates": 0,
        "generator_state_before": before, "generator_state_after": after,
        "state_unchanged": before == after, "provenance": provenance,
        "elapsed_seconds": time.monotonic() - started,
        "rows": rows, "failures": failures, "hidden_face_mae": None,
        "promoted": False, "scope": p["scope"], "missing_families": p["missing_families"],
        "preview_sha256": sha(output / "preview.png") if not failures else None,
    }
    write_json(output / "results.json", report)
    print(json.dumps({"complete": report["complete"], "requests": requests,
                      "nonempty_forwards": forwards, "failures": len(failures),
                      "seconds": report["elapsed_seconds"], "output": str(output)}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
