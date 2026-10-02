"""Frozen visible-region restoration comparison; inference only, no hidden target."""
import argparse
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
from completion_inference import load_restorer
from pretrained_completion import load_codeformer
from scripts.run_practical_lama_comparison import pixels, sha, state_sha, write_json


def quantize(tensor):
    return (tensor[0].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)


def tensor(array):
    return torch.from_numpy(array.copy()).permute(2, 0, 1).float()[None] / 255


def input_statistics(rgb, selected):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    support = cv2.erode((~selected).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    support[:4] = False
    support[-4:] = False
    support[:, :4] = False
    support[:, -4:] = False
    if not support.any():
        return {"visible_support_pixels": 0, "smoothed_laplacian_variance": None}
    smooth = cv2.GaussianBlur(gray, (3, 3), .6)
    laplacian = cv2.Laplacian(smooth, cv2.CV_32F)
    return {"visible_support_pixels": int(support.sum()),
            "smoothed_laplacian_variance": float(laplacian[support].var())}


def visible_metrics(output, reference, selected):
    difference = (output.astype(np.float64) - reference.astype(np.float64)) / 255
    visible = difference[~selected]
    mse = float(np.square(visible).mean())
    return {"visible_mae": float(np.abs(visible).mean()), "visible_mse": mse,
            "visible_psnr": float(-10 * np.log10(mse)) if mse > 0 else None,
            "psnr_infinite": mse == 0, "visible_pixels": int((~selected).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "outputs/practical_restoration_v1/frozen_protocol.json")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs/practical_restoration_outputs_v1")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT) or out.exists():
        raise ValueError("Use a new output folder; preserve prior evidence")
    digest = sha(args.protocol)
    p = json.loads(args.protocol.read_text(encoding="utf-8"))
    if p["format"] != "dgp-practical-restoration-comparison-v1" or p["status"] != "frozen_before_generation":
        raise ValueError("Expected the frozen restoration protocol")
    for collection in (p["artifact_sha256"], p["asset_sha256"]):
        for name, expected in collection.items():
            if sha(ROOT / name) != expected:
                raise ValueError(f"Frozen file changed: {name}")
    torch.set_num_threads(p["threads"])
    out.mkdir()
    for directory in (*p["arms"], "intermediates", "preview"):
        (out / directory).mkdir()
    print("Loading pinned completion and Phase 3 restoration; CPU inference only...", flush=True)
    generator, provenance = load_codeformer(ROOT / p["completion_weights"], "cpu")
    restorer = load_restorer(ROOT / p["restoration_weights"], "cpu")
    states_before = {"completion": state_sha(generator), "restoration": state_sha(restorer)}
    write_json(out / "execution.json", {"protocol_sha256": digest, "torch": str(torch.__version__),
               "device": "cpu", "threads": p["threads"], "optimizer_constructed": False, "optimizer_updates": 0,
               "states_before": states_before, "completion_provenance": provenance})
    rows, failures, statistics, intermediates = [], [], [], {}
    calls = {"new_completion_requests": 0, "new_nonempty_completion_forwards": 0,
             "new_empty_completion_bypasses": 0, "reused_native_completion_outputs": 0,
             "restoration_forwards": 0, "detector_forwards": 0, "optimizer_updates": 0}
    started = time.monotonic()
    with torch.inference_mode():
        for case in p["cases"]:
            tick = time.monotonic()
            try:
                rgb = pixels(ROOT / case["input"])
                reference = pixels(ROOT / case["reference"])
                selected = pixels(ROOT / case["removal"], "L") == 255
                if rgb.shape != reference.shape or rgb.shape != (256, 256, 3):
                    raise ValueError("Unexpected practical crop dimensions")
                x, mask = tensor(rgb), torch.from_numpy(selected.copy()).float()[None, None]
                if case["cached_native_completion"]:
                    baseline = pixels(ROOT / case["cached_native_completion"])
                    calls["reused_native_completion_outputs"] += 1
                else:
                    baseline = quantize(generator(x, mask))
                    calls["new_completion_requests"] += 1
                    calls["new_nonempty_completion_forwards"] += int(selected.any())
                    calls["new_empty_completion_bypasses"] += int(not selected.any())
                if not np.array_equal(baseline[~selected], rgb[~selected]):
                    raise ValueError("Completion changed observed input pixels")
                before = restorer(x)
                calls["restoration_forwards"] += 1
                pre = quantize(generator(before, mask))
                calls["new_completion_requests"] += 1
                calls["new_nonempty_completion_forwards"] += int(selected.any())
                calls["new_empty_completion_bypasses"] += int(not selected.any())
                post = quantize(restorer(tensor(baseline)))
                calls["restoration_forwards"] += 1
                pre_rgb = quantize(before)
                if not np.array_equal(pre[~selected], pre_rgb[~selected]):
                    raise ValueError("Legacy completion changed its restored visible base")
                for name, array in (("pre_dgp", pre_rgb), ("post_dgp", post)):
                    path = out / "intermediates" / f"{case['id']}_{name}.png"
                    Image.fromarray(array).save(path)
                    intermediates[path.relative_to(out).as_posix()] = sha(path)
                post_visible = post.copy()
                post_visible[selected] = baseline[selected]
                quarter = (.75 * baseline.astype(np.float32) + .25 * post.astype(np.float32)).round().astype(np.uint8)
                quarter[selected] = baseline[selected]
                outputs = {"completion_off": baseline, "legacy_pre": pre,
                           "visible_post": post_visible, "visible_post25": quarter}
                statistics.append({"id": case["id"], "synthetically_degraded": case["synthetically_degraded"],
                                   **input_statistics(rgb, selected)})
                for arm, array in outputs.items():
                    if arm.startswith("visible_post") and not np.array_equal(array[selected], baseline[selected]):
                        raise ValueError("Visible-only restoration changed completed facial pixels")
                    path = out / arm / f"{case['id']}.png"
                    Image.fromarray(array).save(path)
                    rows.append({"id": case["id"], "family": case["family"], "arm": arm,
                                 "synthetically_degraded": case["synthetically_degraded"],
                                 "output": path.relative_to(out).as_posix(), "output_sha256": sha(path),
                                 "mask_pixels": int(selected.sum()), **visible_metrics(array, reference, selected),
                                 "outside_mask_changed_pixels_from_input": int(np.any(array != rgb, axis=2)[~selected].sum()),
                                 "hole_changed_pixels_from_completion_off": int(np.any(array != baseline, axis=2)[selected].sum()),
                                 "hidden_face_mae": None})
                print(f"{case['id']}: {time.monotonic()-tick:.2f}s; off/pre/post25 visible MAE="
                      f"{rows[-4]['visible_mae']:.4f}/{rows[-3]['visible_mae']:.4f}/{rows[-1]['visible_mae']:.4f}", flush=True)
            except Exception as error:
                failures.append({"id": case["id"], "error": str(error)})
                print(json.dumps(failures[-1]), flush=True)
            write_json(out / "progress.json", {"complete": False, "rows": rows, "failures": failures, "calls": calls})
    states_after = {"completion": state_sha(generator), "restoration": state_sha(restorer)}
    if states_before != states_after or sha(args.protocol) != digest:
        raise ValueError("Frozen state or protocol changed")
    aggregates = {}
    for degraded in (False, True):
        key = "degraded" if degraded else "native"
        aggregates[key] = {}
        for arm in p["arms"]:
            group = [row for row in rows if row["synthetically_degraded"] == degraded and row["arm"] == arm]
            aggregates[key][arm] = {"cases": len(group), "mean_visible_mae": float(np.mean([row["visible_mae"] for row in group])) if group else None,
                                    "hidden_face_mae": None}
    if not failures:
        indexed = {(row["id"], row["arm"]): row for row in rows}
        for degraded in (False, True):
            chosen = [case for case in p["cases"] if case["synthetically_degraded"] == degraded]
            canvas = Image.new("RGB", (1536, 2830), "#16181c")
            draw = ImageDraw.Draw(canvas)
            for column, title in enumerate(("visible reference", "input", "completion off", "legacy pre", "visible post", "visible post25")):
                draw.text((column * 256 + 5, 10), title, fill="white")
            for index, case in enumerate(chosen):
                tiles = [pixels(ROOT / case["reference"]), pixels(ROOT / case["input"])]
                tiles.extend(pixels(out / indexed[(case["id"], arm)]["output"]) for arm in p["arms"])
                y = 42 + index * 276
                for column, tile in enumerate(tiles):
                    canvas.paste(Image.fromarray(tile), (column * 256, y))
                draw.text((10, y + 258), case["id"], fill="white")
            name = "degraded" if degraded else "native"
            canvas.save(out / f"{name}_preview.png")
            for start in (0, 5):
                canvas.crop((0, 42 + start * 276, 1536, 42 + (start + 5) * 276)).save(out / "preview" / f"{name}_{start+1:02d}_{start+5:02d}.png")
    report = {"complete": len(rows) == p["planned_saved_outputs"] and not failures, "date": "2026-10-02",
              "protocol_sha256": digest, "elapsed_seconds": time.monotonic() - started, "calls": calls,
              "states_before": states_before, "states_after": states_after, "state_unchanged": states_before == states_after,
              "optimizer_constructed": False, "optimizer_updates": 0, "aggregates": aggregates,
              "rows": rows, "input_statistics": statistics, "intermediate_sha256": intermediates,
              "failures": failures, "hidden_face_ground_truth": False, "promoted": False,
              "reference_scope": "Known original visible pixels outside the versioned operator removal footprint only",
              "missing_families": p["missing_families"],
              "preview_sha256": {name: sha(out / name) for name in ("native_preview.png", "degraded_preview.png")} if not failures else {}}
    write_json(out / "results.json", report)
    print(json.dumps({"complete": report["complete"], "saved_outputs": len(rows), "calls": calls,
                      "seconds": report["elapsed_seconds"], "aggregates": aggregates}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
