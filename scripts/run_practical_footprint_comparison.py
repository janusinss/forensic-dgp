"""Compare versioned operator footprints without changing historical labels."""
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
from pretrained_completion import load_codeformer
from scripts.run_practical_lama_comparison import pixels, sha, state_sha, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "outputs/practical_footprints_v3/frozen_protocol.json")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs/practical_footprint_outputs_v3")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT) or out.exists():
        raise ValueError("Preserve existing evidence; use a fresh output directory")
    digest = sha(args.protocol)
    p = json.loads(args.protocol.read_text(encoding="utf-8"))
    if p["format"] != "dgp-practical-footprint-comparison-v3" or p["status"] != "frozen_before_generation":
        raise ValueError("Expected a frozen practical comparison")
    for name, expected in p["artifact_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Frozen artifact changed: {name}")
    for name, expected in p["asset_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Frozen proposal asset changed: {name}")
    source = json.loads((ROOT / p["source_protocol"]).read_text(encoding="utf-8"))
    for name, expected in source["assets_sha256"].items():
        if sha(ROOT / source["gallery_root"] / name) != expected:
            raise ValueError("Historical gallery changed")
    baselines = {}
    for backend, spec in p["backends"].items():
        run = ROOT / spec["previous_run"]
        result = json.loads((run / "results.json").read_text(encoding="utf-8"))
        baselines[backend] = {row["id"]: run / row["output"] for row in result["rows"] if row["arm"] == spec["previous_arm"]}
        for row in result["rows"]:
            if row["arm"] == spec["previous_arm"] and sha(run / row["output"]) != row["output_sha256"]:
                raise ValueError("Prior baseline pixels changed")
    torch.set_num_threads(p["threads"])
    out.mkdir()
    (out / "preview").mkdir()
    for backend in p["backends"]:
        (out / backend).mkdir()
    write_json(out / "execution.json", {"protocol_sha256": digest, "torch": str(torch.__version__),
               "device": "cpu", "threads": p["threads"], "artifact_sha256": p["artifact_sha256"],
               "optimizer_constructed": False, "optimizer_updates": 0})
    loaders = {"codeformer": load_codeformer, "aot": load_aot}
    tick, rows, failures, states, requests, forwards = time.monotonic(), [], [], {}, 0, 0
    for backend, spec in p["backends"].items():
        print(f"Loading {backend}; revised removal proposals; no training...", flush=True)
        model, provenance = loaders[backend](ROOT / spec["weights"], "cpu")
        before = state_sha(model)
        for case in p["cases"]:
            rgb = pixels(ROOT / case["input"])
            removal = pixels(ROOT / case["new_removal"], "L")
            protected = pixels(ROOT / case["preserved_region"], "L") == 255
            if rgb.shape != (256, 256, 3) or not np.isin(removal, (0, 255)).all():
                raise ValueError("Unexpected native input/mask")
            selected = removal == 255
            if (selected & protected).any():
                raise ValueError("Operator-preserved eyewear intersects removal")
            x = torch.from_numpy(rgb).permute(2, 0, 1).float()[None] / 255
            mask = torch.from_numpy(selected.copy()).float()[None, None]
            requests += 1
            started = time.monotonic()
            try:
                result = model(x, mask)
                array = (result[0].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
                if not np.array_equal(array[~selected], rgb[~selected]):
                    raise ValueError("Visible pixels changed outside active footprint")
                path = out / backend / f"{case['id']}.png"
                Image.fromarray(array).save(path)
                previous = pixels(baselines[backend][case["id"]])
                difference = np.abs(array.astype(np.float32) - previous.astype(np.float32)) / 255
                old = pixels(ROOT / case["old_removal"], "L") == 255
                row = {"id": case["id"], "family": case["family"], "backend": backend,
                       "output": path.relative_to(out).as_posix(), "output_sha256": sha(path),
                       "mask_pixels": int(selected.sum()), "empty_mask_bypass": not bool(selected.any()),
                       "exact_outside_own_mask": True, "protected_eyewear_changed_pixels": 0,
                       "mean_change_from_previous_in_added_area": float(difference[selected & ~old].mean()) if (selected & ~old).any() else None,
                       "hidden_face_mae": None, "seconds": time.monotonic() - started}
                rows.append(row)
                forwards += int(selected.any())
                print(f"{backend} {case['id']}: {row['seconds']:.2f}s, removal={row['mask_pixels']}px", flush=True)
            except Exception as error:
                failures.append({"id": case["id"], "backend": backend, "error": str(error)})
                print(json.dumps(failures[-1]), flush=True)
            write_json(out / "progress.json", {"requests": requests, "rows": rows, "failures": failures, "complete": False})
        after = state_sha(model)
        if before != after:
            raise ValueError("Generator state changed")
        states[backend] = {"before": before, "after": after, "unchanged": True, "provenance": provenance}
        del model
    if sha(args.protocol) != digest:
        raise ValueError("Frozen protocol changed")
    if not failures:
        canvas = Image.new("RGB", (1536, 1710), "#16181c")
        draw = ImageDraw.Draw(canvas)
        for column, title in enumerate(("source", "revised removal", "CodeFormer v2", "CodeFormer v3", "AOT v2", "AOT v3")):
            draw.text((column * 256 + 6, 10), title, fill="white")
        for index, case in enumerate(p["cases"]):
            rgb = pixels(ROOT / case["input"])
            selected = pixels(ROOT / case["new_removal"], "L") == 255
            overlay = rgb.astype(np.float32)
            overlay[selected] = .55 * overlay[selected] + .45 * np.array([255, 120, 20])
            tiles = [rgb, overlay.round().astype(np.uint8), pixels(baselines["codeformer"][case["id"]]),
                     pixels(out / "codeformer" / f"{case['id']}.png"), pixels(baselines["aot"][case["id"]]),
                     pixels(out / "aot" / f"{case['id']}.png")]
            y = 42 + index * 276
            for column, tile in enumerate(tiles):
                canvas.paste(Image.fromarray(tile), (column * 256, y))
            draw.text((10, y + 258), case["id"], fill="white")
        canvas.save(out / "preview.png")
        for start in (0, 3):
            canvas.crop((0, 42 + start * 276, 1536, 42 + (start + 3) * 276)).save(out / "preview" / f"rows_{start+1:02d}_{start+3:02d}.png")
    report = {"complete": requests == p["planned_requests"] and not failures, "date": "2026-10-02",
              "protocol_sha256": digest, "requests": requests, "nonempty_generator_forwards": forwards,
              "new_detector_forwards": 0, "optimizer_constructed": False, "optimizer_updates": 0,
              "reused_baseline_rows": len(p["cases"]) * len(p["backends"]), "states": states,
              "elapsed_seconds": time.monotonic() - tick, "rows": rows, "failures": failures,
              "hidden_face_ground_truth": False, "hidden_face_mae": None, "promoted": False,
              "scope": p["scope"], "missing_families": p["missing_families"],
              "preview_sha256": sha(out / "preview.png") if not failures else None}
    write_json(out / "results.json", report)
    print(json.dumps({"complete": report["complete"], "requests": requests, "nonempty_forwards": forwards,
                      "failures": len(failures), "seconds": report["elapsed_seconds"]}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
