"""Inspect legacy native-space signal filters without any face/model inference."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from degradation import adaptive_cctv_denoise, detect_and_deinterlace_cctv, detect_and_smooth_mosaic, estimate_noise_sigma

BASE = ROOT / "outputs/cctv_native_development_v2"
OUT = ROOT / "outputs/cctv_signal_preprocessing_v1"
SOURCE_PIN = "973628b8d27241c8740e5224de2d1ab700e992dd90cd0e67fc80bb7187aa2129"
SUBSET_PIN = "c063985b258b808efaa897c88593cbdbcee2848d686d3d056219f8a8e555a98e"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, data):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False)+"\n")


def display(rgb):
    height, width = rgb.shape[:2]
    side = max(width, height)
    canvas = np.full((side, side, 3), 128, dtype=np.uint8)
    y, x = (side-height)//2, (side-width)//2
    canvas[y:y+height, x:x+width] = rgb
    return Image.fromarray(canvas).resize((160, 160), Image.Resampling.BILINEAR)


def main():
    if OUT.exists():
        raise ValueError("Preserve partial/completed preprocessing audit")
    if sha(ROOT / "degradation.py") != SOURCE_PIN or sha(BASE / "frozen_subset.json") != SUBSET_PIN:
        raise ValueError("Reviewed native source/filter code changed")
    subset = json.loads((BASE / "frozen_subset.json").read_text(encoding="utf-8"))
    cases = [row for row in subset["cases"] if row["role"] == "development"]
    if len(cases) != 24:
        raise ValueError("Require exact frozen development set")
    OUT.mkdir()
    write(OUT / "frozen_protocol.json", {"date": "2026-10-03", "runner_sha256": sha(Path(__file__)),
        "degradation_sha256": SOURCE_PIN, "subset_sha256": SUBSET_PIN, "cases": [c["id"] for c in cases],
        "filter_order": "Exact legacy /reconstruct native-space deinterlace -> mosaic smoothing -> adaptive denoise; no alignment or generation",
        "budget": {"native_cases": 24, "wall_seconds_after_import": 30, "model_forwards": 0, "optimizer_updates": 0},
        "reserved_inputs_used": False, "application_change": False,
        "interpretation": "Measured pixel changes/trigger diagnostics, not restoration error or proof that detected patterns are actual noise/interlacing/mosaic"})
    (OUT / "stages").mkdir()
    start, rows, files, flags = time.monotonic(), [], {}, Counter()
    for case in cases:
        if time.monotonic()-start > 30:
            raise TimeoutError("Finite preprocessing budget exceeded")
        path = BASE / case["source_file"]
        if sha(path) != case["source_sha256"]:
            raise ValueError("Frozen native bytes differ")
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        deint, interlaced, energy_ratio = detect_and_deinterlace_cctv(bgr)
        smoothed, mosaic = detect_and_smooth_mosaic(deint)
        processed = adaptive_cctv_denoise(smoothed)
        native_stages = {"source": rgb, "deinterlaced": cv2.cvtColor(deint, cv2.COLOR_BGR2RGB),
                         "mosaic_stage": cv2.cvtColor(smoothed, cv2.COLOR_BGR2RGB),
                         "processed": cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)}
        saved = {}
        for stage, value in native_stages.items():
            name = f"stages/{case['id']}_{stage}.png"
            Image.fromarray(value).save(OUT / name)
            with Image.open(OUT / name) as check:
                if not np.array_equal(value, np.asarray(check.convert("RGB"))):
                    raise ValueError("Lossless native-stage save differs")
            files[name] = sha(OUT / name)
            saved[stage] = name
        changed = int(np.any(native_stages["processed"] != rgb, axis=2).sum())
        flags["interlacing_triggered"] += int(interlaced)
        flags["mosaic_triggered"] += int(mosaic)
        flags["denoising_changed_pixels"] += int(not np.array_equal(processed, smoothed))
        flags["any_processed_change"] += int(changed > 0)
        rows.append({"id": case["id"], "size_bin": case["size_bin"], "source_sha256": case["source_sha256"],
                     "native_size": [case["native_width"], case["native_height"]], "stages": saved,
                     "interlacing_trigger": bool(interlaced), "interlacing_energy_ratio": energy_ratio,
                     "mosaic_trigger": bool(mosaic), "native_noise_proxy_sigma": estimate_noise_sigma(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)),
                     "changed_native_pixels": changed, "native_pixel_count": rgb.shape[0]*rgb.shape[1],
                     "mean_abs_input_change_255": float(np.abs(native_stages["processed"].astype(np.float32)-rgb).mean())})
    for group in ("le15", "16to23", "24to39", "ge40"):
        group_rows = [row for row in rows if row["size_bin"] == group]
        sheet = Image.new("RGB", (4*164, 6*190+26), (238, 238, 238))
        draw = ImageDraw.Draw(sheet)
        for j, stage in enumerate(("source", "deinterlaced", "mosaic_stage", "processed")):
            draw.text((j*164+2, 3), stage, fill=(0, 0, 0))
            for i, row in enumerate(group_rows):
                y = 26+i*190
                draw.text((j*164+2, y+1), row["id"], fill=(0, 0, 0))
                draw.text((j*164+2, y+14), str(row["native_size"]), fill=(0, 0, 0))
                with Image.open(OUT / row["stages"][stage]) as image:
                    sheet.paste(display(np.asarray(image.convert("RGB"))), (j*164, y+28))
        name = f"signal_{group}.png"
        sheet.save(OUT / name)
        files[name] = sha(OUT / name)
    results = {"complete": True, "protocol_sha256": sha(OUT / "frozen_protocol.json"), "rows": rows,
               "artifacts_sha256": files, "flag_counts": dict(flags), "seconds_after_import": time.monotonic()-start,
               "model_forwards": 0, "optimizer_updates": 0, "reserved_inputs_used": False,
               "application_change": False, "training": False}
    write(OUT / "results.json", results)
    print(json.dumps({"complete": True, "cases": len(rows), "flag_counts": dict(flags),
                      "seconds": results["seconds_after_import"], "results_sha256": sha(OUT / "results.json")}))


if __name__ == "__main__":
    main()
