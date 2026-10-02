"""Cache native/degraded real training inputs; zero optimizer/model operations."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from real_camera_pairs import camera_pair
from supported_real_data import load_supported_manifest, read_item, sha

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset/detector_supported_review_v2/manifest.json"
OUT = ROOT / "outputs/real_camera_pairs_v1"


def main():
    if OUT.exists() or sha(DATASET) != "3054c864bf7612fdcd5d0f55d47130b70f2895b6a0590f1341a60bbb011e2c7e":
        raise ValueError("Preserve existing cache and require the reviewed dataset")
    metadata, all_rows = load_supported_manifest(DATASET)
    rows = [row for row in all_rows if row["split"] == "train"]
    assert len(rows) == 91
    OUT.mkdir()
    (OUT / "degraded").mkdir()
    cases = []
    preview = Image.new("RGB", (1024, 40 + 280 * 8), "#16181c")
    draw = ImageDraw.Draw(preview)
    for col, label in enumerate(("new source native", "camera degraded", "target core unchanged", "purple=unsupervised")):
        draw.text((col * 256 + 3, 8), label, fill="white")
    preview_row = 0
    for index, row in enumerate(rows):
        rgb, target, valid, source = read_item(row, metadata["size"])
        seed = int.from_bytes(hashlib.sha256(("real-camera-v1:" + row["source_sha256"]).encode()).digest()[:4], "little")
        degraded = camera_pair(rgb, target, valid, source, seed)
        output = OUT / "degraded" / f"{index:03d}.png"
        Image.fromarray(degraded["input"]).save(output)
        masks = {key: {"path": Path(row[key + "_path"]).relative_to(ROOT).as_posix(), "sha256": row[key + "_sha256"]}
                 for key in ("mask", "valid", "source_valid")}
        for condition in ("native", "degraded"):
            path = Path(row["image_path"]) if condition == "native" else output
            cases.append({"case_id": len(cases), "real_train_index": index, "condition": condition,
                          "input": path.relative_to(ROOT).as_posix(), "input_sha256": sha(path),
                          "source_sha256": row["source_sha256"], "group": row["group"], "split": "train",
                          "kind": row["kind"], "family": row.get("occlusion_stratum", "previous_untagged"),
                          "data_origin": row["data_origin"], "target": masks,
                          "camera": None if condition == "native" else degraded["camera"],
                          "target_support_or_geometry_changed": False})
        if row["data_origin"] == "mendeley_supported_extension":
            panels = [rgb, degraded["input"]]
            marked = rgb.copy()
            marked[target] = (.55 * rgb[target] + .45 * np.array([16, 185, 129])).round().astype(np.uint8)
            panels.append(marked)
            marked = degraded["input"].copy()
            marked[~valid] = (.45 * marked[~valid] + .55 * np.array([165, 85, 215])).round().astype(np.uint8)
            panels.append(marked)
            y = 40 + 280 * preview_row
            draw.text((3, y), f"{row['source_id']}: {row['occlusion_stratum']}; same target/valid support", fill="white")
            for col, panel in enumerate(panels):
                preview.paste(Image.fromarray(panel), (col * 256, y + 19))
            preview_row += 1
    assert preview_row == 8 and len(cases) == 182
    preview.save(OUT / "preview.png")
    report = {"format": "dgp-reviewed-real-camera-pairs-v1", "date": "2026-10-02", "frozen_before_model_work": True,
              "registry_sha256": sha(DATASET), "registry": DATASET.relative_to(ROOT).as_posix(),
              "dataset_verification_sha256": sha(ROOT / "outputs/mendeley_supported_dataset_validation_v1/verification.json"),
              "builder_sha256": sha(__file__), "camera_helper_sha256": sha(ROOT / "real_camera_pairs.py"),
              "numpy": np.__version__, "opencv": cv2.__version__, "cases": cases,
              "native_source_count": 91, "native_cases": 91, "degraded_cases": 91,
              "held_out_inputs_in_cache": 0, "train_source_group_preserved": True,
              "preview": {"path": "preview.png", "sha256": sha(OUT / "preview.png")},
              "seed_rule": "Little-endian uint32 from SHA256('real-camera-v1:' + source_sha256)[:4]",
              "degradation": {"order": ["observed-ROI Gaussian blur", "area downsample", "bilinear return to original ROI size", "Gaussian RGB noise", "JPEG"],
                              "blur_sigma": [.6, 1.4], "kernel": 9, "low_full_side_choices": [128, 160, 192],
                              "noise_sigma_255": [1, 6], "jpeg_quality_integer_inclusive": [65, 95],
                              "padding_excluded_from_filtering": True, "label_dilation": 0, "label_geometry_change": False},
              "model_forwards": 0, "optimizer_updates": 0, "training_recipe_ready": False,
              "limitations": ["Synthetic input-only perturbations, not validated capture-device degradation distributions",
                              "Real hidden facial ground truth is absent; this cache serves detector training, not generator training",
                              "Fixed one degraded view per source is a bounded diagnostic, not an exhaustive augmentation pipeline",
                              "No claim that camera augmentation or eight related additions improves a model before a VM comparison"]}
    destination = OUT / "manifest.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sources": 91, "cases": 182, "held_out_inputs": 0,
                      "manifest_sha256": sha(destination), "model_forwards": 0, "optimizer_updates": 0}))


if __name__ == "__main__":
    main()
