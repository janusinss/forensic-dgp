"""Freeze training-only V3 native/camera pairs. No torch, model or optimizer."""
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from real_camera_pairs import camera_pair

REGISTRY = ROOT / "dataset/detector_supported_review_v3/manifest.json"
DATA_AUDIT = ROOT / "outputs/cofw_supported_dataset_validation_v1/verification.json"
OUT = ROOT / "outputs/cofw_camera_pairs_v2"
PREVIEW_IDS = ("cofw_train_1030", "cofw_train_0935", "cofw_train_0868",
               "cofw_train_1338", "cofw_train_1057", "cofw_train_1212",
               "cofw_train_0906", "cofw_train_1143", "cofw_train_0930",
               "cofw_train_0246")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, mode):
    with Image.open(path) as image:
        return np.asarray(image.convert(mode)).copy()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    require(not OUT.exists(), "Preserve existing camera cache; use a new version")
    require(sha(REGISTRY) == "abab07e152941c4ae24d8aea3755fa7aaedb18965b25f6d0d1b6a7e00094aadd",
            "Reviewed V3 registry changed")
    require(sha(DATA_AUDIT) == "a403f14457d7ba850ff2dbea4236b1db88276cd9e5d4f8192a65e0a0159261cc",
            "Independent dataset audit changed")
    audit = json.loads(DATA_AUDIT.read_text())
    require(audit["complete"] and audit["registry_sha256"] == sha(REGISTRY), "Dataset audit incomplete")
    metadata = json.loads(REGISTRY.read_text())
    rows = [r for r in metadata["supported_records"] if r["split"] == "train"]
    require(len(rows) == 133 and len(metadata["supported_records"]) == 165, "Fixed membership differs")
    require(sum(r["data_origin"] == "cofw_supported_extension" for r in rows) == 42,
            "Original author-training extension differs")
    OUT.mkdir()
    (OUT / "degraded").mkdir()
    cases, preview_items = [], {}
    for index, row in enumerate(rows):
        source_id = row.get("source_id", f"previous_train_{index:03d}")
        assets = {}
        for key in ("image", "mask", "valid", "source_valid"):
            path = (REGISTRY.parent / row[key]).resolve()
            require(path.is_relative_to(REGISTRY.parent) and sha(path) == row[key + "_sha256"],
                    "Training asset outside registry or changed: " + key)
            assets[key] = path
        rgb = read(assets["image"], "RGB")
        arrays = [read(assets[k], "L") for k in ("mask", "valid", "source_valid")]
        require(rgb.shape == (256, 256, 3) and all(a.shape == (256, 256) and
                np.isin(a, [0, 255]).all() for a in arrays), "Fixed RGB/binary dimensions differ")
        target, valid, source = [a == 255 for a in arrays]
        require(bool(target.any()) == (row["kind"] == "covered"), "Clear/covered target differs")
        seed = int.from_bytes(hashlib.sha256(("real-camera-v1:" + row["source_sha256"]).encode()).digest()[:4], "little")
        pair = camera_pair(rgb, target, valid, source, seed)
        require(all(np.array_equal(pair[k], a) for k, a in
                (("mask", target), ("valid", valid), ("source_valid", source))), "Camera changed targets/support")
        output = OUT / "degraded" / f"{index:03d}.png"
        Image.fromarray(pair["input"]).save(output)
        targets = {k: {"path": assets[k].relative_to(ROOT).as_posix(), "sha256": row[k + "_sha256"]}
                   for k in ("mask", "valid", "source_valid")}
        for condition in ("native", "degraded"):
            path = assets["image"] if condition == "native" else output
            cases.append({"case_id": len(cases), "real_train_index": index, "source_id": source_id,
                          "condition": condition, "input": path.relative_to(ROOT).as_posix(),
                          "input_sha256": sha(path), "source_sha256": row["source_sha256"],
                          "group": row["group"], "kind": row["kind"], "split": "train",
                          "family": row.get("occlusion_stratum", "previous_untagged"),
                          "data_origin": row["data_origin"], "target": targets,
                          "camera": None if condition == "native" else pair["camera"],
                          "target_support_or_geometry_changed": False})
        if source_id in PREVIEW_IDS:
            core = rgb.copy()
            core[target] = (.55 * core[target] + .45 * np.array([16, 185, 129])).round().astype(np.uint8)
            unknown = pair["input"].copy()
            unknown[~valid] = (.45 * unknown[~valid] + .55 * np.array([165, 85, 215])).round().astype(np.uint8)
            preview_items[source_id] = (row, [rgb, pair["input"], core, unknown])
    require(len(preview_items) == 10 and len(cases) == 266, "Fixed pair/preview budget differs")
    previews = []
    for page in range(2):
        sheet = Image.new("RGB", (768, 40 + 219 * 5), "#16181c")
        draw = ImageDraw.Draw(sheet)
        for col, label in enumerate(("native", "camera", "green=target core", "purple=unsupervised")):
            draw.text((192 * col + 3, 8), label, fill="white")
        for line, source_id in enumerate(PREVIEW_IDS[5 * page:5 * page + 5]):
            row, panels = preview_items[source_id]
            y = 40 + line * 219
            for col, panel in enumerate(panels):
                sheet.paste(Image.fromarray(panel).resize((192, 192)), (192 * col, y))
            draw.text((3, y + 195), f"{source_id}: {row['occlusion_stratum']}", fill="white")
        path = OUT / f"preview_{page + 1}.png"
        sheet.save(path)
        previews.append({"path": path.name, "sha256": sha(path), "source_ids": list(PREVIEW_IDS[page * 5:page * 5 + 5])})
    report = {"format": "dgp-cofw-camera-pairs-v1", "date": "2026-10-03",
              "registry": REGISTRY.relative_to(ROOT).as_posix(), "registry_sha256": sha(REGISTRY),
              "dataset_verification": DATA_AUDIT.relative_to(ROOT).as_posix(), "dataset_verification_sha256": sha(DATA_AUDIT),
              "builder_sha256": sha(__file__), "camera_helper_sha256": sha(ROOT / "real_camera_pairs.py"),
              "numpy": np.__version__, "opencv": cv2.__version__, "cases": cases,
              "native_source_count": 133, "native_cases": 133, "degraded_cases": 133,
              "cofw_sources": 42, "previous_sources": 91, "held_out_inputs_in_cache": 0,
              "seed_rule": "Little-endian uint32 from SHA256('real-camera-v1:' + source_sha256)[:4]",
              "degradation": {"order": ["observed-ROI Gaussian blur", "area downsample", "bilinear return", "Gaussian RGB noise", "JPEG"],
                              "blur_sigma": [.6, 1.4], "kernel": 9, "low_full_side_choices": [128, 160, 192],
                              "noise_sigma_255": [1, 6], "jpeg_quality_integer_inclusive": [65, 95],
                              "padding_excluded_from_filtering": True, "label_dilation": 0, "label_geometry_change": False},
              "previews": previews, "frozen_before_new_model_work": True,
              "model_forwards": 0, "optimizer_updates": 0, "training_recipe_ready": False,
              "limitations": ["Fixed synthetic input degradation is not a calibrated camera distribution",
                              "This detector cache contains no true hidden-face/generator target",
                              "Sources are reviewed training data, not an unseen evaluation cohort",
                              "A separate finite VM protocol/package and original model gates remain necessary"]}
    destination = OUT / "manifest.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sources": 133, "cases": 266, "held_out_inputs": 0, "manifest_sha256": sha(destination),
                      "model_forwards": 0, "optimizer_updates": 0}))


if __name__ == "__main__":
    main()
