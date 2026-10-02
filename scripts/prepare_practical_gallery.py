"""Prepare a fixed practical comparison cohort; no model execution or training.

The first cohort intentionally documents missing hair/scarf/object families.
Proposal masks live separately from immutable detector labels and need visual
review before the protocol is frozen or completion outputs are generated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "dataset/detector_supported_review_v1/manifest.json"
MANIFEST_SHA = "860ea1dfba8fa442cab19bf3ab41f6a678109dcdb067c38ec0bd79ce43b51ace"
FROZEN = ROOT / "outputs/frozen_complementarity_v1"
FROZEN_RESULT_SHA = "2880c85f067c013f6998faf1744177be79b8ed0a50d949d6e35ae5da14563928"
SELECTED = (
    (0, "cloth_mask", "face_mask", 3),
    (1, "pink_mask", "face_mask", 3),
    (4, "mask_with_clear_glasses", "face_mask", 3),
    (100, "dark_sunglasses", "sunglasses", 3),
    (108, "sunglasses", "sunglasses", 3),
    (21, "white_glare", "strong_lens_glare", 1),
    (113, "mirrored_glare", "sunglasses_and_glare", 3),
    (104, "hand_over_mask", "hand_over_mask", 3),
    (80, "uncovered", "uncovered_control", 0),
    (106, "clear_glasses", "clear_glasses_control", 0),
)
# Source-only operator corrections include the visible eyewear bridge. These are
# practical removal proposals, not modifications of the old occlusion labels.
BRIDGE_POLYGONS = {
    100: [(109, 93), (133, 92), (144, 101), (140, 111), (124, 113), (110, 105)],
    108: [(116, 108), (139, 106), (144, 116), (140, 125), (126, 121), (114, 120)],
    113: [(117, 119), (138, 117), (144, 127), (138, 135), (122, 131), (115, 129)],
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"))
    if array.shape != (256, 256, 3):
        raise ValueError(f"Expected existing 256px crop: {path}")
    return array


def binary(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.shape != (256, 256) or array.dtype != np.uint8 or not np.isin(array, [0, 255]).all():
        raise ValueError(f"Expected a binary 256px mask: {path}")
    return array


def overlay(image: np.ndarray, mask: np.ndarray) -> Image.Image:
    array = image.copy().astype(np.float32)
    marked = mask == 255
    array[marked] = .55 * array[marked] + .45 * np.array([255, 120, 20])
    return Image.fromarray(array.round().astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs/practical_gallery_v1")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT) or out.exists():
        raise ValueError("Use a new output directory in the workspace")
    if sha(MANIFEST) != MANIFEST_SHA or sha(FROZEN / "results.json") != FROZEN_RESULT_SHA:
        raise ValueError("Previously verified source evidence changed")
    registry = json.loads(MANIFEST.read_text(encoding="utf-8"))["supported_records"]
    train_indices = [i for i, row in enumerate(registry) if row["split"] == "train"]
    result = json.loads((FROZEN / "results.json").read_text(encoding="utf-8"))
    if not result["complete"] or result["optimizer_updates_locally"] != 0:
        raise ValueError("Frozen detector evidence is incomplete or changed")
    cases, assets, tiles = [], {}, []
    out.mkdir(parents=True)
    for directory in ("inputs", "references", "label_masks", "removal_proposals", "cached_original_masks"):
        (out / directory).mkdir()
    for position, (index, title, family, radius) in enumerate(SELECTED):
        row = registry[index]
        if row["split"] != "train":
            raise ValueError("The practical pilot must not consume held-out sources")
        for key in ("image", "mask", "valid", "source_valid"):
            path = MANIFEST.parent / row[key]
            if sha(path) != row[f"{key}_sha256"]:
                raise ValueError(f"Selected source {key} changed")
        if not binary(MANIFEST.parent / row["valid"]).all() or not binary(MANIFEST.parent / row["source_valid"]).all():
            raise ValueError("This first pilot requires fully observed source support")
        image, target = rgb(MANIFEST.parent / row["image"]), binary(MANIFEST.parent / row["mask"])
        painted = Image.fromarray(target.copy())
        if index in BRIDGE_POLYGONS:
            ImageDraw.Draw(painted).polygon(BRIDGE_POLYGONS[index], fill=255)
        proposal = np.asarray(painted.filter(ImageFilter.MaxFilter(2 * radius + 1))) if radius else target.copy()
        name = f"{position:02d}_{title}"
        Image.fromarray(image).save(out / "references" / f"{name}.png")
        Image.fromarray(target).save(out / "label_masks" / f"{name}.png")
        Image.fromarray(proposal).save(out / "removal_proposals" / f"{name}.png")
        cached, masks = {}, {}
        real_index = train_indices.index(index)
        for arm in ("parent", "candidate"):
            key = f"{arm}/real/{real_index:04d}.png"
            path = FROZEN / key
            if sha(path) != result["mask_sha256"][key]:
                raise ValueError("Reused original detector mask differs from verified pixels")
            masks[arm] = binary(path)
            destination = out / "cached_original_masks" / f"{name}_{arm}.png"
            Image.fromarray(masks[arm]).save(destination)
            cached[arm] = destination.relative_to(out).as_posix()
        tiles.append((name, image, proposal, masks["parent"], masks["candidate"]))
        for degraded in (False, True):
            array = image
            if degraded:
                base = Image.fromarray(image).filter(ImageFilter.GaussianBlur(1.2))
                base = base.resize((128, 128), Image.Resampling.BILINEAR).resize((256, 256), Image.Resampling.BILINEAR)
                noise = np.random.default_rng(20261002 + index).normal(0, 3, image.shape)
                array = np.clip(np.asarray(base).astype(np.float32) + noise, 0, 255).round().astype(np.uint8)
            case_id = name + ("_degraded" if degraded else "_native")
            input_path = out / "inputs" / f"{case_id}.png"
            Image.fromarray(array).save(input_path)
            cases.append({
                "id": case_id, "source_manifest_index": index,
                "source": row["source"], "source_sha256": row["source_sha256"],
                "accepted_crop_sha256": row["image_sha256"], "family": family,
                "split": "train", "input": input_path.relative_to(out).as_posix(),
                "reference": f"references/{name}.png", "synthetically_degraded": degraded,
                "reference_scope": "known visible pixels only; no hidden-face ground truth",
                "label_mask": f"label_masks/{name}.png",
                "removal_proposal": f"removal_proposals/{name}.png",
                "removal_margin_pixels_at_256": radius,
                "operator_bridge_polygon": BRIDGE_POLYGONS.get(index),
                "cached_original_masks": None if degraded else cached,
                "automatic_mask_requirement": "Fresh inference on the actual degraded input" if degraded else "Exact original cached pixels or reproduced inference",
                "exposure": "previously inspected detector-training source; not pristine holdout",
            })
    previews = []
    for start in (0, 5):
        canvas = Image.new("RGB", (1024, 1430), "#16181c")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), "DRAFT: input | parent detector | candidate detector | proposed removal", fill="white")
        for offset, (name, image, proposal, parent, candidate) in enumerate(tiles[start:start + 5]):
            y = 42 + offset * 276
            for column, tile in enumerate((Image.fromarray(image), overlay(image, parent), overlay(image, candidate), overlay(image, proposal))):
                canvas.paste(tile, (column * 256, y))
            draw.text((10, y + 258), name, fill="white")
        path = out / f"mask_review_{start // 5 + 1:02d}.png"
        canvas.save(path)
        previews.append(path.name)
    for path in sorted(out.rglob("*.png")):
        assets[path.relative_to(out).as_posix()] = sha(path)
    protocol = {
        "format": "dgp-practical-gallery-v1", "date": "2026-10-02",
        "status": "mask_review_draft", "gallery_frozen": False,
        "scope": "First ten-source practical baseline; full covering-family readiness is incomplete",
        "source_count": len(SELECTED), "case_count": len(cases),
        "source_manifest_sha256": MANIFEST_SHA, "frozen_detector_result_sha256": FROZEN_RESULT_SHA,
        "degradation": {"gaussian_blur_radius": 1.2, "downsample_size": 128, "noise_sigma_255": 3, "seed": "20261002 + source_manifest_index"},
        "missing_families": ["standalone_hand", "obstructing_hair", "scarf_over_face", "other_object"],
        "removal_policy": "Draft square dilation of reviewed covering footprint plus source-reviewed bridge polygons for sunglasses: 3px at256, 1px for glare on clear glasses, 0px for empty controls. Must be visually checked; original labels remain unchanged.",
        "planned_arms": ["parent_detector_codeformer", "candidate_detector_codeformer", "reviewed_removal_codeformer"],
        "restoration_comparison": "Off baseline; compare legacy pre-completion DGP versus visible-only post-completion DGP on the ten declared degraded copies. Do not alter clear inputs in automatic mode.",
        "visual_criteria": ["covering substantially removed", "plausible facial features", "visible appearance preserved outside removal margin", "no conspicuous remnants/seams", "clear controls preserved and manual correction available"],
        "scores": {"hidden_face_mae": None, "reason": "No clean hidden facial ground truth exists for these real occlusions", "visible_degraded_mae": "Score only outside the final removal mask against known original visible pixels", "native_visible_change": "Pixel drift only; not recovery accuracy"},
        "new_model_forwards": 0, "optimizer_updates": 0, "promoted": False,
        "previews": previews, "assets_sha256": assets, "cases": cases,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    if sha(MANIFEST) != MANIFEST_SHA:
        raise ValueError("Immutable source manifest changed during preparation")
    print(json.dumps({"output": str(out), "sources": len(SELECTED), "cases": len(cases), "status": protocol["status"], "model_forwards": 0}))


if __name__ == "__main__":
    main()
