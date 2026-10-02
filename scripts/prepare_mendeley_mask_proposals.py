"""Freeze eight source-grounded detector label proposals, without model work."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/mendeley_occlusion_candidate_v1/uploaded_source_review"
OUT = ROOT / "outputs/mendeley_mask_proposals_v1"
INTEGRITY_SHA = "a3c61ca1e62c284826e968326889a639dd6f37dca2330daaed312b0047a0e493"

# Native 92x112 coordinates inspected before creating proposals. These target
# covering overlap with facial features, not removal of hands/hair elsewhere.
PROPOSALS = [
    (11339, "uncovered", [], "Observed clear face; ordinary hairstyle remains"),
    (11340, "hand", [[(53, 9), (62, 8), (74, 15), (82, 26), (84, 45), (84, 89),
                       (77, 111), (46, 111), (43, 91), (49, 76), (51, 60), (51, 36)]],
     "Vertical hand over one eye and cheek; label the facial overlap"),
    (11349, "hand", [[(14, 57), (28, 55), (48, 56), (66, 55), (81, 53), (86, 60),
                       (85, 77), (82, 93), (76, 107), (39, 111), (28, 99), (19, 84), (14, 72)]],
     "Horizontal hand hides mouth/nose/lower face; avoid the wrist outside facial features"),
    (11353, "hand", [[(14, 53), (24, 51), (43, 56), (56, 59), (68, 61), (80, 63),
                       (85, 69), (86, 91), (77, 94), (64, 94), (48, 97), (32, 98),
                       (15, 99), (9, 88), (12, 72)]],
     "Hand hides central upper features; visible mouth and upper forehead stay outside"),
    (11347, "obstructing_hair", [[(33, 22), (45, 22), (43, 37), (40, 51), (38, 67),
                                  (36, 83), (33, 102), (28, 100), (22, 86), (18, 66),
                                  (18, 50), (23, 35)]],
     "Inferred facial overlap of a hair curtain; keep crown and outer hairstyle. Approximate hidden-face boundary, not anatomy truth"),
    (11343, "sunglasses", [[(13, 30), (26, 28), (42, 29), (46, 31), (53, 29), (69, 29),
                            (74, 32), (75, 43), (71, 51), (64, 58), (56, 56), (51, 52),
                            (49, 44), (47, 40), (44, 48), (41, 55), (35, 58), (25, 58),
                            (19, 55), (14, 48)]],
     "Opaque sunglasses including observed dark rim/bridge; not a clear-glasses label"),
    (11344, "face_mask", [[(15, 58), (30, 54), (45, 50), (61, 56), (80, 59), (80, 75),
                           (73, 96), (59, 105), (41, 111), (27, 104), (17, 94), (13, 82), (12, 67)]],
     "Cloth mask over nose/mouth/chin; visible eyes and ordinary hair stay outside"),
    (11341, "sunglasses_and_mask", [
        [(11, 34), (25, 30), (41, 31), (46, 33), (53, 31), (67, 32), (72, 35),
         (73, 43), (69, 52), (63, 58), (56, 57), (51, 53), (48, 44), (46, 41),
         (43, 49), (39, 55), (31, 59), (22, 58), (15, 53), (12, 45)],
        [(15, 79), (26, 75), (44, 73), (59, 78), (77, 77), (76, 94), (69, 106),
         (48, 111), (32, 108), (20, 100), (17, 90)]],
     "Two separate coverings; visible nose between them remains observed"),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUT.exists() or sha(SOURCE / "integrity.json") != INTEGRITY_SHA:
        raise ValueError("Preserve existing proposals and verified source inventory")
    inventory = json.loads((SOURCE / "integrity.json").read_text())
    records = {item["id"]: item for item in inventory["records"]}
    OUT.mkdir()
    for folder in ("native", "images", "masks", "valid", "source_valid", "full_proposals"):
        (OUT / folder).mkdir()
    rows = []
    canvas = Image.new("RGB", (1024, 40 + 280 * len(PROPOSALS)), "#16181c")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(("source (no model output)", "full proposal", "supervised positive core", "purple=unknown/padding")):
        draw.text((column * 256 + 3, 8), label, fill="white")
    for index, (image_id, family, polygons, rationale) in enumerate(PROPOSALS):
        original = records[image_id]
        if sha(ROOT / original["image"]) != original["sha256"]:
            raise ValueError("Raw source changed")
        with Image.open(ROOT / original["image"]) as photo:
            rgb = np.asarray(photo.convert("RGB")).copy()
        if rgb.shape != (112, 92, 3):
            raise ValueError("Reviewed native geometry differs")
        raw = np.zeros(rgb.shape[:2], np.uint8)
        if polygons:
            cv2.fillPoly(raw, [np.array(polygon, np.int32) for polygon in polygons], 1)
            band = cv2.dilate(raw, np.ones((3, 3), np.uint8)) != cv2.erode(
                raw, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=1)
        else:
            band = np.zeros(raw.shape, bool)
        native_valid = (~band).astype(np.uint8)
        native_core = raw & native_valid
        # Existing supported-data convention: endpoints are pixel centers, with
        # a uniform affine and true padding represented separately from unknown RGB.
        scale = 255 / 111
        affine = np.array([[scale, 0, (255 - 91 * scale) / 2], [0, scale, 0]], np.float64)
        image = cv2.warpAffine(rgb, affine, (256, 256), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(96, 96, 96))
        def warp(array):
            return cv2.warpAffine(array, affine, (256, 256), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        source_support = warp(np.ones(raw.shape, np.uint8))
        valid, mask, full = warp(native_valid), warp(native_core), warp(raw)
        image[~source_support] = 96
        if np.any(mask & ~valid) or np.any(valid & ~source_support):
            raise ValueError("Proposal/support containment differs")
        files = {}
        arrays = {"native": rgb, "images": image, "masks": mask, "valid": valid,
                  "source_valid": source_support, "full_proposals": full}
        for role, array in arrays.items():
            path = OUT / role / f"{image_id}.png"
            Image.fromarray(array if role in ("native", "images") else array.astype(np.uint8) * 255).save(path)
            files[role] = {"path": path.relative_to(OUT).as_posix(), "sha256": sha(path)}
        row = {"index": index, "source_id": image_id, "family": family,
               "source": original["image"], "source_sha256": original["sha256"],
               "rgb_pixel_sha256": original["rgb_pixel_sha256"], "native_size": [92, 112],
               "native_polygons": polygons, "native_unknown_boundary_radius": 1,
               "affine": affine.tolist(), "files": files,
               "kind": "uncovered" if not polygons else "covered",
               "native_positive_core_pixels": int(native_core.sum()),
               "native_unknown_pixels": int(band.sum()), "positive_pixels": int(mask.sum()),
               "supervised_pixels": int(valid.sum()), "source_pixels": int(source_support.sum()),
               "scope": rationale, "group": "mendeley-upload-tail-related-capture-training-only-v1",
               "intended_split": "train", "reviewed": False,
               "uncovered_face_reference": None, "high_resolution_reference": False,
               "annotation_quality": "Approximate assistant polygons; uncertain native boundary excluded; no independent expert adjudication"}
        rows.append(row)
        panels = [image]
        for selected in (full, mask):
            marked = image.copy()
            marked[selected] = (.55 * image[selected] + .45 * np.array([16, 185, 129])).round().astype(np.uint8)
            panels.append(marked)
        support_panel = image.copy()
        support_panel[~valid] = (.45 * image[~valid] + .55 * np.array([165, 85, 215])).round().astype(np.uint8)
        panels.append(support_panel)
        y = 40 + 280 * index
        draw.text((3, y), f"{image_id}: {family}; native 92x112, no pristine target", fill="white")
        for column, panel in enumerate(panels):
            canvas.paste(Image.fromarray(panel), (column * 256, y + 19))
    path = OUT / "preview.png"
    canvas.save(path)
    report = {"format": "dgp-mendeley-detector-mask-proposals-v1", "date": "2026-10-02",
              "source_integrity_sha256": INTEGRITY_SHA, "builder_sha256": sha(__file__),
              "source_url": inventory["source_url"], "doi": inventory["doi"],
              "license_listed_by_publisher": inventory["license_listed_by_publisher"],
              "attribution": inventory["attribution"], "records": rows,
              "preview": {"path": "preview.png", "sha256": sha(path)},
              "target_semantics": "Supervised covering core is one; uncertain +/-1 native pixel boundary is unsupervised but retains observed RGB",
              "split_policy": "All related tail captures train-only together. No new test or validation split; no subject-disjoint claim",
              "training_admitted": False, "training_recipe_ready": False,
              "visual_review_pending": True, "new_model_forwards": 0, "optimizer_updates": 0,
              "limitations": ["Eight sources from a related capture cohort, not eight independent identities",
                              "Low-resolution detector labels only, not clean-face restoration/completion pairs",
                              "Hair overlap includes an approximate inferred facial boundary",
                              "This sample does not supply verified scarf or general-object training examples"]}
    destination = OUT / "manifest.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sources": len(rows), "training_admitted": False,
                      "proposal_manifest_sha256": sha(destination), "preview": str(path)}))


if __name__ == "__main__":
    main()
