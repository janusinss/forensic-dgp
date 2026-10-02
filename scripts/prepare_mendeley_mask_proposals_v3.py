"""Record uncertainty where a covering footprint is clipped by the native crop."""
import copy
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw

from scripts.prepare_mendeley_mask_proposals import ROOT, sha

PARENT = ROOT / "outputs/mendeley_mask_proposals_v2"
OUT = ROOT / "outputs/mendeley_mask_proposals_v3"


def main():
    if OUT.exists() or sha(PARENT / "manifest.json") != "f91154ee48103548469a521f71c20b966efa660f4d314e433a5a2200cc2dd272":
        raise ValueError("Preserve prior proposals and require the reviewed V2 geometry")
    data = json.loads((PARENT / "manifest.json").read_text())
    data = copy.deepcopy(data)
    OUT.mkdir()
    for folder in ("native", "images", "masks", "valid", "source_valid", "full_proposals"):
        (OUT / folder).mkdir()
    changed = []
    canvas = Image.new("RGB", (1024, 40 + 280 * len(data["records"])), "#16181c")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(("source (no model output)", "full proposal", "supervised positive core", "purple=unknown/padding")):
        draw.text((column * 256 + 3, 8), label, fill="white")
    for index, row in enumerate(data["records"]):
        for role, asset in row["files"].items():
            source = PARENT / asset["path"]
            if sha(source) != asset["sha256"]:
                raise ValueError("Parent proposal asset changed")
            shutil.copyfile(source, OUT / asset["path"])
        native_full = np.zeros((112, 92), np.uint8)
        for polygon in row["native_polygons"]:
            cv2.fillPoly(native_full, [np.asarray(polygon, np.int32)], 1)
        band = cv2.dilate(native_full, np.ones((3, 3), np.uint8)) != cv2.erode(
            native_full, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=1)
        crop_rows = [111] if native_full[-1].any() else []
        if crop_rows:
            band[-1] = True
            changed.append(row["source_id"])
        native_valid = ~band
        native_core = native_full.astype(bool) & native_valid
        affine = np.asarray(row["affine"], np.float64)
        def warp(a):
            return cv2.warpAffine(a.astype(np.uint8), affine, (256, 256), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        valid, mask = warp(native_valid), warp(native_core)
        for role, array in (("valid", valid), ("masks", mask)):
            target = OUT / row["files"][role]["path"]
            Image.fromarray(array.astype(np.uint8) * 255).save(target)
            row["files"][role]["sha256"] = sha(target)
        row.update(native_unknown_crop_rows=crop_rows, native_unknown_pixels=int(band.sum()),
                   native_positive_core_pixels=int(native_core.sum()), positive_pixels=int(mask.sum()),
                   supervised_pixels=int(valid.sum()))
        with Image.open(OUT / row["files"]["images"]["path"]) as photo:
            image = np.asarray(photo).copy()
        with Image.open(OUT / row["files"]["full_proposals"]["path"]) as label:
            full = np.asarray(label) == 255
        panels = [image]
        for selected in (full, mask):
            marked = image.copy()
            marked[selected] = (.55 * image[selected] + .45 * np.array([16, 185, 129])).round().astype(np.uint8)
            panels.append(marked)
        marked = image.copy()
        marked[~valid] = (.45 * image[~valid] + .55 * np.array([165, 85, 215])).round().astype(np.uint8)
        panels.append(marked)
        y = 40 + index * 280
        draw.text((3, y), f"{row['source_id']}: {row['family']}; clipped boundary uncertain, no pristine target", fill="white")
        for column, panel in enumerate(panels):
            canvas.paste(Image.fromarray(panel), (column * 256, y + 19))
    canvas.save(OUT / "preview.png")
    data.update(builder_sha256=sha(__file__), parent_manifest_sha256=sha(PARENT / "manifest.json"),
                revision=3, native_crop_edge_policy="When the proposed covering touches bottom native row111, that observed row remains RGB context but is not supervised",
                target_semantics="Supervised covering core is one; +/-1 native pixel boundary and clipped covering bottom crop row are unsupervised; observed RGB retained",
                preview={"path": "preview.png", "sha256": sha(OUT / "preview.png")})
    (OUT / "manifest.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    lineage = {"format": "dgp-native-proposal-refinement-v1", "date": "2026-10-02",
               "parent_manifest_sha256": sha(PARENT / "manifest.json"),
               "new_manifest_sha256": sha(OUT / "manifest.json"), "builder_sha256": sha(__file__),
               "changed_source_ids": changed, "native_polygons_changed": False, "source_images_changed": False,
               "reason": "V2 audit found three supervised OpenCV/Pillow raster disagreements at source11349's clipped native row111. Record crop-edge uncertainty for all proposed coverings touching that row; keep V1/V2 unchanged.",
               "v2_failed_audit": {"source_id": 11349, "supervised_native_pixels": [[40, 111], [41, 111], [42, 111]]},
               "training_admitted": False, "model_forwards": 0, "optimizer_updates": 0}
    (OUT / "refinement.json").write_text(json.dumps(lineage, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sources": 8, "crop_uncertainty_added": changed,
                      "manifest_sha256": sha(OUT / "manifest.json"), "training_admitted": False}))


if __name__ == "__main__":
    main()
