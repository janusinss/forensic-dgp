"""Check source pixels, annotation geometry/support and independent raster agreement."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/mendeley_mask_proposals_v3"
SOURCE = ROOT / "outputs/mendeley_occlusion_candidate_v1/uploaded_source_review"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixels(path):
    with Image.open(path) as image:
        return np.asarray(image).copy()


def main():
    destination = OUT / "independent_verification.json"
    if destination.exists():
        raise ValueError("Preserve completed proposal audit")
    path = OUT / "manifest.json"
    lineage = json.loads((OUT / "refinement.json").read_text())
    assert sha(path) == lineage["new_manifest_sha256"]
    assert lineage["parent_manifest_sha256"] == "f91154ee48103548469a521f71c20b966efa660f4d314e433a5a2200cc2dd272"
    data = json.loads(path.read_text())
    source_audit = SOURCE / "independent_verification.json"
    assert json.loads(source_audit.read_text())["complete"] is True
    rows = data["records"]
    assert [r["source_id"] for r in rows] == [11339, 11340, 11349, 11353, 11347, 11343, 11344, 11341]
    assert len({r["source_sha256"] for r in rows}) == 8
    assert len({r["rgb_pixel_sha256"] for r in rows}) == 8
    assert {r["group"] for r in rows} == {"mendeley-upload-tail-related-capture-training-only-v1"}
    checks = []
    for row in rows:
        source = ROOT / row["source"]
        assert sha(source) == row["source_sha256"]
        with Image.open(source) as image:
            native = np.asarray(image.convert("RGB")).copy()
        assert native.shape == (112, 92, 3) and np.array_equal(native, pixels(OUT / row["files"]["native"]["path"]))
        cv = np.zeros((112, 92), np.uint8)
        pil = Image.new("L", (92, 112), 0)
        draw = ImageDraw.Draw(pil)
        for polygon in row["native_polygons"]:
            assert len(polygon) >= 3 and all(0 <= x < 92 and 0 <= y < 112 for x, y in polygon)
            cv2.fillPoly(cv, [np.asarray(polygon, np.int32)], 1)
            draw.polygon([tuple(point) for point in polygon], fill=1)
        native_full = cv.astype(bool)
        eroded = cv2.erode(cv, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=1)
        dilated = cv2.dilate(cv, np.ones((3, 3), np.uint8))
        band = dilated != eroded
        crop_rows = [111] if native_full[-1].any() else []
        assert row["native_unknown_crop_rows"] == crop_rows
        if crop_rows:
            band[-1] = True
        support = ~band
        # Two rasterizers may disagree on subpixel boundaries; none of those
        # disagreements may affect an actual supervised native target.
        raster_disagreement = native_full != np.asarray(pil).astype(bool)
        assert not (raster_disagreement & support).any()
        affine = np.asarray(row["affine"], np.float64)
        expected = np.array([[255 / 111, 0, (255 - 91 * 255 / 111) / 2], [0, 255 / 111, 0]])
        assert np.allclose(affine, expected, rtol=0, atol=1e-12)
        image = cv2.warpAffine(native, affine, (256, 256), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(96, 96, 96))
        def warped(a):
            return cv2.warpAffine(a.astype(np.uint8), affine, (256, 256), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        source_valid = warped(np.ones(cv.shape, bool))
        valid, mask, full = warped(support), warped(native_full & support), warped(native_full)
        image[~source_valid] = 96
        expected_files = {"native": native, "images": image, "masks": mask, "valid": valid,
                          "source_valid": source_valid, "full_proposals": full}
        for role, expected_array in expected_files.items():
            f = row["files"][role]
            asset = (OUT / f["path"]).resolve()
            assert asset.is_relative_to(OUT.resolve()) and sha(asset) == f["sha256"]
            actual = pixels(asset)
            if role not in ("native", "images"):
                assert actual.dtype == np.uint8 and np.isin(actual, (0, 255)).all()
                actual = actual == 255
            assert np.array_equal(actual, expected_array)
        assert int(mask.sum()) == row["positive_pixels"] and int(valid.sum()) == row["supervised_pixels"]
        assert int(source_valid.sum()) == row["source_pixels"] and int(band.sum()) == row["native_unknown_pixels"]
        assert not (mask & ~valid).any() and not (valid & ~source_valid).any()
        assert np.all(image[~source_valid] == 96) and bool(mask.any()) == (row["kind"] == "covered")
        unknown_observed = source_valid & ~valid
        assert np.array_equal(image[unknown_observed], pixels(OUT / row["files"]["images"]["path"])[unknown_observed])
        checks.append({"source_id": row["source_id"], "native_raster_disagreement_pixels": int(raster_disagreement.sum()),
                       "supervised_native_raster_disagreement_pixels": 0,
                       "unknown_observed_pixels": int(unknown_observed.sum()), "supervised_pixels": int(valid.sum()),
                       "positive_pixels": int(mask.sum())})
    assert sha(OUT / data["preview"]["path"]) == data["preview"]["sha256"]
    record = {"format": "dgp-mendeley-mask-independent-verification-v1", "date": "2026-10-02",
              "complete": True, "manifest_sha256": sha(path), "source_verification_sha256": sha(source_audit),
              "auditor_sha256": sha(__file__), "files_verified": 48, "sources_verified": 8,
              "checks": checks, "source_pixel_and_geometry_verified": True,
              "uncertain_rgb_is_retained": True, "padding_rgb96_verified": True,
              "independent_native_raster_supervised_agreement": True,
              "semantic_mask_accuracy_is_not_proven_by_this_audit": True,
              "training_admitted": False, "model_forwards": 0, "optimizer_updates": 0}
    destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": True, "sources": 8, "verification_sha256": sha(destination)}))


if __name__ == "__main__":
    main()
