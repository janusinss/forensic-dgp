"""Independent COFW source/geometry/overlap audit; never builds an ML model."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "outputs/cofw_training_sources_v2"
SELECTION = ROOT / "outputs/cofw_annotation_sources_v1"
PROPOSALS = ROOT / "outputs/cofw_covering_proposals_v1"
OUT = ROOT / "outputs/cofw_covering_data_validation_v1"


def sha(path, algorithm="sha256"):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def path(base, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts or ":" in relative or "\\" in relative:
        raise ValueError("Unsafe portable relative asset")
    result = (base / relative).resolve()
    if not result.is_relative_to(base.resolve()) or not result.is_file():
        raise ValueError("Missing or escaping source asset")
    return result


def pixels(file, mode=None):
    with Image.open(file) as im:
        im.load()
        return np.asarray(im.convert(mode) if mode else im).copy()


def pixel_sha(array):
    height, width = array.shape[:2]
    return hashlib.sha256(f"RGB:{width}x{height}:".encode() + array.tobytes()).hexdigest()


def raster(size, polygons, pillow):
    if pillow:
        im = Image.new("L", tuple(size), 0)
        draw = ImageDraw.Draw(im)
        for polygon in polygons:
            draw.polygon([tuple(p) for p in polygon], fill=255)
        return np.asarray(im) != 0
    array = np.zeros((size[1], size[0]), np.uint8)
    for polygon in polygons:
        cv2.fillPoly(array, [np.asarray(polygon, np.int32)], 1)
    return array != 0


def main():
    sys.path.insert(0, str(ROOT / "outputs/cofw_read_dependencies"))
    import h5py
    if OUT.exists():
        raise ValueError("Preserve previous independent verification")
    acquire = ROOT / "outputs/cofw_source_acquisition_v3"
    record = read(acquire / "current_record.json")
    official = record["files"]["entries"]["COFW_color.zip"]
    archive = acquire / "COFW_color.zip"
    assert official["size"] == archive.stat().st_size == 503327162
    assert official["checksum"] == "md5:" + sha(archive, "md5") == "md5:8b21d126c4e1fb307cb463578eef0511"
    assert sha(archive) == "bc6a79bda1bd88705082af9ebdd31188cf5841917b09534b25067e424a0130e5"
    assert [r["id"] for r in record["metadata"]["rights"]] == ["cc-by-4.0"]
    extracted = ROOT / "outputs/cofw_source_review_v1"
    extraction = read(extracted / "extraction.json")
    with zipfile.ZipFile(archive) as z:
        assert sorted(z.namelist()) == ["COFW_test_color.mat", "COFW_train_color.mat"]
        for item in extraction["files"]:
            with z.open(item["file"]) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            assert digest == item["sha256"] == sha(extracted / item["file"])
    queue, selection, labels = read(QUEUE / "manifest.json"), read(SELECTION / "manifest.json"), read(PROPOSALS / "manifest.json")
    assert selection["queue_sha256"] == sha(QUEUE / "manifest.json")
    assert labels["source_selection_sha256"] == sha(SELECTION / "manifest.json")
    assert sha(ROOT / queue["matrix"]) == queue["matrix_sha256"] == selection["matrix_sha256"]
    assert len(queue["records"]) == 1345 and len(selection["records"]) == len(labels["records"]) == 42
    assert labels["training_admitted"] is False and selection["training_admitted"] is False
    assert all(r["publisher_split"] == "train" for r in queue["records"])
    train_hashes, crop_hashes, modes = set(), set(), Counter()
    with h5py.File(ROOT / queue["matrix"], "r") as matrix:
        boxes, points = np.asarray(matrix["bboxesTr"]).T, np.asarray(matrix["phisTr"]).T
        refs = np.asarray(matrix["IsTr"]).reshape(-1)
        for index, row in enumerate(queue["records"]):
            assert row["matlab_train_index"] == index + 1
            stored = np.asarray(matrix[refs[index]])
            if stored.ndim == 2:
                rgb = np.ascontiguousarray(np.repeat(stored.T[..., None], 3, axis=2))
                mode = "L"
            else:
                assert stored.shape[0] == 3
                rgb = np.ascontiguousarray(stored.transpose(2, 1, 0))
                mode = "RGB"
            assert rgb.dtype == np.uint8
            assert pixel_sha(rgb) == row["native_rgb_sha256"]
            assert row["native_size"] == [rgb.shape[1], rgb.shape[0]]
            assert row["native_mode"] == mode
            assert np.array_equal(boxes[index], row["matlab_bbox_xywh"])
            assert np.array_equal(points[index, 58:], row["landmark_occlusion_flags"])
            assert np.array_equal(np.column_stack((points[index,:29], points[index,29:58])), row["landmarks_matlab_xy"])
            x0, y0, x1, y1 = row["crop_xyxy_zero_based_half_open"]
            cropped = rgb[y0:y1, x0:x1]
            assert cropped.shape[:2][::-1] == tuple(row["crop_native_size"])
            p = path(ROOT, row["inspection_crop"])
            assert sha(p) == row["inspection_crop_sha256"]
            canvas = Image.new("RGB", (256,256), (96,96,96))
            thumb = ImageOps.contain(Image.fromarray(cropped), (256,256), Image.Resampling.BICUBIC)
            canvas.paste(thumb, ((256-thumb.width)//2, (256-thumb.height)//2))
            assert np.array_equal(np.asarray(canvas), pixels(p))
            train_hashes.add(pixel_sha(rgb)); crop_hashes.add(pixel_sha(cropped)); modes[mode] += 1
        for row in selection["records"]:
            original = queue["records"][row["matlab_train_index"] - 1]
            rgb = pixels(path(ROOT, row["source"]), "RGB")
            cropped = pixels(path(ROOT, row["native_crop"]), "RGB")
            assert sha(ROOT / row["source"]) == row["source_sha256"]
            assert sha(ROOT / row["native_crop"]) == row["native_crop_sha256"]
            assert pixel_sha(rgb) == original["native_rgb_sha256"]
            x0,y0,x1,y1 = original["crop_xyxy_zero_based_half_open"]
            assert np.array_equal(cropped, rgb[y0:y1,x0:x1])
    assert len(train_hashes) == 1345
    inspected_paths = {}
    supported = read(ROOT / "dataset/detector_supported_review_v2/manifest.json")
    for row in supported["supported_records"]:
        for field in ("source", "image"):
            # Preserved legacy source metadata uses Windows separators. Normalize
            # this comparison path without modifying the historical registry.
            p = path(ROOT if field == "source" else ROOT / "dataset/detector_supported_review_v2", row[field].replace("\\", "/"))
            assert sha(p) == row[field + "_sha256"]
            inspected_paths[p] = {"origin": "previous_supported_registry", "split": row["split"]}
    for manifest in ("outputs/broad_covering_gallery_v1/frozen_protocol.json", "outputs/practical_direct_detector_v1/frozen_protocol.json"):
        for row in read(ROOT / manifest)["cases"]:
            for field in ("source", "input"):
                if field in row:
                    p = path(ROOT, row[field])
                    if field + "_sha256" in row: assert sha(p) == row[field + "_sha256"]
                    inspected_paths[p] = {"origin": manifest, "split": "exposed_development_gallery"}
    overlaps = []
    for p, metadata in inspected_paths.items():
        digest = pixel_sha(pixels(p, "RGB"))
        if digest in train_hashes or digest in crop_hashes:
            overlaps.append({"path": p.relative_to(ROOT).as_posix(), **metadata})
    cases, failures = [], []
    for row in labels["records"]:
        width,height = row["crop_native_size"]
        full = raster((width,height), row["native_polygons"], False) & ~raster((width,height), row["native_preserved_holes"], False)
        pil_full = raster((width,height), row["native_polygons"], True) & ~raster((width,height), row["native_preserved_holes"], True)
        mask_image = Image.fromarray(full.astype(np.uint8)*255)
        # Independent Pillow extrema reproduce boundary uncertainty without
        # importing the builder or using its OpenCV morphology implementation.
        dilated = np.asarray(mask_image.filter(ImageFilter.MaxFilter(3))) != 0
        eroded = np.asarray(mask_image.filter(ImageFilter.MinFilter(3))) != 0
        valid = ~(dilated ^ eroded)
        for edge in row["native_unknown_crop_edges"]:
            if edge == "top": valid[0] = False
            elif edge == "bottom": valid[-1] = False
            elif edge == "left": valid[:,0] = False
            elif edge == "right": valid[:,-1] = False
            else: raise ValueError("Unknown crop edge")
        valid &= ~raster((width,height), row["native_unknown_polygons"], False)
        disagreement = (full != pil_full) & valid
        if disagreement.any():
            yy,xx = np.where(disagreement)
            failures.append({"id": row["id"], "cause": "Two native rasterizers disagree on supervised pixels", "pixels_xy": np.column_stack((xx,yy)).tolist()})
        assert int(full.sum()) == row["native_full_pixels"]
        assert int((full & valid).sum()) == row["native_core_pixels"]
        assert int(valid.sum()) == row["native_supervised_pixels"]
        scale = 255 / max(width-1,height-1)
        matrix = np.array([[scale,0,(255-(width-1)*scale)/2],[0,scale,(255-(height-1)*scale)/2]],np.float64)
        assert np.allclose(matrix,row["affine"],atol=1e-12,rtol=0)
        def warp(a):
            return cv2.warpAffine(a.astype(np.uint8),matrix,(256,256),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0) != 0
        source_valid = warp(np.ones(full.shape,bool))
        rgb = pixels(ROOT / row["native_crop"], "RGB")
        image = cv2.warpAffine(rgb,matrix,(256,256),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(96,96,96))
        image[~source_valid] = 96
        targets = {"images": image,"masks":warp(full & valid),"valid":warp(valid),"source_valid":source_valid,"full_proposals":warp(full)}
        for key, expected in targets.items():
            asset = row["files"][key]
            p = path(PROPOSALS,asset["path"])
            assert sha(p) == asset["sha256"]
            actual = pixels(p)
            if key != "images":
                assert actual.dtype == np.uint8 and set(np.unique(actual)) <= {0,255}
                actual = actual == 255
            assert np.array_equal(expected,actual)
        assert bool(targets["masks"].any()) == (row["kind"] == "covered")
        assert not (targets["masks"] & ~targets["valid"]).any()
        assert not (targets["valid"] & ~source_valid).any()
        cases.append({"id": row["id"],"kind":row["kind"],"supervised_native_raster_disagreements":int(disagreement.sum()),
                      "target_pixels":int(targets["masks"].sum()),"valid_pixels":int(targets["valid"].sum())})
    for manifest, base in ((queue,ROOT),(selection,ROOT),(labels,PROPOSALS)):
        for sheet in manifest["sheets"]:
            assert sha(path(base,sheet["path"])) == sheet["sha256"]
    OUT.mkdir()
    report = {"format":"dgp-cofw-source-label-independent-audit-v1","date":"2026-10-03",
              "complete":not failures and not overlaps,"auditor_sha256":sha(__file__),
              "official_rights_record_sha256":sha(acquire / "current_record.json"),"publisher_md5_matched":True,
              "queue_sha256":sha(QUEUE / "manifest.json"),"selection_sha256":sha(SELECTION / "manifest.json"),
              "proposal_sha256":sha(PROPOSALS / "manifest.json"),"native_train_images_verified":1345,
              "unique_train_native_pixel_hashes":len(train_hashes),"native_modes":dict(modes),
              "selected_native_pairs_verified":42,"proposal_assets_verified":210,
              "previous_reference_files_compared":len(inspected_paths),"exact_native_or_crop_rgb_overlaps":overlaps,
              "all_80000_sources_compared":False,"resized_perceptual_identity_overlap_verified":False,
              "test_rgb_read":False,"test_count_metadata":507,"cases":cases,"failures":failures,
              "training_admitted":False,"model_forwards":0,"optimizer_updates":0,
              "independent_expert_semantic_accuracy_verified":False,
              "limitation":"Pixel/geometry/term checks do not approve semantic label accuracy or absence of model-pretraining overlap."}
    with (OUT / "verification.json").open("x",encoding="utf-8",newline="\n") as stream:
        stream.write(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"complete":report["complete"],"geometry_failures":len(failures),"exact_overlaps":len(overlaps),
                      "references_compared":len(inspected_paths),"verification_sha256":sha(OUT / "verification.json")}))


if __name__ == "__main__":
    main()
