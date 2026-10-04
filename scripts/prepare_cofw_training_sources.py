"""Prepare a deterministic COFW TRAINING review queue; no removal targets or ML."""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/cofw_source_review_v1"
OUT = ROOT / "outputs/cofw_training_sources_v2"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pixel_sha(rgb):
    height, width = rgb.shape[:2]
    return hashlib.sha256(f"RGB:{width}x{height}:".encode() + rgb.tobytes()).hexdigest()


def write(path, data):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(data, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Output must stay inside the workspace")
    sys.path.insert(0, str(ROOT / "outputs/cofw_read_dependencies"))
    import h5py
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps

    if output.exists():
        raise ValueError("Preserve previous source queue; use a new version")
    extraction = json.loads((SOURCE / "extraction.json").read_text())
    expected = {row["file"]: row["sha256"] for row in extraction["files"]}
    mat = SOURCE / "COFW_train_color.mat"
    if sha(mat) != expected[mat.name]:
        raise ValueError("Original training matrix changed")
    output.mkdir()
    (output / "crops").mkdir()
    records, pixel_groups, image_sizes = [], defaultdict(list), Counter()
    with h5py.File(mat, "r") as file:
        boxes = np.asarray(file["bboxesTr"]).T
        shapes = np.asarray(file["phisTr"]).T
        refs = np.asarray(file["IsTr"]).reshape(-1)
        if boxes.shape != (1345, 4) or shapes.shape != (1345, 87) or len(refs) != 1345:
            raise ValueError("Publisher training count/schema differs")
        if not np.isfinite(boxes).all() or not np.isfinite(shapes).all() or set(np.unique(shapes[:, 58:])) != {0., 1.}:
            raise ValueError("Invalid box/landmark metadata")
        for index, ref in enumerate(refs):
            stored = np.asarray(file[ref])
            if stored.dtype != np.uint8:
                raise ValueError("Expected MATLAB uint8 pixels")
            # HDF5 stores MATLAB dimension order reversed: C,W,H -> H,W,C.
            if stored.ndim == 3 and stored.shape[0] == 3:
                rgb = np.ascontiguousarray(stored.transpose(2, 1, 0))
                native_mode = "RGB"
            elif stored.ndim == 2:
                rgb = np.repeat(stored.T[..., None], 3, axis=2)
                native_mode = "L"
            else:
                raise ValueError("Unexpected MATLAB image shape")
            height, width = rgb.shape[:2]
            x, y, w, h = boxes[index]
            if min(w, h) <= 0 or min(width, height) <= 1:
                raise ValueError("Degenerate face/image geometry")
            # Publisher positions use MATLAB's 1-based pixel coordinates.
            # A 12% box margin includes visible edge context. There is no alignment,
            # inferred hidden anatomy, or mask derived from sparse landmark flags.
            crop = [max(0, math.floor(x - 1 - .12 * w)),
                    max(0, math.floor(y - 1 - .12 * h)),
                    min(width, math.ceil(x - 1 + 1.12 * w)),
                    min(height, math.ceil(y - 1 + 1.12 * h))]
            if crop[2] <= crop[0] or crop[3] <= crop[1]:
                raise ValueError("Publisher box lies outside image")
            photo = Image.fromarray(rgb)
            native_crop = photo.crop(crop)
            canvas = Image.new("RGB", (256, 256), (96, 96, 96))
            thumb = ImageOps.contain(native_crop, (256, 256), Image.Resampling.BICUBIC)
            canvas.paste(thumb, ((256 - thumb.width) // 2, (256 - thumb.height) // 2))
            dest = output / "crops" / f"train_{index + 1:04d}.png"
            canvas.save(dest)
            fingerprint = pixel_sha(rgb)
            pixel_groups[fingerprint].append(index + 1)
            image_sizes[f"{width}x{height}"] += 1
            records.append({
                "id": f"cofw_train_{index + 1:04d}", "matlab_train_index": index + 1,
                "publisher_split": "train", "project_admission": "source-review-only",
                "native_size": [width, height], "native_rgb_sha256": fingerprint,
                "native_mode": native_mode,
                "hdf5_reference": file[ref].name, "matlab_bbox_xywh": boxes[index].tolist(),
                "landmarks_matlab_xy": np.column_stack((shapes[index, :29], shapes[index, 29:58])).tolist(),
                "landmark_occlusion_flags": shapes[index, 58:].astype(int).tolist(),
                "occluded_landmarks": int(shapes[index, 58:].sum()),
                "crop_xyxy_zero_based_half_open": crop,
                "crop_native_size": list(native_crop.size),
                "inspection_crop": dest.relative_to(ROOT).as_posix(), "inspection_crop_sha256": sha(dest),
                "covering_family_review": None, "removal_mask": None,
                "training_admitted": False, "hidden_face_target_available": False,
            })

    # Broad initial semantic review, selected using only publisher TRAIN metadata.
    # Coverage is not claimed until source inspection and separate annotations.
    covered = [row for row in records if row["occluded_landmarks"] >= 4 and min(row["crop_native_size"]) >= 75]
    ranks = sorted(covered, key=lambda row: (-row["occluded_landmarks"], row["matlab_train_index"]))
    chosen = ranks[:144]
    control = [row for row in records if row["occluded_landmarks"] == 0 and min(row["crop_native_size"]) >= 100]
    control_ids = np.linspace(0, len(control) - 1, 16, dtype=int)
    chosen += [control[int(i)] for i in control_ids]
    if len({row["id"] for row in chosen}) != len(chosen):
        raise ValueError("Review queue duplicates")
    pages = []
    for start in range(0, len(chosen), 40):
        rows = chosen[start:start + 40]
        sheet = Image.new("RGB", (1280, 1030), "#16181c")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), "COFW publisher TRAIN ONLY: source inspection; sparse flags are NOT removal masks", fill="white")
        for pos, row in enumerate(rows):
            x, y = pos % 8 * 160, 36 + pos // 8 * 198
            with Image.open(ROOT / row["inspection_crop"]) as im:
                sheet.paste(im.resize((156, 156), Image.Resampling.LANCZOS), (x + 2, y))
            draw.text((x + 3, y + 159), f"#{row['matlab_train_index']:04d} flags:{row['occluded_landmarks']}", fill="white")
            draw.text((x + 3, y + 176), f"crop {row['crop_native_size'][0]}x{row['crop_native_size'][1]}", fill="#b9c7cf")
        dest = output / f"source_rows_{start + 1:03d}_{start + len(rows):03d}.png"
        sheet.save(dest)
        pages.append({"path": dest.relative_to(ROOT).as_posix(), "sha256": sha(dest), "ids": [row["id"] for row in rows]})
    report = {
        "format": "dgp-cofw-training-source-queue-v1", "date": "2026-10-03",
        "matrix": mat.relative_to(ROOT).as_posix(), "matrix_sha256": sha(mat),
        "archive_sha256": extraction["archive_sha256"], "builder_sha256": sha(__file__),
        "record_url": "https://data.caltech.edu/records/bc0bf-nc666", "doi": "10.22002/D1.20099",
        "publisher_rights_snapshot": "outputs/cofw_source_research_v1/record.json",
        "publisher_rights_snapshot_sha256": sha(ROOT / "outputs/cofw_source_research_v1/record.json"),
        "license_listed_by_publisher": "CC BY 4.0",
        "attribution": "Xavier Burgos-Artizzu, Pietro Perona, Piotr Dollar; COFW; CaltechDATA DOI10.22002/D1.20099; Robust face landmark estimation under occlusion, ICCV2013",
        "train_sources": len(records), "decoded_unique_train_images": len(pixel_groups),
        "exact_train_pixel_duplicate_groups": [group for group in pixel_groups.values() if len(group) > 1],
        "native_dimensions": dict(image_sizes), "initial_review_queue_count": len(chosen),
        "queue_rule": "144 eligible training images ranked by sparse-flag count descending/index ascending, plus16 deterministically spaced zero-flag candidate controls",
        "records": records, "sheets": pages,
        "publisher_test_count_metadata": 507, "test_rgb_read": False,
        "annotation_semantics": "No removal labels. Landmark flags guide source-only review; clear control status needs visual review.",
        "training_admitted": False, "model_forwards": 0, "optimizer_updates": 0,
        "prior_project_pixel_overlap_verified": False, "identity_overlap_verified": False,
        "generator_pretraining_overlap_verified": False,
    }
    write(output / "manifest.json", report)
    print(json.dumps({"train_sources": len(records), "unique_train_images": len(pixel_groups),
                      "source_review_queue": len(chosen), "sheets": len(pages),
                      "training_admitted": False, "manifest_sha256": sha(output / "manifest.json")}), flush=True)


if __name__ == "__main__":
    main()
