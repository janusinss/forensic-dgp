"""Independent archive/extraction inventory verification; no model or optimizer."""
import collections
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/mendeley_occlusion_candidate_v1/uploaded_source_review"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    destination = OUT / "independent_verification.json"
    if destination.exists():
        raise ValueError("Preserve prior verification")
    manifest_path = OUT / "integrity.json"
    data = json.loads(manifest_path.read_text())
    archive_path = ROOT / data["archive"]
    assert sha(archive_path) == data["archive_sha256"]
    files = list((OUT / "extracted").iterdir())
    assert {path.name for path in files} == {str(item["id"]) + ".jpg" for item in data["records"]}
    file_groups, pixel_groups = collections.defaultdict(list), collections.defaultdict(list)
    dimensions = collections.Counter()
    with zipfile.ZipFile(archive_path) as archive:
        members = {item.filename: item for item in archive.infolist()}
        assert len(members) == data["archive_members"] == 11987
        assert all(Path(name).suffix.lower() == ".jpg" for name in members)
        assert {row["archive_member"] for row in data["records"]} == set(members)
        for row in data["records"]:
            path = (ROOT / row["image"]).resolve()
            assert path.is_relative_to((OUT / "extracted").resolve())
            blob = path.read_bytes()
            assert blob == archive.read(row["archive_member"])
            digest = hashlib.sha256(blob).hexdigest()
            assert digest == row["sha256"] and len(blob) == row["bytes"]
            with Image.open(path) as source:
                source.load()
                width, height = source.size
                rgb = source.convert("RGB").tobytes()
                assert source.mode == row["mode"] and source.format == row["format"]
            pixel_hash = hashlib.sha256(f"RGB:{width}x{height}:".encode() + rgb).hexdigest()
            assert pixel_hash == row["rgb_pixel_sha256"]
            assert (width, height) == (row["width"], row["height"])
            file_groups[digest].append(row["id"])
            pixel_groups[pixel_hash].append(row["id"])
            dimensions[f"{width}x{height}"] += 1
    assert len(file_groups) == data["unique_file_hashes"] == 6102
    assert len(pixel_groups) == data["unique_rgb_pixel_hashes"] == 6101
    assert dict(dimensions) == data["dimensions"] and data["decode_failures"] == []
    assert sorted(sorted(g) for g in file_groups.values() if len(g) > 1) == sorted(data["exact_file_duplicate_groups"])
    assert sorted(sorted(g) for g in pixel_groups.values() if len(g) > 1) == sorted(data["exact_pixel_duplicate_groups"])
    previous = ROOT / "dataset/detector_supported_review_v1/manifest.json"
    accepted = json.loads(previous.read_text())
    previous_hashes = {row["source_sha256"] for row in accepted["supported_records"]}
    overlaps = sorted(previous_hashes & set(file_groups))
    assert not overlaps
    compared_raw = 0
    pixel_overlaps = []
    for row in accepted["supported_records"]:
        path = ROOT / row["source"].replace("\\", "/")
        assert sha(path) == row["source_sha256"]
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            pixels = image.convert("RGB").tobytes()
        key = hashlib.sha256(f"RGB:{width}x{height}:".encode() + pixels).hexdigest()
        if key in pixel_groups:
            pixel_overlaps.append({"source": row["source"], "upload_ids": pixel_groups[key]})
        compared_raw += 1
    assert not pixel_overlaps
    for sheet in data["sheets"]:
        assert sha(ROOT / sheet["path"]) == sheet["sha256"]
    report = {"format": "dgp-mendeley-source-independent-verification-v1", "date": "2026-10-02",
              "complete": True, "integrity_sha256": sha(manifest_path), "auditor_sha256": sha(__file__),
              "archive_sha256": sha(archive_path), "files_verified": len(files),
              "unique_file_hashes": len(file_groups), "unique_rgb_pixel_hashes": len(pixel_groups),
              "dimensions": dict(dimensions), "exact_duplicate_groups_verified": True,
              "previous_supported_manifest_sha256": sha(previous), "previous_sources_compared": compared_raw,
              "exact_file_overlap": overlaps, "native_rgb_pixel_overlap": pixel_overlaps,
              "full_80000_prior_overlap_verified": False, "identity_overlap_verified": False,
              "occlusion_masks_in_archive": 0, "splits_in_archive": 0,
              "model_forwards": 0, "optimizer_updates": 0, "training_admitted": False}
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": True, "files_verified": len(files), "previous_sources_compared": compared_raw,
                      "unique_pixels": len(pixel_groups), "verification_sha256": sha(destination)}))


if __name__ == "__main__":
    main()
