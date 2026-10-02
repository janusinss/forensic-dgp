"""Inspect the user-supplied Mendeley candidate without creating training labels."""

import collections
import hashlib
import io
import json
import random
import re
import stat
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "Occluded and Low-Light Human Face Detection Datase.zip"
ARCHIVE_SHA = "2208dfaa4973450073883618ceb8432f4740e9b82fede290c198aa42f51752ac"
OUT = ROOT / "outputs/mendeley_occlusion_candidate_v1/uploaded_source_review"
PREFIX = "Occluded and Low-Light Human Face Detection Datase"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    if OUT.exists():
        raise ValueError("Preserve existing source review; use a new version for changes")
    if sha(ARCHIVE) != ARCHIVE_SHA:
        raise ValueError("Uploaded archive fingerprint differs")
    with zipfile.ZipFile(ARCHIVE) as archive:
        members = archive.infolist()
        if len(members) != 11987 or sum(item.file_size for item in members) > 100_000_000:
            raise ValueError("Uploaded archive count or extraction budget differs")
        normalized = set()
        for item in members:
            # Whitelist this actual archive instead of trusting ZIP paths on Windows.
            if not re.fullmatch(re.escape(PREFIX) + r"/[1-9][0-9]*\.jpg", item.filename):
                raise ValueError(f"Unexpected archive path: {item.filename}")
            name = item.filename.casefold()
            mode = item.external_attr >> 16
            if name in normalized or stat.S_ISLNK(mode) or item.flag_bits & 1:
                raise ValueError("Duplicate, symbolic-link or encrypted ZIP member")
            if not 0 < item.file_size <= 1_000_000:
                raise ValueError("Unexpected member size")
            normalized.add(name)
        members.sort(key=lambda item: int(Path(item.filename).stem))
        ids = [int(Path(item.filename).stem) for item in members]
        if ids != list(range(1, 11988)):
            raise ValueError("Image IDs are not the inspected contiguous range")

        OUT.mkdir(parents=True)
        extracted = OUT / "extracted"
        extracted.mkdir()
        records = []
        failures = []
        byte_groups = collections.defaultdict(list)
        pixel_groups = collections.defaultdict(list)
        dimensions = collections.Counter()
        modes = collections.Counter()
        perceptual_groups = collections.defaultdict(list)
        root_path = extracted.resolve()
        for item in members:
            target = extracted / Path(item.filename).name
            if not target.resolve().is_relative_to(root_path):
                raise ValueError("Extraction target escaped research directory")
            blob = archive.read(item)  # zipfile verifies CRC on each complete read.
            target.write_bytes(blob)
            image_id = int(target.stem)
            file_hash = hashlib.sha256(blob).hexdigest()
            byte_groups[file_hash].append(image_id)
            try:
                with Image.open(io.BytesIO(blob)) as check:
                    check.verify()
                with Image.open(io.BytesIO(blob)) as source:
                    source.load()
                    mode, fmt = source.mode, source.format
                    rgb = np.asarray(source.convert("RGB")).copy()
                height, width = rgb.shape[:2]
                if max(height, width) > 4096 or width * height > 16_000_000:
                    raise ValueError("Unexpected native image dimensions")
                pixel_hash = hashlib.sha256(
                    f"RGB:{width}x{height}:".encode() + rgb.tobytes()
                ).hexdigest()
                small = np.asarray(Image.fromarray(rgb).convert("L").resize((9, 8)))
                dhash = np.packbits(small[:, 1:] > small[:, :-1]).tobytes().hex()
                pixel_groups[pixel_hash].append(image_id)
                perceptual_groups[dhash].append(image_id)
                dimensions[f"{width}x{height}"] += 1
                modes[mode] += 1
                records.append({
                    "id": image_id, "archive_member": item.filename,
                    "image": target.relative_to(ROOT).as_posix(), "bytes": len(blob),
                    "sha256": file_hash, "rgb_pixel_sha256": pixel_hash,
                    "dhash64": dhash, "width": width, "height": height,
                    "mode": mode, "format": fmt,
                    "publisher_split": None, "publisher_occlusion_mask": None,
                    "training_admitted": False, "annotation_reviewed": False,
                })
            except (ValueError, OSError, SyntaxError) as error:
                failures.append({"id": image_id, "error": str(error), "sha256": file_hash})

    # Spread source exposure across sorted unique decoded images, not ZIP order.
    representatives = sorted(min(group) for group in pixel_groups.values())
    count = min(128, len(representatives))
    rng = random.Random(20261002)
    selected = []
    if count:
        for start, end in zip(np.linspace(0, len(representatives), count + 1, dtype=int)[:-1],
                              np.linspace(0, len(representatives), count + 1, dtype=int)[1:]):
            selected.append(representatives[rng.randrange(int(start), int(end))])
    record_map = {record["id"]: record for record in records}
    sheets = []
    for offset in range(0, len(selected), 64):
        sheet = Image.new("RGB", (1024, 1188), "#16181c")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), "SOURCE ONLY: user-uploaded Mendeley candidate; no masks/splits; no model output", fill="white")
        for position, image_id in enumerate(selected[offset:offset + 64]):
            record = record_map[image_id]
            with Image.open(ROOT / record["image"]) as photo:
                native = photo.convert("RGB")
                native.thumbnail((124, 124), Image.Resampling.LANCZOS)
                # Enlarge only for inspection; never record this as source detail.
                if min(native.size) < 90:
                    native = photo.convert("RGB").resize((124, 124), Image.Resampling.NEAREST)
            x, y = position % 8 * 128, 40 + position // 8 * 142
            sheet.paste(native, (x + 2, y))
            draw.text((x + 3, y + 124), f"{image_id}  {record['width']}x{record['height']}", fill="white")
        destination = OUT / f"sources_{offset + 1:03d}_{min(offset + 64, len(selected)):03d}.png"
        sheet.save(destination)
        sheets.append({"path": destination.relative_to(ROOT).as_posix(), "sha256": sha(destination),
                       "ids": selected[offset:offset + 64]})

    report = {
        "format": "dgp-mendeley-uploaded-source-integrity-v1", "date": "2026-10-02",
        "source_url": "https://data.mendeley.com/datasets/s57wnx78vh/1",
        "doi": "10.17632/s57wnx78vh.1", "acquisition": "User uploaded ZIP after automated HTTP 403",
        "archive": ARCHIVE.relative_to(ROOT).as_posix(), "archive_sha256": ARCHIVE_SHA,
        "archive_bytes": ARCHIVE.stat().st_size, "publisher_claimed_images": 12000,
        "publisher_file_checksum_available": False,
        "license_listed_by_publisher": "CC BY 4.0",
        "attribution": "Laxmi Narayan Soni and Akhilesh A Waoo, 2025, DOI 10.17632/s57wnx78vh.1",
        "reader_sha256": sha(__file__), "safe_path_and_crc_verification": True,
        "archive_members": len(members), "image_count": len(records), "decode_failures": failures,
        "dimensions": dict(dimensions), "modes": dict(modes),
        "unique_file_hashes": len(byte_groups), "unique_rgb_pixel_hashes": len(pixel_groups),
        "exact_file_duplicate_groups": [group for group in byte_groups.values() if len(group) > 1],
        "exact_pixel_duplicate_groups": [group for group in pixel_groups.values() if len(group) > 1],
        "dhash_same_value_groups": [group for group in perceptual_groups.values() if len(group) > 1],
        "dhash_note": "Only a review clue; equal perceptual hashes are not proof of duplicates or identity",
        "annotation_files": 0, "publisher_split_files": 0,
        "source_review_sample_ids": selected, "sample_rule": "Seed 20261002, 128 strata over unique RGB representatives sorted by numeric ID",
        "sheets": sheets, "records": records,
        "full_image_family_review_complete": False, "training_admitted": False,
        "new_model_forwards": 0, "optimizer_updates": 0,
        "identity_overlap_verified": False, "full_project_overlap_verified": False,
        "previous_access_status": "outputs/mendeley_occlusion_candidate_v1/access_status.json",
        "status": "Extracted research sources only; dimensions, content and duplicates require review before labels/splits",
    }
    destination = OUT / "integrity.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"images": len(records), "failures": len(failures), "dimensions": dict(dimensions),
                      "unique_files": len(byte_groups), "unique_pixels": len(pixel_groups),
                      "sampled_sources": len(selected), "sheets": len(sheets),
                      "training_admitted": False, "integrity_sha256": sha(destination)}))


if __name__ == "__main__":
    main()
