"""Inventory existing training examples for practical output review, without ML.

Creates descriptive evidence only; never changes the source manifest, labels,
split membership or model state. Contact sheets contain TRAINING cases only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_file(base: Path, value: str) -> Path:
    relative = Path(value.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Expected a portable relative path: {value}")
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()) or not path.is_file():
        raise ValueError(f"Missing or escaping source file: {value}")
    return path


def checked_file(base: Path, row: dict, key: str) -> Path:
    path = relative_file(base, row[key])
    if sha(path) != row[f"{key}_sha256"]:
        raise ValueError(f"{key} fingerprint differs: {row[key]}")
    return path


def load_font(size: int):
    for value in ("C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(value, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "dataset/detector_supported_review_v1/manifest.json")
    parser.add_argument("--output_dir", type=Path,
                        default=ROOT / "outputs/practical_coverage_v1")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError("Output must be a new directory inside the workspace")
    manifest_sha = sha(manifest)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("format") != "dgp-supported-real-masks-v1":
        raise ValueError("Expected supported real-mask manifest")
    rows = data["supported_records"]
    group_splits = defaultdict(set)
    source_splits = defaultdict(set)
    for row in rows:
        group_splits[row["group"]].add(row["split"])
        source_splits[row["source_sha256"]].add(row["split"])
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError("Source group crosses existing splits")
    if any(len(splits) != 1 for splits in source_splits.values()):
        raise ValueError("Exact source bytes cross existing splits")

    entries, thumbnails = [], []
    for index, row in enumerate(rows):
        if row["split"] != "train":
            continue
        paths = {key: checked_file(manifest.parent, row, key)
                 for key in ("image", "mask", "valid", "source_valid")}
        source = checked_file(ROOT, row, "source")
        with Image.open(source) as native:
            native_size = list(native.size)
            native_thumb = ImageOps.contain(native.convert("RGB"), (168, 168))
        with Image.open(paths["image"]) as crop:
            crop_size = list(crop.size)
            crop_thumb = ImageOps.contain(crop.convert("RGB"), (168, 168))
        entries.append({
            "case_id": f"supported_{index:03d}", "manifest_index": index,
            "split": row["split"], "kind": row["kind"],
            "image": row["image"], "image_sha256": row["image_sha256"],
            "mask": row["mask"], "mask_sha256": row["mask_sha256"],
            "valid": row["valid"], "valid_sha256": row["valid_sha256"],
            "source": row["source"], "source_sha256": row["source_sha256"],
            "group": row["group"], "native_size": native_size,
            "crop_size": crop_size, "existing_stratum": row.get("occlusion_stratum"),
            "existing_glare_stratum": row.get("glare_stratum"),
            "data_origin": row["data_origin"],
            "exposure": "previously reviewed; detector-training source; not pristine holdout",
            "practical_visual_review": None,
        })
        thumbnails.append((native_thumb, crop_thumb))

    output.mkdir(parents=True)
    font, small = load_font(15), load_font(12)
    pages = []
    for start in range(0, len(entries), 20):
        selected = entries[start:start + 20]
        canvas = Image.new("RGB", (1440, 1180), "#16181c")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 12), "TRAINING SOURCE COVERAGE ONLY — native source / existing crop",
                  font=font, fill="white")
        for position, row in enumerate(selected):
            x, y = (position % 4) * 360, 50 + (position // 4) * 220
            native, crop = thumbnails[start + position]
            for offset, thumb in ((8, native), (184, crop)):
                canvas.paste(thumb, (x + offset + (168 - thumb.width) // 2,
                                    y + (168 - thumb.height) // 2))
            draw.text((x + 8, y + 174), f"#{row['manifest_index']:03d} {row['kind']}",
                      font=font, fill="#c8e3ed")
            draw.text((x + 8, y + 194), Path(row["image"]).name[:43],
                      font=small, fill="white")
        page = output / f"training_sources_{start // 20 + 1:02d}.png"
        canvas.save(page)
        pages.append({"file": page.name, "sha256": sha(page),
                      "case_ids": [row["case_id"] for row in selected]})

    if sha(manifest) != manifest_sha:
        raise ValueError("Source manifest changed during inventory")
    report = {
        "format": "dgp-practical-coverage-inventory-v1", "date": "2026-10-02",
        "manifest": manifest.relative_to(ROOT).as_posix(),
        "manifest_sha256": manifest_sha,
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "group_split_integrity": "no cross-split group or exact-source-byte overlap",
        "scope": "83 existing training records; metadata-only for other splits",
        "training_counts": dict(Counter(row["kind"] for row in entries)),
        "existing_strata": dict(Counter(row["existing_stratum"] or "untyped" for row in entries)),
        "new_model_forwards": 0, "optimizer_updates": 0,
        "source_mutations": 0, "gallery_frozen": False,
        "pages": pages, "records": entries,
    }
    (output / "inventory.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "training_records": len(entries),
                      "kind_counts": report["training_counts"], "pages": len(pages),
                      "manifest_sha256": manifest_sha, "model_forwards": 0,
                      "gallery_frozen": False}))


if __name__ == "__main__":
    main()
