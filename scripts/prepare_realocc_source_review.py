"""Verify author-split RealOcc source pixels and create source-only review sheets."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "outputs/realocc_source_v1"
ARCHIVE_SHA = "de6a26ecc00457c0067991d95906b15638a2f217c5b5ea01a5db42435510577f"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    source = FOLDER / "extracted/RealOcc"
    out = FOLDER / "source_review"
    if out.exists() or sha(FOLDER / "RealOcc.7z") != ARCHIVE_SHA:
        raise ValueError("Preserve earlier review; source archive must match")
    split_path = source / "split/val.txt"
    split = split_path.read_text(encoding="utf-8").splitlines()
    names = {path.stem for path in (source / "image").glob("*.jpg")}
    mask_names = {path.stem for path in (source / "mask").glob("*.png")}
    if len(split) != len(set(split)) or set(split) != names or names != mask_names or len(names) != 550:
        raise ValueError("Author validation split or pairing differs")
    out.mkdir()
    records, thumbnails = [], []
    for name in sorted(names, key=lambda name: int(name.split("_")[1])):
        image_path, mask_path = source / "image" / f"{name}.jpg", source / "mask" / f"{name}.png"
        for path in (image_path, mask_path):
            with Image.open(path) as image:
                image.verify()
        with Image.open(image_path) as image:
            image.load()
            rgb = np.asarray(image.convert("RGB")).copy()
        with Image.open(mask_path) as image:
            image.load()
            label = np.asarray(image.convert("L")).copy()
        if rgb.shape[:2] != label.shape or not np.isin(label, (0, 1)).all():
            raise ValueError(f"Unexpected native label/dimensions: {name}")
        native = Image.fromarray(rgb).resize((112, 112), Image.Resampling.LANCZOS)
        overlay = rgb.astype(np.float32)
        selected = label == 1
        overlay[selected] = .55 * overlay[selected] + .45 * np.array([255, 120, 20])
        marked = Image.fromarray(overlay.round().astype(np.uint8)).resize((112, 112), Image.Resampling.LANCZOS)
        thumbnails.append((name, native, marked))
        records.append({"id": name, "publisher_split": "val", "image": image_path.relative_to(ROOT).as_posix(),
                        "mask": mask_path.relative_to(ROOT).as_posix(), "image_sha256": sha(image_path),
                        "mask_sha256": sha(mask_path), "width": rgb.shape[1], "height": rgb.shape[0],
                        "mask_values": [0, 1], "label_one_pixels": int(selected.sum()),
                        "label_semantics": "Uninterpreted until source/definition review; value1 must not be assumed to mean occlusion",
                        "training_admitted": False, "new_model_forwards": 0})
    sheets = []
    for start in range(0, len(thumbnails), 50):
        canvas = Image.new("RGB", (2240, 725), "#16181c")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), "SOURCE ONLY: native photo | publisher label value1 orange; original val split; no model outputs", fill="white")
        for index, (name, native, marked) in enumerate(thumbnails[start:start+50]):
            x, y = (index % 10)*224, 40+(index // 10)*135
            canvas.paste(native, (x, y))
            canvas.paste(marked, (x+112, y))
            draw.text((x+4, y+114), name, fill="white")
        path = out / f"sources_{start+1:03d}_{min(start+50,len(thumbnails)):03d}.png"
        canvas.save(path)
        sheets.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)})
    report = {"format": "dgp-realocc-source-integrity-v1", "date": "2026-10-02",
              "archive_sha256": ARCHIVE_SHA, "reader_sha256": sha(__file__), "split_sha256": sha(split_path),
              "image_count": len(records), "mask_count": len(records), "split_count": len(split),
              "source_integrity_verified": True, "records": records, "sheets": sheets,
              "publisher_membership_preserved": True, "new_model_forwards": 0, "optimizer_updates": 0,
              "training_admitted": False, "public_terms_verified": False,
              "scope_review_pending": True, "identity_overlap_verified": False,
              "note": "Public author dataset for developmental source review; not yet project training supervision, hidden facial truth or pristine holdout"}
    destination = out / "integrity.json"
    destination.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"verified_pairs": len(records), "sheets": len(sheets), "training_admitted": False,
                      "integrity_sha256": sha(destination)}))


if __name__ == "__main__":
    main()
