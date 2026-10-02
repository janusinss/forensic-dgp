"""Version practical removal footprints separately from immutable detector labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PROTOCOL = ROOT / "outputs/practical_gallery_v2/frozen_native_protocol_v1.json"
SOURCE_SHA = "74b80e0ec9ebabb7c8b3b3989f5bc4f44b9a4cf1e39e691c073543f2db4651a5"
SELECTED = (2, 3, 5, 6, 8, 9)
# Source-only 256px operator geometry. It contains no hidden-face target.
CHANGES = {
    "02_mask_with_clear_glasses_native": {
        "purpose": "Include the whole blue covering and facial ear-loop segment; retain visible clear eyewear",
        "radius": 3,
        "polygons": [[(96,181),(99,163),(112,149),(128,141),(142,134),(155,120),(163,120),
                      (173,128),(181,145),(190,157),(190,179),(184,199),(172,216),(154,227),
                      (135,230),(119,224),(104,208)]],
        "removal_lines": [{"points": [(60,122),(72,136),(91,158),(110,176)], "width": 5}],
        "preserved_lines": [{"points": [(173,119),(171,127),(168,133),(156,139),(143,143)], "width": 2}],
        "pose_note": "Three-quarter turn; retained as a difficult developmental diagnostic, not proof of frontal/mild-pose coverage",
    },
    "03_dark_sunglasses_native": {
        "purpose": "Remove opaque lenses plus complete visible rim and bridge with a small skin margin",
        "radius": 3,
        "polygons": [[(24,90),(39,83),(59,82),(80,85),(103,88),(122,100),(135,100),(146,95),
                      (170,90),(194,88),(220,88),(236,90),(246,105),(248,128),(238,148),
                      (220,158),(190,162),(158,160),(138,153),(134,129),(121,132),(116,150),
                      (92,157),(70,158),(50,154),(32,148),(22,132),(21,109)]],
        "removal_lines": [], "preserved_lines": [], "pose_note": "Frontal",
    },
    "05_white_glare_native": {
        "purpose": "Estimate obscured lens interiors while retaining clear-frame edges; not a brightness threshold",
        "radius": 0,
        "polygons": [[(78,89),(84,85),(95,84),(109,88),(123,95),(133,101),(131,110),
                      (123,119),(111,123),(96,121),(83,116),(77,105)],
                     [(149,104),(160,104),(177,110),(193,117),(197,126),(194,135),
                      (183,140),(170,138),(159,132),(152,122)]],
        "removal_lines": [], "preserved_lines": [], "pose_note": "Mild turn/roll",
    },
    "06_mirrored_glare_native": {
        "purpose": "Remove mirrored sunglasses including both outer circular rims and bridge",
        "radius": 3,
        "polygons": [[(65,118),(74,111),(91,109),(108,111),(125,118),(124,138),(118,152),
                      (108,158),(91,162),(79,160),(65,148),(62,133)],
                     [(136,119),(151,111),(168,109),(189,110),(198,116),(201,133),
                      (193,149),(181,157),(159,160),(144,154),(137,141)],
                     [(122,118),(141,116),(146,128),(132,130),(123,128)]],
        "removal_lines": [], "preserved_lines": [], "pose_note": "Frontal",
    },
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, mode):
    with Image.open(path) as value:
        return np.asarray(value.convert(mode)).copy()


def overlay(image, mask):
    value = image.astype(np.float32)
    value[mask] = .55 * value[mask] + .45 * np.array([255, 120, 20])
    return Image.fromarray(value.round().astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs/practical_footprints_v3")
    args = parser.parse_args()
    folder = args.output_dir.resolve()
    if folder.exists() or not folder.is_relative_to(ROOT):
        raise ValueError("Use a fresh proposal directory in the workspace")
    if sha(SOURCE_PROTOCOL) != SOURCE_SHA:
        raise ValueError("Frozen source protocol changed")
    p = json.loads(SOURCE_PROTOCOL.read_text(encoding="utf-8"))
    original = ROOT / p["gallery_root"]
    for name, expected in p["assets_sha256"].items():
        if sha(original / name) != expected:
            raise ValueError(f"Frozen source asset changed: {name}")
    folder.mkdir()
    (folder / "masks").mkdir()
    cases, assets, tiles = [], {}, []
    for position in SELECTED:
        case = p["cases"][position]
        rgb = read(original / case["input"], "RGB")
        old = read(original / case["removal_proposal"], "L") == 255
        correction = CHANGES.get(case["id"])
        proposal = Image.fromarray(old.astype(np.uint8) * 255)
        protected = Image.new("L", (256, 256))
        if correction:
            paint = ImageDraw.Draw(proposal)
            for polygon in correction["polygons"]:
                paint.polygon(polygon, fill=255)
            for line in correction["removal_lines"]:
                paint.line(line["points"], fill=255, width=line["width"])
            if correction["radius"]:
                proposal = proposal.filter(ImageFilter.MaxFilter(2 * correction["radius"] + 1))
            protect = ImageDraw.Draw(protected)
            for line in correction["preserved_lines"]:
                protect.line(line["points"], fill=255, width=line["width"])
        new = np.asarray(proposal).copy() == 255
        keep = np.asarray(protected) == 255
        new[keep] = False
        if not correction and new.any():
            raise ValueError("Clear controls must remain empty")
        output = folder / "masks" / f"{case['id']}.png"
        Image.fromarray(new.astype(np.uint8) * 255).save(output)
        preserved_path = folder / "masks" / f"{case['id']}_preserved.png"
        protected.save(preserved_path)
        entry = dict(case)
        entry.update({
            "input": (original / case["input"]).relative_to(ROOT).as_posix(),
            "old_removal": (original / case["removal_proposal"]).relative_to(ROOT).as_posix(),
            "new_removal": output.relative_to(ROOT).as_posix(),
            "preserved_region": preserved_path.relative_to(ROOT).as_posix(),
            "operator_change": correction, "old_pixels": int(old.sum()), "new_pixels": int(new.sum()),
            "added_pixels": int((new & ~old).sum()), "excluded_preserved_pixels": int((old & ~new).sum()),
            "original_label_unchanged": True,
        })
        cases.append(entry)
        tiles.append((case["id"], rgb, old, new, keep))
        for path in (original / case["input"], original / case["removal_proposal"], output, preserved_path):
            assets[path.relative_to(ROOT).as_posix()] = sha(path)
    canvas = Image.new("RGB", (1024, 1710), "#16181c")
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(("source", "frozen v2 removal", "draft v3 removal", "preserved eyewear")):
        draw.text((column * 256 + 6, 10), title, fill="white")
    for index, (title, rgb, old, new, protected) in enumerate(tiles):
        y = 42 + index * 276
        for column, tile in enumerate((Image.fromarray(rgb), overlay(rgb, old), overlay(rgb, new), overlay(rgb, protected))):
            canvas.paste(tile, (column * 256, y))
        draw.text((10, y + 258), title, fill="white")
    canvas.save(folder / "source_mask_review.png")
    for start in (0, 3):
        canvas.crop((0, 42 + start * 276, 1024, 42 + (start + 3) * 276)).save(folder / f"source_rows_{start+1:02d}_{start+3:02d}.png")
    manifest = {
        "format": "dgp-practical-removal-footprint-v3", "date": "2026-10-02",
        "status": "source_review_draft", "source_protocol_sha256": SOURCE_SHA,
        "source_protocol": SOURCE_PROTOCOL.relative_to(ROOT).as_posix(),
        "preparation_sha256": sha(__file__), "cases": cases, "asset_sha256": assets,
        "scope": "Four targeted covering failures plus two clear controls; previously inspected train sources, not a pristine holdout",
        "reviewer": "Assistant operator polygons from observed source only; approximate practical removal proposals, not expert detector labels",
        "exposure": "Earlier outputs informed which cases need correction; new geometry is source-based and must be frozen before new outputs",
        "hidden_face_ground_truth": False, "optimizer_updates": 0, "model_forwards": 0,
        "missing_families": p["missing_families"], "promoted": False,
    }
    (folder / "draft.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prepared": True, "cases": len(cases), "changed_cases": len(CHANGES),
                      "models_run": 0, "draft_sha256": sha(folder / "draft.json")}))


if __name__ == "__main__":
    main()
