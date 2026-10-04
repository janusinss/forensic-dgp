"""Freeze native source crops for a covering-label draft; no masks or training."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "outputs/cofw_training_sources_v2"
OUT = ROOT / "outputs/cofw_annotation_sources_v1"
QUEUE_SHA = "a8479ebfe3055f8aeb353441cfc91816d4beed115b0240a892f61465f0091834"
GROUPS = {
    "hand": [1030, 935, 955, 981, 1324, 906, 975, 1001, 875, 920],
    "obstructing_hair": [868, 1217, 1338, 913, 866, 1063, 1343],
    "cloth_or_scarf": [1068, 1057, 978, 1092, 876],
    "object": [1212, 922, 869, 954, 901],
    "sunglasses": [854, 917, 1143],
    "face_mask_preserve_clear_goggles": [930],
    "uncovered_control": [1, 64, 128, 306, 369, 425, 532, 644],
    "clear_glasses_control": [246, 1310, 834],
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    sys.path.insert(0, str(ROOT / "outputs/cofw_read_dependencies"))
    import h5py
    import numpy as np
    from PIL import Image, ImageDraw

    if OUT.exists() or sha(QUEUE / "manifest.json") != QUEUE_SHA:
        raise ValueError("Preserve previous annotation sources; require frozen queue")
    data = json.loads((QUEUE / "manifest.json").read_text())
    rows = {row["matlab_train_index"]: row for row in data["records"]}
    if sha(ROOT / data["matrix"]) != data["matrix_sha256"]:
        raise ValueError("Verified source matrix changed")
    OUT.mkdir()
    (OUT / "native").mkdir()
    (OUT / "source").mkdir()
    selected = []
    with h5py.File(ROOT / data["matrix"], "r") as file:
        for family, ids in GROUPS.items():
            for index in ids:
                row = rows[index]
                raw = np.asarray(file[row["hdf5_reference"]])
                rgb = np.ascontiguousarray(raw.transpose(2, 1, 0)) if raw.ndim == 3 else np.repeat(raw.T[..., None], 3, axis=2)
                width, height = row["native_size"]
                digest = hashlib.sha256(f"RGB:{width}x{height}:".encode() + rgb.tobytes()).hexdigest()
                if digest != row["native_rgb_sha256"]:
                    raise ValueError("Native source pixels changed")
                source = OUT / "source" / f"train_{index:04d}.png"
                Image.fromarray(rgb).save(source)
                x0, y0, x1, y1 = row["crop_xyxy_zero_based_half_open"]
                crop = rgb[y0:y1, x0:x1].copy()
                native = OUT / "native" / f"train_{index:04d}.png"
                Image.fromarray(crop).save(native)
                selected.append({**row, "family_proposal": family,
                    "source": source.relative_to(ROOT).as_posix(), "source_sha256": sha(source),
                    "native_crop": native.relative_to(ROOT).as_posix(), "native_crop_sha256": sha(native),
                    "annotation_status": "source-review-draft", "training_admitted": False,
                    "pose_review": None, "hidden_face_target_available": False,
                    "group": "cofw-author-training-selected-cohort-only-v1"})
    pages = []
    # Native crops enlarged to2x with a coordinate grid, never added source detail.
    for start in range(0, len(selected), 6):
        entries = selected[start:start + 6]
        sheet = Image.new("RGB", (1200, 1100), "#16181c")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), "Native cropped SOURCE ONLY; 2x nearest inspection; cyan native coordinates; no generated pixels", fill="white")
        for pos, row in enumerate(entries):
            x, y = pos % 3 * 400, 38 + pos // 3 * 526
            with Image.open(ROOT / row["native_crop"]) as image:
                native = image.convert("RGB")
                enlarged = native.resize((native.width * 2, native.height * 2), Image.Resampling.NEAREST)
            if enlarged.width > 390 or enlarged.height > 464:
                raise ValueError("Selected native crop exceeds contact-sheet geometry")
            sheet.paste(enlarged, (x + 4, y + 27))
            draw.text((x + 4, y), f"#{row['matlab_train_index']:04d} {row['family_proposal'][:29]}", fill="white")
            for gx in range(0, native.width, 20):
                draw.line((x + 4 + gx * 2, y + 27, x + 4 + gx * 2, y + 27 + native.height * 2 - 1), fill="#398ba4")
                draw.text((x + 4 + gx * 2, y + 13), str(gx), fill="#73d8f2")
            for gy in range(0, native.height, 20):
                draw.line((x + 4, y + 27 + gy * 2, x + 4 + native.width * 2 - 1, y + 27 + gy * 2), fill="#398ba4")
                draw.text((x + 7, y + 29 + gy * 2), str(gy), fill="#73d8f2")
        target = OUT / f"native_rows_{start + 1:02d}_{start + len(entries):02d}.png"
        sheet.save(target)
        pages.append({"path": target.relative_to(ROOT).as_posix(), "sha256": sha(target), "ids": [row["id"] for row in entries]})
    result = {"format": "dgp-cofw-native-annotation-source-draft-v1", "date": "2026-10-03",
              "queue_sha256": QUEUE_SHA, "matrix_sha256": data["matrix_sha256"],
              "builder_sha256": sha(__file__), "records": selected, "sheets": pages,
              "families_are_proposals_not_final_labels": True, "source_count": len(selected),
              "license_record": "outputs/cofw_source_acquisition_v3/current_record.json",
              "license_record_sha256": sha(ROOT / "outputs/cofw_source_acquisition_v3/current_record.json"),
              "source_url": data["record_url"], "doi": data["doi"], "attribution": data["attribution"],
              "publisher_training_split_only": True, "test_rgb_read": False,
              "training_admitted": False, "model_forwards": 0, "optimizer_updates": 0,
              "exposure": "Assistant-inspected author-training sources. No pristine/unseen claim. Prior/gallery source overlap audit pending."}
    with (OUT / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"selected_native_sources": len(selected), "sheets": len(pages),
                      "manifest_sha256": sha(OUT / "manifest.json"), "training_admitted": False}))


if __name__ == "__main__":
    main()
