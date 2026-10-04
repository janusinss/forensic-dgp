"""Freeze native CCTV development/reserved cases without inference or optimization."""
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import time
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/cctv_native_development_v1"
ARCHIVE = ROOT / "dataset/cctv_survface_raw/QMUL-SurvFace-v1.zip"
ARCHIVE_PIN = "2fbb0876bc4761217c6de5576905e2524b8ca50ad7905720a5b4b378a0ff8e13"
AUDIT = ROOT / "outputs/cctv_survface_structure_audit_v1/verification.json"
AUDIT_PIN = "cf9124983a815776a787eeae577bbbbd173892be2f69dc33c354626641863222"
BINS = ("le15", "16to23", "24to39", "ge40")
SEED = "dgp-native-cctv-v1-2026-10-03"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, data):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + "\n")


def bucket(width, height):
    side = min(width, height)
    return "le15" if side <= 15 else "16to23" if side <= 23 else "24to39" if side <= 39 else "ge40"


def rank(member):
    return hashlib.sha256((SEED + "\0" + member).encode("utf-8")).hexdigest()


def main():
    if OUT.exists():
        raise ValueError("Preserve earlier partial/completed gallery")
    if sha(ARCHIVE) != ARCHIVE_PIN or sha(AUDIT) != AUDIT_PIN:
        raise ValueError("Native source or independent structure audit changed")
    proof = json.loads(AUDIT.read_text(encoding="utf-8"))
    if not proof["complete"] or proof["labeled_train_test_person_overlap"] != 0:
        raise ValueError("Require verified labeled release split")
    OUT.mkdir()
    policy = {
        "format": "native-cctv-selection-policy-v1", "date": "2026-10-03",
        "archive_sha256": ARCHIVE_PIN, "release_audit_sha256": AUDIT_PIN,
        "selector_sha256": sha(Path(__file__)), "seed": SEED,
        "bins_min_native_side_pixels": {"le15": [1, 15], "16to23": [16, 23], "24to39": [24, 39], "ge40": [40, None]},
        "development": {"release_role": "training_set", "per_bin": 6, "total": 24},
        "reserved_evaluation": {"release_role": "Face_Identification_Test_Set/gallery", "per_bin": 8, "total": 32},
        "selection": "SHA256 seed plus full member-name ordering; first unused labeled identity per unfilled size bin. No visual or model-output selection.",
        "budget": {"candidate_headers_per_role": 4096, "selection_wall_seconds": 120,
                   "selected_jpeg_copies": 56, "model_forwards": 0, "optimizer_updates": 0},
        "reserved_exposure": "Encoded byte hashes and JPEG dimensions audited; no rendering, contact sheet or model inference",
        "preprocessing": "Keep native JPEG unchanged; center-pad shorter axis with RGB128 at native size, then Pillow bilinear to256; no alignment, denoise, enhancement, synthetic degradation or extra crop",
        "review_criteria": ["Document frontal/mild versus out-of-scope pose from development inputs before model outputs",
                            "Document whether visible facial structure is sufficient; native size bins are diagnostic, not rejection thresholds",
                            "Compare useful clarity while preserving apparent eyes, nose, mouth, contour and visible appearance",
                            "Reject added generic anatomy, strong artifacts or covering-driven unsupported identity claims",
                            "Inspect raw output separately from legacy display processing; no clean paired CCTV target is available"],
        "limitations": ["Stratified development sample; not proportional population accuracy or original recognition protocol",
                        "Individual country/ethnicity unavailable; native CCTV dataset contains varied pose and occlusion",
                        "Release-labeled identity separation does not prove non-overlap with historical model training",
                        "Reserved sample becomes development evidence if inspected/tuned upon; independent final review still required"],
        "training": False, "application_change": False,
    }
    write(OUT / "selection_policy.json", policy)
    start = time.monotonic()
    rows, scan = [], {}
    with ZipFile(ARCHIVE) as archive:
        for role, prefix, quota in (("development", "QMUL-SurvFace/training_set/", 6),
                                    ("reserved_evaluation", "QMUL-SurvFace/Face_Identification_Test_Set/gallery/", 8)):
            members = sorted((info.filename for info in archive.infolist()
                              if info.filename.startswith(prefix) and info.filename.endswith(".jpg")), key=rank)
            counts, used, checked, errors = Counter(), set(), 0, []
            role_rows = []
            for member in members[:4096]:
                if all(counts[b] == quota for b in BINS):
                    break
                if time.monotonic() - start > 120:
                    raise TimeoutError("Finite gallery selection budget exceeded; preserve receipt")
                name = PurePosixPath(member).name
                prefix_id = name.split("_", 1)[0]
                if not prefix_id.isdecimal():
                    raise ValueError("Use only labeled canonical native images")
                person_id = int(prefix_id)
                if person_id in used:
                    continue
                checked += 1
                raw = archive.read(member)
                try:
                    with Image.open(BytesIO(raw)) as image:
                        width, height = image.size
                        if image.format != "JPEG" or not 1 <= min(width, height) <= max(width, height) <= 4096:
                            raise ValueError("Unexpected native JPEG dimensions/format")
                        image.verify()
                    group = bucket(width, height)
                except (OSError, ValueError) as error:
                    errors.append({"member": member, "error": str(error)})
                    continue
                if counts[group] == quota:
                    continue
                case_id = f"{'dev' if role == 'development' else 'reserved'}_{group}_{counts[group]+1:02d}"
                directory = OUT / role
                directory.mkdir(exist_ok=True)
                relative = f"{role}/{case_id}.jpg"
                with (OUT / relative).open("xb") as target:
                    target.write(raw)
                role_rows.append({"id": case_id, "role": role, "size_bin": group,
                                  "global_person_id": person_id, "archive_member": member,
                                  "source_sha256": hashlib.sha256(raw).hexdigest(), "source_file": relative,
                                  "native_width": width, "native_height": height,
                                  "camera_token": name.split("_")[1], "source_country": None,
                                  "exposure": "input-only development review pending" if role == "development" else "reserved; JPEG header/bytes only"})
                used.add(person_id)
                counts[group] += 1
            scan[role] = {"headers_checked": checked, "bins": dict(counts), "jpeg_errors": errors,
                          "quota_complete": all(counts[b] == quota for b in BINS)}
            rows.extend(sorted(role_rows, key=lambda row: (BINS.index(row["size_bin"]), row["id"])))
            if not scan[role]["quota_complete"]:
                write(OUT / "selection_shortfall.json", {"complete": False, "scan": scan, "selected": rows})
                raise ValueError("Size-bin quotas unavailable within frozen scan cap; do not silently replace them")
    dev_ids = {row["global_person_id"] for row in rows if row["role"] == "development"}
    reserved_ids = {row["global_person_id"] for row in rows if row["role"] == "reserved_evaluation"}
    if dev_ids & reserved_ids or len(dev_ids) != 24 or len(reserved_ids) != 32:
        raise ValueError("Selected identity separation/count differs")
    if len({row["source_sha256"] for row in rows}) != len(rows):
        raise ValueError("Exact image bytes duplicate across selected cases")
    for group in BINS:
        group_rows = [row for row in rows if row["role"] == "development" and row["size_bin"] == group]
        sheet = Image.new("RGB", (3 * 260, 2 * 290), (240, 240, 240))
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(group_rows):
            with Image.open(OUT / row["source_file"]) as image:
                image = image.convert("RGB")
                width, height = image.size
                canvas = Image.new("RGB", (max(width, height),) * 2, (128, 128, 128))
                canvas.paste(image, ((max(width, height)-width)//2, (max(width, height)-height)//2))
                display = canvas.resize((256, 256), Image.Resampling.NEAREST)
            x, y = (i % 3) * 260, (i // 3) * 290
            draw.text((x+2, y+2), row["id"], fill=(0, 0, 0))
            draw.text((x+2, y+15), f"native {width}x{height}; NN display", fill=(0, 0, 0))
            sheet.paste(display, (x, y+31))
        sheet.save(OUT / f"development_inputs_{group}.png")
    protocol = {"format": "dgp-native-cctv-subset-v1", "date": "2026-10-03", "complete": True,
                "selection_policy_sha256": sha(OUT / "selection_policy.json"), "cases": rows,
                "scan": scan, "selected_identity_overlap": 0, "selected_exact_byte_duplicates": 0,
                "source_archive_sha256": ARCHIVE_PIN, "source_native_preserved": True,
                "development_inputs_rendered": 24, "reserved_inputs_rendered": 0,
                "models_executed": False, "optimizer_updates": 0, "selection_seconds": time.monotonic() - start,
                "contact_sheet_sha256": {f"development_inputs_{group}.png": sha(OUT / f"development_inputs_{group}.png") for group in BINS}}
    write(OUT / "frozen_subset.json", protocol)
    print(json.dumps({"complete": True, "development": 24, "reserved": 32, "scan": scan,
                      "subset_sha256": sha(OUT / "frozen_subset.json"), "seconds": protocol["selection_seconds"]}))


if __name__ == "__main__":
    main()
