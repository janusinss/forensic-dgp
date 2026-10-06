"""Freeze source-specific identities/crops, rendering development inputs only."""
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import tarfile
import time

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs/cctv_chokepoint_release_audit_v1_r1"
OUT = ROOT / "outputs/cctv_chokepoint_native_development_v1"
SEED = "dgp-chokepoint-native-person-split-v1-2026-10-05"
CAMERA = "P1E_S1_C1"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    if OUT.exists():
        raise ValueError("Preserve completed/partial cohort; no repeat or overwrite")
    verified = read(AUDIT / "verification.json")
    if not verified["complete"] or verified["acquired_sequence_persons"] != 25:
        raise ValueError("Require audited native frames and25 source-labeled persons")
    annotations = read(AUDIT / "selected_annotations.json")[CAMERA]
    inventory = {row["frame"]: row for row in read(AUDIT / "frame_inventory.json")
                 if row["camera"] == CAMERA and "frame" in row}
    people = sorted({row["person_id"] for row in annotations},
                    key=lambda pid: hashlib.sha256((SEED + "\0ChokePoint/P1/" + pid).encode()).hexdigest())
    if len(people) != 25:
        raise ValueError("Person quota differs; do not borrow another source")
    development, reserved = people[:12], people[12:]
    started = time.monotonic()
    OUT.mkdir()
    (OUT / "LICENSE_SOURCE.html").write_bytes((AUDIT / "LICENSE_SOURCE.html").read_bytes())
    (OUT / "DERIVATIVE_NOTICE.txt").write_text(
        "THESIS RESEARCH DERIVATIVES: These are our unrotated eye-annotation-based native face crops, "
        "padded/resized inputs and contact sheets, not the original ChokePoint release. "
        "Original camera files remain fingerprinted separately. Noncommercial research only. "
        "Retain LICENSE_SOURCE.html with all copies/derivatives. Acknowledge NICTA and Wong, Chen, "
        "Mau, Sanderson and Lovell, CVPR Workshops2011, DOI10.1109/CVPRW.2011.5981881.\n",
        encoding="utf-8", newline="\n")
    paths = [Path(__file__), AUDIT / "verification.json", AUDIT / "selected_annotations.json",
             AUDIT / "frame_inventory.json", OUT / "LICENSE_SOURCE.html", OUT / "DERIVATIVE_NOTICE.txt",
             ROOT / "outputs/cctv_native_development_v2/frozen_subset.json"]
    nested = ROOT / verified["nested_archives"][CAMERA + ".tar.xz"]["path"]
    if sha(nested) != verified["nested_archives"][CAMERA + ".tar.xz"]["sha256"]:
        raise ValueError("Verified original camera container differs")
    paths.append(nested)
    write(OUT / "selection_plan.json", {"date": "2026-10-05", "frozen_before_input_rendering_and_model_outputs": True,
          "source": "ChokePoint original P1E_S1 camera1 frames and author eye/identity annotations",
          "seed": SEED, "identity_namespace": "ChokePoint/P1/publisher_person_id",
          "development_person_ids": development, "reserved_person_ids": reserved,
          "development_person_count": 12, "reserved_person_count": 13,
          "frames_per_identity": 2, "trajectory_positions": [0.33, 0.67],
          "frame_selection": "For each person, order valid two-eye annotations by native frame number and choose round((N-1)*position). No pixel, face-quality or model-output selection; no replacements.",
          "camera_choice": "Publisher README declares P1E_S1_C1 the most frontal camera; no output-driven camera choice",
          "native_crop": "Axis-aligned integer rectangle from mean eye midpoint and Euclidean eye distance d: floor(cx-2d), floor(cy-1.5d), ceil(cx+2d), ceil(cy+3.3d). Clamp to native800x600 bounds and record truncation. No rotation, landmark warp, denoise or synthetic degradation.",
          "prepared_input": "Preserve native crop RGB pixels; center-pad shorter axis with RGB128 at native crop size, then Pillow bilinear256; nearest observed support",
          "input_review_criteria": ["One frontal or mildly turned already cropped face; unsupported pose/framing rejects before inference",
             "Eyes, nose, mouth/lower-face relationship and contour must have usable rough visible structure; ambiguous dark/washout/blur requests a clearer crop",
             "Native ROI dimensions are diagnostic, not an automatic usability threshold",
             "Retain every selected input/exclusion and both temporal positions without replacement; preserve clear glasses and non-obstructing hair",
             "Later review separates raw/restoration/display output and checks useful clarity, visible structure and appearance; generic anatomy/artifacts/no improvement fail. No clean paired reference or recovered-identity claim"],
          "budget": {"development_native_crops": 24, "reserved_metadata_cases": 26, "wall_seconds": 90,
                     "outer_timeout_seconds": 120, "model_forwards": 0, "training_updates": 0},
          "source_country_per_image": None, "ethnicity_labels": None, "zamboanga_samples": False,
          "cross_source_and_historical_training_identity_overlap": "Unknown; namespace labels and report source-specific results. Published QMUL17-source list excludes ChokePoint, which does not prove identity disjointness.",
          "publisher_protocol_reproduced": False,
          "split_note": "Custom identity-disjoint restoration review split, not the publisher's recognition G1/G2 sequence protocol",
          "native_evidence_unpaired": True, "reserved_pixels_decoded_or_rendered": 0,
          "original_qmul24_development_and32_reserved_unchanged": True,
          "sources_sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}})
    cases = []
    for role, ids in (("development", development), ("reserved_evaluation", reserved)):
        for index, pid in enumerate(ids, 1):
            rows = sorted((row for row in annotations if row["person_id"] == pid and row["valid_eyes"]), key=lambda row: int(row["frame"]))
            if len(rows) < 2:
                raise ValueError("Insufficient annotation trajectory; no replacement")
            chosen = [rows[round((len(rows) - 1) * position)] for position in (0.33, 0.67)]
            if chosen[0]["frame"] == chosen[1]["frame"]:
                raise ValueError("Temporal selection duplicates")
            for marker, row in zip(("t033", "t067"), chosen):
                frame = inventory[row["frame"]]
                left, right = row["eyes"]
                cx, cy = (left[0] + right[0]) / 2., (left[1] + right[1]) / 2.
                distance = math.dist(left, right)
                proposed = [math.floor(cx - 2 * distance), math.floor(cy - 1.5 * distance),
                            math.ceil(cx + 2 * distance), math.ceil(cy + 3.3 * distance)]
                bbox = [max(0, proposed[0]), max(0, proposed[1]), min(800, proposed[2]), min(600, proposed[3])]
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    raise ValueError("Invalid native crop; no replacement")
                case_id = f"choke_{'dev' if role == 'development' else 'reserved'}_{index:02d}_{marker}"
                cases.append({"id": case_id, "role": role, "source_person_id": pid,
                    "identity_namespace": "ChokePoint/P1", "camera": CAMERA, "frame": row["frame"],
                    "source_frame_member": frame["member"], "source_frame_sha256": frame["sha256"],
                    "eyes": row["eyes"], "inter_eye_distance_native": distance,
                    "proposed_bbox": proposed, "native_bbox": bbox, "crop_truncated": bbox != proposed,
                    "native_width": bbox[2] - bbox[0], "native_height": bbox[3] - bbox[1],
                    "exposure": "input-only development review pending" if role == "development" else "reserved metadata only; pixels unviewed"})
    if len(cases) != 50 or set(development) & set(reserved):
        raise ValueError("Role count/identity separation differs")
    wanted = {case["source_frame_member"]: case for case in cases if case["role"] == "development"}
    if len(wanted) != 24:
        raise ValueError("Development source frames duplicate")
    for folder in ("native_crops", "inputs", "observed"):
        (OUT / folder).mkdir()
    with tarfile.open(nested, "r|xz") as archive:
        for member in archive:
            if time.monotonic() - started > 90:
                raise TimeoutError("Finite cohort preparation cap; preserve partial outputs")
            if member.name not in wanted:
                continue
            case = wanted.pop(member.name)
            raw = archive.extractfile(member).read()
            if hashlib.sha256(raw).hexdigest() != case["source_frame_sha256"]:
                raise ValueError("Source frame bytes differ")
            with Image.open(BytesIO(raw)) as image:
                crop = image.convert("RGB").crop(tuple(case["native_bbox"]))
            width, height = crop.size
            side, x, y = max(width, height), (max(width, height) - width) // 2, (max(width, height) - height) // 2
            canvas = Image.new("RGB", (side, side), (128, 128, 128))
            canvas.paste(crop, (x, y))
            mask = Image.new("L", (side, side), 0)
            mask.paste(255, (x, y, x + width, y + height))
            for key, folder, image in (("native_crop", "native_crops", crop),
                    ("input", "inputs", canvas.resize((256, 256), Image.Resampling.BILINEAR)),
                    ("observed", "observed", mask.resize((256, 256), Image.Resampling.NEAREST))):
                case[key] = folder + "/" + case["id"] + ".png"
                image.save(OUT / case[key])
                case[key + "_sha256"] = sha(OUT / case[key])
    if wanted:
        raise ValueError("Selected development frames missing")
    development_cases = [case for case in cases if case["role"] == "development"]
    sheets = []
    for first in range(0, 24, 4):
        rows = development_cases[first:first + 4]
        sheet = Image.new("RGB", (520, 612), "white")
        draw = ImageDraw.Draw(sheet)
        for index, case in enumerate(rows):
            x, y = index % 2 * 260, index // 2 * 306
            draw.text((x + 2, y + 2), case["id"] + " / person" + case["source_person_id"], fill="black")
            draw.text((x + 2, y + 16), f"Native ROI {case['native_width']}x{case['native_height']}; NN input", fill="black")
            with Image.open(OUT / case["native_crop"]) as image:
                side = max(image.size)
                canvas = Image.new("RGB", (side, side), (128, 128, 128))
                canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
                sheet.paste(canvas.resize((256, 256), Image.Resampling.NEAREST), (x + 2, y + 43))
        name = f"input-only-{first // 4 + 1:02d}.png"
        sheet.save(OUT / name)
        sheets.append({"path": name, "sha256": sha(OUT / name), "cases": [case["id"] for case in rows]})
    seconds = time.monotonic() - started
    if seconds > 90:
        raise TimeoutError("Cohort completion cap")
    write(OUT / "frozen_subset.json", {"complete": True, "date": "2026-10-05", "seconds": seconds,
          "selection_plan_sha256": sha(OUT / "selection_plan.json"), "cases": cases, "sheets": sheets,
          "development_identities": 12, "reserved_identities": 13, "selected_identity_overlap": 0,
          "development_crops_rendered": 24, "reserved_crops_rendered": 0,
          "reserved_image_pixels_decoded": 0, "model_forwards": 0, "training_updates": 0,
          "input_review_pending": True, "independent_final_review": False,
          "source_country_inferred": False, "ethnicity_inferred": False})
    print(json.dumps({"complete": True, "seconds": seconds, "development_crops": 24,
                      "reserved_metadata_cases": 26, "reserved_pixels_viewed": 0}), flush=True)


if __name__ == "__main__":
    main()
