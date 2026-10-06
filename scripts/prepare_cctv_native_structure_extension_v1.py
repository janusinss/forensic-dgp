"""Bounded metadata selection of an additional native CCTV development stratum."""
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
OUT = ROOT / "outputs/cctv_native_structure_extension_v1"
ARCHIVE = ROOT / "dataset/cctv_survface_raw/QMUL-SurvFace-v1.zip"
ORIGINAL = ROOT / "outputs/cctv_native_development_v2/frozen_subset.json"
AUDIT = ROOT / "outputs/cctv_survface_structure_audit_v1/verification.json"
ARCHIVE_SHA = "2fbb0876bc4761217c6de5576905e2524b8ca50ad7905720a5b4b378a0ff8e13"
ORIGINAL_SHA = "c063985b258b808efaa897c88593cbdbcee2848d686d3d056219f8a8e555a98e"
AUDIT_SHA = "cf9124983a815776a787eeae577bbbbd173892be2f69dc33c354626641863222"
SEED = "dgp-native-structure-extension-v1-2026-10-05"
BINS = ("64to95", "ge96")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def rank(member):
    return hashlib.sha256((SEED + "\0" + member).encode("utf-8")).hexdigest()


def main():
    if OUT.exists():
        raise ValueError("Preserve completed/partial selection; no overwrite or automatic repeat")
    started = time.monotonic()
    if sha(ARCHIVE) != ARCHIVE_SHA or sha(ORIGINAL) != ORIGINAL_SHA or sha(AUDIT) != AUDIT_SHA:
        raise ValueError("Native source, original split or release audit differs")
    original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
    proof = json.loads(AUDIT.read_text(encoding="utf-8"))
    if not proof["complete"] or proof["labeled_train_test_person_overlap"] != 0:
        raise ValueError("Require the independently audited release identity split")
    excluded_ids = {case["global_person_id"] for case in original["cases"]}
    excluded_bytes = {case["source_sha256"] for case in original["cases"]}
    if len(excluded_ids) != 56:
        raise ValueError("Preserve all 24 original development and 32 reserved identities")
    OUT.mkdir()
    (OUT / "source_provenance_snapshot.md").write_bytes((ROOT / "CCTV_BENCHMARK_STATUS.md").read_bytes())
    bindings = [Path(__file__), ORIGINAL, AUDIT,
                ROOT / "outputs/cctv_survface_acquisition_v1/acquisition.json",
                ROOT / "outputs/cctv_survface_acquisition_v1/request.json",
                ROOT / "outputs/cctv_native_development_v2/selection_policy.json",
                ROOT / "outputs/cctv_native_development_v2/input_review.json",
                OUT / "source_provenance_snapshot.md"]
    plan = {"date": "2026-10-05", "frozen_before_input_rendering_and_model_outputs": True,
            "purpose": "Additional larger-native-resolution development stratum to diagnose preservation/usefulness where more visible structure may exist; original low-resolution failures remain in scope",
            "source": "Official QMUL-SurvFace V1, previously acquired and independently audited",
            "archive_sha256": ARCHIVE_SHA, "seed": SEED, "release_role": "training_set",
            "source_country_per_image": None, "ethnicity_labels": None, "zamboanga_samples": False,
            "source_terms": "Previously captured public research download; copyright remains with original owners. No new redistribution or permission claim.",
            "source_terms_snapshot": "source_provenance_snapshot.md",
            "bins_min_native_side": {"64to95": [64, 95], "ge96": [96, None]},
            "maximum_cases_per_bin": 8, "target_total": 16,
            "selection": "SHA256(seed + NUL + full native member name), first unused labeled identity in each available size bin. No pose, appearance, output or ethnicity selection.",
            "excluded_original_global_person_ids": sorted(excluded_ids),
            "budget": {"ranked_candidate_headers": 16384, "wall_seconds_including_hashes": 120,
                       "external_timeout_seconds": 180,
                       "model_forwards": 0, "training_updates": 0},
            "shortfall_policy": "Stop at the fixed header/time cap. Preserve selected cases and explicit shortfall; never lower size cutoffs, borrow reserved cases or replace on visual/model quality.",
            "native_bytes": "Keep encoded bytes/names/provenance; JPEG and PNG headers are accepted even with .jpg extension",
            "preprocessing": "RGB; center-pad shorter native axis with RGB128, then Pillow bilinear256; nearest support mask; no alignment, enhancement, extra crop or synthetic degradation",
            "input_only_review_criteria": [
                "One already cropped face, frontal or mildly turned; classify strong profile, excessive tilt, multiple faces or poor framing as out_of_scope before outputs",
                "Both eye regions, nose position, mouth/lower-face relationship and contour must retain usable rough structure; dark/washout/blur with ambiguous structure requests a clearer image",
                "Resolution strata are diagnostic, not automatic acceptance thresholds; larger crops can still be unusable",
                "Record every selected case and exclusion without replacement, before neural inference; no model output informs input usability",
                "Later output review requires useful clarity with preserved visible features/appearance; invented generic structure, severe artifacts or no useful improvement fails; no aligned clean reference or identity-accuracy claim"],
            "release_labeled_identity_overlap_with_original": 0,
            "historical_dgp_pretrained_identity_overlap": "Unknown; release-label separation is not proof of non-overlap with earlier model training",
            "reserved_inputs_rendered": 0, "final_evaluation": False, "new_training": False,
            "original_24_and_reserved_32_unchanged": True,
            "sources_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in bindings}}
    write(OUT / "selection_plan.json", plan)
    counts, formats, histogram = Counter(), Counter(), Counter()
    selected_ids, selected_bytes, cases, errors = set(), set(), [], []
    checked = 0
    time_stop = False
    with ZipFile(ARCHIVE) as archive:
        members = sorted((info.filename for info in archive.infolist()
                          if info.filename.startswith("QMUL-SurvFace/training_set/")
                          and info.filename.endswith(".jpg")), key=rank)
        for member in members[:16384]:
            if all(counts[bin_name] == 8 for bin_name in BINS):
                break
            if time.monotonic() - started >= 120:
                time_stop = True
                break
            name = PurePosixPath(member).name
            person_id = int(name.split("_", 1)[0])
            if PurePosixPath(member).parent.name != str(person_id):
                raise ValueError("Canonical labeled member differs")
            if person_id in excluded_ids or person_id in selected_ids:
                continue
            checked += 1
            raw = archive.read(member)
            try:
                with Image.open(BytesIO(raw)) as image:
                    width, height = image.size
                    detected = image.format
                    if detected not in ("JPEG", "PNG") or not 1 <= min(width, height) <= max(width, height) <= 4096:
                        raise ValueError("Unexpected native dimensions/format")
                    image.verify()
            except (OSError, ValueError) as error:
                errors.append({"member": member, "error": str(error)})
                continue
            formats[detected] += 1
            short_side = min(width, height)
            group = "ge96" if short_side >= 96 else "64to95" if short_side >= 64 else "below64"
            histogram[group] += 1
            if group not in BINS or counts[group] == 8:
                continue
            fingerprint = hashlib.sha256(raw).hexdigest()
            if fingerprint in selected_bytes or fingerprint in excluded_bytes:
                continue
            case_id = f"ext_{group}_{counts[group] + 1:02d}"
            (OUT / "native").mkdir(exist_ok=True)
            source = "native/" + case_id + ".jpg"
            with (OUT / source).open("xb") as stream:
                stream.write(raw)
            cases.append({"id": case_id, "role": "development_extension", "release_role": "training_set",
                          "global_person_id": person_id, "archive_member": member, "source_file": source,
                          "source_sha256": fingerprint, "native_width": width, "native_height": height,
                          "detected_format": detected, "size_bin": group,
                          "camera_token": next((p for p in name[:-4].split("_") if p.startswith("cam") and p[3:].isdecimal()), None),
                          "source_country": None, "exposure": "input-only development review pending"})
            selected_ids.add(person_id)
            selected_bytes.add(fingerprint)
            counts[group] += 1
    if selected_ids & excluded_ids or len(selected_ids) != len(cases):
        raise ValueError("Native identity overlap/count differs")
    for folder in ("inputs", "observed"):
        (OUT / folder).mkdir()
    for case in cases:
        with Image.open(OUT / case["source_file"]) as image:
            rgb = image.convert("RGB")
        width, height = rgb.size
        side = max(width, height)
        x, y = (side - width) // 2, (side - height) // 2
        canvas = Image.new("RGB", (side, side), (128, 128, 128))
        canvas.paste(rgb, (x, y))
        common = canvas.resize((256, 256), Image.Resampling.BILINEAR)
        support = Image.new("L", (side, side), 0)
        support.paste(255, (x, y, x + width, y + height))
        support = support.resize((256, 256), Image.Resampling.NEAREST)
        case["input"] = "inputs/" + case["id"] + ".png"
        case["observed"] = "observed/" + case["id"] + ".png"
        common.save(OUT / case["input"])
        support.save(OUT / case["observed"])
        case["input_sha256"], case["observed_sha256"] = sha(OUT / case["input"]), sha(OUT / case["observed"])
    sheets = []
    for group in BINS:
        group_cases = [case for case in cases if case["size_bin"] == group]
        for first in range(0, len(group_cases), 4):
            rows = group_cases[first:first + 4]
            sheet = Image.new("RGB", (520, 612), "white")
            draw = ImageDraw.Draw(sheet)
            for index, case in enumerate(rows):
                x, y = (index % 2) * 260, (index // 2) * 306
                draw.text((x + 2, y + 2), case["id"], fill="black")
                draw.text((x + 2, y + 16), f"Native {case['native_width']}x{case['native_height']}; NN input only", fill="black")
                with Image.open(OUT / case["source_file"]) as image:
                    rgb = image.convert("RGB")
                side = max(rgb.size)
                canvas = Image.new("RGB", (side, side), (128, 128, 128))
                canvas.paste(rgb, ((side - rgb.width) // 2, (side - rgb.height) // 2))
                sheet.paste(canvas.resize((256, 256), Image.Resampling.NEAREST), (x + 2, y + 43))
            filename = f"input-only-{group}-{first // 4 + 1:02d}.png"
            sheet.save(OUT / filename)
            sheets.append({"path": filename, "sha256": sha(OUT / filename), "cases": [c["id"] for c in rows]})
    seconds = time.monotonic() - started
    write(OUT / "frozen_subset.json", {"complete": True, "quota_complete": all(counts[b] == 8 for b in BINS),
          "date": "2026-10-05", "selection_plan_sha256": sha(OUT / "selection_plan.json"), "cases": cases,
          "headers_checked": checked, "ranked_candidate_cap": 16384, "header_bin_counts": dict(histogram),
          "selected_counts": {b: counts[b] for b in BINS}, "shortfall": {b: 8 - counts[b] for b in BINS},
          "detected_formats": dict(formats), "errors": errors, "time_stop": time_stop, "seconds": seconds,
          "selection_cap_seconds": 120, "sheets": sheets, "input_review_pending": True,
          "original_identity_overlap": 0, "original_byte_overlap": 0, "model_forwards": 0,
          "optimizer_updates": 0, "reserved_inputs_rendered": 0, "independent_final_review": False})
    print(json.dumps({"complete": True, "selected": len(cases), "counts": dict(counts), "headers_checked": checked,
                      "seconds": seconds, "shortfall": {b: 8 - counts[b] for b in BINS}}))


if __name__ == "__main__":
    main()
