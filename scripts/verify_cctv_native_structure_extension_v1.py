"""Independent bounded replay of native metadata selection; no model inference."""
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import time
from zipfile import ZipFile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/cctv_native_structure_extension_v1"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    receipt = OUT / "independent_selection_audit.json"
    require(not receipt.exists(), "Preserve original audit")
    started = time.monotonic()
    plan, subset, review = [read(OUT / name) for name in
                           ("selection_plan.json", "frozen_subset.json", "input_review.json")]
    require(subset["selection_plan_sha256"] == sha(OUT / "selection_plan.json"), "Plan binding")
    require(review["subset_sha256"] == sha(OUT / "frozen_subset.json"), "Review binding")
    for path, expected in plan["sources_sha256"].items():
        require(sha(ROOT / path) == expected, "Source binding: " + path)
    archive_path = ROOT / "dataset/cctv_survface_raw/QMUL-SurvFace-v1.zip"
    require(sha(archive_path) == plan["archive_sha256"], "Archive binding")
    original = read(ROOT / "outputs/cctv_native_development_v2/frozen_subset.json")
    excluded = {case["global_person_id"] for case in original["cases"]}
    original_bytes = {case["source_sha256"] for case in original["cases"]}
    require(sorted(excluded) == plan["excluded_original_global_person_ids"] and len(excluded) == 56, "Original identity exclusion")
    counts, histogram, selected, used, fingerprints, checked = Counter(), Counter(), [], set(), set(), 0
    with ZipFile(archive_path) as archive:
        names = [info.filename for info in archive.infolist()
                 if info.filename.startswith("QMUL-SurvFace/training_set/") and info.filename.endswith(".jpg")]
        names.sort(key=lambda name: hashlib.sha256((plan["seed"] + "\0" + name).encode()).hexdigest())
        for member in names[:plan["budget"]["ranked_candidate_headers"]]:
            require(time.monotonic() - started < 120, "Finite audit cap")
            if counts["64to95"] == counts["ge96"] == 8:
                break
            pid = int(PurePosixPath(member).name.split("_", 1)[0])
            if pid in excluded or pid in used:
                continue
            checked += 1
            data = archive.read(member)
            with Image.open(BytesIO(data)) as image:
                width, height, detected = image.width, image.height, image.format
                require(detected in ("JPEG", "PNG"), "Native format")
                image.verify()
            short = min(width, height)
            group = "ge96" if short >= 96 else "64to95" if short >= 64 else "below64"
            histogram[group] += 1
            fingerprint = hashlib.sha256(data).hexdigest()
            if group == "below64" or counts[group] >= 8 or fingerprint in fingerprints or fingerprint in original_bytes:
                continue
            selected.append((member, pid, width, height, detected, group, fingerprint))
            counts[group] += 1
            used.add(pid)
            fingerprints.add(fingerprint)
        require(checked == subset["headers_checked"] and dict(histogram) == subset["header_bin_counts"], "Bounded header replay")
        require(len(selected) == len(subset["cases"]) == 3 and not subset["quota_complete"], "Explicit shortfall")
        for expected, case in zip(selected, subset["cases"]):
            require(expected == tuple(case[key] for key in ("archive_member", "global_person_id", "native_width",
                    "native_height", "detected_format", "size_bin", "source_sha256")), "Deterministic selection")
            data = archive.read(case["archive_member"])
            require(data == (OUT / case["source_file"]).read_bytes(), "Original native bytes")
            with Image.open(BytesIO(data)) as image:
                rgb = image.convert("RGB")
            width, height = rgb.size
            side, x, y = max(width, height), (max(width, height) - width) // 2, (max(width, height) - height) // 2
            canvas = Image.new("RGB", (side, side), (128, 128, 128))
            canvas.paste(rgb, (x, y))
            expected_rgb = np.asarray(canvas.resize((256, 256), Image.Resampling.BILINEAR))
            mask = Image.new("L", (side, side), 0)
            mask.paste(255, (x, y, x + width, y + height))
            expected_mask = np.asarray(mask.resize((256, 256), Image.Resampling.NEAREST))
            with Image.open(OUT / case["input"]) as image:
                require(np.array_equal(np.asarray(image), expected_rgb), "Prepared input geometry")
            with Image.open(OUT / case["observed"]) as image:
                require(np.array_equal(np.asarray(image), expected_mask), "Observed mask geometry")
            require(sha(OUT / case["input"]) == case["input_sha256"] and
                    sha(OUT / case["observed"]) == case["observed_sha256"], "Prepared file bindings")
    require(not used & excluded and len(fingerprints) == len(used) == 3, "Identity/byte overlap")
    require({row["id"] for row in review["rows"]} == {case["id"] for case in subset["cases"]}, "Complete input review")
    require(all(row["input_review"] == "out_of_scope" and not row["model_outputs_viewed"]
                for row in review["rows"]) and review["usable_cases"] == 0, "No model comparison admission")
    for sheet in subset["sheets"]:
        require(sha(OUT / sheet["path"]) == sheet["sha256"], "Input sheet binding")
    require(subset["seconds"] <= subset["selection_cap_seconds"] == 120 and not subset["time_stop"], "Selection time")
    require(subset["model_forwards"] == subset["optimizer_updates"] == subset["reserved_inputs_rendered"] == 0, "Scope")
    value = {"complete": True, "date": "2026-10-05", "seconds": time.monotonic() - started,
             "subset_sha256": sha(OUT / "frozen_subset.json"), "input_review_sha256": sha(OUT / "input_review.json"),
             "auditor_sha256": sha(Path(__file__)), "source_bindings": len(plan["sources_sha256"]),
             "ranked_headers_replayed": checked, "selected_native_files_exact": 3,
             "prepared_inputs_and_masks_exact": 3, "shortfall": subset["shortfall"],
             "input_usable": 0, "input_out_of_scope": 3, "original_identity_overlap": 0,
             "reserved_images_decoded_or_rendered": 0, "model_forwards": 0, "training_calls": 0,
             "cross_release_historical_identity_overlap": "Unknown", "independent_final_review": False,
             "decision": "Selection/input shortfall preserved. No neural comparison, source replacement or training admission."}
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"complete": True, "seconds": value["seconds"], "headers_replayed": checked, "usable": 0}))


if __name__ == "__main__":
    main()
