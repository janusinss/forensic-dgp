"""Recompute split, temporal selection and24 native crops from audited camera bytes."""
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import tarfile
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/cctv_chokepoint_native_development_v1"
AUDIT = ROOT / "outputs/cctv_chokepoint_release_audit_v1_r1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    receipt = OUT / "independent_geometry_audit.json"
    require(not receipt.exists(), "Preserve original audit")
    started = time.monotonic()
    plan, subset = read(OUT / "selection_plan.json"), read(OUT / "frozen_subset.json")
    require(subset["complete"] and subset["selection_plan_sha256"] == sha(OUT / "selection_plan.json"), "Plan binding")
    for path, expected in plan["sources_sha256"].items():
        require(sha(ROOT / path) == expected, "Source binding: " + path)
    annotations = read(AUDIT / "selected_annotations.json")["P1E_S1_C1"]
    people = sorted({row["person_id"] for row in annotations}, key=lambda pid:
                    hashlib.sha256((plan["seed"] + "\0ChokePoint/P1/" + pid).encode()).hexdigest())
    require(people[:12] == plan["development_person_ids"] and people[12:] == plan["reserved_person_ids"], "Deterministic person split")
    require(not set(people[:12]) & set(people[12:]) and len(people) == 25, "Source identity disjointness")
    dev = [case for case in subset["cases"] if case["role"] == "development"]
    reserved = [case for case in subset["cases"] if case["role"] == "reserved_evaluation"]
    require(len(dev) == 24 and len(reserved) == 26, "Complete case counts")
    for pid in people:
        rows = sorted((row for row in annotations if row["person_id"] == pid and row["valid_eyes"]), key=lambda row: int(row["frame"]))
        selected = [rows[round((len(rows) - 1) * fraction)] for fraction in (0.33, 0.67)]
        cases = [case for case in subset["cases"] if case["source_person_id"] == pid]
        require(len(cases) == 2, "Two temporal cases per identity")
        for row, case in zip(selected, cases):
            require(case["frame"] == row["frame"] and case["eyes"] == row["eyes"], "Temporal annotation selection")
            (lx, ly), (rx, ry) = row["eyes"]
            cx, cy, distance = (lx + rx) / 2., (ly + ry) / 2., math.hypot(lx - rx, ly - ry)
            proposed = [math.floor(cx - 2 * distance), math.floor(cy - 1.5 * distance),
                        math.ceil(cx + 2 * distance), math.ceil(cy + 3.3 * distance)]
            bbox = [max(0, proposed[0]), max(0, proposed[1]), min(800, proposed[2]), min(600, proposed[3])]
            require(case["proposed_bbox"] == proposed and case["native_bbox"] == bbox, "Native ROI formula")
            require(case["role"] == ("development" if pid in people[:12] else "reserved_evaluation"), "Role assignment")
            if case["role"] == "reserved_evaluation":
                require(not any(key in case for key in ("native_crop", "input", "observed")), "Reserved metadata only")
    wanted = {case["source_frame_member"]: case for case in dev}
    verified = read(AUDIT / "verification.json")
    nested = ROOT / verified["nested_archives"]["P1E_S1_C1.tar.xz"]["path"]
    for folder in ("native_crops", "inputs", "observed"):
        require({path.name for path in (OUT / folder).iterdir()} == {case["id"] + ".png" for case in dev}, "No reserved/unplanned image files")
    with tarfile.open(nested, "r|xz") as archive:
        for member in archive:
            require(time.monotonic() - started <= 120, "Finite geometry audit cap")
            if member.name not in wanted:
                continue
            case = wanted.pop(member.name)
            raw = archive.extractfile(member).read()
            require(hashlib.sha256(raw).hexdigest() == case["source_frame_sha256"], "Original frame SHA256")
            with Image.open(BytesIO(raw)) as frame:
                crop = frame.convert("RGB").crop(tuple(case["native_bbox"]))
            width, height = crop.size
            side, x, y = max(width, height), (max(width, height) - width) // 2, (max(width, height) - height) // 2
            canvas = Image.new("RGB", (side, side), (128, 128, 128))
            canvas.paste(crop, (x, y))
            mask = Image.new("L", (side, side), 0)
            mask.paste(255, (x, y, x + width, y + height))
            expected = {"native_crop": crop, "input": canvas.resize((256, 256), Image.Resampling.BILINEAR),
                        "observed": mask.resize((256, 256), Image.Resampling.NEAREST)}
            for key, image in expected.items():
                require(sha(OUT / case[key]) == case[key + "_sha256"], "Derived file binding")
                with Image.open(OUT / case[key]) as actual:
                    require(np.array_equal(np.asarray(image), np.asarray(actual)), "Original RGB crop/prepared geometry")
    require(not wanted, "All24 original frames replayed")
    for sheet in subset["sheets"]:
        require(sha(OUT / sheet["path"]) == sheet["sha256"], "Input sheet binding")
        with Image.open(OUT / sheet["path"]) as actual:
            require(actual.size == (520, 612), "Input sheet dimensions")
            for index, case_id in enumerate(sheet["cases"]):
                case = next(case for case in dev if case["id"] == case_id)
                with Image.open(OUT / case["native_crop"]) as crop:
                    side = max(crop.size)
                    canvas = Image.new("RGB", (side, side), (128, 128, 128))
                    canvas.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
                    image = canvas.resize((256, 256), Image.Resampling.NEAREST)
                x, y = index % 2 * 260 + 2, index // 2 * 306 + 43
                require(np.array_equal(np.asarray(image), np.asarray(actual.crop((x, y, x + 256, y + 256)))), "Exact input sheet cell")
    value = {"complete": True, "date": "2026-10-05", "seconds": time.monotonic() - started,
             "subset_sha256": sha(OUT / "frozen_subset.json"), "auditor_sha256": sha(Path(__file__)),
             "source_bindings": len(plan["sources_sha256"]), "development_identities": 12,
             "reserved_identities": 13, "source_labeled_identity_overlap": 0,
             "temporal_selections_recomputed": 50, "native_crop_compositions_exact": 24,
             "prepared_inputs_and_masks_exact": 24, "exact_input_sheet_cells": 24,
             "reserved_image_pixels_decoded": 0, "reserved_image_files_created": 0,
             "model_forwards": 0, "training_calls": 0, "paired_restoration_reference": False,
             "cross_source_historical_identity_overlap": "Unknown; no independence claim beyond this source's labels",
             "independent_final_review": False}
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"complete": True, "seconds": value["seconds"], "exact_native_crops": 24,
                      "reserved_pixels_viewed": 0}), flush=True)


if __name__ == "__main__":
    main()
