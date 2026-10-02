"""Independently recount saved practical outputs; no model imports or execution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA = "74b80e0ec9ebabb7c8b3b3989f5bc4f44b9a4cf1e39e691c073543f2db4651a5"
RESULT_SHA = "38776d1a46967c4338a2082fe94ad50c2ca335d767b57c8c4759f681374f5325"
SOURCE_SHA = "860ea1dfba8fa442cab19bf3ab41f6a678109dcdb067c38ec0bd79ce43b51ace"
FROZEN_SHA = "2880c85f067c013f6998faf1744177be79b8ed0a50d949d6e35ae5da14563928"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_image(path, mode):
    with Image.open(path) as image:
        value = np.asarray(image.convert(mode)).copy()
    expected = (256, 256, 3) if mode == "RGB" else (256, 256)
    if value.shape != expected:
        raise ValueError(f"Unexpected image shape: {path}")
    if mode == "L":
        if not np.isin(value, (0, 255)).all():
            raise ValueError(f"Nonbinary mask: {path}")
        return value == 255
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/practical_native_outputs_v1")
    args = parser.parse_args()
    run = args.run.resolve()
    require(run.is_relative_to(ROOT), "Run must be inside the workspace")
    destination = run / "independent_verification.json"
    require(not destination.exists(), "Preserve previous verification")
    protocol_path = ROOT / "outputs/practical_gallery_v2/frozen_native_protocol_v1.json"
    require(digest(protocol_path) == PROTOCOL_SHA, "Protocol changed")
    require(digest(run / "results.json") == RESULT_SHA, "Original result changed")
    p = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = json.loads((run / "results.json").read_text(encoding="utf-8"))
    execution = json.loads((run / "execution.json").read_text(encoding="utf-8"))
    gallery = ROOT / p["gallery_root"]
    for name, expected in p["assets_sha256"].items():
        require(digest(gallery / name) == expected, f"Frozen gallery asset changed: {name}")
    for model in p["models"].values():
        require(digest(ROOT / model["path"]) == model["sha256"], "Checkpoint changed")
    for name, expected in execution["code_sha256"].items():
        require(digest(ROOT / name) == expected, f"Executed source changed: {name}")
    manifest = ROOT / "dataset/detector_supported_review_v1/manifest.json"
    require(digest(manifest) == SOURCE_SHA, "Source manifest changed")
    records = json.loads(manifest.read_text(encoding="utf-8"))["supported_records"]
    train = [i for i, record in enumerate(records) if record["split"] == "train"]
    frozen = ROOT / "outputs/frozen_complementarity_v1"
    require(digest(frozen / "results.json") == FROZEN_SHA, "Cached-detector evidence changed")
    previous = json.loads((frozen / "results.json").read_text(encoding="utf-8"))
    observed = {(row["id"], row["arm"]): row for row in result["rows"]}
    expected_keys = {(case["id"], arm) for case in p["cases"] for arm in p["arms"]}
    require(len(observed) == len(result["rows"]) == 30 and set(observed) == expected_keys,
            "Missing, duplicate or unexpected case/arm")
    recount, nonempty, bypasses, changed_outside = [], 0, 0, 0
    for case in p["cases"]:
        source = records[case["source_manifest_index"]]
        require(source["split"] == "train" and not case["synthetically_degraded"], "Cohort scope changed")
        require(digest(ROOT / source["source"]) == case["source_sha256"], "Native source changed")
        require(digest(manifest.parent / source["image"]) == case["accepted_crop_sha256"], "Accepted crop changed")
        rgb = read_image(gallery / case["input"], "RGB")
        require(np.array_equal(rgb, read_image(manifest.parent / source["image"], "RGB")), "Input pixels differ from accepted native crop")
        target = read_image(gallery / case["label_mask"], "L")
        require(np.array_equal(target, read_image(manifest.parent / source["mask"], "L")), "Old label changed")
        allowed = read_image(gallery / case["removal_proposal"], "L")
        for arm in p["arms"]:
            row = observed[case["id"], arm]
            selected = allowed
            if arm != "reviewed":
                selected = read_image(gallery / case["cached_original_masks"][arm], "L")
                cached_key = f"{arm}/real/{train.index(case['source_manifest_index']):04d}.png"
                require(digest(frozen / cached_key) == previous["mask_sha256"][cached_key], "Prior cached mask changed")
                require(np.array_equal(selected, read_image(frozen / cached_key, "L")), "Reused detector mask differs")
            require(digest(run / row["output"]) == row["output_sha256"], "Saved output hash differs")
            output = read_image(run / row["output"], "RGB")
            outside = int(np.any(output != rgb, axis=2)[~selected].sum())
            changed_outside += outside
            difference = np.abs(output.astype(np.float32) - rgb.astype(np.float32)) / 255
            change = float(difference[~allowed].mean())
            coverage = float((selected & target).sum() / target.sum()) if target.any() else None
            require(outside == 0 and row["exact_outside_own_mask"], "Visible pixels outside active mask changed")
            require(int(selected.sum()) == row["mask_pixels"], "Mask area differs")
            require(not bool(selected.any()) == row["empty_mask_bypass"], "Bypass status differs")
            require(bool(target.any() and not selected.any()) == row["covered_input_with_empty_predicted_mask"], "Covered bypass count differs")
            require(abs(change - row["visible_change_outside_reviewed_area"]) < 1e-8, "Visible-change recount differs")
            require(coverage == row["label_coverage_fraction"], "Label-placement recount differs")
            require(int((selected & ~target).sum()) == row["pixels_outside_original_label"], "Placement-margin count differs")
            require(row["hidden_face_mae"] is None and not row["inference_failed"], "Unsupported hidden-face metric or failure")
            nonempty += int(selected.any())
            bypasses += int(not selected.any())
            recount.append({"id": case["id"], "arm": arm, "outside_own_mask_changed_pixels": outside,
                            "visible_change_outside_reviewed_area": change, "label_coverage_fraction": coverage})
    require(result["complete"] and not result["failures"] and not result["promoted"], "Result status differs")
    require(nonempty == result["nonempty_generator_forwards"] == 22 and bypasses == 8, "Forward/bypass recount differs")
    require(execution["protocol_sha256"] == result["protocol_sha256"] == PROTOCOL_SHA, "Execution protocol differs")
    require(result["generator_state_before"] == result["generator_state_after"] and result["state_unchanged"], "Recorded inference state changed")
    require(result["optimizer_updates"] == execution["optimizer_updates"] == 0 and not execution["optimizer_constructed"], "Recorded training present")
    require(digest(run / "preview.png") == result["preview_sha256"], "Preview changed")
    report = {
        "verified": True, "date": "2026-10-02", "protocol_sha256": PROTOCOL_SHA,
        "results_sha256": RESULT_SHA, "auditor_sha256": digest(__file__),
        "audited_rows": len(recount), "nonempty_outputs": nonempty, "empty_bypasses": bypasses,
        "outside_active_mask_changed_pixels": changed_outside,
        "gallery_assets_verified": len(p["assets_sha256"]), "source_records_verified": len(p["cases"]),
        "model_imports": 0, "new_model_forwards": 0, "optimizer_updates": 0,
        "runtime_state_evidence": "Before/after state digests and zero-update metadata verified; no training replay performed",
        "quality_claim": "Pixel/provenance verification only; visual quality requires separate review",
        "hidden_face_ground_truth": False, "promoted": False, "rows": recount,
    }
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verified": True, "rows": len(recount), "outside_changed_pixels": changed_outside,
                      "verification_sha256": digest(destination)}))


if __name__ == "__main__":
    main()
