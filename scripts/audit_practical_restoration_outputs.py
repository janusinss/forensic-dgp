"""Independent saved-pixel audit of the frozen restoration-stage comparison."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA = "74c652e1d1c604c5229fc953117a45b3c994e17c01fb23158ee9b4bccc17bf00"
RESULT_SHA = "2f1b2ff00237302bb3fd42c9188f63541292f777ba0d2df8aa10cc1c21f6223d"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rgb(path, mode="RGB"):
    with Image.open(path) as image:
        return np.asarray(image.convert(mode)).copy()


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    folder = ROOT / "outputs/practical_restoration_outputs_v1"
    destination = folder / "independent_verification.json"
    require(not destination.exists(), "Preserve prior audit")
    protocol = ROOT / "outputs/practical_restoration_v1/frozen_protocol.json"
    require(sha(protocol) == PROTOCOL_SHA and sha(folder / "results.json") == RESULT_SHA, "Frozen evidence changed")
    p = json.loads(protocol.read_text(encoding="utf-8"))
    result = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    for collection in (p["artifact_sha256"], p["asset_sha256"], result["intermediate_sha256"]):
        prefix = folder if collection is result["intermediate_sha256"] else ROOT
        for name, expected in collection.items():
            require(sha(prefix / name) == expected, f"Frozen file changed: {name}")
    cases = {case["id"]: case for case in p["cases"]}
    keys = {(key, arm) for key in cases for arm in p["arms"]}
    require(len(cases) == 20 and len(result["rows"]) == len(keys) == 80, "Missing or duplicate rows")
    require({(row["id"], row["arm"]) for row in result["rows"]} == keys, "Output keys differ")
    rows = {(row["id"], row["arm"]): row for row in result["rows"]}
    for row in rows.values():
        require(sha(folder / row["output"]) == row["output_sha256"], "Saved output changed")
    recount = {"native": {}, "degraded": {}}
    post_hole_changes, off_outside_changes, reused, degraded_cases = 0, 0, 0, 0
    for case_id, case in cases.items():
        original, reference = rgb(ROOT / case["input"]), rgb(ROOT / case["reference"])
        mask = rgb(ROOT / case["removal"], "L")
        require(np.isin(mask, (0, 255)).all(), "Nonbinary removal footprint")
        selected = mask == 255
        output = {arm: rgb(folder / rows[(case_id, arm)]["output"]) for arm in p["arms"]}
        require(all(array.shape == original.shape == reference.shape == (256, 256, 3) for array in output.values()), "Dimensions differ")
        baseline = output["completion_off"]
        outside = int(np.any(baseline != original, axis=2)[~selected].sum())
        require(outside == 0, "Completion-only visible pixels changed")
        off_outside_changes += outside
        if case["cached_native_completion"]:
            require(np.array_equal(baseline, rgb(ROOT / case["cached_native_completion"])), "Reused native output pixels differ")
            reused += 1
        degraded_cases += int(case["synthetically_degraded"])
        pre = rgb(folder / "intermediates" / f"{case_id}_pre_dgp.png")
        post = rgb(folder / "intermediates" / f"{case_id}_post_dgp.png")
        require(np.array_equal(output["legacy_pre"][~selected], pre[~selected]), "Legacy restored base differs")
        expected_post = post.copy()
        expected_post[selected] = baseline[selected]
        require(np.array_equal(output["visible_post"], expected_post), "Visible-only composition differs")
        expected_quarter = (.75 * baseline.astype(np.float32) + .25 * post.astype(np.float32)).round().astype(np.uint8)
        expected_quarter[selected] = baseline[selected]
        require(np.array_equal(output["visible_post25"], expected_quarter), "Fixed blend differs")
        group = "degraded" if case["synthetically_degraded"] else "native"
        for arm, array in output.items():
            row = rows[(case_id, arm)]
            difference = (array.astype(np.float64) - reference.astype(np.float64))[~selected] / 255
            mae, mse = float(np.abs(difference).mean()), float(np.square(difference).mean())
            require(abs(mae - row["visible_mae"]) < 1e-12 and abs(mse - row["visible_mse"]) < 1e-12, "Known-visible metric differs")
            psnr = float(-10 * np.log10(mse)) if mse else None
            require((psnr is None and row["visible_psnr"] is None) or (psnr is not None and abs(psnr-row["visible_psnr"]) < 1e-10), "PSNR differs")
            require(row["psnr_infinite"] == (mse == 0), "Zero-MSE status differs")
            visible_changed = int(np.any(array != original, axis=2)[~selected].sum())
            hole_changed = int(np.any(array != baseline, axis=2)[selected].sum())
            require(row["outside_mask_changed_pixels_from_input"] == visible_changed and row["hole_changed_pixels_from_completion_off"] == hole_changed, "Pixel-change recount differs")
            if arm.startswith("visible_post"):
                require(hole_changed == 0, "Post restoration changed completed face pixels")
                post_hole_changes += hole_changed
            require(row["mask_pixels"] == int(selected.sum()) and row["visible_pixels"] == int((~selected).sum()), "Support recount differs")
            require(row["hidden_face_mae"] is None, "No hidden ground truth exists")
            recount[group].setdefault(arm, []).append(mae)
    aggregates = {}
    for group, arms in recount.items():
        aggregates[group] = {}
        for arm, values in arms.items():
            observed = result["aggregates"][group][arm]
            mean = float(np.mean(values))
            require(observed["cases"] == len(values) == 10 and abs(observed["mean_visible_mae"]-mean) < 1e-12, "Aggregate differs")
            aggregates[group][arm] = {"cases": len(values), "mean_visible_mae": mean}
    require(reused == degraded_cases == 10 and len(result["intermediate_sha256"]) == 40, "Artifact count differs")
    require(result["calls"] == p["calls_planned"], "Recorded call budget differs")
    require(result["states_before"] == result["states_after"] and result["state_unchanged"], "Recorded model state differs")
    require(result["complete"] and not result["failures"] and not result["promoted"], "Execution status differs")
    require(not result["optimizer_constructed"] and result["optimizer_updates"] == 0, "Training recorded")
    for name, digest in result["preview_sha256"].items():
        require(sha(folder / name) == digest, "Preview changed")
    report = {"verified": True, "date": "2026-10-02", "protocol_sha256": PROTOCOL_SHA, "results_sha256": RESULT_SHA,
              "auditor_sha256": sha(__file__), "saved_outputs_verified": 80, "intermediates_verified": 40,
              "reused_native_outputs_verified": reused, "completion_off_outside_mask_changed_pixels": off_outside_changes,
              "post_restoration_hole_changed_pixels": post_hole_changes, "known_visible_metrics_recounted": True,
              "aggregates": aggregates, "new_model_forwards": 0, "optimizer_updates": 0, "hidden_face_ground_truth": False,
              "scope": "Saved pixels and frozen artifacts; model-call/state counts checked as execution records, not independently replayed"}
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verified": True, "outputs": 80, "sha256": sha(destination), "aggregates": aggregates}))


if __name__ == "__main__":
    main()
