"""Verify saved face-restoration compositions/visible metrics without models."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA = "c384f1b25e9ab80f44ba6ebef3e47c072a4dc57d406d525f9a637f3abf99b229"
RESULT_SHA = "de6a7bf7e177d7f2f8d3ed2c0536de05c64e6662e464046dda597a50bd9f7cb0"


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
    folder = ROOT / "outputs/practical_face_restoration_outputs_v1"
    destination = folder / "independent_verification.json"
    require(not destination.exists(), "Preserve prior audit")
    protocol = ROOT / "outputs/practical_face_restoration_v1/frozen_protocol.json"
    require(sha(protocol) == PROTOCOL_SHA and sha(folder / "results.json") == RESULT_SHA, "Frozen evidence changed")
    p = json.loads(protocol.read_text(encoding="utf-8"))
    result = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    for collection, prefix in ((p["artifact_sha256"], ROOT), (p["asset_sha256"], ROOT), (result["raw_restoration_sha256"], folder)):
        for name, digest in collection.items():
            require(sha(prefix/name) == digest, f"Frozen file changed: {name}")
    cases = {case["id"]: case for case in p["cases"]}
    expected = {(key, arm) for key in cases for arm in p["arms"]}
    require(len(result["rows"]) == len(expected) == 100, "Missing/duplicate rows")
    require({(r["id"], r["arm"]) for r in result["rows"]} == expected, "Output set differs")
    values, wins, hole_changes = {"native": {}, "degraded": {}}, 0, 0
    rows = {(row["id"],row["arm"]): row for row in result["rows"]}
    for case_id, case in cases.items():
        reference = rgb(ROOT/case["reference"])
        original = rgb(ROOT/case["input"])
        baseline = rgb(ROOT/case["cached_completion"])
        mask_pixels = rgb(ROOT/case["removal"], "L")
        require(np.isin(mask_pixels, (0,255)).all(), "Nonbinary proposal")
        selected = mask_pixels == 255
        require(np.array_equal(baseline[~selected], original[~selected]), "Cached completion drifted")
        restored = {weight: rgb(folder/"raw_restoration"/f"{case_id}_w{weight}.png") for weight in p["fidelity_weights"]}
        group = "degraded" if case["synthetically_degraded"] else "native"
        for arm, spec in p["arms"].items():
            row = rows[(case_id,arm)]
            output = rgb(folder/row["output"])
            require(sha(folder/row["output"]) == row["output_sha256"], "Output changed")
            require(output.shape == baseline.shape == reference.shape == (256,256,3), "Dimensions differ")
            expected_output = baseline.copy()
            if spec is not None:
                blend = spec["visible_blend"]
                expected_output = ((1-blend)*baseline.astype(np.float32)+blend*restored[spec["fidelity"]].astype(np.float32)).round().astype(np.uint8)
                expected_output[selected] = baseline[selected]
            require(np.array_equal(output,expected_output), "Declared composition differs")
            changed = int(np.any(output != baseline,axis=2)[selected].sum())
            require(changed == row["hole_changed_pixels"] == 0, "Completed pixels changed")
            hole_changes += changed
            difference = (output.astype(np.float64)-reference.astype(np.float64))[~selected]/255
            mae, mse = float(np.abs(difference).mean()), float(np.square(difference).mean())
            require(abs(mae-row["visible_mae"]) < 1e-12 and abs(mse-row["visible_mse"]) < 1e-12, "Visible metrics differ")
            psnr = float(-10*np.log10(mse)) if mse else None
            require((psnr is None and row["visible_psnr"] is None) or (psnr is not None and abs(psnr-row["visible_psnr"]) < 1e-10), "PSNR differs")
            require(row["mask_pixels"] == int(selected.sum()) and row["visible_pixels"] == int((~selected).sum()), "Support differs")
            require(row["hidden_face_mae"] is None, "No hidden target exists")
            values[group].setdefault(arm,[]).append(mae)
        if case["synthetically_degraded"]:
            wins += int(rows[(case_id,'w1_half')]['visible_mae'] < rows[(case_id,'off')]['visible_mae'])
    aggregates = {}
    for group, arms in values.items():
        aggregates[group] = {}
        for arm, measurements in arms.items():
            observed = result["aggregates"][group][arm]
            mean = float(np.mean(measurements))
            require(len(measurements) == observed["cases"] == 10 and abs(mean-observed["mean_visible_mae"]) < 1e-12, "Aggregate differs")
            aggregates[group][arm] = {"cases":len(measurements),"mean_visible_mae":mean}
    require(result["complete"] and not result["failures"] and not result["promoted"], "Execution status differs")
    require(result["new_restoration_forwards"] == p["planned_restoration_forwards"] == 40 and result["reused_completed_crops"] == 20, "Recorded budget differs")
    require(result["new_completion_forwards"] == result["new_detector_forwards"] == result["optimizer_updates"] == 0 and not result["optimizer_constructed"], "Unexpected execution recorded")
    require(result["state_before"] == result["state_after"] and result["state_unchanged"], "Recorded state differs")
    require(len(result["raw_restoration_sha256"]) == 40, "Raw output count differs")
    for name,digest in result["preview_sha256"].items():
        require(sha(folder/name) == digest, "Preview changed")
    old, new = aggregates['degraded']['off']['mean_visible_mae'], aggregates['degraded']['w1_half']['mean_visible_mae']
    report = {"verified":True,"date":"2026-10-02","protocol_sha256":PROTOCOL_SHA,"results_sha256":RESULT_SHA,
              "auditor_sha256":sha(__file__),"saved_outputs_verified":100,"raw_outputs_verified":40,"reused_completed_crops_verified":20,
              "completed_region_changed_pixels":hole_changes,"w1_half_degraded_cases_improved":wins,
              "w1_half_aggregate_visible_mae_reduction_percent":100*(old-new)/old,"aggregates":aggregates,
              "new_model_forwards":0,"optimizer_updates":0,"hidden_face_ground_truth":False,
              "scope":"Saved pixels/frozen files; recorded call/state budget checked without replay; no population or hidden-identity claim"}
    destination.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"verified":True,"outputs":100,"completed_region_changed_pixels":hole_changes,
                      "degraded_cases_improved":wins,"visible_mae_reduction_percent":report['w1_half_aggregate_visible_mae_reduction_percent'],"sha256":sha(destination)}))


if __name__ == "__main__":
    main()
