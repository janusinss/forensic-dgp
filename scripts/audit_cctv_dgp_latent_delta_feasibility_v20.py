"""Independently reconstruct saved V20 oracle arithmetic; no neural imports."""
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/cctv_dgp_latent_delta_feasibility_v20"
MIXED = ROOT / "outputs/cctv_dgp_mixed_vm_v9_r2"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(value, message):
    if not value:
        raise ValueError(message)


def rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def main():
    receipt = OUT / "independent_saved_output_audit.json"
    require(not receipt.exists(), "Preserve existing audit")
    started = time.monotonic()
    plan, result, execution = [read(OUT / name) for name in ("plan.json", "results.json", "execution.json")]
    require(result["complete"] and result["plan_sha256"] == execution["plan_sha256"] == sha(OUT / "plan.json"), "Plan binding")
    require(plan["frozen_before_inference"] and len(plan["cases"]) == 10, "Frozen ten-case scope")
    for path, expected in plan["sources_sha256"].items():
        require(sha(ROOT / path) == expected, "Frozen source differs: " + path)
    for path, expected in result["artifacts_sha256"].items():
        require(sha(OUT / path) == expected, "Saved artifact differs: " + path)
    expected_counts = {"dgp": 10, "teacher_encoder": 12, "teacher_quantizer": 12, "teacher_generator": 12}
    require(result["model_forwards"] == expected_counts and result["seconds"] <= plan["budget"]["wall_seconds"] == 240, "Reported finite counts/time")
    require(execution["model_state_before"] == result["model_state_after"] and result["model_states_unchanged"], "Reported state parity")
    require(not execution["optimizer_constructed"] and result["optimizer_updates"] == result["backward_calls"] == 0, "Reported no training")
    require(all(not result[key] for key in ("native_used", "reserved_used", "validation_used", "app_changes", "trained_contribution_verified", "goal_complete")), "Scope limits")
    require(result["target_informed_oracle_not_restoration"] and result["paired_synthetic_training_only"], "Oracle declaration")
    require(plan["prospective_feasibility_guards"] == {"clear_raw_and_png_exact_DGP": True,
        "each_degraded_case_MSE_not_worse_than_DGP_tolerance": 1e-12,
        "each_degraded_case_SSIM_not_worse_than_DGP_tolerance": 1e-6,
        "stop_before_conditioner_or_VM_pilot_if_any_failure": True}, "Unchanged prospective guards")
    cases = {case["id"]: case for case in plan["cases"]}
    require(len(cases) == 10 and all(case["role"] == "train" for case in cases.values()), "Training cases")
    records, failures, compositions, clear_controls = {}, [], 0, 0
    for row in result["records"]:
        require(row["id"] in cases and row["id"] not in records, "Unique known row")
        case = cases[row["id"]]
        require(all(row[key] == case[key] for key in ("reference_id", "profile", "source")), "Case labels")
        camera, target = rgb(MIXED / case["input"]), rgb(MIXED / case["target"])
        observed = rgb(MIXED / case["observed"])[..., 0] != 0
        require(camera.shape == target.shape == (256, 256, 3) and observed.any(), "Input support")
        arrays = {}
        for arm, path in row["raw"].items():
            array = np.load(OUT / path, allow_pickle=False)
            require(array.dtype == np.float32 and array.shape == (256, 256, 3) and
                    np.isfinite(array).all() and array.min() >= 0 and array.max() <= 1, "Raw schema/range")
            arrays[arm] = array
        require(set(arrays) == {"dgp", "input_vq", "target_vq_oracle", "prior_delta_oracle"}, "Four raw arms")
        target_vq = np.load(OUT / "raw" / (case["reference_id"] + "_target_vq.npy"), allow_pickle=False)
        require(np.array_equal(target_vq, arrays["target_vq_oracle"]), "Exact cached target rendering")
        expected_oracle = np.clip(arrays["dgp"] + (target_vq - arrays["input_vq"]), 0, 1)
        require(np.array_equal(expected_oracle, arrays["prior_delta_oracle"]), "Exact prior-difference oracle arithmetic")
        scores = {}
        for arm in plan["arms"]:
            if arm == "resize":
                expected = camera
            else:
                expected = np.floor(arrays[arm] * np.float32(255)).astype(np.uint8)
                expected[~observed] = camera[~observed]
                compositions += 1
            delivered = rgb(OUT / row["outputs"][arm])
            require(np.array_equal(expected, delivered), "Exact raw-to-PNG: " + arm)
            require(np.array_equal(delivered[~observed], camera[~observed]), "Padding preserved")
            error = ((delivered.astype(np.float64) - target.astype(np.float64)) / 255.)[observed]
            mse = float(np.mean(error * error))
            interior = cv2.erode(observed.astype(np.uint8), np.ones((7, 7), np.uint8)) != 0
            interior[:3] = interior[-3:] = False
            interior[:, :3] = interior[:, -3:] = False
            require(interior.any(), "Nonempty SSIM support")
            ssim = float(np.mean([structural_similarity(delivered[..., channel], target[..., channel], data_range=255,
                         full=True)[1][interior].mean() for channel in range(3)]))
            scores[arm] = {"MSE_observed": mse, "PSNR_observed": float(-10 * np.log10(max(mse, 1e-12))),
                           "SSIM_valid_observed_windows": ssim}
            require(all(abs(value - row["metrics"][arm][key]) <= 1e-12 for key, value in scores[arm].items()), "Independent paired metrics")
        if row["profile"] == "clear":
            equal = np.array_equal(arrays["dgp"], arrays["prior_delta_oracle"])
            require(equal == row["clear_raw_exact_DGP"], "Clear parity reporting")
            if equal and np.array_equal(rgb(OUT / row["outputs"]["dgp"]), rgb(OUT / row["outputs"]["prior_delta_oracle"])):
                clear_controls += 1
            else:
                failures.append({"id": row["id"], "gate": "clear_exact_DGP"})
        else:
            a, b = scores["dgp"], scores["prior_delta_oracle"]
            if b["MSE_observed"] > a["MSE_observed"] + 1e-12:
                failures.append({"id": row["id"], "gate": "degraded_MSE_preservation"})
            if b["SSIM_valid_observed_windows"] < a["SSIM_valid_observed_windows"] - 1e-6:
                failures.append({"id": row["id"], "gate": "degraded_SSIM_preservation"})
        records[row["id"]] = row
    require(len(records) == 10 and failures == result["prospective_feasibility_failures"], "All rows and prospective failures")
    require(result["stop_before_conditioner_or_VM_pilot"] == bool(failures), "Feasibility stop")
    cells = 0
    for entry in result["sheets"]:
        require(sha(OUT / entry["path"]) == entry["sha256"], "Sheet binding")
        with Image.open(OUT / entry["path"]) as sheet:
            require(sheet.size == (1340, 1504), "Original sheet resolution")
            for row_index, case_id in enumerate(entry["cases"]):
                for column, arm in enumerate(plan["arms"]):
                    x, y = 268 * column + 5, 48 + 292 * row_index
                    require(np.array_equal(np.asarray(sheet.crop((x, y, x + 256, y + 256))),
                                           rgb(OUT / records[case_id]["outputs"][arm])), "Exact sheet cell")
                    cells += 1
    require(cells == 50 and compositions == 40, "Complete saved arithmetic scope")
    seconds = time.monotonic() - started
    require(seconds <= 120, "Two-minute independent audit cap")
    value = {"complete": True, "date": "2026-10-05", "seconds": seconds,
        "results_sha256": sha(OUT / "results.json"), "auditor_sha256": sha(Path(__file__)),
        "source_bindings": len(plan["sources_sha256"]), "artifact_bindings": len(result["artifacts_sha256"]),
        "paired_png_metric_rows": 50, "raw_png_compositions": compositions,
        "exact_prior_difference_compositions": 10, "exact_clear_DGP_controls": clear_controls,
        "exact_sheet_cells": cells, "prospective_failures": failures,
        "stop_before_conditioner_or_VM_pilot": bool(failures),
        "neural_calls": 0, "training_calls": 0, "oracle_is_actual_restoration": False,
        "limits": "Checks saved arrays, independent arithmetic/metrics and reported state/counts; no neural/teacher/CUDA replay or trained contribution."}
    with receipt.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(value), flush=True)


if __name__ == "__main__":
    main()
