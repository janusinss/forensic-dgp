"""Audit saved normalization diagnostic arrays without importing a neural model."""
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/dgp_eval_statistics_diagnostic_v1"
MIXED = ROOT / "outputs/cctv_dgp_mixed_vm_v9_r2"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pixels(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def main():
    receipt = OUT / "saved_output_audit.json"
    require(not receipt.exists(), "Preserve the original audit; no overwrite")
    started = time.monotonic()
    plan, results = read(OUT / "plan.json"), read(OUT / "results.json")
    require(results["complete"] and results["plan_sha256"] == sha(OUT / "plan.json"), "Plan binding")
    for path, expected in plan["sources_sha256"].items():
        require(sha(ROOT / path) == expected, "Source binding: " + path)
    for path, expected in results["artifacts_sha256"].items():
        require(sha(OUT / path) == expected, "Artifact binding: " + path)
    require(plan["arms"] == ["stored_frozen", "per_image_no_writeback"], "Arms")
    require(plan["maximum_dgp_forwards"] == 4 and plan["normalization_layers"] == 5, "Finite counts")
    require(results["forwards"] == {"dgp": 4} and results["seconds"] <= plan["cap_seconds"] == 120, "Reported bounds")
    require(results["state_before"] == results["state_after"] ==
            "808ad791f272eaeaa48fd28b37c47c198ff095e160dd63e5593b6b654798e4f6", "Reported state parity")
    require(results["optimizer_updates"] == results["backward_calls"] == 0, "Reported no training")
    require(not results["native_used"] and not results["reserved_used"] and not results["app_changed"], "Scope")
    cases = {case["id"]: case for case in plan["cases"]}
    require(len(cases) == 2 and all(case["role"] == "train" and case["profile"] == "clear"
                                  for case in cases.values()), "Training clear controls")
    rows = {}
    for row in results["records"]:
        key = (row["id"], row["arm"])
        require(key not in rows and key[0] in cases and key[1] in plan["arms"], "Output row")
        case = cases[key[0]]
        raw = np.load(OUT / row["raw"], allow_pickle=False)
        require(raw.dtype == np.float32 and raw.shape == (256, 256, 3) and np.isfinite(raw).all(), "Raw shape")
        require(raw.min() >= 0 and raw.max() <= 1, "Raw range")
        image, target = pixels(MIXED / case["input"]), pixels(MIXED / case["target"])
        observed = pixels(MIXED / case["observed"])[..., 0] != 0
        expected = np.floor(raw * np.float32(255)).astype(np.uint8)
        expected[~observed] = image[~observed]
        saved = pixels(OUT / row["output"])
        require(np.array_equal(expected, saved), "Raw-to-PNG composition: " + row["output"])
        require(np.array_equal(saved[~observed], image[~observed]), "Padding preservation")
        mse = float(np.mean(((saved.astype(np.float64) - target.astype(np.float64)) / 255.)[observed] ** 2))
        interior = cv2.erode(observed.astype(np.uint8), np.ones((7, 7), np.uint8)) != 0
        interior[:3] = interior[-3:] = False
        interior[:, :3] = interior[:, -3:] = False
        require(interior.any(), "Nonvacuous SSIM support")
        ssim = float(np.mean([structural_similarity(saved[..., c], target[..., c], data_range=255,
                             full=True)[1][interior].mean() for c in range(3)]))
        metrics = {"MSE_observed": mse, "PSNR_observed": float(-10 * np.log10(max(mse, 1e-12))),
                   "SSIM_valid_observed_windows": ssim}
        require(all(abs(metrics[name] - row[name]) <= 1e-12 for name in metrics), "Independent paired metrics")
        rows[key] = metrics
    require(len(rows) == 4, "Four unique outputs")
    failures = []
    for case_id in cases:
        base, candidate = rows[(case_id, "stored_frozen")], rows[(case_id, "per_image_no_writeback")]
        if candidate["MSE_observed"] > base["MSE_observed"] + 1e-12:
            failures.append({"id": case_id, "gate": "clear_MSE_preservation"})
        if candidate["SSIM_valid_observed_windows"] < base["SSIM_valid_observed_windows"] - 1e-6:
            failures.append({"id": case_id, "gate": "clear_SSIM_preservation"})
    require(failures == results["prospective_stop_failures"] and len(failures) == 2, "Prospective stop reproduction")
    activations = read(OUT / "activation_statistics.json")
    require(len(activations) == 20, "Activation count")
    keys, stored = set(), {}
    for activation in activations:
        key = (activation["id"], activation["arm"], activation["layer"])
        require(key not in keys and key[:2] in rows, "Activation key")
        keys.add(key)
        values = [np.asarray(activation[name], dtype=np.float64) for name in
                  ("input_channel_mean", "input_channel_variance", "stored_mean", "stored_variance")]
        require(all(value.ndim == 1 and value.size == values[0].size and np.isfinite(value).all()
                    for value in values), "Finite channel statistics")
        require((values[1] >= 0).all() and (values[3] > 0).all(), "Variance range")
        layer = activation["layer"]
        if layer in stored:
            require(np.array_equal(stored[layer][0], values[2]) and np.array_equal(stored[layer][1], values[3]),
                    "Recorded stored statistics changed")
        else:
            stored[layer] = values[2:]
    require(len(stored) == 5, "Five recorded normalization layers")
    with Image.open(OUT / "training-controls.png") as sheet:
        require(sheet.size == (1072, 608), "Sheet size")
        for index, case in enumerate(cases.values()):
            y = 48 + 292 * index
            paths = (MIXED / case["input"], MIXED / case["target"],
                     OUT / (case["id"] + "_stored_frozen.png"),
                     OUT / (case["id"] + "_per_image_no_writeback.png"))
            for column, path in enumerate(paths):
                x = 268 * column + 5
                require(np.array_equal(np.array(sheet.crop((x, y, x + 256, y + 256))), pixels(path)), "Sheet cell")
    seconds = time.monotonic() - started
    require(seconds <= 120, "Audit time cap")
    value = {"complete": True, "seconds": seconds, "cap_seconds": 120,
             "results_sha256": sha(OUT / "results.json"), "auditor_sha256": sha(Path(__file__)),
             "source_bindings": len(plan["sources_sha256"]), "artifact_bindings": len(results["artifacts_sha256"]),
             "raw_png_compositions": 4, "independently_recomputed_metric_rows": 4,
             "prospective_stop_failures": failures, "exact_sheet_cells": 8,
             "recorded_activation_rows": 20, "recorded_stored_statistics_consistent": True,
             "activation_values_replayed": False, "checkpoint_state_replayed": False,
             "state_check": "Reported execution parity and frozen source/checkpoint bindings; no neural replay",
             "model_forwards": 0, "training_calls": 0, "app_adoption": False,
             "native_used": False, "reserved_used": False, "independent_final_review": False,
             "decision": "Close this variant at the prospective clear-preservation stop; no broader/native inference"}
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"complete": True, "seconds": seconds, "metric_rows": 4, "stop_failures": len(failures)}))


if __name__ == "__main__":
    main()
