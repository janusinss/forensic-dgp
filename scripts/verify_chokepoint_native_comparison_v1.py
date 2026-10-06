"""Audit saved native comparison tensors, policy decisions and display cells; no models."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/cctv_chokepoint_native_comparison_v1"
BASE = ROOT / "outputs/cctv_chokepoint_native_development_v1"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(value, message):
    if not value:
        raise ValueError(message)


def pixels(path, mode="RGB"):
    with Image.open(path) as image:
        return np.asarray(image.convert(mode)).copy()


def main():
    receipt = OUT / "saved_output_audit.json"
    require(not receipt.exists(), "Preserve original audit")
    started = time.monotonic()
    plan, results, execution = [read(OUT / name) for name in ("plan.json", "results.json", "execution.json")]
    require(results["complete"] and results["plan_sha256"] == execution["plan_sha256"] == sha(OUT / "plan.json"), "Plan/execution binding")
    for path, expected in plan["sources_sha256"].items():
        require(sha(ROOT / path) == expected, "Source binding: " + path)
    for path, expected in results["artifacts_sha256"].items():
        require(sha(OUT / path) == expected, "Artifact binding: " + path)
    require(results["model_forwards"] == {"phase3": 24, "identity_v2": 24, "codeformer": 24}, "Reported finite counts")
    require(results["model_state_after"] == execution["model_state_before"], "Reported state parity")
    require(results["seconds"] <= plan["budget"]["wall_seconds_including_loading"] == 300, "Worker timing")
    require(results["PSNR"] is None and results["SSIM"] is None and results["native_evidence_unpaired"], "Unpaired scope")
    require(results["optimizer_updates"] == results["backward_calls"] == 0 and
            not results["reserved_source13_and_original_qmul32_used"] and not results["app_changed"], "Scope")
    require(len(results["records"]) == len(plan["cases"]) == 24, "Complete development cohort")
    choices, checks = Counter(), []
    for case, row in zip(plan["cases"], results["records"]):
        require(case["id"] == row["id"] and case["role"] == "development", "Development row ordering")
        common = pixels(BASE / case["input"])
        observed = pixels(BASE / case["observed"], "L") != 0
        require(np.array_equal(common, pixels(OUT / row["outputs"]["resize"])), "Resize control")
        for arm in ("phase3", "identity_v2", "codeformer_w1"):
            raw = np.load(OUT / f"raw/{case['id']}_{arm}.npy", allow_pickle=False)
            require(raw.shape == (256, 256, 3) and raw.dtype == np.float32 and np.isfinite(raw).all()
                    and raw.min() >= 0 and raw.max() <= 1, "Raw shape/range")
            expected = np.floor(raw * np.float32(255)).astype(np.uint8)
            expected[~observed] = common[~observed]
            image = pixels(OUT / row["outputs"][arm])
            require(np.array_equal(expected, image), "Exact raw-to-PNG/padding composition")
            change = float(np.mean(np.abs(image.astype(np.float64) - common.astype(np.float64))[observed]))
            require(abs(change - row["observed_MAE_change_255_diagnostic_only"][arm]) <= 1e-12, "Diagnostic input change")
        # Recompute the fixed blur/noise policy from captured support, not the
        # execution's saved decision and never the generated images.
        support = cv2.erode(observed.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        support[:4] = support[-4:] = False
        support[:, :4] = support[:, -4:] = False
        require(support.sum() >= 512, "Nonvacuous quality support")
        gray = cv2.cvtColor(common, cv2.COLOR_RGB2GRAY).astype(np.float32)
        smoothed = cv2.GaussianBlur(gray, (3, 3), .6)
        blur = float(cv2.Laplacian(smoothed, cv2.CV_32F)[support].var())
        low = cv2.GaussianBlur(gray, (7, 7), 1.5)
        gradient = np.hypot(cv2.Sobel(low, cv2.CV_32F, 1, 0), cv2.Sobel(low, cv2.CV_32F, 0, 1))
        flat = support & (gradient <= np.percentile(gradient[support], 35))
        response = cv2.filter2D(gray, cv2.CV_32F, np.asarray([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32))
        noise = float(np.median(np.abs(response[flat])) / (6 * .67448975))
        signals = row["quality_signals"]
        require(abs(signals["blur_variance"] - blur) <= 1e-12 and
                abs(signals["noise_sigma_255"] - noise) <= 1e-12, "Independent input signals")
        chosen = "identity_v2" if blur < 24 or noise >= 8 else "resize"
        require(chosen == row["auto_selected"] and signals["blur_threshold"] == 24 and signals["noise_threshold"] == 8, "Frozen routing")
        require(np.array_equal(pixels(OUT / row["outputs"]["identity_v2_auto"]),
                               pixels(OUT / row["outputs"][chosen])), "Exact Auto alias")
        choices[chosen] += 1
        checks.append({"id": case["id"], "auto_selected": chosen, "raw_png_compositions": 3,
                       "padding_exact": True, "input_change_is_quality_metric": False})
    sheets = []
    for first in range(0, 24, 4):
        name = f"comparison-{first // 4 + 1:02d}.png"
        with Image.open(OUT / name) as sheet:
            require(sheet.size == (1340, 1192), "Full-resolution sheet size")
            for index, row in enumerate(results["records"][first:first + 4]):
                y = 48 + 292 * index
                for column, arm in enumerate(plan["arms"]):
                    x = 268 * column + 5
                    require(np.array_equal(np.asarray(sheet.crop((x, y, x + 256, y + 256))),
                                           pixels(OUT / row["outputs"][arm])), "Exact saved sheet cell")
        sheets.append({"path": name, "sha256": sha(OUT / name),
                       "cases": [row["id"] for row in results["records"][first:first + 4]]})
    seconds = time.monotonic() - started
    require(seconds <= 120, "Audit cap")
    value = {"complete": True, "date": "2026-10-05", "seconds": seconds,
             "results_sha256": sha(OUT / "results.json"), "auditor_sha256": sha(Path(__file__)),
             "source_bindings": len(plan["sources_sha256"]), "artifact_bindings": len(results["artifacts_sha256"]),
             "raw_png_compositions": 72, "resize_controls": 24, "exact_auto_aliases": 24,
             "independently_recomputed_input_signals": 24, "exact_sheet_cells": 120,
             "auto_choices": dict(choices), "sheets": sheets, "checks": checks,
             "model_forwards": 0, "training_calls": 0, "checkpoint_state_replayed": False,
             "state_scope": "Reported execution parity with frozen source/checkpoint bindings; no neural replay",
             "PSNR": None, "SSIM": None, "identity_accuracy": None,
             "reserved_source13_and_original_qmul32_used": False,
             "restoration_qualified": False, "independent_final_review": False, "app_changed": False}
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"complete": True, "seconds": seconds, "raw_compositions": 72,
                      "auto_choices": dict(choices)}), flush=True)


if __name__ == "__main__":
    main()
