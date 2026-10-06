"""Analytic TRAINING-only capacity bound; no model, fitting or image output.

The target-informed interval projection is deliberately more permissive than a
learned scalar spatial gate. It is not an inference recipe or quality verdict.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pixels(path):
    with Image.open(path) as source:
        return np.asarray(source).copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--plan-sha", required=True)
    args = parser.parse_args()
    out = args.root.resolve()
    assert out.is_relative_to(ROOT) and not (out / "results.json").exists()
    assert sha(out / "plan.json") == args.plan_sha
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    assert plan["format"] == "dgp-detail-gate-analytic-capacity-bound-v21"
    started = time.monotonic()

    def clock():
        assert time.monotonic() - started < plan["budgets"]["worker_seconds"]

    for rel, expected in plan["sources_sha256"].items():
        clock()
        path = (ROOT / rel).resolve()
        assert path.is_relative_to(ROOT) and sha(path) == expected, rel
    assert len(plan["cases"]) == 50 and all(c["role"] == "train" for c in plan["cases"])
    points = np.arange(-6, 7, dtype=np.float64)
    kernel = np.exp(-0.5 * (points / 2.0) ** 2)
    kernel /= kernel.sum()

    def blur(value):
        return convolve1d(convolve1d(value, kernel, axis=0, mode="reflect"),
                          kernel, axis=1, mode="reflect")

    records = []
    for case in plan["cases"]:
        clock()
        original = pixels(ROOT / case["input"])
        target_png = pixels(ROOT / case["target"])
        mask = pixels(ROOT / case["observed"]) > 0
        raw = np.load(ROOT / case["raw_dgp"], allow_pickle=False)
        png = pixels(ROOT / case["png_dgp"])
        assert original.shape == target_png.shape == raw.shape == png.shape == (256, 256, 3)
        assert raw.dtype == np.float32 and np.isfinite(raw).all()
        assert 0 <= raw.min() <= raw.max() <= 1
        assert np.array_equal(png, np.where(mask[..., None],
            np.floor(raw * np.float32(255)), original).astype(np.uint8))
        camera = (original.astype(np.float32) / np.float32(255)).astype(np.float64)
        retained = raw.astype(np.float64)
        target = target_png.astype(np.float64) / 255.0
        residual = retained - camera
        denominator = blur(mask.astype(np.float64))
        low = blur(residual * mask[..., None]) / np.maximum(denominator[..., None], 1e-15)
        band = residual - low
        lo = np.maximum(0.0, retained - np.abs(band))
        hi = np.minimum(1.0, retained + np.abs(band))
        projection = np.minimum(np.maximum(target, lo), hi)
        # An extra integer on each side deliberately enlarges the delivered set
        # to cover float32 cast/multiply rounding. This is a conservative bound,
        # not necessarily a realizable delivered image. Padding is never scored.
        lo_q = np.maximum(0, np.floor(lo * 255).astype(np.int16) - 1)
        hi_q = np.minimum(255, np.floor(hi * 255).astype(np.int16) + 1)
        q_projection = np.minimum(np.maximum(target_png.astype(np.int16), lo_q), hi_q)
        baseline_raw_mse = float(((retained - target)[mask] ** 2).mean())
        bound_raw_mse = float(((projection - target)[mask] ** 2).mean())
        baseline_png_mse = float((((png.astype(np.float64) - target_png) / 255)[mask] ** 2).mean())
        bound_png_mse = float((((q_projection.astype(np.float64) - target_png) / 255)[mask] ** 2).mean())
        assert bound_raw_mse <= baseline_raw_mse + 1e-15
        assert bound_png_mse <= baseline_png_mse + 1e-15
        records.append({"id": case["id"], "source": case["source"], "role": "train",
                        "profile": case["profile"],
                        "retained_raw_MSE": baseline_raw_mse,
                        "optimistic_raw_MSE_lower_bound": bound_raw_mse,
                        "retained_PNG_MSE": baseline_png_mse,
                        "conservative_PNG_MSE_lower_bound": bound_png_mse})

    grouped = collections.defaultdict(list)
    for row in records:
        src, profile = row["source"], row["profile"]
        for key in ("all", "clear" if profile == "clear" else "degraded", src + "/all",
                    src + ("/clear" if profile == "clear" else "/degraded"), src + "/" + profile):
            grouped[key].append(row)
    fields = [key for key in records[0] if key.endswith("MSE") or key.endswith("bound")]
    groups = []
    for key, rows in sorted(grouped.items()):
        avg = {field: float(np.mean([r[field] for r in rows])) for field in fields}
        avg["raw_MSE_improvement_ceiling"] = 1 - avg["optimistic_raw_MSE_lower_bound"] / avg["retained_raw_MSE"]
        avg["conservative_PNG_MSE_improvement_ceiling"] = 1 - avg["conservative_PNG_MSE_lower_bound"] / avg["retained_PNG_MSE"]
        groups.append({"group": key, "cases": len(rows), "metrics": avg})
    degraded = next(g for g in groups if g["group"] == "degraded")
    ceiling = degraded["metrics"]["conservative_PNG_MSE_improvement_ceiling"]
    minimum = plan["prospective_stop"]["minimum_degraded_PNG_MSE_relative_gain"]
    clock()
    result = {"complete": True, "date": "2026-10-05", "plan_sha256": args.plan_sha,
              "runner_sha256": sha(Path(__file__)), "seconds": time.monotonic() - started,
              "source_bindings_verified": len(plan["sources_sha256"]), "cases": 50,
              "records": records, "groups": groups,
              "conservative_degraded_PNG_MSE_improvement_ceiling": ceiling,
              "necessary_capacity_condition_pass": ceiling >= minimum,
              "close_detail_gate_before_VM_pilot": ceiling < minimum,
              "target_informed_analytic_bound": True, "restoration_outputs_produced": 0,
              "neural_calls": 0, "training_calls": 0, "native_or_reserved_used": False,
              "app_or_checkpoint_changed": False, "historical_gates_waived": False,
              "independent_final_review_complete": False, "goal_complete": False,
              "limits": "Target-informed pointwise interval bound is not restoration or a fitted model. It permits separate RGB coefficients and ignores learnability/spatial smoothness, making it more permissive than a scalar gate. The PNG set is additionally expanded one integer at each end; no SSIM, identity or structural-quality bound is implied. Failure only closes this restricted band-correction route under the declared capacity condition, not all possible DGP architectures."}
    with (out / "results.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("complete", "seconds", "cases",
        "conservative_degraded_PNG_MSE_improvement_ceiling", "necessary_capacity_condition_pass",
        "close_detail_gate_before_VM_pilot")}))


if __name__ == "__main__":
    main()
