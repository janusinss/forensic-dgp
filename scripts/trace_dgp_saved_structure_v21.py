"""Bounded saved-array diagnostic; no model import, output correction or training."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, convolve1d


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def image(path):
    with Image.open(path) as source:
        return np.asarray(source).copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha", required=True)
    args = parser.parse_args()
    out = args.root.resolve()
    assert out.is_relative_to(ROOT) and not (out / "results.json").exists()
    assert sha(out / "plan.json") == args.plan_sha
    plan = read(out / "plan.json")
    assert plan["format"] == "dgp-saved-output-structure-trace-v21"
    assert len(plan["cases"]) == plan["budgets"]["cases"] == 74
    started = time.monotonic()

    def clock():
        assert time.monotonic() - started < plan["budgets"]["worker_seconds"]

    for rel, expected in plan["sources_sha256"].items():
        clock()
        path = (ROOT / rel).resolve()
        assert path.is_relative_to(ROOT) and sha(path) == expected, rel

    spec = plan["diagnostic_kernel"]
    assert spec["sigma_pixels_256"] == 2.0 and spec["radius_pixels_256"] == 6
    positions = np.arange(-6, 7, dtype=np.float64)
    kernel = np.exp(-0.5 * (positions / 2.0) ** 2)
    kernel /= kernel.sum()
    luma = np.array(spec["luma_weights"], dtype=np.float64)

    def luminance(rgb):
        return np.sum(rgb.astype(np.float64) * luma, axis=2)

    def high_pass(value):
        low = convolve1d(convolve1d(value, kernel, axis=0, mode="reflect"),
                         kernel, axis=1, mode="reflect")
        return value - low

    def compare(camera, prediction, delivered, mask, target):
        ih = high_pass(camera)[mask]
        oh = high_pass(prediction)[mask]
        residual = oh - ih
        denominator = float(np.dot(ih, ih))
        energy = float(np.dot(oh, oh))
        assert len(ih) and denominator > 1e-20
        slope = float(np.dot(ih, oh) / denominator)
        residue_energy = float(np.dot(residual, residual))
        rho = (float(np.dot(ih, residual) / np.sqrt(denominator * residue_energy))
               if residue_energy > 1e-20 else None)
        dx_in, dy_in = np.diff(camera, axis=1), np.diff(camera, axis=0)
        dx_out, dy_out = np.diff(prediction, axis=1), np.diff(prediction, axis=0)
        mx = mask[:, 1:] & mask[:, :-1]
        my = mask[1:, :] & mask[:-1, :]
        grad_in = float(np.sum(dx_in[mx] ** 2) + np.sum(dy_in[my] ** 2))
        grad_out = float(np.sum(dx_out[mx] ** 2) + np.sum(dy_out[my] ** 2))
        assert grad_in > 1e-20
        stats = {
            "pixels": int(mask.sum()),
            "input_high_frequency_rms": float(np.sqrt(denominator / len(ih))),
            "output_high_frequency_rms": float(np.sqrt(energy / len(ih))),
            "input_aligned_detail_slope": slope,
            "effective_residual_detail_slope": slope - 1.0,
            "effective_residual_input_correlation": rho,
            "high_frequency_energy_ratio": energy / denominator,
            "first_difference_energy_ratio": grad_out / grad_in,
            "raw_input_luma_MAE": float(np.abs(prediction[mask] - camera[mask]).mean()),
            "PNG_raw_luma_MAE": float(np.abs(delivered[mask] - prediction[mask]).mean()),
            "PNG_raw_high_frequency_RMSE": float(np.sqrt(np.mean(
                (high_pass(delivered)[mask] - oh) ** 2))),
        }
        if target is not None:
            th = high_pass(target)[mask]
            stats.update({
                "input_target_high_frequency_RMSE": float(np.sqrt(np.mean((ih - th) ** 2))),
                "output_target_high_frequency_RMSE": float(np.sqrt(np.mean((oh - th) ** 2))),
            })
        return stats

    records = []
    compositions = 0
    for case in plan["cases"]:
        clock()
        original = image(ROOT / case["input"])
        assert original.shape == (256, 256, 3) and original.dtype == np.uint8
        captured = image(ROOT / case["observed"]) > 0
        assert captured.shape == (256, 256)
        interior = binary_erosion(captured, structure=np.ones((13, 13), bool))
        camera = original.astype(np.float32) / np.float32(255)
        camera_y = luminance(camera)
        target = (image(ROOT / case["target"]).astype(np.float64) / 255
                  if case["target"] is not None else None)
        regions = {"observed_interior": interior}
        patches = np.zeros((256, 256), bool)
        points = case.get("landmarks5_canvas_xy") or case["eyes_canvas_xy"]
        for point in points:
            x, y = np.floor(point).astype(int)
            patches[max(0, y - 12):min(256, y + 12),
                    max(0, x - 12):min(256, x + 12)] = True
        regions["input_landmark_patches"] = patches & interior
        raw_paths = case.get("raw_arms", {"dgp": case.get("raw_dgp")})
        png_paths = case.get("png_arms", {"dgp": case.get("png_dgp")})
        for arm in case["arms"]:
            raw = np.load(ROOT / raw_paths[arm], allow_pickle=False)
            assert raw.dtype == np.float32 and raw.shape == original.shape
            assert np.isfinite(raw).all() and 0 <= raw.min() <= raw.max() <= 1
            delivered = image(ROOT / png_paths[arm])
            expected = np.where(captured[..., None], np.floor(raw * np.float32(255)),
                                original).astype(np.uint8)
            assert np.array_equal(delivered, expected), case["id"] + ":" + arm
            compositions += 1
            pred_y = luminance(raw)
            stats = {name: compare(camera_y, pred_y,
                                   luminance(delivered.astype(np.float64) / 255),
                                   mask, None if target is None else luminance(target))
                     for name, mask in regions.items()}
            record = {
                "id": case["id"], "source": case["source"], "role": case["role"],
                "profile": case["profile"], "source_person_or_reference": case["source_person_or_reference"],
                "arm": arm, "regions": stats,
                "exact_original_raw_PNG_delivery": True,
                "max_observed_quantization_error": float(np.abs(
                    raw.astype(np.float64)[captured] - delivered.astype(np.float64)[captured] / 255).max()),
                "normalization_scope": case["normalization_scope"],
                "native_unpaired": target is None,
            }
            if target is not None:
                record["paired_photographic_input_MSE"] = float(((camera.astype(np.float64) - target)[captured] ** 2).mean())
                record["paired_photographic_raw_DGP_MSE"] = float(((raw.astype(np.float64) - target)[captured] ** 2).mean())
            records.append(record)

    grouped = collections.defaultdict(list)
    for row in records:
        for region, values in row["regions"].items():
            key = (row["source"], row["profile"], row["arm"], region)
            grouped[key].append(values)
    summaries = []
    for key, values in sorted(grouped.items()):
        available = [field for field in values[0] if field != "pixels"]
        summaries.append({
            "source": key[0], "profile": key[1], "arm": key[2], "region": key[3],
            "cases": len(values),
            "effective_residual_cancellation_cases": sum(v["effective_residual_detail_slope"] < 0 for v in values),
            "mean": {field: float(np.mean([v[field] for v in values if v[field] is not None]))
                     for field in available},
            "median": {field: float(np.median([v[field] for v in values if v[field] is not None]))
                       for field in available},
        })
    clock()
    result = {
        "complete": True, "date": "2026-10-05", "plan_sha256": args.plan_sha,
        "runner_sha256": sha(Path(__file__)), "seconds": time.monotonic() - started,
        "source_bindings_verified": len(plan["sources_sha256"]),
        "case_count": len(plan["cases"]), "arm_rows": len(records),
        "exact_raw_PNG_compositions": compositions, "records": records, "summaries": summaries,
        "neural_calls": 0, "training_calls": 0, "app_or_model_changed": False,
        "correction_outputs_produced": False, "reserved_pixels_used": False,
        "native_quality_or_identity_metric": False,
        "independent_final_review_complete": False, "goal_complete": False,
        "limits": "Input-aligned detail may include noise and compression. Negative effective residual projection proves attenuation in these saved outputs, not a particular latent training cause, facial identity preservation or loss recovery. Native eyes are input metadata regions; other features are not automatically localized.",
    }
    with (out / "results.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in
                     ("complete", "seconds", "case_count", "arm_rows", "exact_raw_PNG_compositions")}))
    for summary in summaries:
        if summary["source"].startswith("ChokePoint"):
            print(json.dumps({key: summary[key] for key in
                             ("arm", "region", "effective_residual_cancellation_cases", "median")}))


if __name__ == "__main__":
    main()
