"""Independent saved-array/PIL audit: no Torch, model forward or training."""
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/dgp_app_covering_review_v3"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixels(path):
    with Image.open(path) as source:
        return np.asarray(source.convert("RGB")).copy()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def overlap(a, b):
    union = int((a | b).sum())
    return {"iou_against_operator_footprint": int((a & b).sum()) / union if union else 1.,
            "missed_operator_pixels": int((b & ~a).sum()), "extra_pixels": int((a & ~b).sum()),
            "proposal_pixels": int(a.sum()), "operator_pixels": int(b.sum())}


def visible_metrics(output, target, removal):
    visible = ~removal
    difference = output.astype(np.float64) / 255 - target.astype(np.float64) / 255
    mse = float(np.square(difference[visible]).mean())
    ssim_values = []
    interior = cv2.erode(visible.astype(np.uint8), np.ones((7, 7), np.uint8)) != 0
    interior[:3] = interior[-3:] = False
    interior[:, :3] = interior[:, -3:] = False
    for channel in range(3):
        _, field = structural_similarity(output[..., channel], target[..., channel], data_range=255, full=True)
        ssim_values.append(float(field[interior].mean()))
    return {"MSE_visible": mse, "PSNR_visible": float(-10 * np.log10(max(mse, 1e-12))),
            "SSIM_visible_valid_windows": float(np.mean(ssim_values)),
            "visible_pixels": int(visible.sum()), "ssim_centers": int(interior.sum()),
            "scope": "Paired synthetic degradation on known visible pixels only; no hidden-region accuracy"}


def main():
    started = time.monotonic()
    result = read(OUT / "results.json")
    plan = read(OUT / "plan.json")
    assert result["complete"] and result["plan_sha256"] == sha(OUT / "plan.json")
    assert result["state_before"] == result["state_after"]
    for path, expected in plan["sources_sha256"].items():
        assert sha(ROOT / path) == expected, path
    for path, expected in result["artifacts_sha256"].items():
        assert sha(OUT / path) == expected, path
    comparisons, metrics, sums = [], [], defaultdict(list)
    verified_outputs, exact_aliases, protected = 0, 0, []
    cases = {case["id"]: case for case in plan["cases"]}
    expected_completion = expected_dgp = 0
    for row in result["rows"]:
        if time.monotonic() - started > 180:
            raise TimeoutError("180-second saved-output audit cap exceeded")
        case = cases[row["id"]]
        original = pixels(ROOT / case["input"])
        removal = pixels(ROOT / case["reviewed"]).mean(axis=-1) >= 127.5
        assert original.shape == (256, 256, 3) and removal.shape == (256, 256)
        automatic = pixels(OUT / "images" / (row["id"] + "_automatic.png"))[..., 0] != 0
        counts = overlap(automatic, removal)
        for key, value in counts.items():
            assert value == row["automatic"][key], (row["id"], key)
        comparisons.append({"id": row["id"], "family": row["family"], "condition": row["condition"],
                            "model": "app_retained_detector", **counts})
        for name, path in case["cached_comparison_proposals"].items():
            computed = overlap(pixels(ROOT / path)[..., 0] != 0, removal)
            for key, value in computed.items():
                assert value == row["cached_unqualified_comparisons"][name][key]
            comparisons.append({"id": row["id"], "family": row["family"], "condition": row["condition"],
                                "model": name, **computed})
        outputs = {}
        for mode, record in row["assisted"].items():
            if record["rejected"]:
                continue
            metadata = read(OUT / "metadata" / (row["id"] + "_" + mode + ".json"))
            output = pixels(OUT / record["output"])
            assert sha(OUT / record["output"]) == result["artifacts_sha256"][record["output"]]
            expected = original.copy()
            with np.load(OUT / "stages" / (row["id"] + "_" + mode + ".npz"), allow_pickle=False) as stages:
                if metadata["restoration_applied"]:
                    raw = stages["dgp"][0].transpose(1, 2, 0)
                    assert raw.dtype == np.float32 and raw.shape == (256, 256, 3)
                    assert np.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 1
                    assert hashlib.sha256(raw.tobytes()).hexdigest() == metadata["raw_dgp"]["sha256"]
                    expected[~removal] = np.floor(raw[~removal] * np.float32(255)).astype(np.uint8)
                    expected_dgp += 1
                if removal.any():
                    completed = stages["completion"][0].transpose(1, 2, 0)
                    expected[removal] = np.floor(completed[removal] * np.float32(255)).astype(np.uint8)
                    expected_completion += 1
            colour = metadata["display_processing"]["colour_policy"]
            if colour["applied"]:
                gray = np.repeat(cv2.cvtColor(expected, cv2.COLOR_RGB2GRAY)[..., None], 3, axis=-1)
                if metadata["restoration_applied"]:
                    expected = gray
                else:
                    expected[removal] = gray[removal]
            np.testing.assert_array_equal(output, expected)
            assert hashlib.sha256(output.tobytes()).hexdigest() == metadata["output_rgb_sha256"]
            if mode == "off":
                np.testing.assert_array_equal(output[~removal], original[~removal])
            outputs[mode] = output
            verified_outputs += 1
            if case["protected"]:
                preserve = pixels(ROOT / case["protected"])[..., 0] != 0
                assert not (preserve & removal).any()
                protected.append({"id": row["id"], "mode": mode,
                    "protected_changed_pixels": int(np.any(output != original, axis=-1)[preserve].sum()),
                    "Off_required_exact": mode == "off"})
            if case["synthetically_degraded"]:
                reference = cases[row["id"].replace("_degraded", "_native")]
                target = pixels(ROOT / reference["input"])
                measure = visible_metrics(output, target, removal)
                metrics.append({"id": row["id"], "family": row["family"], "mode": mode, **measure})
                sums[(row["family"], mode)].append(measure)
        if "off" in outputs and "on" in outputs:
            np.testing.assert_array_equal(outputs["off"][removal], outputs["on"][removal])
        if "auto" in outputs:
            selected = "on" if row["assisted"]["auto"]["restoration_applied"] else "off"
            np.testing.assert_array_equal(outputs["auto"], outputs[selected])
            exact_aliases += 1
    assert expected_completion == result["forwards"]["completion"]
    assert expected_dgp == result["forwards"]["dgp"]
    groups = []
    for (family, mode), rows in sums.items():
        mse = float(np.mean([r["MSE_visible"] for r in rows]))
        groups.append({"family": family, "mode": mode, "cases": len(rows), "mean_MSE_visible": mse,
                       "PSNR_from_mean_MSE": float(-10 * np.log10(max(mse, 1e-12))),
                       "mean_SSIM_visible_valid_windows": float(np.mean([r["SSIM_visible_valid_windows"] for r in rows]))})
    receipt = {"complete": True, "seconds": time.monotonic() - started, "cap_seconds": 180,
               "results_sha256": sha(OUT / "results.json"), "source_bindings": len(plan["sources_sha256"]),
               "output_bindings": len(result["artifacts_sha256"]), "exact_compositions": verified_outputs,
               "exact_auto_aliases": exact_aliases, "mask_comparisons": comparisons,
               "paired_synthetic_visible_metrics": metrics, "paired_visible_groups": groups,
               "protected_regions": protected, "model_forwards": 0, "training_calls": 0,
               "native_cctv_metrics": None, "hidden_metrics": None, "family_qualification": False,
               "independent_final_review": False}
    with (OUT / "saved_output_audit.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"complete": True, "seconds": receipt["seconds"],
                      "exact_compositions": verified_outputs, "exact_auto_aliases": exact_aliases}))


if __name__ == "__main__":
    main()
