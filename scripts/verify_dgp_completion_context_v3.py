"""Saved-array verification of the fixed context ablation; no neural calls."""
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "outputs/dgp_app_covering_review_v3"
OUT = ROOT / "outputs/dgp_app_completion_context_v3"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pixels(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB")).copy()


def measure(image, target, removal):
    keep = ~removal
    residual = (image.astype(np.float64) - target.astype(np.float64)) / 255.
    mse = float(np.mean(residual[keep] ** 2))
    interior = cv2.erode(keep.astype(np.uint8), np.ones((7, 7), np.uint8)) != 0
    interior[:3] = interior[-3:] = False
    interior[:, :3] = interior[:, -3:] = False
    fields = [structural_similarity(image[..., i], target[..., i], data_range=255,
                                   full=True)[1] for i in range(3)]
    return {"MSE_visible": mse, "PSNR_visible": float(-10 * np.log10(max(mse, 1e-12))),
            "SSIM_visible_valid_windows": float(np.mean([x[interior].mean() for x in fields])),
            "visible_pixels": int(keep.sum()), "ssim_centers": int(interior.sum())}


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    started = time.monotonic()
    result = read(OUT / "results.json")
    plan = read(OUT / "plan.json")
    parent = read(PARENT / "results.json")
    parent_plan = read(PARENT / "plan.json")
    cases = {c["id"]: c for c in parent_plan["cases"]}
    assert result["complete"] and result["state_before"] == result["state_after"]
    assert result["dgp_forwards"] == 32 and result["seconds"] <= plan["cap_seconds"]
    assert not result["training"] and not result["app_adopted"]
    assert all(result[k] == 0 for k in ("completion_forwards", "detector_forwards",
                                       "optimizer_updates", "backward_calls"))
    assert plan["source_sha256"] == sha(ROOT / "scripts/compare_dgp_completion_context_v3.py")
    assert plan["parent_results_sha256"] == sha(PARENT / "results.json")
    assert plan["parent_audit_sha256"] == sha(PARENT / "saved_output_audit.json")
    assert parent["plan_sha256"] == sha(PARENT / "plan.json")
    for path, expected in parent_plan["sources_sha256"].items():
        assert sha(ROOT / path) == expected, path
    for path, expected in parent["artifacts_sha256"].items():
        assert sha(PARENT / path) == expected, path
    for path, expected in result["artifacts_sha256"].items():
        assert sha(OUT / path) == expected, path
    groups = defaultdict(list)
    counts = {name: {"MSE_better": 0, "SSIM_better": 0, "cases": 0}
              for name in ("off", "v3_covered_input")}
    control_checks = 0
    verified = []
    for row in result["rows"]:
        assert time.monotonic() - started < 180
        case = cases[row["id"]]
        recorded = next(r for r in parent["rows"] if r["id"] == row["id"])
        context = pixels(PARENT / recorded["assisted"]["off"]["output"])
        removal = pixels(ROOT / case["reviewed"])[..., 0] != 0
        raw = np.load(OUT / (row["id"] + "_raw.npy"), allow_pickle=False)
        assert raw.shape == (256, 256, 3) and raw.dtype == np.float32
        assert np.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 1
        expected = np.floor(raw * np.float32(255)).astype(np.uint8)
        metadata = read(PARENT / "metadata" / (row["id"] + "_on.json"))
        if metadata["display_processing"]["colour_policy"]["applied"]:
            expected = np.repeat(cv2.cvtColor(expected, cv2.COLOR_RGB2GRAY)[..., None], 3, axis=-1)
        expected[removal] = context[removal]
        actual = pixels(OUT / (row["id"] + "_output.png"))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual[removal], context[removal])
        if not removal.any():
            with np.load(PARENT / "stages" / (row["id"] + "_on.npz"), allow_pickle=False) as stages:
                np.testing.assert_array_equal(raw, stages["dgp"][0].transpose(1, 2, 0))
            np.testing.assert_array_equal(actual, pixels(PARENT / recorded["assisted"]["on"]["output"]))
            control_checks += 1
        if case["synthetically_degraded"]:
            target = pixels(ROOT / cases[row["id"].replace("_degraded", "_native")]["input"])
            observations = {"off": measure(context, target, removal),
                "v3_covered_input": measure(pixels(PARENT / recorded["assisted"]["on"]["output"]), target, removal),
                "completed_context": measure(actual, target, removal)}
            for mode, values in observations.items():
                for key, value in values.items():
                    assert abs(value - row["paired_known_nonremoved"][mode][key]) < 1e-12
                groups[(row["family"], mode)].append(values)
            for baseline in counts:
                counts[baseline]["cases"] += 1
                counts[baseline]["MSE_better"] += int(observations["completed_context"]["MSE_visible"] < observations[baseline]["MSE_visible"])
                counts[baseline]["SSIM_better"] += int(observations["completed_context"]["SSIM_visible_valid_windows"] > observations[baseline]["SSIM_visible_valid_windows"])
        verified.append({"id": row["id"], "removal_pixels": int(removal.sum()),
                         "raw_sha256": sha(OUT / (row["id"] + "_raw.npy")),
                         "output_sha256": sha(OUT / (row["id"] + "_output.png"))})
    aggregated = []
    for (family, mode), values in groups.items():
        mse = float(np.mean([r["MSE_visible"] for r in values]))
        aggregated.append({"family": family, "mode": mode, "cases": len(values),
            "mean_MSE_visible": mse, "PSNR_from_mean_MSE": float(-10*np.log10(max(mse, 1e-12))),
            "mean_SSIM_visible_valid_windows": float(np.mean([r["SSIM_visible_valid_windows"] for r in values]))})
    # Four rows per sheet keep 256-pixel cells readable at original resolution.
    sheets = []
    for start in range(0, len(result["rows"]), 4):
        page = Image.new("RGB", (1072, 1192), "white")
        draw = ImageDraw.Draw(page)
        for col, label in enumerate(("Input", "Reviewed completion / Off", "Current v3 DGP On", "DGP on completed context")):
            draw.text((col*268+5, 5), label, fill="black")
        for index, row in enumerate(result["rows"][start:start+4]):
            y = 24 + index*292
            case = cases[row["id"]]
            source = next(r for r in parent["rows"] if r["id"] == row["id"])
            paths = (ROOT/case["input"], PARENT/source["assisted"]["off"]["output"],
                     PARENT/source["assisted"]["on"]["output"], OUT/(row["id"]+"_output.png"))
            draw.text((5, y), row["id"], fill="black")
            for col, path in enumerate(paths):
                page.paste(Image.fromarray(pixels(path)), (col*268+5, y+24))
        path = OUT / ("context-comparison-%02d.png" % (start//4+1))
        if path.exists():
            raise ValueError("Preserve original context sheet")
        page.save(path)
        sheets.append({"file": path.name, "sha256": sha(path), "cases": [r["id"] for r in result["rows"][start:start+4]]})
    receipt = {"complete": True, "seconds": time.monotonic()-started, "cap_seconds": 180,
        "results_sha256": sha(OUT/"results.json"), "source_sha256": sha(Path(__file__)),
        "artifact_bindings": len(result["artifacts_sha256"]), "exact_compositions": len(verified),
        "empty_mask_exact_controls": control_checks, "verified": verified,
        "improvement_counts": counts, "paired_known_nonremoved_groups": aggregated,
        "sheets": sheets, "model_forwards": 0, "training_calls": 0,
        "metric_scope": "Reused paired photographic synthetic degradation outside frozen operator footprint; background/unmarked covering can remain. No aligned clean hidden face or native CCTV reference.",
        "native_cctv_metrics": None, "hidden_metrics": None, "app_adopted": False,
        "family_qualification": False, "independent_final_review": False}
    write(OUT/"saved_output_audit.json", receipt)
    support_rows = []
    for case in cases.values():
        if not case["protected"]:
            continue
        support = pixels(ROOT / case["protected"])[..., 0] != 0
        source = next(r for r in parent["rows"] if r["id"] == case["id"])
        generated = not source["assisted"]["off"]["rejected"]
        support_rows.append({"id": case["id"], "protected_file": case["protected"],
            "protected_file_sha256": sha(ROOT / case["protected"]),
            "protected_pixels": int(support.sum()), "generated": generated,
            "nonvacuous_generated_protection": bool(support.any() and generated)})
    assert not any(r["nonvacuous_generated_protection"] for r in support_rows)
    clarification = {"date": "2026-10-05", "complete": True,
        "parent_saved_output_audit_sha256": sha(PARENT/"saved_output_audit.json"),
        "rows": support_rows,
        "finding": "All generated cases with a protected-region file have zero protected pixels. Only the excluded three-quarter case has a 60-pixel protected region. Zero changed-pixel records in the original receipt are vacuous; they do not establish DGP On eyewear/hair preservation.",
        "valid_nonvacuous_check": "Every accepted Off output exactly preserves all original pixels outside removal. The four empty-mask Off controls preserve all 65,536 pixels. DGP On changes visible appearance and is not byte-preserving.",
        "historical_receipt_modified": False, "model_forwards": 0, "training_calls": 0}
    write(PARENT/"protected_support_clarification.json", clarification)
    print(json.dumps({"complete": True, "seconds": receipt["seconds"], "exact_compositions": len(verified),
                      "empty_mask_exact_controls": control_checks, "improvement_counts": counts}))


if __name__ == "__main__":
    main()
