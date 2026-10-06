"""Verify saved context-margin arrays/composition without neural inference."""
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/"outputs/dgp_app_covering_review_v3"
OUT = ROOT/"outputs/completion_margin_full_v3"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pixels(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB")).copy()


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    started = time.monotonic()
    result = read(OUT/"results.json")
    plan = read(OUT/"plan.json")
    parent = read(PARENT/"results.json")
    assert result["complete"] and result["plan_sha256"] == sha(OUT/"plan.json")
    assert result["seconds"] <= plan["cap_seconds"] and result["state_before"] == result["state_after"]
    assert result["forwards"] == {"completion": 29, "dgp": 0, "detector": 0}
    assert result["optimizer_updates"] == result["backward_calls"] == 0
    for path, expected in plan["sources_sha256"].items():
        assert sha(ROOT/path) == expected, path
    for path, expected in result["artifacts_sha256"].items():
        assert sha(OUT/path) == expected, path
    cases = {c["id"]: c for c in plan["cases"]}
    control = plan["zero_radius_control"]
    raw0 = np.load(OUT/"raw/zero_control.npy", allow_pickle=False)
    with np.load(PARENT/"stages"/(control+"_off.npz"), allow_pickle=False) as stages:
        np.testing.assert_array_equal(raw0, stages["completion"][0].transpose(1, 2, 0))
    checks, supported = [], []
    exact_outputs = exact_aliases = empty_controls = rejected = nonempty = 0
    for row in result["rows"]:
        assert time.monotonic()-started < 180
        case = cases[row["id"]]
        assert row["rejected"] == (not case["eligible"])
        if row["rejected"]:
            assert case["input_review"] in ("needs_clearer", "out_of_scope")
            assert "outputs" not in row and row["rejection_message"]
            rejected += 1
            continue
        supported.append(row)
        source = next(r for r in parent["rows"] if r["id"] == row["id"])
        image = pixels(ROOT/case["input"])
        mask = pixels(ROOT/case["removal"])[..., 0] != 0
        context = cv2.dilate(mask.astype(np.uint8), np.ones((13, 13), np.uint8)) != 0
        np.testing.assert_array_equal(pixels(OUT/"conditioning_masks"/(row["id"]+".png"))[..., 0] != 0, context)
        assert int(mask.sum()) == case["removal_pixels"] and int(context.sum()) == case["conditioning_pixels"]
        if mask.any():
            raw = np.load(OUT/"raw"/row["raw_completion"], allow_pickle=False)
            assert raw.dtype == np.float32 and raw.shape == (256, 256, 3)
            assert np.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 1
            np.testing.assert_array_equal(raw[~mask], (image.astype(np.float32)/np.float32(255))[~mask])
            generated = np.floor(raw*np.float32(255)).astype(np.uint8)
            nonempty += 1
        else:
            assert row["raw_completion"] is None
            generated = image.copy()
            empty_controls += 1
        meta = read(PARENT/"metadata"/(row["id"]+"_off.json"))
        if meta["display_processing"]["colour_policy"]["applied"]:
            generated = np.repeat(cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY)[..., None], 3, axis=-1)
        outputs = {}
        for mode in ("off", "on"):
            baseline = pixels(PARENT/source["assisted"][mode]["output"])
            expected = image.copy() if mode == "off" else baseline.copy()
            expected[mask] = generated[mask]
            output = pixels(OUT/row["outputs"][mode])
            np.testing.assert_array_equal(output, expected)
            np.testing.assert_array_equal(output[~mask], baseline[~mask])
            if mode == "off":
                np.testing.assert_array_equal(output[~mask], image[~mask])
            outputs[mode] = output
            exact_outputs += 1
        selected = "on" if source["assisted"]["auto"]["restoration_applied"] else "off"
        assert selected == row["auto_selected"]
        np.testing.assert_array_equal(pixels(OUT/row["outputs"]["auto"]), outputs[selected])
        np.testing.assert_array_equal(outputs["off"][mask], outputs["on"][mask])
        exact_outputs += 1
        exact_aliases += 1
        previous = pixels(PARENT/source["assisted"]["off"]["output"])
        difference = np.abs(outputs["off"].astype(np.float64)-previous.astype(np.float64))
        checks.append({"id": row["id"], "family": row["family"], "condition": row["condition"],
            "removal_pixels": int(mask.sum()), "added_conditioning_pixels": int((context & ~mask).sum()),
            "changed_completion_pixels": int(np.any(outputs["off"] != previous, axis=-1)[mask].sum()),
            "mean_abs_completion_change_255": float(difference[mask].mean()) if mask.any() else 0.,
            "changed_outside_removal_pixels": 0, "change_is_quality_metric": False})
    assert (exact_outputs, exact_aliases, empty_controls, rejected, nonempty) == (96, 32, 4, 4, 28)
    sheets = []
    for start in range(0, len(supported), 4):
        page = Image.new("RGB", (1340, 1192), "white")
        draw = ImageDraw.Draw(page)
        for column, title in enumerate(("Input", "Current Off completion", "Context6 Off", "Context6 + cached DGP On", "Context6 + fixed Auto")):
            draw.text((268*column+5, 5), title, fill="black")
        for index, row in enumerate(supported[start:start+4]):
            y = 24+292*index
            case = cases[row["id"]]
            baseline = next(r for r in parent["rows"] if r["id"] == row["id"])
            images = [pixels(ROOT/case["input"]), pixels(PARENT/baseline["assisted"]["off"]["output"])]
            images.extend(pixels(OUT/row["outputs"][mode]) for mode in ("off", "on", "auto"))
            draw.text((5, y), row["id"], fill="black")
            for column, image in enumerate(images):
                page.paste(Image.fromarray(image), (268*column+5, y+24))
        path = OUT/("full-comparison-%02d.png" % (start//4+1))
        if path.exists():
            raise ValueError("Preserve original full comparison sheet")
        page.save(path)
        sheets.append({"path": path.name, "sha256": sha(path), "cases": [r["id"] for r in supported[start:start+4]]})
    receipt = {"complete": True, "seconds": time.monotonic()-started, "cap_seconds": 180,
        "results_sha256": sha(OUT/"results.json"), "source_sha256": sha(Path(__file__)),
        "source_bindings": len(plan["sources_sha256"]), "output_bindings": len(result["artifacts_sha256"]),
        "zero_control_raw_exact": True, "exact_compositions": exact_outputs, "exact_auto_aliases": exact_aliases,
        "empty_mask_exact_controls": empty_controls, "input_exclusions": rejected,
        "visible_off_original_and_on_cached_exact": True, "completed_pixels_identical_across_modes": True,
        "context_masks_exact_dilation": True, "sheets": sheets, "checks": checks,
        "paired_nonremoved_metrics": "Unchanged exactly because nonremoved pixels and valid SSIM windows are identical to parent outputs",
        "native_cctv_metrics": None, "hidden_metrics": None, "model_forwards": 0, "training_calls": 0,
        "family_qualification": False, "independent_final_review": False, "app_changed": False}
    write(OUT/"saved_output_audit.json", receipt)
    print(json.dumps({"complete": True, "seconds": receipt["seconds"], "exact_compositions": exact_outputs,
                      "exact_auto_aliases": exact_aliases, "input_exclusions": rejected}))


if __name__ == "__main__":
    main()
