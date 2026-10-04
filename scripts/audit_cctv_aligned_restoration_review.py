"""Independent geometry/control/raw-pixel checks, no model imports or forwards."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/cctv_alignment_restoration_comparison_v1"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pixels(path, mode="RGB"):
    with Image.open(path) as image:
        return np.asarray(image.convert(mode)).copy()


def require(value, reason):
    if not value:
        raise ValueError(reason)


def main():
    destination = OUT / "independent_verification.json"
    require(not destination.exists(), "Preserve earlier audit")
    policy, execution, results = (read(OUT / name) for name in ("frozen_protocol.json", "execution.json", "results.json"))
    require(results["complete"] and results["protocol_sha256"] == execution["protocol_sha256"] == sha(OUT / "frozen_protocol.json"), "Comparison protocol differs/incomplete")
    for name, pin in policy["assets_sha256"].items():
        require(sha(ROOT / name) == pin, "Frozen review asset changed: "+name)
    for name, pin in results["artifacts_sha256"].items():
        require(sha(OUT / name) == pin, "Saved review artifact changed: "+name)
    require([c["id"] for c in policy["cases"]] == [row["id"] for row in results["rows"]] == ["dev_16to23_05", "dev_24to39_06", "dev_ge40_04"], "Missing/reordered cases")
    checked = []
    for case, row in zip(policy["cases"], results["rows"]):
        matrix = np.asarray(case["matrix"], dtype=np.float64)
        common = pixels(ROOT / case["common_input"])
        aligned = cv2.warpAffine(common, matrix, (256, 256), borderMode=cv2.BORDER_REFLECT)
        require(np.array_equal(aligned, pixels(ROOT / case["aligned_input"])) and np.array_equal(aligned, pixels(OUT / row["outputs"]["input_aligned"])), "Aligned input pixels differ")
        source = case["source"]
        width, height = source["native_width"], source["native_height"]
        side = max(width, height)
        mask = Image.new("L", (side, side), 0)
        mask.paste(255, ((side-width)//2, (side-height)//2, (side-width)//2+width, (side-height)//2+height))
        common_mask = np.asarray(mask.resize((256, 256), Image.Resampling.NEAREST))
        expected_mask = cv2.warpAffine(common_mask, matrix, (256, 256), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        actual_mask = pixels(OUT / row["outputs"]["observed_support"], "L")
        require(np.array_equal(actual_mask, expected_mask), "Native observed-support mask differs")
        observed = actual_mask == 255
        require(int(observed.sum()) == row["observed_pixels"] and float(observed.mean()) == row["observed_source_fraction"], "Observed support count differs")
        for name in ("dgp", "codeformer"):
            control = np.load(OUT / f"stages/{case['id']}_{name}_control.npy", allow_pickle=False)
            baseline = np.load(ROOT / case["cached_raw_controls"][name], allow_pickle=False)
            expected = cv2.warpAffine(baseline, matrix, (256, 256), borderMode=cv2.BORDER_REFLECT)
            require(np.array_equal(control, expected), "Geometry-matched cached float control differs")
            restored = np.load(OUT / f"stages/{case['id']}_{name}_aligned.npy", allow_pickle=False)
            for arm, value in ((name+"_control", control), (name+"_aligned", restored)):
                require(value.shape == (256, 256, 3) and value.dtype == np.float32 and np.isfinite(value).all() and 0 <= value.min() <= value.max() <= 1, "Invalid raw/control float")
                require(np.array_equal((value*255).astype(np.uint8), pixels(OUT / row["outputs"][arm])), "Raw/control PNG differs from captured float")
            change = float(np.abs(restored-control)[observed].mean())
            require(abs(change-row["same_frame_change_mae_diagnostic_only"][name]) < 1e-12, "Same-frame diagnostic differs")
        checked.append({"id": case["id"], "six_pngs_reconstructed": True, "controls_warped_in_same_frame": True, "observed_pixels": int(observed.sum())})
    require(results["model_forwards"] == {"dgp": 3, "codeformer": 3}, "Forward count differs")
    require(results["state_after"] == execution["state_before"] and not execution["optimizer_constructed"] and results["optimizer_updates"] == 0, "Frozen inference state/scope differs")
    require(results["seconds_after_loading"] <= 90 and results["PSNR"] is None and results["SSIM"] is None, "Budget/unpaired metric claim differs")
    require(not results["reserved_evaluation_used"] and not results["checkpoint_selected"] and not results["application_change"], "Unreviewed selection or reserved inference")
    report = {"verified": True, "date": "2026-10-03", "auditor_sha256": sha(Path(__file__)),
              "protocol_sha256": sha(OUT / "frozen_protocol.json"), "results_sha256": sha(OUT / "results.json"),
              "raw_and_control_float_stages": 12, "pngs_reconstructed": 18, "rows": checked,
              "model_forwards_in_audit": 0, "optimizer_updates": 0, "reserved_inputs_used": False,
              "limitations": "Captured outputs/geometry are reproducible; this does not prove clean anatomical truth, alignment correctness or restoration quality"}
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print({"verified": True, "pngs_reconstructed": 18, "verification_sha256": sha(destination)})


if __name__ == "__main__":
    main()
