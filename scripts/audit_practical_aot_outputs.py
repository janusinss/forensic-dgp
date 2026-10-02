"""Independent saved-pixel and frozen-artifact audit of AOT-GAN outputs."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA = "a53c4b8e3a27f30d9ccf0c49c7e04454f6303d72b3c6c08c5977ee168fb7d09b"
RESULT_SHA = "bf77388894dd27b68d7f8065a237b6d4a21855770eded8c88e1b79e5ce32a278"


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def pixels(path, mode):
    with Image.open(path) as image:
        return np.asarray(image.convert(mode)).copy()


def require(condition, text):
    if not condition:
        raise ValueError(text)


def main():
    folder = ROOT / "outputs/practical_aot_outputs_v1"
    destination = folder / "independent_verification.json"
    require(not destination.exists(), "Preserve prior audit")
    protocol = ROOT / "outputs/practical_aot_protocol_v1.json"
    require(sha(protocol) == PROTOCOL_SHA and sha(folder / "results.json") == RESULT_SHA, "Frozen evidence changed")
    p = json.loads(protocol.read_text(encoding="utf-8"))
    result = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    for name, expected in p["artifact_sha256"].items():
        require(sha(ROOT / name) == expected, f"Frozen artifact changed: {name}")
    native = json.loads((ROOT / p["native_protocol"]).read_text(encoding="utf-8"))
    gallery = ROOT / native["gallery_root"]
    for name, expected in native["assets_sha256"].items():
        require(sha(gallery / name) == expected, "Frozen gallery changed")
    cases = {case["id"]: case for case in native["cases"]}
    require(len(result["rows"]) == 10 and {row["id"] for row in result["rows"]} == set(cases), "Missing/duplicate rows")
    nonempty, bypasses, outside_changed = 0, 0, 0
    for row in result["rows"]:
        case = cases[row["id"]]
        original = pixels(gallery / case["input"], "RGB")
        removal = pixels(gallery / case["removal_proposal"], "L")
        require(np.isin(removal, (0, 255)).all(), "Nonbinary mask")
        mask = removal == 255
        output = pixels(folder / row["output"], "RGB")
        require(sha(folder / row["output"]) == row["output_sha256"], "Output changed")
        require(original.shape == output.shape == (256, 256, 3), "Dimensions differ")
        changed = int(np.any(output != original, axis=2)[~mask].sum())
        require(changed == 0 and row["exact_outside_reviewed_mask"], "Visible pixels changed")
        require(row["mask_pixels"] == int(mask.sum()) and row["empty_mask_bypass"] == (not bool(mask.any())), "Mask/bypass recount differs")
        require(row["arm"] == "aot512" and row["hidden_face_mae"] is None, "Arm or hidden-target metric differs")
        outside_changed += changed
        nonempty += int(mask.any())
        bypasses += int(not mask.any())
    require(result["complete"] and not result["failures"] and not result["promoted"], "Execution status differs")
    require(result["nonempty_generator_forwards"] == nonempty == 8 and bypasses == 2 and result["requests"] == 10, "Call recount differs")
    require(result["generator_state_before"] == result["generator_state_after"] and result["state_unchanged"], "Recorded generator state changed")
    require(result["optimizer_updates"] == 0 and not result["optimizer_constructed"], "Training recorded")
    require(sha(folder / "preview.png") == result["preview_sha256"], "Preview changed")
    baseline_count = 0
    for value in p["baselines"].values():
        old = json.loads((ROOT / value["root"] / "results.json").read_text(encoding="utf-8"))
        for row in old["rows"]:
            if row["arm"] == value["arm"]:
                require(sha(ROOT / value["root"] / row["output"]) == row["output_sha256"], "Reused baseline changed")
                baseline_count += 1
    require(baseline_count == result["reused_baseline_rows"] == 20, "Baseline recount differs")
    report = {"verified": True, "date": "2026-10-02", "protocol_sha256": PROTOCOL_SHA, "results_sha256": RESULT_SHA,
              "auditor_sha256": sha(__file__), "audited_outputs": 10, "reused_baseline_rows_verified": baseline_count,
              "nonempty_outputs": nonempty, "empty_bypasses": bypasses,
              "outside_reviewed_mask_changed_pixels": outside_changed,
              "new_model_forwards": 0, "optimizer_updates": 0, "hidden_face_ground_truth": False,
              "quality_claim": "Artifact/pixel verification only; separate assistant visual review is developmental evidence"}
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verified": True, "rows": 10, "outside_changed_pixels": outside_changed,
                      "sha256": sha(destination)}))


if __name__ == "__main__":
    main()
