"""Verify cached palette outputs and the actual browser response; no model loading."""
import base64
import hashlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/grayscale_palette_comparison_v2"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pixels(path, mode="RGB"):
    with Image.open(path) as image:
        return np.asarray(image.convert(mode)).copy()


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    destination = OUT / "independent_verification.json"
    require(not destination.exists(), "Preserve previous audit")
    require(sha(OUT / "frozen_protocol.json") == "893909bc8d020ad4239cd48ffdc34bf5a5007e55a352f3f17a22b14f69a3b23e", "Protocol changed")
    require(sha(OUT / "results.json") == "3d6e4b90fc3b6f39c230fa06dabc783412542f664d21a9a844d552be95214242", "Results changed")
    protocol = json.loads((OUT / "frozen_protocol.json").read_text())
    result = json.loads((OUT / "results.json").read_text())
    base_dir = ROOT / "outputs/broad_face_workflow_outputs_v1"
    base = json.loads((base_dir / "results.json").read_text())
    source = json.loads((ROOT / "outputs/broad_covering_gallery_v1/frozen_protocol.json").read_text())
    cases = {row["id"]: row for row in source["cases"]}
    baseline = {(row["id"], row["arm"]): row for row in base["rows"] if row["output"]}
    require(len(result["rows"]) == 28 and result["complete"], "Incomplete palette comparison")
    require({(row["id"], row["arm"]) for row in result["rows"]} == set(baseline), "Rows differ")
    for path, digest in protocol["assets_sha256"].items():
        require(sha(ROOT / path) == digest, "Frozen asset changed: " + path)
    require(sha(ROOT / "face_color_policy.py") == protocol["policy_sha256"], "Runtime palette policy changed")
    old = ROOT / "outputs/grayscale_palette_comparison_v1"
    old_protocol = json.loads((old / "frozen_protocol.json").read_text())
    require(sha(old / "policy_snapshot.py") == old_protocol["policy_sha256"], "V1 policy snapshot differs")
    chosen, off_changes, color_changes, gray_rows = 0, 0, 0, []
    for row in result["rows"]:
        case = cases[row["id"]]
        prior = baseline[(row["id"], row["arm"])]
        original = pixels(ROOT / case["input"])
        reference = pixels(ROOT / case["reference"])
        previous = pixels(base_dir / prior["output"])
        output = pixels(OUT / row["output"])
        mask = pixels((base_dir if prior["mask_root"] == "output" else ROOT) / prior["mask"], "L") == 255
        require(sha(OUT / row["output"]) == row["output_sha256"], "Output fingerprint differs")
        support = cv2.erode((~mask).astype(np.uint8), np.ones((9, 9), np.uint8)) != 0
        sample = cv2.GaussianBlur(original.astype(np.float32), (5, 5), 1)
        chroma = sample.max(-1) - sample.min(-1)
        measured = float(chroma[support].mean()) if int(support.sum()) >= 512 else None
        gray_input = measured is not None and measured <= 4
        require(gray_input == row["signal"]["grayscale_input"], "Input-only selection differs")
        require(measured == row["signal"]["mean_visible_channel_range_255"], "Input chroma differs")
        expected = previous.copy()
        if gray_input:
            chosen += 1
            gray = np.repeat(cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)[..., None], 3, -1)
            if row["restoration_applied"]:
                expected = gray
            else:
                expected[mask] = gray[mask]
            gray_rows.append({"id": row["id"], "arm": row["arm"], "visible_chroma": measured})
        require(np.array_equal(expected, output), "Palette composition differs")
        changed = int(np.any(output != previous, -1)[~mask].sum())
        if not row["restoration_applied"]:
            off_changes += changed
            require(not np.any(output[~mask] != original[~mask]), "Off output changed visible input")
        if not gray_input:
            color_changes += int(np.any(output != previous, -1).sum())
        reviewed = pixels(ROOT / case["proposal"], "L") == 255
        mae = float(np.abs(output.astype(float) - reference.astype(float))[~reviewed].mean() / 255)
        require(abs(mae - row["visible_mae_outside_fixed_operator_proposal"]) < 1e-12, "Visible MAE differs")
    require(chosen == 8 and off_changes == 0 and color_changes == 0, "Palette preservation criteria failed")
    require(sha(OUT / "assisted_grayscale_preview.png") == result["preview_sha256"], "Preview changed")
    browser_path = ROOT / "scratch/browser-gray-v2.json"
    browser = json.loads(browser_path.read_text())
    decoded = {}
    for name, mode in (("original", "RGB"), ("mask", "L"), ("output", "RGB")):
        with Image.open(io.BytesIO(base64.b64decode(browser[name].split(",", 1)[1]))) as image:
            decoded[name] = np.asarray(image.convert(mode)).copy()
    case_id = "val_25_hand_mouth_native"
    require(np.array_equal(decoded["original"], pixels(ROOT / cases[case_id]["input"])), "Browser source differs")
    require(np.array_equal(decoded["mask"], pixels(ROOT / cases[case_id]["proposal"], "L")), "Browser reviewed mask differs")
    require(np.array_equal(decoded["output"], pixels(OUT / ("assisted_" + case_id + ".png"))), "Browser pixels differ from cached V2")
    metadata = browser["metadata"]
    require(metadata["policy"] == "reviewed-face-workflow-v2" and metadata["color_policy"]["applied"], "Browser policy differs")
    require(hashlib.sha256(decoded["output"].tobytes()).hexdigest() == metadata["output_rgb_sha256"], "Browser decoded-output hash differs")
    report = {"verified": True, "date": "2026-10-02", "protocol_sha256": sha(OUT / "frozen_protocol.json"),
              "results_sha256": sha(OUT / "results.json"), "auditor_sha256": sha(__file__),
              "saved_outputs_verified": 28, "palette_selected": chosen, "off_visible_palette_changes": off_changes,
              "color_input_changed_pixels": color_changes, "v1_policy_snapshot_verified": True,
              "browser_response_sha256": sha(browser_path), "browser_matches_cached_v2_pixels": True,
              "browser_case": case_id, "selected_rows": gray_rows, "new_model_forwards": 0, "optimizer_updates": 0,
              "scope": "Known saved pixels and input-only palette selection; no hidden-face or unseen population claim"}
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: report[key] for key in ("verified", "saved_outputs_verified", "palette_selected", "off_visible_palette_changes", "browser_matches_cached_v2_pixels")}))


if __name__ == "__main__":
    main()
