"""Independent artifact/pixel recount of the frozen LaMa comparison, no models."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "64ff7321ca0580ba9af8df0ab247a22bf3c9171cdfed5f6f01021b2023829d5b"
RESULT = "2dde4543215701641323aa0661bd3314a00e87fc2027df2450360770dd055b94"


def sha(path):
    with Path(path).open("rb") as value:
        return hashlib.file_digest(value, "sha256").hexdigest()


def pixels(path, mode):
    with Image.open(path) as image:
        return np.asarray(image.convert(mode)).copy()


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    folder = ROOT / "outputs/practical_lama_outputs_v1"
    destination = folder / "independent_verification.json"
    require(not destination.exists(), "Preserve previous audit")
    protocol_path = ROOT / "outputs/practical_lama_protocol_v1.json"
    require(sha(protocol_path) == PROTOCOL and sha(folder / "results.json") == RESULT, "Frozen evidence changed")
    p = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    for name, expected in p["artifact_sha256"].items():
        require(sha(ROOT / name) == expected, f"Artifact changed: {name}")
    native = json.loads((ROOT / p["native_protocol"]).read_text(encoding="utf-8"))
    gallery = ROOT / native["gallery_root"]
    for name, expected in native["assets_sha256"].items():
        require(sha(gallery / name) == expected, "Gallery changed")
    cases = {case["id"]: case for case in native["cases"]}
    keys = {(row["id"], row["arm"]) for row in result["rows"]}
    require(len(keys) == len(result["rows"]) == 20 and keys == {(key, arm) for key in cases for arm in p["arms"]}, "Missing/duplicate outputs")
    changed, nonempty, bypasses = 0, 0, 0
    for row in result["rows"]:
        case = cases[row["id"]]
        original = pixels(gallery / case["input"], "RGB")
        mask = pixels(gallery / case["removal_proposal"], "L")
        require(np.isin(mask, (0, 255)).all(), "Nonbinary removal mask")
        mask = mask == 255
        output = pixels(folder / row["output"], "RGB")
        require(sha(folder / row["output"]) == row["output_sha256"], "Saved output changed")
        require(output.shape == original.shape == (256, 256, 3), "Unexpected dimensions")
        outside = int(np.any(output != original, axis=2)[~mask].sum())
        require(outside == 0 and row["exact_outside_reviewed_mask"], "Visible pixels changed")
        require(row["mask_pixels"] == int(mask.sum()) and row["empty_mask_bypass"] == (not bool(mask.any())), "Mask/bypass count differs")
        require(row["hidden_face_mae"] is None, "Invented hidden target metric")
        changed += outside
        nonempty += int(mask.any())
        bypasses += int(not mask.any())
    require(result["complete"] and not result["failures"] and not result["promoted"], "Execution/selection status differs")
    require(result["requests"] == 20 and result["nonempty_generator_forwards"] == nonempty == 16 and bypasses == 4, "Call recount differs")
    require(result["generator_state_before"] == result["generator_state_after"] and result["state_unchanged"], "Recorded state differs")
    require(not result["optimizer_constructed"] and result["optimizer_updates"] == 0, "Training recorded")
    require(sha(folder / "preview.png") == result["preview_sha256"], "Preview changed")
    report = {"verified": True, "date": "2026-10-02", "protocol_sha256": PROTOCOL, "results_sha256": RESULT,
              "auditor_sha256": sha(__file__), "audited_outputs": 20, "nonempty_outputs": nonempty,
              "bypasses": bypasses, "outside_reviewed_mask_changed_pixels": changed,
              "new_model_forwards": 0, "optimizer_updates": 0, "hidden_face_ground_truth": False,
              "quality_claim": "Artifact and visible-byte preservation only; separate visual review remains necessary"}
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verified": True, "outputs": 20, "outside_changed_pixels": changed, "sha256": sha(destination)}))


if __name__ == "__main__":
    main()
