"""Recount saved V3 pixels and frozen sources without loading any model."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA = "4a256b9a00c25f68414010877a8cf96c1e3a9a283842360ed64978a676a5c5d3"
RESULT_SHA = "1c79c0627697a353e57f47adb8943e1e5fda8af1ed51ec00f53aa01dbf3ef17c"


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def pixels(path, mode):
    with Image.open(path) as image:
        return np.asarray(image.convert(mode)).copy()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    folder = ROOT / "outputs/practical_footprint_outputs_v3"
    destination = folder / "independent_verification.json"
    require(not destination.exists(), "Preserve the previous verification")
    protocol_path = ROOT / "outputs/practical_footprints_v3/frozen_protocol.json"
    require(sha(protocol_path) == PROTOCOL_SHA, "Frozen protocol changed")
    require(sha(folder / "results.json") == RESULT_SHA, "Recorded results changed")
    p = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    for collection in (p["artifact_sha256"], p["asset_sha256"]):
        for name, expected in collection.items():
            require(sha(ROOT / name) == expected, f"Frozen artifact changed: {name}")
    source_path = ROOT / p["source_protocol"]
    require(sha(source_path) == p["source_protocol_sha256"], "Historical protocol changed")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    for name, expected in source["assets_sha256"].items():
        require(sha(ROOT / source["gallery_root"] / name) == expected, "Historical gallery changed")
    cases = {case["id"]: case for case in p["cases"]}
    expected_keys = {(key, backend) for key in cases for backend in p["backends"]}
    require(len(result["rows"]) == len(expected_keys) == 12, "Missing or duplicate requests")
    require({(row["id"], row["backend"]) for row in result["rows"]} == expected_keys, "Request set differs")
    baselines, reused = {}, 0
    for backend, spec in p["backends"].items():
        old_root = ROOT / spec["previous_run"]
        old = json.loads((old_root / "results.json").read_text(encoding="utf-8"))
        baselines[backend] = {}
        for row in old["rows"]:
            if row["arm"] == spec["previous_arm"] and row["id"] in cases:
                path = old_root / row["output"]
                require(sha(path) == row["output_sha256"], "Reused baseline changed")
                baselines[backend][row["id"]] = path
                reused += 1
        require(set(baselines[backend]) == set(cases), "Missing reused baseline")
    nonempty, bypasses, outside_changed, protected_changed = 0, 0, 0, 0
    rows = []
    for row in result["rows"]:
        case = cases[row["id"]]
        original = pixels(ROOT / case["input"], "RGB")
        mask_pixels = pixels(ROOT / case["new_removal"], "L")
        protected_pixels = pixels(ROOT / case["preserved_region"], "L")
        require(np.isin(mask_pixels, (0, 255)).all() and np.isin(protected_pixels, (0, 255)).all(), "Nonbinary proposal")
        selected, protected = mask_pixels == 255, protected_pixels == 255
        require(not (selected & protected).any(), "Preserved eyewear intersects removal")
        output_path = folder / row["output"]
        output = pixels(output_path, "RGB")
        require(sha(output_path) == row["output_sha256"], "Saved output changed")
        require(original.shape == output.shape == (256, 256, 3), "Dimensions differ")
        changed = np.any(output != original, axis=2)
        outside = int(changed[~selected].sum())
        preserved = int(changed[protected].sum())
        require(outside == preserved == 0, "Visible or preserved-eyewear pixels changed")
        require(row["exact_outside_own_mask"] and row["protected_eyewear_changed_pixels"] == preserved, "Recorded preservation differs")
        require(row["mask_pixels"] == int(selected.sum()), "Removal count differs")
        require(row["empty_mask_bypass"] == (not bool(selected.any())), "Bypass differs")
        old = pixels(ROOT / case["old_removal"], "L") == 255
        added = selected & ~old
        excluded = old & ~selected
        require(case["old_pixels"] == int(old.sum()) and case["new_pixels"] == int(selected.sum()), "Proposal recount differs")
        require(case["added_pixels"] == int(added.sum()) and case["excluded_preserved_pixels"] == int(excluded.sum()), "Proposal delta differs")
        previous = pixels(baselines[row["backend"]][row["id"]], "RGB")
        difference = np.abs(output.astype(np.float32) - previous.astype(np.float32)) / 255
        mean_change = float(difference[added].mean()) if added.any() else None
        observed = row["mean_change_from_previous_in_added_area"]
        require((mean_change is None and observed is None) or (mean_change is not None and observed is not None and abs(mean_change - observed) < 1e-7), "Saved-pixel delta differs")
        require(row["hidden_face_mae"] is None, "No hidden target is available")
        nonempty += int(selected.any())
        bypasses += int(not selected.any())
        outside_changed += outside
        protected_changed += preserved
        rows.append({"id": row["id"], "backend": row["backend"], "outside_changed_pixels": outside,
                     "protected_changed_pixels": preserved, "added_pixels": int(added.sum()),
                     "excluded_pixels": int(excluded.sum())})
    require(result["complete"] and not result["failures"] and not result["promoted"], "Execution status differs")
    require(result["requests"] == 12 and result["nonempty_generator_forwards"] == nonempty == 8 and bypasses == 4, "Call recount differs")
    require(reused == result["reused_baseline_rows"] == 12, "Baseline recount differs")
    for state in result["states"].values():
        require(state["before"] == state["after"] and state["unchanged"], "Recorded generator state differs")
    require(result["new_detector_forwards"] == result["optimizer_updates"] == 0 and not result["optimizer_constructed"], "Unexpected training or detection recorded")
    require(sha(folder / "preview.png") == result["preview_sha256"], "Preview changed")
    report = {"verified": True, "date": "2026-10-02", "protocol_sha256": PROTOCOL_SHA,
              "results_sha256": RESULT_SHA, "auditor_sha256": sha(__file__), "audited_outputs": 12,
              "historical_gallery_assets_verified": len(source["assets_sha256"]),
              "reused_baseline_rows_verified": reused, "nonempty_outputs": nonempty, "empty_bypasses": bypasses,
              "outside_own_mask_changed_pixels": outside_changed, "protected_eyewear_changed_pixels": protected_changed,
              "new_model_forwards": 0, "optimizer_updates": 0, "hidden_face_ground_truth": False, "rows": rows,
              "quality_claim": "Saved-artifact and pixel verification; visual usefulness is a separate developmental review"}
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verified": True, "outputs": 12, "outside_changed_pixels": outside_changed,
                      "protected_changed_pixels": protected_changed, "sha256": sha(destination)}))


if __name__ == "__main__":
    main()
