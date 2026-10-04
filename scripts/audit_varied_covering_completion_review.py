"""Reconstruct each estimated-face PNG from captured floats; no model inference."""
import hashlib
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_varied_covering_results import require, sha, read, write, binary, exact
from scripts.audit_real_camera_completion_review import rgb

RUN = ROOT / "outputs/varied_covering_completion_review_v1"


def main():
    destination = RUN / "independent_verification.json"
    require(not destination.exists(), "Preserve previous independent output audit")
    protocol, result = read(RUN / "frozen_protocol.json"), read(RUN / "results.json")
    require(result["complete"] is True and result["protocol_sha256"] == sha(RUN / "frozen_protocol.json")
            and result["state_before"] == result["state_after"] and result["promoted"] is False, "Output state/binding differs")
    exact({key: result[key] for key in ("generation_requests", "detector_forwards", "optimizer_updates")},
          {"generation_requests": 20, "detector_forwards": 0, "optimizer_updates": 0}, "downstream scope")
    require(type(result["seconds_excluding_loading"]) is float and 0 < result["seconds_excluding_loading"] < 300,
            "Output comparison exceeds finite budget")
    require(len(protocol["cases"]) == 10 and protocol["arms"] == ["existing91", "varied133"]
            and protocol["restoration"] == "auto" and protocol["training"] is False
            and protocol["application_checkpoint_selection"] is False, "Frozen output scope differs")
    for name, pin in protocol["assets_sha256"].items(): require(sha(ROOT / name) == pin, "Downstream asset changed")
    expected = {(case["id"], arm) for case in protocol["cases"] for arm in protocol["arms"]}
    require(len(result["rows"]) == 20 and {(row["id"], row["arm"]) for row in result["rows"]} == expected,
            "Downstream case membership differs")
    files, counts, verified = {"preview.png"}, {"completion": 0, "restoration": 0}, []
    for row in result["rows"]:
        case = next(case for case in protocol["cases"] if case["id"] == row["id"])
        require(case["hidden_ground_truth"] is None and row["hidden_ground_truth"] is None, "No hidden facial target exists")
        require(row["mask"] == case["masks"][row["arm"]], "Automatic mask binding differs")
        original, output = rgb(ROOT / case["input"]), rgb(RUN / row["output"])
        require(row["output"] == f"outputs/{row['arm']}_{row['id']}.png"
                and row["output_sha256"] == sha(RUN / row["output"]), "Output filename/digest differs")
        mask = binary(ROOT / row["mask"])
        observed = original.transpose(2, 0, 1)[None].astype(np.float32) / 255
        stages = {}
        for kind, name in row["stage_arrays"].items():
            require(kind in counts and name == f"stages/{row['arm']}_{row['id']}_{kind}.npy", "Stage name differs")
            value = np.load(RUN / name, allow_pickle=False)
            require(value.dtype == np.float32 and value.shape == (1, 3, 256, 256) and np.isfinite(value).all(), "Invalid captured stage")
            stages[kind] = value; counts[kind] += 1; files.add(name)
        require(("completion" in stages) == bool(mask.any()), "Empty-hair bypass/actual completion differs")
        completed = np.where(mask[None, None], stages.get("completion", observed), observed)
        restored = "restoration" in stages
        require(row["metadata"]["restoration_applied"] is restored, "Restoration stage declaration differs")
        blended = np.where(mask[None, None], completed, .5 * stages["restoration"] + .5 * completed) if restored else completed
        generated = mask[None, None].repeat(3, axis=1)
        require(np.array_equal(blended[generated], completed[generated]), "Visible restoration changed generated region")
        quantized = (np.clip(blended[0].transpose(1, 2, 0), 0, 1) * 255).round().astype(np.uint8)
        if not restored: quantized[~mask] = original[~mask]
        support = cv2.erode((~mask).astype(np.uint8), np.ones((9, 9), np.uint8)) != 0
        filtered = cv2.GaussianBlur(original.astype(np.float32), (5, 5), 1)
        chroma = filtered.max(-1) - filtered.min(-1)
        signal = float(chroma[support].mean()) if int(support.sum()) >= 512 else None
        grayscale = signal is not None and signal <= 4
        if grayscale:
            projected = np.repeat(cv2.cvtColor(quantized, cv2.COLOR_RGB2GRAY)[..., None], 3, -1)
            if restored: quantized = projected
            else: quantized[mask] = projected[mask]
        require(np.array_equal(quantized, output), "Captured stages do not reproduce estimated-face PNG")
        metadata = row["metadata"]
        require(metadata["policy"] == "reviewed-face-workflow-v2" and metadata["restoration_requested"] == "auto"
                and metadata["additional_mask_expansion"] == 0 and metadata["optimizer_updates"] == 0
                and metadata["mask_source"] == "frozen candidate automatic proposal; no manual edits", "Downstream processing policy differs")
        for key, array in (("original_rgb_sha256", original), ("removal_mask_sha256", mask.astype(np.uint8)), ("output_rgb_sha256", output)):
            exact(metadata[key], hashlib.sha256(array.tobytes()).hexdigest(), "Decoded pixel binding")
        exact(metadata["color_policy"]["grayscale_input"], grayscale, "Grayscale decision")
        exact(metadata["color_policy"]["mean_visible_channel_range_255"], signal, "Visible grayscale signal")
        files.add(row["output"])
        verified.append({"id": row["id"], "arm": row["arm"], "final_png_reconstructed_exactly": True,
                         "mask_empty": not bool(mask.any()), "restoration_applied": restored,
                         "generated_region_preserved_before_palette": True, "grayscale_projection": grayscale,
                         "hidden_ground_truth": None})
    exact(result["forwards"], counts, "Captured forward membership")
    require(counts["completion"] == 18 and counts["restoration"] <= 20, "Finite generation/restoration budget differs")
    require(set(result["artifacts_sha256"]) == files, "Downstream artifact membership differs")
    for name, pin in result["artifacts_sha256"].items(): require(sha(RUN / name) == pin, "Downstream artifact changed")
    record = {"format": "dgp-varied-covering-downstream-independent-verification-v1", "date": "2026-10-03", "complete": True,
              "auditor_sha256": sha(__file__), "protocol_sha256": sha(RUN / "frozen_protocol.json"),
              "results_sha256": sha(RUN / "results.json"), "files_verified": len(files), "rows": verified,
              "captured_model_forwards": counts, "model_forwards_in_audit": 0, "optimizer_updates": 0,
              "promoted": False, "visual_review_pending": True,
              "limitations": ["Ten previously inspected degraded diagnostic sources; native conditions and assisted results remain separate",
                              "Captured floats verify final composition; plausibility/removal/visible appearance require visual review",
                              "Hidden facial truth is unknown; empty hair outputs do not demonstrate covering removal"]}
    write(destination, record)
    print({"complete": True, "pngs_reconstructed": 20, "verification_sha256": sha(destination)})


if __name__ == "__main__": main()
