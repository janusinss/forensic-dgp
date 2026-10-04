"""Recount conditioning support and reconstruct saved output floats; zero model forwards."""
import hashlib
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_real_camera_results import require, sha, read, write, binary, exact
from scripts.audit_real_camera_completion_review import rgb

RUN = ROOT / "outputs/completion_context_review_v1"


def main():
    destination = RUN / "independent_verification.json"
    require(not destination.exists(), "Preserve earlier independent context audit")
    p, result = read(RUN / "frozen_protocol.json"), read(RUN / "results.json")
    require(result["complete"] is True and result["protocol_sha256"] == sha(RUN / "frozen_protocol.json")
            and result["state_before"] == result["state_after"] and result["application_changed"] is False
            and result["promoted"] is False, "Context processing/state binding differs")
    require(len(p["cases"]) == 4 and p["radii_at256"] == [2, 6] and p["restoration"] == "auto"
            and p["application_change"] is False and p["optimizer_constructed"] is False, "Context experiment scope differs")
    for name, pin in p["assets_sha256"].items():
        require(sha(ROOT / name) == pin, "Frozen context asset changed: " + name)
    expected = {(case["id"], radius) for case in p["cases"] for radius in (2, 6)} | {(p["baseline_reproduction_case"], 0)}
    require(len(result["rows"]) == 9 and {(r["id"], r["radius_at256"]) for r in result["rows"]} == expected,
            "Context request membership differs")
    exact({key: result[key] for key in ("generation_requests", "detector_forwards", "optimizer_updates", "forwards")},
          {"generation_requests": 9, "detector_forwards": 0, "optimizer_updates": 0,
           "forwards": {"completion": 9, "restoration": 9}}, "Context forward budget")
    require(type(result["seconds_excluding_loading"]) is float and 0 < result["seconds_excluding_loading"] < 180,
            "Context inference processing cap exceeded")
    files, verified = {"preview.png"}, []
    for row in result["rows"]:
        case = next(c for c in p["cases"] if c["id"] == row["id"])
        original, output, baseline, reference = (rgb((RUN if key == "output" else ROOT) / (row[key] if key == "output" else case[key]))
                                                  for key in ("input", "output", "baseline", "reference"))
        mask = binary(ROOT / case["mask"])
        radius = row["radius_at256"]
        require(type(radius) is int and radius in (0, 2, 6), "Context radius differs")
        tag = f"context{radius}_{row['id']}"
        require(row["context_mask"] == "masks/" + tag + ".png" and row["output"] == "outputs/" + tag + ".png",
                "Context artifact name differs")
        context = binary(RUN / row["context_mask"])
        wanted = cv2.dilate(mask.astype(np.uint8), np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)) != 0
        require(np.array_equal(context, wanted) and np.all(context[mask]) and float(context.mean()) < .85,
                "Suppression support differs from declared fixed radius")
        stages = {}
        require(set(row["stage_arrays"]) == {"completion", "restoration"}, "Actual stage membership differs")
        for kind, name in row["stage_arrays"].items():
            require(name == "stages/" + tag + "_" + kind + ".npy", "Stage filename differs")
            stage = np.load(RUN / name, allow_pickle=False)
            require(stage.shape == (1, 3, 256, 256) and stage.dtype == np.float32 and np.isfinite(stage).all(),
                    "Invalid completion context stage")
            stages[kind] = stage
            files.add(name)
        observed = original.transpose(2, 0, 1)[None].astype(np.float32) / 255
        require(np.array_equal(stages["completion"][~np.broadcast_to(context[None, None], observed.shape)],
                               observed[~np.broadcast_to(context[None, None], observed.shape)]),
                "Wrapped backend changed pixels outside conditioning area")
        completed = np.where(mask[None, None], stages["completion"], observed)
        require(np.array_equal(completed[~np.broadcast_to(mask[None, None], observed.shape)],
                               observed[~np.broadcast_to(mask[None, None], observed.shape)]),
                "Context suppression leaked into output support")
        blended = np.where(mask[None, None], completed, .5 * stages["restoration"] + .5 * completed)
        require(np.array_equal(blended[np.broadcast_to(mask[None, None], observed.shape)],
                               completed[np.broadcast_to(mask[None, None], observed.shape)]),
                "Visible restoration changed generated pixels")
        quantized = (np.clip(blended[0].transpose(1, 2, 0), 0, 1) * 255).round().astype(np.uint8)
        support = cv2.erode((~mask).astype(np.uint8), np.ones((9, 9), np.uint8)) != 0
        blurred = cv2.GaussianBlur(original.astype(np.float32), (5, 5), 1)
        measured = float((blurred.max(-1) - blurred.min(-1))[support].mean()) if int(support.sum()) >= 512 else None
        gray = measured is not None and measured <= 4
        if gray:
            quantized = np.repeat(cv2.cvtColor(quantized, cv2.COLOR_RGB2GRAY)[..., None], 3, -1)
        require(np.array_equal(quantized, output), "Context floats do not reproduce final estimated-face PNG")
        metadata = row["metadata"]
        exact({key: metadata[key] for key in ("additional_mask_expansion", "context_suppression_radius_at256",
              "conditioning_support_changed", "final_removal_support_changed", "optimizer_updates", "restoration_applied")},
              {"additional_mask_expansion": 0, "context_suppression_radius_at256": radius,
               "conditioning_support_changed": radius > 0, "final_removal_support_changed": False,
               "optimizer_updates": 0, "restoration_applied": True}, "Fixed reviewed output support")
        exact(metadata["color_policy"]["grayscale_input"], gray, "Input-only grayscale routing")
        require(metadata["mask_source"] == p["mask_source"] and metadata["restoration_requested"] == "auto"
                and metadata["restoration_fidelity"] == 1.0 and metadata["visible_restoration_blend"] == .5,
                "Context inference policy differs")
        for key, array in (("original_rgb_sha256", original), ("removal_mask_sha256", mask.astype(np.uint8)), ("output_rgb_sha256", output)):
            exact(metadata[key], hashlib.sha256(array.tobytes()).hexdigest(), "Input/mask/output decoded pixels")
        for key, array in (("visible_mae", output), ("baseline_visible_mae", baseline)):
            exact(row[key], float((np.abs(array.astype(float) - reference.astype(float))[~mask] / 255).mean()), "Visible-only diagnostic")
        if radius == 0:
            require(np.array_equal(output, baseline), "Zero-radius cached baseline differs")
        require(row["hidden_ground_truth"] is None and case["hidden_ground_truth"] is None, "Hidden-face target must remain unknown")
        files.update((row["output"], row["context_mask"]))
        verified.append({"id": row["id"], "radius_at256": radius, "context_pixels": int(context.sum()),
                         "reviewed_output_pixels": int(mask.sum()), "png_reconstructed_exactly": True,
                         "final_removal_support_changed": False, "hidden_ground_truth": None})
    require(set(result["artifacts_sha256"]) == files and len(files) == 37, "Context artifact membership differs")
    for name, pin in result["artifacts_sha256"].items():
        require(sha(RUN / name) == pin, "Context artifact changed")
    write(destination, {"format": "dgp-completion-context-independent-verification-v1", "date": "2026-10-03", "complete": True,
          "protocol_sha256": sha(RUN / "frozen_protocol.json"), "results_sha256": sha(RUN / "results.json"),
          "auditor_sha256": sha(__file__), "files_verified": len(files), "rows": verified, "model_forwards_in_audit": 0,
          "optimizer_updates": 0, "application_changed": False, "promoted": False, "visual_review_pending": True,
          "limitations": ["Only four exposed degraded assisted cases, not automatic detector or unseen validation evidence",
                          "Visual removal/plausibility review is required; hidden facial anatomy is unknown",
                          "Fidelity of reported model-state/forward counters relies on frozen runner and captured outputs"]})
    print({"complete": True, "pngs_reconstructed": 9, "verification_sha256": sha(destination)})


if __name__ == "__main__":
    main()
