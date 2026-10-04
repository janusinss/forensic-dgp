"""Independent saved-pixel/geometry checks; no alignment or generator execution."""
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from degradation import detect_and_deinterlace_cctv, detect_and_smooth_mosaic, adaptive_cctv_denoise

BASE = ROOT / "outputs/cctv_native_development_v2"
SIGNAL = ROOT / "outputs/cctv_signal_preprocessing_v1"
ALIGN = ROOT / "outputs/cctv_alignment_feasibility_v2"
DESTINATION = ROOT / "outputs/cctv_processing_evidence_v1.json"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pixels(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    require(not DESTINATION.exists(), "Preserve earlier independent processing audit")
    subset = read(BASE / "frozen_subset.json")
    development = {case["id"]: case for case in subset["cases"] if case["role"] == "development"}
    signal, alignment = read(SIGNAL / "results.json"), read(ALIGN / "results.json")
    require(signal["complete"] and alignment["complete"], "Incomplete processing diagnostic")
    for directory, result in ((SIGNAL, signal), (ALIGN, alignment)):
        policy = read(directory / "frozen_protocol.json")
        require(result["protocol_sha256"] == sha(directory / "frozen_protocol.json"), "Diagnostic protocol changed")
        runner = "scripts/audit_cctv_signal_preprocessing.py" if directory == SIGNAL else "scripts/audit_cctv_alignment_feasibility_v2.py"
        require(policy["runner_sha256"] == sha(ROOT / runner), "Frozen diagnostic runner changed")
        source = policy.get("degradation_sha256", policy.get("legacy_geometry_sha256"))
        require(source == sha(ROOT / "degradation.py"), "Filter/alignment implementation changed")
        require(policy["subset_sha256"] == sha(BASE / "frozen_subset.json"), "Native source subset changed")
        for name, pin in result["artifacts_sha256"].items():
            require(sha(directory / name) == pin, "Saved processing artifact changed: " + name)
        require(result["optimizer_updates"] == 0 and not result.get("reserved_inputs_used") and not result["application_change"], "Diagnostic scope differs")
    require(signal["model_forwards"] == 0 and alignment["restoration_forwards"] == 0, "Unexpected restoration inference")
    require(len(signal["rows"]) == 24 and {row["id"] for row in signal["rows"]} == set(development), "Signal cases missing/repeated")
    changed_cases = []
    for row in signal["rows"]:
        case = development[row["id"]]
        require(sha(BASE / case["source_file"]) == case["source_sha256"], "Native crop changed")
        source = pixels(BASE / case["source_file"])
        bgr = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
        deinterlaced, flag, ratio = detect_and_deinterlace_cctv(bgr)
        mosaic, mosaic_flag = detect_and_smooth_mosaic(deinterlaced)
        final = adaptive_cctv_denoise(mosaic)
        expected = {"source": source, "deinterlaced": cv2.cvtColor(deinterlaced, cv2.COLOR_BGR2RGB),
                    "mosaic_stage": cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB), "processed": cv2.cvtColor(final, cv2.COLOR_BGR2RGB)}
        require(flag == row["interlacing_trigger"] and ratio == row["interlacing_energy_ratio"] and mosaic_flag == row["mosaic_trigger"], "Filter routing differs")
        for stage, value in expected.items():
            require(np.array_equal(value, pixels(SIGNAL / row["stages"][stage])), "Native signal stage pixels differ")
        changed = int(np.any(expected["processed"] != source, axis=2).sum())
        require(changed == row["changed_native_pixels"], "Native pixel change count differs")
        if changed:
            changed_cases.append({"id": row["id"], "native_pixels_changed": changed})
    require(len(alignment["rows"]) == 12 and alignment["alignment_requests"] == 12, "Alignment request count differs")
    require(alignment["state_before"] == alignment["state_after"], "Alignment model states changed")
    require(alignment["seconds_after_loading"] <= 90 and signal["seconds_after_import"] <= 30, "Diagnostic budget differs")
    comparison = ROOT / "outputs/cctv_native_comparison_v1"
    comparison_result = read(comparison / "results.json")
    comparison_proof = read(comparison / "independent_verification.json")
    require(comparison_proof["verified"] and comparison_proof["results_sha256"] == sha(comparison / "results.json"), "Initial shared-input audit differs")
    for case_id in {row["id"] for row in alignment["rows"]}:
        native = next(row for row in alignment["rows"] if row["id"] == case_id and row["space"] == "native_rgb")
        upscaled = next(row for row in alignment["rows"] if row["id"] == case_id and row["space"] == "common_padded256_rgb")
        for row in (native, upscaled):
            if row["space"] == "native_rgb":
                source = pixels(BASE / development[case_id]["source_file"])
            else:
                saved = next(r for r in comparison_result["rows"] if r["id"] == case_id)["outputs"]["input"]
                require(sha(comparison / saved) == comparison_result["artifacts_sha256"][saved], "Shared256 input changed")
                source = pixels(comparison / saved)
            require(list(source.shape) == row["source_shape"], "Alignment source space differs")
            if row["method"] == "CANONICAL_5POINT":
                require(not row["errors_caught_by_legacy_helper"], "Errored alignment labeled canonical")
                call = [c for c in alignment["landmark_api_calls"] if c["case"] == case_id and c["space"] == row["space"]][-1]
                points = np.asarray(call["landmarks"][0], dtype=np.float32)
                left, right = points[36:42].mean(0), points[42:48].mean(0)
                five = np.asarray([left, right, points[30], points[48], points[54]])
                require(np.array_equal(five, np.asarray(row["predicted_five_points"], dtype=np.float32)), "Recorded five points differ")
                distance = float(np.linalg.norm(right-left))
                angle = float(np.degrees(np.arctan2(right[1]-left[1], right[0]-left[0])))
                require(distance >= 12 and abs(angle) <= 45 and (points[48, 1]+points[54, 1])/2 > (left[1]+right[1])/2, "Canonical geometry gate differs")
                require(abs(distance-row["predicted_eye_distance_in_call_pixels"]) < 1e-5, "Eye-distance value differs")
                value = cv2.warpAffine(source, np.asarray(row["transform"]), (256, 256), borderMode=cv2.BORDER_REFLECT)
            else:
                require(row["method"] == "FALLBACK_CENTER", "Unknown alignment method")
                height, width = source.shape[:2]
                side = min(height, width)
                y, x = (height-side)//2, (width-side)//2
                value = cv2.resize(source[y:y+side, x:x+side], (256, 256), interpolation=cv2.INTER_CUBIC)
                matrix = np.asarray([[256/side, 0, -x*256/side], [0, 256/side, -y*256/side]], dtype=np.float32)
                require(np.allclose(matrix, np.asarray(row["transform"]), atol=1e-6), "Fallback coordinate matrix differs")
            require(np.array_equal(value, pixels(ALIGN / row["image"])), "Saved alignment pixels differ")
    counts = {space: sum(row["space"] == space and row["method"] == "CANONICAL_5POINT" for row in alignment["rows"]) for space in ("native_rgb", "common_padded256_rgb")}
    require(counts == alignment["canonical_counts"] == {"native_rgb": 0, "common_padded256_rgb": 3}, "Alignment method counts differ")
    require(len(alignment["legacy_caught_errors"]) == 5 and all(row["space"] == "native_rgb" for row in alignment["legacy_caught_errors"]), "Legacy exception census differs")
    report = {"verified": True, "date": "2026-10-03", "auditor_sha256": sha(Path(__file__)),
              "native_subset_sha256": sha(BASE / "frozen_subset.json"), "signal_results_sha256": sha(SIGNAL / "results.json"),
              "alignment_results_sha256": sha(ALIGN / "results.json"), "initial_comparison_results_sha256": sha(comparison / "results.json"),
              "signal_pngs_exact": 96, "alignment_pngs_exact": 12, "signal_changed_cases": changed_cases,
              "canonical_counts": counts, "legacy_detector_exceptions_recorded": 5,
              "model_forwards_in_audit": 0, "optimizer_updates": 0, "reserved_inputs_rendered": False,
              "limits": "Pixel/routing/geometry checks do not establish true landmark anatomy, useful restoration or Zamboanga accuracy"}
    with DESTINATION.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"verified": True, "signal_pngs_exact": 96, "alignment_pngs_exact": 12,
                      "signal_changed_cases": changed_cases, "audit_sha256": sha(DESTINATION)}))


if __name__ == "__main__":
    main()
