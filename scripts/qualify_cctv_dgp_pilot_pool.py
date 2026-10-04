"""Input-only candidate deduplication and geometric admission; never training."""
import argparse
import collections
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_cctv_native_comparison import read, write, sha
from cctv_camera_stress import reference_canvas

POOL = ROOT/"outputs/cctv_dgp_pilot_pool_v1"
OUT = ROOT/"outputs/cctv_dgp_reference_gate_v1"
POOL_PIN = "422d0646b2685664a42e3eca593acd10ac2bc2a58505381699f3fdb5b10dc79f"
DETECTOR = Path.home()/".insightface/models/buffalo_l/det_10g.onnx"
# These exceptions were seen in the preceding input-only review, before a new pilot.
EXCEPTIONS = {
    "dataset/asian_faces/asian_face_04768.jpg": "sunglasses: no observed uncovered-eye reference",
    "dataset/asian_faces/asian_face_09092.jpg": "closed eyes and covering over cheek/face",
    "dataset/asian_faces/asian_face_05352.jpg": "grainy/scanned reference with uncertain detail",
    "dataset/asian_faces/asian_face_01271.jpg": "already visibly blocky/soft reference",
    "dataset/thumbnails128x128/46922.png": "sunglasses diagnostic, not an uncovered-eye target",
    "dataset/asian_faces/asian_face_05500.jpg": "strong profile outside initial frontal/mild scope",
    "dataset/asian_faces/asian_face_08886.jpg": "strong profile outside initial frontal/mild scope",
    "dataset/asian_faces/asian_face_09381.jpg": "strong profile with facial detail obscured by foliage",
}
RULES = {"detection_threshold": .6, "exactly_one_face": True, "minimum_eye_span_256": 24,
         "maximum_abs_roll_degrees": 35, "nose_projection_between_eyes": [.18, .82],
         "mouth_below_eyes_minimum_eye_span_fraction": .2,
         "minimum_observed_fraction_in112_crop": .90}


def admit(landmarks, boxes, observed, matrix):
    reasons = []
    if len(boxes) != 1:
        return ["reference_detection_count"], {}
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (1, 5, 2) or not np.isfinite(points).all():
        return ["invalid_reference_landmarks"], {}
    eye = points[0, 1]-points[0, 0]
    span = float(np.linalg.norm(eye))
    roll = float(math.degrees(math.atan2(float(eye[1]), float(eye[0]))))
    if span < RULES["minimum_eye_span_256"]:
        reasons.append("insufficient_reference_eye_span")
    if abs(roll) > RULES["maximum_abs_roll_degrees"]:
        reasons.append("reference_roll_proxy")
    nose_t = float((points[0, 2]-points[0, 0])@eye/max(span**2, 1e-8))
    if not RULES["nose_projection_between_eyes"][0] <= nose_t <= RULES["nose_projection_between_eyes"][1]:
        reasons.append("reference_yaw_proxy")
    perpendicular = np.array([-eye[1], eye[0]], dtype=np.float32)/max(span, 1e-8)
    mouth_gap = float((points[0, 3:5].mean(0)-points[0, :2].mean(0))@perpendicular)
    if mouth_gap < RULES["mouth_below_eyes_minimum_eye_span_fraction"]*span:
        reasons.append("reference_mouth_geometry")
    if matrix is None or not np.isfinite(matrix).all():
        reasons.append("invalid_reference_matrix")
        coverage = 0.0
    else:
        crop_mask = cv2.warpAffine(observed.astype(np.uint8), matrix, (112, 112),
                                   flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        coverage = float(crop_mask.mean())
        if coverage < RULES["minimum_observed_fraction_in112_crop"]:
            reasons.append("reference_crop_padding")
    return reasons, {"eye_span_256": span, "roll_degrees": roll, "nose_projection": nose_t,
                     "mouth_gap_256": mouth_gap, "observed112_fraction": coverage}


def prepare():
    if OUT.exists():
        raise ValueError("Preserve completed/partial reference gate")
    if sha(POOL/"data_audit.json") != POOL_PIN:
        raise ValueError("Audited candidate pool differs")
    audit = read(POOL/"data_audit.json")
    if audit["cross_role_byte_duplicates"]:
        raise ValueError("Candidate train/validation byte overlap")
    seen, unique, omitted = {}, [], []
    for row in audit["candidate_references"]:
        if row["sha256"] in seen:
            omitted.append({**row, "kept_path": seen[row["sha256"]], "reason": "exact_byte_duplicate_keep_first_ranked"})
        else:
            seen[row["sha256"]] = row["source_file"]
            unique.append(row)
    if len(unique) != 1150 or len(omitted) != 2:
        raise ValueError("Expected two deduplicated training references")
    assets = {"scripts/qualify_cctv_dgp_pilot_pool.py": sha(Path(__file__)),
              "scripts/cctv_camera_stress.py": sha(ROOT/"scripts/cctv_camera_stress.py"),
              "outputs/cctv_dgp_pilot_pool_v1/data_audit.json": POOL_PIN,
              "outputs/cctv_dgp_pilot_pool_v1/input_review.json": sha(POOL/"input_review.json")}
    protocol = {"format": "dgp-cctv-reference-input-gate-v1", "date": "2026-10-03",
                "frozen_before_new_training": True, "assets_sha256": assets,
                "detector": {"path": str(DETECTOR), "sha256": sha(DETECTOR), "input": [256,256], "providers": ["CPUExecutionProvider"]},
                "reference_geometry_rules": RULES, "input_review_exceptions": EXCEPTIONS,
                "references": unique, "duplicates_omitted": omitted,
                "budget": {"reference_detection_requests": 1150, "seconds_after_loading": 300, "optimizer_updates": 0},
                "scope": "Input-admissible processed-reference cohort for a bounded restoration pilot. Geometric proxies are not true pose/quality/uncovered labels or guarantees that eyes are open. Existing manual input exceptions are separate. No generated outputs decide eligibility.",
                "quality_flags": "Report grayscale-like RGB, luma/contrast and low captured resolution without inferring sensor noise or rejecting skin tones via brightness thresholds. Imperfect processed references remain an explicit limitation.",
                "training_ready": False, "reserved_native_cctv_used": False}
    OUT.mkdir()
    write(OUT/"frozen_protocol.json", protocol)
    print(json.dumps({"prepared": True, "unique_references": len(unique), "omitted_duplicates": len(omitted), "protocol_sha256": sha(OUT/"frozen_protocol.json")}))


def run():
    if (OUT/"execution.json").exists():
        raise ValueError("Preserve completed/partial reference gate execution")
    protocol = read(OUT/"frozen_protocol.json")
    for name, pin in protocol["assets_sha256"].items():
        if sha(ROOT/name) != pin:
            raise ValueError("Frozen reference asset differs: "+name)
    if sha(DETECTOR) != protocol["detector"]["sha256"]:
        raise ValueError("Detector weight differs")
    import onnxruntime as ort
    import insightface
    from insightface.model_zoo import get_model
    from insightface.utils.face_align import estimate_norm
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
    options.log_severity_level = 3  # Known static640 metadata warns on every valid256 forward; retain shape guards.
    detector = get_model(str(DETECTOR), providers=["CPUExecutionProvider"], sess_options=options)
    detector.prepare(ctx_id=-1, input_size=(256,256), det_thresh=RULES["detection_threshold"])
    cv2.setNumThreads(4)
    write(OUT/"execution.json", {"protocol_sha256": sha(OUT/"frozen_protocol.json"),
          "onnxruntime": ort.__version__, "insightface": insightface.__version__,
          "detector_providers": detector.session.get_providers(), "optimizer_constructed": False,
          "only_reference_inputs": True, "runtime_note": "ORT severity3 suppresses known static640 shape-metadata warnings at valid256 input; detection/result geometry and weight checks remain enforced."})
    rows, start = [], time.monotonic()
    for index, ref in enumerate(protocol["references"]):
        if sha(ROOT/ref["source_file"]) != ref["sha256"]:
            raise ValueError("Source bytes changed")
        target, observed, bounds = reference_canvas(ROOT/ref["source_file"])
        boxes, landmarks = detector.detect(cv2.cvtColor(target, cv2.COLOR_RGB2BGR), max_num=0)
        matrix = estimate_norm(landmarks[0].astype(np.float32), image_size=112) if len(boxes)==1 and landmarks is not None else None
        reasons, measures = admit(landmarks, boxes, observed, matrix)
        manual = EXCEPTIONS.get(ref["source_file"])
        if manual:
            reasons.append("reviewed_input_exception")
        luminance = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)[observed].astype(np.float32)/255
        rgb = target[observed].astype(np.float32)
        gray_fraction = float((np.ptp(rgb, axis=1)<=2).mean())
        rows.append({**ref, "admitted": not reasons, "reasons": reasons,
                     "input_review_exception": manual, "bounds": bounds,
                     "target_rgb_sha256": __import__('hashlib').sha256(target.tobytes()).hexdigest(),
                     "observed_sha256": __import__('hashlib').sha256(observed.astype(np.uint8).tobytes()).hexdigest(),
                     "bboxes": boxes.tolist(), "landmarks5": landmarks.tolist() if landmarks is not None else None,
                     "matrix112": matrix.tolist() if matrix is not None else None,
                     "geometry_diagnostics": measures,
                     "quality_diagnostics": {"luma_mean": float(luminance.mean()), "luma_std": float(luminance.std()),
                                             "grayscale_like_pixel_fraction": gray_fraction,
                                             "captured_min_edge": min(ref["native_size"])}})
        if time.monotonic()-start > 300:
            write(OUT/"partial_findings.json", {"rows": rows, "complete": False, "reason": "finite input-gate cap", "optimizer_updates": 0})
            raise TimeoutError("Finite reference-gate cap exceeded; partial evidence retained")
        if (index+1)%128==0:
            print(f"Reference input gate {index+1}/1150; admitted={sum(r['admitted'] for r in rows)}; elapsed={time.monotonic()-start:.1f}s", flush=True)
    seconds = time.monotonic()-start
    groups = {}
    for ref in rows:
        key = ref["role"]+"/"+ref["source"]
        group = groups.setdefault(key, {"candidates": 0, "admitted": 0, "reasons": collections.Counter()})
        group["candidates"] += 1
        group["admitted"] += int(ref["admitted"])
        group["reasons"].update(ref["reasons"])
    artifacts = {}
    for key in groups:
        source_rows = [r for r in rows if r["role"]+"/"+r["source"]==key]
        selected = [r for r in source_rows if r["admitted"]][:8]+[r for r in source_rows if not r["admitted"]][:8]
        name = "preview_"+key.replace("/", "_")+".png"
        sheet = Image.new("RGB", (4*164, 4*198), (238,238,238))
        draw = ImageDraw.Draw(sheet)
        for i, ref in enumerate(selected):
            target, _, _ = reference_canvas(ROOT/ref["source_file"])
            x,y = (i%4)*164, (i//4)*198
            draw.text((x+2,y+2), "PASS" if ref["admitted"] else "REJECT", fill="black")
            draw.text((x+2,y+15), Path(ref["source_file"]).name[:23], fill="black")
            sheet.paste(Image.fromarray(target).resize((160,160),Image.Resampling.BILINEAR),(x,y+34))
        sheet.save(OUT/name)
        artifacts[name] = sha(OUT/name)
    if seconds > 300:
        raise ValueError("Finite reference gate elapsed cap differs")
    report = {"complete": True, "protocol_sha256": sha(OUT/"frozen_protocol.json"), "rows": rows,
              "groups": groups, "artifacts_sha256": artifacts, "seconds_after_loading": seconds,
              "reference_detection_requests": len(rows), "restoration_forwards": 0, "optimizer_updates": 0,
              "training_ready": False, "reserved_native_cctv_used": False,
              "limits": "Only input-geometric admission and declared prior manual exceptions. No all-cohort pristine/uncovered quality labels, calibrated CCTV model, same-person overlap audit or final review."}
    write(OUT/"results.json", report)
    print(json.dumps({"complete": True, "groups": groups, "seconds": seconds, "results_sha256": sha(OUT/"results.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare","run"))
    args = parser.parse_args()
    prepare() if args.action=="prepare" else run()
