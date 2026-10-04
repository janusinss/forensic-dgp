"""Bounded landmark/alignment inference on six input-selected coarse frontal cases."""
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import face_alignment
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import degradation

BASE = ROOT / "outputs/cctv_native_development_v2"
OUT = ROOT / "outputs/cctv_alignment_feasibility_v2"
IDS = ("dev_16to23_01", "dev_16to23_05", "dev_24to39_01", "dev_24to39_02", "dev_24to39_06", "dev_ge40_04")
CACHE = Path.home() / ".cache/torch/hub/checkpoints"
PINS = {"2DFAN4-11f355bf06.pth.tar": "11f355bf0693120222f5955ce3f9dc8fb5763ebb30a47d7906e509490d32e4aa",
        "s3fd-619a316812.pth": "619a31681264d3f7f7fc7a16a42cbbe8b23f31a256f75a366e5a1bcd59b33543"}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, data):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False)+"\n")


def state_sha(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode()+str(tuple(value.shape)).encode()+str(value.dtype).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    if OUT.exists():
        raise ValueError("Preserve partial/completed alignment probe")
    subset = json.loads((BASE / "frozen_subset.json").read_text(encoding="utf-8"))
    review = json.loads((BASE / "input_review.json").read_text(encoding="utf-8"))
    eligible = {row["id"] for row in review["rows"] if row["pose_review"] == "frontal_or_mild_approximate" and row["input_structure_review"] == "coarse"}
    if eligible != set(IDS) or sha(BASE / "frozen_subset.json") != "c063985b258b808efaa897c88593cbdbcee2848d686d3d056219f8a8e555a98e" or sha(BASE / "input_review.json") != "41912033f5070547e120a6b384024fa929eeaf3a7ecca408c0c67c53d37d36e2":
        raise ValueError("Require exactly six input-only eligible cases, not selection by restored outputs")
    if sha(ROOT / "degradation.py") != "973628b8d27241c8740e5224de2d1ab700e992dd90cd0e67fc80bb7187aa2129":
        raise ValueError("Legacy alignment geometry changed")
    for name, pin in PINS.items():
        if sha(CACHE / name) != pin:
            raise ValueError("Require existing fingerprinted aligner cache; no new model acquisition")
    cases = [row for row in subset["cases"] if row["id"] in eligible]
    for case in cases:
        if sha(BASE / case["source_file"]) != case["source_sha256"]:
            raise ValueError("Native source changed")
    package = Path(face_alignment.__file__).parent
    package_files = list(package.glob("*.py")) + list((package / "detection/sfd").glob("*.py"))
    OUT.mkdir()
    protocol = {"date": "2026-10-03", "runner_sha256": sha(Path(__file__)), "cases": list(IDS),
        "selection": "All six coarse frontal/mild cases in the pre-output input review; no cases chosen for model success",
        "subset_sha256": sha(BASE / "frozen_subset.json"), "input_review_sha256": sha(BASE / "input_review.json"),
        "prior_incomplete_probe_protocol_sha256": sha(ROOT / "outputs/cctv_alignment_feasibility_v1/frozen_protocol.json"),
        "reporting_fix": "Preserve legacy-caught detection exceptions as explicit fallback findings, rather than aborting before saving them; budget and frozen-state checks remain fatal",
        "focused_error_diagnostic_sha256": sha(ROOT / "outputs/cctv_alignment_failure_diagnostic_v1.json"),
        "legacy_geometry_sha256": sha(ROOT / "degradation.py"), "face_alignment_version": face_alignment.__version__,
        "package_py_sha256": {str(path.relative_to(package)): sha(path) for path in sorted(package_files)},
        "cached_weights_sha256": PINS, "spaces": ["native_rgb", "common_padded256_rgb"],
        "geometry": "Unchanged align_canonical_face: 5-point estimate, native-space eye distance>=12, angle<=45, mouth below eyes; otherwise center-crop fallback",
        "runtime": "CPU, four threads, eager compile=False; cached weights only; no warmup/generator/optimizer",
        "budget": {"alignment_requests": 12, "landmark_api_calls_max": 24, "wall_seconds_after_loading": 90,
                   "restoration_forwards": 0, "optimizer_updates": 0},
        "limits": "Predicted landmarks are not ground-truth structure. A successful transform does not prove improved restoration, identity preservation or a good default. No reserved input or application change."}
    write(OUT / "frozen_protocol.json", protocol)
    torch.set_num_threads(4)
    torch.manual_seed(20261003)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device="cpu", compile=False)
    fa.face_alignment_net.eval().requires_grad_(False)
    detector = fa.face_detector.face_detector
    detector.eval().requires_grad_(False)
    models = {"fan": fa.face_alignment_net, "s3fd": detector}
    before = {name: state_sha(model) for name, model in models.items()}
    counts = {"fan": 0, "s3fd": 0}
    handles = []
    for name, model in models.items():
        def count(module, args, name=name):
            counts[name] += 1
        handles.append(model.register_forward_pre_hook(count))
    calls, current, all_calls = [], {}, []
    original = fa.get_landmarks

    def captured(rgb):
        if len(all_calls) >= 24:
            raise ValueError("Landmark API budget exceeded")
        record = {"case": current["id"], "space": current["space"], "shape": list(rgb.shape)}
        all_calls.append(record)
        calls.append(record)
        try:
            with torch.inference_mode():
                points = original(rgb)
            record["faces"] = 0 if points is None else len(points)
            record["landmarks"] = [] if points is None else [np.asarray(p).tolist() for p in points]
            return points
        except Exception as error:
            record["error"] = type(error).__name__+": "+str(error)
            raise

    fa.get_landmarks = captured
    degradation._GLOBAL_FA = fa
    (OUT / "images").mkdir()
    rows, artifacts = [], {}
    start = time.monotonic()
    for case in cases:
        with Image.open(BASE / case["source_file"]) as native:
            rgb = np.asarray(native.convert("RGB")).copy()
        common_path = ROOT / f"outputs/cctv_native_comparison_v1/images/{case['id']}_input.png"
        with Image.open(common_path) as image:
            common = np.asarray(image.convert("RGB")).copy()
        for space, source in (("native_rgb", rgb), ("common_padded256_rgb", common)):
            if time.monotonic()-start > 90:
                raise TimeoutError("Finite alignment feasibility budget exceeded")
            calls.clear()
            current.update(id=case["id"], space=space)
            with torch.inference_mode():
                aligned, matrix, method = degradation.align_canonical_face(cv2.cvtColor(source, cv2.COLOR_RGB2BGR))
            final = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            image_name = f"images/{case['id']}_{space}.png"
            Image.fromarray(final).save(OUT / image_name)
            artifacts[image_name] = sha(OUT / image_name)
            row = {"id": case["id"], "space": space, "method": method, "source_shape": list(source.shape),
                   "transform": matrix.tolist(), "image": image_name, "landmark_api_calls": len(calls),
                   "errors_caught_by_legacy_helper": [c["error"] for c in calls if "error" in c]}
            if method == "CANONICAL_5POINT":
                pts = np.asarray(calls[-1]["landmarks"][0], dtype=np.float32)
                left, right = pts[36:42].mean(0), pts[42:48].mean(0)
                row["predicted_eye_distance_in_call_pixels"] = float(np.linalg.norm(right-left))
                row["predicted_five_points"] = np.array([left, right, pts[30], pts[48], pts[54]]).tolist()
                replay = cv2.warpAffine(cv2.cvtColor(source, cv2.COLOR_RGB2BGR), np.asarray(row["transform"]), (256, 256), borderMode=cv2.BORDER_REFLECT)
            else:
                height, width = source.shape[:2]
                side = min(height, width)
                top, left = (height-side)//2, (width-side)//2
                replay = cv2.resize(cv2.cvtColor(source[top:top+side, left:left+side], cv2.COLOR_RGB2BGR), (256, 256), interpolation=cv2.INTER_CUBIC)
            if not np.array_equal(final, cv2.cvtColor(replay, cv2.COLOR_BGR2RGB)):
                raise ValueError("Saved alignment not reproduced by recorded geometry")
            row["transform_pixels_replayed_exactly"] = True
            rows.append(row)
            print(f"{case['id']} {space}: {method}; calls={len(calls)}", flush=True)
    seconds = time.monotonic()-start
    after = {name: state_sha(model) for name, model in models.items()}
    if seconds > 90 or before != after:
        raise ValueError("Alignment budget/state/error check failed; preserve partial evidence")
    for handle in handles:
        handle.remove()
    sheet = Image.new("RGB", (3*164, 6*190+26), (238, 238, 238))
    draw = ImageDraw.Draw(sheet)
    for j, label in enumerate(("common_input", "native_alignment", "upscaled_alignment")):
        draw.text((j*164+2, 3), label, fill=(0, 0, 0))
        for i, case in enumerate(cases):
            y = 26+i*190
            draw.text((j*164+2, y+1), case["id"], fill=(0, 0, 0))
            if j == 0:
                path = ROOT / f"outputs/cctv_native_comparison_v1/images/{case['id']}_input.png"
                label = "native padded256"
            else:
                row = next(r for r in rows if r["id"] == case["id"] and r["space"] == ("native_rgb" if j == 1 else "common_padded256_rgb"))
                path, label = OUT / row["image"], row["method"]
            draw.text((j*164+2, y+14), label, fill=(0, 0, 0))
            with Image.open(path) as image:
                sheet.paste(image.resize((160, 160), Image.Resampling.BILINEAR), (j*164, y+28))
    sheet.save(OUT / "alignment_inputs.png")
    artifacts["alignment_inputs.png"] = sha(OUT / "alignment_inputs.png")
    report = {"complete": True, "protocol_sha256": sha(OUT / "frozen_protocol.json"), "rows": rows,
              "landmark_api_calls": all_calls, "model_forwards": counts, "alignment_requests": 12,
              "forward_count_scope": "Pre-hooks count attempted forwards, including detector failures",
              "legacy_caught_errors": [{"id": row["id"], "space": row["space"], "errors": row["errors_caught_by_legacy_helper"]} for row in rows if row["errors_caught_by_legacy_helper"]],
              "state_before": before, "state_after": after, "model_states_unchanged": True,
              "artifacts_sha256": artifacts, "seconds_after_loading": seconds,
              "optimizer_updates": 0, "restoration_forwards": 0, "reserved_inputs_used": False,
              "application_change": False, "canonical_counts": {space: sum(row["space"] == space and row["method"] == "CANONICAL_5POINT" for row in rows) for space in ("native_rgb", "common_padded256_rgb")}}
    write(OUT / "results.json", report)
    print(json.dumps({"complete": True, "canonical_counts": report["canonical_counts"], "model_forwards": counts,
                      "seconds": seconds, "results_sha256": sha(OUT / "results.json")}))


if __name__ == "__main__":
    main()
