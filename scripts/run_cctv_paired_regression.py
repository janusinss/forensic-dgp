"""Freeze and run a small inference-only paired camera-stress regression."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import DGPSynthesizer
from pretrained_face_restoration import load_face_restorer
from cctv_camera_stress import PROFILES, reference_canvas, degrade
from run_cctv_native_comparison import sha, read, write, state_sha, DGP, DGP_PIN, CF, CF_PIN

OUT = ROOT / "outputs/cctv_paired_regression_v1"
SPLIT = "outputs/downloaded_phase4/outputs/phase4_with_progress/split.json"
SPLIT_PIN = "5c71bc358a351d50e3c0a7d76abe749cb4412aca80d2c7eb4de5fb33dbd29071"
RANK_SEED = "cctv-paired-regression-v1-2026-10-03"
SOURCES = ("dataset/thumbnails128x128", "dataset/asian_faces")
INSIGHT = Path.home()/".insightface/models/buffalo_l"


def normalize(path):
    return str(path).replace("\\", "/")


def selection(split):
    training = [normalize(p) for p in split["train"]]
    validation = [normalize(p) for p in split["validation"]]
    if len(training) != 76000 or len(validation) != 4000 or len(set(training)) != len(training) or len(set(validation)) != len(validation) or set(training)&set(validation):
        raise ValueError("Historical split integrity differs")
    chosen = []
    for source in SOURCES:
        cohort = [p for p in validation if p.startswith(source+"/")]
        expected = 3500 if source == SOURCES[0] else 500
        if len(cohort) != expected:
            raise ValueError("Historical validation source counts differ")
        ranking = sorted(cohort, key=lambda p: hashlib.sha256((RANK_SEED+"\0"+p).encode()).hexdigest())
        chosen.extend((source, path) for path in ranking[:4])
    return chosen


def image_sheet(rows, columns, path, cell=144):
    sheet = Image.new("RGB", (len(columns)*(cell+4), len(rows)*(cell+28)+24), (238, 238, 238))
    draw = ImageDraw.Draw(sheet)
    for j, col in enumerate(columns):
        draw.text((j*(cell+4)+2, 3), col, fill="black")
        for i, row in enumerate(rows):
            y = 24+i*(cell+28)
            draw.text((j*(cell+4)+2, y+2), row["id"], fill="black")
            with Image.open(OUT/row["images"][col]) as img:
                sheet.paste(img.resize((cell, cell), Image.Resampling.BILINEAR), (j*(cell+4), y+24))
    sheet.save(path)


def prepare():
    if OUT.exists():
        raise ValueError("Preserve completed/partial regression; create a new version")
    if sha(ROOT/SPLIT) != SPLIT_PIN:
        raise ValueError("Original Phase4 split differs")
    chosen = selection(read(ROOT/SPLIT))
    files = ["scripts/run_cctv_paired_regression.py", "scripts/audit_cctv_paired_regression.py",
             "scripts/cctv_camera_stress.py", "scripts/run_cctv_native_comparison.py", SPLIT, DGP, CF,
             "models/dgp_synthesizer.py", "models/fpn_mobilenet.py", "models/mobilenet_v2.py",
             "models/__init__.py", "models/morphological_loss.py", "models/losses.py",
             "pretrained_face_restoration.py", "third_party/codeformer/LICENSE"]
    files.extend(p.relative_to(ROOT).as_posix() for p in sorted((ROOT/"third_party/codeformer").glob("*.py")))
    files.extend(path for _, path in chosen)
    assets = {p: sha(ROOT/p) for p in files}
    if assets[DGP] != DGP_PIN or assets[CF] != CF_PIN:
        raise ValueError("Restoration weights differ")
    external = {name: {"path": str(INSIGHT/name), "sha256": sha(INSIGHT/name)}
                for name in ("det_10g.onnx", "w600k_r50.onnx")}
    OUT.mkdir()
    (OUT/"images").mkdir()
    refs, cases, artifacts = [], [], {}
    for index, (source, path) in enumerate(chosen):
        ref_id = f"ref_{index+1:02d}"
        with Image.open(ROOT/path) as src:
            dimensions = list(src.size)
        target, observed, bounds = reference_canvas(ROOT/path)
        target_file, mask_file = f"images/{ref_id}_target.png", f"images/{ref_id}_observed.png"
        Image.fromarray(target).save(OUT/target_file)
        Image.fromarray(observed.astype(np.uint8)*255).save(OUT/mask_file)
        artifacts.update({target_file: sha(OUT/target_file), mask_file: sha(OUT/mask_file)})
        refs.append({"id": ref_id, "source": source, "source_file": path,
                     "source_sha256": assets[path], "native_size": dimensions,
                     "target": target_file, "observed": mask_file, "bounds": bounds,
                     "reference_note": "Original processed photo, potentially imperfect; enlargement is not new captured detail"})
        for profile in PROFILES:
            case_id = ref_id+"_"+profile["id"]
            seed = int.from_bytes(hashlib.sha256((RANK_SEED+"\0"+case_id).encode()).digest()[:8], "big")
            degraded, meta = degrade(target, bounds, profile, seed)
            inp = f"images/{case_id}_input.png"
            Image.fromarray(degraded).save(OUT/inp)
            artifacts[inp] = sha(OUT/inp)
            cases.append({"id": case_id, "reference_id": ref_id, "source": source,
                          "profile": profile["id"], "seed": seed, "input": inp, "degradation": meta})
    protocol = {"format": "dgp-cctv-paired-camera-stress-v1", "date": "2026-10-03",
                "frozen_before_restoration_outputs": True, "historical_split": SPLIT,
                "historical_split_sha256": SPLIT_PIN, "selection_seed": RANK_SEED,
                "source_validation_counts": {SOURCES[0]: 3500, SOURCES[1]: 500},
                "sampled_references_per_source": 4, "references": refs, "cases": cases,
                "profiles": PROFILES, "assets_sha256": assets, "external_assets": external,
                "prepared_artifacts_sha256": artifacts,
                "input_policy": "Native RGB center-pad128 then Pillow bilinear256; degrade only observed bounding rectangle; keep original padding outside. No face alignment or display enhancement; preserve source aspect.",
                "metric_policy": "PSNR/MAE on observed pixels; SSIM map averaged on a 3-pixel eroded observed mask, data_range1, channel_axis=-1, win_size7. Perfect PSNR is null with perfect_match=true, never JSON Infinity. Scores describe processed-reference agreement, not perceptual or identity proof.",
                "identity_policy": "One SCRFD detection on each reference target only, confidence>=0.6, exactly one face, eye span>=8. Estimate InsightFace similarity112 once from those five reference points; identical float warp to reference/input/DGP/CodeFormer. No output-based detection or eligibility. Frozen CPU ONNX cosine is a development proxy, not independent identity accuracy.",
                "criteria": ["Report each source and stress profile separately; clear control separately from degraded aggregate.",
                             "Prioritize reference facial structure over sharpening; inspect all 40 cases including clear controls.",
                             "Do not select a checkpoint, app default or new training solely by an eight-reference metric average.",
                             "Retain low-information/pose/covering reference limitations and use identical identity-eligible cases for all arms.",
                             "Use source/profile regression and native CCTV evidence together to justify a finite VM-only pilot; preserve all historical failures."],
                "limitations": ["Synthetic processed-RGB proxy, not calibrated Zamboanga CCTV, RAW ISP, H264 video or temporal restoration.",
                                "FFHQ thumbnails are128px and sampled Asian originals119-211px; reference256 enlargement adds no observed detail.",
                                "Historical validation paths are reused for development; this is not an untouched final test, and pretrained overlap is unknown.",
                                "Aspect-preserving targets differ from the legacy square-stretched Phase4 validation; scores are not interchangeable.",
                                "Observed mask excludes padding, not hair/background within the crop; SSIM and recognizer cosine do not prove identity preservation."],
                "research_sources": ["https://arxiv.org/abs/2107.10833", "https://arxiv.org/abs/1811.11127", "https://github.com/NVlabs/ffhq-dataset"],
                "budget": {"references": 8, "cases": 40, "dgp_forwards": 40, "codeformer_forwards": 40,
                           "reference_detection_requests": 8, "maximum_recognition_forwards": 128,
                           "cpu_threads": 4, "reference_wall_seconds_after_loading": 60,
                           "restoration_wall_seconds_after_loading": 480, "optimizer_updates": 0},
                "codeformer": {"fidelity": 1.0, "adain": True, "input": 512, "output": 256},
                "training": False, "reserved_native_cctv_used": False, "application_change": False}
    write(OUT/"frozen_protocol.json", protocol)
    rows = []
    for ref in refs:
        row = {"id": ref["id"], "images": {"reference": ref["target"]}}
        for c in cases:
            if c["reference_id"] == ref["id"]:
                row["images"][c["profile"]] = c["input"]
        rows.append(row)
    image_sheet(rows, ("reference", *[p["id"] for p in PROFILES]), OUT/"input_review_grid.png", 144)
    print(json.dumps({"prepared": True, "protocol_sha256": sha(OUT/"frozen_protocol.json"), "cases": len(cases), "restoration_forwards": 0}))


def pixel_metrics(rgb, target, observed):
    reference = target.astype(np.float32)/255
    error = rgb-reference
    mse = float(np.square(error[observed]).astype(np.float64).mean())
    _, scoremap = structural_similarity(reference, rgb, data_range=1, channel_axis=-1,
                                        win_size=7, full=True)
    # Zero border is essential: the all-observed square must still exclude map boundary effects.
    interior = cv2.erode(observed.astype(np.uint8), np.ones((7, 7), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    return {"PSNR": float(-10*np.log10(mse)) if mse else None, "perfect_match": mse == 0,
            "MAE": float(np.abs(error[observed]).astype(np.float64).mean()),
            "SSIM": float(scoremap[interior].astype(np.float64).mean()),
            "observed_pixels": int(observed.sum()), "ssim_interior_pixels": int(interior.sum())}


def summary(rows):
    result = {}
    for source in (*SOURCES, "all_sources"):
        result[source] = {}
        for profile in [p["id"] for p in PROFILES] + ["degraded_only"]:
            group = [r for r in rows if (source == "all_sources" or r["source"] == source) and
                     ((profile == "degraded_only" and r["profile"] != "clear") or r["profile"] == profile)]
            arms = {}
            for arm in ("input", "dgp", "codeformer"):
                items = [r["metrics"][arm] for r in group]
                eligible = [m for m in items if m["ArcFace_fixed"] is not None]
                arms[arm] = {"cases": len(items), "perfect_matches": sum(m["perfect_match"] for m in items),
                             "finite_PSNR_cases": sum(m["PSNR"] is not None for m in items),
                             "PSNR": float(np.mean([m["PSNR"] for m in items if m["PSNR"] is not None])) if any(m["PSNR"] is not None for m in items) else None,
                             "SSIM": float(np.mean([m["SSIM"] for m in items])),
                             "MAE": float(np.mean([m["MAE"] for m in items])),
                             "identity_eligible_cases": len(eligible),
                             "ArcFace_fixed": float(np.mean([m["ArcFace_fixed"] for m in eligible])) if eligible else None}
            result[source][profile] = arms
    return result


def run():
    if (OUT/"execution.json").exists():
        raise ValueError("Preserve completed/partial inference; no repeated run")
    protocol = read(OUT/"frozen_protocol.json")
    review = read(OUT/"input_review.json")
    if review["reference_ids"] != [r["id"] for r in protocol["references"]] or not review["reviewed_before_restoration_outputs"]:
        raise ValueError("Input-only reference review required")
    for name, pin in protocol["assets_sha256"].items():
        if sha(ROOT/name) != pin:
            raise ValueError("Frozen asset differs: "+name)
    for name, pin in protocol["prepared_artifacts_sha256"].items():
        if sha(OUT/name) != pin:
            raise ValueError("Frozen input differs: "+name)
    for asset in protocol["external_assets"].values():
        if sha(asset["path"]) != asset["sha256"]:
            raise ValueError("Frozen detector/recognizer differs")
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    torch.manual_seed(20261003)
    dgp = DGPSynthesizer().cpu().eval().requires_grad_(False)
    dgp.load_state_dict(torch.load(ROOT/DGP, map_location="cpu", weights_only=True), strict=True)
    cf, cf_provenance = load_face_restorer(ROOT/CF, device="cpu")
    from insightface.model_zoo import get_model
    from insightface.utils.face_align import estimate_norm
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    detector = get_model(str(INSIGHT/"det_10g.onnx"), providers=["CPUExecutionProvider"], sess_options=options)
    detector.prepare(ctx_id=-1, input_size=(256, 256), det_thresh=.6)
    recognizer = ort.InferenceSession(str(INSIGHT/"w600k_r50.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    models = {"dgp": dgp, "codeformer": cf}
    before = {name: state_sha(model) for name, model in models.items()}
    counts = {"dgp": 0, "codeformer": 0}
    recognition_count = 0

    def count(name):
        def hook(module, args, result):
            counts[name] += 1
        return hook

    handles = [m.register_forward_hook(count(name)) for name, m in models.items()]
    write(OUT/"execution.json", {"protocol_sha256": sha(OUT/"frozen_protocol.json"),
          "input_review_sha256": sha(OUT/"input_review.json"), "model_state_before": before,
          "models_eval_and_frozen": all(not m.training and not any(p.requires_grad for p in m.parameters()) for m in models.values()),
          "optimizer_constructed": False, "torch_version": torch.__version__, "onnxruntime_version": ort.__version__,
          "numpy_version": np.__version__, "opencv_version": cv2.__version__, "cpu_threads": 4,
          "onnx_providers": {"detector": detector.session.get_providers(), "recognizer": recognizer.get_providers()},
          "codeformer_provenance": cf_provenance, "budget": protocol["budget"]})
    if not read(OUT/"execution.json")["models_eval_and_frozen"]:
        raise ValueError("Only frozen inference permitted")
    (OUT/"stages").mkdir()
    artifacts, geometry, references = {}, {}, {}

    def embedding(rgb, matrix):
        nonlocal recognition_count
        crop = cv2.warpAffine(rgb.astype(np.float32), matrix, (112, 112), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        inp = np.ascontiguousarray((crop*2-1).transpose(2, 0, 1)[None])
        emb = recognizer.run(None, {recognizer.get_inputs()[0].name: inp})[0][0]
        recognition_count += 1
        if not np.isfinite(emb).all() or np.linalg.norm(emb) <= 0:
            raise ValueError("Invalid frozen recognition embedding")
        return (emb/np.linalg.norm(emb)).astype(np.float32)

    start = time.monotonic()
    for ref in protocol["references"]:
        target = np.asarray(Image.open(OUT/ref["target"]).convert("RGB")).copy()
        observed = np.asarray(Image.open(OUT/ref["observed"])) > 0
        bboxes, landmarks = detector.detect(cv2.cvtColor(target, cv2.COLOR_RGB2BGR), max_num=0)
        eligible = len(bboxes) == 1 and landmarks is not None and np.isfinite(landmarks).all() and np.linalg.norm(landmarks[0, 0]-landmarks[0, 1]) >= 8
        item = {"reference_id": ref["id"], "detections": len(bboxes), "bboxes": bboxes.tolist(),
                "reference_landmarks": landmarks.tolist() if landmarks is not None else None,
                "eligible": bool(eligible), "matrix": None, "reference_embedding": None}
        matrix, ref_emb = None, None
        if eligible:
            matrix = estimate_norm(landmarks[0].astype(np.float32), image_size=112)
            ref_emb = embedding(target.astype(np.float32)/255, matrix)
            item.update(matrix=matrix.tolist(), reference_embedding=ref_emb.tolist())
        geometry[ref["id"]] = item
        references[ref["id"]] = (target, observed, matrix, ref_emb)
        if time.monotonic()-start > 60:
            raise TimeoutError("Finite reference-alignment cap exceeded")
    ref_seconds = time.monotonic()-start
    write(OUT/"reference_geometry.json", {"reference_detection_requests": 8, "rows": list(geometry.values()),
          "reference_recognition_forwards": recognition_count, "seconds_after_loading": ref_seconds,
          "no_output_detection": True, "only_reference_targets_used": True})
    rows = []
    start = time.monotonic()
    for index, case in enumerate(protocol["cases"]):
        if time.monotonic()-start > 480:
            raise TimeoutError("Finite camera-stress inference cap exceeded")
        target, observed, matrix, ref_emb = references[case["reference_id"]]
        common = np.asarray(Image.open(OUT/case["input"]).convert("RGB")).copy()
        tensor = torch.from_numpy(common).permute(2, 0, 1).float().unsqueeze(0)/255
        with torch.inference_mode():
            generated = {"dgp": dgp(tensor), "codeformer": cf(tensor, fidelity=1)}
        floats = {"input": common.astype(np.float32)/255}
        images, metrics = {"reference": next(r["target"] for r in protocol["references"] if r["id"] == case["reference_id"]), "input": case["input"]}, {}
        for arm, value in generated.items():
            if value.shape != (1, 3, 256, 256) or not torch.isfinite(value).all() or value.min() < 0 or value.max() > 1:
                raise ValueError("Invalid raw restoration stage")
            floats[arm] = value[0].permute(1, 2, 0).cpu().numpy().copy()
            stage = f"stages/{case['id']}_{arm}.npy"
            with (OUT/stage).open("xb") as stream:
                np.save(stream, floats[arm], allow_pickle=False)
            artifacts[stage] = sha(OUT/stage)
            png = f"images/{case['id']}_{arm}.png"
            Image.fromarray((floats[arm]*255).astype(np.uint8)).save(OUT/png)
            artifacts[png] = sha(OUT/png)
            images[arm] = png
        for arm, value in floats.items():
            metric = pixel_metrics(value, target, observed)
            metric["ArcFace_fixed"] = None
            metric["embedding_file"] = None
            if matrix is not None:
                emb = embedding(value, matrix)
                emb_file = f"stages/{case['id']}_{arm}_embedding.npy"
                with (OUT/emb_file).open("xb") as stream:
                    np.save(stream, emb, allow_pickle=False)
                artifacts[emb_file] = sha(OUT/emb_file)
                metric["embedding_file"] = emb_file
                metric["ArcFace_fixed"] = float(np.clip(emb@ref_emb, -1, 1))
            metrics[arm] = metric
        rows.append({**case, "images": images, "metrics": metrics})
        print(f"Paired regression {index+1}/40: {case['id']}; elapsed={time.monotonic()-start:.1f}s", flush=True)
    seconds = time.monotonic()-start
    for handle in handles:
        handle.remove()
    after = {name: state_sha(model) for name, model in models.items()}
    expected_recognition = sum(g["eligible"] for g in geometry.values())*16
    if before != after or counts != {"dgp": 40, "codeformer": 40} or seconds > 480 or recognition_count != expected_recognition or recognition_count > 128:
        raise ValueError("Inference state/forward/time guard differs")
    for profile in PROFILES:
        file = f"comparison_{profile['id']}.png"
        image_sheet([r for r in rows if r["profile"] == profile["id"]], ("reference", "input", "dgp", "codeformer"), OUT/file, 160)
        artifacts[file] = sha(OUT/file)
    image_sheet(rows[:10], ("reference", "input", "dgp", "codeformer"), OUT/"preview_10_rows.png", 160)
    artifacts["preview_10_rows.png"] = sha(OUT/"preview_10_rows.png")
    result = {"complete": True, "protocol_sha256": sha(OUT/"frozen_protocol.json"), "rows": rows,
              "summary": summary(rows), "artifacts_sha256": artifacts, "model_forwards": counts,
              "reference_detection_requests": 8, "recognition_forwards": recognition_count,
              "reference_geometry_sha256": sha(OUT/"reference_geometry.json"), "model_state_after": after,
              "model_states_unchanged": True, "reference_seconds_after_loading": ref_seconds,
              "restoration_seconds_after_loading": seconds, "optimizer_updates": 0,
              "checkpoint_selected": False, "application_change": False, "reserved_native_cctv_used": False}
    write(OUT/"results.json", result)
    print(json.dumps({"complete": True, "model_forwards": counts, "recognition_forwards": recognition_count,
                      "seconds": seconds, "results_sha256": sha(OUT/"results.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    args = parser.parse_args()
    prepare() if args.action == "prepare" else run()
