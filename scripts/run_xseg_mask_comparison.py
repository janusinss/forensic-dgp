"""Run the frozen 36-case segmentation-only comparison; no generator or optimizer."""
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from face_workflow import FaceWorkflow, visibility_check
from scripts.run_practical_lama_comparison import state_sha
from xseg_occlusion import XSegVisibleFace, covering_proposal, facial_domain

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/xseg_mask_comparison_v1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pixels(path, mode="RGB"):
    with Image.open(path) as image:
        return np.asarray(image.convert(mode)).copy()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def mask_metrics(mask, reviewed, protected):
    predicted, truth = mask.astype(bool), reviewed.astype(bool)
    hit = int((predicted & truth).sum())
    union = int((predicted | truth).sum())
    tolerance = cv2.dilate(truth.astype(np.uint8), np.ones((7, 7), np.uint8)) != 0
    return {"predicted_pixels": int(predicted.sum()), "reference_pixels": int(truth.sum()),
            "intersection_pixels": hit, "union_pixels": union,
            "recall": hit / int(truth.sum()) if truth.any() else None,
            "precision": hit / int(predicted.sum()) if predicted.any() else None,
            "iou": hit / union if truth.any() and union else None,
            "excess_pixels_outside_3px_tolerance": int((predicted & ~tolerance).sum()),
            "protected_wire_pixels_changed": int((predicted & protected).sum()),
            "empty_control_marked_fraction": float(predicted.mean()) if not truth.any() else None,
            "guard": visibility_check(mask)}


def preview(cases, rows, first, index):
    selected = cases[first:first+6]
    canvas = Image.new("RGB", (1280, 32 + 281 * len(selected)), "#16181c")
    draw = ImageDraw.Draw(canvas)
    for col, label in enumerate(("input", "retained proposal", "XSeg visible probability", "XSeg covering proposal", "reviewed reference")):
        draw.text((256*col+3, 8), label, fill="white")
    def marked(original, mask):
        color = original.astype(float).copy()
        color[mask != 0] = .55*color[mask != 0]+.45*np.array([16,185,129])
        return Image.fromarray(color.round().astype(np.uint8))
    for i, case in enumerate(selected):
        row = rows[first+i]
        original = pixels(ROOT / case["input"])
        parent = pixels(OUT / row["baseline_mask"], "L") != 0
        candidate = pixels(OUT / row["xseg_mask"], "L") != 0
        reviewed = pixels(ROOT / case["reviewed"], "L") != 0
        probability = np.load(OUT / row["visible_probability"], allow_pickle=False)
        gray = Image.fromarray((probability*255).round().astype(np.uint8)).convert("RGB")
        yy = 32+i*281
        draw.text((3, yy), case["id"], fill="white")
        values = (Image.fromarray(original), marked(original,parent), gray, marked(original,candidate), marked(original,reviewed))
        for col, value in enumerate(values):
            canvas.paste(value, (256*col, yy+19))
    path = OUT / "preview" / ("rows_%02d_%02d.png" % (first+1, first+len(selected)))
    canvas.save(path)
    return path.relative_to(OUT).as_posix(), sha(path)


def main():
    if not (OUT / "frozen_protocol.json").exists() or (OUT / "execution.json").exists():
        raise ValueError("Need a fresh frozen protocol; preserve existing runs")
    p = json.loads((OUT / "frozen_protocol.json").read_text())
    if not p["frozen_before_first_forward"] or len(p["cases"]) != 36:
        raise ValueError("Protocol differs")
    for path, digest in p["assets_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError("Frozen asset changed: " + path)
    for name in ("probabilities", "raw_masks", "xseg_masks", "baseline_masks", "preview"):
        (OUT / name).mkdir()
    segmenter = XSegVisibleFace(ROOT / "outputs/xseg_pretrained_v1/xseg_1.onnx")
    baseline = FaceWorkflow(device="cpu")
    baseline._runtime()
    model = baseline._detector()
    before = state_sha(model)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Baseline must be frozen")
    count = [0]
    def count_forward(*args):
        count[0] += 1
    hook = model.segmenter.register_forward_hook(count_forward)
    write_json(OUT / "execution.json", {"protocol_sha256": sha(OUT / "frozen_protocol.json"),
               "budget": p["budget"], "state_before": before, "cpu_threads":4,
               "optimizer_constructed": False, "generator_loaded":False, "restorer_loaded":False,
               "onnx_provider":segmenter.session.get_providers()})
    started, rows, failures, hashes = time.monotonic(), [], [], {}
    for case in p["cases"]:
        if time.monotonic()-started > 300:
            failures.append({"id":case["id"], "reason":"Wall-time budget exhausted"})
            break
        try:
            original = pixels(ROOT / case["input"])
            truth = pixels(ROOT / case["reviewed"], "L") != 0
            protected = pixels(ROOT / case["protected"], "L") != 0 if case["protected"] else np.zeros((256,256),bool)
            probability = segmenter(original)
            raw, proposal = covering_proposal(probability)
            parent = (pixels(ROOT / case["baseline_cached"], "L") != 0).astype(np.uint8) if case["baseline_cached"] else baseline.review_mask(original)["mask"]
            paths = {"visible_probability":"probabilities/"+case["id"]+".npy",
                     "raw_mask":"raw_masks/"+case["id"]+".png", "xseg_mask":"xseg_masks/"+case["id"]+".png",
                     "baseline_mask":"baseline_masks/"+case["id"]+".png"}
            np.save(OUT / paths["visible_probability"], probability, allow_pickle=False)
            for key, array in (("raw_mask",raw), ("xseg_mask",proposal), ("baseline_mask",parent)):
                Image.fromarray(array.astype(np.uint8)*255).save(OUT / paths[key])
            for path in paths.values():
                hashes[path] = sha(OUT / path)
            rows.append({"id":case["id"], "family":case["family"], "synthetically_degraded":case["synthetically_degraded"],
                         "pose_scope":case["pose_scope"], "expected_rejection":case["expected_rejection"], **paths,
                         "visible_probability_min":float(probability.min()), "visible_probability_max":float(probability.max()),
                         "visible_probability_mean_in_face_domain":float(probability[facial_domain()].mean()),
                         "baseline":mask_metrics(parent,truth,protected), "xseg":mask_metrics(proposal,truth,protected)})
        except Exception as exc:
            failures.append({"id":case["id"],"reason":str(exc)})
            break
    hook.remove()
    after = state_sha(model)
    if count[0] > 20 or segmenter.forward_count > 36 or baseline.generator is not None or baseline.restorer is not None or before != after:
        raise ValueError("Inference-only execution invariant failed")
    previews = {}
    if len(rows) == 36:
        for first in range(0,36,6):
            name,digest = preview(p["cases"],rows,first,first//6)
            previews[name] = digest
    aggregates = {}
    for row in rows:
        key = row["family"] + ("/degraded" if row["synthetically_degraded"] else "/native")
        group = aggregates.setdefault(key, {"cases":0, "baseline":{}, "xseg":{}})
        group["cases"] += 1
        for arm in ("baseline", "xseg"):
            for metric in ("recall","precision","iou","excess_pixels_outside_3px_tolerance","empty_control_marked_fraction"):
                value = row[arm][metric]
                if value is not None:
                    group[arm].setdefault(metric,[]).append(value)
    for group in aggregates.values():
        for arm in ("baseline","xseg"):
            group[arm] = {metric:float(np.mean(values)) for metric,values in group[arm].items()}
    report = {"format":"dgp-xseg-mask-results-v1", "complete":len(rows)==36 and not failures, "date":"2026-10-02",
              "protocol_sha256":sha(OUT / "frozen_protocol.json"), "rows":rows, "failures":failures,
              "assets_sha256":hashes, "previews_sha256":previews, "aggregates":aggregates,
              "xseg_forwards":segmenter.forward_count, "baseline_detector_forwards":count[0],
              "completion_forwards":0, "restoration_forwards":0, "optimizer_updates":0,
              "model_state_before":before, "model_state_after":after, "state_unchanged":before==after,
              "elapsed_seconds_after_loading":time.monotonic()-started, "promoted":False,
              "visual_review_pending":True, "hidden_ground_truth":None, "training_admitted":False}
    write_json(OUT / "results.json", report)
    print(json.dumps({key:report[key] for key in ("complete","xseg_forwards","baseline_detector_forwards","elapsed_seconds_after_loading","failures","promoted")}),flush=True)
    for name,value in aggregates.items():
        print(json.dumps({"group":name,**value}),flush=True)


if __name__ == "__main__":
    main()
