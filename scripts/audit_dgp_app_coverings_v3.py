"""Finite current-route covering review; preserve all historical masks and gates."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cctv_dgp_pilot import state_hash
from dgp_face_restoration import sha, prepare_crop
from dgp_face_workflow_v3 import DGPFaceWorkflow, decode_crop, decode_mask
from face_workflow import png_bytes

OUT = ROOT / "outputs/dgp_app_covering_review_v3"
PARENT = "outputs/varied_covering_practical_protocol_v1/protocol.json"
PARENT_SHA = "15f6ca20827c4221f7e12630f7d3fab54d7eceae51235eb9401c6f529563382e"
CACHE_ROOT = ROOT / "outputs/varied_covering_practical_review_v1"
CAP = 600


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def prepare():
    if (OUT / "plan.json").exists():
        raise ValueError("Preserve frozen covering plan")
    if sha(ROOT / PARENT) != PARENT_SHA:
        raise ValueError("Changed historical covering inventory")
    previous = read(ROOT / PARENT)
    inventory = read(OUT / "input_inventory.json")
    if len(previous["cases"]) != 36 or inventory["protocol_sha256"] != PARENT_SHA:
        raise ValueError("Changed input inventory")
    cache = read(CACHE_ROOT / "results.json")
    files = [PARENT, "outputs/varied_covering_practical_review_v1/results.json",
             "dgp_face_workflow_v3.py", "face_workflow.py", "face_color_policy.py",
             "dgp_face_restoration.py", "dgp_frozen_inference_v2.py", "cctv_dgp_frozen_norm.py",
             "cctv_input_quality.py", "pretrained_completion.py", "completion_inference.py",
             "models/__init__.py", "models/dgp_synthesizer.py", "completion.py", "third_party/codeformer/codeformer_arch.py",
             "scripts/audit_dgp_app_coverings_v3.py"]
    engine = DGPFaceWorkflow(device="cpu")
    files.extend(str(p.relative_to(ROOT)).replace("\\", "/") for p in (
        engine.detector_path, engine.completion_path, engine.restoration_path))
    cases = []
    for row in previous["cases"]:
        review = "out_of_scope" if "three-quarter" in row["pose_scope"] else (
            "needs_clearer" if row["expected_rejection"] else "usable")
        note = "Readable remaining facial context, cropped frontal/mild turn" if review == "usable" else (
            "Strong three-quarter diagnostic excluded from supported pose" if review == "out_of_scope" else
            "Hands hide nearly all central facial context; hair/background alone is insufficient")
        cached_row = next(r for r in cache["rows"] if r["id"] == row["id"])
        comparisons = {}
        for name in ("existing91", "varied133"):
            path = cached_row["models"][name]["proposal"]
            if sha(CACHE_ROOT / path) != cache["artifact_sha256"][path]:
                raise ValueError("Changed historical comparison mask")
            relative = str((CACHE_ROOT / path).relative_to(ROOT)).replace("\\", "/")
            comparisons[name] = relative
            files.append(relative)
        for path in [row["input"], row["reviewed"]] + ([row["protected"]] if row["protected"] else []):
            if sha(ROOT / path) != previous["assets_sha256"][path]:
                raise ValueError("Changed historical input/operator footprint")
            files.append(path)
        cases.append({**row, "input_review": review, "input_only_note": note,
                      "condition": "synthetic_degraded_photo" if row["synthetically_degraded"] else "original_photo",
                      "cached_comparison_proposals": comparisons})
    write(OUT / "plan.json", {"date": "2026-10-05", "format": "dgp-app-covering-inference-review-v3",
          "cases": cases, "sources_sha256": {f: sha(ROOT / f) for f in dict.fromkeys(files)},
          "input_sheets_sha256": {f: sha(OUT / f) for f in ("inputs-original-photos.png", "inputs-degraded-photos.png")},
          "input_review": "All 36 input cells manually viewed before generation; operator policy, not automatic classification",
          "cap_seconds": CAP, "max_forwards": {"detector": 36, "completion": 84, "dgp": 64},
          "max_requests": 108, "device": "cpu", "training": False, "reserved_native_cctv_used": False,
          "automatic_output_policy": "Unreviewed automatic masks are diagnostic proposals only; no generation from rejected areas",
          "assisted_output_policy": "Previously frozen operator mask, no new expansion, Auto/On/Off",
          "no_hidden_ground_truth": True, "qualification": "Development evidence, not independent final review"})
    print(json.dumps({"prepared": True, "cases": len(cases), "cap_seconds": CAP}), flush=True)


def overlap(proposal, reference):
    a, b = proposal.astype(bool), reference.astype(bool)
    intersection, union = int((a & b).sum()), int((a | b).sum())
    return {"iou_against_operator_footprint": intersection / union if union else 1.,
            "missed_operator_pixels": int((b & ~a).sum()), "extra_pixels": int((a & ~b).sum()),
            "proposal_pixels": int(a.sum()), "operator_pixels": int(b.sum()),
            "meaning": "Approximate reviewed removal-area overlap, not segmentation or hidden-identity ground truth"}


def run():
    if (OUT / "execution.json").exists():
        raise ValueError("Preserve completed or failed execution")
    plan = read(OUT / "plan.json")
    for path, expected in plan["sources_sha256"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("Changed planned source: " + path)
    started = time.monotonic()
    engine = DGPFaceWorkflow(device="cpu")
    engine._runtime()
    models = {"detector": engine._detector(), "completion": engine._generator(), "dgp": engine._restorer()}
    before = {name: state_hash(model) for name, model in models.items()}
    counts = dict.fromkeys(models, 0)
    captured = {}
    def hook(name):
        def save(module, args, result):
            counts[name] += 1
            if counts[name] > plan["max_forwards"][name]:
                raise ValueError("Finite forward count exceeded")
            if name != "detector":
                captured[name] = result.detach().cpu().numpy().copy()
        return save
    handles = [models["completion"].register_forward_hook(hook("completion")),
               models["dgp"].register_forward_hook(hook("dgp"))]
    # CompletionNet.detect is an explicit method, so instrument that method separately.
    original_detect = models["detector"].detect
    def detect(image):
        counts["detector"] += 1
        if counts["detector"] > 36:
            raise ValueError("Finite detector count exceeded")
        return original_detect(image)
    models["detector"].detect = detect
    write(OUT / "execution.json", {"plan_sha256": sha(OUT / "plan.json"), "state_before": before,
          "device": "cpu", "optimizer_constructed": False, "cap_seconds": CAP})
    for folder in ("images", "stages", "metadata"):
        (OUT / folder).mkdir()
    rows, requests, failures = [], 0, []
    try:
        for case in plan["cases"]:
            if time.monotonic() - started > CAP:
                raise TimeoutError("600-second covering inference cap exceeded")
            rgb = decode_crop((ROOT / case["input"]).read_bytes())
            mask = decode_mask((ROOT / case["reviewed"]).read_bytes(), rgb.shape[:2])
            proposal = engine.review_mask(rgb)
            (OUT / "images" / (case["id"] + "_automatic.png")).write_bytes(png_bytes(proposal["mask"] * 255))
            cached = {name: overlap(decode_mask((ROOT / path).read_bytes(), rgb.shape[:2]), mask)
                      for name, path in case["cached_comparison_proposals"].items()}
            row = {"id": case["id"], "family": case["family"], "condition": case["condition"],
                   "input_review": case["input_review"], "automatic": overlap(proposal["mask"], mask),
                   "cached_unqualified_comparisons": cached, "assisted": {}, "hidden_ground_truth": None}
            outputs = {}
            for mode in ("off", "on", "auto"):
                requests += 1
                if requests > plan["max_requests"] or time.monotonic() - started > CAP:
                    raise TimeoutError("Finite request/time budget exceeded")
                captured.clear()
                try:
                    result = engine.generate(rgb, mask, mode, case["input_review"], True)
                except ValueError as exc:
                    row["assisted"][mode] = {"rejected": True, "message": str(exc)}
                    if case["input_review"] == "usable":
                        failures.append({"id": case["id"], "mode": mode, "reason": str(exc)})
                    continue
                output = result["output"]
                output_path = "images/" + case["id"] + "_" + mode + ".png"
                (OUT / output_path).write_bytes(png_bytes(output))
                write(OUT / "metadata" / (case["id"] + "_" + mode + ".json"), result["metadata"])
                np.savez_compressed(OUT / "stages" / (case["id"] + "_" + mode + ".npz"), **captured)
                outputs[mode] = output
                ignored = result["mask"].astype(bool)
                if mode == "off":
                    np.testing.assert_array_equal(output[~ignored], result["original"][~ignored])
                if mode == "on":
                    np.testing.assert_array_equal(output[ignored], outputs["off"][ignored])
                if mode == "auto":
                    selected = "on" if result["metadata"]["restoration_applied"] else "off"
                    np.testing.assert_array_equal(output, outputs[selected])
                row["assisted"][mode] = {"rejected": False, "output": output_path,
                     "restoration_applied": result["metadata"]["restoration_applied"],
                     "mask_source": result["metadata"]["mask_source"], "completion_nonempty": bool(mask.any()),
                     "visible_change_pixels": int(np.any(output != result["original"], axis=-1)[~ignored].sum()),
                     "display_processing": result["metadata"]["display_processing"]}
            rows.append(row)
            print(json.dumps({"case": case["id"], "requests": requests, "forwards": counts,
                              "seconds": round(time.monotonic() - started, 2)}), flush=True)
        after = {name: state_hash(model) for name, model in models.items()}
        if before != after:
            raise ValueError("Inference changed a model state")
        for condition in ("original_photo", "synthetic_degraded_photo"):
            chosen = [r for r in rows if r["condition"] == condition]
            for page in range(3):
                sheet = Image.new("RGB", (6 * 268, 36 + 6 * 288), "#020617")
                draw = ImageDraw.Draw(sheet)
                for i, label in enumerate(("Input", "Automatic proposal", "Reviewed footprint", "Off", "DGP On", "Auto")):
                    draw.text((i * 268 + 4, 8), label, fill="white")
                for row_index, row in enumerate(chosen[page * 6:(page + 1) * 6]):
                    case = next(c for c in plan["cases"] if c["id"] == row["id"])
                    y = 36 + row_index * 288
                    draw.text((4, y), case["id"], fill="white")
                    images = [decode_crop((ROOT / case["input"]).read_bytes()),
                              decode_crop((OUT / "images" / (case["id"] + "_automatic.png")).read_bytes()),
                              np.repeat(decode_mask((ROOT / case["reviewed"]).read_bytes(), (256, 256))[..., None] * 255, 3, axis=-1)]
                    for mode in ("off", "on", "auto"):
                        record = row["assisted"][mode]
                        if record["rejected"]:
                            images.append(None)
                        else:
                            images.append(decode_crop((OUT / record["output"]).read_bytes()))
                    for i, image in enumerate(images):
                        if image is None:
                            draw.text((i * 268 + 8, y + 110), "REJECTED: " + case["input_review"], fill="#f87171")
                        else:
                            sheet.paste(Image.fromarray(image), (i * 268 + 4, y + 24))
                sheet.save(OUT / (condition + "_" + str(page + 1) + ".png"))
        seconds = time.monotonic() - started
        if seconds > CAP:
            raise TimeoutError("600-second covering cap exceeded")
        artifacts = {str(p.relative_to(OUT)).replace("\\", "/"): sha(p)
                     for p in OUT.rglob("*") if p.is_file()}
        write(OUT / "results.json", {"complete": True, "plan_sha256": sha(OUT / "plan.json"),
              "rows": rows, "requests": requests, "forwards": counts, "seconds": seconds,
              "state_before": before, "state_after": after, "unexpected_rejections": failures,
              "artifacts_sha256": artifacts, "optimizer_updates": 0, "backward_calls": 0,
              "native_cctv_used": False, "reserved_native_used": False, "hidden_metrics": None,
              "automatic_proposals_reviewed_for_generation": False, "visual_review_pending": True,
              "family_qualification": False, "independent_final_review": False})
        print(json.dumps({"complete": True, "seconds": seconds, "forwards": counts,
                          "unexpected_rejections": len(failures)}), flush=True)
    except Exception as exc:
        write(OUT / "failure.json", {"complete": False, "reason": repr(exc), "requests": requests,
                                     "forwards": counts, "rows": rows})
        raise
    finally:
        for handle in handles:
            handle.remove()
        models["detector"].detect = original_detect


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else run()
