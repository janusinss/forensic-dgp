"""Finite full-gallery check of the previously promising fixed context radius.

No training, threshold fitting, new detector/DGP forwards or app changes.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cctv_dgp_pilot import state_hash
from completion_context_margin import ContextMarginCompletion, conditioning_mask
from dgp_face_restoration import sha
from dgp_face_workflow_v3 import DGPFaceWorkflow, canonical_tensor, review_input
from face_workflow import png_bytes, visibility_check

PARENT = ROOT / "outputs/dgp_app_covering_review_v3"
PRIOR = ROOT / "outputs/completion_context_review_v1"
OUT = ROOT / "outputs/completion_margin_full_v3"
CAP = 300


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def pixels(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB")).copy()


def prepare():
    if OUT.exists():
        raise ValueError("Preserve any existing full context-margin comparison")
    parent = read(PARENT/"results.json")
    proof = read(PARENT/"saved_output_audit.json")
    visual = read(PARENT/"visual_review.json")
    prior = read(PRIOR/"results.json")
    prior_visual = read(PRIOR/"visual_review.json")
    previous_plan = read(PARENT/"plan.json")
    assert parent["complete"] and proof["complete"] and visual["complete"] and prior["complete"]
    assert proof["results_sha256"] == sha(PARENT/"results.json")
    assert prior_visual["results_sha256"] == sha(PRIOR/"results.json")
    assert visual["input_review_before_inference"] and len(previous_plan["cases"]) == 36
    files = ["scripts/compare_completion_margin_full_v3.py", "completion_context_margin.py",
             "pretrained_completion.py", "completion.py", "dgp_face_workflow_v3.py", "face_workflow.py",
             "face_color_policy.py", "third_party/codeformer/codeformer_arch.py",
             "third_party/codeformer/vqgan_arch.py", "checkpoints/codeformer_inpainting.pth"]
    files.extend(str(path.relative_to(ROOT)).replace("\\", "/") for path in (
        PARENT/"plan.json", PARENT/"results.json", PARENT/"saved_output_audit.json", PARENT/"visual_review.json",
        PRIOR/"frozen_protocol.json", PRIOR/"results.json", PRIOR/"independent_verification.json", PRIOR/"visual_review.json"))
    for path, expected in previous_plan["sources_sha256"].items():
        assert sha(ROOT/path) == expected, path
    for path, expected in parent["artifacts_sha256"].items():
        assert sha(PARENT/path) == expected, path
    cases = []
    for case in previous_plan["cases"]:
        row = next(r for r in parent["rows"] if r["id"] == case["id"])
        removal = pixels(ROOT/case["reviewed"])[..., 0] != 0
        context = cv2.dilate(removal.astype(np.uint8), np.ones((13, 13), np.uint8)) != 0
        eligible = not row["assisted"]["off"]["rejected"]
        if eligible:
            assert not visibility_check(removal.astype(np.uint8))["rejected"]
            assert context.mean() < .85
        files.extend([case["input"], case["reviewed"]])
        cases.append({"id": case["id"], "family": case["family"], "condition": case["condition"],
            "input": case["input"], "removal": case["reviewed"], "input_review": case["input_review"],
            "eligible": eligible, "nonempty": bool(removal.any()), "removal_pixels": int(removal.sum()),
            "conditioning_pixels": int(context.sum()), "added_conditioning_pixels": int((context & ~removal).sum())})
        if eligible:
            for mode in ("off", "on", "auto"):
                files.extend(str((PARENT/path).relative_to(ROOT)).replace("\\", "/") for path in (
                    row["assisted"][mode]["output"], "metadata/"+case["id"]+"_"+mode+".json",
                    "stages/"+case["id"]+"_"+mode+".npz"))
    assert sum(c["eligible"] for c in cases) == 32
    assert sum(c["eligible"] and c["nonempty"] for c in cases) == 28
    OUT.mkdir()
    write(OUT/"plan.json", {"date": "2026-10-05", "frozen_before_inference": True, "cases": cases,
        "sources_sha256": {f: sha(ROOT/f) for f in dict.fromkeys(files)},
        "hypothesis": "A prior four-case diagnostic found less scarf texture at fixed context6; full-family validation is required before integration.",
        "fixed_radius_at256": 6, "output_removal_area_changed": False,
        "conditioning_rule": "Suppress a square six-pixel ring from neural context only, then discard generated ring pixels. The operator removal footprint is unchanged.",
        "zero_radius_control": "val_336_scarf_gloves_degraded", "cap_seconds": CAP,
        "outer_timeout_seconds": 350, "maximum_completion_forwards": 29,
        "maximum_dgp_forwards": 0, "maximum_detector_forwards": 0,
        "criteria": ["Reproduce current completion raw/PNG in the zero-radius control exactly",
            "Preserve original Off visible bytes, current On visible bytes and exact input-only Auto routing",
            "Compare all supported completion outputs by family for removal, coherent rough anatomy and joins",
            "Do not adopt a universal setting with material visual regressions; no hidden identity claim"],
        "training": False, "fitting": False, "app_changes": False,
        "scope": "Same exposed 36 photographic development cases. No native CCTV, reserved inputs or independent final review."})
    print(json.dumps({"prepared": True, "cases": 36, "completion_forwards": 29,
                      "plan_sha256": sha(OUT/"plan.json")}))


def run():
    if (OUT/"execution.json").exists():
        raise ValueError("Preserve original execution; no rerun")
    plan = read(OUT/"plan.json")
    for path, expected in plan["sources_sha256"].items():
        assert sha(ROOT/path) == expected, path
    engine = DGPFaceWorkflow(device="cpu")
    engine._runtime()
    started = time.monotonic()
    base = engine._generator()
    before = state_hash(base)
    parent = read(PARENT/"results.json")
    assert before == parent["state_before"]["completion"]
    assert engine.detector is None and engine.restorer is None
    assert not base.training and not any(p.requires_grad for p in base.parameters())
    write(OUT/"execution.json", {"plan_sha256": sha(OUT/"plan.json"), "state_before": before,
        "device": "cpu", "torch": str(torch.__version__), "threads": torch.get_num_threads(),
        "cap_seconds": CAP, "optimizer_constructed": False})
    for directory in ("raw", "images", "conditioning_masks"):
        (OUT/directory).mkdir()
    counts = {"completion": 0, "dgp": 0, "detector": 0}
    handle = base.register_forward_hook(lambda *args: counts.__setitem__("completion", counts["completion"]+1))
    rows = []
    try:
        # One fresh zero-radius call binds this ablation to the current route.
        case = next(c for c in plan["cases"] if c["id"] == plan["zero_radius_control"])
        image = pixels(ROOT/case["input"])
        mask = pixels(ROOT/case["removal"])[..., 0] != 0
        tensor = canonical_tensor(image, "cpu")
        m = torch.from_numpy(mask.astype(np.float32))[None, None]
        zero = ContextMarginCompletion(base, 0).eval()
        raw = zero(tensor, m)[0].permute(1, 2, 0).numpy().copy()
        with np.load(PARENT/"stages"/(case["id"]+"_off.npz"), allow_pickle=False) as stages:
            np.testing.assert_array_equal(raw, stages["completion"][0].transpose(1, 2, 0))
        np.save(OUT/"raw/zero_control.npy", raw, allow_pickle=False)
        wrapper = ContextMarginCompletion(base, 6).eval()
        for case in plan["cases"]:
            if time.monotonic()-started > CAP:
                raise TimeoutError("300-second full context comparison cap exceeded")
            row = {"id": case["id"], "family": case["family"], "condition": case["condition"],
                   "rejected": not case["eligible"], "nonempty": case["nonempty"]}
            if not case["eligible"]:
                try:
                    review_input(case["input_review"])
                except ValueError as error:
                    row["rejection_message"] = str(error)
                else:
                    raise ValueError("Expected input-only exclusion")
                rows.append(row)
                continue
            image = pixels(ROOT/case["input"])
            mask = pixels(ROOT/case["removal"])[..., 0] != 0
            tensor = canonical_tensor(image, "cpu")
            m = torch.from_numpy(mask.astype(np.float32))[None, None]
            context = conditioning_mask(m, 6)[0, 0].numpy().astype(np.uint8)
            assert int(context.sum()) == case["conditioning_pixels"]
            (OUT/"conditioning_masks"/(case["id"]+".png")).write_bytes(png_bytes(context*255))
            if case["nonempty"]:
                raw = wrapper(tensor, m)[0].permute(1, 2, 0).numpy().copy()
                assert raw.dtype == np.float32 and np.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 1
                np.save(OUT/"raw"/(case["id"]+".npy"), raw, allow_pickle=False)
                completion = np.floor(raw*np.float32(255)).astype(np.uint8)
            else:
                completion = image.copy()
            metadata = read(PARENT/"metadata"/(case["id"]+"_off.json"))
            if metadata["display_processing"]["colour_policy"]["applied"]:
                completion = np.repeat(cv2.cvtColor(completion, cv2.COLOR_RGB2GRAY)[..., None], 3, axis=-1)
            parent_row = next(r for r in parent["rows"] if r["id"] == case["id"])
            outputs = {}
            for mode in ("off", "on"):
                output = image.copy() if mode == "off" else pixels(PARENT/parent_row["assisted"][mode]["output"])
                output[mask] = completion[mask]
                outputs[mode] = output
                expected_visible = image if mode == "off" else pixels(PARENT/parent_row["assisted"][mode]["output"])
                np.testing.assert_array_equal(output[~mask], expected_visible[~mask])
            selected = "on" if parent_row["assisted"]["auto"]["restoration_applied"] else "off"
            outputs["auto"] = outputs[selected].copy()
            row.update(auto_selected=selected, raw_completion=case["id"]+".npy" if case["nonempty"] else None,
                       outside_removal_exact=True, output_removal_area_changed=False,
                       outputs={mode: "images/"+case["id"]+"_"+mode+".png" for mode in outputs})
            for mode, output in outputs.items():
                (OUT/row["outputs"][mode]).write_bytes(png_bytes(output))
            rows.append(row)
            write(OUT/("progress_%02d.json" % len(rows)), {"cases_processed": len(rows), "forwards": counts,
                  "seconds": time.monotonic()-started, "id": case["id"]})
            print(case["id"]+": fixed context6; completed cases "+str(len(rows)), flush=True)
        after = state_hash(base)
        seconds = time.monotonic()-started
        assert before == after and counts == {"completion": 29, "dgp": 0, "detector": 0}
        assert seconds <= CAP and len(rows) == 36 and engine.detector is None and engine.restorer is None
        files = [p for p in OUT.rglob("*") if p.is_file()]
        write(OUT/"results.json", {"complete": True, "seconds": seconds, "cap_seconds": CAP,
            "plan_sha256": sha(OUT/"plan.json"), "state_before": before, "state_after": after,
            "forwards": counts, "rows": rows, "reused_dgp_output_cases": 32,
            "zero_radius_control_raw_exact": True, "artifacts_sha256": {str(p.relative_to(OUT)).replace("\\", "/"): sha(p) for p in files},
            "optimizer_updates": 0, "backward_calls": 0, "native_cctv_used": False,
            "reserved_native_used": False, "hidden_metrics": None, "app_changed": False,
            "visual_review_pending": True, "family_qualification": False, "independent_final_review": False})
        print(json.dumps({"complete": True, "seconds": seconds, "forwards": counts}))
    except Exception as error:
        write(OUT/"failure.json", {"error": str(error), "type": type(error).__name__, "completed_cases": len(rows),
            "forwards": counts, "seconds": time.monotonic()-started, "training_calls": 0})
        raise
    finally:
        handle.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else run()
