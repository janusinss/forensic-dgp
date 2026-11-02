"""Prospective TRAIN-only paired CodeFormer baseline probe; preparation has no torch import."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/completion_codeformer_hq_train_probe_v1"
READINESS = ROOT / "outputs/completion_codeformer_quality_triage_v1/training_data_readiness.json"
READINESS_SHA = "87dabe0b2c81350d91f9c0a00666c44dd8b514d1ac23d83c2b571f4cef80f0d6"
ADMISSION = ROOT / "outputs/completion_codeformer_manual_v1/source_admission.json"
ADMISSION_SHA = "188e54fd998ad97e9bef84607b8e92643f555565bce5a5c64c4a7a759f2c61ab"
CHECKPOINT_SHA = "b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797"
KINDS = ("lower_face", "eye_band", "image_left_lens", "cheek_object", "empty_control")
BUDGET = {"requests": 30, "neural_forwards": 24, "empty_bypasses": 6, "case_seconds": 120,
          "worker_seconds": 600, "external_seconds": 630, "output_bytes": 160 * 1024**2}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new(path, obj):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False)


def pixels(path, mode="RGB"):
    with Image.open(path) as image:
        require(image.format == "PNG" and image.size == (256, 256), "Prepared256 PNG required")
        return np.array(image.convert(mode), copy=True)


def pin(path):
    path = Path(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path)}


def verify_pin(record):
    path = Path(record["path"])
    require(path.is_file() and path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], f"File drift: {path}")


def fixed_mask(kind):
    """Pixel coordinates fixed prospectively, independently of model outputs."""
    image = Image.new("L", (256, 256), 0)
    draw = ImageDraw.Draw(image)
    if kind == "lower_face":
        draw.polygon([(48, 140), (208, 140), (218, 211), (180, 232), (76, 232), (38, 211)], fill=255)
    elif kind == "eye_band":
        draw.rectangle((48, 78, 207, 131), fill=255)
    elif kind == "image_left_lens":
        draw.ellipse((54, 82, 116, 126), fill=255)
    elif kind == "cheek_object":
        draw.ellipse((148, 126, 207, 187), fill=255)
    else:
        require(kind == "empty_control", "Unknown prospective mask")
    return np.array(image, copy=True)


def selected_sources():
    require(sha(READINESS) == READINESS_SHA, "Readiness receipt changed")
    ready = read(READINESS)
    require(ready["complete"] and ready["independent"], "Independent readiness required")
    sources = ready["genuine_HQ_candidate_sources"]["training_sources"]
    require(len(sources) == 391 and all(row["original_role"] == row["legacy_completion_split"] == "train" for row in sources), "TRAIN membership mismatch")
    # Six lowest numeric FFHQ IDs, because every ID is zero-padded in this registry.
    chosen = sorted(sources, key=lambda row: row["reference_id"])[:6]
    require(len({row["reference_id"] for row in chosen}) == 6, "Duplicate reference")
    for row in chosen:
        require(row["source_review_decision"] == "accept_clean_restoration", "Clean source not admitted")
        require(row["source_size"] == [1024, 1024] and row["target_size"] == [256, 256], "Genuine source geometry mismatch")
        verify_pin(row["source"])
        verify_pin(row["target256"])
    return chosen


def admitted_sources():
    require(sha(ADMISSION) == ADMISSION_SHA, "Source admission changed")
    receipt = read(ADMISSION)
    require(receipt["complete"] and receipt["independent"] and receipt["safe_state_dict"], "Source admission incomplete")
    require(receipt["checkpoint"]["sha256"] == CHECKPOINT_SHA, "Wrong checkpoint admitted")
    for record in receipt["source_files"]:
        verify_pin(record)
    verify_pin(receipt["checkpoint"])
    return receipt


def prepare():
    require(not OUT.exists(), "Probe namespace exists; never overwrite/rerun")
    admission = admitted_sources()
    sources = selected_sources()
    OUT.mkdir()
    for folder in ("targets", "inputs", "masks", "input_preview"):
        (OUT / folder).mkdir()
    cases = []
    for source in sources:
        reference = source["reference_id"]
        target = pixels(source["target256"]["path"])
        target_path = OUT / "targets" / (reference + ".png")
        # Preserve the exact retained PNG bytes, not only a derived equivalent.
        with target_path.open("xb") as stream:
            stream.write(Path(source["target256"]["path"]).read_bytes())
        require(sha(target_path) == source["target256"]["sha256"], "Target copy mismatch")
        sheet = Image.new("RGB", (768, 1148), "#181a1e")
        draw = ImageDraw.Draw(sheet)
        for col, label in enumerate(("TRAIN reference256", "prepared white hole", "reviewed fixed mask")):
            draw.text((col * 256 + 4, 5), label, fill="white")
        for number, kind in enumerate(KINDS):
            mask = fixed_mask(kind)
            require(np.isin(mask, [0, 255]).all() and mask.mean() / 255 < .85, "Invalid prospective mask")
            image = target.copy()
            image[mask == 255] = 255
            identity = reference + "__" + kind
            image_path, mask_path = OUT / "inputs" / (identity + ".png"), OUT / "masks" / (identity + ".png")
            for path, array in ((image_path, image), (mask_path, mask)):
                with path.open("xb") as stream:
                    Image.fromarray(array).save(stream, format="PNG")
            cases.append({"id": identity, "reference_id": reference, "mask_kind": kind, "role": "TRAIN_calibration",
                          "source1024": source["source"], "target256": pin(target_path), "input256": pin(image_path),
                          "mask256": pin(mask_path), "mask_pixels": int((mask == 255).sum()),
                          "attribution": source["attribution"], "hidden_target": "Same clean TRAIN reference; procedural missing support only"})
            if number < 4:
                for col, array in enumerate((target, image, np.repeat(mask[:, :, None], 3, axis=2))):
                    sheet.paste(Image.fromarray(array), (col * 256, 28 + number * 280))
                draw.text((4, 286 + number * 280), reference + " / " + kind, fill="white")
        with (OUT / "input_preview" / (reference + ".png")).open("xb") as stream:
            sheet.save(stream, format="PNG")
    protocol = {"format": "codeformer-train-only-paired-baseline-probe-v1", "status": "frozen_before_inference",
                "frozen_UTC": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(__file__),
                "readiness": pin(READINESS), "source_admission": pin(ADMISSION), "checkpoint": admission["checkpoint"],
                "source_files": admission["source_files"], "source_selection": "First six eligible clean TRAIN references sorted by zero-padded reference_id; no output/error ranking",
                "selected_sources": sources, "cases": cases, "budgets": BUDGET, "device": "cpu", "threads": 4,
                "precision": "FP32", "seed": 240, "internal_resolution": 512, "delivered_resolution": 256,
                "model": {"backend": "codeformer-inpainting", "codebook_size": 512, "embedding_dim": 512,
                          "n_head": 8, "layers": 9, "connect_list": ["32", "64", "128"], "w": 1, "adain": False},
                "mask_geometry": {"lower_face_polygon": [[48,140],[208,140],[218,211],[180,232],[76,232],[38,211]],
                                  "eye_band_inclusive_rectangle": [48,78,207,131], "image_left_lens_inclusive_ellipse": [54,82,116,126],
                                  "cheek_object_inclusive_ellipse": [148,126,207,187], "empty_control": "All zeros"},
                "input_policy": "Retained clean256 target; replace only M=1 pixels with255; existing adapter then applies visible-normalized bilinear512 and nearest mask, white hole, RGB*2-1",
                "output_policy": "Existing adapter returns composedfloat256; np.rint(*255)uint8 then exact original prepared input copy outsideM",
                "paired_metrics": {"hole_MAE": "Mean abs(completed_float256-clean_target256/255) over masked RGB scalars; empty=None",
                                   "gradient_edge_error": "Mean abs of completed-target finite-difference errors over RGB along horizontal/vertical edges crossing M boundary; empty=None"},
                "review_criteria": ["Observe eye/nose/mouth/gaze consistency and seams at exact256 against paired TRAIN target",
                                    "Report MAE/edge errors per source/mask and mean/worst without tuning or pass threshold invention",
                                    "Every generated sheet inspected; exact known bytes and empty bypass required",
                                    "Procedural TRAIN holes do not qualify real covering removal, hidden identity, DEV, FINAL or full families"],
                "new_training": False, "optimizer_updates": 0, "reserved_FINAL_used": False,
                "unknown_pretraining_overlap": True, "subject_independence_unestablished": True,
                "clear_glasses_policy": "Any visible accessories in clean target remain true target; hole fixtures are not claims that every accessory must disappear",
                "input_previews": [pin(path) for path in sorted((OUT / "input_preview").glob("*.png"))], "goal_complete": False}
    require(len(cases) == 30 and sum(case["mask_pixels"] > 0 for case in cases) == 24, "Fixed request count mismatch")
    write_new(OUT / "protocol.json", protocol)
    print(json.dumps({"prepared": True, "protocol_sha256": sha(OUT / "protocol.json"), "script_sha256": sha(__file__),
                      "sources": [row["reference_id"] for row in sources], "requests": 30, "neural_calls": 0}, indent=2))


def validate(protocol_sha, review_path):
    require(sha(OUT / "protocol.json") == protocol_sha, "Protocol pin mismatch")
    protocol = read(OUT / "protocol.json")
    require(protocol["script_sha256"] == sha(__file__) and protocol["budgets"] == BUDGET, "Runner/budgets drift")
    admission = admitted_sources()
    require(protocol["source_files"] == admission["source_files"] and protocol["checkpoint"] == admission["checkpoint"], "Source admission binding mismatch")
    require(protocol["selected_sources"] == selected_sources() and len(protocol["cases"]) == 30, "TRAIN selection changed")
    for case in protocol["cases"]:
        require(case["role"] == "TRAIN_calibration", "Non-TRAIN request refused")
        for key in ("source1024", "target256", "input256", "mask256"):
            verify_pin(case[key])
        mask = pixels(case["mask256"]["path"], "L")
        require(np.array_equal(mask, fixed_mask(case["mask_kind"])), "Mask differs from fixed geometry")
    review = read(review_path)
    require(review.get("complete") is True and review.get("independent") is True, "Independent prospective probe review required")
    require(review.get("protocol_sha256") == protocol_sha and review.get("script_sha256") == sha(__file__), "Prospective review pin mismatch")
    require(review.get("neural_forwards_authorized") == 24, "Prospective review forward cap mismatch")
    return protocol


def state_sha(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(value.dtype).encode()); digest.update(str(value.shape).encode()); digest.update(value.tobytes())
    return digest.hexdigest()


def metrics(completed, target, removal):
    if not removal.any():
        return {"hole_MAE": None, "gradient_edge_error": None, "boundary_edges": 0}
    target = target.astype(np.float32) / np.float32(255)
    error = completed - target
    edge_x = removal[:, 1:] != removal[:, :-1]
    edge_y = removal[1:, :] != removal[:-1, :]
    values = np.concatenate((np.abs(error[:, 1:] - error[:, :-1])[edge_x].reshape(-1),
                             np.abs(error[1:, :] - error[:-1, :])[edge_y].reshape(-1)))
    return {"hole_MAE": float(np.abs(error[removal]).mean()), "gradient_edge_error": float(values.mean()),
            "boundary_edges": int(edge_x.sum() + edge_y.sum())}


def progress(obj):
    temporary = OUT / ".progress.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(obj, stream, allow_nan=False)
    os.replace(temporary, OUT / "progress.json")


def run(protocol_sha, review_path):
    started = time.monotonic()
    protocol = validate(protocol_sha, review_path)
    require(not (OUT / "execution.json").exists() and not (OUT / "images").exists(), "Probe was already started; do not rerun")
    write_new(OUT / "execution.json", {"complete": False, "started_unix": time.time(), "pid": os.getpid(),
                                      "protocol_sha256": protocol_sha, "review_sha256": sha(review_path), "neural_calls": 0})
    # Model/runtime imports occur only after exact protocol and independent source/run review.
    sys.path.insert(0, str(ROOT))
    import torch
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.set_grad_enabled(False); torch.manual_seed(240)
    adapter = importlib.import_module("pretrained_completion")
    require(Path(adapter.__file__).resolve() == ROOT / "pretrained_completion.py", "Noncanonical adapter")
    model, provenance = adapter.load_codeformer(protocol["checkpoint"]["path"], device="cpu")
    for record in protocol["source_files"]:
        verify_pin(record)
    for name, relative in {"completion":"completion.py", "third_party":"third_party/__init__.py", "third_party.codeformer":"third_party/codeformer/__init__.py", "third_party.codeformer.codeformer_arch":"third_party/codeformer/codeformer_arch.py", "third_party.codeformer.vqgan_arch":"third_party/codeformer/vqgan_arch.py", "third_party.codeformer.registry":"third_party/codeformer/registry.py"}.items():
        require(name in sys.modules and Path(sys.modules[name].__file__).resolve() == ROOT / relative, "Noncanonical dependency: " + name)
    model = model.cpu().float().eval().requires_grad_(False)
    require(not model.training and not any(value.requires_grad for value in model.parameters()), "Model is not frozen")
    before = state_sha(model)
    for name in ("images", "stages", "preview"):
        (OUT / name).mkdir()
    rows, calls, active = [], 0, {"phase":"loaded", "forward_started":0, "forward_finished":0, "finished_requests":0}
    capture = []

    def save_raw(_module, _inputs, outputs):
        nonlocal calls
        calls += 1
        require(calls <= 24 and not capture, "Maximum24 actual neural forwards; one capture per case")
        raw = outputs[0]
        require(raw.shape == (1, 3, 512, 512) and raw.dtype == torch.float32 and bool(torch.isfinite(raw).all()), "Invalid raw512")
        capture.append(raw.detach()[0].permute(1, 2, 0).cpu().numpy().copy())

    hook = model.net.register_forward_hook(save_raw)
    try:
        for number, case in enumerate(protocol["cases"], 1):
            require(number <= 30 and time.monotonic() - started < 600, "Finite request/time cap")
            began = time.monotonic()
            active.update(phase="case", active_case=case["id"], active_started_unix=time.time())
            progress(active)
            image = pixels(case["input256"]["path"]); target = pixels(case["target256"]["path"])
            mask8 = pixels(case["mask256"]["path"], "L"); removal = mask8 == 255
            if removal.any():
                active.update(phase="forward", forward_started=calls + 1); progress(active)
                with torch.inference_mode():
                    x = torch.from_numpy(image.copy()).permute(2,0,1)[None].float() / 255
                    m = torch.from_numpy(removal.astype(np.float32))[None,None]
                    result = model(x, m)
                    require(result.shape == x.shape and bool(torch.isfinite(result).all()), "Invalid composed output")
                    completed = result[0].permute(1,2,0).cpu().numpy().copy()
                require(len(capture) == 1, "Exactly one raw512 capture expected")
                raw = capture.pop()
                active.update(phase="saving", forward_finished=calls); progress(active)
            else:
                raw = None
                completed = image.astype(np.float32) / np.float32(255)
            require(time.monotonic() - began <= 120, "Case/forward120s exceeded")
            require(completed.dtype == np.float32 and completed.min() >= 0 and completed.max() <= 1, "Output range/precision")
            require(np.array_equal(completed[~removal], (image.astype(np.float32)/255)[~removal]), "Composed outside pixels changed")
            final = np.rint(completed * 255).clip(0,255).astype(np.uint8)
            final[~removal] = image[~removal]
            require(np.array_equal(final[~removal], target[~removal]), "Paired visible source copy failed")
            output = OUT / "images" / (case["id"] + ".png")
            with output.open("xb") as stream: Image.fromarray(final).save(stream, format="PNG")
            stage = OUT / "stages" / (case["id"] + ".npz")
            arrays = {"input_rgb":image,"target_rgb":target,"removal_mask":removal.astype(np.uint8),"completed256":completed,"output_rgb":final}
            if raw is not None: arrays["raw512"] = raw
            with stage.open("xb") as stream: np.savez_compressed(stream, **arrays)
            seconds = time.monotonic() - began
            require(seconds <= 120, "Total case120s exceeded")
            row = {"id":case["id"],"reference_id":case["reference_id"],"mask_kind":case["mask_kind"],"role":"TRAIN_calibration",
                   "output":output.relative_to(OUT).as_posix(),"output_sha256":sha(output),"stages":stage.relative_to(OUT).as_posix(),"stages_sha256":sha(stage),
                   "neural_calls":int(removal.any()),"empty_bypass":not bool(removal.any()),"mask_pixels":int(removal.sum()),"seconds":seconds,
                   "outside_source_bytes_exact":int((~removal).sum())*3,"outside_changed_pixels":0,"paired_metrics":metrics(completed,target,removal)}
            rows.append(row); write_new(OUT / f"progress_{number:02d}.json", row)
            active.update(phase="between_cases", active_started_unix=None, finished_requests=number); progress(active)
            require(sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file()) <= BUDGET["output_bytes"], "160MiB artifact cap")
            print(json.dumps({"finished":number,"of":30,"id":case["id"],"seconds":seconds,"neural_calls":calls}), flush=True)
        require(calls == 24 and len(rows) == 30 and sum(row["empty_bypass"] for row in rows) == 6, "Fixed counts changed")
        after = state_sha(model); require(before == after, "Model state changed")
        validate(protocol_sha, review_path)
        for source in protocol["selected_sources"]:
            sheet = Image.new("RGB", (1024,1148), "#181a1e"); draw = ImageDraw.Draw(sheet)
            for col,label in enumerate(("TRAIN reference256","prepared white hole","fixed removal mask","CodeFormer result256")):draw.text((col*256+4,5),label,fill="white")
            cases = [case for case in protocol["cases"] if case["reference_id"] == source["reference_id"] and case["mask_pixels"]]
            for slot,case in enumerate(cases):
                mask = pixels(case["mask256"]["path"],"L")
                arrays = (pixels(case["target256"]["path"]),pixels(case["input256"]["path"]),np.repeat(mask[:,:,None],3,axis=2),pixels(OUT/"images"/(case["id"]+".png")))
                for col,array in enumerate(arrays):sheet.paste(Image.fromarray(array),(col*256,28+slot*280))
                draw.text((4,286+slot*280),case["id"],fill="white")
            with (OUT/"preview"/(source["reference_id"]+".png")).open("xb") as stream:sheet.save(stream,format="PNG")
        nonempty = [row for row in rows if not row["empty_bypass"]]
        summary = {}
        for key in ("hole_MAE","gradient_edge_error"):
            summary[key] = {"mean":float(np.mean([row["paired_metrics"][key] for row in nonempty])),"worst":max(nonempty,key=lambda row:row["paired_metrics"][key])["id"],"maximum":max(row["paired_metrics"][key] for row in nonempty)}
        require(time.monotonic()-started <= 600, "Worker600s cap")
        write_new(OUT/"results.json",{"complete":True,"protocol_sha256":protocol_sha,"review_sha256":sha(review_path),"requests":30,"neural_forwards":calls,"empty_bypasses":6,
                  "rows":rows,"paired_TRAIN_summary":summary,"generator_state_before":before,"generator_state_after":after,"state_unchanged":True,"provenance":provenance,
                  "seconds":time.monotonic()-started,"optimizer_constructed":False,"optimizer_updates":0,"gradient_calls":0,"reserved_FINAL_used":False,
                  "scope":"Paired procedural TRAIN calibration only; no real hidden identity, DEV/generalization/fullfamily qualification","quality_qualified":False,"visual_review_pending":True,"goal_complete":False})
        active.update(phase="finished", complete=True); progress(active)
    except BaseException as exc:
        write_new(OUT/"failure.json",{"complete":False,"cause":repr(exc),"traceback":traceback.format_exc(),"rows":rows,"neural_forwards":calls,"seconds":time.monotonic()-started,"goal_complete":False})
        raise
    finally:
        hook.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true"); mode.add_argument("--run", action="store_true")
    parser.add_argument("--protocol-sha"); parser.add_argument("--review", type=Path, default=OUT / "independent_probe_admission.json")
    args = parser.parse_args()
    if args.prepare: prepare()
    else:
        require(args.protocol_sha is not None and args.review is not None, "Explicit protocol SHA and independent prospective review required")
        run(args.protocol_sha,args.review)


if __name__ == "__main__":
    main()
