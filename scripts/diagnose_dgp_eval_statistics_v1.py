"""Four DGP forwards on two training controls; no learning or buffer adaptation."""
import json
import sys
import time
from pathlib import Path
from types import MethodType

import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cctv_dgp_pilot import state_hash
from dgp_face_restoration import sha
from dgp_face_workflow_v3 import canonical_tensor, DGP_SHA256, DEFAULT_DGP
from dgp_frozen_inference_v2 import load_frozen_dgp_restorer

PARENT = ROOT/"outputs/cctv_dgp_broader_codes_vm_v16_r2/broader_codes_protocol_v16_r2.json"
MIXED = ROOT/"outputs/cctv_dgp_mixed_vm_v9_r2"
OUT = ROOT/"outputs/dgp_eval_statistics_diagnostic_v1"
PARENT_SHA = "4f64cf2204458f8fadcab1824fc4bc293c65803e123fdebb304f7b5bd3a502a0"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def pixels(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB")).copy()


def per_image_statistics(module, value):
    if module.training or not module.track_running_stats:
        raise RuntimeError("Require a frozen evaluation module, never training mode")
    # No running buffers are passed to the operator; nothing is written back.
    return F.instance_norm(value, None, None, module.weight, module.bias,
                           True, 0., module.eps)


def metrics(image, target, observed):
    error = (image.astype(np.float64)-target.astype(np.float64))/255.
    mse = float(np.square(error[observed]).mean())
    interior = cv2.erode(observed.astype(np.uint8), np.ones((7, 7), np.uint8)) != 0
    interior[:3] = interior[-3:] = False
    interior[:, :3] = interior[:, -3:] = False
    scores = [structural_similarity(image[..., c], target[..., c], data_range=255,
                                   full=True)[1][interior].mean() for c in range(3)]
    return {"MSE_observed": mse, "PSNR_observed": float(-10*np.log10(max(mse, 1e-12))),
            "SSIM_valid_observed_windows": float(np.mean(scores))}


def main():
    if OUT.exists():
        raise ValueError("Preserve the original diagnostic; no repeat or overwrite")
    assert sha(PARENT) == PARENT_SHA
    parent = read(PARENT)
    ids = [parent["train_preview_reference_ids"][i] for i in (0, 5)]
    refs = {r["id"]: r for r in parent["references"]}
    cases = [next(c for c in parent["training_cases"] if c["reference_id"] == rid and c["profile"] == "clear") for rid in ids]
    assert all(refs[rid]["role"] == "train" for rid in ids) and len({c["source"] for c in cases}) == 2
    files = [str(Path(__file__).relative_to(ROOT)).replace("\\", "/"), "models/dgp_synthesizer.py",
        "models/fpn_mobilenet.py", "models/mobilenet_v2.py", "dgp_face_workflow_v3.py", "dgp_face_restoration.py",
        "dgp_frozen_inference_v2.py", "cctv_dgp_frozen_norm.py", str(PARENT.relative_to(ROOT)).replace("\\", "/"),
        str(DEFAULT_DGP.relative_to(ROOT)).replace("\\", "/")]
    assert sha(DEFAULT_DGP) == DGP_SHA256
    bound_cases = []
    for case in cases:
        ref = refs[case["reference_id"]]
        for path in (case["input"], ref["target"], ref["observed"]):
            assert sha(MIXED/path) == parent["data_assets_sha256"][path], path
            files.append(str((MIXED/path).relative_to(ROOT)).replace("\\", "/"))
        bound_cases.append({**case, "role": "train", "target": ref["target"], "observed": ref["observed"]})
    OUT.mkdir()
    write(OUT/"plan.json", {"date": "2026-10-05", "frozen_before_inference": True,
        "cases": bound_cases, "sources_sha256": {p: sha(ROOT/p) for p in files},
        "hypothesis": "Stored normalization may suppress useful detail when feature statistics differ. Test only inference normalization, with identical trained parameters and no statistics writeback.",
        "arms": ["stored_frozen", "per_image_no_writeback"], "normalization_layers": 5,
        "maximum_dgp_forwards": 4, "cap_seconds": 120, "outer_timeout_seconds": 180,
        "prospective_stop": "Stop before broader or native evaluation if either clear training control increases observed MSE by more than 1e-12 or loses valid-window SSIM by more than 1e-6",
        "training": False, "parameter_updates": False, "buffer_adaptation": False,
        "threshold_fitting": False, "native_used": False, "reserved_used": False,
        "scope": "Two existing training-role clear controls; source folder is provenance, not ethnicity. Not generalization or app adoption."})
    started = time.monotonic()
    torch.set_num_threads(4)
    model, provenance = load_frozen_dgp_restorer(DEFAULT_DGP, expected_sha256=DGP_SHA256, device="cpu")
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    before = state_hash(model)
    norm_layers = [(name, layer) for name, layer in model.net.named_modules() if isinstance(layer, nn.InstanceNorm2d)]
    assert len(norm_layers) == 5
    activations, records, output_count = [], [], {"dgp": 0}
    mode = "stored_frozen"
    case_id = ""
    hooks = [model.register_forward_hook(lambda *args: output_count.__setitem__("dgp", output_count["dgp"]+1))]
    for name, layer in norm_layers:
        def capture(module, args, layer_name=name):
            value = args[0]
            actual_mean = value.mean((2, 3))[0]
            actual_var = value.var((2, 3), unbiased=False)[0]
            activations.append({"id": case_id, "arm": mode, "layer": layer_name,
                "input_channel_mean": actual_mean.detach().cpu().tolist(),
                "input_channel_variance": actual_var.detach().cpu().tolist(),
                "stored_mean": module.running_mean.detach().cpu().tolist(),
                "stored_variance": module.running_var.detach().cpu().tolist(),
                "median_input_to_stored_variance_ratio": float((actual_var/(module.running_var+module.eps)).median()),
                "mean_abs_input_to_stored_mean_gap": float((actual_mean-module.running_mean).abs().mean())})
        hooks.append(layer.register_forward_pre_hook(capture))
    try:
        for mode in ("stored_frozen", "per_image_no_writeback"):
            if mode == "per_image_no_writeback":
                for _, layer in norm_layers:
                    layer._apply_instance_norm = MethodType(per_image_statistics, layer)
            for case in bound_cases:
                assert time.monotonic()-started <= 120
                case_id = case["id"]
                image = pixels(MIXED/case["input"])
                target = pixels(MIXED/case["target"])
                observed = pixels(MIXED/case["observed"])[..., 0] != 0
                with torch.inference_mode():
                    raw = model(canonical_tensor(image, "cpu"))[0].permute(1, 2, 0).numpy().copy()
                assert raw.dtype == np.float32 and raw.shape == (256, 256, 3) and np.isfinite(raw).all()
                output = np.floor(raw*np.float32(255)).astype(np.uint8)
                output[~observed] = image[~observed]
                filename = case_id+"_"+mode
                np.save(OUT/(filename+".npy"), raw, allow_pickle=False)
                Image.fromarray(output).save(OUT/(filename+".png"))
                records.append({"id": case_id, "source": case["source"], "role": "train", "arm": mode,
                    "raw": filename+".npy", "output": filename+".png", **metrics(output, target, observed)})
        after = state_hash(model)
        seconds = time.monotonic()-started
        assert before == after and output_count == {"dgp": 4} and len(activations) == 20 and seconds <= 120
        stops = []
        for case in bound_cases:
            base = next(r for r in records if r["id"] == case["id"] and r["arm"] == "stored_frozen")
            prospective = next(r for r in records if r["id"] == case["id"] and r["arm"] == "per_image_no_writeback")
            if prospective["MSE_observed"] > base["MSE_observed"]+1e-12:
                stops.append({"id": case["id"], "gate": "clear_MSE_preservation"})
            if prospective["SSIM_valid_observed_windows"] < base["SSIM_valid_observed_windows"]-1e-6:
                stops.append({"id": case["id"], "gate": "clear_SSIM_preservation"})
        write(OUT/"activation_statistics.json", activations)
        page = Image.new("RGB", (1072, 608), "white")
        draw = ImageDraw.Draw(page)
        for col, title in enumerate(("Input", "Paired target", "Stored frozen stats", "Per-image stats / no writes")):
            draw.text((268*col+5, 5), title, fill="black")
        for index, case in enumerate(bound_cases):
            y = 24+292*index
            draw.text((5, y), case["id"]+" / training control", fill="black")
            paths = (MIXED/case["input"], MIXED/case["target"], OUT/(case["id"]+"_stored_frozen.png"),
                     OUT/(case["id"]+"_per_image_no_writeback.png"))
            for col, path in enumerate(paths):
                page.paste(Image.fromarray(pixels(path)), (268*col+5, y+24))
        page.save(OUT/"training-controls.png")
        files = [p for p in OUT.iterdir() if p.is_file()]
        write(OUT/"results.json", {"complete": True, "seconds": seconds, "cap_seconds": 120,
            "plan_sha256": sha(OUT/"plan.json"), "state_before": before, "state_after": after,
            "forwards": output_count, "records": records, "prospective_stop_failures": stops,
            "provenance": provenance, "stored_statistics_unchanged": True,
            "artifacts_sha256": {p.name: sha(p) for p in files},
            "optimizer_updates": 0, "backward_calls": 0, "native_used": False, "reserved_used": False,
            "app_changed": False, "independent_final_review": False, "visual_review_pending": True,
            "next": "Close the per-image normalization hypothesis; no broader/native inference" if stops else "No adoption; require broader fixed training-cohort preservation checks before development/native use"})
        print(json.dumps({"complete": True, "seconds": seconds, "forwards": output_count, "clear_stop_failures": len(stops)}))
    except Exception as error:
        write(OUT/"failure.json", {"error": str(error), "type": type(error).__name__, "forwards": output_count,
            "completed_records": len(records), "seconds": time.monotonic()-started, "training_calls": 0})
        raise
    finally:
        for hook in hooks:
            hook.remove()


if __name__ == "__main__":
    main()
