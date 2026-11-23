"""One reviewed-mask CodeFormer CPU request; the parent enforces a hard120s timeout.

Inputs are prepared RGB256 input.png and binary L256 mask.png (255 means remove).
Success publishes result.png, stages.npz and receipt.json. Empty masks bypass all
neural imports and checkpoint access. No restoration, detection or training runs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "outputs/completion_codeformer_finetune_v1_r1_candidate/codeformer_completion_finetune_r1.pth"
CANDIDATE_MANIFEST = ROOT / "outputs/completion_codeformer_finetune_v1_r1_candidate/candidate_manifest.json"
MAX_SECONDS = 120
MAX_INPUT_BYTES = 10 * 1024**2
POLICY = "reviewed-codeformer-finetune-r1-completion-cpu-v1"
CANONICAL_SOURCES = {"pretrained_completion_finetune_r1": "pretrained_completion_finetune_r1.py", "face_workflow": "face_workflow.py", "pretrained_completion": "pretrained_completion.py",
                     "completion": "completion.py", "third_party": "third_party/__init__.py",
                     "third_party.codeformer": "third_party/codeformer/__init__.py",
                     "third_party.codeformer.codeformer_arch": "third_party/codeformer/codeformer_arch.py",
                     "third_party.codeformer.vqgan_arch": "third_party/codeformer/vqgan_arch.py",
                     "third_party.codeformer.registry": "third_party/codeformer/registry.py"}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_sha(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def check_time(started):
    if time.monotonic() - started > MAX_SECONDS:
        raise TimeoutError("CodeFormer request exceeded120 seconds; parent must enforce hard timeout")


def load_request(request_dir):
    """Read exact prepared pixels; no resizing, thresholding or inferred mask."""
    request_dir = Path(request_dir).resolve(strict=True)
    if not request_dir.is_dir():
        raise ValueError("Request directory is missing")
    arrays, bindings = {}, {}
    for filename, mode in (("input.png", "RGB"), ("mask.png", "L")):
        path = request_dir / filename
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_INPUT_BYTES:
            raise ValueError(f"Invalid prepared request file: {filename}")
        bindings[filename] = sha(path)
        with Image.open(path) as image:
            if image.format != "PNG" or image.size != (256, 256) or image.mode != mode:
                raise ValueError(f"{filename} must be a prepared {mode}256 PNG")
            arrays[filename] = np.array(image, copy=True)
        if sha(path) != bindings[filename]:
            raise ValueError(f"Request file changed while reading: {filename}")
    rgb, mask8 = arrays["input.png"], arrays["mask.png"]
    if rgb.dtype != np.uint8 or rgb.shape != (256, 256, 3) or not np.isfinite(rgb).all():
        raise ValueError("Expected finite RGB256 uint8 input")
    if mask8.dtype != np.uint8 or mask8.shape != rgb.shape[:2] or not np.isin(mask8, (0, 255)).all():
        raise ValueError("Use a matching binary0/255 removal mask")
    removal = (mask8 == 255).astype(np.uint8)
    if float(removal.mean()) >= .85:
        raise ValueError("Too little visible face remains; use a less-covered image")
    bounds = [0, 0, 256, 256]
    request_path = request_dir / "request.json"
    if request_path.exists():
        if request_path.is_symlink() or not request_path.is_file() or request_path.stat().st_size > MAX_INPUT_BYTES:
            raise ValueError("Invalid request metadata")
        bindings["request.json"] = sha(request_path)
        metadata = json.loads(request_path.read_text(encoding="utf-8"))
        bounds = metadata.get("prepared_bounds", bounds)
        if not isinstance(bounds, list) or len(bounds) != 4 or any(type(value) is not int for value in bounds):
            raise ValueError("prepared_bounds must be four integer content coordinates")
        if sha(request_path) != bindings["request.json"]:
            raise ValueError("Request metadata changed while reading")
    left, top, right, bottom = bounds
    if not (0 <= left < right <= 256 and 0 <= top < bottom <= 256):
        raise ValueError("Invalid prepared content bounds")
    unavailable = np.ones((256, 256), dtype=np.uint8)
    unavailable[top:bottom, left:right] = 0
    if removal[unavailable != 0].any():
        raise ValueError("Removal mask must not include prepared padding")
    unavailable = np.maximum(unavailable, removal)
    visibility = visibility_check256(unavailable)
    if visibility["rejected"]:
        raise ValueError("Too much face is hidden. Use a less-covered face image")
    return request_dir, rgb, removal, bindings, bounds, unavailable, visibility


def visibility_check256(mask):
    """Existing face_workflow.visibility_check geometry, on already256 binary pixels.

    This NumPy form lets empty requests validate padding without importing torch.
    Nonempty requests additionally compare the actual canonical function result.
    """
    m = mask != 0
    yy, xx = np.mgrid[:256, :256]
    face = ((xx - 127.5) / 83) ** 2 + ((yy - 137) / 104) ** 2 <= 1
    bands = {"left_eye": (60, 84, 119, 133), "right_eye": (137, 84, 196, 133),
             "nose": (108, 119, 149, 166), "mouth": (82, 165, 174, 208)}
    fractions = {name: float(m[y1:y2, x1:x2].mean())
                 for name, (x1, y1, x2, y2) in bands.items()}
    face_covered = float(m[face].mean())
    all_features = (fractions["left_eye"] >= .85 and fractions["right_eye"] >= .85
                    and fractions["nose"] >= .9 and fractions["mouth"] >= .9)
    rejected = face_covered >= .80 or all_features or m.mean() >= .85
    return {"rejected": bool(rejected), "face_covered_fraction": face_covered,
            "feature_covered_fractions": fractions,
            "scope": "Approximate frontal-crop geometry, conditional on the reviewed removal area"}


def encode_output(rgb, removal, completed):
    """Match existing CodeFormer rounding and copy original bytes outside the mask."""
    if completed.shape != rgb.shape or completed.dtype != np.float32 or not np.isfinite(completed).all():
        raise FloatingPointError("Invalid CodeFormer completion output")
    if completed.min() < 0 or completed.max() > 1:
        raise FloatingPointError("CodeFormer completion values must lie in[0,1]")
    final = np.rint(completed * np.float32(255)).clip(0, 255).astype(np.uint8)
    final[removal == 0] = rgb[removal == 0]
    return final


def canonical_module(name):
    # Put this checkout first even when invoked with another cwd/PYTHONPATH.
    root_text = str(ROOT)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    module = importlib.import_module(name)
    if Path(module.__file__).resolve() != ROOT / (name + ".py"):
        raise RuntimeError(f"Refusing noncanonical {name} module")
    return module


def verify_loaded_sources(require_all=False):
    """Refuse a preloaded wrong checkout before import and verify every dependency after load."""
    loaded = {}
    for name, relative in CANONICAL_SOURCES.items():
        module = sys.modules.get(name)
        if module is None:
            if require_all:
                raise RuntimeError(f"Expected canonical dependency was not loaded: {name}")
            continue
        filename = getattr(module, "__file__", None)
        if filename is None or Path(filename).resolve() != (ROOT / relative).resolve():
            raise RuntimeError(f"Refusing noncanonical loaded dependency: {name}")
        loaded[name] = str(Path(filename).resolve())
    return loaded


def source_bindings():
    sources = [Path(__file__).resolve(), ROOT / "pretrained_completion_finetune_r1.py", ROOT / "pretrained_completion.py", ROOT / "completion.py", ROOT / "face_workflow.py",
               ROOT / "third_party/__init__.py"]
    sources.extend(sorted((ROOT / "third_party/codeformer").rglob("*.py")))
    return {str(path.relative_to(ROOT).as_posix()): sha(path) for path in sources}


def complete_nonempty(rgb, removal, checkpoint, started, unavailable, candidate_manifest_sha):
    """Exactly one adapter call and one pretrained net forward, after validation."""
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    torch.manual_seed(240)
    verify_loaded_sources()
    workflow = canonical_module("face_workflow")
    visibility = workflow.visibility_check(unavailable)
    if visibility != visibility_check256(unavailable):
        raise RuntimeError("Canonical visibility geometry changed; review worker before use")
    if visibility["rejected"]:
        raise ValueError("Too much face is hidden. Use a less-covered face image")
    checkpoint = Path(checkpoint)
    adapter = canonical_module("pretrained_completion_finetune_r1")
    candidate = adapter.validate_candidate_manifest(candidate_manifest_sha, checkpoint=checkpoint)
    check_time(started)
    load_started = time.monotonic()
    model, provenance = adapter.load_candidate(manifest_sha=candidate_manifest_sha, checkpoint=checkpoint, device="cpu")
    loaded_sources = verify_loaded_sources(require_all=True)
    model = model.cpu().float().eval().requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("CodeFormer model must be frozen in evaluation mode")
    load_seconds = time.monotonic() - load_started
    check_time(started)
    neural_calls = 0

    def count_forward(_module, _inputs):
        nonlocal neural_calls
        neural_calls += 1
        if neural_calls > 1:
            raise RuntimeError("One-forward request cap exceeded")

    hook = model.net.register_forward_pre_hook(count_forward)
    forward_started = time.monotonic()
    try:
        with torch.inference_mode():
            image = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float() / 255
            mask = torch.from_numpy(removal.copy())[None, None].float()
            result = model(image, mask)
            if result.shape != image.shape or result.dtype != torch.float32 or not torch.isfinite(result).all():
                raise FloatingPointError("Invalid CodeFormer adapter output")
            outside = (mask == 0).expand_as(image)
            if not torch.equal(result[outside], image[outside]):
                raise RuntimeError("CodeFormer adapter changed pixels outside the reviewed removal mask")
            completed = result[0].permute(1, 2, 0).cpu().numpy().copy()
    finally:
        hook.remove()
    forward_seconds = time.monotonic() - forward_started
    check_time(started)
    if neural_calls != 1:
        raise RuntimeError("Expected exactly one pretrained CodeFormer forward")
    if sha(checkpoint) != candidate["checkpoint"]["sha256"]:
        raise RuntimeError("Candidate checkpoint changed during request")
    adapter.validate_candidate_manifest(candidate_manifest_sha, checkpoint=checkpoint)
    return completed, provenance, visibility, {"load_seconds": load_seconds, "forward_seconds": forward_seconds,
                                               "torch": str(torch.__version__), "neural_calls": neural_calls,
                                               "checkpoint": str(checkpoint), "checkpoint_sha256": candidate["checkpoint"]["sha256"],
                                               "candidate_manifest_sha256": candidate_manifest_sha,
                                               "loaded_canonical_sources": loaded_sources}


def write_json_exclusive(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def run_request(request_dir, checkpoint=DEFAULT_CHECKPOINT, candidate_manifest_sha=None):
    started = time.monotonic()
    request_dir, rgb, removal, inputs_sha256, bounds, unavailable, visibility = load_request(request_dir)
    for name in ("result.png", "stages.npz", "receipt.json", "failure.json"):
        if (request_dir / name).exists():
            raise ValueError("Request already has outputs; use a fresh request directory")
    sources_before = source_bindings()
    try:
        if not removal.any():
            # No torch import, checkpoint existence/hash check, or model loading here.
            completed = rgb.astype(np.float32) / np.float32(255)
            final = rgb.copy()
            provenance = {"backend": "codeformer-inpainting", "bypassed": "empty reviewed removal mask",
                          "checkpoint_loaded": False, "candidate_manifest_sha256_expected": candidate_manifest_sha}
            runtime = {"load_seconds": 0.0, "forward_seconds": 0.0, "torch": None, "neural_calls": 0,
                       "checkpoint": None, "checkpoint_sha256": None}
        else:
            completed, provenance, visibility, runtime = complete_nonempty(rgb, removal, checkpoint, started, unavailable, candidate_manifest_sha)
            final = encode_output(rgb, removal, completed)
        check_time(started)
        if not np.array_equal(final[removal == 0], rgb[removal == 0]):
            raise RuntimeError("Output changed visible source bytes")
        if any(sha(request_dir / name) != digest for name, digest in inputs_sha256.items()):
            raise RuntimeError("Prepared request input changed during generation")
        if source_bindings() != sources_before:
            raise RuntimeError("Completion worker/source changed during generation")
        with (request_dir / "result.png").open("xb") as stream:
            Image.fromarray(final).save(stream, format="PNG")
        with (request_dir / "stages.npz").open("xb") as stream:
            np.savez_compressed(stream, input_rgb=rgb, removal_mask=removal,
                                completed_rgb=completed, output_rgb=final)
        with Image.open(request_dir / "result.png") as saved_image:
            saved = np.array(saved_image, copy=True)
        if not np.array_equal(saved, final):
            raise RuntimeError("Saved result PNG differs from composed output")
        check_time(started)
        outside_bytes = int((removal == 0).sum()) * 3
        receipt = {"format": POLICY, "complete": True, "created_UTC": datetime.now(timezone.utc).isoformat(),
                   "backend": "codeformer-inpainting-finetune-r1", "candidate_manifest_sha256": candidate_manifest_sha, "model_qualified": False, "device": "cpu", "threads": 4, "precision": "FP32",
                   "input_shape": [256, 256, 3], "mask_shape": [256, 256], "output_shape": [256, 256, 3],
                   "prepared_bounds": bounds, "visibility_counts_padding_as_unavailable": True,
                   "internal_resolution": 512 if removal.any() else None, "delivered_resolution": 256,
                   "mask_pixels": int(removal.sum()), "mask_fraction": float(removal.mean()),
                   "empty_mask_bypass": not bool(removal.any()), "neural_calls": runtime["neural_calls"],
                   "optimizer_constructed": False, "optimizer_updates": 0, "gradient_calls": 0,
                   "exact_outside_reviewed_mask": True, "outside_source_bytes": outside_bytes,
                   "outside_source_rgb_sha256": array_sha(rgb[removal == 0]),
                   "outside_output_rgb_sha256": array_sha(final[removal == 0]),
                   "input_rgb_sha256": array_sha(rgb), "removal_mask_sha256": array_sha(removal),
                   "completed_rgb_sha256": array_sha(completed), "output_rgb_sha256": array_sha(final),
                   "inputs_sha256": inputs_sha256,
                   "outputs_sha256": {name: sha(request_dir / name) for name in ("result.png", "stages.npz")},
                   "sources_sha256": sources_before, "provenance": provenance, "visibility": visibility,
                   "timing": {"total_seconds": time.monotonic() - started, "maximum_seconds": MAX_SECONDS,
                              "load_seconds": runtime["load_seconds"], "forward_seconds": runtime["forward_seconds"]},
                   "runtime": runtime, "assisted_only": True, "additional_mask_expansion": 0,
                   "png_conversion": "clamp[0,1], multiply255, np.rint, uint8; original bytes recopied outside removal mask",
                   "visible_restoration_applied": False, "hidden_PSNR_SSIM": None,
                   "meaning": "Hidden facial features are plausible estimates, not verified identity"}
        write_json_exclusive(request_dir / "receipt.json", receipt)
        return receipt
    except Exception as exc:
        if not (request_dir / "failure.json").exists():
            write_json_exclusive(request_dir / "failure.json", {"complete": False, "error_type": type(exc).__name__,
                                                                "error": str(exc), "seconds": time.monotonic() - started,
                                                                "inputs_sha256": inputs_sha256})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--candidate-manifest-sha", required=True)
    args = parser.parse_args()
    try:
        receipt = run_request(args.request_dir, args.checkpoint, args.candidate_manifest_sha)
    except Exception as exc:
        print(json.dumps({"complete": False, "error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"complete": True, "neural_calls": receipt["neural_calls"],
                      "seconds": receipt["timing"]["total_seconds"],
                      "result_sha256": receipt["outputs_sha256"]["result.png"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
