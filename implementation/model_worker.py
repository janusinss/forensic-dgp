"""One isolated, frozen CPU forward. No model is constructed on import."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import time

from .runtime import MAX_IMAGE_BYTES, decode_crop, delivered_views, load_frozen_cpu_restorer, model_contract, need, prepare_crop


def _write_status(output, phase, counters, error=None):
    """Persist the last known forward boundary even if the process is stopped."""
    value = {"phase": phase, **counters}
    if error is not None:
        value.update({"failed": True, "error_type": type(error).__name__, "error": str(error)[:1000]})
    target = output / "status.json"
    temporary = output / "status.json.tmp"
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def run_request(request):
    """Execute {input_path, model, output_dir}; return RAW/PNG paths and metadata."""
    need(type(request) is dict and set(request) == {"input_path", "model", "output_dir"},
         "Expected input_path, model and output_dir only.")
    need(all(isinstance(request[key], str) and request[key] for key in request), "Request values must be strings.")
    contract = model_contract(request["model"])
    output = Path(request["output_dir"]).absolute()
    output.mkdir(parents=True, exist_ok=True)
    need(output.is_dir() and not output.is_symlink(), "Ordinary output directory required.")
    phase = "loading"
    counters = {"forward_attempts": 0, "forward_returns": 0}
    _write_status(output, phase, counters)
    try:
        return _execute_request(request, contract, output, counters)
    except Exception as exc:
        # The persisted phase is authoritative if inference or delivery failed.
        status = json.loads((output / "status.json").read_text(encoding="utf-8"))
        _write_status(output, status["phase"], counters, error=exc)
        raise


def _execute_request(request, contract, output, counters):
    source = Path(request["input_path"]).absolute()
    info = source.lstat()
    need(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode)
         and not getattr(info, "st_file_attributes", 0) & 1024 and 0 < info.st_size <= MAX_IMAGE_BYTES,
         "Bounded ordinary input image required.")
    contents = source.read_bytes()
    rgb = decode_crop(contents)
    canvas, observed, _, geometry = prepare_crop(rgb)

    import numpy as np
    import torch
    from torch.nn import functional as F
    from PIL import Image
    torch.set_num_threads(4)
    started = time.monotonic()
    net = load_frozen_cpu_restorer(request["model"])
    x = torch.from_numpy((canvas.astype(np.float32) / np.float32(255)).copy()).permute(2, 0, 1)[None]
    with torch.no_grad():
        sample = F.interpolate(x, (512, 512), mode="bilinear", align_corners=False) * 2 - 1
        counters["forward_attempts"] = 1
        _write_status(output, "forward_entered", counters)
        native = net(sample, w=1.0, detach_16=True, adain=True)[0]
        counters["forward_returns"] = 1
        _write_status(output, "forward_returned", counters)
        raw = F.interpolate(((native.float() + 1) / 2).clamp(0, 1), (256, 256), mode="bilinear",
                            align_corners=False)[0].permute(1, 2, 0).contiguous().numpy().copy()
    raw, pixels = delivered_views(raw, canvas, observed)
    raw_path, png_path = output / "raw.npy", output / "restoration.png"
    with raw_path.open("xb") as stream:
        np.save(stream, raw, allow_pickle=False)
    with png_path.open("xb") as stream:
        Image.fromarray(pixels).save(stream, format="PNG")
    metadata = {
        **contract, "geometry": geometry, "device": "cpu", "precision": "float32",
        "uploaded_bytes_sha256": hashlib.sha256(contents).hexdigest(),
        "raw_rgb_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        "PNG_RGB8_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
        "PNG_file_sha256": hashlib.sha256(png_path.read_bytes()).hexdigest(),
        "forward_sites": list(net.last_forward_sites), "base_calls": net.last_base_calls,
        **counters,
        "outside_observed_support_RAW_and_PNG_match_input_exact": True,
        "RAW_float_definition": "Native RGB clipped to[0,1], bilinear-resized to256; before RGB8 floor quantization.",
        "seconds": time.monotonic() - started,
    }
    _write_status(output, "complete", counters)
    return {"raw_path": str(raw_path), "png_path": str(png_path), "metadata": metadata}


def main(argv=None):
    parser = argparse.ArgumentParser(description="One frozen CPU restoration request.")
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args(argv)
    # Small local protocol, separate from uploaded image bytes.
    need(0 < args.request.stat().st_size <= 16384, "Bounded request JSON required.")
    request = json.loads(args.request.read_text(encoding="utf-8"))
    result = run_request(request)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.result.with_name(args.result.name + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
    os.replace(temporary, args.result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
