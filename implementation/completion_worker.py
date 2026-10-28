"""One bounded reviewed-mask R1 request, supervised by the owning local app.

Run with -m implementation.completion_worker --request request.json --result
result.json. All trusted server input/output paths must stay within the request
directory. A refusal is distinct from a model/runtime failure. No restoration,
covering detection, training, network or research-folder access occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time

import numpy as np
from PIL import Image

from . import completion_runtime as runtime

MAX_INPUT_BYTES = 8 * 1024**2
MAX_SECONDS = 120


def _pin(path):
    path = Path(path)
    return {"filename": path.name, "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _path(value, base, *, directory=False, exists=True):
    runtime.need(isinstance(value, str) and value, "Expected a request-local path.")
    path = Path(value)
    path = path if path.is_absolute() else base / path
    path = Path(os.path.abspath(path))
    runtime.need(path.is_relative_to(base), "Completion files must stay inside this request directory.")
    for parent in (path, *path.parents):
        if parent.exists() or parent.is_symlink():
            info = parent.lstat()
            runtime.need(not stat.S_ISLNK(info.st_mode) and not getattr(info, "st_file_attributes", 0) & 1024,
                         "Linked/reparse request paths are refused.")
        if parent == base:
            break
    if exists:
        runtime.need(path.is_dir() if directory else path.is_file(), "Missing ordinary request file/directory.")
    return path


def _write_json(path, value, *, replace=False):
    path = Path(path)
    if not replace:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush(); os.fsync(stream.fileno())
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_png(path, mode):
    runtime.need(0 < path.stat().st_size <= MAX_INPUT_BYTES, "Bounded prepared PNG required.")
    pin = _pin(path)
    with Image.open(path) as im:
        runtime.need(im.format == "PNG" and im.mode == mode and im.size == (256, 256),
                     f"Use exact prepared {mode}256 PNG pixels; no worker resizing occurs.")
        pixels = np.array(im, copy=True)
    runtime.need(_pin(path) == pin, "Completion input changed while reading.")
    return pixels, pin


def run_request(request, *, request_root=None):
    runtime.need(type(request) is dict and set(request) == {"input_path", "mask_path", "output_dir", "prepared_bounds"},
                 "Expected input_path, mask_path, output_dir and prepared_bounds only.")
    base = Path(request_root or Path.cwd()).absolute()
    runtime.need(base.is_dir() and not base.is_symlink(), "Ordinary request directory required.")
    output = _path(request["output_dir"], base, directory=True, exists=False)
    output.mkdir(parents=True, exist_ok=True)
    _path(str(output), base, directory=True)
    names = ("status.json", "raw.npy", "raw512.npy", "input512.npy", "stages.npz", "completion.png", "receipt.json")
    runtime.need(not any((output / name).exists() for name in names), "Completion request already has evidence; use a fresh directory.")
    counters = {"neural_attempts": 0, "neural_returns": 0, "neural_calls": 0,
                "optimizer_updates": 0, "gradient_calls": 0, "restoration_calls": 0, "detector_calls": 0}
    started = time.monotonic()

    def status(phase, *, error=None, refused=False):
        data = {"component": "completion_r1", "phase": phase, "complete": phase == "complete",
                "status": "refused" if refused else "error" if error else phase,
                "seconds": time.monotonic() - started, **counters,
                "forward_attempts": counters["neural_attempts"], "forward_returns": counters["neural_returns"]}
        if error:
            data.update({"error_type": type(error).__name__, "error": str(error)[:1000]})
        _write_json(output / "status.json", data, replace=True)
        return data

    status("validating")
    try:
        image_path = _path(request["input_path"], base)
        mask_path = _path(request["mask_path"], base)
        rgb, image_pin = _load_png(image_path, "RGB")
        mask8, mask_pin = _load_png(mask_path, "L")
        runtime.need(np.isin(mask8, [0, 255]).all(), "Removal PNG must contain only0/255 pixels.")
        mask = (mask8 == 255).astype(np.uint8)
        status("loading")

        def forward_event(event):
            key = "neural_attempts" if event == "entered" else "neural_returns"
            counters[key] += 1
            if event == "returned":
                counters["neural_calls"] += 1
            runtime.need(counters[key] <= 1, "One R1 forward maximum.")
            status("forward_entered" if event == "entered" else "forward_returned")

        result = runtime.complete(rgb, mask, request["prepared_bounds"], forward_event=forward_event)
        runtime.need(time.monotonic() - started <= MAX_SECONDS, "Completion exceeded its120-second cooperative limit.")
        runtime.need(_pin(image_path) == image_pin and _pin(mask_path) == mask_pin, "Completion request source changed.")
        final, completed = result["output_rgb"], result["completed_rgb"]
        runtime.need(np.array_equal(final[mask == 0], rgb[mask == 0]), "Completion changed visible canvas bytes.")
        with (output / "raw.npy").open("xb") as stream:
            np.save(stream, completed, allow_pickle=False)
        with (output / "stages.npz").open("xb") as stream:
            np.savez_compressed(stream, input_rgb=rgb, removal_mask=mask, completed_rgb=completed, output_rgb=final)
        with (output / "completion.png").open("xb") as stream:
            Image.fromarray(final).save(stream, format="PNG")
        if result["raw512"] is not None:
            for name, array in (("raw512.npy", result["raw512"]), ("input512.npy", result["input512"])):
                runtime.need(array.dtype == np.float32 and array.shape == (1, 3, 512, 512)
                             and np.isfinite(array).all(), "Invalid raw512 completion evidence.")
                with (output / name).open("xb") as stream:
                    np.save(stream, array, allow_pickle=False)
        with Image.open(output / "completion.png") as image:
            runtime.need(np.array_equal(np.array(image), final), "Saved completion PNG differs.")
        pins = {name: _pin(output / name) for name in names[1:] if (output / name).is_file()}
        metadata = {**result["metadata"], **counters, "complete": True, "status": "complete",
                    "forward_attempts": counters["neural_attempts"], "forward_returns": counters["neural_returns"],
                    "source_pins": {"input": image_pin, "mask": mask_pin}, "output_pins": pins,
                    "input_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                    "removal_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                    "outside_source_rgb_sha256": hashlib.sha256(rgb[mask == 0].tobytes()).hexdigest(),
                    "outside_output_rgb_sha256": hashlib.sha256(final[mask == 0].tobytes()).hexdigest(),
                    "outside_mask_changed_bytes": 0, "supplied_canvas_preserved_outside_mask": True,
                    "automatic_detection_used_in_worker": False, "mask_source": "reviewed request mask",
                    "raw_float_definition": "R1 composed RGB256 float32[0,1] before np.rint PNG quantization.",
                    "raw512_definition": "Unclamped generator output NCHW512 float32[-1,1] nominal range; no claim of bounded raw values.",
                    "seconds": time.monotonic() - started}
        _write_json(output / "receipt.json", metadata)
        status("complete")
        return {"complete": True, "status": "complete", "raw_path": str(output / "raw.npy"),
                "png_path": str(output / "completion.png"), "stages_path": str(output / "stages.npz"),
                "raw512_path": str(output / "raw512.npy") if result["raw512"] is not None else None,
                "receipt_path": str(output / "receipt.json"), "metadata": metadata}
    except Exception as exc:
        status("refused" if isinstance(exc, runtime.CompletionRefused) else "failed",
               error=exc, refused=isinstance(exc, runtime.CompletionRefused))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args(argv)
    base = args.request.absolute().parent
    try:
        source = _path(str(args.request.absolute()), base)
        target = _path(str(args.result.absolute()), base, exists=False)
        runtime.need(not target.exists() and source != target, "Result must be a new request-local file.")
        runtime.need(0 < source.stat().st_size <= 16384, "Bounded completion request JSON required.")
        request = json.loads(source.read_text(encoding="utf-8"))
        result = run_request(request, request_root=base)
        _write_json(target, result)
        return 0
    except Exception as exc:
        refused = isinstance(exc, runtime.CompletionRefused)
        result = {"complete": False, "status": "refused" if refused else "error",
                  "error_type": type(exc).__name__, "error": str(exc)[:1000]}
        try:
            target = _path(str(args.result.absolute()), base, exists=False)
            if not target.exists():
                _write_json(target, result)
        except Exception:
            pass
        print(json.dumps(result), file=sys.stderr)
        return 2 if refused else 1


if __name__ == "__main__":
    raise SystemExit(main())
