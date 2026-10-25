"""Independent non-neural audits for the own-state24-to96 continuation.

This module never imports Torch, the model, or the worker.  Serialized Torch
files are physical byte bindings only; tensor/Adam contents require a separate
local returned-checkpoint audit.  Feature sufficient statistics permit scalar
and decomposition consistency checks, not full activation replay.
"""
from __future__ import annotations

import collections
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import zipfile
import io

import numpy as np
from PIL import Image

SITES = ("32", "64", "128", "256")
CHANNELS = {"32": 256, "64": 256, "128": 128, "256": 128}
VIEWS = ("raw", "native_png")
REFS = ("tr_ffhq_03761", "tr_ffhq_55791", "tr_ffhq_60978",
        "tr_ffhq_34217", "tr_ffhq_31449", "tr_ffhq_52422")
READABLE_STRONG_REFS = ("tr_ffhq_03761", "tr_ffhq_55791",
                        "tr_ffhq_34217", "tr_ffhq_52422")
METRIC_NAMES = ("mae", "mse", "psnr", "ssim", "luminance_bias",
                "signed_luminance_bias", "coarse_rgb_mae", "adjacent_rgb_error")


def require(value, message):
    if not value:
        raise ValueError(message)


def _finite_number(value, message="Finite number required"):
    require(type(value) in (int, float) and math.isfinite(value), message)
    return float(value)


def _near(actual, expected, label, *, atol=2e-10, rtol=2e-10):
    if actual is None or expected is None:
        require(actual is None and expected is None, label + ": null differs")
    else:
        require(math.isclose(_finite_number(actual), _finite_number(expected),
                             abs_tol=atol, rel_tol=rtol), label + ": scalar differs")


def _compare(actual, expected, label="value", *, atol=2e-10, rtol=2e-10):
    if isinstance(expected, dict):
        require(type(actual) is dict and set(actual) == set(expected), label + ": keys differ")
        for key, value in expected.items():
            _compare(actual[key], value, label + "/" + str(key), atol=atol, rtol=rtol)
    elif isinstance(expected, list):
        require(type(actual) is list and len(actual) == len(expected), label + ": list differs")
        for index, value in enumerate(expected):
            _compare(actual[index], value, label + "/" + str(index), atol=atol, rtol=rtol)
    elif type(expected) in (int, float) and type(expected) is not bool:
        _near(actual, expected, label, atol=atol, rtol=rtol)
    else:
        require(type(actual) is type(expected) and actual == expected, label + ": value differs")


def _unique_object(pairs):
    answer = {}
    for key, value in pairs:
        require(key not in answer, "Duplicate JSON key")
        answer[key] = value
    return answer


def _finite_json_float(token):
    value = float(token)
    require(math.isfinite(value), "Nonfinite or overflowing JSON float")
    return value


def read_json(path, *, cap=8 * 1024**2):
    path = Path(path)
    require(path.stat().st_size <= cap, "Bounded JSON required")
    return json.loads(path.read_bytes(), object_pairs_hook=_unique_object,
                      parse_float=_finite_json_float,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def _safe_file(root, name):
    root = Path(root).resolve()
    require(type(name) is str and "\\" not in name and ":" not in name, "Relative POSIX path required")
    rel = PurePosixPath(name)
    require(not rel.is_absolute() and rel.parts and
            all(part not in (".", "..", "") for part in rel.parts) and
            str(rel) == name, "Canonical relative file required")
    path = root.joinpath(*rel.parts)
    require(path.resolve().is_relative_to(root), "File escaped root")
    cursor = path
    while cursor != root:
        info = cursor.lstat()
        require(not stat.S_ISLNK(info.st_mode) and
                not getattr(info, "st_file_attributes", 0) & 1024,
                "Linked/reparse path refused")
        cursor = cursor.parent
    info = path.stat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "Unique regular file required")
    return path


def pin(root, name):
    path = _safe_file(root, name)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), "File changed while hashing")
    return {"path": name, "bytes": after.st_size, "sha256": digest.hexdigest()}


def bound_file(root, binding):
    require(type(binding) is dict and set(binding) == {"path", "bytes", "sha256"},
            "Exact physical binding schema required")
    require(type(binding["bytes"]) is int and binding["bytes"] >= 0 and
            type(binding["sha256"]) is str and len(binding["sha256"]) == 64,
            "Physical size/digest required")
    require(pin(root, binding["path"]) == binding, "Physical binding changed: " + binding["path"])
    return _safe_file(root, binding["path"])


def read_npy(root, binding, shape, dtype):
    path = bound_file(root, binding)
    dtype = np.dtype(dtype)
    require(not dtype.hasobject, "Object arrays prohibited")
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        require(version in ((1, 0), (2, 0)), "NPY version1/2 required")
        parse = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                 else np.lib.format.read_array_header_2_0)
        actual_shape, fortran, actual_dtype = parse(stream)
        header_bytes = stream.tell()
    require(actual_shape == tuple(shape) and not fortran and actual_dtype == dtype,
            "NPY shape/order/dtype mismatch")
    require(path.stat().st_size == header_bytes + math.prod(shape) * dtype.itemsize,
            "NPY trailing/truncated bytes")
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    require(np.isfinite(value).all(), "Nonfinite saved array")
    return value


def rgb(root, binding, *, size=(256, 256)):
    path = bound_file(root, binding)
    with Image.open(path) as image:
        require(image.format == "PNG" and image.mode == "RGB" and image.size == size,
                "Exact RGB PNG geometry required")
        return np.asarray(image).copy()


def independent_metrics(raw, png, target, support=None, evidence="paired_synthetic_TRAIN"):
    """Independent critical metrics; same retained float32 RGB8 target convention."""
    if support is None:
        support = np.ones((256, 256), dtype=np.bool_)
    require(raw.shape == png.shape == (256, 256, 3) and raw.dtype == np.float32 and
            png.dtype == np.uint8 and support.shape == (256, 256) and
            support.dtype == np.bool_ and support.any() and np.isfinite(raw).all() and
            0 <= raw.min() <= raw.max() <= 1, "Delivered finite RGB256 required")
    if evidence in ("native_unpaired", "unpaired_behavior_anchor"):
        require(target is None, "An unpaired input cannot acquire a clean target")
        return {view: dict.fromkeys(METRIC_NAMES) for view in VIEWS}
    require(evidence in ("paired_synthetic", "paired_synthetic_TRAIN",
                         "paired_synthetic_filter_stress") and target is not None and
            target.shape == raw.shape and target.dtype == np.float32 and
            np.isfinite(target).all() and 0 <= target.min() <= target.max() <= 1,
            "Explicit paired target required")
    truth = target.astype(np.float64)
    result = {}
    for view, image in (("raw", raw), ("native_png", png.astype(np.float32) / np.float32(255))):
        delta = image.astype(np.float64) - truth
        measured = delta[support]
        mse = float(np.mean(measured * measured))
        signed = float(np.mean(measured @ np.array([.2126, .7152, .0722])))
        row = dict(mae=float(np.mean(np.abs(measured))), mse=mse,
                   psnr=(-10 * math.log10(mse) if mse else None), ssim=None,
                   luminance_bias=abs(signed), signed_luminance_bias=signed,
                   coarse_rgb_mae=None, adjacent_rgb_error=None)
        if bool(support.all()):
            from skimage.metrics import structural_similarity
            pred = image.astype(np.float64)
            pool = lambda value: value.reshape(16, 16, 16, 16, 3).mean((1, 3), dtype=np.float64)
            row["coarse_rgb_mae"] = float(np.abs(pool(pred) - pool(truth)).mean())
            row["ssim"] = float(structural_similarity(pred, truth, data_range=1., channel_axis=2))
            row["adjacent_rgb_error"] = float(.5 * (
                np.abs(delta[:, 1:] - delta[:, :-1]).mean() +
                np.abs(delta[1:] - delta[:-1]).mean()))
        result[view] = row
    return result


def _ordered_sum(values):
    """Exact VM Python3.10 left-to-right float addition; no gate tolerance."""
    total = 0
    for value in values:
        total += value
    return total


def independent_guard(current, initial):
    """Reconstruct the old24-TRAIN stop; exact <=, no quality tolerance added."""
    require(type(current) is dict and type(initial) is dict and
            set(current) == set(initial) and len(current) == 24, "Exact24 guard IDs required")
    require(collections.Counter(r["role"] for r in current.values()) ==
            {"clear": 8, "dark": 8, "filter": 8}, "Guard role census differs")
    result = {"pass": True, "additional_TRAIN_stop_only": True, "model_qualified": False,
              "groups": {}, "arithmetic": "Python float64; exact <=; no added tolerance"}
    for key in current:
        require(current[key]["role"] == initial[key]["role"] and
                current[key]["split"] == initial[key]["split"] == "TRAIN", "Guard role/split differs")
    for role in ("clear", "dark", "filter"):
        ids = sorted(k for k, row in current.items() if row["role"] == role)
        group = {"pass": True, "cases": ids, "views": {}}
        for view in VIEWS:
            group["views"][view] = {}
            fields = ("mae", "adjacent_rgb_error") if role == "filter" else ("behavior_distance",)
            for field in fields:
                get = lambda bank, key: (bank[key]["behavior_distance"][view]
                                         if field == "behavior_distance" else bank[key]["metrics"][view][field])
                start = {key: _finite_number(get(initial, key)) for key in ids}
                now = {key: _finite_number(get(current, key)) for key in ids}
                require(min(start.values()) >= 0 and min(now.values()) >= 0, "Negative guard distance")
                if role == "filter":
                    measured = {key: now[key] - start[key] for key in ids}
                    limits = dict.fromkeys(ids, .03)
                    mean_limit = .01
                else:
                    measured = now
                    limits = {key: max(.03, start[key]) for key in ids}
                    mean_limit = max(.01, _ordered_sum(start.values()) / 8)
                mean = _ordered_sum(measured.values()) / 8
                failed = [key for key in ids if measured[key] > limits[key]]
                passed = not failed and mean <= mean_limit
                group["views"][view][field] = {
                    "pass": passed, "current": now, "initial": start, "case_limits": limits,
                    "failed_case_ids": failed, "mean_measurement": mean, "mean_limit": mean_limit,
                    "measurement": "increase from initial" if role == "filter" else "distance from frozen teacher"}
                group["pass"] = bool(group["pass"] and passed)
        result["groups"][role] = group
        result["pass"] = bool(result["pass"] and group["pass"])
    return result


def objective_fixtures():
    """Counterexamples protect centering axes, DC nullspace and stop semantics."""
    def centered(value):
        return value - value.mean(axis=(-2, -1), keepdims=True, dtype=np.float64)
    def u(value, target, scale):
        return float(np.abs(value - target).mean() / scale)
    def s(value, target, scale):
        return float(np.abs(centered(value) - centered(target)).mean() / scale)
    target = np.arange(60, dtype=np.float64).reshape(2, 2, 3, 5) / 17
    offset = np.array([[1., -2.], [3., -.5]])[:, :, None, None]
    value = target + offset
    scale = math.sqrt(float(np.square(target).mean()))
    require(u(value, target, scale) > .1 and s(value, target, scale) < 1e-14,
            "DC-only change must distinguish U and S")
    # A3x5 map exposes accidentally centering only rows, only columns or channels.
    spatial = np.zeros_like(target)
    spatial[:, :, 0, 0] = 4
    spatial[:, :, 2, 4] = -4
    require(s(target + spatial, target, scale) > .05, "Non-square spatial signal lost")
    require(np.allclose(centered(target + offset), centered(target), atol=2e-15),
            "Per-example/per-channel centering differs")
    # Scalar finite differences, not autodiff, verify the output-bias nullspace.
    step = 1e-5
    plus = s(target + spatial + step * offset, target, scale)
    minus = s(target + spatial - step * offset, target, scale)
    require(abs((plus - minus) / (2 * step)) < 1e-9, "Centered objective output-bias derivative")
    require(s(target + spatial, target, 2 * scale) == s(target + spatial, target, scale) / 2,
            "Fixed clean RMS scaling differs")
    require(s(target, target, scale) == u(target, target, scale) == 0,
            "Identity zero-residual fixture differs")
    # Mean-only and worst-case-only failures are independently binding.
    base = {}
    for role in ("clear", "dark", "filter"):
        for number in range(8):
            key = role + str(number)
            base[key] = {"role": role, "split": "TRAIN",
                         "behavior_distance": dict.fromkeys(VIEWS, .005),
                         "metrics": {v: {"mae": .1, "adjacent_rgb_error": .05} for v in VIEWS}}
    require(independent_guard(base, base)["pass"], "Initial guard must pass its additional stop")
    change = json.loads(json.dumps(base))
    for row in change.values():
        if row["role"] == "clear":
            row["behavior_distance"]["raw"] = .011
    require(not independent_guard(change, base)["pass"], "Guard mean failure not detected")
    change = json.loads(json.dumps(base))
    change["dark0"]["behavior_distance"]["native_png"] = .03000000000000001
    require(not independent_guard(change, base)["pass"], "Exact worst-case threshold not binding")
    raw = np.full((256, 256, 3), .25, dtype=np.float32)
    png = np.full((256, 256, 3), 64, dtype=np.uint8)
    truth = np.zeros_like(raw)
    measured = independent_metrics(raw, png, truth)
    _near(measured["raw"]["mae"], .25, "Known MAE", atol=0, rtol=0)
    _near(measured["raw"]["mse"], .0625, "Known MSE", atol=0, rtol=0)
    _near(measured["raw"]["psnr"], 10 * math.log10(16), "Known PSNR")
    require(measured["raw"]["adjacent_rgb_error"] == 0 and
            measured["native_png"]["mae"] != measured["raw"]["mae"], "RAW and PNG conflated")
    require(all(x is None for row in independent_metrics(raw, png, None,
            evidence="native_unpaired").values() for x in row.values()), "Unpaired accuracy invented")
    require(all(x is None for row in independent_metrics(raw, png, None,
            evidence="unpaired_behavior_anchor").values() for x in row.values()),
            "Dark behavior anchor cannot acquire photographic target metrics")
    for payload in ('{"x":1e999}', '{"x":NaN}', '{"x":1,"x":2}'):
        try:
            json.loads(payload, object_pairs_hook=_unique_object, parse_float=_finite_json_float,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
        except ValueError:
            pass
        else:
            raise ValueError("Strict JSON overflow/nonfinite/duplicate fixture failed")
    require(_ordered_sum([1e16, 1., -1e16]) == 0.,
            "VM Python3.10 ordered float sum cannot become local compensated sum")
    old72 = [{"id": "case" + str(i % 10), "update": i // 3 + 1, "slot": i % 3 + 1}
             for i in range(72)]
    continued = [dict(r, update=r["update"] + 24 + repeat * 24)
                 for repeat in range(3) for r in old72]
    require(len(continued) == 216 and continued[0]["update"] == 25 and
            continued[-1]["update"] == 96 and
            all(continued[i]["id"] == old72[i % 72]["id"] for i in range(216)),
            "Absolute25..96 deliberately repeats original72 exposure order")
    require(8 + 2 * 5 * 24 + 2 * 3 * 4 == 272 and
            2 * 72 * 3 * 4 + 2 * 4 * 10 * 4 + 272 * 4 == 3136,
            "Initial own24 parity/statistics included in finite continuation counts")
    return {"pass": True, "fixtures": ["DC-only U/S separation", "non-square3x5 spatial axes",
            "per-example/channel DC nullspace finite difference", "same fixed total-clean RMS",
            "identity error zero", "guard mean and exact-case failures", "known MAE/MSE/PSNR",
            "RAW-PNG distinction", "native-unpaired clean scores null",
            "dark behavior-anchor clean scores null separately from native CCTV",
            "JSON overflow/nonfinite/duplicate rejection", "exact Python3.10 ordered float sum",
            "absolute25..96 repeats original72 sampler", "272full/3136adapter initial-parity closure"],
            "Torch_neural_gradient_optimizer_VM_browser_calls": 0}


def _cached_output(root, row):
    raw = read_npy(root, row["outputs"]["raw"], (256, 256, 3), "float32")
    png = rgb(root, row["outputs"]["png"])
    require(np.array_equal(np.floor(raw * np.float32(255)).astype(np.uint8), png),
            "Delivered PNG must be exact historical floor(RAW256*255)")
    if row["target"] is None:
        require(row.get("role") == "dark" and row.get("behavior_teacher") is not None,
                "Only explicitly unpaired dark behavior anchors may omit a target")
        measured = independent_metrics(raw, png, None, evidence="unpaired_behavior_anchor")
    else:
        target = rgb(root, row["target"]).astype(np.float32) / np.float32(255)
        measured = independent_metrics(raw, png, target)
    _compare(row["metrics"], measured, "Cached paired metrics")
    if row.get("role") in ("clear", "dark"):
        teacher = row["behavior_teacher"]
        truth_raw = read_npy(root, teacher["raw"], (256, 256, 3), "float32")
        truth_png = rgb(root, teacher["png"]).astype(np.float32) / np.float32(255)
        distances = {
            "raw": float(np.abs(raw.astype(np.float64) - truth_raw.astype(np.float64)).mean()),
            "native_png": float(np.abs((png.astype(np.float32) / np.float32(255)).astype(np.float64) -
                                        truth_png.astype(np.float64)).mean())}
        _compare(row["behavior_distance"], distances, "Cached teacher distances")
        require(teacher["kind"] == ("original" if row["role"] == "clear" else "r2"),
                "Clear untouched-CodeFormer / dark R2 teacher required")
    return measured


def _clean_statistics(root, binding, site):
    array = read_npy(root, binding, (1, CHANNELS[site], int(site), int(site)), "float32")
    channels, height = CHANNELS[site], int(site)
    means = np.empty(channels, dtype=np.float64)
    squares = np.empty(channels, dtype=np.float64)
    pooled = np.empty((channels, 8, 8), dtype=np.float64)
    for start in range(0, channels, 8):
        part = array[0, start:start + 8].astype(np.float64)
        means[start:start + len(part)] = part.mean((1, 2))
        squares[start:start + len(part)] = np.square(part).mean((1, 2))
        pooled[start:start + len(part)] = part.reshape(len(part), 8, height // 8,
                                                      8, height // 8).mean((2, 4))
    return {"means": means, "squares": squares, "pooled": pooled,
            "RMS": max(1e-6, math.sqrt(float(squares.mean())))}




MOMENTS = ("wanted_abs", "final_abs", "wanted_AC_abs", "final_AC_abs", "wanted_sq",
           "correction_sq", "final_sq", "wanted_spatial_sq", "correction_spatial_sq",
           "spatial_dot", "observed_sq", "target_sq")


def feature_summary(moments, means, pooled, scale, *, clean=None, initial=False):
    """Recompute scalar reductions from12xC moments,3xC means and pooled8."""
    require(moments.dtype == means.dtype == pooled.dtype == np.float64 and
            moments.ndim == 2 and moments.shape[0] == 12 and
            means.shape == (3, moments.shape[1]) and
            pooled.shape == (3, moments.shape[1], 8, 8) and
            all(np.isfinite(a).all() for a in (moments, means, pooled)) and
            math.isfinite(scale) and scale >= 1e-6, "Exact sufficient-statistic schema required")
    require(np.allclose(pooled.mean((2, 3)), means, atol=2e-10, rtol=2e-10),
            "Pooled/channel mean association differs")
    require(np.all(moments[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11]] >= 0),
            "Negative absolute/squared moment")
    wanted_DC = np.square(means[1]); correction_DC = np.square(means[2])
    final_DC = np.square(means[2] - means[1])
    total_dot = (moments[5] + moments[4] - moments[6]) / 2
    require(np.allclose(moments[4], wanted_DC + moments[7], atol=2e-10, rtol=2e-10),
            "Wanted DC/spatial decomposition differs")
    require(np.allclose(moments[5], correction_DC + moments[8], atol=2e-10, rtol=2e-10),
            "Correction DC/spatial decomposition differs")
    require(np.allclose(total_dot, means[1] * means[2] + moments[9], atol=2e-10, rtol=2e-10),
            "Dot DC/spatial decomposition differs")
    final_AC_sq = moments[7] + moments[8] - 2 * moments[9]
    require(np.all(final_AC_sq >= -2e-10) and
            np.allclose(moments[6], final_DC + final_AC_sq, atol=2e-10, rtol=2e-10),
            "Final residual DC/spatial decomposition differs")
    require(np.all(np.square(total_dot) <= moments[4] * moments[5] + 2e-10) and
            np.all(np.square(moments[9]) <= moments[7] * moments[8] + 2e-10),
            "Moment Cauchy bound differs")
    for absolute, square in ((0, 4), (1, 6), (2, 7)):
        require(np.all(np.square(moments[absolute]) <= moments[square] + 2e-10),
                "Absolute moment exceeds RMS")
    require(np.all(np.square(moments[3]) <= final_AC_sq + 2e-10),
            "Centered final absolute moment exceeds RMS")
    if clean is not None:
        require(np.allclose(means[0] + means[1], clean["means"], atol=2e-10, rtol=2e-10) and
                np.allclose(pooled[0] + pooled[1], clean["pooled"], atol=2e-10, rtol=2e-10) and
                np.allclose(moments[11], clean["squares"], atol=2e-10, rtol=2e-10),
                "Clean target moments/means/pooled association differs")
        _near(scale, clean["RMS"], "Fixed total-clean RMS", atol=1e-9, rtol=1e-9)
    if initial:
        require(np.count_nonzero(means[2]) == np.count_nonzero(pooled[2]) == 0 and
                np.count_nonzero(moments[5]) == np.count_nonzero(moments[8]) ==
                np.count_nonzero(moments[9]) == 0 and
                np.array_equal(moments[0], moments[1]) and
                np.array_equal(moments[2], moments[3]) and
                np.array_equal(moments[4], moments[6]), "Identity initial arithmetic differs")
    scalar = {name: float(moments[i].mean()) for i, name in enumerate(MOMENTS)}
    cDC, wDC = float(correction_DC.mean()), float(wanted_DC.mean())
    dotDC, dot = float((means[1] * means[2]).mean()), float(total_dot.mean())
    cosine = lambda product, aa, bb: product / math.sqrt(aa * bb) if aa > 0 and bb > 0 else None
    gain = lambda start, end: 1 - end / start if start > 0 else None
    return {"initial_normalized_L1": scalar["wanted_abs"] / scale,
            "final_normalized_L1": scalar["final_abs"] / scale,
            "initial_centered_normalized_L1": scalar["wanted_AC_abs"] / scale,
            "final_centered_normalized_L1": scalar["final_AC_abs"] / scale,
            "raw_relative_gain": gain(scalar["wanted_abs"], scalar["final_abs"]),
            "centered_relative_gain": gain(scalar["wanted_AC_abs"], scalar["final_AC_abs"]),
            "correction_RMS_over_clean_RMS": math.sqrt(scalar["correction_sq"]) / scale,
            "wanted_RMS_over_clean_RMS": math.sqrt(scalar["wanted_sq"]) / scale,
            "final_RMS_over_clean_RMS": math.sqrt(scalar["final_sq"]) / scale,
            "correction_DC_energy": cDC, "correction_spatial_energy": scalar["correction_spatial_sq"],
            "wanted_DC_energy": wDC, "wanted_spatial_energy": scalar["wanted_spatial_sq"],
            "correction_DC_energy_fraction": cDC / scalar["correction_sq"] if scalar["correction_sq"] else None,
            "direction_alignment": cosine(dot, scalar["correction_sq"], scalar["wanted_sq"]),
            "DC_direction_alignment": cosine(dotDC, cDC, wDC),
            "spatial_direction_alignment": cosine(scalar["spatial_dot"], scalar["correction_spatial_sq"],
                                                   scalar["wanted_spatial_sq"])}


def read_statistics_npz(root, binding):
    path = bound_file(root, binding)
    require(path.stat().st_size <= 2 * 1024**2, "Bounded case statistics NPZ required")
    shapes = {site + "_means": (3, CHANNELS[site]) for site in SITES}
    shapes.update({site + "_moments": (12, CHANNELS[site]) for site in SITES})
    shapes.update({site + "_pooled": (3, CHANNELS[site], 8, 8) for site in SITES})
    result = {}
    with zipfile.ZipFile(path) as archive:
        require(len(archive.infolist()) == 12 and
                set(archive.namelist()) == {key + ".npy" for key in shapes},
                "Exact12 site statistics arrays required")
        for info in archive.infolist():
            key = info.filename[:-4]
            require(not info.is_dir() and not info.flag_bits & 1 and
                    info.file_size <= 512 * 1024 and info.compress_type in
                    (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED), "Bounded unencrypted NPY member required")
            payload = archive.read(info)
            stream = io.BytesIO(payload)
            version = np.lib.format.read_magic(stream)
            require(version in ((1, 0), (2, 0)), "NPZ NPY version1/2 only")
            parse = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                     else np.lib.format.read_array_header_2_0)
            shape, fortran, dtype = parse(stream)
            require(shape == shapes[key] and not fortran and dtype == np.dtype("float64") and
                    len(payload) == stream.tell() + math.prod(shape) * 8, "NPZ array shape/dtype/EOF differs")
            stream.seek(0)
            result[key] = np.load(stream, allow_pickle=False)
            require(np.isfinite(result[key]).all(), "Nonfinite feature statistics")
    return result


def audit_feature_row(root, protocol, row, *, reference_root=None, initial=False, _clean_cache=None):
    """Check source-bound sufficient statistics, explicitly without activation replay."""
    refs = Path(root if reference_root is None else reference_root).resolve()
    cases = {r["id"]: r for r in protocol["train_rows"]}
    require(row["id"] in cases, "Statistics outside admitted10 TRAIN cases")
    case = cases[row["id"]]
    clean = protocol["clean_features"][case["reference_id"]]
    require(row["reference_id"] == case["reference_id"] and row["source_input"] == case["input"] and
            row["clean_targets"] == clean["features"] and row["scales"] == clean["scales"] and
            row["base515_sha256"] == protocol["base515_sha256"] and
            row["initial_identity"] is initial, "Statistics case/target/frozen-base join differs")
    require(set(row["site_metrics"]) == set(SITES), "Four exact feature sites required")
    arrays = read_statistics_npz(root, row["arrays"])
    answer = {}
    drift = {}
    for site in SITES:
        cache = {} if _clean_cache is None else _clean_cache
        key = (case["reference_id"], site)
        if key not in cache:
            cache[key] = _clean_statistics(refs, clean["features"][site], site)
        target = cache[key]
        summary = feature_summary(arrays[site + "_moments"], arrays[site + "_means"],
                                  arrays[site + "_pooled"], row["scales"][site], clean=target, initial=initial)
        _compare(row["site_metrics"][site], summary, "Feature scalar reductions")
        answer[site] = summary
        drift[site] = float(np.abs(arrays[site + "_means"][2]).mean())
    return {"id": row["id"], "arm": row["arm"], "update": row["update"],
            "site_metrics": answer, "mean_absolute_channel_drift": drift,
            "clean_feature_reference_root_explicit": reference_root is not None,
            "arrays": row["arrays"], "full_activation_replay": False,
            "absolute_moments_source_captured_consistency_only": True}


def _json_lines(root, name, cap=16 * 1024**2):
    path = _safe_file(root, name)
    require(path.stat().st_size <= cap, "Finite journal cap exceeded")
    value = path.read_bytes()
    require(not value or value.endswith(b"\n"), "Truncated durable journal line")
    return [json.loads(line, object_pairs_hook=_unique_object,
                       parse_float=_finite_json_float,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSONL")))
            for line in value.splitlines()]






def _closed_namespace(root, protocol, *, returned):
    root = Path(root).resolve()
    files = {}
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs:
            info = (Path(directory) / name).lstat()
            require(not stat.S_ISLNK(info.st_mode) and
                    not getattr(info, "st_file_attributes", 0) & 1024, "Linked directory refused")
        for name in names:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            files[relative] = _safe_file(root, relative)
    allowed = set(protocol["files"]) | set(protocol["allowed_outputs"]) | {"protocol.json"}
    require(set(files) <= allowed, "File outside sealed namespace")
    vm_only = set(protocol["vm_only_static"])
    require(vm_only <= set(protocol["files"]) and
            set(protocol["return_static"]) == set(protocol["files"]) - vm_only,
            "Return/static partition differs")
    required = set(protocol["files"]) - (vm_only if returned else set())
    require(required <= set(files), "Required static files missing")
    output_bytes = 0
    cache_bytes = 0
    for name, path in files.items():
        size = path.stat().st_size
        if name.startswith("cache/"):
            cache_bytes += size
        elif name not in vm_only:
            output_bytes += size
    require(output_bytes <= protocol["limits"]["return_bytes"] and
            cache_bytes <= protocol["limits"]["cache_bytes"] and
            len(files) <= protocol["limits"]["files"], "Closed namespace size/file cap")
    if returned:
        require(not any(n.startswith("cache/") for n in files) and not vm_only.intersection(files),
                "Returned payload may not duplicate VM-only weights/features/cache")
    return {"files": sorted(files), "file_count": len(files),
            "return_namespace_bytes": output_bytes, "VM_only_cache_bytes": cache_bytes,
            "returned_payload": returned}


def _audit_output_row(root, source, row, *, arm, update, phase):
    require(row["id"] == source["id"] and row["reference_id"] == source["reference_id"] and
            row["profile"] == source["profile"] and row["role"] == source.get("role") and
            row["input"] == source["input"] and row["target"] == source["target"] and
            row["split"] == "TRAIN" and row["arm"] == arm and row["update"] == update and
            row["phase"] == phase and row["upstream"] == source["upstream"] and
            row["RAW_PNG_separate"] is True and row["model_qualified"] is False,
            "Output cohort/input/frozen-upstream provenance differs")
    if source.get("behavior_teacher") is not None:
        require(row["behavior_teacher"] == source["behavior_teacher"], "Behavior teacher changed")
    measured = _cached_output(root, row)
    if phase == "parity":
        actual_raw = read_npy(root, row["outputs"]["raw"], (256, 256, 3), "float32")
        baseline_raw = read_npy(root, source["outputs"]["raw"], (256, 256, 3), "float32")
        require(np.array_equal(actual_raw, baseline_raw) and
                np.array_equal(rgb(root, row["outputs"]["png"]), rgb(root, source["outputs"]["png"])) and
                row["own24_cached_RAW_PNG_upstream_exact"] is True, "Exact cached own24 CUDA parity differs")
    return measured


def _bank(root, relative, sources, *, arm, update, phase):
    bank = read_json(_safe_file(root, relative))
    require(bank["complete"] is True and bank["arm"] == arm and bank["update"] == update and
            bank["model_qualified"] is False and len(bank["cases"]) == len(sources), "Full output bank metadata")
    current = {row["id"]: row for row in bank["cases"]}
    require(len(current) == len(sources) and set(current) == {row["id"] for row in sources},
            "Full bank exact cohort coverage")
    stem = relative[:-len("bank.json")]
    for source in sources:
        row = current[source["id"]]
        sidecar = read_json(_safe_file(root, stem + row["id"] + ".json"))
        require(sidecar == row, "Embedded output bank/sidecar differs")
        require(row["outputs"]["raw"]["path"] == stem + row["id"] + ".npy" and
                row["outputs"]["png"]["path"] == stem + row["id"] + ".png", "Output payload path differs")
        _audit_output_row(root, source, row, arm=arm, update=update, phase=phase)
    return current


def _export_manifest(root, protocol, namespace, supervisor):
    name = "export.json"
    receipt = read_json(_safe_file(root, name))
    protocol_sha = hashlib.sha256((Path(root) / "protocol.json").read_bytes()).hexdigest()
    require(receipt["format"] == "codeformer-spatial-continuation-export-manifest-v1" and
            receipt["complete"] is True and receipt["protocol_sha256"] == protocol_sha and
            receipt["worker_terminal"] is True and receipt["worker_exit"] == supervisor["worker_exit"] and
            receipt["training_success_not_implied"] is True and receipt["model_qualified"] is False and
            receipt["VM_only_static_and_observed_features_not_returned"] is True and
            receipt["checkpoint_pickle_contents_not_audited_by_export"] is True,
            "Exact terminal export boundary receipt required")
    members = receipt["members"]
    require(type(members) is list and len({r["path"] for r in members}) == len(members),
            "Unique export member bindings")
    expected = {name for name in namespace["files"] if name != "export.json" and
                not name.startswith("cache/") and name not in protocol["vm_only_static"]}
    require({r["path"] for r in members} == expected, "Export did not retain exact closed return subset")
    for binding in members:
        bound_file(root, binding)
    return {"complete": True, "members": len(members), "export_manifest": pin(root, "export.json"),
            "all_member_byte_bindings_verified": True,
            "archive_container_bytes_independently_verified": False}


O = "outputs/spatial_continuation_v1/"
CAPS = {"prefix": 10, "adapter_train": 1728, "adapter_stats": 320,
        "full_forward": 272, "adapter_full": 1088, "backward": 432, "optimizer": 144}
GUARD_POINTS = (30, 36, 48, 72, 96)
STAT_POINTS = (24, 48, 72, 96)
STRONG_POINTS = (48, 72, 96)


def _parent_static(root, protocol):
    """Byte-bind audited own24 states without deserializing a Torch checkpoint."""
    binding = protocol["parent_protocol"]
    parent = read_json(bound_file(root, binding))
    require(binding["sha256"] == "3fe24e684a835268ac805ca56bfafefff429ef5f1aa13a90ec11faa644380dd9",
            "Exact approved parent protocol required")
    require(protocol["parent_protocol_sha256"] == binding["sha256"] and
            protocol["base515_sha256"] == parent["base515_sha256"] and
            protocol["sampler_original72"] == parent["sampler"], "Exact parent sampler/base join")
    # Content-addressed paths may change; physical digests and RGB association may not.
    for new, old in zip(protocol["train_rows"], parent["train_rows"]):
        require(all(new[k] == old[k] for k in
                    ("id", "reference_id", "profile", "split", "input_only_decision", "input_RGB8_sha256",
                     "target_RGB8_sha256")) and
                all(new[k][v] == old[k][v] for k in ("input", "target") for v in ("bytes", "sha256")),
                "No new training observation or photographic target in continuation")
    require(len(parent["train_rows"]) == len(protocol["train_rows"]) == 10,
            "Exact original10 training observations")
    for new, old in zip(protocol["guards"], parent["guards"]):
        require(all(new[k] == old[k] for k in ("id", "role", "reference_id", "split", "profile", "metrics",
                                               "upstream")), "Unchanged guard0 cohort/metrics/upstream")
        if old["target"] is None:
            require(new["target"] is None, "Dark anchor must remain without photographic target")
        for key in ("input", "outputs"):
            if key == "outputs":
                require(all(new[key][v]["sha256"] == old[key][v]["sha256"] for v in ("raw", "png")),
                        "Guard0 output changed")
            else:
                require(new[key]["sha256"] == old[key]["sha256"], "Guard0 input changed")
        if old.get("behavior_teacher"):
            require(new["behavior_distance"] == old["behavior_distance"] and
                    new["behavior_teacher"]["kind"] == old["behavior_teacher"]["kind"] and
                    all(new["behavior_teacher"][v]["sha256"] == old["behavior_teacher"][v]["sha256"]
                        for v in ("raw", "png")), "Behavior teachers/distance changed")
    require(len(parent["guards"]) == len(protocol["guards"]) == 24 and
            [r["id"] for r in protocol["strong"]] == [r["id"] for r in parent["strong"]],
            "Unchanged guard24 and strong4 cohorts")
    for ref, entry in protocol["clean_features"].items():
        require(entry["scales"] == parent["clean_features"][ref]["scales"] and
                all(entry["features"][s]["sha256"] == parent["clean_features"][ref]["features"][s]["sha256"]
                    for s in SITES), "Original24 clean features and total RMS unchanged")
    require(set(protocol["resume"]) == {"U", "S"}, "Both own24 states required")
    evidence = protocol["parent_evidence"]
    require(set(evidence) == {"checkpoint_content_audit", "pure_return_audit", "parent_result", "export_manifest"},
            "Explicit independently audited parent state/evidence chain required")
    state_audit = read_json(bound_file(root, evidence["checkpoint_content_audit"]))
    pure_audit = read_json(bound_file(root, evidence["pure_return_audit"]))
    parent_result = read_json(bound_file(root, evidence["parent_result"]))
    parent_export = read_json(bound_file(root, evidence["export_manifest"]))
    require(evidence["checkpoint_content_audit"]["sha256"] ==
            "a3d3f8b5533b2d98a8445bbfc010fe8cf83e046296cd9eb0711894e0e3fb9704" and
            evidence["pure_return_audit"]["sha256"] ==
            "94344dc1494c2da4818222bc8c28d9b8986179b9ccf0ee14b2587e84bca52f4a" and
            evidence["parent_result"]["sha256"] ==
            "4dc5a10bb73eea23b2f495c0de07432b7bc17671ae501736979ea7065e39b5dc" and
            evidence["export_manifest"]["sha256"] ==
            "9da40ef8b7bd237050596b4e97de6fa3be4d8316d65a03587df211bd39d82929" and
            state_audit["complete"] is True and state_audit["pass"] is True and
            pure_audit["audit_pass"] is True and parent_result["complete"] is True,
            "Exact authoritative parent evidence and successful independent audits")
    exported = {r["path"]: r for r in parent_export["members"]}
    require(parent_export["protocol_sha256"] == binding["sha256"] and
            parent_export["complete"] is True and parent_export["worker_exit"] == 0,
            "Successful original parent return manifest")
    audited24 = {r["arm"]: r for r in state_audit["checkpoints"] if r["update"] == 24}
    parent_branches = {r["arm"]: r for r in parent_result["branches"]}
    state_report = []
    for arm, resume in protocol["resume"].items():
        bound_file(root, resume["checkpoint"])
        require(resume["completed_updates"] == 24 and len(resume["adapter_sha256"]) == 64 and
                resume["adapter_sha256"] != parent["initial_adapter_sha256"], "Actual own24 learned checkpoint")
        previous = audited24[arm]
        require(resume["checkpoint"]["sha256"] == previous["checkpoint"]["sha256"] and
                resume["checkpoint"]["bytes"] == previous["checkpoint"]["bytes"] and
                resume["adapter_sha256"] == previous["adapter_sha256"] and
                previous["state_tensor_count"] == 24 and previous["Adam_parameter_states"] == 24 and
                previous["Adam_steps_exact"] == 24 and parent_branches[arm]["completed_updates"] == 24 and
                parent_branches[arm]["terminal_status"] == "complete" and
                parent_branches[arm]["last_checkpoint"]["sha256"] == resume["checkpoint"]["sha256"],
                "Own24 checkpoint adapter/Adam counter and terminal parent chain differs")
        require(len(resume["parity_rows"]) == 4 and
                [r["id"] for r in resume["parity_rows"]] == protocol["parity_ids"],
                "Own24 four parity witnesses per branch")
        for row in resume["parity_rows"]:
            require(row["arm"] == arm and row["update"] == 24 and row["split"] == "TRAIN",
                    "Own24 parity is not initial identity R2")
            original_input = next(g for g in protocol["guards"] if g["id"] == row["id"])
            require(all(row[field] == original_input[field] for field in
                        ("id", "reference_id", "profile", "role", "split", "input", "target",
                         "input_RGB8_sha256", "target_RGB8_sha256")),
                    "Own24 parity retains exact original decoded-input and target association")
            pixels = rgb(root, row["input"])
            require(hashlib.sha256(pixels.tobytes()).hexdigest() == row["input_RGB8_sha256"],
                    "Own24 full-parity decoded RGB8 input association")
            if row["target"] is not None:
                target = rgb(root, row["target"])
                require(hashlib.sha256(target.tobytes()).hexdigest() == row["target_RGB8_sha256"],
                        "Own24 full-parity decoded photographic target association")
            for view, suffix in (("raw", ".npy"), ("png", ".png")):
                original = exported[f"outputs/spatial_objectives_v1/guards/{arm}/u0024/{row['id']}" + suffix]
                require(all(row["outputs"][view][k] == original[k] for k in ("bytes", "sha256")),
                        "Own24 parity RAW/PNG must match actual original arm24 exported guard output")
            _cached_output(root, row)
        feature_rows = resume["feature_rows"]
        old_feature_rows = {r["id"]: r for r in parent_branches[arm]["feature_statistics"][-1]["rows"]}
        require(len(feature_rows) == 10 and [r["id"] for r in feature_rows] ==
                [r["id"] for r in protocol["train_rows"]], "All10 own24 feature targets")
        for row in feature_rows:
            require(row["arm"] == arm and row["update"] == 24 and
                    row["adapter_state_sha256"] == resume["adapter_sha256"] and
                    row["base515_sha256"] == protocol["base515_sha256"] and
                    row["initial_identity"] is False, "Own24 feature state provenance")
            previous_row = old_feature_rows[row["id"]]
            require(row["arrays"]["sha256"] == previous_row["arrays"]["sha256"] and
                    row["site_metrics"] == previous_row["site_metrics"] and
                    row["scales"] == previous_row["scales"], "Exact parent own24 feature evidence")
            read_statistics_npz(root, row["arrays"])
        state_report.append({"arm": arm, "checkpoint": resume["checkpoint"],
                             "adapter_sha256": resume["adapter_sha256"], "completed_updates": 24,
                             "own_output_witnesses": 4, "own_feature_case_targets": 10,
                             "Torch_checkpoint_contents_reread_by_this_pure_audit": False})
    return parent, state_report


def audit_static(root, protocol):
    root = Path(root).resolve()
    require(protocol["vm_name"] == "cctv_codeformer_spatial_continuation_v1_vm" and
            protocol["max_calls"] == CAPS, "Distinct finite continuation packet required")
    for name, expected in protocol["files"].items():
        bound_file(root, {"path": name, **expected})
    namespace = _closed_namespace(root, protocol, returned=False)
    parent, resume_report = _parent_static(root, protocol)
    rows = protocol["train_rows"]
    require(len(rows) == 10 and len({r["id"] for r in rows}) == 10 and
            set(r["reference_id"] for r in rows) == set(REFS) and
            all(r["split"] == "TRAIN" and r["input_only_decision"] == "readable" for r in rows),
            "Only original readable10 TRAIN observations from6 photos")
    require(collections.Counter("strong" if r["profile"] == "strong_fixed_blur3p2_lr64_jpeg50" else
                                "moderate" if "gaussian_2p4" in r["profile"] else "unexpected"
                                for r in rows) == {"moderate": 6, "strong": 4},
            "Six moderate and four strong; no borderline/mosaic admission")
    sampler = protocol["sampler"]
    require(len(sampler) == 216, "72 additional updates times3 micros required")
    for i, row in enumerate(sampler):
        original = protocol["sampler_original72"][i % 72]
        require(row == {"id": original["id"], "update": i // 3 + 25, "slot": i % 3 + 1},
                "Same original72-row pattern repeated3times per arm; absolute25..96")
    guards, strong = protocol["guards"], protocol["strong"]
    require(len(guards) == 24 and collections.Counter(r["role"] for r in guards) ==
            {"clear": 8, "dark": 8, "filter": 8} and len(strong) == 4 and
            set(r["reference_id"] for r in strong) == set(READABLE_STRONG_REFS), "Guard/probe scope")
    for row in rows + guards + strong:
        image = rgb(root, row["input"])
        require(hashlib.sha256(image.tobytes()).hexdigest() == row["input_RGB8_sha256"], "Exact RGB8 input")
        if row["target"] is None:
            require(row.get("role") == "dark" and row.get("target_RGB8_sha256") is None,
                    "Only dark behavior anchors have no aligned target")
        else:
            target = rgb(root, row["target"])
            require(hashlib.sha256(target.tobytes()).hexdigest() == row["target_RGB8_sha256"],
                    "Exact RGB8 photographic target")
    clean_cache, scale_rows = {}, []
    require(set(protocol["clean_features"]) == set(REFS), "Same six reference photos")
    for ref, entry in protocol["clean_features"].items():
        require(set(entry["features"]) == set(entry["scales"]) == set(SITES), "All24 clean target arrays")
        for site in SITES:
            stats = _clean_statistics(root, entry["features"][site], site)
            _near(entry["scales"][site], stats["RMS"], "Unchanged total-clean RMS", atol=1e-9, rtol=1e-9)
            clean_cache[ref, site] = stats
            scale_rows.append({"reference_id": ref, "site": site, "RMS": stats["RMS"]})
    for row in guards + strong:
        _cached_output(root, row)
    for resume in protocol["resume"].values():
        for row in resume["feature_rows"]:
            audit_feature_row(root, protocol, row, initial=False, _clean_cache=clean_cache)
    guard0 = independent_guard({r["id"]: r for r in guards}, {r["id"]: r for r in guards})
    require(guard0["pass"], "Initial old24 additional stop")
    metric_sources = [name for name, entry in protocol["files"].items()
                      if name.endswith("/metrics.py") and entry["sha256"] ==
                      "3e9a129f41c7b62e2b7cda89b2bf05b7cf8478930b50ce5611a2c17ef741107c"]
    require(metric_sources, "Unmodified historical metrics/gate source retained as reference")
    require(protocol["base515_sha256"] ==
            "a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f",
            "Retained frozen R2 base")
    return {"complete": True, "audit_pass": True, "static_files": len(protocol["files"]),
            "same10_TRAIN_observations": True, "clean_target_arrays_physically_checked": 24,
            "original_guards": 24, "own24_output_parity_bindings": 8,
            "own24_feature_parity_NPZ_bindings": 20, "sampler_rows_per_arm": 216,
            "absolute_update_range": [25, 96], "additional_optimizer_upperbound": 144,
            "backward_upperbound": 432, "resume": resume_report, "fixed_scales": scale_rows,
            "guard0": guard0, "namespace": namespace, "objective_fixtures": objective_fixtures(),
            "source_contract": source_contract(root, protocol),
            "Torch_checkpoint_tensor_Adam_RNG_contents_audited_here": False,
            "Torch_neural_gradient_optimizer_VM_browser_calls": 0, "model_qualified": False}




def _event_audit(root, protocol):
    event_name = "outputs/spatial_continuation_v1/events.jsonl"
    events = _json_lines(root, event_name) if (Path(root) / event_name).is_file() else []
    caps = protocol["max_calls"]
    counts = dict.fromkeys(caps, 0)
    returns = dict.fromkeys(caps, 0)
    pending, reservations, completed, signatures = {}, {}, {}, set()
    extras = []
    sequence = 0
    for position, event in enumerate(events):
        name = event["event"]
        if name == "reserved":
            sequence += 1
            token, kind = event["token"], event["kind"]
            require(type(token) is int and token == sequence and kind in caps,
                    "Reservation token/kind sequence differs")
            counts[kind] += 1
            require(event["count"] == counts[kind] <= caps[kind], "Reservation count/cap differs")
            context = event["context"]
            require(type(context) is dict, "Reservation context required")
            if kind == "prefix":
                require(set(context) == {"id"} and context["id"] in
                        {r["id"] for r in protocol["train_rows"]}, "Unplanned prefix context")
            elif kind in ("adapter_train", "backward", "optimizer"):
                require(context["arm"] in ("U", "S") and type(context["update"]) is int and
                        25 <= context["update"] <= 96, "Unplanned optimization context")
                if kind != "optimizer":
                    require(type(context["slot"]) is int and 1 <= context["slot"] <= 3 and
                            protocol["sampler"][(context["update"] - 25) * 3 + context["slot"] - 1]["id"] ==
                            context["id"], "Optimization sampler association differs")
                if kind == "adapter_train":
                    require(context["site"] in SITES, "Unplanned training adapter site")
            elif kind == "adapter_stats":
                require(context["arm"] in ("U", "S") and context["update"] in (24, 48, 72, 96) and
                        context["id"] in {r["id"] for r in protocol["train_rows"]} and
                        context["site"] in SITES, "Unplanned statistics call")
            elif kind in ("full_forward", "adapter_full"):
                require(context["phase"] in ("parity", "guard", "strong"), "Unplanned full-model phase")
                phase = context["phase"]
                if phase == "parity":
                    require(context["arm"] in ("U", "S") and context["update"] == 24 and
                            context["id"] in protocol["parity_ids"],
                            "Unplanned initial parity call")
                else:
                    cohort = "guards" if phase == "guard" else "strong"
                    require(context["arm"] in ("U", "S") and context["update"] in (GUARD_POINTS if phase == "guard" else STRONG_POINTS) and
                            context["id"] in {r["id"] for r in protocol[cohort]}, "Unplanned output cohort")
                if kind == "adapter_full":
                    parent = context["parent_full_reservation"]
                    require(parent in pending and reservations[parent]["kind"] == "full_forward" and
                            context["site"] in SITES, "Full adapter call outside live parent forward")
                    parent_context = reservations[parent]["context"]
                    require(all(context[k] == parent_context[k] for k in ("arm", "update", "id", "phase")),
                            "Adapter full parent context differs")
            signature = (kind, json.dumps(context, sort_keys=True, separators=(",", ":")))
            require(signature not in signatures, "Duplicate prospective operation context")
            signatures.add(signature)
            reservations[token] = {"kind": kind, "context": context, "position": position,
                                   "signature": signature}
            pending[token] = kind
        elif name == "returned":
            token, kind = event["token"], event["kind"]
            require(token in pending and pending[token] == kind, "Unreserved/repeated operation return")
            del pending[token]
            returns[kind] += 1
            completed[token] = {**reservations[token], "returned_position": position,
                                "returned_context": event["context"]}
        else:
            require(name in ("train_microbatch", "optimizer_commit", "branch_resume"),
                    "Undeclared execution journal event")
            extras.append(event)
    # Full readouts must have exactly four source-bound site calls on success.
    for token, event in completed.items():
        if event["kind"] == "full_forward":
            children = [v for v in completed.values() if v["kind"] == "adapter_full" and
                        v["context"]["parent_full_reservation"] == token]
            sites = [v["context"]["site"] for v in children]
            require(sorted(sites) == sorted(SITES), "Successful full forward lacks four adapter returns")
            require(all(event["position"] < v["position"] < v["returned_position"] < event["returned_position"]
                        for v in children), "Child adapter return escaped parent full-forward interval")
    return {"counts": counts, "returns": returns, "pending": pending,
            "reservations": reservations, "completed": completed, "extras": extras,
            "events": events}


def source_contract(root, protocol):
    prefix = "research/codeformer_spatial_continuation_v1/"
    trees = {}
    for name in ("model.py", "worker.py", "audit.py", "packet.py"):
        relative = prefix + name
        bound_file(root, {"path": relative, **protocol["files"][relative]})
        trees[name] = ast.parse(_safe_file(root, relative).read_text(encoding="utf8"))
        for statement in trees[name].body:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                modules = ([statement.module] if isinstance(statement, ast.ImportFrom)
                           else [alias.name for alias in statement.names])
                require(not any(m and m.split(".")[0] == "torch" for m in modules),
                        "No module-level Torch import")
    functions = {node.name: node for node in trees["model.py"].body if isinstance(node, ast.FunctionDef)}
    objective = ast.unparse(functions["feature_objective"])
    require(all(expression in objective for expression in
                ("z.mean(dim=(2, 3), keepdim=True)", "t.mean(dim=(2, 3), keepdim=True)",
                 "(z - t).abs().mean() / scales[site]", "sum(terms.values()) / 4.0",
                 "scalar / SLOT_DIVISOR")), "Exact unchanged U/S fixed-RMS centering objective")
    require(not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
                    node.func.attr == "detach" for node in ast.walk(functions["feature_objective"])),
            "Differentiable centering/adapter loss cannot detach")
    guard = ast.unparse(functions["manual_runtime_guard"])
    activate = ast.unparse(functions["activate_runtime"])
    require("platform.system() == 'Linux'" in guard and "'TMUX'" in guard and
            "manual_runtime_guard(root, protocol_sha)" in activate and "'NVIDIA L4'" in activate,
            "Exact manual guarded continuation runtime")
    worker = {node.name: node for node in trees["worker.py"].body if isinstance(node, ast.FunctionDef)}
    stats = ast.unparse(worker["_statistics"])
    for formula in ("wanted = b - a", "correction = z[start:end].astype(np.float64) - a",
                    "final = correction - wanted", "wa = wanted - means[1, start:end, None, None]",
                    "ca = correction - means[2, start:end, None, None]", "fa = ca - wa"):
        require(formula in stats, "Unchanged sufficient-statistic formula")
    stops = [node for node in ast.walk(trees["worker.py"]) if isinstance(node, ast.Try) and
             any(isinstance(h.type, ast.Name) and h.type.id == "GuardStop" for h in node.handlers)]
    require(len(stops) == 1 and any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
                node.func.attr == "require" and node.args and "guard_elapsed <= 900" in ast.unparse(node.args[0])
                for statement in stops[0].finalbody for node in ast.walk(statement)),
            "Guard-stop path cannot bypass total output timing cap")
    return {"pass": True, "objective_centering_fixed_scale_and_gradient_path_checked": True,
            "statistics_formula_and_global_resource_stop_checked": True,
            "manual_Linux_tmux_L4_source_guard_checked": True,
            "no_neural_execution": True, "source_not_checkpoint_content_or_activation_replay": True}


def _logical_stats_parity(root, actual, expected):
    a = read_statistics_npz(root, actual["arrays"])
    b = read_statistics_npz(root, expected["arrays"])
    require(set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a),
            "Own24 logical NPZ array parity differs; ZIP metadata is not a numerical claim")
    require(actual["site_metrics"] == expected["site_metrics"] and
            actual["adapter_state_sha256"] == expected["adapter_state_sha256"] and
            actual["scales"] == expected["scales"], "Own24 scalar/state/RMS parity differs")
    return True


def audit_return(root, protocol, result=None, *, reference_root=None):
    """Prospective pure return audit; actual checkpoint contents need separate audit."""
    root = Path(root).resolve()
    full_stage = all((root / name).is_file() for name in protocol["vm_only_static"])
    returned = not full_stage
    namespace = _closed_namespace(root, protocol, returned=returned)
    for name, value in protocol["files"].items():
        if (root / name).is_file():
            bound_file(root, {"path": name, **value})
    require(reference_root is not None or full_stage, "Explicit complete reference root for VM-only targets")
    refs = Path(root if reference_root is None else reference_root).resolve()
    targets = []
    for entry in protocol["clean_features"].values():
        for binding in entry["features"].values():
            bound_file(refs, binding)
            targets.append(binding)
    _parent_static(root, protocol)
    supervisor = read_json(_safe_file(root, "supervisor.json"))
    require(supervisor["complete"] is True and type(supervisor["worker_exit"]) is int and
            type(supervisor["timeout"]) is bool and supervisor["worker_terminal"] is True and
            supervisor["owned_child_group_terminal"] is True and
            supervisor["training_success_not_implied"] is True and
            0 <= _finite_number(supervisor["seconds"]) <= protocol["limits"]["supervisor_seconds"] +
            protocol["limits"]["whole_deadline_cleanup_seconds"],
            "Successful terminal owned-child supervisor evidence")
    export = _export_manifest(root, protocol, namespace, supervisor)
    ledger = _event_audit(root, protocol)
    resource_name = O + "resources.jsonl"
    resources = _json_lines(root, resource_name) if (root / resource_name).is_file() else []
    samples = [r for r in resources if r.get("context") != "projection"]
    require([r["sample"] for r in samples] == list(range(1, len(samples) + 1)), "Ordered resource samples")
    limits = protocol["limits"]
    # Linux /proc VmRSS and ru_maxrss are separate accounting samples; neither
    # is asserted to order the other. Each retains the exact same10GiB cap.
    resource_pass = bool(samples) and all(
        0 <= _finite_number(r["seconds"]) <= limits["worker_seconds"] and
        0 < _finite_number(r["RSS_bytes"]) <= limits["RSS_bytes"] and
        0 < _finite_number(r["peak_RSS_bytes"]) <= limits["RSS_bytes"] and
        0 <= _finite_number(r["CUDA_reserved_bytes"]) <= limits["CUDA_reserved_bytes"] and
        _finite_number(r["free_bytes"]) >= limits["disk_reserve_bytes"] for r in samples)
    for row in resources:
        if row.get("context") == "projection":
            require(0 < row["completed"] <= row["total"] and row["seconds"] > 0 and row["cap_seconds"] > 0,
                    "Finite resource projection")
            _near(row["projected_seconds"], row["seconds"] / row["completed"] * row["total"], "Projection")
    resource_summary = {"samples": len(samples), "within_declared_limits": resource_pass,
                        "maximum_proc_VmRSS_bytes": max((r["RSS_bytes"] for r in samples), default=None),
                        "maximum_ru_maxrss_bytes": max((r["peak_RSS_bytes"] for r in samples), default=None),
                        "independent_RSS_bounds_no_cross_source_order_assumption": True}
    result_path = root / O / "result.json"
    if result is None and result_path.is_file():
        result = read_json(result_path)
    failure_path = root / O / "failure.json"
    if result is None:
        require(failure_path.is_file() and supervisor["worker_exit"] != 0,
                "Infrastructure partial evidence requires explicit failure")
        failure = read_json(failure_path)
        require(failure["partial_evidence_preserved"] is True and
                failure["automatic_resume_authorized"] is False and failure["model_qualified"] is False and
                failure["counts"] == ledger["counts"] and failure["returns"] == ledger["returns"] and
                failure["pending"] == {str(k): v for k, v in ledger["pending"].items()},
                "Infrastructure failure/counter closure")
        return {"complete": True, "audit_pass": True, "terminal_kind": "infrastructure_failure_partial_evidence",
                "counts": ledger["counts"], "returns": ledger["returns"], "pending": failure["pending"],
                "resources": resource_summary, "namespace": namespace, "export": export,
                "partial_feature_and_pixel_readouts_fully_audited": False,
                "Torch_checkpoint_tensor_Adam_RNG_contents_audited": False,
                "training_success_not_implied": True, "model_qualified": False,
                "Torch_neural_gradient_optimizer_VM_browser_calls": 0}
    require(not failure_path.exists() and supervisor["worker_exit"] == 0 and not supervisor["timeout"] and
            resource_pass and result_path.is_file() and read_json(result_path) == result,
            "Completed continuation requires physical successful bounded result")
    protocol_sha = hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest()
    require(result["complete"] is True and result["protocol_sha256"] == protocol_sha and
            result["counts"] == ledger["counts"] == ledger["returns"] == result["returns"] and
            not result["pending"] and not ledger["pending"] and
            result["base515_sha256"] == protocol["base515_sha256"], "Terminal operation/state closure")
    for key, expected in {"no_DEV_FINAL_native_optimized": True, "no_automatic_model_selection": True,
                          "model_qualified": False, "face_visual_review_pending": True,
                          "full478_development_gates_not_replaced": True}.items():
        require(result[key] == expected, "Research boundary changed: " + key)
    require(0 <= result["optimization_seconds"] <= 900 and
            0 <= result["guard_probe_seconds"] <= 900 and
            result["guard_probe_seconds_include_initial8_parity"] is True and
            0 <= result["parity_seconds"] <= result["guard_probe_seconds"],
            "Finite measured stages include all8 own24 parity outputs")
    expected_runtime = {O + n for n in ("started.json", "preflight.json", "initial.json", "resume_parity.json", "result.json",
                                       "events.jsonl", "resources.jsonl")}
    actual_calls = {kind: set() for kind in CAPS}
    key = lambda context: json.dumps(context, sort_keys=True, separators=(",", ":"))
    completed = ledger["completed"]
    extras = ledger["extras"]
    micros = [r for r in extras if r["event"] == "train_microbatch"]
    commits = [r for r in extras if r["event"] == "optimizer_commit"]
    resume_events = [r for r in extras if r["event"] == "branch_resume"]
    require(len(micros) == ledger["returns"]["backward"] and len(commits) == ledger["returns"]["optimizer"],
            "Exact returned optimizer commits and micros")
    for row in protocol["train_rows"]:
        actual_calls["prefix"].add(key({"id": row["id"]}))
    clean_cache, feature_rows, own24_rows = {}, [], {}
    for arm in ("U", "S"):
        resume = protocol["resume"][arm]
        own = {r["id"]: r for r in resume["parity_rows"]}
        for identifier in protocol["parity_ids"]:
            stem = O + "parity/" + arm + "/" + identifier
            row = read_json(_safe_file(root, stem + ".json"))
            _audit_output_row(root, own[identifier], row, arm=arm, update=24, phase="parity")
            require(row["resume_checkpoint"] == resume["checkpoint"], "Own24 full parity source checkpoint")
            expected_runtime.update(stem + ext for ext in (".json", ".npy", ".png"))
            actual_calls["full_forward"].add(key({"arm": arm, "update": 24, "id": identifier, "phase": "parity"}))
        expected_features = {r["id"]: r for r in resume["feature_rows"]}
        for case in protocol["train_rows"]:
            stem = O + f"stats/{arm}/u0024/" + case["id"]
            row = read_json(_safe_file(root, stem + ".json"))
            require(row["arm"] == arm and row["update"] == 24 and row["arrays"]["path"] == stem + ".npz",
                    "Both initial own24 feature readout sets")
            _logical_stats_parity(root, row, expected_features[row["id"]])
            parity = row["resume_parity"]
            require(parity["source_checkpoint"] == resume["checkpoint"] and
                    parity["expected_adapter_sha256"] == resume["adapter_sha256"] and
                    parity["reference_stats"] == expected_features[row["id"]]["arrays"] and
                    parity["arrays_exact"] is True and parity["metrics_exact"] is True and
                    parity["ZIP_container_bytes_not_required_equal"] is True,
                    "Initial own24 feature parity source/state receipt")
            feature_rows.append(audit_feature_row(root, protocol, row, reference_root=refs,
                                                 initial=False, _clean_cache=clean_cache))
            own24_rows[arm, row["id"]] = row
            expected_runtime.update((stem + ".json", stem + ".npz"))
            for site in SITES:
                actual_calls["adapter_stats"].add(key({"arm": arm, "update": 24,
                                                       "id": case["id"], "site": site}))
    initial_end = max(r["returned_position"] for r in completed.values()
                      if r["kind"] in ("prefix", "full_forward", "adapter_stats") and
                      (r["kind"] == "prefix" or r["context"]["update"] == 24))
    require(all(r["position"] > initial_end for r in completed.values()
                if r["kind"] in ("adapter_train", "backward", "optimizer")),
            "ANY optimization before BOTH own24 output and feature parity completed")
    branches = result["branches"]
    require(len(branches) == 2 and [b["arm"] for b in branches] == ["U", "S"], "Both original arms")
    reports, gates = [], []
    by_id = {r["id"]: r for r in protocol["train_rows"]}
    guard0 = {r["id"]: r for r in protocol["guards"]}
    for branch in branches:
        arm, end = branch["arm"], branch["completed_updates"]
        require(end in GUARD_POINTS and branch["additional_updates"] == end - 24 and
                branch["terminal_status"] in ("complete", "guard_stop") and
                branch["model_qualified"] is False and branch["automatic_candidate_selection"] is False and
                read_json(_safe_file(root, O + "branches/" + arm + ".json")) == branch,
                "Finite absolute24to96 terminal branch")
        expected_runtime.add(O + "branches/" + arm + ".json")
        resume = protocol["resume"][arm]
        arm_commits = [r for r in commits if r["arm"] == arm]
        arm_micros = [r for r in micros if r["arm"] == arm]
        require([r["update"] for r in arm_commits] == list(range(25, end + 1)) and
                len(arm_micros) == (end - 24) * 3, "Additional72 maximum, absolute updates not reset")
        state = resume["adapter_sha256"]
        for commit in arm_commits:
            update = commit["update"]
            require(commit["adapter_before_sha256"] == state and commit["all515_frozen_exact"] is True and
                    commit["S_expected_null_up_biases"] == (arm == "S") and
                    commit["all24_tensors_changed_required"] is False and commit["gradient_clip_norm"] == 1.,
                    "Own-state continuation chain and frozen base")
            state = commit["adapter_after_sha256"]
            _near(commit["LR"], 1e-4, "No restarted warmup at25", atol=0, rtol=0)
            require(_finite_number(commit["preclip_norm"]) >= 0, "Finite preclip norm")
            subset = [r for r in arm_micros if r["update"] == update]
            require([r["slot"] for r in subset] == [1, 2, 3], "Exact3 continuation micros")
            for micro in subset:
                context = {"arm": arm, "update": update, "slot": micro["slot"], "id": micro["id"]}
                require(micro["id"] == protocol["sampler"][(update - 25) * 3 + micro["slot"] - 1]["id"] and
                        micro["adapter_state_before_update_sha256"] == commit["adapter_before_sha256"] and
                        micro["clean_targets"] == protocol["clean_features"][by_id[micro["id"]]["reference_id"]]["features"] and
                        micro["scales"] == protocol["clean_features"][by_id[micro["id"]]["reference_id"]]["scales"] and
                        set(micro["site_losses"]) == set(SITES) and min(micro["site_losses"].values()) >= 0,
                        "Same216 sampler/state/clean-target/scale joins")
                _near(micro["LR"], 1e-4, "Constant continuation LR", atol=0, rtol=0)
                _near(micro["scalar"], sum(micro["site_losses"].values()) / 4,
                      "FP32 site mean", atol=5e-6, rtol=1e-6)
                _near(micro["divided_scalar"], micro["scalar"] / 3,
                      "FP32 accumulated three micros", atol=5e-6, rtol=1e-6)
                actual_calls["backward"].add(key(context))
                for site in SITES:
                    actual_calls["adapter_train"].add(key({**context, "site": site}))
            actual_calls["optimizer"].add(key({"arm": arm, "update": update}))
        reached = [u for u in GUARD_POINTS if u <= end]
        stat_points = [u for u in STRONG_POINTS if u <= end]
        checkpoints = []
        for update in (24, *reached):
            name = O + f"checkpoints/{arm}/u{update:04d}.pt"
            physical = pin(root, name)
            require(physical["bytes"] <= limits["checkpoint_bytes"], "Bounded adapter+Adam checkpoint")
            expected_runtime.add(name)
            checkpoints.append(physical)
        require(branch["last_checkpoint"] == checkpoints[-1], "Saved checkpoint before every guard including failure")
        snapshots = branch["feature_statistics"]
        require([row["update"] for row in snapshots] == [24, *stat_points],
                "Initial own24 and all48/72/96 reached feature snapshots")
        require(snapshots[0]["rows"] == [own24_rows[arm, row["id"]] for row in protocol["train_rows"]],
                "Embedded branch initial24 rows equal actual parity sidecars")
        for snapshot in snapshots[1:]:
            update = snapshot["update"]
            require([r["id"] for r in snapshot["rows"]] == [r["id"] for r in protocol["train_rows"]],
                    "All10 feature cases per scheduled snapshot")
            expected_state = next(r["adapter_after_sha256"] for r in arm_commits if r["update"] == update)
            for row in snapshot["rows"]:
                stem = O + f"stats/{arm}/u{update:04d}/" + row["id"]
                require(row == read_json(_safe_file(root, stem + ".json")) and row["arm"] == arm and
                        row["update"] == update and row["adapter_state_sha256"] == expected_state and
                        row["arrays"]["path"] == stem + ".npz", "Scheduled feature state/array join")
                feature_rows.append(audit_feature_row(root, protocol, row, reference_root=refs,
                                                     initial=False, _clean_cache=clean_cache))
                expected_runtime.update((stem + ".json", stem + ".npz"))
                for site in SITES:
                    actual_calls["adapter_stats"].add(key({"arm": arm, "update": update,
                                                           "id": row["id"], "site": site}))
        failed = None
        for update in reached:
            stem = O + f"guards/{arm}/u{update:04d}/"
            bank = _bank(root, stem + "bank.json", protocol["guards"], arm=arm, update=update, phase="guard")
            gate = independent_guard(bank, guard0)
            require(read_json(_safe_file(root, stem + "gates.json")) == gate, "Exact ordered VM guard0 thresholds")
            gates.append({"arm": arm, "update": update, "gate": gate})
            expected_runtime.update((stem + "bank.json", stem + "gates.json"))
            for identifier in bank:
                expected_runtime.update(stem + identifier + ext for ext in (".json", ".npy", ".png"))
                actual_calls["full_forward"].add(key({"arm": arm, "update": update,
                                                      "id": identifier, "phase": "guard"}))
            if not gate["pass"]:
                failed = update
                require(update == reached[-1], "Guard-stopped branch continued")
            elif update in STRONG_POINTS:
                stem = O + f"strong/{arm}/u{update:04d}/"
                bank = _bank(root, stem + "bank.json", protocol["strong"], arm=arm, update=update, phase="strong")
                expected_runtime.add(stem + "bank.json")
                for identifier in bank:
                    expected_runtime.update(stem + identifier + ext for ext in (".json", ".npy", ".png"))
                    actual_calls["full_forward"].add(key({"arm": arm, "update": update,
                                                          "id": identifier, "phase": "strong"}))
        require(branch["guard_failed_update"] == failed and
                branch["terminal_status"] == ("guard_stop" if failed is not None else "complete") and
                (failed is not None or end == 96), "Terminal branch must honor first guard stop")
        expected_delta = {"prefix": 0, "adapter_train": (end - 24) * 3 * 4,
                          "adapter_stats": len(stat_points) * 10 * 4,
                          "full_forward": len(reached) * 24 + 4 * (len(stat_points) - int(failed in STRONG_POINTS)),
                          "adapter_full": 4 * (len(reached) * 24 + 4 * (len(stat_points) - int(failed in STRONG_POINTS))),
                          "backward": (end - 24) * 3, "optimizer": end - 24}
        require({k: branch["end_counts"][k] - branch["start_counts"][k] for k in CAPS} == expected_delta and
                branch["returns"] == branch["end_counts"], "Branch exact operation delta closure")
        reports.append({"arm": arm, "absolute_completed_updates": end, "additional_updates": end - 24,
                        "status": branch["terminal_status"], "guard_failed_update": failed,
                        "checkpoints_byte_bound_only": checkpoints, "operation_delta": expected_delta})
    initial_counts = {"prefix": 10, "adapter_train": 0, "adapter_stats": 80,
                      "full_forward": 8, "adapter_full": 32, "backward": 0, "optimizer": 0}
    require(branches[0]["start_counts"] == initial_counts and
            branches[0]["end_counts"] == branches[1]["start_counts"] and
            branches[1]["end_counts"] == result["counts"], "Initial parity and cumulative branch count chain")
    for kind, expected in actual_calls.items():
        if kind != "adapter_full":
            require({key(r["context"]) for r in completed.values() if r["kind"] == kind} == expected,
                    "Actual operations differ from reached plan: " + kind)
    require(len(resume_events) == 2 and [r["arm"] for r in resume_events] == ["U", "S"],
            "Two durable own24 branch restore events")
    for event in resume_events:
        resume = protocol["resume"][event["arm"]]
        require(event["source_checkpoint"] == resume["checkpoint"] and
                event["resume_adapter_sha256"] == resume["adapter_sha256"] and
                event["starting_absolute_update"] == 24 and event["own24_Adam_restored"] is True and
                event["fresh_Adam_empty"] is False and event["no_warmup_restart"] is True and
                event["same216_sampler_rows"] is True and
                event["base515_sha256"] == protocol["base515_sha256"],
                "Own Adam24/RNG restore provenance; no fresh restart")
        branch = next(b for b in branches if b["arm"] == event["arm"])
        require(all(event[k] == v for k, v in branch["resume"].items()), "Durable branch resume join")
    initial = read_json(_safe_file(root, O + "initial.json"))
    parity = read_json(_safe_file(root, O + "resume_parity.json"))
    require(initial["complete"] is True and initial["base515_sha256"] == protocol["base515_sha256"] and
            initial["identity_factory_not_training_start"] is True and initial["no_parent_VM_dependency"] is True and
            parity["complete"] is True and parity["both_arms_own24_FULL_RAW_PNG_upstream_exact"] is True and
            parity["both_arms_all10_logical_feature_arrays_metrics_exact"] is True and
            parity["optimizer_updates"] == parity["backwards"] == 0 and
            parity["source_states"] == initial["own24_resume_proofs"], "Both own24 pre-optimization parity receipt")
    for arm in ("U", "S"):
        proof = initial["own24_resume_proofs"][arm]
        event = next(e for e in resume_events if e["arm"] == arm)
        require(proof["source_checkpoint"] == protocol["resume"][arm]["checkpoint"] and
                proof["adapter_sha256"] == protocol["resume"][arm]["adapter_sha256"] and
                proof["completed_updates"] == 24 and len(proof["optimizer_steps"]) == 24 and
                set(proof["optimizer_steps"].values()) == {24.} and
                proof["optimizer_tree_sha256"] == event["optimizer_tree_sha256"] and
                proof["RNG_tree_sha256"] == event["RNG_tree_sha256"], "Exact source-owned Adam/RNG restore trees")
    require({name for name in namespace["files"] if name.startswith(O)} == expected_runtime,
            "Closed reached/stopped continuation runtime namespace")
    cache = result["observed_VM_only_cache"]
    require(set(cache) == set(by_id), "Original10 fresh prefix cache bindings")
    for identifier, entries in cache.items():
        require(set(entries) == set(SITES), "Forty observed feature cache bindings")
        for site, binding in entries.items():
            require(binding["path"] == f"cache/observed/{identifier}/encoder{site}.npy", "VM-only cache scope")
            if not returned:
                read_npy(root, binding, (1, CHANNELS[site], int(site), int(site)), "float32")
    require(all(row["observed_cache"] == cache[row["id"]] for row in micros), "Observed cache training source join")
    return {"complete": True, "audit_pass": True, "terminal_kind": "finite_two_arm_own24_continuation",
            "protocol_sha256": protocol_sha, "counts": ledger["counts"], "returns": ledger["returns"],
            "branches": reports, "guards": gates, "feature_case_readouts": len(feature_rows),
            "feature_site_readouts": len(feature_rows) * 4, "feature_NPZ_arrays": len(feature_rows) * 12,
            "all_initial_own24_logical_feature_and_output_parity_exact": True,
            "resources": resource_summary, "namespace": namespace, "export": export,
            "explicit_reference_root": reference_root is not None,
            "physically_reread_clean_feature_arrays": targets, "source_contract": source_contract(root, protocol),
            "Torch_checkpoint_tensor_Adam_RNG_contents_audited": False,
            "checkpoint_content_audit_still_required": True, "full_activation_replay": False,
            "source_absolute_moments_not_independently_recomputed_from_full_activations": False,
            "independent_visual_FULL478_DEV_FINAL_app_user_acceptance_still_required": True,
            "training_success_not_implied": True, "model_qualified": False,
            "Torch_neural_gradient_optimizer_VM_browser_calls": 0}

