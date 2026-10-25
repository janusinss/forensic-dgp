"""Independent, non-neural audits for the prospective paired objective screen.

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
                    mean_limit = max(.01, sum(start.values()) / 8)
                mean = sum(measured.values()) / 8
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
    return {"pass": True, "fixtures": ["DC-only U/S separation", "non-square3x5 spatial axes",
            "per-example/channel DC nullspace finite difference", "same fixed total-clean RMS",
            "identity error zero", "guard mean and exact-case failures", "known MAE/MSE/PSNR",
            "RAW-PNG distinction", "native-unpaired clean scores null",
            "dark behavior-anchor clean scores null separately from native CCTV",
            "JSON overflow/nonfinite/duplicate rejection"],
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


def audit_static(root, protocol):
    """Audit declared static assets without importing worker/model/Torch code."""
    root = Path(root).resolve()
    require(type(protocol) is dict and type(protocol.get("files")) is dict,
            "Sealed static files map required")
    assets = []
    for name, expected in sorted(protocol["files"].items()):
        require(set(expected) == {"bytes", "sha256"}, "Closed file binding fields")
        binding = {"path": name, **expected}
        bound_file(root, binding)
        assets.append(binding)
    rows = protocol["train_rows"]
    require(len(rows) == 10 and len({r["id"] for r in rows}) == 10,
            "Exact10 distinct training observations")
    require(set(r["reference_id"] for r in rows) == set(REFS) and
            all(r["split"] == "TRAIN" and r["input_only_decision"] == "readable" for r in rows),
            "Six readable TRAIN photographs only")
    moderate = [r for r in rows if "gaussian_2p4" in r["profile"]]
    strong = [r for r in rows if r["profile"] == "strong_fixed_blur3p2_lr64_jpeg50"]
    require(len(moderate) == 6 and set(r["reference_id"] for r in moderate) == set(REFS) and
            len(strong) == 4 and set(r["reference_id"] for r in strong) == set(READABLE_STRONG_REFS),
            "Exact6moderate/4readable strong; borderline2 excluded")
    by_id = {r["id"]: r for r in rows}
    sampler = protocol["sampler"]
    require(type(sampler) is list and len(sampler) == 72, "Same72 sampler rows required")
    ids = [r["id"] for r in rows]
    for index, row in enumerate(sampler):
        require(row["id"] == ids[index % 10] and
                row["update"] == index // 3 + 1 and row["slot"] == index % 3 + 1,
                "Frozen72 row update/slot/id schedule differs")
    require(collections.Counter(r["id"] for r in sampler) ==
            dict(zip(ids, [8, 8] + [7] * 8)), "Exact per-input repeated exposure counts")
    guards = protocol["guards"]
    require(len(guards) == 24 and len({r["id"] for r in guards}) == 24 and
            collections.Counter(r["role"] for r in guards) == {"filter": 8, "clear": 8, "dark": 8} and
            all(r["split"] == "TRAIN" for r in guards), "Exact24 old TRAIN guards required")
    probes = protocol["strong"]
    require(len(probes) == 4 and {r["id"] for r in probes} == {r["id"] for r in strong},
            "All four optimized strong inputs require full-output probes")
    for row in probes:
        require(row["input"] == by_id[row["id"]]["input"] and
                row["target"] == by_id[row["id"]]["target"] and row["split"] == "TRAIN",
                "Strong probe fitting input/target association differs")
    for row in rows + guards + probes:
        pixels = rgb(root, row["input"])
        require(hashlib.sha256(pixels.tobytes()).hexdigest() == row["input_RGB8_sha256"],
                "Decoded RGB8 input association differs")
        if row["target"] is None:
            require(row.get("role") == "dark" and row.get("behavior_teacher") is not None and
                    row.get("target_RGB8_sha256") is None,
                    "Only dark behavior anchors may omit photographic target/hash")
        else:
            target = rgb(root, row["target"])
            require(hashlib.sha256(target.tobytes()).hexdigest() == row["target_RGB8_sha256"],
                    "Decoded RGB8 paired-target association differs")
    require(len(protocol["parity_ids"]) == 4 and len(set(protocol["parity_ids"])) == 4 and
            set(protocol["parity_ids"]).issubset({r["id"] for r in guards}),
            "Four existing guard parity witnesses required")
    clean = protocol["clean_features"]
    require(set(clean) == set(REFS), "Six exact clean-feature references")
    scales = {}
    for reference, entry in clean.items():
        require(set(entry["features"]) == set(entry["scales"]) == set(SITES),
                "All24 clean features/scales required")
        scales[reference] = {}
        for site in SITES:
            stats = _clean_statistics(root, entry["features"][site], site)
            _near(entry["scales"][site], stats["RMS"], "Original total clean RMS", atol=1e-9, rtol=1e-9)
            scales[reference][site] = stats["RMS"]
    for row in guards + probes:
        _cached_output(root, row)
    initial_guard = {r["id"]: r for r in guards}
    guard0 = independent_guard(initial_guard, initial_guard)
    require(guard0["pass"], "Guard0 additional stop required")
    # A source-pinned old implementation is a cross-check, not the independent formula.
    metric_names = [n for n in protocol["files"] if n.endswith("/metrics.py")]
    require(len(metric_names) == 1, "One exact retained non-neural gate implementation")
    metric_name = metric_names[0]
    require(protocol["files"][metric_name]["sha256"] ==
            "3e9a129f41c7b62e2b7cda89b2bf05b7cf8478930b50ce5611a2c17ef741107c",
            "Retained gate metrics source changed")
    spec = importlib.util.spec_from_file_location("_spatial_old_metrics_pure", root / metric_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _compare(module.guard_interim(initial_guard, initial_guard), guard0, "Independent guard0 cross-check")
    caps = protocol["max_calls"]
    require(caps == {"prefix": 10, "adapter_train": 576, "adapter_stats": 240,
                     "full_forward": 172, "adapter_full": 688, "backward": 144, "optimizer": 48},
            "Finite36? No: exact10prefix/172full/1504adapter/48update scope required")
    require(protocol["initial_adapter_sha256"] ==
            "20398c0179f55436cc03333ed439c3f61e9a33e026274af548423e2753fd6b4b" and
            protocol["base515_sha256"] ==
            "a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f",
            "Fresh identity adapter / frozen R2 source contract")
    namespace = _closed_namespace(root, protocol, returned=False)
    return {"complete": True, "audit_pass": True, "static_files": len(assets),
            "TRAIN_observations": 10, "photographs": 6, "moderate": 6, "readable_strong": 4,
            "clean_feature_arrays": 24, "old_TRAIN_guards": 24, "strong_output_probes": 4,
            "sampler_rows_per_arm": 72, "unchanged_total_clean_RMS": scales,
            "guard0": guard0, "objective_fixtures": objective_fixtures(),
            "source_contract": source_contract(root, protocol), "namespace": namespace,
            "Torch_pickle_contents_audited": False,
            "Torch_neural_gradient_optimizer_VM_browser_calls": 0,
            "model_qualified": False}


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


def _event_audit(root, protocol):
    event_name = "outputs/spatial_objectives_v1/events.jsonl"
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
                        1 <= context["update"] <= 24, "Unplanned optimization context")
                if kind != "optimizer":
                    require(type(context["slot"]) is int and 1 <= context["slot"] <= 3 and
                            protocol["sampler"][(context["update"] - 1) * 3 + context["slot"] - 1]["id"] ==
                            context["id"], "Optimization sampler association differs")
                if kind == "adapter_train":
                    require(context["site"] in SITES, "Unplanned training adapter site")
            elif kind == "adapter_stats":
                require(context["arm"] in ("U", "S") and context["update"] in (6, 12, 24) and
                        context["id"] in {r["id"] for r in protocol["train_rows"]} and
                        context["site"] in SITES, "Unplanned statistics call")
            elif kind in ("full_forward", "adapter_full"):
                require(context["phase"] in ("parity", "guard", "strong"), "Unplanned full-model phase")
                phase = context["phase"]
                if phase == "parity":
                    require(context["update"] == 0 and context["id"] in protocol["parity_ids"],
                            "Unplanned initial parity call")
                else:
                    cohort = "guards" if phase == "guard" else "strong"
                    require(context["arm"] in ("U", "S") and context["update"] in (6, 12, 24) and
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
            require(name in ("train_microbatch", "optimizer_commit", "branch_reset"),
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
    """Critical formula/guarded-import AST checks; not neural execution."""
    prefix = "research/codeformer_spatial_objectives_v1/"
    files = {}
    for name in ("model.py", "worker.py", "audit.py", "packet.py"):
        relative = prefix + name
        require(relative in protocol["files"], "All four implementation sources bound")
        bound_file(root, {"path": relative, **protocol["files"][relative]})
        files[name] = ast.parse(_safe_file(root, relative).read_text(encoding="utf8"))
    for name, tree in files.items():
        for statement in tree.body:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                modules = ([statement.module] if isinstance(statement, ast.ImportFrom)
                           else [alias.name for alias in statement.names])
                require(not any(x and x.split(".")[0] == "torch" for x in modules),
                        "No module-level Torch import permitted")
    model = {node.name: node for node in files["model.py"].body if isinstance(node, ast.FunctionDef)}
    objective = ast.unparse(model["feature_objective"])
    require("z.mean(dim=(2, 3), keepdim=True)" in objective and
            "t.mean(dim=(2, 3), keepdim=True)" in objective and
            "(z - t).abs().mean() / scales[site]" in objective and
            "sum(terms.values()) / 4.0" in objective and "scalar / SLOT_DIVISOR" in objective,
            "Critical U/S axes/fixed-scale/site-average/microbatch formula differs")
    require(not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
                    node.func.attr == "detach" for node in ast.walk(model["feature_objective"])),
            "Objective may not detach centering or optimizing path")
    activate = ast.unparse(model["activate_runtime"])
    guard = ast.unparse(model["manual_runtime_guard"])
    require("platform.system() == 'Linux'" in guard and "'TMUX'" in guard and
            "manual_runtime_guard(root, protocol_sha)" in activate and
            "'NVIDIA L4'" in activate and "protocol_sha" in activate and
            "import torch" in activate, "Manual exact Linux/tmux/L4 activation guard differs")
    worker = {node.name: node for node in files["worker.py"].body if isinstance(node, ast.FunctionDef)}
    statistic = ast.unparse(worker["_statistics"])
    for formula in ("wanted = b - a", "correction = z[start:end].astype(np.float64) - a",
                    "final = correction - wanted", "wa = wanted - means[1, start:end, None, None]",
                    "ca = correction - means[2, start:end, None, None]", "fa = ca - wa"):
        require(formula in statistic, "Source sufficient-statistic formula differs: " + formula)
    require("moments[index, start:end] = op().mean(axis=(1, 2))" in statistic,
            "Saved per-channel spatial moments differ")
    return {"pass": True, "critical_U_S_axes_scales_divisors_checked": True,
            "centering_not_detached": True, "moment_definitions_source_checked": True,
            "runtime_import_guard_source_checked": True, "neural_execution": False,
            "not_a_proof_of_numerical_Torch_execution": True}


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
                row["initial_cached_RAW_PNG_upstream_exact"] is True, "Exact cached CUDA parity differs")
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
    require(receipt["format"] == "codeformer-spatial-objectives-export-manifest-v1" and
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


def audit_return(root, protocol, result=None, *, reference_root=None):
    """Audit actual returned arrays/journals. Never train or load a Torch pickle.

    ``reference_root`` must explicitly name the retained complete sealed stage
    when VM-only clean targets are absent from the portable return. The original
    protocol hashes still bind those physically re-read reference arrays.
    """
    root = Path(root).resolve()
    output = "outputs/spatial_objectives_v1/"
    full_stage = all((root / name).is_file() for name in protocol["vm_only_static"])
    returned = not full_stage
    namespace = _closed_namespace(root, protocol, returned=returned)
    for name, value in protocol["files"].items():
        if (root / name).is_file():
            bound_file(root, {"path": name, **value})
    require(reference_root is not None or full_stage,
            "Explicit retained complete reference root required for omitted clean targets")
    refs = Path(root if reference_root is None else reference_root).resolve()
    reference_pins = []
    for entry in protocol["clean_features"].values():
        for binding in entry["features"].values():
            bound_file(refs, binding)
            reference_pins.append(binding)
    supervisor = read_json(_safe_file(root, "supervisor.json"))
    require(supervisor["complete"] is True and type(supervisor["worker_exit"]) is int and
            type(supervisor["timeout"]) is bool and supervisor["training_success_not_implied"] is True and
            0 <= _finite_number(supervisor["seconds"]) <= protocol["limits"]["supervisor_seconds"] + 10,
            "Terminal bounded supervisor receipt required")
    export = _export_manifest(root, protocol, namespace, supervisor)
    ledger = _event_audit(root, protocol)
    resource_name = output + "resources.jsonl"
    resource_rows = (_json_lines(root, resource_name) if (root / resource_name).is_file() else [])
    samples = [r for r in resource_rows if r.get("context") != "projection"]
    require([r["sample"] for r in samples] == list(range(1, len(samples) + 1)),
            "Ordered resource self-samples required")
    limits = protocol["limits"]
    resource_pass = bool(samples) and all(0 <= _finite_number(r["seconds"]) <= limits["worker_seconds"] and
                        0 < _finite_number(r["RSS_bytes"]) <= r["peak_RSS_bytes"] <= limits["RSS_bytes"] and
                        0 <= _finite_number(r["CUDA_reserved_bytes"]) <= limits["CUDA_reserved_bytes"] and
                        _finite_number(r["free_bytes"]) >= limits["disk_reserve_bytes"] for r in samples)
    projections = [r for r in resource_rows if r.get("context") == "projection"]
    for row in projections:
        require(0 < row["completed"] <= row["total"] and row["seconds"] > 0 and row["cap_seconds"] > 0,
                "Finite timing projection required")
        _near(row["projected_seconds"], row["seconds"] / row["completed"] * row["total"], "Timing projection")
    resource_report = {"samples": len(samples), "within_declared_limits": resource_pass,
                       "peak_self_RSS_bytes": max((r["peak_RSS_bytes"] for r in samples), default=None),
                       "minimum_free_bytes": min((r["free_bytes"] for r in samples), default=None),
                       "maximum_CUDA_reserved_bytes": max((r["CUDA_reserved_bytes"] for r in samples), default=None),
                       "source_bound_self_samples_not_independent_live_measurement": True}
    result_path = root / output / "result.json"
    if result is None and result_path.is_file():
        result = read_json(result_path)
    failure_path = root / output / "failure.json"
    if result is None:
        require(failure_path.is_file() and supervisor["worker_exit"] != 0,
                "Missing terminal result requires explicit infrastructure failure")
        failure = read_json(failure_path)
        require(failure["partial_evidence_preserved"] is True and
                failure["automatic_resume_authorized"] is False and failure["model_qualified"] is False and
                failure["counts"] == ledger["counts"] and failure["returns"] == ledger["returns"] and
                {str(k): v for k, v in ledger["pending"].items()} == failure["pending"],
                "Partial failure/counter receipt differs")
        return {"complete": True, "audit_pass": True, "terminal_kind": "infrastructure_failure_partial_evidence",
                "quality_pass": False, "training_success_not_implied": True,
                "counts": ledger["counts"], "returns": ledger["returns"],
                "pending": failure["pending"], "resources": resource_report, "namespace": namespace,
                "export": export,
                "partial_feature_pixel_contents_fully_audited": False,
                "checkpoint_Torch_tensor_Adam_RNG_contents_audited": False,
                "reference_root_explicit": reference_root is not None,
                "Torch_neural_gradient_optimizer_VM_browser_calls": 0, "model_qualified": False}
    require(not failure_path.exists() and supervisor["worker_exit"] == 0 and not supervisor["timeout"] and
            resource_pass and result_path.is_file() and read_json(result_path) == result,
            "Completed screen requires exact physical result and successful bounded supervisor")
    protocol_sha = hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest()
    require(result["complete"] is True and result["protocol_sha256"] == protocol_sha and
            result["base515_sha256"] == protocol["base515_sha256"] and
            result["counts"] == ledger["counts"] == ledger["returns"] == result["returns"] and
            not result["pending"] and not ledger["pending"], "Terminal protocol/base/counter closure")
    for key, expected in {"no_DEV_FINAL_native_optimized": True, "no_automatic_model_selection": True,
                          "model_qualified": False, "face_visual_review_pending": True,
                          "full478_development_gates_not_replaced": True,
                          "training_updates_maximum": 48}.items():
        require(result[key] == expected, "Research status boundary differs: " + key)
    require(0 <= result["optimization_seconds"] <= 900 and 0 <= result["guard_probe_seconds"] <= 900,
            "Finite measured optimization and output stage bounds")
    require(result["guard_probe_seconds_include_initial4_parity"] is True and
            0 <= result["parity_seconds"] <= result["guard_probe_seconds"],
            "All172 prospective output calls include initial4 parity timing")
    initial = read_json(_safe_file(root, output + "initial.json"))
    require(initial["adapter_sha256"] == protocol["initial_adapter_sha256"] and
            initial["base515_sha256"] == protocol["base515_sha256"] and
            initial["two_arm_reset_required"] is True and initial["model_qualified"] is False,
            "Exact fresh identity/R2 initialization receipt")
    expected_runtime = {output + name for name in
                        ("started.json", "preflight.json", "initial.json", "result.json",
                         "events.jsonl", "resources.jsonl")}
    guard_sources = {r["id"]: r for r in protocol["guards"]}
    for identifier in protocol["parity_ids"]:
        stem = output + "parity/" + identifier
        row = read_json(_safe_file(root, stem + ".json"))
        _audit_output_row(root, guard_sources[identifier], row, arm="shared", update=0, phase="parity")
        expected_runtime.update(stem + ext for ext in (".json", ".npy", ".png"))
    clean_cache = {}
    shared = result["shared_initial_feature_stats"]
    require(len(shared) == 10 and [r["id"] for r in shared] == [r["id"] for r in protocol["train_rows"]],
            "Exact shared10 initial feature rows")
    feature_rows = []
    for row in shared:
        stem = output + "shared_features0/" + row["id"]
        require(read_json(_safe_file(root, stem + ".json")) == row and row["arm"] == "shared" and
                row["update"] == 0 and row["adapter_state_sha256"] == protocol["initial_adapter_sha256"] and
                row["arrays"]["path"] == stem + ".npz", "Shared initial feature/state join")
        feature_rows.append(audit_feature_row(root, protocol, row, reference_root=refs,
                                             initial=True, _clean_cache=clean_cache))
        expected_runtime.update((stem + ".json", stem + ".npz"))
    branches = result["branches"]
    require(len(branches) == 2 and [b["arm"] for b in branches] == ["U", "S"], "Both fixed arms required")
    branch_reports, guard_reports = [], []
    planned_contexts = {kind: set() for kind in protocol["max_calls"]}
    context_key = lambda context: json.dumps(context, sort_keys=True, separators=(",", ":"))
    for identifier in protocol["parity_ids"]:
        planned_contexts["full_forward"].add(context_key(
            {"arm": "shared", "update": 0, "id": identifier, "phase": "parity"}))
    for row in protocol["train_rows"]:
        planned_contexts["prefix"].add(context_key({"id": row["id"]}))
    full_successful = [r for r in ledger["completed"].values() if r["kind"] == "full_forward"]
    commits = [r for r in ledger["extras"] if r["event"] == "optimizer_commit"]
    micros = [r for r in ledger["extras"] if r["event"] == "train_microbatch"]
    require(len(commits) == ledger["returns"]["optimizer"] and
            len(micros) == ledger["returns"]["backward"], "Commit/microbatch event count differs")
    for branch in branches:
        arm, updates = branch["arm"], branch["completed_updates"]
        require(type(updates) is int and updates in (6, 12, 24) and
                branch["terminal_status"] in ("complete", "guard_stop") and
                branch["automatic_candidate_selection"] is False and branch["model_qualified"] is False and
                read_json(_safe_file(root, output + "branches/" + arm + ".json")) == branch,
                "Terminal branch state/physical receipt differs")
        expected_runtime.add(output + "branches/" + arm + ".json")
        reset = branch["reset"]
        require(reset == {"initial_adapter_sha256": protocol["initial_adapter_sha256"],
                          "CPU_RNG_sha256": initial["CPU_RNG_sha256"],
                          "CUDA_RNG_sha256": initial["CUDA_RNG_sha256"], "fresh_Adam_empty": True},
                "Both branches require exact shared identity/RNG and fresh empty Adam")
        arm_commits = [r for r in commits if r["arm"] == arm]
        arm_micros = [r for r in micros if r["arm"] == arm]
        require([r["update"] for r in arm_commits] == list(range(1, updates + 1)) and
                len(arm_micros) == updates * 3, "Finite committed updates/three micros differ")
        before = protocol["initial_adapter_sha256"]
        for commit in arm_commits:
            update = commit["update"]
            require(commit["adapter_before_sha256"] == before and
                    commit["all515_frozen_exact"] is True and commit["gradient_clip_norm"] == 1. and
                    commit["S_expected_null_up_biases"] == (arm == "S") and
                    commit["all24_tensors_changed_required"] is False, "Committed state/base/nullspace chain")
            before = commit["adapter_after_sha256"]
            lr = 1e-4 * min(1., update / 10)
            _near(commit["LR"], lr, "Adam LR", atol=0, rtol=0)
            require(_finite_number(commit["preclip_norm"]) >= 0, "Finite optimizer norm")
            subset = [r for r in arm_micros if r["update"] == update]
            require([r["slot"] for r in subset] == [1, 2, 3], "Exact3 microbatch order")
            for micro in subset:
                index = (update - 1) * 3 + micro["slot"] - 1
                source = {r["id"]: r for r in protocol["train_rows"]}[micro["id"]]
                require(micro["id"] == protocol["sampler"][index]["id"] and
                        micro["adapter_state_before_update_sha256"] == commit["adapter_before_sha256"] and
                        micro["clean_targets"] == protocol["clean_features"][source["reference_id"]]["features"] and
                        micro["scales"] == protocol["clean_features"][source["reference_id"]]["scales"] and
                        set(micro["site_losses"]) == set(SITES), "Microbatch sampler/state/clean/RMS join")
                _near(micro["LR"], lr, "Microbatch LR", atol=0, rtol=0)
                _near(micro["scalar"], sum(micro["site_losses"].values()) / 4,
                      "FP32 site scalar", atol=5e-6, rtol=1e-6)
                _near(micro["divided_scalar"], micro["scalar"] / 3,
                      "FP32 accumulated microbatch divisor", atol=5e-6, rtol=1e-6)
                require(min(micro["site_losses"].values()) >= 0, "Negative feature absolute loss")
                context = {"arm": arm, "update": update, "slot": micro["slot"], "id": micro["id"]}
                planned_contexts["backward"].add(context_key(context))
                for site in SITES:
                    planned_contexts["adapter_train"].add(context_key({**context, "site": site}))
            planned_contexts["optimizer"].add(context_key({"arm": arm, "update": update}))
        points = [point for point in (6, 12, 24) if point <= updates]
        require([r["update"] for r in branch["feature_statistics"]] == points,
                "All reached feature readout snapshots required")
        checkpoints = []
        for update in (0, *points):
            name = output + f"checkpoints/{arm}/u{update:04d}.pt"
            physical = pin(root, name)
            require(physical["bytes"] <= limits["checkpoint_bytes"], "Adapter-only checkpoint byte cap")
            checkpoints.append(physical)
            expected_runtime.add(name)
        require(branch["last_checkpoint"] == checkpoints[-1], "Final stopped/complete checkpoint pin")
        failed = None
        for snapshot in branch["feature_statistics"]:
            update, rows = snapshot["update"], snapshot["rows"]
            require(len(rows) == 10 and [r["id"] for r in rows] ==
                    [r["id"] for r in protocol["train_rows"]], "All10 feature cases per reached snapshot")
            state = next(r["adapter_after_sha256"] for r in arm_commits if r["update"] == update)
            for row in rows:
                stem = output + f"stats/{arm}/u{update:04d}/" + row["id"]
                require(read_json(_safe_file(root, stem + ".json")) == row and row["arm"] == arm and
                        row["update"] == update and row["adapter_state_sha256"] == state and
                        row["arrays"]["path"] == stem + ".npz", "Stats state/snapshot/physical join")
                feature_rows.append(audit_feature_row(root, protocol, row, reference_root=refs,
                                                     initial=False, _clean_cache=clean_cache))
                expected_runtime.update((stem + ".json", stem + ".npz"))
                for site in SITES:
                    planned_contexts["adapter_stats"].add(context_key(
                        {"arm": arm, "update": update, "id": row["id"], "site": site}))
            stem = output + f"guards/{arm}/u{update:04d}/"
            bank = _bank(root, stem + "bank.json", protocol["guards"], arm=arm, update=update, phase="guard")
            gate = independent_guard(bank, guard_sources)
            require(read_json(_safe_file(root, stem + "gates.json")) == gate, "Exact unchanged RAW/PNG guard gate")
            guard_reports.append({"arm": arm, "update": update, "gate": gate})
            expected_runtime.update(stem + name for name in ("bank.json", "gates.json"))
            for identifier in bank:
                expected_runtime.update(stem + identifier + ext for ext in (".json", ".npy", ".png"))
                planned_contexts["full_forward"].add(context_key(
                    {"arm": arm, "update": update, "id": identifier, "phase": "guard"}))
            if not gate["pass"]:
                failed = update
                require(update == points[-1], "Failed branch continued to later feature/output snapshot")
                require(not any(r["context"]["arm"] == arm and r["context"]["phase"] == "strong" and
                                r["context"]["update"] >= update for r in full_successful),
                        "Strong probes ran after guard failure")
            else:
                strong_stem = output + f"strong/{arm}/u{update:04d}/"
                strong_bank = _bank(root, strong_stem + "bank.json", protocol["strong"],
                                    arm=arm, update=update, phase="strong")
                expected_runtime.add(strong_stem + "bank.json")
                for identifier in strong_bank:
                    expected_runtime.update(strong_stem + identifier + ext for ext in (".json", ".npy", ".png"))
                    planned_contexts["full_forward"].add(context_key(
                        {"arm": arm, "update": update, "id": identifier, "phase": "strong"}))
        require(branch["guard_failed_update"] == failed and
                branch["terminal_status"] == ("guard_stop" if failed is not None else "complete") and
                (failed is not None or updates == 24), "Branch terminal stop/result mismatch")
        expected_delta = {"prefix": 0, "adapter_train": updates * 3 * 4,
                          "adapter_stats": len(points) * 10 * 4,
                          "full_forward": len(points) * 24 + (len(points) - int(failed is not None)) * 4,
                          "adapter_full": (len(points) * 24 + (len(points) - int(failed is not None)) * 4) * 4,
                          "backward": updates * 3, "optimizer": updates}
        require({kind: branch["end_counts"][kind] - branch["start_counts"][kind] for kind in
                 protocol["max_calls"]} == expected_delta and
                branch["returns"] == branch["end_counts"], "Branch operation delta/returns differs")
        branch_reports.append({"arm": arm, "completed_updates": updates,
                               "terminal_status": branch["terminal_status"], "guard_failed_update": failed,
                               "operation_delta": expected_delta, "checkpoint_byte_bindings": checkpoints,
                               "checkpoint_Torch_tensor_Adam_RNG_contents_audited": False})
    require(branches[0]["end_counts"] == branches[1]["start_counts"] and
            branches[1]["end_counts"] == result["counts"], "Cumulative two-arm counter chain differs")
    expected_initial_counts = {"prefix": 10, "adapter_train": 0, "adapter_stats": 0,
                               "full_forward": 4, "adapter_full": 16, "backward": 0, "optimizer": 0}
    require(branches[0]["start_counts"] == expected_initial_counts, "Only4parity/10prefix before optimizers")
    for kind, expected_contexts in planned_contexts.items():
        if kind == "adapter_full":
            continue
        actual_contexts = {context_key(row["context"]) for row in ledger["completed"].values()
                           if row["kind"] == kind}
        require(actual_contexts == expected_contexts, "Actual operation contexts differ from reached plan: " + kind)
    reset_events = [r for r in ledger["extras"] if r["event"] == "branch_reset"]
    require(len(reset_events) == 2 and [r["arm"] for r in reset_events] == ["U", "S"],
            "Both durable branch reset events required")
    for event, branch in zip(reset_events, branches):
        require(event["same72_sampler_rows"] is True and
                event["base515_sha256"] == protocol["base515_sha256"] and
                all(event[k] == v for k, v in branch["reset"].items()), "Durable branch reset provenance")
    actual_runtime = {name for name in namespace["files"] if name.startswith(output)}
    require(actual_runtime == expected_runtime, "Reached/stopped closed runtime namespace differs")
    # VM cache bindings survive as provenance when payload arrays are omitted.
    cache = result["observed_VM_only_cache"]
    require(set(cache) == {r["id"] for r in protocol["train_rows"]}, "Exact10 observed feature caches")
    for identifier, entries in cache.items():
        require(set(entries) == set(SITES), "All40 observed feature cache bindings")
        for site, binding in entries.items():
            require(binding["path"] == f"cache/observed/{identifier}/encoder{site}.npy", "VM cache namespace")
            if not returned:
                read_npy(root, binding, (1, CHANNELS[site], int(site), int(site)), "float32")
    require(all(row["observed_cache"] == cache[row["id"]] for row in micros),
            "Training microbatch observed feature cache binding differs")
    groups = []
    for arm in ("shared", "U", "S"):
        for update in sorted({r["update"] for r in feature_rows if r["arm"] == arm}):
            for site in SITES:
                selected = [r["site_metrics"][site] for r in feature_rows
                            if r["arm"] == arm and r["update"] == update]
                groups.append({"arm": arm, "update": update, "site": site, "TRAIN_cases": len(selected),
                               "mean_initial_L1": sum(r["initial_normalized_L1"] for r in selected) / len(selected),
                               "mean_final_L1": sum(r["final_normalized_L1"] for r in selected) / len(selected),
                               "mean_initial_centered_L1": sum(r["initial_centered_normalized_L1"] for r in selected) / len(selected),
                               "mean_final_centered_L1": sum(r["final_centered_normalized_L1"] for r in selected) / len(selected)})
    return {"complete": True, "audit_pass": True, "terminal_kind": "finite_two_arm_screen",
            "training_success_not_implied": True, "protocol_sha256": protocol_sha,
            "counts": ledger["counts"], "returns": ledger["returns"], "pending": {},
            "branches": branch_reports, "guards": guard_reports, "feature_groups": groups,
            "feature_case_readouts": len(feature_rows), "feature_site_readouts": len(feature_rows) * 4,
            "feature_NPZ_arrays": len(feature_rows) * 12, "resources": resource_report,
            "namespace": namespace, "export": export, "source_contract": source_contract(root, protocol),
            "reference_root_explicit": reference_root is not None,
            "physically_reread_clean_feature_arrays": reference_pins,
            "calibration_not_subtracted": True, "full_activation_replay": False,
            "source_captured_absolute_moments_independently_recomputed_from_full_activations": False,
            "checkpoint_Torch_tensor_Adam_RNG_contents_audited": False,
            "checkpoint_content_audit_required_before_learning_claims_or_continuation": True,
            "independent_face_visual_review_pending": True,
            "historical_original_preservation_R2_dark_failures_retained": True,
            "full478_DEV_FINAL_app_user_acceptance_not_established": True,
            "Torch_neural_gradient_optimizer_VM_browser_calls": 0,
            "model_qualified": False, "app_default_changed": False, "goal_complete": False}
