"""Rebuild V8 training-only pixels, embeddings, traces and states without inference."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import tarfile

import numpy as np
from PIL import Image
import torch


def close(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            close(actual[key], expected[key])
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert math.isfinite(actual) and math.isclose(actual, expected, rel_tol=2e-6, abs_tol=2e-7)
    else:
        assert actual == expected


def audit(root, parent_root, out, capacity_results, expected_protocol_sha):
    spec = importlib.util.spec_from_file_location("blur_fit_pinned", root / "scripts/cctv_dgp_blur_fit_v8.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert module.sha(root / "blur_fit_protocol_v8.json") == expected_protocol_sha
    p = module.verify(root, weights=False)
    assert module.sha(parent_root / "capacity_protocol_v7.json") == module.PARENT_SHA
    parent = module.read(parent_root / "capacity_protocol_v7.json")
    assert parent["cases"] == p["cases"] and parent["references"] == p["references"]
    for name, pin in p["cache_assets"].items():
        assert module.sha(parent_root / name) == pin
    sys.path.insert(0, str(root))
    from cctv_dgp_pilot import aggregate, exported_pixel_metrics, state_hash
    from models import DGPSynthesizer
    result = module.read(out / "results.json")
    assert result["complete"] and result["protocol_sha256"] == expected_protocol_sha
    assert result["total_optimizer_updates"] == result["training_backward_calls"] == 1000
    assert result["preflight_autograd_calls"] == 1 and 0 < result["elapsed_seconds"] <= 600
    assert result["buffers_and_teachers_unchanged"] and result["training_only"]
    assert not any(result[k] for k in ("validation_used", "native_used", "native_reserved_used", "production_promoted", "model_improvement_established"))
    for name, pin in result["artifacts_sha256"].items():
        path = out / name
        assert path.resolve().is_relative_to(out.resolve()) and module.sha(path) == pin
    execution = module.read(out / "execution.json")
    assert execution["host"].split('.')[0] == "forensic-dgp-thesis" and "L4" in execution["gpu"]
    assert execution["starting_state_hash"] == module.START_STATE and execution["batch_size"] == 2
    assert execution["runtime_cap_seconds"] == 600
    assert execution["source_sha256"] == module.sha(root / "scripts/cctv_dgp_blur_fit_v8.py") == module.sha(out / "executed_source.py")
    preflight = module.read(out / "preflight.json")
    assert preflight["passed"] and preflight["zero_optimizer_updates"] and preflight["autograd_calls"] == 1
    assert preflight["active_gradient_tensors"] > 0 and preflight["training_cases"] == module.TRAIN_IDS
    assert preflight["starting_state_hash"] == module.START_STATE and preflight["buffers_unchanged"]
    timing = module.read(out / "timing_at16.json")
    assert timing["cap_seconds"] == 600 and 0 < timing["projected_seconds"] <= 600
    traces = [json.loads(line) for line in (out / "updates.jsonl").read_text().splitlines()]
    assert len(traces) == 1000
    for update, row in enumerate(traces, 1):
        assert row["update"] == update and row["cases"] == module.TRAIN_IDS
        assert math.isfinite(row["preclip_norm"]) and row["preclip_norm"] > 0
        assert set(row["components"]) == {"pixel_mse"} and row["loss"] >= 0
        close(row["loss"], row["components"]["pixel_mse"])
    refs = {r["id"]: r for r in p["references"]}
    target_embeds = {ref: np.load(out / "target_embeddings" / (ref + ".npy"), allow_pickle=False) for ref in refs}
    for embed in target_embeds.values():
        assert embed.shape == (512,) and np.isfinite(embed).all() and np.isclose(np.linalg.norm(embed), 1.0, atol=1e-5)
    metrics = {}; count = 0
    for stage in ["baseline", "update20", "update100", "update1000"]:
        saved = module.read(out / stage / "metrics.json")
        assert len(saved["rows"]) == 10 and saved["training_only"]
        rebuilt = []
        for case, row in zip(p["cases"], saved["rows"]):
            for key, value in case.items():
                assert row[key] == value
            ref = refs[case["reference_id"]]
            rgb = np.asarray(Image.open(out / row["prediction"]).convert("RGB"))
            target = np.asarray(Image.open(root / ref["target"]).convert("RGB"))
            support = np.asarray(Image.open(root / ref["observed"])) > 0
            input_rgb = np.asarray(Image.open(root / case["input"]).convert("RGB"))
            raw = np.load(out / row["raw_float"], allow_pickle=False)
            assert raw.shape == (256, 256, 3) and raw.dtype == np.float32 and np.isfinite(raw).all()
            assert raw.min() >= 0 and raw.max() <= 1
            quantized = np.clip(raw * 255, 0, 255).astype(np.uint8)
            quantized[~support] = input_rgb[~support]
            np.testing.assert_array_equal(rgb, quantized)
            np.testing.assert_array_equal(rgb[~support], input_rgb[~support])
            values = exported_pixel_metrics(rgb, target, support)
            for key, value in values.items():
                close(row[key], value)
            embed = np.load(out / row["embedding"], allow_pickle=False)
            assert embed.shape == (512,) and np.isfinite(embed).all() and np.isclose(np.linalg.norm(embed), 1.0, atol=1e-5)
            cosine = float(np.clip(embed @ target_embeds[ref["id"]], -1, 1))
            close(row["ArcFace_observed_fixed"], cosine)
            if stage == "baseline":
                prior = np.asarray(Image.open(capacity_results / "baseline/images" / (case["id"] + ".png")).convert("RGB"))
                np.testing.assert_array_equal(rgb, prior)
            rebuilt.append({**row, **values, "ArcFace_observed_fixed": cosine}); count += 1
        close(saved["summary"], aggregate(rebuilt))
        sheet = np.asarray(Image.open(out / saved["preview"]).convert("RGB"))
        assert sheet.shape == (2904, 780, 3)
        for i, row in enumerate(saved["rows"]):
            y = 24 + i * 288 + 28; ref = refs[row["reference_id"]]
            for j, file in enumerate([root / row["input"], out / row["prediction"], root / ref["target"]]):
                np.testing.assert_array_equal(sheet[y:y + 256, j * 260 + 2:j * 260 + 258], np.asarray(Image.open(file).convert("RGB")))
        metrics[stage] = saved["summary"]
    start = torch.load(parent_root / p["weights"]["start"], map_location="cpu", weights_only=True)
    assert state_hash(start) == module.START_STATE
    model = DGPSynthesizer().eval(); model.load_state_dict(start, strict=True)
    buffers = {name for name, _ in model.named_buffers()}; parameters = {name for name, _ in model.named_parameters()}
    assert [s["update"] for s in result["snapshots"]] == [20, 100, 1000]
    changes = []
    for snapshot in result["snapshots"]:
        weights = torch.load(out / snapshot["checkpoint"], map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=True)
        assert state_hash(weights) == snapshot["state_hash"]
        assert all(torch.equal(weights[name], start[name]) for name in buffers)
        changed = sum(not torch.equal(weights[name], start[name]) for name in parameters)
        assert changed > 0
        assert snapshot["metrics"] == f"update{snapshot['update']}/metrics.json"
        changes.append({"update": snapshot["update"], "changed_parameter_tensors": changed})
    key = "dataset/thumbnails128x128/blur_lr24"
    ratio = metrics["update1000"][key]["MSE"] / metrics["baseline"][key]["MSE"]
    close(result["fixed_final_blur_training_mse_ratio"], ratio)
    assert result["training_fit_observed"] == (ratio <= 0.8)
    parent_result = module.read(capacity_results / "results.json")
    assert parent_result["protocol_sha256"] == module.PARENT_SHA and parent_result["complete"]
    comparisons = {"v7_pixel100_blur_training_mse": module.read(capacity_results / "pixel_mse_update100/metrics.json")["summary"][key]["MSE"],
        "v8_blur_training_mse": {stage: metrics[stage][key]["MSE"] for stage in metrics},
        "v8_all_training_profile_summaries": metrics,
        "v8_pair_exposure_matches_v7_pixel100_at_update": 20}
    assert not list(out.rglob("best.pth"))
    return {"complete": True, "protocol_sha256": expected_protocol_sha, "results_sha256": module.sha(out / "results.json"),
        "training_pngs_checked": count, "raw_float_predictions_checked": count,
        "embedding_arrays_checked": count + 10, "update_trace_checked": 1000,
        "checkpoint_changes": changes, "v7_baseline_pngs_identical": 10,
        "fixed_final_blur_training_mse_ratio": ratio, "training_fit_observed": ratio <= 0.8,
        "comparisons": comparisons, "training_only": True, "local_model_forwards": 0,
        "local_backward_calls": 0, "local_optimizer_updates": 0, "native_reserved_used": False,
        "production_promoted": False, "model_improvement_established": False,
        "limitation": "Rebuilds training-only arithmetic/states; no recognizer/CUDA gradient replay, generalization, useful native-output or identity-recovery claim"}


def extract(archive, destination):
    import hashlib
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    assert Path(str(archive) + ".sha256").read_text().strip().split() == [digest.hexdigest(), archive.name]
    assert not destination.exists(), "Preserve existing return"
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        assert len(members) < 500 and sum(m.size for m in members) < 200 * 1024 ** 2
        for member in members:
            path = Path(member.name)
            assert (member.isfile() or member.isdir()) and not path.is_absolute() and ".." not in path.parts
            assert "\\" not in member.name and ":" not in member.name
        destination.mkdir(parents=True)
        for member in members:
            if member.isfile():
                target = destination / member.name; target.parent.mkdir(parents=True, exist_ok=True)
                with stream.extractfile(member) as source, target.open("xb") as dest:
                    shutil.copyfileobj(source, dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--capacity-results", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha", required=True)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--extract-to", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.archive:
        assert args.extract_to is not None
        extract(args.archive, args.extract_to)
        args.results = args.extract_to / "outputs/cctv_dgp_blur_fit_v8"
    result = audit(args.root, args.parent_root, args.results, args.capacity_results, args.expected_protocol_sha)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print({k: v for k, v in result.items() if k != "comparisons"}, flush=True)
