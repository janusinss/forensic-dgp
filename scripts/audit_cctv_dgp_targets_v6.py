"""Independent data/output arithmetic audit; never trains or replays CUDA gradients."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import tarfile
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from cctv_dgp_targets_v6 import ARMS, SEED, START_STATE, read, safe_path, sha, verify, write
from cctv_camera_stress import degrade
from cctv_dgp_pilot import aggregate, exported_pixel_metrics, qualifies, state_hash


def assert_close(actual, expected, message):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(message + ": keys differ")
        for key in expected:
            assert_close(actual[key], expected[key], message + "/" + key)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if actual is None or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=2e-6, abs_tol=2e-7):
            raise ValueError(message + ": value differs")
    elif actual != expected:
        raise ValueError(message + ": value differs")


def audit_preparation(root, workspace=None):
    start = time.monotonic()
    p = verify(root)
    refs = {r["id"]: r for r in p["references"]}
    source_receipt = read(root / "hq_source_derivation_receipt.json")
    eligible = read(root / "lineage/eligible_references.json")
    checked = {r["id"]: r for r in source_receipt["references"]}
    if len(checked) != 444 or set(checked) != {r["reference_id"] for r in eligible["references"]}:
        raise ValueError("HQ derivation evidence differs")
    sources_rebuilt = 0
    for item in eligible["references"]:
        record = checked[item["reference_id"]]
        if record["source_sha256"] != item["source_sha256"] or record["target_sha256"] != item["target_sha256"] or not record["canonical_target_rebuilt_from_1024"]:
            raise ValueError("HQ derivation binding differs")
        if workspace:
            file = Path(workspace) / item["source_path_from_workspace"]
            if sha(file) != item["source_sha256"]:
                raise ValueError("Local original HQ source differs")
            with Image.open(file) as source:
                canonical = source.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
            np.testing.assert_array_equal(np.asarray(canonical), np.asarray(Image.open(root / refs[item["reference_id"]]["target"])))
            sources_rebuilt += 1
    count = 0
    for case in sum(p["training_epochs"].values(), []) + p["validation_cases"]:
        ref = refs[case["reference_id"]]
        target = np.asarray(Image.open(root / ref["target"]).convert("RGB"))
        rebuilt, details = degrade(target, ref["bounds"], case["camera"], case["proxy_details"]["seed"])
        np.testing.assert_array_equal(np.asarray(Image.open(root / case["input"]).convert("RGB")), rebuilt)
        if details != case["proxy_details"]:
            raise ValueError("Input camera provenance differs")
        count += 1
    return {"complete": True, "protocol_sha256": sha(root / "targets_protocol_v6.json"),
        "input_pngs_rebuilt": count, "reduced_targets_rebuilt": 391, "hq_sources_independently_rebuilt": sources_rebuilt,
        "training_references": 391, "validation_references": 104, "expected_vm_updates": 196,
        "local_model_forwards": 0, "local_backward_calls": 0, "local_optimizer_updates": 0,
        "native_reserved_used": False, "seconds": time.monotonic() - start}


def audit_results(root, out):
    p = verify(root)
    report = read(out / "results.json")
    if not report["complete"] or report["protocol_sha256"] != sha(root / "targets_protocol_v6.json"):
        raise ValueError("Returned completion/protocol differs")
    if (report["total_optimizer_updates"], report["training_backward_calls"], report["preflight_autograd_calls"]) != (196, 196, 2):
        raise ValueError("Returned execution budget differs")
    if not 0 < report["elapsed_seconds"] <= 1200 or report["production_promoted"] or report["native_reserved_used"] or report["goal_complete"]:
        raise ValueError("Returned timing/release boundary differs")
    for name, pin in report["artifacts_sha256"].items():
        if sha(safe_path(out, name)) != pin:
            raise ValueError("Returned artifact differs: " + name)
    for name, pin in p["assets_sha256"].items():
        if name.endswith((".py", ".sh")) and sha(out / "runtime_sources" / name) != pin:
            raise ValueError("Executed runtime source differs")
    execution, preflight = read(out / "execution.json"), read(out / "preflight.json")
    if execution["host"].split(".")[0] != "forensic-dgp-thesis" or execution["device"] != "cuda":
        raise ValueError("Training was not recorded on authorized VM")
    if execution["starting_state_hash"] != START_STATE or preflight["starting_state_hash"] != START_STATE:
        raise ValueError("Starting state differs")
    if not preflight["passed"] or not preflight["zero_optimizer_updates"] or preflight["autograd_calls"] != 2 or preflight["tail_batch_size"] != 7 or "L4" not in preflight["gpu"]:
        raise ValueError("CUDA preflight evidence differs")
    for check, arm in zip(preflight["arms"], ARMS):
        if check["arm"] != arm["id"] or check["optimizer_constructed"] or check["optimizer_updates"] or check["active_gradient_tensors"] <= 0:
            raise ValueError("Target-specific gradient preflight differs")
    traces = [json.loads(line) for line in (out / "updates.jsonl").read_text().splitlines()]
    if len(traces) != 196:
        raise ValueError("Training trace length differs")
    offset = 0
    for arm in ARMS:
        update = 0
        for epoch in (1, 2):
            cases = p["training_epochs"][str(epoch)]
            for index in range(0, len(cases), 8):
                update += 1
                row = traces[offset]
                offset += 1
                if (row["arm"], row["epoch"], row["update"], row["cases"], row["backward_calls"]) != (
                    arm["id"], epoch, update, [c["id"] for c in cases[index:index+8]], 1):
                    raise ValueError("Matched traversal differs")
                expected = row["components"]["pixel"] + .05 * row["components"]["color"] + .1 * row["components"]["vgg"] + .05 * row["components"]["sobel"] + .1 * row["identity_loss"]
                assert_close(row["loss"], expected, "update objective")
                if not math.isfinite(row["preclip_norm"]) or row["preclip_norm"] <= 0:
                    raise ValueError("Nonfinite/empty recorded update gradient")
    # Check immutable buffers and actual parameter changes without model forwards.
    from models import DGPSynthesizer
    model = DGPSynthesizer().eval()
    start = torch.load(root / p["weights"]["start"], map_location="cpu", weights_only=True)
    model.load_state_dict(start, strict=True)
    parameter_keys = {k for k, _ in model.named_parameters()}
    buffer_keys = {k for k, _ in model.named_buffers()}
    refs = {r["id"]: r for r in p["references"]}
    canonical_embeds = {}
    for ref in refs.values():
        embed = np.load(out / "reference_embeddings" / (ref["id"] + ".npy"), allow_pickle=False)
        if embed.shape != (512,) or not np.isfinite(embed).all() or not np.isclose(np.linalg.norm(embed), 1., atol=1e-5):
            raise ValueError("Malformed reference embedding")
        canonical_embeds[ref["id"]] = embed
    metrics_reports = {}
    png_count, float_count, cosines = 0, 0, 0
    stages = ["baseline"] + [f"{a['id']}_epoch{e}" for a in ARMS for e in (1, 2)]
    cases_by_id = {c["id"]: c for c in p["validation_cases"]}
    for stage in stages:
        metrics = read(out / stage / "metrics.json")
        if len(metrics["rows"]) != 520 or {r["id"] for r in metrics["rows"]} != set(cases_by_id):
            raise ValueError("Common validation IDs differ")
        rebuilt_rows, rebuilt_input = [], []
        for row in metrics["rows"]:
            case, ref = cases_by_id[row["id"]], refs[row["reference_id"]]
            for key, value in case.items():
                if row[key] != value:
                    raise ValueError("Evaluation case binding differs")
            actual = np.asarray(Image.open(out / row["prediction"]).convert("RGB"))
            target = np.asarray(Image.open(root / ref["target"]).convert("RGB"))
            mask = np.asarray(Image.open(root / ref["observed"])) > 0
            input_rgb = np.asarray(Image.open(root / case["input"]).convert("RGB"))
            np.testing.assert_array_equal(actual[~mask], input_rgb[~mask])
            values = exported_pixel_metrics(actual, target, mask)
            for key in values:
                assert_close(row[key], values[key], "exact PNG metric")
            embed = np.load(out / row["embedding"], allow_pickle=False)
            cosine = float(np.clip(embed @ canonical_embeds[ref["id"]], -1, 1))
            assert_close(row["ArcFace_observed_fixed"], cosine, "embedding cosine")
            rebuilt_rows.append({**row, **values, "ArcFace_observed_fixed": cosine})
            png_count += 1
            cosines += 1
            if row["id"] in p["preview_case_ids"]:
                with np.load(out / stage / "float_preview" / (row["id"] + ".npz"), allow_pickle=False) as data:
                    network, observed = data["network_rgb"], data["observed_rgb"]
                    if network.shape != (256, 256, 3) or not np.isfinite(network).all() or network.min() < 0 or network.max() > 1:
                        raise ValueError("Raw float preview differs")
                    np.testing.assert_allclose(observed, network * mask[:, :, None] + input_rgb / 255 * (~mask)[:, :, None], atol=1e-7)
                    np.testing.assert_array_equal((observed * 255).astype(np.uint8), actual)
                float_count += 1
            if stage == "baseline":
                input_row = metrics["input_rows"][len(rebuilt_input)]
                if input_row["id"] != row["id"]:
                    raise ValueError("Input comparator order differs")
                values = exported_pixel_metrics(input_rgb, target, mask)
                for key in values:
                    assert_close(input_row[key], values[key], "input metric")
                embed = np.load(out / input_row["embedding"], allow_pickle=False)
                cosine = float(np.clip(embed @ canonical_embeds[ref["id"]], -1, 1))
                assert_close(input_row["ArcFace_observed_fixed"], cosine, "input cosine")
                rebuilt_input.append({**input_row, **values, "ArcFace_observed_fixed": cosine})
                cosines += 1
        assert_close(metrics["summary"], aggregate(rebuilt_rows), "common target summary")
        if rebuilt_input:
            assert_close(metrics["input_summary"], aggregate(rebuilt_input), "input summary")
        # Verify displayed 256-pixel cells against actual files, not only their checksum.
        sheet = np.asarray(Image.open(out / metrics["preview"]).convert("RGB"))
        by_id = {row["id"]: row for row in metrics["rows"]}
        for i, case_id in enumerate(p["preview_case_ids"]):
            row, y = by_id[case_id], 24 + i * 288 + 28
            for j, file in enumerate((root / row["input"], out / row["prediction"], root / refs[row["reference_id"]]["target"])):
                np.testing.assert_array_equal(sheet[y:y+256, j*260+2:j*260+258], np.asarray(Image.open(file).convert("RGB")))
        metrics_reports[stage] = metrics
    selections, changes = [], []
    if [b["arm"] for b in report["branches"]] != ARMS:
        raise ValueError("Returned arms differ")
    for branch in report["branches"]:
        arm = branch["arm"]["id"]
        best, epoch_selected, source = metrics_reports["baseline"]["summary"], 0, f"{arm}/baseline.pth"
        if state_hash(torch.load(out / source, map_location="cpu", weights_only=True)) != START_STATE:
            raise ValueError("Arm baseline state differs")
        for entry in branch["epochs"]:
            epoch = entry["epoch"]
            if epoch not in (1, 2) or entry["cumulative_updates"] != epoch * 49:
                raise ValueError("Epoch budget differs")
            weights = torch.load(out / entry["checkpoint"], map_location="cpu", weights_only=True)
            model.load_state_dict(weights, strict=True)
            for key in buffer_keys:
                np.testing.assert_array_equal(weights[key].numpy(), start[key].numpy())
            changed = sum(not torch.equal(weights[key], start[key]) for key in parameter_keys)
            if changed <= 0 or state_hash(weights) != entry["state_hash"] or sha(out / entry["checkpoint"]) != entry["checkpoint_sha256"]:
                raise ValueError("Checkpoint change evidence differs")
            candidate = metrics_reports[f"{arm}_epoch{epoch}"]["summary"]
            accepted = qualifies(candidate, metrics_reports["baseline"]["summary"], best)
            if entry["accepted"] != accepted:
                raise ValueError("Strict guard selection differs")
            if accepted:
                best, epoch_selected, source = candidate, epoch, entry["checkpoint"]
            changes.append({"arm": arm, "epoch": epoch, "changed_parameter_tensors": changed})
        selection = read(out / arm / "best_selection.json")
        if selection != branch["selection"] or selection["selected_epoch"] != epoch_selected or selection["selected_source"] != source:
            raise ValueError("Best checkpoint selection differs")
        if sha(out / arm / "best.pth") != sha(out / source) or selection["best_sha256"] != sha(out / source):
            raise ValueError("Best bytes differ from selected checkpoint")
        selections.append(selection)
    return {"complete": True, "protocol_sha256": report["protocol_sha256"], "returned_results_sha256": sha(out / "results.json"),
        "png_predictions_checked": png_count, "raw_float_previews_checked": float_count,
        "embedding_cosines_rebuilt": cosines, "update_trace_checked": len(traces),
        "checkpoint_changes": changes, "selections": selections, "local_model_forwards": 0,
        "local_backward_calls": 0, "local_optimizer_updates": 0, "cuda_gradients_recomputed": False,
        "production_promoted": False, "native_reserved_used": False,
        "limitation": "Arithmetic/state/source audit; recognizer forwards and CUDA gradients are not independently replayed; useful native output and independent final review remain required"}


def extract(archive, destination):
    checksum_file = Path(str(archive) + ".sha256")
    fields = checksum_file.read_text(encoding="ascii").strip().split()
    if len(fields) != 2 or fields[1].lstrip("*") != Path(archive).name or fields[0] != sha(archive):
        raise ValueError("Return archive checksum differs")
    if destination.exists():
        raise ValueError("Preserve existing extracted return")
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        if len(members) > 12000 or sum(item.size for item in members) > 2 * 1024**3:
            raise ValueError("Unexpected return archive size")
        for item in members:
            safe_path(destination, item.name)
            if not item.isfile() and not item.isdir():
                raise ValueError("Archive links/special files are forbidden")
        destination.mkdir(parents=True)
        for item in members:
            if item.isfile():
                file = safe_path(destination, item.name)
                file.parent.mkdir(parents=True, exist_ok=True)
                with stream.extractfile(item) as source, file.open("xb") as target:
                    import shutil
                    shutil.copyfileobj(source, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--verify-preparation", action="store_true")
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--extract-to", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.archive:
        if not args.extract_to:
            parser.error("--archive requires --extract-to")
        extract(args.archive, args.extract_to)
        args.results = args.extract_to / "outputs/cctv_dgp_targets_v6"
    result = audit_preparation(args.root, args.workspace_root) if args.verify_preparation else audit_results(args.root, args.results)
    if args.receipt:
        write(args.receipt, result)
    print(result)
