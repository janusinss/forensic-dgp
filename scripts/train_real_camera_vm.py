"""CUDA/Linux-only, fixed-budget camera-versus-source detector diagnostic."""
import argparse
import copy
import gc
import json
from pathlib import Path
import sys
import tarfile
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from native_expert import (MODEL_SHA, OPTIMIZER_SHA, PARENT_SHA, require, require_vm_gpu,
                           validate_source, validate_source_optimizer, expert_loss)
from real_camera_experiment import ARMS, EPOCHS, STEPS, fixed_schedules
from supported_real_data import SupportedMasks
from scripts.train_native_expert_vm import (sha, read_json, write_json, safe_path, verify_fixture,
                                            dependency_versions, counts, aggregate)


def load_case(case):
    def array(path):
        with Image.open(safe_path(ROOT, path)) as image:
            return np.asarray(image).copy()
    rgb = array(case["input"])
    target = array(case["target"]["mask"]["path"]) == 255
    valid = array(case["target"]["valid"]["path"]) == 255
    require(rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8
            and target.shape == valid.shape == (256, 256) and valid.any()
            and not (target & ~valid).any(), "Frozen real-camera case differs")
    return (torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255,
            torch.from_numpy(target.astype(np.float32))[None], torch.from_numpy(valid.astype(np.float32))[None])


def fixture_tensors(item):
    return item["input"].float() / 255, item["mask"].float(), item["valid"].float()


@torch.inference_mode()
def measure(model, cases, cache, folder, wall_check):
    model.eval()
    folder.mkdir(parents=True)
    groups, rows, hashes = {}, [], {}
    for pool, iterable in (("real", cases), ("fixture", range(280))):
        (folder / pool).mkdir()
        for item in iterable:
            wall_check()
            if pool == "real":
                index = item["case_id"]
                x, target, valid = load_case(item)
                group = ("old" if item["real_train_index"] < 83 else "new") + "_" + item["condition"]
                name = f"real/{index:03d}"
            else:
                index = item
                x, target, valid = fixture_tensors(cache[index])
                group, name = "reflection", f"fixture/{index:03d}"
            prediction = (model.detect(x.cuda()[None]).sigmoid()[0, 0] >= .5).cpu()
            measured = counts(prediction, target[0].bool(), valid[0].bool())
            groups.setdefault(group, []).append(measured)
            path = folder / (name + ".png")
            Image.fromarray(prediction.numpy().astype(np.uint8) * 255).save(path)
            hashes[name + ".png"] = sha(path)
            rows.append({"id": name, "group": group, "counts": measured})
    require(len(rows) == 462, "Fixed train-cohort measurement budget differs")
    return {"groups": {k: aggregate(v) for k, v in groups.items()}, "rows": rows,
            "mask_sha256": hashes, "forward_images": 462, "held_out_forward_images": 0,
            "scope": "Training-cohort diagnosis only; new sources are trained by camera91 and are not a holdout"}


def decisions(before, final):
    def retained(a, b):
        return a["iou"] >= b["iou"] and all(a[k] <= b[k] for k in
                    ("missed_fraction", "visible_false_positive", "empty_mask_cases", "negative_false_positive_cases"))
    a, b, c = (final[arm]["groups"] for arm in ARMS)
    native = before["groups"]
    checks = {
        "camera83_degraded_fit_beats_native83": b["old_degraded"]["iou"] > a["old_degraded"]["iou"],
        "camera83_native_and_replay_fit_retained_vs_native83": all(retained(b[g], a[g]) for g in ("old_native", "reflection")),
        "camera91_new_native_and_degraded_fit_gain_vs_camera83": all(c[g]["iou"] > b[g]["iou"] for g in ("new_native", "new_degraded")),
        "camera91_old_native_degraded_and_replay_fit_retained": all(retained(c[g], b[g]) for g in ("old_native", "old_degraded", "reflection")),
        "all_final_clear_controls_empty": all(group["negative_false_positive_cases"] == 0 for arm in final.values() for group in arm["groups"].values()),
    }
    return {"checks": checks, "all_training_fit_checks_pass": all(checks.values()),
            "source42_baseline_groups": native, "training_fit_only": True,
            "original_425_case_gates_evaluated": False, "historical_gate_failure_unchanged": True,
            "promoted": False, "selected_checkpoint": None,
            "next": "Independently audit returned states/masks and review the fixed grid before any original-gate or end-to-end evaluation"}


def preview(cases, ids, root):
    width, height = 192, 219
    canvas = Image.new("RGB", (6 * width, 42 + height * len(ids)), "#16181c")
    draw = ImageDraw.Draw(canvas)
    labels = ("input", "supervised core", "source42", *ARMS)
    for col, text in enumerate(labels):
        draw.text((col * width + 3, 7), text, fill="white")
    draw.text((3, 24), "TRAIN-COHORT MASKS ONLY: green TP, red FP, yellow FN, purple unsupervised. No generated faces.", fill="white")
    for row, case_id in enumerate(ids):
        case = cases[case_id]
        x, target, valid = load_case(case)
        image = np.rint(x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        truth, support = target[0].numpy().astype(bool), valid[0].numpy().astype(bool)
        def overlay(pred):
            color = image.copy()
            marked = truth | pred | ~support
            color[~support] = (165, 85, 215)
            color[truth & pred & support] = (16, 185, 129)
            color[~truth & pred & support] = (245, 50, 50)
            color[truth & ~pred & support] = (255, 220, 40)
            result = image.copy()
            result[marked] = (.55 * image[marked] + .45 * color[marked]).round().astype(np.uint8)
            return result
        panels = [image, overlay(truth)]
        for folder in (root / "initial_masks", *(root / arm / "final_masks" for arm in ARMS)):
            with Image.open(folder / "real" / f"{case_id:03d}.png") as photo:
                panels.append(overlay(np.asarray(photo) == 255))
        y = 42 + row * height
        for col, panel in enumerate(panels):
            canvas.paste(Image.fromarray(panel).resize((width, width)), (col * width, y))
            draw.text((col * width + 3, y + width + 3), f"real{case['real_train_index']:03d}/{case['condition']}", fill="white")
    canvas.save(root / "preview.png")


def main(args):
    require_vm_gpu()  # First: Windows/CPU cannot create files, models or optimizers.
    require(args.batch_size == (1 if args.dry_run else 8), "Use batch1 only for dry run; actual fixed batch8")
    protocol_path = ROOT / "inputs/real_camera_protocol.json"
    protocol = read_json(protocol_path)
    inventory = read_json(ROOT / "inventory.json")
    for name, digest in inventory.items():
        require(sha(safe_path(ROOT, name)) == digest, "Bundle asset changed: " + name)
    existing = Path(args.existing).expanduser().resolve() if args.existing else ROOT.parent / "coverage_vm_bundle"
    assets = {role: safe_path(existing, row["vm_path"]) for role, row in protocol["existing_assets"].items()}
    for role, path in assets.items():
        require(sha(path) == protocol["existing_assets"][role]["sha256"], "Read-only VM asset differs: " + role)
    require(sha(assets["model"]) == MODEL_SHA and sha(assets["optimizer"]) == OPTIMIZER_SHA
            and sha(assets["parent"]) == PARENT_SHA, "Source42 moment/parent binding differs")
    require(all((ROOT.parent / "dataset" / name).is_dir() for name in ("asian_faces", "thumbnails128x128")), "Original repository dataset directories required")
    split_path = ROOT / "inputs/phase4_split.json"
    require(sha(split_path) == protocol["phase4_split_sha256"], "Original split snapshot differs")
    repository_split = ROOT.parent / "outputs/phase4_with_progress/split.json"
    require(repository_split.is_file() and sha(repository_split) == protocol["phase4_split_sha256"], "VM original split missing/different")
    split = read_json(split_path)
    training = {path.replace("\\", "/") for path in split["train"]}
    require(not training.intersection(path.replace("\\", "/") for path in split["validation"]), "Original split overlap")
    real = SupportedMasks(ROOT / "dataset/detector_supported_review_v2/manifest.json", split="train")
    parent_manifest = read_json(ROOT / "inputs/parent_supported_manifest.json")
    require(len(real) == 91 and real.metadata["supported_records"][:115] == parent_manifest["supported_records"], "Supported source/split lineage differs")
    fixture_rows = read_json(assets["fixture_manifest"])["cases"]
    require(all(row["source"].replace("\\", "/") in training for row in real.rows[73:83] + fixture_rows), "Original native/fixture source outside Phase4 training")
    require(fixed_schedules(real.rows, fixture_rows) == protocol["schedules"], "Fixed schedule differs")
    pairs = read_json(ROOT / "outputs/real_camera_pairs_v1/manifest.json")
    require(pairs["registry_sha256"] == sha(ROOT / "dataset/detector_supported_review_v2/manifest.json") and len(pairs["cases"]) == 182,
            "Real camera cache binding differs")
    cases = pairs["cases"]
    torch.set_num_threads(4)
    cache = torch.load(assets["fixture_pixels"], map_location="cpu", weights_only=True)
    require(cache["format"] == "dgp-reflection-coverage-pixels-v1" and set(cache["pixels"]) == set(range(280)), "Fixture cache differs")
    pixels = cache["pixels"]
    for i, row in enumerate(fixture_rows):
        verify_fixture(pixels[i], row)
    deps_path = existing / "outputs/face_occlusion_dependencies"
    sys.path.insert(0, str(deps_path))
    versions = dependency_versions(deps_path)
    require(sha(deps_path / "setup.json") == protocol["setup_sha256"] and str(torch.__version__) == "2.9.1+cu129", "Pinned existing runtime differs")
    require(torch.cuda.get_device_properties(0).total_memory >= 4 * 1024**3 and torch.cuda.mem_get_info()[0] >= 2 * 1024**3, "Require4GiB total/2GiB free VRAM")
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    from face_occlusion_adapter import load_adapter, parameter_groups
    moments = torch.load(assets["optimizer"], map_location="cpu", weights_only=True)
    model, payload = load_adapter(assets["model"], "cuda")
    validate_source(payload)
    moment_check = validate_source_optimizer(moments, parameter_groups(model), MODEL_SHA)
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    forward_count = [0]
    def count_forward(module, inputs, output):
        forward_count[0] += len(inputs[0])
    model.network.register_forward_hook(count_forward)
    frozen = {key: value for key, value in initial.items() if key.startswith("reference_visible_head.")
              or key.endswith(("running_mean", "running_var", "num_batches_tracked"))}
    out = ROOT / "outputs/real_camera_vm"
    require(not (out / "execution.json").exists(), "Preserve completed/partial execution; do not repeat in place")
    if args.dry_run:
        require(not (out / "preflight.json").exists(), "Preserve earlier preflight")
        x, target, valid = load_case(cases[146])  # Source171: actual partial support.
        from reflection_coverage import supported_segmentation_loss
        with torch.inference_mode():
            loss = supported_segmentation_loss(model.detect(x.cuda()[None]), target.cuda()[None], valid.cuda()[None])
        require(torch.isfinite(loss) and all(torch.equal(t.detach().cpu(), initial[k]) for k, t in model.state_dict().items()), "Dry-run loss/state differs")
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "preflight.json", {"complete": True, "batch_size": 1, "loss": float(loss),
                   "protocol_sha256": sha(protocol_path), "gpu": torch.cuda.get_device_name(), "dependencies": versions,
                   "cuda_available": True, "moment_check": moment_check, "optimizer_constructed": False,
                   "optimizer_updates": 0, "actual_model_forward_images": forward_count[0],
                   "held_out_forward_images": 0, "training_quality_verified": False})
        print("CUDA preflight passed: one batch, zero updates; full pilot still pending", flush=True)
        return
    require((out / "preflight.json").is_file() and read_json(out / "preflight.json")["complete"], "Successful one-batch preflight required")
    require(read_json(out / "preflight.json")["protocol_sha256"] == sha(protocol_path), "Preflight protocol changed")
    start = time.monotonic()
    def wall_check():
        require(time.monotonic() - start < 1800, "Fixed30-minute execution budget exceeded; retain partial results")
    def check_frozen(current):
        require(all(torch.equal(current.state_dict()[key].detach().cpu(), value) for key, value in frozen.items()), "Frozen head/BN state changed")
    write_json(out / "execution.json", {"protocol_sha256": sha(protocol_path), "inventory_sha256": sha(ROOT / "inventory.json"),
               "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "moment_check": moment_check,
               "arms": list(ARMS), "updates_per_arm": 112, "total_update_budget": 336,
               "held_out_forward_images": 0, "source_gate_failure_unchanged": True, "promoted": False})
    baseline = measure(model, cases, pixels, out / "initial_masks", wall_check)
    write_json(out / "baseline.json", baseline)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    final, checkpoint_hashes = {}, {}
    for arm in ARMS:
        wall_check()
        torch.manual_seed(42)
        model, payload = load_adapter(assets["model"], "cuda")
        model.network.register_forward_hook(count_forward)
        require(all(torch.equal(t.detach().cpu(), initial[k]) for k, t in model.state_dict().items()), "Arm does not start from identical source42")
        optimizer = torch.optim.AdamW(parameter_groups(model), weight_decay=1e-4)
        optimizer.load_state_dict(copy.deepcopy(moments["state"]))
        restored = optimizer.state_dict()
        require(restored["param_groups"] == moments["state"]["param_groups"] and
                all(torch.equal(t.detach().cpu(), moments["state"]["state"][i][key])
                    for i, state in restored["state"].items() for key, t in state.items()), "Inherited moments changed/reset")
        folder = out / arm
        folder.mkdir()
        updates = 0
        for epoch, batches in enumerate(protocol["schedules"][arm], 1):
            model.train()
            for step, batch in enumerate(batches, 1):
                wall_check()
                real_values = [load_case(cases[2 * r["index"] + int(r["degraded"])]) for r in batch["real"]]
                fixture_values = [fixture_tensors(pixels[i]) for i in batch["fixture"]]
                r = tuple(torch.stack([v[j] for v in real_values]).cuda() for j in range(3))
                f = tuple(torch.stack([v[j] for v in fixture_values]).cuda() for j in range(3))
                optimizer.zero_grad(set_to_none=True)
                logits = model.detect(torch.cat((r[0], f[0])))
                loss, real_loss, replay_loss = expert_loss(logits[:3], *r[1:], logits[3:], *f[1:])
                require(torch.isfinite(loss), "Nonfinite actual training loss")
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(model.network.parameters(), 1., error_if_nonfinite=True))
                optimizer.step()
                updates += 1
                record = {"arm": arm, "experiment_epoch": epoch, "step": step, "experiment_updates": updates,
                          "optimizer_state_step": 672 + updates, "cumulative_model_updates": 882 + updates,
                          "real": batch["real"], "fixture": batch["fixture"], "loss": float(loss.detach()),
                          "real_loss": float(real_loss.detach()), "fixture_loss": float(replay_loss.detach()), "pre_clip_norm": norm}
                with (folder / "steps.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(record) + "\n")
                if step == 1 or step % 14 == 0:
                    print(f"{arm} epoch{epoch}/2 update{updates}/112 loss={record['loss']:.5f} elapsed={time.monotonic()-start:.0f}s", flush=True)
            check_frozen(model)
        require(updates == EPOCHS * STEPS == 112 and len(optimizer.state) == 92
                and all(state["step"].item() == 784 for state in optimizer.state.values()), "Fixed update/moment budget differs")
        final[arm] = measure(model, cases, pixels, folder / "final_masks", wall_check)
        write_json(folder / "metrics.json", final[arm])
        checkpoint = folder / "last.pth"
        torch.save({**payload, "model": model.state_dict(), "epoch": 44, "additional_epoch": 34,
                    "optimizer_updates": 994, "fresh_optimizer_updates": 784, "experiment_updates": 112,
                    "source_camera_checkpoint_counters": protocol["source_counters"],
                    "real_camera_arm": arm, "real_camera_protocol_sha256": sha(protocol_path),
                    "detector_only": True, "source_camera_selection": payload["selection"],
                    "selection": {"selected": False, "training_fit_only": True, "historical_gates_not_evaluated": True}}, checkpoint)
        moment_path = folder / "final_optimizer.pth"
        torch.save({"state": optimizer.state_dict(), "model_sha256": sha(checkpoint), "experiment_updates": 112,
                    "optimizer_state_step": 784, "cumulative_model_updates": 994}, moment_path)
        checkpoint_hashes[arm] = {"model_sha256": sha(checkpoint), "optimizer_sha256": sha(moment_path),
                                 "optimizer_retained_on_vm_not_in_return_archive": True}
        check_frozen(model)
        del model, optimizer, restored
        gc.collect()
        torch.cuda.empty_cache()
    decision = decisions(baseline, final)
    write_json(out / "fit_decision.json", decision)
    preview(cases, protocol["preview_case_ids"], out)
    for role, path in assets.items():
        require(sha(path) == protocol["existing_assets"][role]["sha256"], "Read-only source asset changed")
    wall_check()
    require(forward_count[0] == 1848 + 336 * 8, "Actual forward-image budget differs")
    write_json(out / "complete.json", {"complete": True, "protocol_sha256": sha(protocol_path),
               "updates_per_arm": 112, "total_optimizer_updates": 336, "moment_steps_per_arm": 784,
               "cumulative_model_updates_per_arm": 994, "actual_model_forward_images": forward_count[0],
               "held_out_forward_images": 0, "seconds": time.monotonic() - start,
               "checkpoints": checkpoint_hashes, "fit_decision": decision,
               "original_425_case_gates_evaluated": False, "promoted": False, "selected_checkpoint": None,
               "preview_sha256": sha(out / "preview.png"), "original_assets_unchanged": True})
    files = {p.relative_to(ROOT).as_posix(): p for p in out.rglob("*") if p.is_file() and p.name != "final_optimizer.pth"}
    files["inputs/real_camera_protocol.json"] = protocol_path
    files["bundle_inventory.json"] = ROOT / "inventory.json"
    return_inventory = {name: sha(path) for name, path in sorted(files.items())}
    write_json(out / "return_inventory.json", return_inventory)
    files[(out / "return_inventory.json").relative_to(ROOT).as_posix()] = out / "return_inventory.json"
    archive = ROOT / "real-camera-results.tar.gz"
    require(not archive.exists(), "Preserve existing return archive")
    with tarfile.open(archive, "x:gz") as tar:
        for name, path in sorted(files.items()):
            tar.add(path, arcname="real_camera_vm_results/" + name, recursive=False)
    with archive.with_name(archive.name + ".sha256").open("x", encoding="ascii", newline="\n") as stream:
        stream.write(sha(archive) + "  " + archive.name + "\n")
    print("Three-arm diagnostic complete. Download real-camera-results.tar.gz and .sha256; no checkpoint selected.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    main(parser.parse_args())
