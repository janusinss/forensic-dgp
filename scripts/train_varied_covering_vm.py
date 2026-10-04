"""Linux/CUDA-only finite data comparison; no local training or selection."""
import argparse
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
from native_expert import require_vm_gpu
from supported_real_data import SupportedMasks
from varied_covering_experiment import (ARMS, EPOCHS, STEPS, SOURCE_SHA, require, family,
                                        validate_source, fixed_schedules, fit_decisions, preflight_case)
from scripts.train_native_expert_vm import (sha, read_json, write_json, safe_path, verify_fixture,
                                            dependency_versions, counts, aggregate)
from scripts.train_real_camera_vm import load_case, fixture_tensors


def loss_for_batch(real_logits, real_mask, real_valid, fixture_logits, fixture_mask, fixture_valid):
    from reflection_coverage import supported_segmentation_loss
    require(real_logits.shape[0] == fixture_logits.shape[0] == 4, "Fixed4-real/4-fixture batch required")
    real = supported_segmentation_loss(real_logits, real_mask, real_valid, .25, .1)
    fixture = supported_segmentation_loss(fixture_logits, fixture_mask, fixture_valid, .25, .1)
    return .5 * real + .5 * fixture, real, fixture


@torch.inference_mode()
def measure(model, cases, real_rows, pixels, folder, wall_check):
    model.eval()
    folder.mkdir(parents=True)
    groups, rows, hashes = {}, [], {}
    for pool, iterable in (("real", cases), ("fixture", range(280))):
        (folder / pool).mkdir()
        for item in iterable:
            wall_check()
            if pool == "real":
                case_id, index = item["case_id"], item["real_train_index"]
                x, target, valid = load_case(item)
                if index < 91:
                    group = "old_" + item["condition"]
                elif real_rows[index]["kind"] == "uncovered":
                    group = "cofw_clear_" + item["condition"]
                else:
                    group = "cofw_" + family(real_rows[index]) + "_" + item["condition"]
                name = f"real/{case_id:03d}"
            else:
                x, target, valid = fixture_tensors(pixels[item])
                group, name = "reflection", f"fixture/{item:03d}"
            prediction = (model.detect(x.cuda()[None]).sigmoid()[0, 0] >= .5).cpu()
            scored = counts(prediction, target[0].bool(), valid[0].bool())
            groups.setdefault(group, []).append(scored)
            path = folder / (name + ".png")
            Image.fromarray(prediction.numpy().astype(np.uint8) * 255).save(path)
            hashes[name + ".png"] = sha(path)
            rows.append({"id": name, "group": group, "counts": scored})
    require(len(rows) == 546, "Fixed train-cohort measurement budget differs")
    return {"groups": {k: aggregate(v) for k, v in groups.items()}, "rows": rows,
            "mask_sha256": hashes, "forward_images": 546, "held_out_forward_images": 0,
            "scope": "Exposed training-cohort diagnosis only; added sources are not an unseen holdout"}


def preview(cases, ids, out):
    width, height = 192, 218
    canvas = Image.new("RGB", (5 * width, 42 + height * len(ids)), "#16181c")
    draw = ImageDraw.Draw(canvas)
    for col, title in enumerate(("input", "target core", "camera91 source", *ARMS)):
        draw.text((col * width + 3, 7), title, fill="white")
    draw.text((3, 24), "TRAIN-COHORT MASKS: green TP, red FP, yellow FN, purple ignored; no generated faces.", fill="white")
    for row, case_id in enumerate(ids):
        case = cases[case_id]
        x, target, valid = load_case(case)
        rgb = np.rint(x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        truth, support = target[0].numpy().astype(bool), valid[0].numpy().astype(bool)

        def overlay(prediction):
            color = rgb.copy()
            color[~support] = (165, 85, 215)
            color[truth & prediction & support] = (16, 185, 129)
            color[~truth & prediction & support] = (245, 50, 50)
            color[truth & ~prediction & support] = (255, 220, 40)
            marked = truth | prediction | ~support
            result = rgb.copy()
            result[marked] = (.55 * rgb[marked] + .45 * color[marked]).round().astype(np.uint8)
            return result

        panels = [rgb, overlay(truth)]
        for folder in (out / "initial_masks", *(out / arm / "final_masks" for arm in ARMS)):
            with Image.open(folder / "real" / f"{case_id:03d}.png") as image:
                panels.append(overlay(np.asarray(image) == 255))
        y = 42 + row * height
        for col, panel in enumerate(panels):
            canvas.paste(Image.fromarray(panel).resize((width, width)), (col * width, y))
        draw.text((3, y + width + 3), f"{case['source_id']}/{case['condition']}", fill="white")
    canvas.save(out / "preview.png")


def main(args):
    require_vm_gpu()  # First: refuse Windows/CPU before files, models or optimizer construction.
    require(args.batch_size == (1 if args.dry_run else 8), "Use batch1 only for dry run; full fixed batch8")
    out = ROOT / "outputs/varied_covering_vm"
    require(not (out / "execution.json").exists() and not (out / "complete.json").exists(),
            "Preserve executed/partial pilot; never restart in place")
    protocol_path = ROOT / "inputs/varied_covering_protocol.json"
    protocol = read_json(protocol_path)
    inventory = read_json(ROOT / "inventory.json")
    require(all(sha(safe_path(ROOT, path)) == digest for path, digest in inventory.items()), "Bundle asset changed")
    require(protocol["arms"] == list(ARMS) and protocol["updates_per_arm"] == EPOCHS * STEPS == 128
            and protocol["optimizer_reset"] is True and protocol["held_out_forward_images"] == 0
            and protocol["original_425_case_gates_evaluated"] is False and protocol["promoted"] is False,
            "Frozen diagnostic scope differs")
    coverage = Path(args.existing).expanduser().resolve() if args.existing else ROOT.parent / "coverage_vm_bundle"
    camera = Path(args.camera).expanduser().resolve() if args.camera else ROOT.parent / "real_camera_vm_bundle"
    roots = {"coverage": coverage, "camera": camera}
    assets = {role: safe_path(roots[row["root"]], row["vm_path"]) for role, row in protocol["existing_assets"].items()}
    for role, path in assets.items():
        require(sha(path) == protocol["existing_assets"][role]["sha256"], "Read-only source asset changed: " + role)
    require(sha(assets["model"]) == SOURCE_SHA, "Fixed camera91 source changed")
    require(all((ROOT.parent / "dataset" / name).is_dir() for name in ("asian_faces", "thumbnails128x128")),
            "Require original repository dataset directories")
    split_path = ROOT / "inputs/phase4_split.json"
    repository_split = ROOT.parent / "outputs/phase4_with_progress/split.json"
    require(repository_split.is_file() and sha(repository_split) == sha(split_path) == protocol["phase4_split_sha256"],
            "Original VM split missing/changed")
    split = read_json(split_path)
    training = {p.replace("\\", "/") for p in split["train"]}
    require(not training.intersection(p.replace("\\", "/") for p in split["validation"]), "Original split overlap")
    registry_path = ROOT / "dataset/detector_supported_review_v3/manifest.json"
    require(sha(registry_path) == protocol["registry_sha256"], "Reviewed registry changed")
    real = SupportedMasks(registry_path, split="train")
    parent = read_json(ROOT / "inputs/parent_supported_manifest.json")
    require(real.metadata["supported_records"][:123] == parent["supported_records"] and len(real) == 133,
            "Parent records/train-only extension differ")
    for row in real.metadata["supported_records"]:
        raw = safe_path(ROOT, row["source"].replace("\\", "/"))
        require(sha(raw) == row["source_sha256"], "Registered raw-source provenance changed")
    fixture_rows = read_json(assets["fixture_manifest"])["cases"]
    require(all(r["source"].replace("\\", "/") in training for r in real.rows[73:83] + fixture_rows),
            "Original native/fixture source outside Phase4 training")
    require(fixed_schedules(real.rows, fixture_rows) == protocol["schedules"], "Frozen schedule differs")
    pairs_path = ROOT / "outputs/cofw_camera_pairs_v2/manifest.json"
    require(sha(pairs_path) == protocol["pairs_sha256"], "Paired input cache changed")
    pairs = read_json(pairs_path)
    require(pairs["registry_sha256"] == sha(registry_path) and len(pairs["cases"]) == 266
            and pairs["held_out_inputs_in_cache"] == 0, "Fixed paired inputs differ")
    cases = pairs["cases"]
    preflight = preflight_case(cases)
    require(preflight["case_id"] == protocol["preflight_case_id"], "Frozen preflight source differs")
    torch.set_num_threads(4)
    cache = torch.load(assets["fixture_pixels"], map_location="cpu", weights_only=True)
    require(cache["format"] == "dgp-reflection-coverage-pixels-v1" and set(cache["pixels"]) == set(range(280)), "Fixture cache differs")
    pixels = cache["pixels"]
    for i, row in enumerate(fixture_rows):
        verify_fixture(pixels[i], row)
    deps = coverage / "outputs/face_occlusion_dependencies"
    sys.path.insert(0, str(deps))
    versions = dependency_versions(deps)
    require(sha(deps / "setup.json") == protocol["setup_sha256"] and str(torch.__version__) == "2.9.1+cu129",
            "Pinned existing VM runtime differs")
    require(torch.cuda.get_device_properties(0).total_memory >= 4 * 1024**3
            and torch.cuda.mem_get_info()[0] >= 2 * 1024**3, "Require4GiB total/2GiB free VRAM")
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    from face_occlusion_adapter import load_adapter, parameter_groups
    model, payload = load_adapter(assets["model"], "cuda")
    validate_source(payload)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    frozen = {k: v for k, v in initial.items() if k.startswith("reference_visible_head.")
              or k.endswith(("running_mean", "running_var", "num_batches_tracked"))}
    forward_count = [0]

    def count_forward(module, inputs, output):
        forward_count[0] += len(inputs[0])

    model.network.register_forward_hook(count_forward)
    if args.dry_run:
        require(not out.exists(), "Preserve previous preflight/partial output")
        x, mask, valid = load_case(preflight)  # Explicit native hair source868, never an inferred positional index.
        from reflection_coverage import supported_segmentation_loss
        with torch.inference_mode():
            loss = supported_segmentation_loss(model.detect(x.cuda()[None]), mask.cuda()[None], valid.cuda()[None])
        require(torch.isfinite(loss) and all(torch.equal(t.detach().cpu(), initial[k]) for k, t in model.state_dict().items()),
                "Dry-run loss/model state differs")
        out.mkdir(parents=True)
        write_json(out / "preflight.json", {"complete": True, "batch_size": 1, "loss": float(loss),
                   "protocol_sha256": sha(protocol_path), "gpu": torch.cuda.get_device_name(), "dependencies": versions,
                   "cuda_available": True, "fresh_optimizer_planned": True, "optimizer_constructed": False,
                   "optimizer_updates": 0, "actual_model_forward_images": forward_count[0], "held_out_forward_images": 0})
        print("CUDA preflight passed: one batch, zero updates; finite varied-covering pilot pending", flush=True)
        return
    require((out / "preflight.json").is_file() and read_json(out / "preflight.json")["complete"]
            and read_json(out / "preflight.json")["protocol_sha256"] == sha(protocol_path), "Matching one-batch preflight required")
    start = time.monotonic()

    def wall_check():
        require(time.monotonic() - start < 1800, "Fixed30-minute training/measurement cap exceeded; preserve partial results")

    def check_frozen(current):
        require(all(torch.equal(current.state_dict()[k].detach().cpu(), v) for k, v in frozen.items()), "Frozen head/BN state changed")

    write_json(out / "execution.json", {"protocol_sha256": sha(protocol_path), "inventory_sha256": sha(ROOT / "inventory.json"),
               "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "arms": list(ARMS),
               "updates_per_arm": 128, "total_update_budget": 256, "optimizer_reset": True,
               "source_model_updates": 994, "fresh_moment_start_step": 0, "held_out_forward_images": 0, "promoted": False})
    baseline = measure(model, cases, real.rows, pixels, out / "initial_masks", wall_check)
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
        require(all(torch.equal(t.detach().cpu(), initial[k]) for k, t in model.state_dict().items()), "Branch initialization differs")
        optimizer = torch.optim.AdamW(parameter_groups(model), weight_decay=1e-4)
        require(not optimizer.state, "Both branches must start with empty fresh optimizer state")
        folder = out / arm
        folder.mkdir()
        updates = 0
        for epoch, batches in enumerate(protocol["schedules"][arm], 1):
            model.train()
            for step, batch in enumerate(batches, 1):
                wall_check()
                rv = [load_case(cases[2 * row["index"] + int(row["degraded"])]) for row in batch["real"]]
                fv = [fixture_tensors(pixels[index]) for index in batch["fixture"]]
                r = tuple(torch.stack([v[j] for v in rv]).cuda() for j in range(3))
                f = tuple(torch.stack([v[j] for v in fv]).cuda() for j in range(3))
                optimizer.zero_grad(set_to_none=True)
                logits = model.detect(torch.cat((r[0], f[0])))
                loss, real_loss, replay_loss = loss_for_batch(logits[:4], *r[1:], logits[4:], *f[1:])
                require(torch.isfinite(loss), "Nonfinite actual training loss")
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(model.network.parameters(), 1., error_if_nonfinite=True))
                optimizer.step()
                updates += 1
                record = {"arm": arm, "experiment_epoch": epoch, "step": step, "experiment_updates": updates,
                          "optimizer_state_step": updates, "cumulative_model_updates": 994 + updates,
                          "real": batch["real"], "fixture": batch["fixture"], "loss": float(loss.detach()),
                          "real_loss": float(real_loss.detach()), "fixture_loss": float(replay_loss.detach()), "pre_clip_norm": norm}
                with (folder / "steps.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(record) + "\n")
                if step == 1 or step % 16 == 0:
                    print(f"{arm} epoch{epoch}/2 update{updates}/128 loss={record['loss']:.5f} elapsed={time.monotonic()-start:.0f}s", flush=True)
            check_frozen(model)
        require(updates == 128 and len(optimizer.state) == 92
                and all(state["step"].item() == 128 for state in optimizer.state.values()), "Fresh update/moment budget differs")
        final[arm] = measure(model, cases, real.rows, pixels, folder / "final_masks", wall_check)
        write_json(folder / "metrics.json", final[arm])
        checkpoint = folder / "last.pth"
        torch.save({**payload, "model": model.state_dict(), "epoch": 46, "additional_epoch": 36,
                    "optimizer_updates": 1122, "fresh_optimizer_updates": 128, "experiment_updates": 128,
                    "source_varied_checkpoint_counters": protocol["source_counters"], "optimizer_reset": True,
                    "varied_covering_arm": arm, "varied_covering_protocol_sha256": sha(protocol_path), "detector_only": True,
                    "source_varied_selection": payload["selection"],
                    "selection": {"selected": False, "training_fit_only": True, "historical_gates_not_evaluated": True}}, checkpoint)
        moments = folder / "final_optimizer.pth"
        torch.save({"state": optimizer.state_dict(), "model_sha256": sha(checkpoint), "experiment_updates": 128,
                    "optimizer_state_step": 128, "cumulative_model_updates": 1122, "optimizer_reset": True}, moments)
        checkpoint_hashes[arm] = {"model_sha256": sha(checkpoint), "optimizer_sha256": sha(moments),
                                 "optimizer_retained_on_vm_not_in_return_archive": True}
        check_frozen(model)
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    decision = fit_decisions(final)
    write_json(out / "fit_decision.json", decision)
    preview(cases, protocol["preview_case_ids"], out)
    for role, path in assets.items():
        require(sha(path) == protocol["existing_assets"][role]["sha256"], "Read-only asset changed: " + role)
    require(all(sha(safe_path(ROOT, path)) == digest for path, digest in inventory.items()),
            "Bundled consumed code/data changed during execution")
    wall_check()
    require(forward_count[0] == 1638 + 256 * 8 == 3686, "Actual forward-image budget differs")
    write_json(out / "complete.json", {"complete": True, "protocol_sha256": sha(protocol_path),
               "updates_per_arm": 128, "total_optimizer_updates": 256, "moment_steps_per_arm": 128,
               "cumulative_model_updates_per_arm": 1122, "optimizer_reset": True,
               "actual_model_forward_images": forward_count[0], "held_out_forward_images": 0,
               "seconds": time.monotonic() - start, "checkpoints": checkpoint_hashes,
               "fit_decision": decision, "original_425_case_gates_evaluated": False,
               "promoted": False, "selected_checkpoint": None,
               "preview_sha256": sha(out / "preview.png"), "original_assets_unchanged": True})
    files = {p.relative_to(ROOT).as_posix(): p for p in out.rglob("*") if p.is_file() and p.name != "final_optimizer.pth"}
    files["inputs/varied_covering_protocol.json"] = protocol_path
    files["bundle_inventory.json"] = ROOT / "inventory.json"
    inventory = {name: sha(path) for name, path in sorted(files.items())}
    write_json(out / "return_inventory.json", inventory)
    files[(out / "return_inventory.json").relative_to(ROOT).as_posix()] = out / "return_inventory.json"
    archive = ROOT / "varied-covering-results.tar.gz"
    require(not archive.exists(), "Preserve existing return archive")
    with tarfile.open(archive, "x:gz") as tar:
        for name, path in sorted(files.items()):
            tar.add(path, arcname="varied_covering_vm_results/" + name, recursive=False)
    with archive.with_name(archive.name + ".sha256").open("x", encoding="ascii", newline="\n") as stream:
        stream.write(sha(archive) + "  " + archive.name + "\n")
    print("Finite varied-covering diagnostic complete. Download varied-covering-results.tar.gz and .sha256; no checkpoint selected.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing")
    parser.add_argument("--camera")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    main(parser.parse_args())
