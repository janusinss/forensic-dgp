"""Finite V6 target-bandwidth pilot; all backward paths require the Linux L4 VM."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cctv_dgp_targets_v6 import ARMS, START_STATE, arm_protocol, require_vm, verify
from cctv_dgp_pilot import (FixedObservedIdentity, PilotDataset, PilotPerceptual, aggregate,
    base_loss, buffer_hash, composite, exported_pixel_metrics, freeze_normalization,
    grid112, identity_crop, image_tensor, qualifies, sha, state_hash, write)
from cctv_dgp_frozen_norm import install_frozen_instance_norm
from models import DGPSynthesizer

CONTEXT = {"updates": 0, "backward_calls": 0, "autograd_calls": 0}


def check_clock(deadline):
    if time.monotonic() > deadline:
        raise TimeoutError("Frozen 20-minute V6 budget exceeded")


def device_batch(batch):
    for key in ("low", "target", "mask", "grid"):
        batch[key] = batch[key].cuda(non_blocking=True)
    return batch


def save_state(path, model):
    if path.exists():
        raise ValueError("Preserve existing checkpoint")
    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, path)


def preview(root, out, p, rows, stage):
    by_id = {r["id"]: r for r in rows}
    refs = {r["id"]: r for r in p["references"]}
    sheet = Image.new("RGB", (3 * 260, 10 * 288 + 24), (238, 238, 238))
    draw = ImageDraw.Draw(sheet)
    for j, label in enumerate(("input", "raw DGP observed PNG", "common reference")):
        draw.text((j * 260 + 2, 3), label, fill="black")
    for i, case_id in enumerate(p["preview_case_ids"]):
        row, y = by_id[case_id], 24 + i * 288
        for j, path in enumerate((root / row["input"], out / row["prediction"], root / refs[row["reference_id"]]["target"])):
            draw.text((j * 260 + 2, y + 2), case_id, fill="black")
            with Image.open(path) as image:
                sheet.paste(image.convert("RGB"), (j * 260 + 2, y + 28))
    name = f"{stage}/preview_10_rows.png"
    sheet.save(out / name)
    return name


def evaluate(root, out, p, model, identity, embeds, stage, deadline, include_input=False):
    folder = out / stage
    folder.mkdir()
    for name in ("images", "embeddings", "float_preview"):
        (folder / name).mkdir()
    dataset = PilotDataset(root, p, p["validation_cases"])
    model.eval()
    rows, input_rows, artifacts = [], [], {}
    start, calls = time.monotonic(), 0
    for batch in DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0):
        check_clock(deadline)
        batch = device_batch(batch)
        with torch.no_grad():
            network = model(batch["low"])
            calls += 1
            observed = composite(network, batch["low"], batch["mask"])
            quantized = torch.floor(observed * 255).clamp(0, 255) / 255
            generated_embeddings = identity.embedding(quantized, batch["mask"], batch["grid"])
            input_embeddings = identity.embedding(batch["low"], batch["mask"], batch["grid"]) if include_input else None
        for j, index in enumerate(batch["index"].tolist()):
            case = dataset.cases[index]
            ref = dataset.references[case["reference_id"]]
            prediction = (quantized[j].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            target = np.asarray(Image.open(root / ref["target"]).convert("RGB"))
            mask = np.asarray(Image.open(root / ref["observed"])) > 0
            file = f"{stage}/images/{case['id']}.png"
            Image.fromarray(prediction).save(out / file)
            artifacts[file] = sha(out / file)
            embedding = generated_embeddings[j].cpu().numpy()
            embed_file = f"{stage}/embeddings/{case['id']}.npy"
            np.save(out / embed_file, embedding, allow_pickle=False)
            artifacts[embed_file] = sha(out / embed_file)
            rows.append({**case, **exported_pixel_metrics(prediction, target, mask), "prediction": file,
                         "embedding": embed_file, "ArcFace_observed_fixed": float(np.clip(embedding @ embeds[ref["id"]], -1, 1))})
            if case["id"] in p["preview_case_ids"]:
                file = f"{stage}/float_preview/{case['id']}.npz"
                np.savez_compressed(out / file, network_rgb=network[j].permute(1, 2, 0).cpu().numpy(),
                                    observed_rgb=observed[j].permute(1, 2, 0).cpu().numpy())
                artifacts[file] = sha(out / file)
            if include_input:
                rgb = np.asarray(Image.open(root / case["input"]).convert("RGB"))
                embedding = input_embeddings[j].cpu().numpy()
                file = f"{stage}/embeddings/{case['id']}_input.npy"
                np.save(out / file, embedding, allow_pickle=False)
                artifacts[file] = sha(out / file)
                input_rows.append({**case, **exported_pixel_metrics(rgb, target, mask), "embedding": file,
                    "ArcFace_observed_fixed": float(np.clip(embedding @ embeds[ref["id"]], -1, 1))})
        if len(rows) % 80 < 8:
            print(f"{stage} validation {len(rows)}/520 elapsed={time.monotonic()-start:.0f}s", flush=True)
    name = preview(root, out, p, rows, stage)
    artifacts[name] = sha(out / name)
    report = {"complete": True, "rows": rows, "summary": aggregate(rows), "input_rows": input_rows,
              "input_summary": aggregate(input_rows) if include_input else None, "artifact_sha256": artifacts,
              "seconds": time.monotonic() - start, "dgp_batch_forwards": calls, "preview": name,
              "evaluation_basis": "common exact exported RGB PNG; unsupported padding preserved"}
    write(folder / "metrics.json", report)
    return report


def preflight(root, out, p, model, identity, perceptual, deadline):
    check_clock(deadline)
    before, teachers = state_hash(model), {"identity": state_hash(identity), "perceptual": state_hash(perceptual)}
    reports = []
    for arm in ARMS:
        dataset = PilotDataset(root, arm_protocol(p, arm), p["training_epochs"]["1"][:8])
        batch = device_batch(next(iter(DataLoader(dataset, batch_size=8))))
        common_dataset = PilotDataset(root, p, p["training_epochs"]["1"][:8])
        common = device_batch(next(iter(DataLoader(common_dataset, batch_size=8))))
        model.eval()
        generated = composite(model(batch["low"]), batch["low"], batch["mask"])
        reconstruction, _ = base_loss(generated, batch["target"], batch["mask"], perceptual)
        with torch.no_grad():
            target_embed = identity.embedding(common["target"], common["mask"], common["grid"])
        identity_term = (1 - (identity.embedding(generated, batch["mask"], batch["grid"]) * target_embed).sum(1)).mean()
        total = reconstruction + .1 * identity_term
        gradients = torch.autograd.grad(total, tuple(model.parameters()), allow_unused=True)
        CONTEXT["autograd_calls"] += 1
        active = [g for g in gradients if g is not None]
        if not active or not all(torch.isfinite(g).all() for g in active) or sum(float(g.abs().sum()) for g in active) <= 0:
            raise ValueError("Empty/nonfinite V6 preflight gradients")
        if any(parameter.grad is not None for teacher in (model, identity, perceptual) for parameter in teacher.parameters()):
            raise ValueError("Zero-update preflight accumulated parameter gradients")
        if state_hash(model) != before or {"identity": state_hash(identity), "perceptual": state_hash(perceptual)} != teachers:
            raise ValueError("Zero-update preflight changed student/teacher state")
        reports.append({"arm": arm["id"], "loss": float(total), "active_gradient_tensors": len(active),
                        "autograd_calls": 1, "optimizer_constructed": False, "optimizer_updates": 0})
        del gradients, active, total, generated, reconstruction, identity_term
    # Tail batch is seven images: catch PyTorch InstanceNorm state writebacks.
    dataset = PilotDataset(root, p, p["training_epochs"]["1"][-7:])
    batch = device_batch(next(iter(DataLoader(dataset, batch_size=7))))
    with torch.no_grad():
        tail = model(batch["low"])
    if not torch.isfinite(tail).all() or state_hash(model) != before:
        raise ValueError("Seven-image evaluation altered model state")
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
    encoder = ort.InferenceSession(str(root / p["weights"]["arcface"]), sess_options=options, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        crop = identity_crop(batch["target"][:1], batch["mask"][:1], batch["grid"][:1]) * 2 - 1
        actual = identity.encoder(crop).cpu().numpy()
    expected = encoder.run(None, {encoder.get_inputs()[0].name: crop.cpu().numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
    report = {"passed": True, "arms": reports, "batch_size": 8, "tail_batch_size": 7,
        "zero_optimizer_updates": True, "autograd_calls": 2, "backward_calls": 0,
        "starting_state_hash": before, "normalization_state_unchanged": True,
        "teachers": teachers, "gpu": torch.cuda.get_device_name(0),
        "onnx_max_error": float(np.abs(expected - actual).max()),
        "torch": torch.__version__, "full_pilot_pending": True}
    write(out / "preflight.json", report)
    print("V6 CUDA preflight passed: matched targets, batch8/tail7, zero updates", flush=True)
    return teachers


def run(root, out, preflight_only=False):
    require_vm(root)  # Before output creation, model/optimizer construction or backward.
    start = time.monotonic()
    deadline = start + 1200
    p = verify(root)
    if out.exists():
        raise ValueError("Preserve existing V6 output; no automatic resume/repeat")
    out.mkdir(parents=True)
    for name in p["assets_sha256"]:
        if name.endswith((".py", ".sh")):
            target = out / "runtime_sources" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
    torch.set_num_threads(4)
    torch.manual_seed(p["seed"])
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = False, False
    model = DGPSynthesizer().cuda().eval()
    CONTEXT["model"] = model
    if install_frozen_instance_norm(model) != 5:
        raise ValueError("Expected five normalization adapters")
    starting = torch.load(root / p["weights"]["start"], map_location="cpu", weights_only=True)
    model.load_state_dict(starting, strict=True)
    if state_hash(model) != START_STATE:
        raise ValueError("Audited V2 starting tensors differ")
    identity = FixedObservedIdentity(root / p["weights"]["arcface"], "cuda")
    perceptual = PilotPerceptual(root / p["weights"]["vgg_trunk"], "cuda")
    teachers = preflight(root, out, p, model, identity, perceptual, deadline)
    if preflight_only:
        return
    buffers = buffer_hash(model)
    write(out / "execution.json", {"protocol_sha256": sha(root / "targets_protocol_v6.json"),
        "host": platform.node(), "device": "cuda", "started_unix_seconds": time.time(),
        "starting_state_hash": state_hash(model), "initial_buffers_hash": buffers, "teacher_states": teachers,
        "python": sys.version, "package_versions": {k: importlib.metadata.version(k) for k in
            ("torch", "torchvision", "numpy", "Pillow", "onnx", "onnx2torch", "onnxruntime", "scikit-image")},
        "runtime_cap_seconds": 1200, "normalization_frozen": True, "amp": False, "ema": False,
        "grid_sample_cuda_backward_may_be_nondeterministic": True})
    (out / "reference_embeddings").mkdir()
    embeds = {}
    for offset in range(0, len(p["references"]), 8):
        check_clock(deadline)
        refs = p["references"][offset:offset + 8]
        targets = torch.stack([image_tensor(Image.open(root / r["identity_target"]).convert("RGB")) for r in refs]).cuda()
        masks = torch.stack([torch.from_numpy((np.asarray(Image.open(root / r["observed"])) > 0).copy()).float()[None] for r in refs]).cuda()
        grids = torch.stack([torch.from_numpy(grid112(r["matrix112"])) for r in refs]).cuda()
        with torch.no_grad():
            values = identity.embedding(targets, masks, grids).cpu().numpy()
        for ref, embedding in zip(refs, values):
            embeds[ref["id"]] = embedding
            np.save(out / "reference_embeddings" / (ref["id"] + ".npy"), embedding, allow_pickle=False)
    baseline = evaluate(root, out, p, model, identity, embeds, "baseline", deadline, True)
    if state_hash(model) != START_STATE:
        raise ValueError("Baseline evaluation changed starting state")
    branches = []
    for arm in ARMS:
        check_clock(deadline)
        torch.manual_seed(p["seed"])
        model.load_state_dict(starting, strict=True)
        if state_hash(model) != START_STATE:
            raise ValueError("Matched arm starting state differs")
        backbone = list(model.fpn.features.parameters())
        ids = {id(x) for x in backbone}
        optimizer = torch.optim.Adam([{"params": backbone, "lr": 2e-6},
            {"params": [x for x in model.parameters() if id(x) not in ids], "lr": 1e-5}], weight_decay=1e-5)
        branch = out / arm["id"]
        branch.mkdir()
        save_state(branch / "baseline.pth", model)
        best, best_epoch, best_file, updates = baseline["summary"], 0, f"{arm['id']}/baseline.pth", 0
        entries = []
        training_protocol = arm_protocol(p, arm)
        for epoch in (1, 2):
            epoch_start = time.monotonic()
            cases = p["training_epochs"][str(epoch)]
            loader = DataLoader(PilotDataset(root, training_protocol, cases), batch_size=8,
                                shuffle=False, num_workers=2, pin_memory=True)
            model.train()
            freeze_normalization(model)
            for batch in loader:
                check_clock(deadline)
                batch = device_batch(batch)
                optimizer.zero_grad(set_to_none=True)
                generated = composite(model(batch["low"]), batch["low"], batch["mask"])
                reconstruction, components = base_loss(generated, batch["target"], batch["mask"], perceptual)
                teacher_target = torch.from_numpy(np.stack([embeds[r] for r in batch["reference_id"]])).cuda()
                identity_loss = (1 - (identity.embedding(generated, batch["mask"], batch["grid"]) * teacher_target).sum(1)).mean()
                total = reconstruction + .1 * identity_loss
                if not torch.isfinite(total):
                    raise FloatingPointError("Nonfinite V6 objective")
                if updates >= 98 or CONTEXT["updates"] >= 196:
                    raise ValueError("Finite update cap would be exceeded")
                total.backward()
                CONTEXT["backward_calls"] += 1
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
                if any(x.grad is not None for teacher in (identity, perceptual) for x in teacher.parameters()):
                    raise ValueError("Frozen teacher accumulated parameter gradient")
                optimizer.step()
                updates += 1
                CONTEXT["updates"] += 1
                record = {"arm": arm["id"], "epoch": epoch, "update": updates,
                    "cases": list(batch["case_id"]), "loss": float(total), "components": components,
                    "identity_loss": float(identity_loss), "preclip_norm": norm,
                    "backward_calls": 1, "elapsed_total_seconds": time.monotonic() - start}
                with (out / "updates.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                if updates == 16:
                    projected = time.monotonic() - start + (196 - CONTEXT["updates"]) * (time.monotonic() - epoch_start) / 16 + 5 * baseline["seconds"]
                    write(branch / "timing_at16.json", {"projected_seconds": projected, "cap_seconds": 1200})
                    if projected > 1200:
                        raise TimeoutError("Measured timing cannot fit V6 budget")
                if updates == 1 or updates % 16 == 0:
                    print(f"{arm['id']} epoch{epoch}/2 update{updates}/98 loss={float(total):.5f} elapsed={time.monotonic()-start:.0f}s", flush=True)
            model.eval()
            file = f"{arm['id']}/epoch_{epoch}.pth"
            save_state(out / file, model)
            if buffer_hash(model) != buffers:
                raise ValueError("Frozen student buffers changed")
            stage = f"{arm['id']}_epoch{epoch}"
            metrics = evaluate(root, out, p, model, identity, embeds, stage, deadline)
            accepted = qualifies(metrics["summary"], baseline["summary"], best)
            if accepted:
                best, best_epoch, best_file = metrics["summary"], epoch, file
            entry = {"epoch": epoch, "cumulative_updates": updates, "checkpoint": file,
                     "checkpoint_sha256": sha(out / file), "state_hash": state_hash(model),
                     "buffers_hash": buffer_hash(model), "metrics": f"{stage}/metrics.json", "accepted": accepted}
            entries.append(entry)
            write(branch / f"epoch_{epoch}_record.json", entry)
            print(f"{arm['id']} epoch{epoch} saved; metric_qualified={accepted}", flush=True)
        shutil.copyfile(out / best_file, branch / "best.pth")
        selection = {"selected_epoch": best_epoch, "selected_source": best_file, "best_sha256": sha(branch / "best.pth"),
                     "reason": "Metric-eligible research candidate; visual/native review pending" if best_epoch else "No trained candidate passed; retain starting baseline"}
        write(branch / "best_selection.json", selection)
        if updates != 98:
            raise ValueError("Arm update count differs")
        branches.append({"arm": arm, "epochs": entries, "optimizer_updates": updates, "selection": selection})
    if CONTEXT["updates"] != 196 or CONTEXT["backward_calls"] != 196 or CONTEXT["autograd_calls"] != 2:
        raise ValueError("Finite execution count differs")
    if {"identity": state_hash(identity), "perceptual": state_hash(perceptual)} != teachers:
        raise ValueError("Teacher state changed")
    check_clock(deadline)
    artifacts = {f.relative_to(out).as_posix(): sha(f) for f in sorted(out.rglob("*")) if f.is_file()}
    write(out / "results.json", {"complete": True, "protocol_sha256": sha(root / "targets_protocol_v6.json"),
        "branches": branches, "total_optimizer_updates": 196, "training_backward_calls": 196,
        "preflight_autograd_calls": 2, "elapsed_seconds": time.monotonic() - start,
        "teacher_states_unchanged": True, "normalization_buffers_unchanged": True,
        "reference_embedding_files": len(embeds), "artifacts_sha256": artifacts,
        "production_promoted": False, "native_reserved_used": False, "goal_complete": False})
    print("V6 target-bandwidth comparison complete; no app checkpoint promoted", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.verify:
        p = verify(args.root)
        print({"bundle_verified": True, "training_references": 391, "validation_cases": 520, "optimizer_updates": 0})
    else:
        out = args.output or args.root / "outputs/cctv_dgp_targets_v6"
        try:
            run(args.root, out, args.preflight_only)
        except Exception as error:
            if out.exists() and not (out / "failure.json").exists():
                if "model" in CONTEXT:
                    save_state(out / "partial_state.pth", CONTEXT["model"])
                write(out / "failure.json", {"complete": False, "error_type": type(error).__name__, "error": str(error),
                    "optimizer_updates": CONTEXT["updates"], "backward_calls": CONTEXT["backward_calls"],
                    "autograd_calls": CONTEXT["autograd_calls"], "resume_permitted": False})
            raise
