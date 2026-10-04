"""Isolated two-pair blur fitting: prepare locally; actual fitting on the L4 VM.

Fixed training-fit endpoints do not select a production checkpoint. Existing
validation/native images are neither copied nor forwarded by this diagnostic.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time

PARENT_SHA = "a1d2fbb2c543c95ec1d0c7afe4670d74e7ad8009a8608daf5a2cce6209540b61"
DESIGN_SHA = "1537890efea27da622e495d2b071cbd770c54a5e160a4f847b6a8af39a76e9ac"
START_STATE = "d89f4e52777b5260e94a3e533aa21a0204cbd79b578137881fc6a4c67f1bb0c3"
TRAIN_IDS = ["e1_tr_ffhq_00178", "e1_tr_ffhq_01210"]
CONTEXT = {"out": None, "model": None, "updates": 0, "backwards": 0}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify(root, weights=True):
    root = Path(root)
    p = read(root / "blur_fit_protocol_v8.json")
    assert (root / "blur_fit_protocol_v8.sha256").read_text().strip() == sha(root / "blur_fit_protocol_v8.json")
    assert sha(root / "lineage/capacity_protocol_v7.json") == PARENT_SHA
    assert sha(root / "lineage/proposed_protocol_v8.json") == DESIGN_SHA
    parent = read(root / "lineage/capacity_protocol_v7.json")
    design = read(root / "lineage/proposed_protocol_v8.json")
    assert p["parent_protocol_sha256"] == PARENT_SHA and p["design_sha256"] == DESIGN_SHA
    assert p["references"] == parent["references"] and p["cases"] == parent["cases"] == design["evaluation_cases"]
    assert len(p["cases"]) == 10 and len(p["references"]) == 10
    assert all(r["role"] == "train" for r in p["references"])
    selected = [c for c in p["cases"] if c["profile"] == "blur_lr24"]
    assert p["training_cases"] == selected == design["training_cases"]
    assert [c["id"] for c in selected] == TRAIN_IDS
    assert p["objective"] == "per-image observed RGB squared error mean"
    assert p["optimizer"] == "fresh Adam" and p["seed"] == parent["seed"] == 107
    assert p["lr"] == design["lr"] == {"backbone": 2e-5, "other": 1e-4}
    assert p["weight_decay"] == 1e-5 and p["clip_norm"] == 1.0 and p["batch_size"] == 2
    assert p["updates"] == design["updates"] == 1000 and p["snapshots"] == [20, 100, 1000]
    assert p["runtime_cap_seconds"] == design["trainer_cap_seconds"] == 600
    assert p["supervisor_cap_seconds"] == 900 and p["fixed_final_fit_mse_ratio"] == 0.8
    assert p["weights"] == parent["weights"] and p["starting_state_hash"] == START_STATE
    assert not any(p[k] for k in ("validation_used", "native_used", "native_reserved_used", "promotion_permitted", "best_checkpoint_selection_permitted"))
    for name, pin in p["assets_sha256"].items():
        path = root / name
        assert not Path(name).is_absolute() and ".." not in Path(name).parts and ":" not in name and "\\" not in name
        assert path.resolve().is_relative_to(root.resolve())
        if weights or name not in p["cache_assets"]:
            assert sha(path) == pin, "Pinned asset differs: " + name
    for name, pin in parent["assets_sha256"].items():
        assert p["assets_sha256"][name] == pin
    return p


def prepare(parent_root, design_file, out):
    started = time.monotonic()
    assert not out.exists(), "Preserve the existing V8 package"
    assert sha(parent_root / "capacity_protocol_v7.json") == PARENT_SHA
    assert sha(design_file) == DESIGN_SHA
    parent = read(parent_root / "capacity_protocol_v7.json")
    out.mkdir(parents=True)
    pins = dict(parent["assets_sha256"])
    cache = parent["cache_assets"]
    for name, pin in sorted(pins.items()):
        if name in cache:
            continue
        assert sha(parent_root / name) == pin
        destination = out / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent_root / name, destination)
    additions = {
        "lineage/capacity_protocol_v7.json": parent_root / "capacity_protocol_v7.json",
        "lineage/proposed_protocol_v8.json": design_file,
        "scripts/cctv_dgp_blur_fit_v8.py": Path(__file__).resolve(),
        "scripts/audit_cctv_dgp_blur_fit_v8.py": Path(__file__).with_name("audit_cctv_dgp_blur_fit_v8.py"),
        "scripts/run_cctv_dgp_blur_fit_v8_supervised.py": Path(__file__).with_name("run_cctv_dgp_blur_fit_v8_supervised.py"),
    }
    for name, source in additions.items():
        destination = out / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        assert not destination.exists()
        shutil.copyfile(source, destination)
        pins[name] = sha(destination)
    p = {
        "version": 8, "purpose": "Training-only isolated blur fit; no model selection",
        "parent_protocol_sha256": PARENT_SHA, "design_sha256": DESIGN_SHA,
        "references": parent["references"], "cases": parent["cases"],
        "training_cases": [c for c in parent["cases"] if c["profile"] == "blur_lr24"],
        "objective": "per-image observed RGB squared error mean", "optimizer": "fresh Adam",
        "seed": 107, "lr": {"backbone": 2e-5, "other": 1e-4}, "weight_decay": 1e-5,
        "clip_norm": 1.0, "batch_size": 2, "updates": 1000, "snapshots": [20, 100, 1000],
        "runtime_cap_seconds": 600, "supervisor_cap_seconds": 900,
        "fixed_final_fit_mse_ratio": 0.8, "criterion_profile": "blur_lr24",
        "criterion_is_training_fit_only": True, "matched_v7_pair_exposure_at_update": 20,
        "starting_state_hash": START_STATE, "weights": parent["weights"],
        "cache_assets": cache, "assets_sha256": pins,
        "validation_used": False, "native_used": False, "native_reserved_used": False,
        "promotion_permitted": False, "best_checkpoint_selection_permitted": False,
        "training_device": "existing NVIDIA L4 forensic-dgp-thesis VM",
        "hypothesis_limits": "V7 profile/case/exposure confounds prevent sole causal attribution; fitting two images does not establish generalization",
    }
    write(out / "blur_fit_protocol_v8.json", p)
    (out / "blur_fit_protocol_v8.sha256").write_text(sha(out / "blur_fit_protocol_v8.json") + "\n", encoding="ascii", newline="\n")
    verify(out, weights=False)
    from PIL import Image
    for case in p["cases"]:
        with Image.open(out / case["input"]) as image:
            assert image.size == (256, 256) and image.mode == "RGB"
    archive = out.parent / "cctv-dgp-blur-fit-v8.tar.gz"
    with tarfile.open(archive, "x:gz") as stream:
        stream.add(out, arcname=out.name)
    Path(str(archive) + ".sha256").write_text(sha(archive) + "  " + archive.name + "\n", encoding="ascii", newline="\n")
    receipt = {"complete": True, "protocol_sha256": sha(out / "blur_fit_protocol_v8.json"),
        "archive_sha256": sha(archive), "bytes": archive.stat().st_size, "seconds": time.monotonic() - started,
        "training_pairs": 2, "training_only_evaluation_pairs": 10, "expected_vm_updates": 1000,
        "copied_exact_assets": len(pins) - len(cache), "local_model_forwards": 0,
        "local_backward_calls": 0, "local_optimizer_updates": 0, "native_used": False}
    write(out.parent / "cctv_dgp_blur_fit_v8_preparation.json", receipt)
    print(receipt, flush=True)


def train(root):
    sys.path.insert(0, str(root))
    from cctv_dgp_targets_v6 import require_vm
    require_vm(root)  # Guard before model, backward, optimizer or output creation.
    p = verify(root)
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from torch.utils.data import DataLoader
    from models import DGPSynthesizer
    from cctv_dgp_frozen_norm import install_frozen_instance_norm
    from cctv_dgp_pilot import (PilotDataset, FixedObservedIdentity, PilotPerceptual,
        aggregate, buffer_hash, composite, exported_pixel_metrics, freeze_normalization, state_hash)
    started = time.monotonic()
    deadline = started + 600
    out = root / "outputs/cctv_dgp_blur_fit_v8"
    assert not out.exists(), "Preserve existing outputs; no automatic resume/repeat"
    out.mkdir(parents=True)
    CONTEXT["out"] = out
    torch.set_num_threads(4)
    torch.manual_seed(p["seed"])
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = False, False
    model = DGPSynthesizer().cuda().eval()
    CONTEXT["model"] = model
    assert install_frozen_instance_norm(model) == 5
    starting = torch.load(root / p["weights"]["start"], map_location="cpu", weights_only=True)
    model.load_state_dict(starting, strict=True)
    assert state_hash(model) == START_STATE
    buffers = buffer_hash(model)
    identity = FixedObservedIdentity(root / p["weights"]["arcface"], "cuda")
    perceptual = PilotPerceptual(root / p["weights"]["vgg_trunk"], "cuda")
    teacher_states = [state_hash(identity), state_hash(perceptual)]
    dataset = PilotDataset(root, p, p["cases"])
    fitting = PilotDataset(root, p, p["training_cases"])

    def clock():
        if time.monotonic() > deadline:
            raise TimeoutError("600-second V8 trainer cap exceeded")

    def device(batch):
        return {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def objective(generated, batch):
        values = ((generated - batch["target"]).square() * batch["mask"]).sum((1, 2, 3)) / (3 * batch["mask"].sum((1, 2, 3)))
        return values.mean()

    fixed_batch = device(next(iter(DataLoader(fitting, batch_size=2))))
    assert list(fixed_batch["case_id"]) == TRAIN_IDS
    loss = objective(composite(model(fixed_batch["low"]), fixed_batch["low"], fixed_batch["mask"]), fixed_batch)
    gradients = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
    active = [g for g in gradients if g is not None]
    assert active and all(torch.isfinite(g).all() for g in active) and sum(float(g.abs().sum()) for g in active) > 0
    assert state_hash(model) == START_STATE and buffer_hash(model) == buffers
    assert all(x.grad is None for x in model.parameters())
    write(out / "preflight.json", {"passed": True, "autograd_calls": 1, "zero_optimizer_updates": True,
        "active_gradient_tensors": len(active), "training_cases": TRAIN_IDS,
        "starting_state_hash": START_STATE, "buffers_unchanged": True})
    del gradients, active, loss
    write(out / "execution.json", {"protocol_sha256": sha(root / "blur_fit_protocol_v8.json"),
        "starting_state_hash": START_STATE, "host": __import__("platform").node(),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "batch_size": 2,
        "runtime_cap_seconds": 600, "source_sha256": sha(Path(__file__)), "teacher_states": teacher_states,
        "validation_used": False, "native_used": False, "production_promoted": False})
    shutil.copyfile(Path(__file__), out / "executed_source.py")
    with torch.no_grad():
        target_embeds = {}
        for batch in DataLoader(dataset, batch_size=2):
            clock(); batch = device(batch)
            embeds = identity.embedding(batch["target"], batch["mask"], batch["grid"]).cpu().numpy()
            for ref, embed in zip(batch["reference_id"], embeds):
                target_embeds[ref] = embed
                file = out / "target_embeddings" / (ref + ".npy")
                file.parent.mkdir(exist_ok=True)
                np.save(file, embed, allow_pickle=False)

    def evaluate(stage):
        evaluation_start = time.monotonic()
        model.eval()
        rows = []
        for batch in DataLoader(dataset, batch_size=2):
            clock(); batch = device(batch)
            with torch.no_grad():
                predictions = composite(model(batch["low"]), batch["low"], batch["mask"]).cpu().numpy().transpose(0, 2, 3, 1)
                rgbs = np.clip(predictions * 255, 0, 255).astype(np.uint8)
                supports = batch["mask"].cpu().numpy()[:, 0] > 0
                for i, index in enumerate(batch["index"].cpu().tolist()):
                    input_rgb = np.asarray(Image.open(root / dataset.cases[index]["input"]).convert("RGB"))
                    rgbs[i][~supports[i]] = input_rgb[~supports[i]]
                quantized = torch.from_numpy((rgbs.astype(np.float32) / 255).transpose(0, 3, 1, 2)).cuda()
                embeds = identity.embedding(quantized, batch["mask"], batch["grid"]).cpu().numpy()
            for i, index in enumerate(batch["index"].cpu().tolist()):
                case = dataset.cases[index]; ref = dataset.references[case["reference_id"]]
                target = np.asarray(Image.open(root / ref["target"]).convert("RGB"))
                mask = np.asarray(Image.open(root / ref["observed"])) > 0
                file = out / stage / "images" / (case["id"] + ".png")
                file.parent.mkdir(parents=True, exist_ok=True); Image.fromarray(rgbs[i]).save(file)
                embedfile = out / stage / "embeddings" / (case["id"] + ".npy")
                embedfile.parent.mkdir(exist_ok=True); np.save(embedfile, embeds[i], allow_pickle=False)
                rawfile = out / stage / "raw_float" / (case["id"] + ".npy")
                rawfile.parent.mkdir(exist_ok=True); np.save(rawfile, predictions[i], allow_pickle=False)
                rows.append({**case, "source": ref["source"], **exported_pixel_metrics(rgbs[i], target, mask),
                    "ArcFace_observed_fixed": float(np.clip(embeds[i] @ target_embeds[ref["id"]], -1, 1)),
                    "prediction": file.relative_to(out).as_posix(), "embedding": embedfile.relative_to(out).as_posix(),
                    "raw_float": rawfile.relative_to(out).as_posix()})
        sheet = Image.new("RGB", (780, 24 + 288 * 10), "#eeeeee"); draw = ImageDraw.Draw(sheet)
        for j, title in enumerate(["fixed training input", "training-only DGP output", "canonical HQ target"]):
            draw.text((j * 260 + 2, 3), title, fill="black")
        for i, row in enumerate(rows):
            ref = dataset.references[row["reference_id"]]
            for j, file in enumerate([root / row["input"], out / row["prediction"], root / ref["target"]]):
                y = 24 + i * 288
                draw.text((j * 260 + 2, y + 2), row["id"], fill="black")
                sheet.paste(Image.open(file).convert("RGB"), (j * 260 + 2, y + 28))
        preview = out / stage / "preview_10_rows.png"; sheet.save(preview)
        result = {"rows": rows, "summary": aggregate(rows), "preview": preview.relative_to(out).as_posix(),
            "training_only": True, "seconds": time.monotonic() - evaluation_start}
        write(out / stage / "metrics.json", result)
        return result

    baseline = evaluate("baseline")
    assert state_hash(model) == START_STATE
    backbone = list(model.fpn.features.parameters()); ids = {id(x) for x in backbone}
    optimizer = torch.optim.Adam([
        {"params": backbone, "lr": p["lr"]["backbone"]},
        {"params": [x for x in model.parameters() if id(x) not in ids], "lr": p["lr"]["other"]},
    ], weight_decay=p["weight_decay"])
    fitting_started = time.monotonic(); snapshots = []
    for update in range(1, 1001):
        clock(); model.train(); freeze_normalization(model); optimizer.zero_grad(set_to_none=True)
        loss = objective(composite(model(fixed_batch["low"]), fixed_batch["low"], fixed_batch["mask"]), fixed_batch)
        assert torch.isfinite(loss)
        loss.backward(); CONTEXT["backwards"] += 1
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
        assert norm > 0 and all(x.grad is None for teacher in (identity, perceptual) for x in teacher.parameters())
        optimizer.step(); CONTEXT["updates"] += 1
        row = {"update": update, "cases": TRAIN_IDS, "loss": float(loss.detach()),
            "components": {"pixel_mse": float(loss.detach())}, "preclip_norm": norm}
        with (out / "updates.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if update == 16:
            projected = time.monotonic() - started + (1000 - update) * (time.monotonic() - fitting_started) / 16 + 3 * baseline["seconds"]
            write(out / "timing_at16.json", {"projected_seconds": projected, "cap_seconds": 600})
            if projected > 600:
                raise TimeoutError("Measured V8 projection exceeds its budget")
        if update == 1 or update % 100 == 0:
            print(f"isolated_blur update{update}/1000 loss={row['loss']:.6f} elapsed={time.monotonic()-started:.0f}s", flush=True)
        if update in p["snapshots"]:
            assert buffer_hash(model) == buffers
            stage = f"update{update}"; evaluate(stage)
            file = out / "checkpoints" / (f"update_{update}.pth"); file.parent.mkdir(exist_ok=True)
            torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, file)
            snapshots.append({"update": update, "checkpoint": file.relative_to(out).as_posix(),
                "state_hash": state_hash(model), "metrics": stage + "/metrics.json"})
    assert CONTEXT["updates"] == CONTEXT["backwards"] == 1000 and buffer_hash(model) == buffers
    assert [state_hash(identity), state_hash(perceptual)] == teacher_states
    verify(root)
    final = read(out / "update1000/metrics.json")["summary"]
    key = "dataset/thumbnails128x128/blur_lr24"
    ratio = final[key]["MSE"] / baseline["summary"][key]["MSE"]
    result = {"complete": True, "protocol_sha256": sha(root / "blur_fit_protocol_v8.json"),
        "total_optimizer_updates": 1000, "training_backward_calls": 1000, "preflight_autograd_calls": 1,
        "elapsed_seconds": time.monotonic() - started, "snapshots": snapshots,
        "fixed_final_blur_training_mse_ratio": ratio, "training_fit_observed": ratio <= 0.8,
        "artifacts_sha256": {f.relative_to(out).as_posix(): sha(f) for f in sorted(out.rglob("*")) if f.is_file()},
        "buffers_and_teachers_unchanged": True, "training_only": True, "validation_used": False,
        "native_used": False, "native_reserved_used": False, "production_promoted": False,
        "model_improvement_established": False}
    write(out / "results.json", result)
    print({k: v for k, v in result.items() if k not in ("artifacts_sha256", "snapshots")}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--parent-root", type=Path)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.parent_root, args.design, args.root)
    else:
        try:
            train(args.root.resolve())
        except Exception as error:
            out = CONTEXT["out"]
            if out is not None and out.exists() and not (out / "failure.json").exists():
                failure = {"complete": False, "error_type": type(error).__name__, "error": str(error),
                    "optimizer_updates_recorded": CONTEXT["updates"], "backward_calls_recorded": CONTEXT["backwards"],
                    "resume_permitted": False, "production_promoted": False}
                if CONTEXT["model"] is not None:
                    try:
                        import torch
                        file = out / "partial_state.pth"
                        torch.save({k: v.detach().cpu().clone() for k, v in CONTEXT["model"].state_dict().items()}, file)
                        failure["partial_state_sha256"] = sha(file)
                    except Exception as export_error:
                        failure["partial_export_error"] = str(export_error)
                write(out / "failure.json", failure)
            raise
