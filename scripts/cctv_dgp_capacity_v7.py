"""Finite training-only fit diagnostic; preparation/audit local, training on L4.

Two matched arms fit ten existing approved training pairs, two per camera profile.
This probes learnability at a larger diagnostic learning rate, not generalization.
There is deliberately no best.pth, validation/native forwarding or promotion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time

PARENT_SHA = "0fe7d036bf8567bf0860790df553afd627ac50bd5096cfda29cc7d09d71d320c"
START_STATE = "d89f4e52777b5260e94a3e533aa21a0204cbd79b578137881fc6a4c67f1bb0c3"
ARMS = ["pixel_mse", "retained_objective"]
SEED = 107
CONTEXT = {"out": None, "model": None, "updates": 0, "backwards": 0}


def sha(path):
    d = hashlib.sha256()
    with path.open("rb") as stream:
        for b in iter(lambda: stream.read(1024*1024), b""):
            d.update(b)
    return d.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def selected(parent):
    cases = parent["training_epochs"]["1"]
    rows = []
    for profile in sorted({c["profile"] for c in cases}):
        rows.extend(sorted((c for c in cases if c["profile"] == profile), key=lambda c: c["id"])[:2])
    assert len(rows) == 10 and len({c["reference_id"] for c in rows}) == 10
    return rows


def verify(root, weights=True):
    p = read(root / "capacity_protocol_v7.json")
    assert (root / "capacity_protocol_v7.sha256").read_text().strip() == sha(root / "capacity_protocol_v7.json")
    assert sha(root / "lineage/targets_protocol_v6.json") == PARENT_SHA
    parent = read(root / "lineage/targets_protocol_v6.json")
    assert p["cases"] == selected(parent)
    ids = {c["reference_id"] for c in p["cases"]}
    expected = [r for r in parent["references"] if r["id"] in ids]
    assert p["references"] == expected and all(r["role"] == "train" for r in expected)
    assert p["arms"] == ARMS and p["seed"] == SEED and p["updates_per_arm"] == 100 and p["batch_size"] == 2
    assert p["lr"] == {"backbone": 2e-5, "other": 1e-4} and p["snapshots"] == [25, 50, 100]
    assert p["runtime_cap_seconds"] == 600 and p["learnability_mse_ratio"] == .8
    assert p["weight_decay"] == 1e-5 and p["clip_norm"] == 1.
    assert p["learnability_groups"] == ["blur_lr24", "motion_lr48"]
    assert not any(p[k] for k in ("validation_used", "native_used", "production_promotion_permitted"))
    assert p["weights"] == parent["weights"]
    for name, pin in p["assets_sha256"].items():
        path = root / name
        assert not Path(name).is_absolute() and ".." not in Path(name).parts and path.resolve().is_relative_to(root.resolve())
        if weights or name not in p["cache_assets"]:
            assert sha(path) == pin, "Pinned asset differs: " + name
    return p


def prepare(parent_root, out):
    started = time.monotonic()
    assert not out.exists(), "Preserve existing diagnostic package"
    assert sha(parent_root / "targets_protocol_v6.json") == PARENT_SHA
    parent = read(parent_root / "targets_protocol_v6.json")
    cases = selected(parent)
    ids = {c["reference_id"] for c in cases}
    refs = [r for r in parent["references"] if r["id"] in ids]
    out.mkdir(parents=True)
    assets = set(c["input"] for c in cases)
    for ref in refs:
        assets.update(ref[k] for k in ("target", "identity_target", "observed"))
    assets.update(name for name in parent["assets_sha256"] if name.startswith("models/") and name.endswith(".py"))
    assets.update(["cctv_dgp_pilot.py", "cctv_dgp_frozen_norm.py", "cctv_dgp_targets_v6.py"])
    pins = {}
    for name in sorted(assets):
        source = parent_root / name
        assert sha(source) == parent["assets_sha256"][name]
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        pins[name] = sha(target)
    target = out / "lineage/targets_protocol_v6.json"
    target.parent.mkdir()
    shutil.copyfile(parent_root / "targets_protocol_v6.json", target)
    pins["lineage/targets_protocol_v6.json"] = sha(target)
    source = Path(__file__).resolve()
    target = out / "scripts/cctv_dgp_capacity_v7.py"
    target.parent.mkdir()
    shutil.copyfile(source, target)
    pins["scripts/cctv_dgp_capacity_v7.py"] = sha(target)
    cache = {name: parent["assets_sha256"][name] for name in parent["weights"].values()}
    pins.update(cache)
    p = {"version": 7, "purpose": "Training-only loss/learnability diagnostic, not model selection",
         "parent_protocol_sha256": PARENT_SHA, "references": refs, "cases": cases,
         "arms": ARMS, "seed": SEED, "batch_size": 2, "updates_per_arm": 100,
         "snapshots": [25, 50, 100], "lr": {"backbone": 2e-5, "other": 1e-4},
         "weight_decay": 1e-5, "clip_norm": 1., "runtime_cap_seconds": 600,
         "objectives": {"pixel_mse": "per-image observed RGB squared error mean",
                        "retained_objective": "Charbonnier + .05 color + .1 postactivation VGG + .05 Sobel + .1 fixed-observed identity"},
         "learnability_mse_ratio": .8, "learnability_groups": ["blur_lr24", "motion_lr48"],
         "learnability_is_training_fit_only": True, "validation_used": False, "native_used": False,
         "production_promotion_permitted": False, "weights": parent["weights"],
         "cache_assets": cache, "assets_sha256": pins}
    write(out / "capacity_protocol_v7.json", p)
    (out / "capacity_protocol_v7.sha256").write_text(sha(out / "capacity_protocol_v7.json") + "\n", encoding="ascii")
    verify(out, weights=False)
    from PIL import Image
    for c in cases:
        with Image.open(out / c["input"]) as image:
            assert image.size == (256, 256) and image.mode == "RGB"
    archive = out.parent / "cctv-dgp-capacity-v7.tar.gz"
    with tarfile.open(archive, "x:gz") as stream:
        stream.add(out, arcname=out.name)
    Path(str(archive) + ".sha256").write_text(sha(archive) + "  " + archive.name + "\n", encoding="ascii", newline="\n")
    receipt = {"complete": True, "protocol_sha256": sha(out / "capacity_protocol_v7.json"),
               "archive_sha256": sha(archive), "bytes": archive.stat().st_size, "seconds": time.monotonic()-started,
               "training_cases": 10, "training_references": 10, "expected_vm_updates": 200,
               "copied_exact_assets": len(pins)-len(cache), "local_model_forwards": 0,
               "local_backward_calls": 0, "local_optimizer_updates": 0, "native_used": False}
    write(out.parent / "cctv_dgp_capacity_v7_preparation.json", receipt)
    print(receipt, flush=True)


def train(root):
    sys.path.insert(0, str(root))
    from cctv_dgp_targets_v6 import require_vm
    require_vm(root)  # Before any model/optimizer/backward or output creation.
    p = verify(root)
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from torch.utils.data import DataLoader
    from models import DGPSynthesizer
    from cctv_dgp_frozen_norm import install_frozen_instance_norm
    from cctv_dgp_pilot import (PilotDataset, FixedObservedIdentity, PilotPerceptual,
        aggregate, base_loss, buffer_hash, composite, exported_pixel_metrics, freeze_normalization, state_hash)
    started, updates, backwards = time.monotonic(), 0, 0
    out = root / "outputs/cctv_dgp_capacity_v7"
    assert not out.exists(), "Preserve existing diagnostic; no automatic resume/repeat"
    out.mkdir(parents=True)
    CONTEXT["out"] = out
    deadline = started + p["runtime_cap_seconds"]
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
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
    teacher_before = [state_hash(identity), state_hash(perceptual)]
    dataset = PilotDataset(root, p, p["cases"])

    def clock():
        if time.monotonic() > deadline:
            raise TimeoutError("600-second diagnostic cap exceeded")

    def device(batch):
        return {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    with torch.no_grad():
        target_embeds = {}
        for batch in DataLoader(dataset, batch_size=2):
            batch = device(batch)
            e = identity.embedding(batch["target"], batch["mask"], batch["grid"]).cpu().numpy()
            for ref, embed in zip(batch["reference_id"], e):
                target_embeds[ref] = embed
                file = out / "target_embeddings" / (ref + ".npy")
                file.parent.mkdir(exist_ok=True)
                np.save(file, embed, allow_pickle=False)

    def objective(arm, generated, batch):
        if arm == "pixel_mse":
            value = ((generated-batch["target"]).square()*batch["mask"]).sum((1,2,3))/(3*batch["mask"].sum((1,2,3)))
            loss = value.mean()
            return loss, {"pixel_mse": float(loss.detach())}
        reconstruction, components = base_loss(generated, batch["target"], batch["mask"], perceptual)
        wanted = torch.from_numpy(np.stack([target_embeds[r] for r in batch["reference_id"]])).cuda()
        term = (1-(identity.embedding(generated,batch["mask"],batch["grid"])*wanted).sum(1)).mean()
        return reconstruction + .1*term, {**components, "identity": float(term.detach())}

    preflight = []
    for arm in ARMS:
        batch = device(next(iter(DataLoader(dataset, batch_size=2))))
        loss, _ = objective(arm, composite(model(batch["low"]),batch["low"],batch["mask"]), batch)
        gradients = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
        active = [g for g in gradients if g is not None]
        assert active and all(torch.isfinite(g).all() for g in active) and sum(float(g.abs().sum()) for g in active)>0
        assert state_hash(model)==START_STATE and [state_hash(identity),state_hash(perceptual)]==teacher_before
        assert all(x.grad is None for x in model.parameters())
        preflight.append({"arm":arm,"active_gradient_tensors":len(active),"optimizer_updates":0})
        del gradients,active,loss
    write(out/"preflight.json",{"passed":True,"autograd_calls":2,"zero_optimizer_updates":True,"arms":preflight})
    write(out/"execution.json",{"protocol_sha256":sha(root/"capacity_protocol_v7.json"),"starting_state_hash":START_STATE,
        "host":__import__("platform").node(),"torch":torch.__version__,"gpu":torch.cuda.get_device_name(0),
        "batch_size":2,"runtime_cap_seconds":600,"source_sha256":sha(Path(__file__)),"teacher_states":teacher_before,
        "validation_used":False,"native_used":False,"production_promoted":False})
    shutil.copyfile(Path(__file__),out/"executed_source.py")

    def evaluate(stage):
        evaluation_start = time.monotonic()
        model.eval()
        rows = []
        for batch in DataLoader(dataset,batch_size=2):
            clock(); batch=device(batch)
            with torch.no_grad():
                pred=composite(model(batch["low"]),batch["low"],batch["mask"]).cpu().numpy().transpose(0,2,3,1)
                rgb_batch=np.clip(pred*255,0,255).astype(np.uint8)
                supports=batch["mask"].cpu().numpy()[:,0]>0
                for i,index in enumerate(batch["index"].cpu().tolist()):
                    input_rgb=np.asarray(Image.open(root/dataset.cases[index]["input"]).convert("RGB"))
                    rgb_batch[i][~supports[i]]=input_rgb[~supports[i]]
                quantized=torch.from_numpy((rgb_batch.astype(np.float32)/255).transpose(0,3,1,2)).cuda()
                embeds=identity.embedding(quantized,batch["mask"],batch["grid"]).cpu().numpy()
            for i,index in enumerate(batch["index"].cpu().tolist()):
                case=dataset.cases[index]; ref=dataset.references[case["reference_id"]]
                target=np.asarray(Image.open(root/ref["target"]).convert("RGB"))
                mask=np.asarray(Image.open(root/ref["observed"]))>0
                rgb=rgb_batch[i]
                file=out/stage/"images"/(case["id"]+".png"); file.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(rgb).save(file)
                embedfile=out/stage/"embeddings"/(case["id"]+".npy");embedfile.parent.mkdir(exist_ok=True)
                np.save(embedfile,embeds[i],allow_pickle=False)
                rows.append({**case,"source":ref["source"],**exported_pixel_metrics(rgb,target,mask),
                    "ArcFace_observed_fixed":float(np.clip(embeds[i]@target_embeds[ref["id"]],-1,1)),
                    "prediction":file.relative_to(out).as_posix(),"embedding":embedfile.relative_to(out).as_posix()})
        sheet=Image.new("RGB",(780,24+288*10),"#eeeeee");draw=ImageDraw.Draw(sheet)
        for j,title in enumerate(["fixed training input","training-only DGP output","canonical HQ target"]):
            draw.text((j*260+2,3),title,fill="black")
        for i,row in enumerate(rows):
            ref=dataset.references[row["reference_id"]]
            for j,file in enumerate([root/row["input"],out/row["prediction"],root/ref["target"]]):
                y=24+i*288;draw.text((j*260+2,y+2),row["id"],fill="black");sheet.paste(Image.open(file).convert("RGB"),(j*260+2,y+28))
        preview=out/stage/"preview_10_rows.png";sheet.save(preview)
        result={"rows":rows,"summary":aggregate(rows),"preview":preview.relative_to(out).as_posix(),
                "training_only":True,"seconds":time.monotonic()-evaluation_start}
        write(out/stage/"metrics.json",result)
        return result

    baseline=evaluate("baseline")
    assert state_hash(model)==START_STATE
    branches=[]
    for arm in ARMS:
        arm_start=time.monotonic()
        model.load_state_dict(starting,strict=True);torch.manual_seed(SEED)
        backbone=list(model.fpn.features.parameters());ids={id(x) for x in backbone}
        optimizer=torch.optim.Adam([{"params":backbone,"lr":2e-5},{"params":[x for x in model.parameters() if id(x) not in ids],"lr":1e-4}],weight_decay=1e-5)
        snapshots=[]
        for update in range(1,101):
            clock(); cycle=(update-1)//5; offset=((update-1)%5)*2
            order=np.random.default_rng(SEED+cycle).permutation(10).tolist()
            batch=device(next(iter(DataLoader(torch.utils.data.Subset(dataset,order[offset:offset+2]),batch_size=2))))
            model.train();freeze_normalization(model);optimizer.zero_grad(set_to_none=True)
            loss,components=objective(arm,composite(model(batch["low"]),batch["low"],batch["mask"]),batch)
            assert torch.isfinite(loss)
            loss.backward();backwards+=1;CONTEXT["backwards"]+=1
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
            assert norm>0 and all(x.grad is None for teacher in (identity,perceptual) for x in teacher.parameters())
            optimizer.step();updates+=1;CONTEXT["updates"]+=1
            row={"arm":arm,"update":update,"cases":list(batch["case_id"]),"loss":float(loss.detach()),"components":components,"preclip_norm":norm}
            with (out/"updates.jsonl").open("a",encoding="utf-8") as stream: stream.write(json.dumps(row,allow_nan=False)+"\n")
            if update==16:
                projected=time.monotonic()-started+(200-updates)*(time.monotonic()-arm_start)/16+6*baseline["seconds"]
                write(out/arm/"timing_at16.json",{"projected_seconds":projected,"cap_seconds":600})
                if projected>600:
                    raise TimeoutError("Measured projection exceeds diagnostic budget")
            if update==1 or update%25==0:
                print(f"{arm} update{update}/100 loss={row['loss']:.5f} elapsed={time.monotonic()-started:.0f}s",flush=True)
            if update in p["snapshots"]:
                assert buffer_hash(model)==buffers
                stage=f"{arm}_update{update}"; metrics=evaluate(stage)
                weights=out/arm/(f"update_{update}.pth");weights.parent.mkdir(exist_ok=True)
                torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},weights)
                snapshots.append({"update":update,"checkpoint":weights.relative_to(out).as_posix(),"state_hash":state_hash(model),"metrics":stage+"/metrics.json"})
        final=read(out/f"{arm}_update100/metrics.json")["summary"]
        ratios={profile:final["dataset/thumbnails128x128/"+profile]["MSE"]/baseline["summary"]["dataset/thumbnails128x128/"+profile]["MSE"] for profile in p["learnability_groups"]}
        branches.append({"arm":arm,"snapshots":snapshots,"fixed_final_training_mse_ratios":ratios,"training_fit_observed":all(v<=.8 for v in ratios.values())})
    assert updates==backwards==200 and buffer_hash(model)==buffers
    assert [state_hash(identity),state_hash(perceptual)]==teacher_before
    verify(root)
    artifacts={file.relative_to(out).as_posix():sha(file) for file in sorted(out.rglob("*")) if file.is_file()}
    result={"complete":True,"protocol_sha256":sha(root/"capacity_protocol_v7.json"),"total_optimizer_updates":updates,
        "training_backward_calls":backwards,"preflight_autograd_calls":2,"elapsed_seconds":time.monotonic()-started,
        "branches":branches,"artifacts_sha256":artifacts,"buffers_and_teachers_unchanged":True,
        "training_only":True,"validation_used":False,"native_used":False,"native_reserved_used":False,
        "production_promoted":False,"model_improvement_established":False}
    write(out/"results.json",result)
    print({k:v for k,v in result.items() if k not in ("artifacts_sha256","branches")},flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--parent-root",type=Path)
    parser.add_argument("--root",type=Path,required=True)
    args=parser.parse_args()
    if args.prepare:
        prepare(args.parent_root,args.root)
    else:
        try:
            train(args.root.resolve())
        except Exception as error:
            out=CONTEXT["out"]
            if out is not None and out.exists() and not (out/"failure.json").exists():
                failure={"complete":False,"error_type":type(error).__name__,"error":str(error),
                         "optimizer_updates_recorded":CONTEXT["updates"],"backward_calls_recorded":CONTEXT["backwards"],
                         "resume_permitted":False,"production_promoted":False}
                if CONTEXT["model"] is not None:
                    try:
                        import torch
                        file=out/"partial_state.pth"
                        torch.save({k:v.detach().cpu().clone() for k,v in CONTEXT["model"].state_dict().items()},file)
                        failure["partial_state_sha256"]=sha(file)
                    except Exception as export_error:
                        failure["partial_export_error"]=str(export_error)
                write(out/"failure.json",failure)
            raise
