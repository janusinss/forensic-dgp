"""Matched camera/identity diagnostic, guarded to the user's Linux L4 VM only."""
import argparse
import json
import math
import platform
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image,ImageDraw
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models import DGPSynthesizer
from cctv_dgp_pilot import (sha,read,write,state_hash,require_cuda,FixedObservedIdentity,
    PilotPerceptual,PilotDataset,base_loss,composite,freeze_normalization,exported_pixel_metrics,buffer_hash,
    aggregate,qualifies,verify_bundle,identity_crop,image_tensor)

RUN_CONTEXT={}


def require_vm(root):
    require_cuda()
    if sys.platform!="linux" or platform.node().split(".")[0]!="forensic-dgp-thesis":
        raise RuntimeError("Actual training/backward probes must run on forensic-dgp-thesis Linux VM")
    project=Path.home()/"forensic-dgp"
    if not Path(root).resolve().is_relative_to(project.resolve()):
        raise RuntimeError("VM bundle must be under ~/forensic-dgp")
    if "L4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This diagnostic is frozen for the existing NVIDIA L4 VM")


def check_clock(deadline):
    if time.monotonic()>deadline:
        raise TimeoutError("Finite90-minute pilot cap exceeded; preserve partial evidence")


def device_batch(batch):
    return {k:v.to("cuda") if isinstance(v,torch.Tensor) else v for k,v in batch.items()}


def save_weights(path,model):
    if path.exists():
        raise ValueError("Refuse checkpoint overwrite")
    torch.save(model.state_dict(),path)


def preflight(root,out,protocol,model,identity,perceptual,deadline):
    import onnxruntime as ort
    before=state_hash(model)
    teacher_before={"identity":state_hash(identity),"perceptual":state_hash(perceptual)}
    loader=DataLoader(PilotDataset(root,protocol,protocol["training_epochs"]["1"]),batch_size=8,shuffle=False,num_workers=0)
    batch=device_batch(next(iter(loader)))
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    reference=ort.InferenceSession(str(root/protocol["weights"]["arcface"]),sess_options=options,providers=["CPUExecutionProvider"])
    with torch.no_grad():
        crop=identity_crop(batch["target"][:1],batch["mask"][:1],batch["grid"][:1])*2-1
        expected=reference.run(None,{reference.get_inputs()[0].name:crop.cpu().numpy()})[0]
        actual=identity.encoder(crop).cpu().numpy()
    np.testing.assert_allclose(actual,expected,rtol=1e-3,atol=1e-4)
    model.eval().requires_grad_(True)
    generated=composite(model(batch["low"]),batch["low"],batch["mask"])
    generated.retain_grad()
    loss,_=base_loss(generated,batch["target"],batch["mask"],perceptual)
    with torch.no_grad():
        reference_embed=identity.embedding(batch["target"],batch["mask"],batch["grid"])
    identity_term=(1-(identity.embedding(generated,batch["mask"],batch["grid"])*reference_embed).sum(1)).mean()
    # Check the identity term's path separately before the composite backward.
    identity_gradient=torch.autograd.grad(identity_term,generated,retain_graph=True)[0]
    if not torch.isfinite(identity_gradient).all() or identity_gradient.abs().sum()<=0:
        raise ValueError("Identity supervision has no finite restoration-input gradient")
    total=loss+.1*identity_term
    total.backward()
    if not torch.isfinite(total) or not any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()):
        raise ValueError("Nonfinite/empty restoration gradients")
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError("Nonfinite restoration gradients")
    if any(p.requires_grad or p.grad is not None for m in (identity,perceptual) for p in m.parameters()):
        raise ValueError("Frozen teacher received parameter gradients")
    model.zero_grad(set_to_none=True)
    if state_hash(model)!=before or {"identity":state_hash(identity),"perceptual":state_hash(perceptual)}!=teacher_before:
        raise ValueError("Zero-update preflight changed weights/buffers")
    check_clock(deadline)
    report={"passed":True,"batch_size":len(batch["low"]),"optimizer_constructed":False,"optimizer_updates":0,
            "onnx_conversion_max_error":float(np.abs(expected-actual).max()),
            "identity_input_gradient_mean":float(identity_gradient.abs().mean().item()),
            "model_state_unchanged":True,"model_state_hash":before,"teacher_state_before":teacher_before,
            "gpu":torch.cuda.get_device_name(0),"vram_bytes":torch.cuda.get_device_properties(0).total_memory,
            "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(),"torch":torch.__version__,
            "protocol_sha256":sha(root/"protocol.json"),"full_pilot_pending":True}
    write(out/"preflight.json",report)
    print("CUDA preflight passed: batch8, zero updates; full matched pilot pending",flush=True)
    return teacher_before


def preview(out,root,protocol,rows,stage):
    by_id={r["id"]:r for r in rows};refs={r["id"]:r for r in protocol["references"]}
    sheet=Image.new("RGB",(3*164,10*194+24),(238,238,238));draw=ImageDraw.Draw(sheet)
    for j,label in enumerate(("input","DGP observed PNG","reference")):
        draw.text((j*164+2,3),label,fill="black")
    for i,case_id in enumerate(protocol["preview_case_ids"]):
        row=by_id[case_id];ref=refs[row["reference_id"]]
        for j,path in enumerate((root/row["input"],out/row["prediction"],root/ref["target"])):
            y=24+i*194;draw.text((j*164+2,y+2),case_id[:26],fill="black")
            with Image.open(path) as img:
                sheet.paste(img.convert("RGB").resize((160,160),Image.Resampling.BILINEAR),(j*164,y+28))
    file=f"{stage}/preview_10_rows.png";sheet.save(out/file)
    return file


def evaluate(root,out,protocol,model,identity,reference_embeddings,stage,deadline,include_input=False):
    target_dir=out/stage;target_dir.mkdir()
    (target_dir/"images").mkdir();(target_dir/"embeddings").mkdir();(target_dir/"float_preview").mkdir()
    dataset=PilotDataset(root,protocol,protocol["validation_cases"])
    loader=DataLoader(dataset,batch_size=8,shuffle=False,num_workers=0)
    model.eval();rows=[];input_rows=[];artifacts={};start=time.monotonic();calls=0
    for batch in loader:
        check_clock(deadline)
        batch=device_batch(batch)
        with torch.no_grad():
            network=model(batch["low"]);calls+=1
            observed=composite(network,batch["low"],batch["mask"])
            # Metrics follow the exact exported image. This is different from prior raw-float protocols.
            quantized=torch.floor(observed*255).clamp(0,255)/255
            generated_embeddings=identity.embedding(quantized,batch["mask"],batch["grid"])
            input_embeddings=identity.embedding(batch["low"],batch["mask"],batch["grid"]) if include_input else None
        for j,index in enumerate(batch["index"].tolist()):
            case=dataset.cases[index];ref=dataset.references[case["reference_id"]]
            prediction=(quantized[j].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
            target=np.asarray(Image.open(root/ref["target"]).convert("RGB"));mask=np.asarray(Image.open(root/ref["observed"]))>0
            image_file=f"{stage}/images/{case['id']}.png";Image.fromarray(prediction).save(out/image_file);artifacts[image_file]=sha(out/image_file)
            embedding=generated_embeddings[j].cpu().numpy();embed_file=f"{stage}/embeddings/{case['id']}.npy"
            np.save(out/embed_file,embedding,allow_pickle=False);artifacts[embed_file]=sha(out/embed_file)
            reference=reference_embeddings[ref["id"]]
            metrics=exported_pixel_metrics(prediction,target,mask)
            rows.append({**case,**metrics,"prediction":image_file,"embedding":embed_file,
                         "ArcFace_observed_fixed":float(np.clip(embedding@reference,-1,1))})
            if case["id"] in protocol["preview_case_ids"]:
                file=f"{stage}/float_preview/{case['id']}.npz"
                np.savez_compressed(out/file,network_rgb=network[j].permute(1,2,0).cpu().numpy(),
                                    observed_rgb=observed[j].permute(1,2,0).cpu().numpy())
                artifacts[file]=sha(out/file)
            if include_input:
                rgb=np.asarray(Image.open(root/case["input"]).convert("RGB"))
                emb=input_embeddings[j].cpu().numpy();file=f"{stage}/embeddings/{case['id']}_input.npy"
                np.save(out/file,emb,allow_pickle=False);artifacts[file]=sha(out/file)
                input_rows.append({**case,**exported_pixel_metrics(rgb,target,mask),"embedding":file,
                                   "ArcFace_observed_fixed":float(np.clip(emb@reference,-1,1))})
        if len(rows)%80<8:
            print(f"{stage} validation {len(rows)}/{len(dataset)} elapsed={time.monotonic()-start:.0f}s",flush=True)
    preview_file=preview(out,root,protocol,rows,stage);artifacts[preview_file]=sha(out/preview_file)
    report={"complete":True,"rows":rows,"summary":aggregate(rows),"input_rows":input_rows,
            "input_summary":aggregate(input_rows) if include_input else None,
            "artifact_sha256":artifacts,"seconds":time.monotonic()-start,"dgp_batch_forwards":calls,
            "preview":preview_file,"evaluation_basis":"exported RGB PNG with preserved nonobserved padding"}
    write(out/stage/"metrics.json",report)
    return report


def run(root,out,only_preflight=False):
    require_vm(root)  # Before any optimizer, backward, weight loading or output creation.
    start=time.monotonic();deadline=start+5400
    RUN_CONTEXT.clear();RUN_CONTEXT["start"]=start;RUN_CONTEXT["total_updates"]=0
    protocol=verify_bundle(root)
    if out.exists():
        raise ValueError("Run already exists: preserve partial/completed evidence; no automatic resume/repeat")
    out.mkdir(parents=True)
    torch.set_num_threads(4);torch.manual_seed(protocol["seed"])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=DGPSynthesizer().cuda().eval()
    model.load_state_dict(torch.load(root/protocol["weights"]["dgp"],map_location="cpu",weights_only=True),strict=True)
    RUN_CONTEXT["model"]=model
    identity=FixedObservedIdentity(root/protocol["weights"]["arcface"],"cuda")
    perceptual=PilotPerceptual(root/protocol["weights"]["vgg_trunk"],"cuda")
    teacher_before=preflight(root,out,protocol,model,identity,perceptual,deadline)
    if only_preflight:
        return
    import importlib.metadata
    initial_buffers=buffer_hash(model)
    write(out/"execution.json",{"protocol_sha256":sha(root/"protocol.json"),"host":platform.node(),"device":"cuda",
          "torch":torch.__version__,"started_unix_seconds":time.time(),"normalization_running_stats_frozen":True,
          "ema":False,"amp":False,"grid_sample_cuda_backward_may_be_nondeterministic":True,
          "initial_model_state":state_hash(model),"initial_buffers_hash":initial_buffers,
          "teacher_states":teacher_before,"runtime_cap_seconds":5400,
          "python":sys.version,"package_versions":{k:importlib.metadata.version(k) for k in
              ("torch","torchvision","numpy","Pillow","onnx","onnx2torch","onnxruntime","scikit-image")}})
    (out/"reference_embeddings").mkdir();reference_embeddings={};references={r["id"]:r for r in protocol["references"]}
    from cctv_dgp_pilot import grid112
    for offset in range(0,len(protocol["references"]),8):
        check_clock(deadline);refs=protocol["references"][offset:offset+8]
        targets=torch.stack([image_tensor(Image.open(root/r["target"]).convert("RGB")) for r in refs]).cuda()
        masks=torch.stack([torch.from_numpy((np.asarray(Image.open(root/r["observed"]))>0).copy()).float()[None] for r in refs]).cuda()
        grids=torch.stack([torch.from_numpy(grid112(r["matrix112"])) for r in refs]).cuda()
        with torch.no_grad():
            embeds=identity.embedding(targets,masks,grids).cpu().numpy()
        for ref,embed in zip(refs,embeds):
            reference_embeddings[ref["id"]]=embed
            np.save(out/"reference_embeddings"/(ref["id"]+".npy"),embed,allow_pickle=False)
    baseline=evaluate(root,out,protocol,model,identity,reference_embeddings,"baseline",deadline,True)
    baseline_state=state_hash(model);branches=[];total_updates=0
    for arm in protocol["arms"]:
        check_clock(deadline);torch.manual_seed(protocol["seed"])
        model.load_state_dict(torch.load(root/protocol["weights"]["dgp"],map_location="cpu",weights_only=True),strict=True)
        if state_hash(model)!=baseline_state:
            raise ValueError("Matched branch did not start at identical baseline")
        backbone=list(model.fpn.features.parameters());ids={id(p) for p in backbone}
        optimizer=torch.optim.Adam([{"params":backbone,"lr":2e-6},
            {"params":[p for p in model.parameters() if id(p) not in ids],"lr":1e-5}],weight_decay=1e-5)
        RUN_CONTEXT["optimizer"]=optimizer;RUN_CONTEXT["arm"]=arm["id"]
        branch=out/arm["id"];branch.mkdir()
        save_weights(branch/"baseline.pth",model)
        best=baseline["summary"];best_epoch=0;best_file=f"{arm['id']}/baseline.pth";updates=0;epochs=[]
        for epoch in (1,2):
            epoch_start=time.monotonic();cases=protocol["training_epochs"][str(epoch)]
            loader=DataLoader(PilotDataset(root,protocol,cases),batch_size=8,shuffle=False,num_workers=2,pin_memory=True)
            model.train();freeze_normalization(model);records=[]
            for batch in loader:
                check_clock(deadline);batch=device_batch(batch);optimizer.zero_grad(set_to_none=True)
                predicted=composite(model(batch["low"]),batch["low"],batch["mask"])
                loss,components=base_loss(predicted,batch["target"],batch["mask"],perceptual)
                id_loss=predicted.sum()*0
                if arm["lambda_identity"]:
                    target_emb=torch.from_numpy(np.stack([reference_embeddings[r] for r in batch["reference_id"]])).cuda()
                    id_loss=(1-(identity.embedding(predicted,batch["mask"],batch["grid"])*target_emb).sum(1)).mean()
                total=loss+arm["lambda_identity"]*id_loss
                if not torch.isfinite(total):
                    raise FloatingPointError("Nonfinite pilot loss")
                total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();updates+=1;total_updates+=1
                RUN_CONTEXT["total_updates"]=total_updates;RUN_CONTEXT["arm_updates"]=updates;RUN_CONTEXT["epoch"]=epoch
                if updates>256 or total_updates>512:
                    raise ValueError("Finite update cap violated")
                record={"arm":arm["id"],"epoch":epoch,"update":updates,"cases":list(batch["case_id"]),
                        "loss":float(total.item()),"identity_loss":float(id_loss.item()),"components":components,
                        "elapsed_total_seconds":time.monotonic()-start}
                records.append(record)
                with (out/"updates.jsonl").open("a",encoding="utf-8") as stream:
                    stream.write(json.dumps(record,allow_nan=False)+"\n")
                if updates==32:
                    per_update=(time.monotonic()-epoch_start)/32
                    remaining=protocol["expected_total_updates"]-total_updates
                    projected=time.monotonic()+remaining*per_update+5*baseline["seconds"]
                    write(branch/"timing_at32.json",{"seconds_per_update":per_update,"projected_finish_with_validation_reserve_seconds":projected-start,"cap_seconds":5400})
                    if projected>deadline:
                        raise TimeoutError("Measured timing cannot fit the frozen90-minute budget")
                if updates==1 or updates%32==0:
                    print(f"{arm['id']} epoch{epoch}/2 update{updates}/{protocol['updates_per_arm']} loss={total.item():.5f} elapsed={time.monotonic()-start:.0f}s",flush=True)
            model.eval();checkpoint=f"{arm['id']}/epoch_{epoch}.pth";save_weights(out/checkpoint,model)
            if buffer_hash(model)!=initial_buffers:
                raise ValueError("Frozen student normalization buffers changed")
            stage=f"{arm['id']}_epoch{epoch}"
            metrics=evaluate(root,out,protocol,model,identity,reference_embeddings,stage,deadline)
            accepted=qualifies(metrics["summary"],baseline["summary"],best)
            if accepted:
                best,best_epoch,best_file=metrics["summary"],epoch,checkpoint
            entry={"epoch":epoch,"updates":len(records),"cumulative_updates":updates,"metrics":f"{stage}/metrics.json",
                   "checkpoint":checkpoint,"checkpoint_sha256":sha(out/checkpoint),"state_hash":state_hash(model),
                   "buffers_hash":buffer_hash(model),"accepted":accepted}
            epochs.append(entry)
            write(branch/f"epoch_{epoch}_record.json",entry)
            print(f"{arm['id']} epoch{epoch} saved; qualified_against_baseline={accepted}",flush=True)
        import shutil
        shutil.copyfile(out/best_file,branch/"best.pth")
        selection={"selected_epoch":best_epoch,"selected_source":best_file,"best_sha256":sha(branch/"best.pth"),
                   "reason":"Pilot metric guards only; independent native/visual review still pending" if best_epoch else "No trained candidate passed; retain starting baseline"}
        write(branch/"best_selection.json",selection)
        if updates!=protocol["updates_per_arm"]:
            raise ValueError("Actual branch update count differs")
        branches.append({"arm":arm,"epochs":epochs,"optimizer_updates":updates,"selection":selection})
    if total_updates!=protocol["expected_total_updates"]:
        raise ValueError("Matched total updates differ")
    if {"identity":state_hash(identity),"perceptual":state_hash(perceptual)}!=teacher_before:
        raise ValueError("Frozen teacher states changed")
    check_clock(deadline)
    artifacts={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    report={"complete":True,"protocol_sha256":sha(root/"protocol.json"),"branches":branches,
            "total_optimizer_updates":total_updates,"elapsed_seconds":time.monotonic()-start,
            "teacher_states_unchanged":True,"normalization_buffers_unchanged":True,"reference_embedding_files":len(reference_embeddings),
            "artifacts_sha256":artifacts,"production_checkpoint_promoted":False,
            "native_reserved_used":False,"goal_complete":False}
    write(out/"results.json",report)
    print("Matched CCTV DGP diagnostic complete. Export results; no app checkpoint promoted.",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify",action="store_true")
    parser.add_argument("--preflight-only",action="store_true")
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.verify:
        protocol=verify_bundle(args.root);print(json.dumps({"bundle_verified":True,"references":len(protocol["references"]),"optimizer_updates":0}))
    else:
        output=args.output or args.root/("outputs/preflight" if args.preflight_only else "outputs/cctv_dgp_pilot")
        output_existed=output.exists()
        try:
            run(args.root,output,args.preflight_only)
        except Exception as error:
            if not output_existed and output.exists() and not (output/"failure.json").exists():
                partial=None;partial_error=None
                if "model" in RUN_CONTEXT:
                    try:
                        partial={k:v for k,v in RUN_CONTEXT.items() if k not in ("model","optimizer","start")}
                        partial["model"]=RUN_CONTEXT["model"].state_dict()
                        if "optimizer" in RUN_CONTEXT:
                            partial["optimizer"]=RUN_CONTEXT["optimizer"].state_dict()
                        torch.save(partial,output/"partial_state.pth")
                        partial=sha(output/"partial_state.pth")
                    except Exception as export_error:
                        partial_error=type(export_error).__name__+": "+str(export_error)
                write(output/"failure.json",{"complete":False,"error_type":type(error).__name__,"error":str(error),
                      "partial_state_sha256":partial,"partial_export_error":partial_error,
                      "optimizer_updates_recorded":RUN_CONTEXT.get("total_updates",0),"resume_permitted":False})
            raise
