"""Manual-L4-only CodeFormer gradient diagnostic. Zero optimizer updates.

--verify-transfer hashes files only. --run starts one capped child; --worker is
still VM-gated. No training module, optimizer or inference wrapper is imported.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import urllib.request

VM_NAME = "cctv_completion_codeformer_gradient_v1_vm"
WEIGHTS_SHA = "b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797"
LIMITS = {"seconds": 600, "forward_slots": 24, "gradient_queries": 16, "optimizer_updates": 0,
          "references": 4, "cases": 8, "output_bytes": 128 * 1024**2}
SOURCE_MODULES = {"third_party": "third_party/__init__.py", "third_party.codeformer": "third_party/codeformer/__init__.py",
                  "third_party.codeformer.registry": "third_party/codeformer/registry.py",
                  "third_party.codeformer.vqgan_arch": "third_party/codeformer/vqgan_arch.py",
                  "third_party.codeformer.codeformer_arch": "third_party/codeformer/codeformer_arch.py"}

def need(ok, message):
    if not ok: raise ValueError(message)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def write_new(path, data):
    with Path(path).open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(data,stream,indent=2,allow_nan=False);stream.write("\n")

def member(root, relative):
    need(isinstance(relative,str) and relative and "\\" not in relative,"Invalid packet path")
    path=Path(relative)
    need(not path.is_absolute() and ".." not in path.parts,"Packet traversal refused")
    target=root/path
    for part in [target,*target.parents]:
        need(not part.is_symlink(),"Packet symlink refused")
        if part==root: break
    need(target.resolve().is_relative_to(root),"Packet escaped root")
    return target

def verify(root, expected):
    root=root.resolve(strict=True)
    path=root/"protocol.json"
    need(sha(path)==expected,"Protocol SHA mismatch")
    p=read(path)
    need(p["format"]=="completion-codeformer-gradient-diagnostic-v1" and p["budgets"]==LIMITS,"Wrong diagnostic contract")
    need(p["optimizer_updates"]==0 and p["reserved_FINAL_used"] is False,"Diagnostic role changed")
    need(len(p["references"])==4 and len(p["cases"])==8,"Finite case count changed")
    ids=[r["reference_id"] for r in p["references"]]
    need(len(set(ids))==4 and ids==sorted(ids),"Reference selection differs")
    need(all(r["original_role"]==r["legacy_completion_split"]=="train" for r in p["references"]),"Non-TRAIN source refused")
    need({(r["reference_id"],r["mask_kind"]) for r in p["cases"]}=={(i,k) for i in ids for k in ("lower_face","eye_band")},"TRAIN mask cases differ")
    for row in p["files"]:
        file=member(root,row["path"])
        need(file.is_file() and file.stat().st_size==row["bytes"] and sha(file)==row["sha256"],"Packet file drift: "+row["path"])
    file_names={r["path"] for r in p["files"]}
    required={"scripts/cctv_completion_codeformer_gradient_v1_vm.py","weights/codeformer_inpainting.pth",*SOURCE_MODULES.values()}
    need(required<=file_names,"Missing exact executable/source/checkpoint pin")
    for row in p["references"]: need(row["source1024"] in file_names,"Unpinned clean source")
    for row in p["cases"]: need(row["mask256"] in file_names,"Unpinned reviewed mask")
    need(sha(root/"weights/codeformer_inpainting.pth")==WEIGHTS_SHA,"Wrong original inpainting weights")
    return p

def require_vm(root):
    need(os.name=="posix" and root.resolve()==(Path.home()/"forensic-dgp"/VM_NAME).resolve(),"Run only at the named ~/forensic-dgp L4 VM root")
    metadata="http://metadata.google.internal/computeMetadata/v1/instance/machine-type"
    request=urllib.request.Request(metadata,headers={"Metadata-Flavor":"Google"})
    with urllib.request.urlopen(request,timeout=4) as response: machine=response.read(4096).decode().strip()
    need(machine.rsplit("/",1)[-1]=="g2-standard-4","Expected Google Cloud g2-standard-4")
    names=subprocess.check_output(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],text=True,timeout=10).strip().splitlines()
    need(len(names)==1 and "L4" in names[0],"Expected one NVIDIA L4")
    active=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True,timeout=10).strip()
    need(not active,"Competing GPU workload; do not run diagnostic")
    return {"machine_type":machine,"gpu":names[0]}

def run_worker(root, p, expected):
    hardware=require_vm(root)
    out=root/"outputs"
    need(not out.exists(),"Diagnostic outputs already exist; preserve prior evidence")
    out.mkdir()
    started=time.monotonic(); counters={"forward_slots":0,"gradient_queries":0,"optimizer_updates":0}
    forward_kinds={"clean_encoder":0,"clean_quantizer":0,"masked_code_only":0,"masked_RGB":0}
    def guard():
        need(time.monotonic()-started<LIMITS["seconds"],"600-second diagnostic cap reached")
        need(counters["forward_slots"]<=24 and counters["gradient_queries"]<=16,"Neural/gradient bound exceeded")
        need(sum(f.stat().st_size for f in out.rglob("*") if f.is_file())<=LIMITS["output_bytes"],"Output byte cap reached")
    # This is the first neural-library import and occurs after all transfer and VM checks.
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    need(torch.cuda.is_available(),"CUDA unavailable")
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(240)
    torch.cuda.manual_seed_all(240);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(root))
    for module, relative in SOURCE_MODULES.items():
        imported=importlib.import_module(module)
        need(Path(imported.__file__).resolve()==(root/relative).resolve(),"Unexpected import origin: "+module)
    CodeFormer=importlib.import_module("third_party.codeformer.codeformer_arch").CodeFormer
    checkpoint=root/"weights/codeformer_inpainting.pth"
    loaded=torch.load(checkpoint,map_location="cpu",weights_only=True)
    need(isinstance(loaded,dict) and isinstance(loaded.get("params_ema"),dict),"Expected original params_ema state")
    net=CodeFormer(dim_embd=512,codebook_size=512,n_head=8,n_layers=9,connect_list=["32","64","128"])
    net.load_state_dict(loaded["params_ema"],strict=True);del loaded
    net.to("cuda",dtype=torch.float32).eval().requires_grad_(False)
    def state_hash():
        digest=hashlib.sha256()
        for name,value in sorted(net.state_dict().items()):
            cpu=value.detach().cpu().contiguous()
            need(torch.isfinite(cpu).all().item(),"Non-finite original state")
            digest.update(name.encode());digest.update(str(tuple(cpu.shape)).encode());digest.update(str(cpu.dtype).encode());digest.update(cpu.numpy().tobytes())
        return digest.hexdigest()
    original_state=state_hash();guard()
    def array_sha(value): return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    def png(path, tensor):
        data=(tensor.detach().clamp(0,1)[0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
        Image.fromarray(data,"RGB").save(path)
    def tensor_image(path):
        with Image.open(path) as image:
            need(image.format=="PNG" and image.size==(1024,1024),"Exact HQ1024 PNG required")
            a=np.array(image.convert("RGB").resize((512,512),Image.Resampling.LANCZOS),copy=True)
        return torch.from_numpy(a).permute(2,0,1).unsqueeze(0).to("cuda",torch.float32)/255
    def mask_image(path):
        with Image.open(path) as image:
            need(image.format=="PNG" and image.size==(256,256) and image.mode=="L","Exact L256 mask required")
            a=np.array(image,copy=True);need(np.isin(a,[0,255]).all(),"Non-binary mask")
            a=np.array(image.resize((512,512),Image.Resampling.NEAREST),copy=True)
        return torch.from_numpy(a).to("cuda",torch.float32)[None,None]/255
    def set_groups(route):
        net.requires_grad_(False)
        if route=="code":
            groups={"encoder":list(net.encoder.named_parameters()),
                    "position_emb":[("position_emb",net.position_emb)],
                    "feat_emb":list(net.feat_emb.named_parameters()),
                    "transformer":list(net.ft_layers.named_parameters()),
                    "idx_pred":list(net.idx_pred_layer.named_parameters())}
        else:
            groups={"fusion"+size:list(net.fuse_convs_dict[size].named_parameters()) for size in ("32","64","128")}
        for group,rows in groups.items():
            need(bool(rows),"Empty intended group: "+group)
            for _,parameter in rows:parameter.requires_grad_(True)
        return groups
    def query(loss, groups):
        guard();need(loss.ndim==0 and torch.isfinite(loss).item(),"Non-finite scalar loss")
        ordered=[(group,name,parameter) for group,rows in groups.items() for name,parameter in rows]
        counters["gradient_queries"]+=1;guard()
        gradients=torch.autograd.grad(loss,[p for _,_,p in ordered],allow_unused=True)
        summary={group:{"squared_norm":0.,"tensors":0,"missing":[],"zero_tensors":[],"gradient_tensors":[]} for group in groups}
        for (group,name,parameter),gradient in zip(ordered,gradients):
            row=summary[group];row["tensors"]+=1
            if gradient is None:row["missing"].append(name);continue
            need(torch.isfinite(gradient).all().item(),"Non-finite gradient: "+group+"."+name)
            norm=float(torch.linalg.vector_norm(gradient.detach().double()).item())
            row["squared_norm"]+=norm*norm
            if norm==0:row["zero_tensors"].append(name)
            row["gradient_tensors"].append({"name":name,"shape":list(parameter.shape),"L2":norm,"sha256":array_sha(gradient)})
        for group,row in summary.items():
            row["L2"]=math.sqrt(row.pop("squared_norm"));row["finite"]=True
            row["nonzero"]=row["L2"]>0;row["all_tensors_connected"]=not row["missing"]
        return summary
    clean_cache={};case_results=[]
    try:
        for reference in p["references"]:
            guard();clean=tensor_image(root/reference["source1024"]);normalized=clean*2-1
            with torch.no_grad():
                counters["forward_slots"]+=1;forward_kinds["clean_encoder"]+=1;latent=net.encoder(normalized)
                counters["forward_slots"]+=1;forward_kinds["clean_quantizer"]+=1;quantized,_,stats=net.quantize(latent)
                codes=stats["min_encoding_indices"].reshape(1,-1).long()
            need(latent.shape==(1,256,16,16) and codes.shape==(1,256),"Clean quantizer API geometry mismatch")
            need(codes.min().item()>=0 and codes.max().item()<512,"Clean code index out of range")
            clean_cache[reference["reference_id"]]=(clean,latent.detach(),codes.detach())
            write_new(out/(reference["reference_id"]+"_teacher.json"),{"reference_id":reference["reference_id"],"teacher":"same frozen inpainting encoder and nearest quantizer; not independent clean prior","latent_sha256":array_sha(latent),"codes_sha256":array_sha(codes),"codes":codes.cpu().tolist()[0]})
            png(out/(reference["reference_id"]+"_clean512.png"),clean);guard()
        for case in p["cases"]:
            guard();clean,teacher_latent,codes=clean_cache[case["reference_id"]]
            mask=mask_image(root/case["mask256"])
            masked=clean*(1-mask)+mask;normalized=masked*2-1
            code_groups=set_groups("code")
            counters["forward_slots"]+=1;forward_kinds["masked_code_only"]+=1;logits,lq_feat=net(normalized,code_only=True)
            ce=F.cross_entropy(logits.reshape(-1,512),codes.reshape(-1))
            feature=F.mse_loss(lq_feat,teacher_latent)
            objective=ce+feature
            code_gradient=query(objective,code_groups)
            code_values={"CE":float(ce.detach().item()),"latent_MSE":float(feature.detach().item()),"objective_CE_plus_latent":float(objective.detach().item())}
            logits_np=logits.detach().cpu().numpy();lq_np=lq_feat.detach().cpu().numpy()
            del logits,lq_feat,ce,feature,objective,code_groups
            fusion_groups=set_groups("fusion")
            counters["forward_slots"]+=1;forward_kinds["masked_RGB"]+=1;raw,_,_=net(normalized,w=1,detach_16=True,adain=False)
            generated=(raw+1)/2
            # Compute losses on un-clamped RGB; clamp is only for retained display.
            eroded_mask=1-F.max_pool2d(1-mask,kernel_size=7,stride=1,padding=3)
            inner=(mask-eroded_mask).clamp(0,1)
            need(mask.sum().item()>0 and inner.sum().item()>0,"Empty loss support")
            rgb=(abs(generated-clean)*mask).sum()/(mask.sum()*3)
            boundary=(abs(generated-clean)*inner).sum()/(inner.sum()*3)
            fusion_objective=rgb+boundary
            fusion_gradient=query(fusion_objective,fusion_groups)
            row={"id":case["id"],"reference_id":case["reference_id"],"mask_kind":case["mask_kind"],"code_losses":code_values,"fusion_losses":{"mask_RGB_L1":float(rgb.detach().item()),"inner_boundary_RGB_L1":float(boundary.detach().item()),"objective_RGB_plus_boundary":float(fusion_objective.detach().item())},"code_gradients":code_gradient,"fusion_gradients":fusion_gradient,"seconds":time.monotonic()-started,"counts":dict(counters)}
            row["intended_groups_finite_nonzero_connected"]=all(g["finite"] and g["nonzero"] and g["all_tensors_connected"] for route in (code_gradient,fusion_gradient) for g in route.values())
            write_new(out/(case["id"]+".json"),row);case_results.append(row)
            png(out/(case["id"]+"_white512.png"),masked)
            composed=clean*(1-mask)+generated.clamp(0,1)*mask
            png(out/(case["id"]+"_estimate512.png"),composed)
            np.savez_compressed(out/(case["id"]+"_raw.npz"),logits=logits_np,lq_latent=lq_np,raw_RGB=generated.detach().cpu().numpy(),mask=mask.detach().cpu().numpy())
            del raw,generated,composed,rgb,boundary,fusion_objective,fusion_groups
            print(json.dumps({"case":case["id"],"groups_pass":row["intended_groups_finite_nonzero_connected"],"counts":counters,"seconds":time.monotonic()-started}),flush=True);guard()
        net.requires_grad_(False)
        final_state=state_hash();need(final_state==original_state,"Model state changed without optimizer")
        need(sha(checkpoint)==WEIGHTS_SHA,"Original checkpoint changed")
        need(counters=={"forward_slots":24,"gradient_queries":16,"optimizer_updates":0},"Finite execution count differs")
        passed=all(row["intended_groups_finite_nonzero_connected"] for row in case_results)
        write_new(out/"results.json",{"format":"completion-codeformer-gradient-diagnostic-results-v1","complete":True,"gradient_path_pass":passed,"protocol_sha256":expected,"hardware":hardware,"torch":torch.__version__,"precision":"float32_no_autocast_no_TF32","teacher_limitation":"same frozen inpainting encoder, not independent clean prior","state_before":original_state,"state_after":final_state,"checkpoint_sha256_after":WEIGHTS_SHA,"counts":counters,"forward_kinds":forward_kinds,"seconds":time.monotonic()-started,"cases":case_results,"model_qualified":False,"training_success":False,"training_updates":0,"reserved_FINAL_used":False})
        guard();return 0 if passed else 2
    except BaseException as exc:
        failure={"complete":False,"error":repr(exc),"traceback":traceback.format_exc(),"counts":counters,"seconds":time.monotonic()-started,"optimizer_updates":0,"model_qualified":False}
        if not (out/"failure.json").exists():write_new(out/"failure.json",failure)
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--protocol-sha",required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-transfer",action="store_true")
    mode.add_argument("--run",action="store_true")
    mode.add_argument("--worker",action="store_true")
    a=parser.parse_args();root=a.root.resolve(strict=True);p=verify(root,a.protocol_sha)
    if a.verify_transfer:
        print(json.dumps({"complete":True,"files":len(p["files"]),"cases":8,"optimizer_updates":0,"neural_calls":0,"gradient_queries":0}));return 0
    if a.worker:return run_worker(root,p,a.protocol_sha)
    hardware=require_vm(root)
    need(not (root/"outputs").exists() and not (root/"supervisor.json").exists(),"Retain previous diagnostic namespace; no rerun")
    command=[sys.executable,"-B","-u",str(root/"scripts/cctv_completion_codeformer_gradient_v1_vm.py"),"--root",str(root),"--protocol-sha",a.protocol_sha,"--worker"]
    started=time.monotonic();timeout=False;child=None;code=1;interruption=None
    previous_term=signal.getsignal(signal.SIGTERM)
    def terminate_request(signum,frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate_request)
    try:
        with (root/"diagnostic.log").open("x",encoding="utf-8") as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=child.wait(timeout=600)
            except subprocess.TimeoutExpired:
                timeout=True;code=124
    except BaseException as exc:
        interruption={"type":type(exc).__name__,"message":str(exc)}
        code=130 if isinstance(exc,KeyboardInterrupt) else int(exc.code) if isinstance(exc,SystemExit) and isinstance(exc.code,int) else 1
    finally:
        try:
            if child is not None and child.poll() is None:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait(timeout=15)
        finally:
            signal.signal(signal.SIGTERM,previous_term)
            write_new(root/"supervisor.json",{"complete":True,"child_pid":None if child is None else child.pid,"child_exit":code,"actual_child_returncode":None if child is None else child.returncode,"timeout":timeout,"interruption":interruption,"external_cap_seconds":600,"seconds":time.monotonic()-started,"hardware":hardware,"optimizer_updates":0,"training_success_not_implied":True,"owned_group_cleanup_finally":True})
    print(json.dumps({"child_exit":code,"timeout":timeout,"interruption":interruption,"log":"diagnostic.log","training_success_not_implied":True}));return code

if __name__=="__main__":raise SystemExit(main())
