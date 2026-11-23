"""Independent saved-array/receipt audit. No model forward or gradient execution."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
import zipfile
import numpy as np
from PIL import Image

BRANCH = Path(r"C:\xampp\htdocs\YEAR 4\Testing_FACE_MASK_COMPLETION")
DATA = Path(r"C:\xampp\htdocs\YEAR 4\Testing")
RETURN = BRANCH / "outputs/completion_codeformer_gradient_v1_return"
ROOT = RETURN / "cctv_completion_codeformer_gradient_v1_results"
PACKET = BRANCH / "outputs/completion_codeformer_gradient_v1_packet_r1/cctv_completion_codeformer_gradient_v1_vm"
REPAIR = BRANCH / "outputs/completion_python310_repair_v1/payload"
EXPECTED = "e79183a9f4e3ea50e9ac1d6705d9667f046364132e1f1be0628befd024835d9a"
WEIGHTS_SHA = "b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797"
COUNTS = {"forward_slots": 24, "gradient_queries": 16, "optimizer_updates": 0}
GROUP_COUNTS = {"encoder":164, "position_emb":1, "feat_emb":2, "transformer":108,
                "idx_pred":3, "fusion32":18, "fusion64":18, "fusion128":18}
PREFIX = {"encoder":"encoder.", "position_emb":"", "feat_emb":"feat_emb.",
          "transformer":"ft_layers.", "idx_pred":"idx_pred_layer.",
          "fusion32":"fuse_convs_dict.32.", "fusion64":"fuse_convs_dict.64.",
          "fusion128":"fuse_convs_dict.128."}
HEX = re.compile(r"^[0-9a-f]{64}$")
U32 = float(np.finfo(np.float32).eps / 2)

def need(ok, message):
    if not ok: raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024), b""): h.update(block)
    return h.hexdigest()

def pin(path):
    path=Path(path)
    need(path.is_file() and not path.is_symlink(), "Missing/linked file: "+str(path))
    return {"path":str(path.resolve()),"bytes":path.stat().st_size,"sha256":sha(path)}

def read(path):
    def reject(value): raise ValueError("Non-finite JSON: "+value)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),parse_constant=reject)

def validate_pin(path, expected):
    actual=pin(path)
    need(actual["bytes"]==expected["bytes"] and actual["sha256"]==expected["sha256"],"File binding mismatch: "+str(path))
    return actual

def member(root, name):
    need(isinstance(name,str) and name and "\\" not in name,"Invalid relative path")
    parts=Path(name).parts
    need(not Path(name).is_absolute() and ".." not in parts,"Traversal")
    target=root/name
    need(target.resolve().is_relative_to(root.resolve()),"Escape")
    return target

def rgb(path, size=(512,512)):
    with Image.open(path) as image:
        need(image.format=="PNG" and image.mode=="RGB" and image.size==size,"Invalid RGB PNG: "+str(path))
        return np.array(image,copy=True)

def gamma(n, u=U32):
    need(n*u<1,"Arithmetic bound undefined")
    return (n*u)/(1-n*u)

def float32_objective(a,b,reported,label):
    expected=float(np.float32(np.float32(a)+np.float32(b)))
    need(expected==reported,"FP32 objective addition differs: "+label)
    return expected

def local_state():
    # CPU weights-only deserialization, never imports model architecture or constructs a network.
    import torch
    path=DATA/"checkpoints/codeformer_inpainting.pth"
    checkpoint=pin(path)
    need(checkpoint["sha256"]==WEIGHTS_SHA,"Original checkpoint drift")
    obj=torch.load(path,map_location="cpu",weights_only=True)
    need(isinstance(obj,dict) and isinstance(obj.get("params_ema"),dict),"params_ema missing")
    state=obj["params_ema"]; digest=hashlib.sha256(); shapes={}; state_groups={}
    state_prefixes={**PREFIX,"quantizer":"quantize.","generator":"generator."}
    for name,value in sorted(state.items()):
        need(isinstance(name,str) and isinstance(value,torch.Tensor) and value.device.type=="cpu","Unexpected checkpoint member")
        a=value.detach().contiguous().numpy()
        need(np.isfinite(a).all(),"Non-finite original state")
        shapes[name]=list(a.shape)
        matched=[group for group,prefix in state_prefixes.items() if (name=="position_emb" if group=="position_emb" else prefix and name.startswith(prefix))]
        group=matched[0] if len(matched)==1 else "other"
        stats=state_groups.setdefault(group,{"tensors":0,"numel":0,"float32_bytes":0,"dtype_counts":{}})
        stats["tensors"]+=1;stats["numel"]+=a.size;stats["float32_bytes"]+=a.nbytes
        stats["dtype_counts"][str(a.dtype)]=stats["dtype_counts"].get(str(a.dtype),0)+1
        digest.update(name.encode());digest.update(str(tuple(a.shape)).encode())
        digest.update(str(value.dtype).encode());digest.update(a.tobytes())
    result={"checkpoint":checkpoint,"params_ema_tensors":len(state),"state_sha256":digest.hexdigest(),
            "loader":"torch.load(map_location=cpu, weights_only=True); no architecture import or model object",
            "torch":torch.__version__,"groups":state_groups,"model_instantiations":0,"forward_calls":0,"gradient_calls":0}
    del state,obj
    return result,shapes

def main():
    started=time.monotonic()
    protocol_pin=pin(ROOT/"protocol.json")
    need(protocol_pin["sha256"]==EXPECTED,"Wrong returned protocol")
    need(sha(REPAIR/"protocol.json")==EXPECTED,"Local compatible protocol differs")
    p=read(ROOT/"protocol.json")
    need(len(p["files"])==27 and len(p["references"])==4 and len(p["cases"])==8,"Packet cardinality changed")
    need(p["optimizer_updates"]==0 and p["reserved_FINAL_used"] is False,"Protocol role changed")
    source_pins=[]
    for row in p["files"]:
        rel=row["path"]; path=REPAIR/rel if (REPAIR/rel).is_file() else PACKET/rel
        source_pins.append(validate_pin(path,row))
    file_map={row["path"]:row for row in p["files"]}
    manifest_pin=pin(ROOT/"export_manifest.json"); manifest=read(ROOT/"export_manifest.json")
    need(manifest["protocol_sha256"]==EXPECTED and len(manifest["files"])==44,"Manifest contract differs")
    returned_pins=[validate_pin(member(ROOT,row["path"]),row) for row in manifest["files"]]
    need({Path(q["path"]).relative_to(ROOT).as_posix() for q in returned_pins}==
         {q.relative_to(ROOT).as_posix() for q in ROOT.rglob("*") if q.is_file()}-{"export_manifest.json"},"Returned namespace differs")
    transport_pin=pin(RETURN/"transport_audit.json")
    need(read(transport_pin["path"])["complete"] is True,"Transport audit incomplete")
    results_pin=pin(ROOT/"outputs/results.json"); result=read(results_pin["path"])
    supervisor=read(ROOT/"supervisor.json")
    need(result["complete"] is True and result["protocol_sha256"]==EXPECTED,"Incomplete result")
    need(result["counts"]==COUNTS and result["forward_kinds"]=={"clean_encoder":4,"clean_quantizer":4,"masked_code_only":8,"masked_RGB":8},"Finite counters differ")
    need(result["training_updates"]==0 and result["model_qualified"] is False and result["training_success"] is False and result["reserved_FINAL_used"] is False,"Qualification fields changed")
    need(supervisor["complete"] and supervisor["child_exit"]==supervisor["actual_child_returncode"]==0 and not supervisor["timeout"] and supervisor["interruption"] is None,"Supervisor did not complete cleanly")
    need(0<result["seconds"]<supervisor["seconds"]<600 and supervisor["external_cap_seconds"]==600,"Runtime cap differs")
    need(result["hardware"]==supervisor["hardware"] and result["hardware"]["gpu"]=="NVIDIA L4" and result["hardware"]["machine_type"].endswith("/g2-standard-4"),"Hardware receipt differs")
    need(result["precision"]=="float32_no_autocast_no_TF32","Precision differs")
    state,shapes=local_state()
    need(state["state_sha256"]==result["state_before"]==result["state_after"],"Reported state differs from original params_ema")
    need(result["checkpoint_sha256_after"]==WEIGHTS_SHA,"VM checkpoint receipt differs")
    readiness=read(PACKET/"evidence/training_data_readiness.json")
    train=readiness["genuine_HQ_candidate_sources"]["training_sources"]
    selected=sorted(train,key=lambda row:row["reference_id"])[:4]
    need([r["reference_id"] for r in p["references"]]==[r["reference_id"] for r in selected],"Not first four pinned TRAIN")
    clean_by_id={}; teachers={}; references=[]
    for ref in p["references"]:
        rid=ref["reference_id"]
        need(ref["original_role"]==ref["legacy_completion_split"]=="train","Non-TRAIN reference")
        path=Path(ref["original_source_pin"]["path"])
        original=validate_pin(path,ref["original_source_pin"])
        need(original["sha256"]==file_map[ref["source1024"]]["sha256"],"Original/payload source differs")
        with Image.open(path) as image:
            need(image.mode=="RGB" and image.format=="PNG" and image.size==(1024,1024),"HQ source changed")
            clean=np.array(image.resize((512,512),Image.Resampling.LANCZOS),copy=True)
        need(np.array_equal(clean,rgb(ROOT/"outputs"/(rid+"_clean512.png"))),"LANCZOS clean PNG differs")
        clean_by_id[rid]=clean
        teacher=read(ROOT/"outputs"/(rid+"_teacher.json"))
        need(teacher["reference_id"]==rid and len(teacher["codes"])==256 and HEX.fullmatch(teacher["latent_sha256"]),"Teacher schema differs")
        need(all(type(x) is int and 0<=x<512 for x in teacher["codes"]),"Invalid teacher indices")
        codes=np.asarray(teacher["codes"],dtype=np.int64).reshape(1,256)
        need(hashlib.sha256(codes.tobytes()).hexdigest()==teacher["codes_sha256"],"Teacher code hash differs")
        teachers[rid]=codes
        references.append({"id":rid,"original":original,"clean512":pin(ROOT/"outputs"/(rid+"_clean512.png")),
                           "original_role":"train","legacy_completion_split":"train","LANCZOS512_exact":True,
                           "teacher_codes_int64_hash_exact":True,"teacher_latent_sha256":teacher["latent_sha256"],
                           "teacher_latent_independent_replay":False})
    need([r["id"] for r in result["cases"]]==[r["id"] for r in p["cases"]],"Result case order differs")
    logs=[]
    for line in (ROOT/"diagnostic.log").read_text().splitlines():
        if line.startswith("{"):
            value=json.loads(line)
            if "case" in value: logs.append(value)
    need(len(logs)==8,"Log event count differs")
    case_audits=[]; total_tensors=0; total_outside=0; total_raw=0; groups_summary={}
    npz_shapes={"logits":(1,256,512),"lq_latent":(1,256,16,16),"raw_RGB":(1,3,512,512),"mask":(1,1,512,512)}
    for index,(case,row,event) in enumerate(zip(p["cases"],result["cases"],logs)):
        cid=case["id"]; rid=case["reference_id"]; out=ROOT/"outputs"
        need(row==read(out/(cid+".json")),"Duplicated case receipt differs")
        expected_counts={"forward_slots":10+index*2,"gradient_queries":2+index*2,"optimizer_updates":0}
        need(row["counts"]==event["counts"]==expected_counts and event["case"]==cid and event["groups_pass"] is True,"Event progression differs")
        need(row["reference_id"]==rid and row["mask_kind"]==case["mask_kind"] and row["intended_groups_finite_nonzero_connected"] is True,"Case metadata differs")
        npz_path=out/(cid+"_raw.npz")
        with zipfile.ZipFile(npz_path) as z:
            need(set(z.namelist())=={k+".npy" for k in npz_shapes} and len(z.infolist())==4,"NPZ namespace differs")
            need(all(0<q.file_size<8*1024**2 for q in z.infolist()),"NPZ size differs")
        with np.load(npz_path,allow_pickle=False) as z:
            arrays={k:np.array(z[k],copy=True) for k in npz_shapes}
        for name,a in arrays.items():
            need(a.dtype==np.dtype("float32") and a.shape==npz_shapes[name] and np.isfinite(a).all(),"Invalid saved array: "+cid+" "+name)
            total_raw+=a.nbytes
        mask_path=PACKET/case["mask256"]
        validate_pin(mask_path,file_map[case["mask256"]])
        with Image.open(mask_path) as image:
            need(image.mode=="L" and image.size==(256,256) and image.format=="PNG","Mask schema differs")
            mask256=np.array(image,copy=True);need(np.isin(mask256,[0,255]).all(),"Non-binary mask")
            mask=np.asarray(image.resize((512,512),Image.Resampling.NEAREST))==255
        need(np.array_equal(arrays["mask"][0,0],mask.astype(np.float32)),"Nearest mask differs")
        clean_u8=clean_by_id[rid]; clean=np.transpose(clean_u8.astype(np.float32)/np.float32(255),(2,0,1))[None]
        expected_white=np.where(mask[:,:,None],np.uint8(255),clean_u8)
        need(np.array_equal(expected_white,rgb(out/(cid+"_white512.png"))),"White-hole PNG differs")
        m=mask[None,None].astype(np.float32); generated=arrays["raw_RGB"]
        composed=clean*(np.float32(1)-m)+np.clip(generated,0,1)*m
        expected_png=np.rint(np.transpose(composed[0],(1,2,0))*np.float32(255)).astype(np.uint8)
        actual_png=rgb(out/(cid+"_estimate512.png"))
        need(np.array_equal(expected_png,actual_png),"Raw-to-PNG differs")
        outside=int(np.count_nonzero(~mask))*3
        need(np.array_equal(actual_png[~mask],clean_u8[~mask]),"Outside PNG pixel changed")
        total_outside+=outside
        # max_pool2d uses -inf padding on inverse mask; equivalent erosion pads mask with one.
        pad=np.pad(mask,3,mode="constant",constant_values=True);eroded=np.ones_like(mask)
        for dy in range(7):
            for dx in range(7): eroded &= pad[dy:dy+512,dx:dx+512]
        ring=mask & ~eroded
        need(mask.any() and ring.any(),"Empty loss support")
        abs_error=np.abs(generated.astype(np.float64)-clean.astype(np.float64))
        losses={}; differences={}
        for name,support in (("mask_RGB_L1",mask),("inner_boundary_RGB_L1",ring)):
            n=int(np.count_nonzero(support))*3
            value=float(np.sum(abs_error[0,:,support],dtype=np.float64)/n)
            reported=row["fusion_losses"][name]
            need(math.isfinite(reported) and reported>=0,"Invalid reported fusion loss")
            # A conservative any-order positive FP32 reduction bound: at most n-1 rounded additions.
            # Plus a small elementwise subtraction/division allowance. This is trace arithmetic,
            # never a visual-quality threshold. Record actual discrepancy, not just pass/fail.
            budget=gamma(n+8)*max(value,np.finfo(np.float32).tiny)+8*U32*(1+float(np.max(np.abs(generated))))
            delta=abs(value-reported);need(delta<=budget,"Fusion trace arithmetic exceeds FP32 roundoff envelope")
            losses[name]={"float64_from_saved_arrays":value,"reported_CUDA_float32":reported,"absolute_difference":delta,
                          "absolute_roundoff_envelope":float(budget),"support_RGB_elements":n}
            differences[name]=delta
        float32_objective(row["fusion_losses"]["mask_RGB_L1"],row["fusion_losses"]["inner_boundary_RGB_L1"],
                          row["fusion_losses"]["objective_RGB_plus_boundary"],"fusion")
        logits=arrays["logits"][0].astype(np.float64);codes=teachers[rid][0]
        max_logit=np.max(logits,axis=1)
        ce_tokens=np.log(np.sum(np.exp(logits-max_logit[:,None]),axis=1))+max_logit-logits[np.arange(256),codes]
        ce=float(np.mean(ce_tokens,dtype=np.float64));reported_ce=row["code_losses"]["CE"]
        ce_budget=4*gamma(512+256+8)*max(1.,float(np.max(np.abs(logits))),abs(ce))
        need(math.isfinite(reported_ce) and abs(ce-reported_ce)<=ce_budget,"CE trace exceeds FP32 arithmetic envelope")
        mse=row["code_losses"]["latent_MSE"]
        need(math.isfinite(mse) and mse>=0,"Invalid receipt latent MSE")
        float32_objective(reported_ce,mse,row["code_losses"]["objective_CE_plus_latent"],"code")
        group_rows={};case_tensors=0
        for route,expected_groups in (("code_gradients",list(GROUP_COUNTS)[:5]),("fusion_gradients",list(GROUP_COUNTS)[5:])):
            need(list(row[route])==expected_groups,"Gradient group names/order differ")
            for name,g in row[route].items():
                tensors=g["gradient_tensors"]
                need(g["tensors"]==len(tensors)==GROUP_COUNTS[name] and g["missing"]==g["zero_tensors"]==[],"Gradient count/missing/zero differs")
                need(g["finite"] is g["nonzero"] is g["all_tensors_connected"] is True,"Gradient summary rejected")
                seen=set();squared=0.
                for tensor in tensors:
                    key=PREFIX[name]+tensor["name"]
                    need(key not in seen and key in shapes and tensor["shape"]==shapes[key],"Gradient parameter name/shape differs")
                    seen.add(key);norm=tensor["L2"]
                    need(type(norm) in (float,int) and math.isfinite(norm) and norm>0 and HEX.fullmatch(tensor["sha256"]),"Invalid gradient norm/hash receipt")
                    squared+=norm*norm
                expected_keys={key for key in shapes if key=="position_emb"} if name=="position_emb" else {key for key in shapes if key.startswith(PREFIX[name])}
                need(seen==expected_keys,"Selected group omits original parameter")
                expected_norm=math.sqrt(squared);delta=abs(expected_norm-g["L2"])
                need(delta<=2*math.ulp(expected_norm),"Group norm aggregation differs")
                group_rows[name]={"tensors":len(tensors),"L2":g["L2"],"aggregate_recomputed":expected_norm,"absolute_difference":delta,
                                  "all_norms_finite_positive":True,"all_parameter_names_shapes_match_original":True,
                                  "all_sha256_well_formed":True,"gradient_array_hashes_replayed":False}
                case_tensors+=len(tensors)
                groups_summary.setdefault(name,[]).append(g["L2"])
        need(case_tensors==332,"Case gradient tensor total differs");total_tensors+=case_tensors
        case_audits.append({"id":cid,"raw":pin(npz_path),"estimate512":pin(out/(cid+"_estimate512.png")),
                           "array_shapes_dtypes_finite_exact":True,"mask256_to512_NEAREST_exact":True,"white_input_exact":True,
                           "clamp_composition_rint_PNG_exact":True,"outside_RGB_bytes_exact":outside,
                           "raw_RGB_min":float(generated.min()),"raw_RGB_max":float(generated.max()),
                           "mask_pixels":int(mask.sum()),"inner3px_ring_pixels":int(ring.sum()),
                           "fusion_losses":losses,"CE":{"float64_recomputed":ce,"reported_CUDA_float32":reported_ce,
                           "absolute_difference":abs(ce-reported_ce),"absolute_roundoff_envelope":ce_budget},
                           "latent_MSE":{"reported":mse,"independent_recompute":False,
                           "reason":"Returned lq_latent exists, but clean teacher latent array does not; only its SHA is exported."},
                           "objective_receipt_FP32_additions_exact":True,"gradient_tensor_receipts":case_tensors,"groups":group_rows,
                           "model_calls_performed_by_auditor":0,"gradient_queries_performed_by_auditor":0})
    need(total_tensors==2656 and result["gradient_path_pass"] is True,"Final diagnostic receipt differs")
    # Fresh rehash: do not publish a complete audit if another process modified evidence during reading.
    for bound in [protocol_pin,manifest_pin,results_pin,transport_pin,*source_pins,*returned_pins]:
        validate_pin(bound["path"],bound)
    audit={"format":"codeformer-completion-gradient-return-independent-saved-result-audit-v1","complete":True,"independent":True,
           "audited_UTC":datetime.now(timezone.utc).isoformat(),"auditor":pin(Path(__file__)),
           "protocol":protocol_pin,"results":results_pin,"transport_audit":transport_pin,"export_manifest":manifest_pin,
           "returned_files_rehashed":len(returned_pins)+1,"local_protocol_files_rehashed":len(source_pins),
           "source_pins":source_pins,"references":references,"cases":case_audits,"case_count":8,
           "raw_arrays":32,"raw_array_bytes_checked":total_raw,"PNG_replays":{"clean":4,"white":8,"composed":8},
           "exact_outside_RGB_bytes":total_outside,"gradient_tensor_receipts_checked":total_tensors,
           "gradient_group_L2_ranges":{name:{"minimum":min(v),"maximum":max(v)}for name,v in groups_summary.items()},
           "checkpoint_state_binding":state,"reported_state_before":result["state_before"],"reported_state_after":result["state_after"],
           "state_limitation":"Original local params_ema hashes to both reported state digests. This is not independent observation of all VM memory or proof of no hidden calls.",
           "teacher_limitation":result["teacher_limitation"],
           "gradient_witness_limitation":"Only names/shapes/L2/hash receipts were exported. Their consistency and checkpoint shapes were checked; actual gradient arrays/hashes/connectivity were not independently replayed.",
           "arithmetic_policy":{"float64_replay":True,"FP32_unit_roundoff":U32,
             "fusion":"gamma(n+8)*recomputed_positive_mean + 8*u*(1+maxabs(raw_RGB)); n=support_RGB_elements. Conservative sequential-reduction envelope, no quality gate.",
             "CE":"4*gamma(512+256+8)*max(1,maxabs(logits),CE). Conservative FP32 elementary/reduction trace envelope; not a bitwise reimplementation of CUDA exp/log.",
             "objectives":"Exact np.float32 addition of reported operands.",
             "norm_aggregation":"Recompute Python-double sum(norm*norm) in receipt order then sqrt; allow at most 2 final double ULP."},
           "execution_receipts":{"counts":result["counts"],"forward_kinds":result["forward_kinds"],
             "worker_seconds":result["seconds"],"supervisor":supervisor,"claimed_graph_witness_pass":result["gradient_path_pass"]},
           "scope":{"TRAIN_references":4,"reserved_VAL_or_FINAL_payload_reads":0,"role_and_packet_pins_checked":True,
             "native_CCTV_evidence":False,"automatic_mask_evidence":False,"trained_checkpoint_created":False,
             "quality_or_full_family_qualified":False,"learning_improvement_established":False},
           "auditor_execution":{"CPU_weights_only_loads":1,"model_instantiations":0,"model_forwards":0,"gradient_queries":0,
             "optimizer_updates":0,"VM_calls":0,"archive_code_executed":False},
           "audit_seconds":time.monotonic()-started}
    target=RETURN/"independent_result_audit.json"
    with target.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(audit,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({"complete":True,"audit":pin(target),"cases":8,"gradient_tensor_receipts":total_tensors,
                      "outside_RGB_bytes":total_outside,"state":state["state_sha256"],
                      "max_CE_abs_difference":max(r["CE"]["absolute_difference"] for r in case_audits),
                      "max_fusion_abs_difference":max(v["absolute_difference"]for r in case_audits for v in r["fusion_losses"].values()),
                      "seconds":audit["audit_seconds"]},indent=2))

if __name__=="__main__": main()

