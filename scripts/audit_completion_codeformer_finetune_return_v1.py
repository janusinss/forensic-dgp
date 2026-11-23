"""Independent failed preflight return audit; passive pickle then CPU weights_only.
Never constructs a model or executes a forward, backward, optimizer step or VM call.
"""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import pickletools
import random
import re
import sys
import time
import zipfile
import numpy as np

BASE=Path(r"C:\xampp\htdocs\YEAR 4\Testing_FACE_MASK_COMPLETION")
DATA=Path(r"C:\xampp\htdocs\YEAR 4\Testing")
PARENT=BASE/"outputs/completion_codeformer_finetune_v1_return"
ROOT=PARENT/"cctv_completion_codeformer_finetune_v1_results"
EXPECTED="0dfe963bd29856ea6201d6397aa25d651ca0291661fa3ec26539c9c770e31b8f"
TERMINAL_SHA="08db0663b2b0820ce497e7ef21e58410db89bdcdbabb60346dce78bb3f97d9ea"
BASE_SHA="b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797"
LOGICAL="586a8854ed7b5e3f0e202c258addb7a87de63cd3068d153741b065379ff6668f"
COUNTS={"forward_slots":82,"backward_calls":0,"code_optimizer_steps":0,"fusion_optimizer_steps":0,"empty_bypasses":0}
TYPED={"teacher_encoder":40,"teacher_quantizer":40,"parity":2,"fit_code":0,"fit_RGB":0,"monitor":0,"public_development":0}
SAFE_GLOBALS={"collections OrderedDict","torch._utils _rebuild_tensor_v2","torch FloatStorage","torch ByteStorage","torch LongStorage"}
PREFIX={"encoder":"encoder.","position_emb":"","feat_emb":"feat_emb.","transformer":"ft_layers.","idx_pred":"idx_pred_layer.",
        "fusion32":"fuse_convs_dict.32.","fusion64":"fuse_convs_dict.64.","fusion128":"fuse_convs_dict.128."}

def need(ok,message):
    if not ok:raise ValueError(message)

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb")as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()

def pin(path):
    path=Path(path);need(path.is_file() and not path.is_symlink(),"Missing/linked evidence")
    return {"path":str(path.resolve()),"bytes":path.stat().st_size,"sha256":sha(path)}

def read(path):
    def reject(x):raise ValueError("Non-finiteJSON"+x)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),parse_constant=reject)

def bound(path,record):
    actual=pin(path);need(actual["bytes"]==record["bytes"] and actual["sha256"]==record["sha256"],"Binding changed: "+str(path))
    return actual

def passive(path):
    with zipfile.ZipFile(path)as archive:
        rows=archive.infolist();names=[r.filename for r in rows]
        need(len(names)==len(set(names)) and all(not r.is_dir() and not r.flag_bits&1 for r in rows),"CheckpointZIP namespace")
        need(all(not Path(n).is_absolute() and ".." not in Path(n).parts and "\\" not in n for n in names),"CheckpointZIP traversal")
        bad=archive.testzip();need(bad is None,"CheckpointZIP CRC failed")
        candidates=[n for n in names if n.endswith("/data.pkl")]
        need(len(candidates)==1,"Expected one passive pickle")
        data=archive.read(candidates[0]);need(len(data)<2*1024**2,"Pickle metadata unbounded")
        globals_=[];operations=0
        for op,arg,pos in pickletools.genops(data):
            operations+=1
            need(op.name!="STACK_GLOBAL","Stack global requires separate analysis")
            if op.name=="GLOBAL":
                need(arg in SAFE_GLOBALS,"Unexpected pickle global: "+str(arg));globals_.append(arg)
        need(not any(n.endswith((".py",".pyc",".so",".dll"))for n in names),"Code inside checkpointZIP")
        return {"checkpoint":pin(path),"ZIP_entries":len(rows),"ZIP_uncompressed_bytes":sum(r.file_size for r in rows),
                "CRC_all_members_pass":True,"pickle":candidates[0],"pickle_bytes":len(data),"pickle_opcodes":operations,
                "pickle_globals":sorted(set(globals_)),"globals_in_known_tensor_only_allowlist":True,
                "arbitrary_pickle_load_used":False,"unknown_safe_globals_added":False}

def logical(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        array=value.detach().cpu().contiguous().numpy()
        need(np.isfinite(array).all(),"Nonfinite state tensor")
        digest.update(name.encode());digest.update(str(tuple(array.shape)).encode())
        digest.update(str(value.dtype).encode());digest.update(array.tobytes())
    return digest.hexdigest()

def main():
    started=time.monotonic()
    ppin=pin(ROOT/"protocol.json");need(ppin["sha256"]==EXPECTED,"Wrong protocol")
    protocol=read(ROOT/"protocol.json");manifest_pin=pin(ROOT/"export_manifest.json");manifest=read(manifest_pin["path"])
    need(manifest["protocol_sha256"]==EXPECTED,"Manifest protocol differs")
    returns=[]
    for row in manifest["files"]:
        name=row["path"];need(isinstance(name,str) and not Path(name).is_absolute() and ".." not in Path(name).parts and "\\" not in name,"Manifest traversal")
        returns.append(bound(ROOT/name,row))
    expected_names={"protocol.json","supervisor.json","trainer.log","outputs/failure.json","outputs/terminal_training_state.pth"}
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()}
    need(actual==expected_names|{"export_manifest.json"} and {r["path"] for r in manifest["files"]}==expected_names,"Unexpected evidence namespace")
    failure=read(ROOT/"outputs/failure.json");supervisor=read(ROOT/"supervisor.json")
    need(failure["complete"] is False and "Fresh frozen app pipeline baseline parity failed" in failure["exception"],"Different failure")
    need(failure["counts"]==COUNTS and failure["typed_forward_slots"]==TYPED,"Nonzero training work or changed typedcounts")
    need(failure["last_full_cycle"]==0 and failure["partial_progress"]=={"cycle":0,"route":"initialization","microbatch":0,"phase":"before_first_update"},"Failure stage differs")
    need(failure["model_qualified"] is False and len(failure["snapshots"])==1,"Failure scope differs")
    saved=failure["snapshots"][0];terminal=ROOT/"outputs/terminal_training_state.pth"
    terminal_pin=pin(terminal);need(terminal_pin["sha256"]==TERMINAL_SHA==saved["sha256"] and terminal_pin["bytes"]==saved["bytes"],"Terminal identity differs")
    need(supervisor["complete"] is True and supervisor["worker_exit"]==supervisor["actual_child_exit"]==1 and not supervisor["timeout"] and supervisor["interruption"] is None,"Supervisor failure notclean")
    need(supervisor["terminal_state_present"] is True and supervisor["external_cap_seconds"]==1800 and 0<failure["seconds"]<supervisor["seconds"]<1800,"Runtime/terminal receipt differs")
    original=DATA/"checkpoints/codeformer_inpainting.pth";base_pin=pin(original);need(base_pin["sha256"]==BASE_SHA,"Original changed")
    passive_reports={"terminal":passive(terminal),"original":passive(original)}
    # Only after passive CRC/global review. Never adds custom safe globals or uses weights_only=False.
    import torch
    base=torch.load(original,map_location="cpu",weights_only=True)
    saved=torch.load(terminal,map_location="cpu",weights_only=True)
    need(isinstance(base,dict) and isinstance(base.get("params_ema"),dict) and isinstance(saved,dict),"State dictionaries missing")
    state=base["params_ema"];trainable=saved["trainable_state"]
    need(saved["base_checkpoint_sha256"]==BASE_SHA and saved["protocol_sha256"]==EXPECTED and saved["contract"]==protocol["contract"],"Saved protocol/basecontract differs")
    trainable_names={n for n in state if not n.startswith(("quantize.","generator."))}
    frozen_names=set(state)-trainable_names
    need(set(trainable)==trainable_names and len(trainable)==332 and len(frozen_names)==165 and len(state)==497,"State partition differs")
    byte_count=numel=0
    for name,value in trainable.items():
        need(isinstance(value,torch.Tensor) and value.device.type=="cpu" and value.dtype==torch.float32 and value.shape==state[name].shape,"State tensor schema differs")
        a=value.detach().contiguous().numpy();b=state[name].detach().contiguous().numpy()
        need(np.isfinite(a).all() and np.array_equal(a,b),"Saved trainable updated/differs from base: "+name)
        byte_count+=a.nbytes;numel+=a.size
    need(byte_count==233839360 and numel==58459840,"Trainable size differs")
    base_logical=logical(state);reconstructed=dict(state);reconstructed.update(trainable)
    reconstructed_logical=logical(reconstructed);frozen_logical=logical({n:state[n]for n in frozen_names})
    need(base_logical==reconstructed_logical==LOGICAL==saved["logical_state_sha256"],"Original/reconstructed logical hash differs")
    need(frozen_logical==saved["frozen_state_sha256"],"Frozen hash differs")
    need(saved["cycle"]==saved["completed_paired_cycles"]==0 and saved["next_cycle"]==1 and saved["next_clean_sampler_index"]==0,"Saved cycles/sampler index differ")
    need(saved["counts"]==COUNTS and saved["progress"]==failure["partial_progress"] and saved["sampler_schedule"]==protocol["schedule"],"Saved progress/schedule differs")
    need(saved["partial_accumulated_gradients"]=={} and saved["automatic_resume_permitted"] is False and saved["model_qualified"] is False,"Unexpected gradient/resume flags")
    need(saved["reason"]=="failure_retained_current_partial_state" and saved["resume_progress_requires_new_manual_audit"] is True,"Saved reason changed")
    diagnostic=read(BASE/"outputs/completion_codeformer_gradient_v1_return/cctv_completion_codeformer_gradient_v1_results/outputs/results.json")["cases"][0]
    expected_parameter_names={}
    for route,dest in (("code_gradients","code"),("fusion_gradients","fusion")):
        expected_parameter_names[dest]=[PREFIX[group]+tensor["name"] for group,record in diagnostic[route].items() for tensor in record["gradient_tensors"]]
    optimizer_reviews={}
    for route,count in (("code",278),("fusion",54)):
        names=saved["optimizer_parameter_names"][route];need(names==expected_parameter_names[route] and len(set(names))==count,"Ordered optimizer names differ")
        optimizer=saved[route+"_optimizer"];need(optimizer["state"]=={} and len(optimizer["param_groups"])==1,"Adam moments/updates unexpectedly exist")
        group=optimizer["param_groups"][0]
        need(group["params"]==list(range(count)) and group["lr"]==1e-5 and tuple(group["betas"])==(.9,.99) and group["eps"]==1e-8 and group["weight_decay"]==0,"Adam mapping/hyperparameters differ")
        optimizer_reviews[route]={"parameter_tensors":count,"ordered_names_match_graph_diagnostic_and_original":True,
                                  "parameter_names":names,"optimizer_state_entries":0,"moment_tensors":0,"step_counters":0,
                                  "parameter_ids":group["params"],"hyperparameters":{k:v for k,v in group.items()if k!="params"}}
    need(saved["python_random_state"]==random.Random(240).getstate(),"Python RNG differs from seeded nofit state")
    np_expected=np.random.RandomState(240).get_state();np_saved=saved["numpy_random_state"]
    need(np_saved["algorithm"]==np_expected[0] and np.array_equal(np.asarray(np_saved["keys"],dtype=np.uint32),np_expected[1])
         and np_saved["position"]==np_expected[2] and np_saved["has_gauss"]==np_expected[3] and np_saved["cached_gaussian"]==np_expected[4],"NumPy RNG differs")
    cpu_rng=saved["torch_CPU_RNG"];cuda_rng=saved["CUDA_RNG"]
    need(isinstance(cpu_rng,torch.Tensor) and cpu_rng.dtype==torch.uint8 and cpu_rng.ndim==1 and cpu_rng.numel()>0 and len(cuda_rng)==1,"RNG schema differs")
    rng_pins=[]
    for kind,value in [("CPU",cpu_rng),("CUDA0",cuda_rng[0])]:
        need(isinstance(value,torch.Tensor) and value.dtype==torch.uint8 and value.ndim==1 and value.numel()>0,"RNG tensor invalid")
        array=value.cpu().contiguous().numpy()
        rng_pins.append({"kind":kind,"shape":list(array.shape),"dtype":"uint8","bytes":array.nbytes,"sha256":hashlib.sha256(array.tobytes()).hexdigest(),
                         "generator_state_replay_not_performed":True})
    need(not list(ROOT.rglob("*.png")) and not list(ROOT.rglob("*.npz")) and not (ROOT/"outputs/results.json").exists(),"Unexpected evaluated pixels/results")
    for q in [ppin,manifest_pin,*returns,base_pin]:bound(q["path"],q)
    audit={"format":"completion-codeformer-finetune-v1-independent-failed-return-audit","complete":True,"independent":True,
           "audited_UTC":datetime.now(timezone.utc).isoformat(),"auditor":pin(Path(__file__)),"protocol":ppin,"export_manifest":manifest_pin,
           "returned_file_pins":returns,"terminal_checkpoint":terminal_pin,"original_checkpoint":base_pin,"passive_checkpoint_review":passive_reports,
           "safe_loading":{"torch":torch.__version__,"map_location":"cpu","weights_only":True,"custom_safe_globals_added":False,
                           "arbitrary_pickle_load":False,"network_class_import":False,"model_instantiations":0},
           "failure_status":"Fresh frozen app pipeline parity preflight failed on first case before any fit update or monitor/DEV evaluation.",
           "failure_receipt":failure,"supervisor":supervisor,"reported_counts":COUNTS,"reported_typed_forward_slots":TYPED,
           "training_executed":False,"optimizer_updates":0,"backward_calls":0,"training_quality_result_available":False,
           "trainable_state":{"tensors":332,"numel":numel,"FP32_bytes":byte_count,"every_tensor_exact_original":True},
           "frozen_original_state":{"tensors":165,"logical_sha256":frozen_logical,"included_in_terminal":False},
           "full_reconstruction":{"tensors":497,"original_logical_sha256":base_logical,"reconstructed_logical_sha256":reconstructed_logical,"all_original_values_exact":True,
                                  "limitation":"Saved artifact equals retained base. It is not an independent observation of every past VM memory state or hidden operation."},
           "optimizers":optimizer_reviews,"partial_accumulated_gradients":0,"sampler_schedule_exact":True,
           "RNG":{"Python_seed240_unchanged_exact":True,"NumPy_seed240_unchanged_exact":True,"tensor_states":rng_pins,
                  "Torch_expected_seed_only_comparison":"Not claimed: CPU architecture initialization legitimately consumes random numbers; tensor schema/hash retained without replay."},
           "parity_limits":{"case_inferred_from_fixed_order":protocol["fit_cases"][0]["id"],"declared_raw256_max_error_threshold":1e-6,
             "actual_error_recorded":False,"first_case_completed_comparison_calls":2,
             "not_executed":["Remaining3 parity comparisons","128case TRAIN fit schedule","all snapshots/monitors/publicDEV evaluations"],
             "no_PNG_or_raw_NPZ_payloads":True,"formula_static_comparison":"Same visible-normalized bilinear512, nearestmaskwhite, clamp andbilinear256 delivered composition. Own explicitdetach16=True matches modeldefault; wrapper w1/adainFalse same.",
             "structural_differences":["Canonical wrapper Boolean active-batch indexing copies/layout versus direct tensor path.","Canonical wrapper inference_mode versus direct parity no_grad."],
             "cause_established":False,"GPU_rounding_or_layout_magnitude":"Unknown; actualerror and intermediateparityarrays were not returned. Do not infer a harmless tolerance issue."},
           "retention":{"preserve_failed_archive_and_state":True,"resume_or_rerun_unchanged":False,"earlier_snapshot_rescue_available":False,
                        "terminal_state_is_initial_base_not_trained_checkpoint":True},
           "local_execution":{"weights_only_CPU_loads":2,"model_instantiations":0,"neural_forwards":0,"backward_calls":0,"optimizer_steps":0,"VM_calls":0,
                              "returned_executable_code_run":False},
           "model_qualified":False,"goal_complete":False,"audit_seconds":time.monotonic()-started}
    target=PARENT/"independent_return_audit.json"
    with target.open("x",encoding="utf-8",newline="\n")as stream:json.dump(audit,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({"complete":True,"audit":pin(target),"training_executed":False,"trainable_tensors":332,"all_exact_original":True,
                      "optimizer_moments":0,"logical_sha256":reconstructed_logical,"seconds":audit["audit_seconds"]},indent=2))

if __name__=="__main__":main()

