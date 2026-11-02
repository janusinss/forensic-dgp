"""Independent local packet/data audit. No neural imports, VM calls or execution."""
from __future__ import annotations
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
import time
import numpy as np
from PIL import Image, ImageDraw

BASE=Path(r"C:\xampp\htdocs\YEAR 4\Testing_FACE_MASK_COMPLETION")
DATA=Path(r"C:\xampp\htdocs\YEAR 4\Testing")
PARENT=BASE/"outputs/completion_codeformer_finetune_v1_packet"
ROOT=PARENT/"cctv_completion_codeformer_finetune_v1_vm"
EXPECTED="0dfe963bd29856ea6201d6397aa25d651ca0291661fa3ec26539c9c770e31b8f"
WORKER_SHA="8f3a69bc3209590a6919adfa67afa9dbd3da4857e9180c337123a6ee65e332cb"
EXPORTER_SHA="34c1db228ca0053dda3eba5e19b49379c4e8bd8387ac0a4d0a18747d15c3355f"
SHELL_SHA="9afa228c5e775d97a1207110f7a621a09227826fda7e040fe3268eb1e83a2c9a"
WEIGHT_SHA="b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797"
HISTORY=DATA/"outputs/completion_mat_mirror_comparison_v1/protocol.json"
HISTORY_SHA="d3b35fe043c198007515f3dc0a221b7c7b369b7edd8d7b9bd45122cc550cf76f"
INPUT_REVIEW_SHA="c103673ca233d28a5a3a82855611886dc7543ddd8922271ea943de02f2456295"
KINDS=("lower_face","eye_band","image_left_lens","cheek_object")

def need(ok,message):
    if not ok:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def pin(path):
    path=Path(path);need(path.is_file() and not path.is_symlink(),"Missing/linked: "+str(path))
    return {"path":str(path.resolve()),"bytes":path.stat().st_size,"sha256":sha(path)}

def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def check(path,expected):
    actual=pin(path)
    need(actual["bytes"]==expected["bytes"] and actual["sha256"]==expected["sha256"],"Pin mismatch: "+str(path))
    return actual

def member(root,name):
    need(isinstance(name,str) and name and "\\" not in name and ":" not in name,"Bad member name")
    p=PurePosixPath(name);need(not p.is_absolute() and all(x not in (".","..") for x in p.parts),"Traversal")
    target=root.joinpath(*p.parts)
    need(target.resolve().is_relative_to(root.resolve()),"Escape")
    return target

def image(path,size,mode):
    with Image.open(path) as im:
        need(im.format=="PNG" and im.size==size and im.mode==mode,"Image format/shape: "+str(path))
        return np.array(im,copy=True)

def fixed(kind):
    result=Image.new("L",(256,256));draw=ImageDraw.Draw(result)
    if kind=="lower_face":draw.polygon([(48,140),(208,140),(218,211),(180,232),(76,232),(38,211)],fill=255)
    elif kind=="eye_band":draw.rectangle((48,78,207,131),fill=255)
    elif kind=="image_left_lens":draw.ellipse((54,82,116,126),fill=255)
    elif kind=="cheek_object":draw.ellipse((148,126,207,187),fill=255)
    else:raise ValueError("Unknown geometry")
    return np.array(result,copy=True)

def import_pure(path,name):
    tree=ast.parse(Path(path).read_text(encoding="utf-8-sig"),feature_version=(3,10))
    need(not any(isinstance(n,ast.Attribute) and n.attr=="file_digest" for n in ast.walk(tree)),"Python3.11 hash dependency")
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);need("torch" not in sys.modules,"Unexpected neural import")
    return module

def main():
    started=time.monotonic();need("torch" not in sys.modules,"Audit must start without Torch")
    protocol_pin=pin(ROOT/"protocol.json");need(protocol_pin["sha256"]==EXPECTED,"Unexpected stage")
    p=read(ROOT/"protocol.json");files={r["path"]:r for r in p["files"]}
    need(len(files)==len(p["files"])==329,"File count/duplicates")
    actual={f.relative_to(ROOT).as_posix() for f in ROOT.rglob("*") if f.is_file()}
    need(actual==set(files)|{"protocol.json"},"Unlisted packet file")
    file_pins=[check(member(ROOT,rel),row) for rel,row in files.items()]
    need(sum(r["bytes"] for r in files.values())==28653275,"Payload bytes differ")
    need(not any(Path(rel).suffix.lower() in (".pth",".pt",".ckpt") or rel.startswith("weights/") for rel in files),"Weights embedded in packet")
    worker=import_pure(ROOT/"scripts/cctv_completion_codeformer_finetune_v1_vm.py","independent_pilot_contract")
    exporter=import_pure(ROOT/"scripts/export_completion_codeformer_finetune_v1.py","independent_pilot_exporter")
    need(sha(BASE/"scripts/cctv_completion_codeformer_finetune_v1_vm.py")==WORKER_SHA==files["scripts/cctv_completion_codeformer_finetune_v1_vm.py"]["sha256"],"Worker source differs")
    need(sha(BASE/"scripts/export_completion_codeformer_finetune_v1.py")==EXPORTER_SHA==files["scripts/export_completion_codeformer_finetune_v1.py"]["sha256"],"Exporter differs")
    shell_path=ROOT/"scripts/run_completion_codeformer_finetune_v1.sh"
    need(sha(shell_path)==SHELL_SHA and b"\r" not in shell_path.read_bytes(),"Shell bytes/LF differ")
    need(p["contract"]==worker.CONTRACT and p["manual_execution_only"] and p["optimizer_updates_locally"]==p["local_neural_calls"]==0,"Contract/manual status differs")
    source_review_path=ROOT/"evidence/finetune_source_admission.json";source_review=read(source_review_path)
    need(source_review["complete"] and source_review["independent"] and source_review["script_sha256"]==WORKER_SHA and source_review["independent_tests"]["passed"]==12,"Source admission incomplete")
    for row in source_review["source_files"]:check(row["path"],row)
    for relative in worker.SOURCE_MODULES.values():
        need(sha(ROOT/relative)==sha(BASE/relative),"Loaded model source differs: "+relative)
    py_files=[]
    for path in ROOT.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8-sig"),filename=str(path),feature_version=(3,10))
        need(not any(isinstance(n,ast.Attribute) and n.attr=="file_digest" for n in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig")))),"file_digest dependency")
        py_files.append(pin(path))
    had=hasattr(hashlib,"file_digest");old=getattr(hashlib,"file_digest",None)
    try:
        if had:delattr(hashlib,"file_digest")
        for module in (worker,exporter):
            need(module.sha(ROOT/"protocol.json")==EXPECTED,"Chunked hash helper failed without file_digest")
    finally:
        if had:hashlib.file_digest=old
    need(exporter.MAX_BYTES==1536*1024**2 and exporter.MAX_SECONDS==900 and exporter.VM_NAME==ROOT.name,"Exporter bounds/root differ")
    review_path=ROOT/"evidence/root_input_review.json";need(sha(review_path)==INPUT_REVIEW_SHA,"Input review differs")
    review=read(review_path)
    need(review["complete"] and review["all5_source_pages_viewed"] and review["all40_source_targets_reviewed"] and not review["model_outputs_used_for_selection"],"Prospective input review incomplete")
    readiness=read(ROOT/"evidence/training_data_readiness.json")
    need(sha(ROOT/"evidence/training_data_readiness.json")=="87dabe0b2c81350d91f9c0a00666c44dd8b514d1ac23d83c2b571f4cef80f0d6","Readiness changed")
    selected=sorted(readiness["genuine_HQ_candidate_sources"]["training_sources"],key=lambda x:x["reference_id"])[:40]
    need(len(selected)==len(p["references"])==40,"Reference count differs")
    need([r["id"] for r in p["references"]]==[r["reference_id"] for r in selected],"Fixed first40 selection differs")
    need(review["fit"]==[r["id"] for r in p["references"][:32]] and review["monitor"]==[r["id"] for r in p["references"][32:]],"Root reviewed role IDs differ")
    cells={c["id"]:c for page in review["pages"] for c in page["cells"]}
    reference_audits=[];source_hashes={};target_hashes={};pixel_hashes={};clean={}
    original_pins=[]
    for i,(row,oldrow) in enumerate(zip(p["references"],selected)):
        rid=row["id"];role="fit" if i<32 else "monitor"
        need(row["role"]==role and row["original_role"]==row["legacy_completion_split"]==oldrow["original_role"]==oldrow["legacy_completion_split"]=="train","Role changed")
        need(row["original_source1024"]==oldrow["source"] and row["retained_target256"]==oldrow["target256"] and row["attribution"]==oldrow["attribution"],"Provenance differs")
        need(row["clean_review_decision"]==oldrow["source_review_decision"]=="accept_clean_restoration","Clean decision differs")
        original=check(oldrow["source"]["path"],oldrow["source"]);target=check(oldrow["target256"]["path"],oldrow["target256"]);original_pins.extend([original,target])
        clean256=member(ROOT,row["clean256"]);clean512=member(ROOT,row["clean512"])
        need(sha(clean256)==target["sha256"]==cells[rid]["input"]["sha256"],"Exact retained/visible target bytes differ")
        with Image.open(original["path"]) as im:
            need(im.format=="PNG" and im.mode=="RGB" and im.size==(1024,1024),"HQ1024 schema changed")
            expected256=np.array(im.resize((256,256),Image.Resampling.LANCZOS),copy=True)
            expected512=np.array(im.resize((512,512),Image.Resampling.LANCZOS),copy=True)
        got256=image(clean256,(256,256),"RGB");got512=image(clean512,(512,512),"RGB")
        need(np.array_equal(got256,expected256) and np.array_equal(got512,expected512),"Direct LANCZOS derivation differs")
        source_hashes.setdefault(role,set()).add(original["sha256"]);target_hashes.setdefault(role,set()).add(target["sha256"])
        pixel_hashes.setdefault(role,set()).add(hashlib.sha256(got256.tobytes()).hexdigest());clean[rid]=got256
        reference_audits.append({"id":rid,"role":role,"source1024":original,"retained256":target,"clean256":pin(clean256),"clean512":pin(clean512),"direct_LANCZOS256_and512_exact":True,"source_role_TRAIN":True})
    need(not source_hashes["fit"]&source_hashes["monitor"] and not target_hashes["fit"]&target_hashes["monitor"] and not pixel_hashes["fit"]&pixel_hashes["monitor"],"FIT/MONITOR exact source/content overlap")
    pages=[]
    for page in review["pages"]:
        path=ROOT/"evidence/input_review"/Path(page["path"]).name
        need(sha(path)==page["sha256"] and sha(page["path"])==page["sha256"],"Review grid changed")
        with Image.open(path) as sheet:
            for cell in page["cells"]:
                crop=np.array(sheet.crop((cell["x"],cell["y"],cell["x"]+256,cell["y"]+256)).convert("RGB"),copy=True)
                need(np.array_equal(crop,clean[cell["id"]]),"Grid source cell differs")
        pages.append(pin(path))
    need(len(pages)==5 and len(cells)==40,"Review cell coverage differs")
    cases=p["fit_cases"]+p["monitor_cases"];need(len(cases)==160 and len({r["id"] for r in cases})==160,"Case count differs")
    mask_counts={};mask_pins=[]
    expected_pairs={(r["id"],kind) for r in p["references"] for kind in KINDS}
    need({(r["reference_id"],r["mask_kind"]) for r in cases}==expected_pairs,"Geometry coverage differs")
    for row in cases:
        path=member(ROOT,row["mask"]);mask=image(path,(256,256),"L");expect=fixed(row["mask_kind"])
        need(np.array_equal(mask,expect) and row["mask_pixels"]==int((expect==255).sum()),"Fixed mask differs")
        need(row["role"]==next(r["role"] for r in p["references"] if r["id"]==row["reference_id"]),"Mask role mismatch")
        mask_counts[row["mask_kind"]]=int((expect==255).sum());mask_pins.append(pin(path))
    schedule=worker.training_schedule(p["fit_cases"]);need(p["schedule"]==schedule,"Schedule differs")
    flattened=[ident for cycle in schedule for ident in cycle]
    need(len(schedule)==32 and len(flattened)==len(set(flattened))==128 and set(flattened)=={r["id"] for r in p["fit_cases"]},"Schedule is not complete exactly-once128")
    need(sha(HISTORY)==HISTORY_SHA==sha(ROOT/"evidence/public_development_protocol.json"),"Historical DEV protocol changed")
    olddev=read(HISTORY);need(p["development_excluded_before_generation"]==olddev["excluded_before_generation"] and len(olddev["excluded_before_generation"])==4,"Historical exclusion decision changed")
    need([r["id"] for r in p["development_cases"]]==[r["id"] for r in olddev["cases"]],"DEV membership/order differs")
    dev=[];empty=[]
    for row,oldrow in zip(p["development_cases"],olddev["cases"]):
        need(all(row[k]==oldrow[k] for k in ("id","family","condition","operator_input_review")),"DEV metadata differs")
        need(row["historical_input"]==oldrow["input"] and row["historical_reviewed_mask"]==oldrow["reviewed"],"Historical path drift")
        src=DATA/oldrow["input"];msk=DATA/oldrow["reviewed"];ip=member(ROOT,row["input256"]);mp=member(ROOT,row["mask"])
        need(sha(ip)==sha(src)==olddev["sources_sha256"][oldrow["input"]] and sha(mp)==sha(msk)==olddev["sources_sha256"][oldrow["reviewed"]],"DEV input/mask bytes differ")
        a=image(ip,(256,256),"RGB");m=image(mp,(256,256),"L");need(np.isin(m,[0,255]).all(),"Nonbinary DEV mask")
        pixels=int((m==255).sum());need(pixels==row["mask_pixels"],"DEV mask count differs")
        if pixels==0:empty.append(row["id"])
        dev.append({"id":row["id"],"input":pin(ip),"mask":pin(mp),"pixels":pixels,"historical_bytes_exact":True})
    need(len(empty)==4 and len(dev)==32,"Expected28active4empty DEV")
    need(p["development_family_counts"]==dict(Counter(r["family"] for r in olddev["cases"])),"Family counts differ")
    need(p["checkpoint"]=={"path":"../cctv_completion_codeformer_gradient_v1_vm/weights/codeformer_inpainting.pth","bytes":370786611,"sha256":WEIGHT_SHA},"External checkpoint path/identity differs")
    retained=BASE/"outputs/completion_codeformer_gradient_v1_packet_r1/cctv_completion_codeformer_gradient_v1_vm/weights/codeformer_inpainting.pth"
    checkpoint=pin(retained);need(checkpoint["sha256"]==WEIGHT_SHA and checkpoint["bytes"]==370786611,"Retained gradient-base checkpoint differs")
    original_checkpoint=pin(DATA/"checkpoints/codeformer_inpainting.pth");need(checkpoint["sha256"]==original_checkpoint["sha256"],"Original checkpoint differs")
    need(p["export"]=={"stem":"cctv-completion-codeformer-finetune-v1","cap_bytes":1536*1024**2,"cap_seconds":900,"original_weight_archive_included":False,"terminal_state_max_count":1},"Export contract differs")
    need(p["automatic_resume_or_promotion"] is False and p["quality_qualified"] is False and p["goal_complete"] is False,"Qualification changed")
    # Rehash all packet and original selected TRAIN files after inspection.
    for q in [protocol_pin,*file_pins,*original_pins,checkpoint,original_checkpoint]:check(q["path"],q)
    need("torch" not in sys.modules,"Unexpected Torch import")
    audit={"format":"completion-codeformer-finetune-v1-independent-packet-admission","complete":True,"independent":True,"audited_UTC":datetime.now(timezone.utc).isoformat(),
           "auditor":pin(Path(__file__)),"protocol":protocol_pin,"protocol_sha256":EXPECTED,"worker_sha256":WORKER_SHA,
           "source_admission":pin(source_review_path),"file_count":len(file_pins),"payload_bytes":sum(q["bytes"] for q in file_pins),
           "all_packet_files":file_pins,"references":reference_audits,"reference_count":40,"fit_references":32,"monitor_references":8,
           "fit_monitor_disjoint_by_source_SHA_target_SHA_and_RGB_pixels":True,"person_identity_disjointness_established":False,
           "original_FFHQ_pretraining_overlap_known":False,"root_input_review":pin(review_path),"review_pages":pages,"source_cells_exact":40,
           "procedural_masks":len(mask_pins),"mask_geometry_pixels":mask_counts,"all160_fixed_geometries_exact":True,
           "schedule_128_fit_pairs_once":True,"schedule":schedule,"public_development":dev,"public_DEV_cases":32,"public_DEV_active":28,
           "public_DEV_empty_controls":empty,"historical_protocol":pin(HISTORY),"excluded_before_generation":olddev["excluded_before_generation"],
           "exclusions_unchanged":True,"external_checkpoint_contract":p["checkpoint"],"retained_local_external_equivalent":checkpoint,
           "original_checkpoint":original_checkpoint,"checkpoint_embedded_in_packet":False,"no_original_VAL_or_FINAL_pixels_read":True,
           "Python310_grammar_files":py_files,"file_digest_absence_helpers_tested":["worker.sha","exporter.sha"],"actual_Python310_execution":False,
           "exporter_review":{"source":pin(ROOT/"scripts/export_completion_codeformer_finetune_v1.py"),"shell":pin(shell_path),
             "max_uncompressed_and_compressed_bytes":1536*1024**2,"time_guard_seconds":900,"time_guard_kind":"cooperative, no separate OS-enforced export timeout",
             "evidence_scope":"protocol+terminal supervisor+trainer log+all regular outputs includingfailed.partial; no source/assets/baseweights",
             "mutations":"new archive/sidecar/exportreceipt only; exact old source/root/outputs retained",
             "regular_member_only":True,"source_rehash_before_and_after_each_member":True,"atomic_archive_publication":True,"LF_shell":True,
             "wrapper_always_attempts_export_after_worker_exit":True,"wrapper_preserves_nonzero_worker_status":True},
           "status":"packet_admitted_for_explicit_manual_L4_execution_after_archive_transfer_and_VM_verification",
           "newly_trained_model_qualified":False,"all_runtime_and_return_checks_still_required":True,
           "remaining_requirements":["Verify exactarchive/filepins onVM beforeexplicitmanualrun.","Four fresh canonicalwrapperGPU paritycases before firstupdate.","Return safe-load/reconstruction/optimizer/RNG/pixel/preservation audit and everyplannedvisualreview.","No automaticresume, best-snapshot rescue orapp promotion."],
           "local_execution":{"Torch_imported":False,"model_instantiations":0,"neural_calls":0,"gradients":0,"optimizer_updates":0,"VM_calls":0,"training_or_run_mode_executed":False},
           "audit_seconds":time.monotonic()-started}
    target=PARENT/"independent_packet_admission.json"
    with target.open("x",encoding="utf-8",newline="\n") as stream:json.dump(audit,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({"complete":True,"audit":pin(target),"files":len(file_pins),"refs":40,"masks":160,"DEV":32,"empty":empty,"seconds":audit["audit_seconds"]},indent=2))

if __name__=="__main__":main()

