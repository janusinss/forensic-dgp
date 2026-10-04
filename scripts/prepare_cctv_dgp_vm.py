"""Freeze a self-contained matched DGP camera pilot. Preparation makes zero updates."""
import collections
import hashlib
from pathlib import Path
import shutil
import sys
import tarfile
import time

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cctv_dgp_pilot import sha,read,write,verify_bundle
from cctv_camera_stress import PROFILES,reference_canvas,degrade

DEST=ROOT/"outputs/cctv_dgp_vm_bundle_v1"
ARCHIVE=ROOT/"outputs/cctv-dgp-vm-bundle.tar.gz"
REPORT=ROOT/"outputs/cctv_dgp_pilot_protocol_v1"
PREFIX="cctv_dgp_vm_bundle"
SEED=20261003
GATE="outputs/cctv_dgp_reference_gate_v2"
PINS={
 GATE+"/frozen_protocol.json":"f29d9b728f0e5c1aef954832547115682c59d93803dda58e9046417f51deff9b",
 GATE+"/results.json":"3cee64b945a4b6e43179f2f1c93526ad9295e0a0ae1c5b0b5ebe461cb5366f32",
 GATE+"/independent_audit.json":"64cbaeb0ac36ce7bb918ec4fd417ae84e3e60dda1df4cbeffcbc29e16d7a9aaf",
 GATE+"/input_review.json":"69b72552932275de8f55393f90c4f94ad778bbd0f42d29999910982dca4fba92",
 "outputs/downloaded_phase4/outputs/phase4_with_progress/split.json":"5c71bc358a351d50e3c0a7d76abe749cb4412aca80d2c7eb4de5fb33dbd29071",
 "checkpoints/dgp_zamboanga_final.pth":"b6376f56c4161ef1f26a8f9efb0bb309b3014f4ecdeaca5399a47c1c4607586c"}
TEACHERS={
 "arcface":(Path.home()/".insightface/models/buffalo_l/w600k_r50.onnx","4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"),
 "vgg":(Path.home()/".cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth","dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44")}
REQUIREMENTS="""# Keep the existing matching CUDA torch/torchvision installation.
numpy==1.26.4
Pillow==11.3.0
opencv-python-headless==4.11.0.86
scipy==1.15.3
scikit-image==0.24.0
onnx==1.17.0
onnxruntime==1.22.1
onnx2torch==1.5.15
face-alignment==1.4.1
"""
SETUP='''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - <<'PY'
import sys, torch, torchvision
assert sys.version_info >= (3,10), 'Python 3.10 or newer required'
assert torch.cuda.is_available(), 'Activate your existing CUDA environment first'
assert 'L4' in torch.cuda.get_device_name(0), 'Existing L4 VM required'
print('Existing CUDA runtime:',torch.__version__,torchvision.__version__,torch.cuda.get_device_name(0))
open('cuda_runtime_before.txt','x').write(torch.__version__+'\\n'+torchvision.__version__+'\\n')
PY
if ! python3 -m venv --system-site-packages .venv; then
  echo 'Install matching python3-venv (for Python3.10: sudo apt-get install python3.10-venv), then use a new package directory.' >&2
  exit 1
fi
source .venv/bin/activate
timeout 20m python -m pip install -r requirements_cctv_dgp_vm.txt
python - <<'PY'
from pathlib import Path
import torch, torchvision
assert Path('cuda_runtime_before.txt').read_text().splitlines()==[torch.__version__,torchvision.__version__], 'CUDA torch/vision changed; stop'
assert torch.cuda.is_available(), 'CUDA unavailable after setup'
PY
python -m pip freeze > environment.txt
python -u scripts/run_cctv_dgp_pilot_vm.py --verify
'''
RUN='''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
test -f .venv/bin/activate
source .venv/bin/activate
test ! -e outputs/cctv_dgp_pilot
python -u scripts/run_cctv_dgp_pilot_vm.py 2>&1 | tee pilot.log
python -u scripts/audit_cctv_dgp_pilot_results.py --root "$PWD" --results outputs/cctv_dgp_pilot
test ! -e cctv-dgp-results.tar.gz
tar -czf cctv-dgp-results.tar.gz protocol.json protocol.sha256 environment.txt cuda_runtime_before.txt pilot.log outputs/cctv_dgp_pilot
sha256sum cctv-dgp-results.tar.gz > cctv-dgp-results.tar.gz.sha256
echo 'Pilot and file audit complete. Download cctv-dgp-results.tar.gz and its .sha256. Native/visual review still required.'
'''


def write_text(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8",newline="\n") as stream:
        stream.write(value)


def jitter(profile,rng):
    p=dict(profile)
    if p["id"]=="blur_lr24":
        p.update(long_edge=int(rng.integers(24,33)),sigma=float(rng.uniform(.6,2)),jpeg=int(rng.integers(35,66)))
    elif p["id"]=="lowlight_lr32":
        p.update(long_edge=int(rng.integers(32,65)),exposure=float(rng.uniform(.2,.65)),sigma=float(rng.uniform(.5,1.4)),
                 shot=float(rng.uniform(.0004,.0012)),read=float(rng.uniform(.001,.003)),jpeg=int(rng.integers(30,61)))
    elif p["id"]=="motion_lr48":
        p.update(long_edge=int(rng.integers(40,65)),motion=int(rng.choice([5,7,9,11])),jpeg=int(rng.integers(25,61)))
    elif p["id"]=="compound_lr24":
        p.update(long_edge=int(rng.integers(24,41)),exposure=float(rng.uniform(.15,.4)),sigma=float(rng.uniform(.7,2)),
                 motion=int(rng.choice([3,5,7])),shot=float(rng.uniform(.0006,.0018)),read=float(rng.uniform(.002,.004)),
                 jpeg=int(rng.integers(20,51)))
    return p


def main():
    if any(p.exists() for p in (DEST,ARCHIVE,ARCHIVE.with_name(ARCHIVE.name+".sha256"),REPORT)):
        raise ValueError("Preserve prepared/partial package; version any new preparation")
    for name,pin in PINS.items():
        if sha(ROOT/name)!=pin:
            raise ValueError("Audited parent changed: "+name)
    for path,pin in TEACHERS.values():
        if sha(path)!=pin:
            raise ValueError("Cached teacher changed")
    gate=read(ROOT/GATE/"results.json");review=read(ROOT/GATE/"input_review.json")
    if not gate["complete"] or not read(ROOT/GATE/"independent_audit.json")["complete"]:
        raise ValueError("Reference gate/audit incomplete")
    selected=[r for r in gate["rows"] if r["admitted"] and r["source_file"] not in review["new_training_exclusions"]]
    sources=["dataset/thumbnails128x128","dataset/asian_faces"]
    quota=min(sum(r["role"]=="train" and r["source"]==s for r in selected) for s in sources)
    picked=[];balance_exclusions=[]
    for role in ("train","validation"):
        for source in sources:
            rows=[r for r in selected if r["role"]==role and r["source"]==source]
            chosen=rows[:quota] if role=="train" else rows
            picked.extend(chosen)
            balance_exclusions.extend(r["source_file"] for r in rows[len(chosen):])
    if quota!=451 or len(picked)!=1012:
        raise ValueError("Reviewed source cohort differs")
    DEST.mkdir();REPORT.mkdir()
    start=time.monotonic();references=[];assets={}
    def register(path):
        assets[path.relative_to(DEST).as_posix()]=sha(path)
    def copy(source,name):
        dst=DEST/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dst);register(dst)
    def save(rgb,name):
        dst=DEST/name;dst.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(rgb).save(dst);register(dst)
    split=read(ROOT/"outputs/downloaded_phase4/outputs/phase4_with_progress/split.json")
    inherited={role:{p.replace('\\','/') for p in split[key]} for role,key in (("train","train"),("validation","validation"))}
    for i,row in enumerate(picked):
        path=ROOT/row["source_file"]
        if row["source_file"] not in inherited[row["role"]] or sha(path)!=row["sha256"]:
            raise ValueError("Source role/bytes differ")
        refid=("tr_" if row["role"]=="train" else "va_")+("ffhq_" if row["source"]==sources[0] else "asian_")+path.stem.rsplit('_',1)[-1]
        target,observed,bounds=reference_canvas(path)
        if bounds!=row["bounds"] or hashlib.sha256(target.tobytes()).hexdigest()!=row["target_rgb_sha256"] or hashlib.sha256(observed.tobytes()).hexdigest()!=row["observed_sha256"]:
            raise ValueError("Source target canvas differs")
        native=f"data/native/{refid}{path.suffix}";target_path=f"data/targets/{refid}.png";mask=f"data/observed/{refid}.png"
        copy(path,native);save(target,target_path);save(observed.astype(np.uint8)*255,mask)
        references.append({"id":refid,"source":row["source"],"role":row["role"],"source_file":row["source_file"],
          "source_sha256":row["sha256"],"native":native,"native_size":row["native_size"],"target":target_path,"observed":mask,
          "bounds":bounds,"matrix112":row["matrix112"],"landmarks5":row["landmarks5"],
          "target_rgb_sha256":row["target_rgb_sha256"],"observed_sha256":row["observed_sha256"],
          "geometric_support":row["observed_feature_support"],"quality_diagnostics":row["quality_diagnostics"]})
        if (i+1)%256==0:
            print(f"Prepared references {i+1}/{len(picked)} elapsed={time.monotonic()-start:.0f}s",flush=True)
    epochs={};training=[r for r in references if r["role"]=="train"]
    for epoch in (1,2):
        rng=np.random.default_rng(SEED+epoch);order=rng.permutation(len(training));cases=[]
        for slot,index in enumerate(order):
            ref=training[index];profile=jitter(PROFILES[(slot+epoch-1)%5],rng)
            noise_seed=int(rng.integers(0,2**32));target=np.asarray(Image.open(DEST/ref["target"]).convert("RGB"))
            rgb,details=degrade(target,ref["bounds"],profile,noise_seed)
            caseid=f"e{epoch}_{ref['id']}";name=f"data/train/e{epoch}/{caseid}.png";save(rgb,name)
            cases.append({"id":caseid,"reference_id":ref["id"],"source":ref["source"],"profile":profile["id"],
                          "input":name,"camera":profile,"proxy_details":details})
        epochs[str(epoch)]=cases
    cases=[];validation=[r for r in references if r["role"]=="validation"]
    for index,ref in enumerate(validation):
        target=np.asarray(Image.open(DEST/ref["target"]).convert("RGB"))
        for pindex,profile in enumerate(PROFILES):
            rgb,details=degrade(target,ref["bounds"],profile,SEED+100000+index*10+pindex)
            caseid=f"{ref['id']}_{profile['id']}";name=f"data/validation/{caseid}.png";save(rgb,name)
            cases.append({"id":caseid,"reference_id":ref["id"],"source":ref["source"],"profile":profile["id"],
                          "input":name,"camera":dict(profile),"proxy_details":details})
    preview_ids=[]
    for source in sources:
        ref=next(r for r in validation if r["source"]==source)
        preview_ids.extend(c["id"] for c in cases if c["reference_id"]==ref["id"])
    files=["cctv_dgp_pilot.py","scripts/run_cctv_dgp_pilot_vm.py","scripts/audit_cctv_dgp_pilot_results.py",
           "scripts/audit_cctv_dgp_bundle.py","scripts/cctv_camera_stress.py","scripts/prepare_cctv_dgp_vm.py",
           "tests/test_cctv_dgp_pilot.py","CCTV_DGP_VM.md","CCTV_DGP_NEXT_PILOT.md"]
    files.extend(p.relative_to(ROOT).as_posix() for p in sorted((ROOT/"models").glob("*.py")))
    for name in files:
        copy(ROOT/name,name)
    for name in PINS:
        if not name.startswith("checkpoints/"):
            copy(ROOT/name,"evidence/"+name)
    copy(ROOT/"checkpoints/dgp_zamboanga_final.pth","weights/dgp_zamboanga_final.pth")
    copy(TEACHERS["arcface"][0],"weights/w600k_r50.onnx")
    vgg=torch.load(TEACHERS["vgg"][0],map_location="cpu",weights_only=True)
    trunk={k.removeprefix("features."):v for k,v in vgg.items() if k.startswith("features.") and int(k.split('.')[1])<27}
    torch.save(trunk,DEST/"weights/vgg19_first27.pth");register(DEST/"weights/vgg19_first27.pth")
    for name,text in (("scripts/setup_cctv_dgp_vm.sh",SETUP),("scripts/run_cctv_dgp_vm.sh",RUN),("requirements_cctv_dgp_vm.txt",REQUIREMENTS)):
        write_text(DEST/name,text);register(DEST/name)
    protocol={"format":"dgp-cctv-matched-pilot-v1","date":"2026-10-03","seed":SEED,"references":references,
      "source_selection":{"qualification_results_sha256":PINS[GATE+"/results.json"],"input_review_sha256":PINS[GATE+"/input_review.json"],
        "training_per_source":quota,"training_references":902,"validation_references":110,"validation_source_counts":{s:sum(r["source"]==s for r in validation) for s in sources},
        "balance_only_training_exclusions":balance_exclusions,"new_input_review_exclusions":review["new_training_exclusions"],
        "all_references_visually_reviewed":False,"full80000_identity_overlap_audited":False},
      "training_epochs":epochs,"validation_cases":cases,"preview_case_ids":preview_ids,
      "arms":[{"id":"camera_no_identity","lambda_identity":0.},{"id":"camera_identity","lambda_identity":.1}],
      "epochs":2,"batch_size":8,"updates_per_arm":226,"expected_total_updates":452,"runtime_cap_seconds":5400,
      "optimizer":{"type":"Adam","backbone_lr":2e-6,"head_lr":1e-5,"weight_decay":1e-5,"gradient_clip_norm":1.,"amp":False,"ema":False,
        "normalization_running_statistics_frozen":True,"fresh_each_arm":True,"grid_sample_cuda_backward_may_be_nondeterministic":True},
      "loss":{"observed_charbonnier_weight":1.,"charbonnier_epsilon":.001,"pooled_color_weight":.05,"vgg_weight":.1,
              "vgg_taps":[3,8,17,26],"vgg_tap_weights":[.1,.2,1.,1.],"sobel_weight":.05,"fan":False,"fft":False},
      "identity":{"diagnostic":"ArcFace_observed_fixed","input":"RGB112 float bilinear fixed target inverse affine; nearest observed support, unsupported pixels RGB128/255; normalize to[-1,1]",
                  "eligibility":"Same input-only qualified reference cohort and matrix for every arm/epoch; no output-dependent detection",
                  "claim":"Reference appearance diagnostic, not identification accuracy or exact recovered identity"},
      "metrics":{"pixel_basis":"Exact RGB8 PNG, observation mask only; SSIM7 map within mask eroded3px",
                 "PSNR_aggregation":"-10log10(mean case MSE); zeroMSE PSNR null plus perfect-match flag; differs from previous mean per-image PSNR",
                 "selection":"At least0.1dB degraded MSE-based PSNR gain versus current best; every source/profile and clear group MSE no higher than starting baseline+1e-12; SSIM and ArcFace_observed_fixed no lower than baseline-1e-6; all fixed eligible pairs present",
                 "resizing_baseline_reported":True,"production_promotion":False},
      "weights":{"dgp":"weights/dgp_zamboanga_final.pth","arcface":"weights/w600k_r50.onnx","vgg_trunk":"weights/vgg19_first27.pth"},
      "teacher_source_sha256":{k:pin for k,(_,pin) in TEACHERS.items()},"validation_camera_profiles":PROFILES,
      "precomputed_proxy_inputs":True,"historical_split_sha256":PINS["outputs/downloaded_phase4/outputs/phase4_with_progress/split.json"],
      "local_preparation_model_forwards":0,"local_optimizer_updates":0,"native_reserved_used":False,
      "limitations":["Processed reference photos, usually below256 captured pixels; not pristine HQ or calibrated CCTV sensor data",
                     "Unequal qualified validation source counts; report each source/profile separately",
                     "Historical development validation and supervised teacher diagnostic, not untouched identity evaluation",
                     "No Zamboanga CCTV validation or personal ethnicity inference",
                     "Matched new loss/data/norm policy is not the immutable historical Phase5 recipe; only identity coefficient differs between new arms"],
      "assets_sha256":assets}
    write(DEST/"protocol.json",protocol);write_text(DEST/"protocol.sha256",sha(DEST/"protocol.json")+"\n")
    verify_bundle(DEST)
    with tarfile.open(ARCHIVE,"x:gz",compresslevel=5) as archive:
        for p in sorted(DEST.rglob('*')):
            if p.is_file():
                info=tarfile.TarInfo(PREFIX+"/"+p.relative_to(DEST).as_posix());info.size=p.stat().st_size;info.mtime=0;info.mode=0o644
                with p.open('rb') as stream:
                    archive.addfile(info,stream)
    write_text(ARCHIVE.with_name(ARCHIVE.name+".sha256"),sha(ARCHIVE)+"  "+ARCHIVE.name+"\n")
    write(REPORT/"build.json",{"complete":True,"bundle":DEST.relative_to(ROOT).as_posix(),"archive":ARCHIVE.relative_to(ROOT).as_posix(),
       "archive_sha256":sha(ARCHIVE),"archive_bytes":ARCHIVE.stat().st_size,"protocol_sha256":sha(DEST/"protocol.json"),
       "asset_files":len(assets),"training_references":902,"validation_references":110,"validation_cases":550,
       "updates_per_arm":226,"total_updates":452,"seconds":time.monotonic()-start,
       "local_model_forwards":0,"local_optimizer_updates":0,"actual_cuda_verified":False,"training_run_started":False})
    print(read(REPORT/"build.json"))


if __name__=="__main__":
    main()
