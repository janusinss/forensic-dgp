"""Reconstruct frozen pilot selection and prepared data; never train or run models."""
import argparse
import collections
import hashlib
from pathlib import Path
import sys
import tarfile

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cctv_dgp_pilot import read,write,sha,verify_bundle
from cctv_camera_stress import reference_canvas,degrade


def require(condition,message):
    if not condition:
        raise ValueError(message)


def audit(root,archive=None,reconstruct_camera=False):
    root=Path(root);p=verify_bundle(root)
    gate=read(root/"evidence/outputs/cctv_dgp_reference_gate_v2/results.json")
    review=read(root/"evidence/outputs/cctv_dgp_reference_gate_v2/input_review.json")
    split=read(root/"evidence/outputs/downloaded_phase4/outputs/phase4_with_progress/split.json")
    require(sha(root/"evidence/outputs/cctv_dgp_reference_gate_v2/results.json")==p["source_selection"]["qualification_results_sha256"],"Gate binding differs")
    require(sha(root/"evidence/outputs/cctv_dgp_reference_gate_v2/input_review.json")==p["source_selection"]["input_review_sha256"],"Review binding differs")
    require(sha(root/"evidence/outputs/downloaded_phase4/outputs/phase4_with_progress/split.json")==p["historical_split_sha256"],"Inherited split binding differs")
    sources=sorted({r["source"] for r in p["references"]})
    selected=[r for r in gate["rows"] if r["admitted"] and r["source_file"] not in review["new_training_exclusions"]]
    quota=min(sum(r["role"]=="train" and r["source"]==s for r in selected) for s in sources)
    expected=[];truncated=[]
    for role in ("train","validation"):
        for source in ("dataset/thumbnails128x128","dataset/asian_faces"):
            rows=[r for r in selected if r["role"]==role and r["source"]==source]
            chosen=rows[:quota] if role=="train" else rows
            expected.extend(r["source_file"] for r in chosen);truncated.extend(r["source_file"] for r in rows[len(chosen):])
    require([r["source_file"] for r in p["references"]]==expected,"Derived balanced selection differs")
    require(truncated==p["source_selection"]["balance_only_training_exclusions"],"Balance exclusions differ")
    inherited={role:{s.replace('\\','/') for s in split[key]} for role,key in (("train","train"),("validation","validation"))}
    parent={r["source_file"]:r for r in gate["rows"]}
    refs={r["id"]:r for r in p["references"]};targets={};masks={}
    for r in refs.values():
        require(r["source_file"] in inherited[r["role"]],"Inherited role differs")
        native=root/r["native"];require(sha(native)==r["source_sha256"],"Native source changed")
        target,observed,bounds=reference_canvas(native)
        require(bounds==r["bounds"] and r["matrix112"]==parent[r["source_file"]]["matrix112"],"Reference geometry differs")
        require(hashlib.sha256(target.tobytes()).hexdigest()==r["target_rgb_sha256"],"Target pixels differ")
        require(hashlib.sha256(observed.tobytes()).hexdigest()==r["observed_sha256"],"Observation pixels differ")
        require(np.array_equal(target,np.asarray(Image.open(root/r["target"]).convert("RGB"))),"Stored target differs")
        require(np.array_equal(observed,np.asarray(Image.open(root/r["observed"]))>0),"Stored observation differs")
        targets[r["id"]]=target;masks[r["id"]]=observed
    cases=[c for cs in p["training_epochs"].values() for c in cs]+p["validation_cases"]
    regenerated=0
    for c in cases:
        ref=refs[c["reference_id"]];rgb=np.asarray(Image.open(root/c["input"]).convert("RGB"));mask=masks[ref["id"]]
        require(rgb.shape==(256,256,3) and np.array_equal(rgb[~mask],targets[ref["id"]][~mask]),"Case shape/context differs")
        if c["profile"]=="clear":
            require(np.array_equal(rgb,targets[ref["id"]]),"Clear anchor differs")
        if reconstruct_camera:
            rebuilt,details=degrade(targets[ref["id"]],ref["bounds"],c["camera"],c["proxy_details"]["seed"])
            require(np.array_equal(rgb,rebuilt) and details==c["proxy_details"],"Prepared camera proxy differs")
            regenerated+=1
    # Rebuild the RNG schedule independently of VM training/output scores.
    from prepare_cctv_dgp_vm import jitter
    from cctv_camera_stress import PROFILES
    training=[r for r in p["references"] if r["role"]=="train"]
    for epoch in (1,2):
        rng=np.random.default_rng(p["seed"]+epoch);order=rng.permutation(len(training))
        for slot,(index,c) in enumerate(zip(order,p["training_epochs"][str(epoch)])):
            profile=jitter(PROFILES[(slot+epoch-1)%5],rng);noise_seed=int(rng.integers(0,2**32))
            require(c["reference_id"]==training[index]["id"] and c["camera"]==profile and c["proxy_details"]["seed"]==noise_seed,"Frozen training schedule differs")
    for index,r in enumerate(r for r in p["references"] if r["role"]=="validation"):
        for profile_index,profile in enumerate(PROFILES):
            c=p["validation_cases"][index*5+profile_index]
            require(c["reference_id"]==r["id"] and c["camera"]==profile and c["proxy_details"]["seed"]==p["seed"]+100000+index*10+profile_index,"Fixed validation schedule differs")
    count=0
    if archive:
        expected_names={"cctv_dgp_vm_bundle/"+n for n in p["assets_sha256"]}|{"cctv_dgp_vm_bundle/protocol.json","cctv_dgp_vm_bundle/protocol.sha256"}
        with tarfile.open(archive,"r:gz") as tar:
            seen=set()
            for member in tar:
                require(member.isfile() and member.name in expected_names and member.name not in seen,"Unexpected/duplicate archive entry")
                digest=hashlib.sha256()
                with tar.extractfile(member) as stream:
                    for block in iter(lambda:stream.read(1024*1024),b''):
                        digest.update(block)
                relative=member.name.split('/',1)[1]
                require(digest.hexdigest()==sha(root/relative),"Archive member differs")
                seen.add(member.name);count+=1
            require(seen==expected_names,"Incomplete archive inventory")
        checksum=Path(str(archive)+".sha256").read_bytes()
        require(b'\r' not in checksum and checksum.decode('ascii')==sha(archive)+"  "+Path(archive).name+"\n","Checksum digest/LF format differs")
    return {"complete":True,"protocol_sha256":sha(root/"protocol.json"),"references_reconstructed":len(refs),
            "training_references":len(training),"validation_references":len(refs)-len(training),"prepared_cases_checked":len(cases),
            "camera_cases_regenerated":regenerated,"archive_members_checked":count,"archive_sha256":sha(archive) if archive else None,
            "source_groups":dict(collections.Counter(r["role"]+"/"+r["source"] for r in refs.values())),
            "local_model_forwards":0,"optimizer_updates":0,"cuda_execution_verified":False,"reserved_native_used":False}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT/"outputs/cctv_dgp_vm_bundle_v1")
    parser.add_argument("--archive",type=Path)
    parser.add_argument("--reconstruct-camera",action="store_true")
    parser.add_argument("--save",type=Path)
    args=parser.parse_args();report=audit(args.root,args.archive,args.reconstruct_camera)
    if args.save:
        write(args.save,report)
    print(report)
