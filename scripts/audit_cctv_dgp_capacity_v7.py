"""Independent training-fit arithmetic/state audit; no inference or backward."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import shutil
import tarfile

import numpy as np
from PIL import Image
import torch

PROTOCOL_SHA = "a1d2fbb2c543c95ec1d0c7afe4670d74e7ad8009a8608daf5a2cce6209540b61"


def close(a, b):
    if isinstance(b, dict):
        assert set(a) == set(b)
        for k in b:
            close(a[k], b[k])
    elif isinstance(b, (int, float)) and not isinstance(b, bool):
        assert math.isfinite(a) and math.isclose(a,b,rel_tol=2e-6,abs_tol=2e-7)
    else:
        assert a == b


def audit(root, parent, out):
    spec=importlib.util.spec_from_file_location("capacity_pinned",root/"scripts/cctv_dgp_capacity_v7.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.sha(root/"capacity_protocol_v7.json")==PROTOCOL_SHA
    p=module.verify(root,weights=False)
    assert module.sha(parent/"targets_protocol_v6.json")==module.PARENT_SHA
    for name,pin in p["cache_assets"].items():
        assert module.sha(parent/name)==pin
    import sys
    sys.path.insert(0,str(root))
    from cctv_dgp_pilot import aggregate,exported_pixel_metrics,state_hash
    from models import DGPSynthesizer
    report=module.read(out/"results.json")
    assert report["complete"] and report["protocol_sha256"]==PROTOCOL_SHA
    assert report["total_optimizer_updates"]==report["training_backward_calls"]==200
    assert report["preflight_autograd_calls"]==2 and 0<report["elapsed_seconds"]<=600
    assert report["buffers_and_teachers_unchanged"] and report["training_only"]
    assert not any(report[k] for k in ["validation_used","native_used","native_reserved_used","production_promoted","model_improvement_established"])
    for name,pin in report["artifacts_sha256"].items():
        path=out/name
        assert path.resolve().is_relative_to(out.resolve()) and module.sha(path)==pin
    execution=module.read(out/"execution.json")
    assert execution["host"].split('.')[0]=="forensic-dgp-thesis" and 'L4' in execution["gpu"]
    assert execution["starting_state_hash"]==module.START_STATE and execution["batch_size"]==2
    assert execution["source_sha256"]==module.sha(root/"scripts/cctv_dgp_capacity_v7.py")==module.sha(out/"executed_source.py")
    preflight=module.read(out/"preflight.json")
    assert preflight["passed"] and preflight["zero_optimizer_updates"] and preflight["autograd_calls"]==2
    assert [a["arm"] for a in preflight["arms"]]==module.ARMS
    assert all(a["optimizer_updates"]==0 and a["active_gradient_tensors"]>0 for a in preflight["arms"])
    traces=[json.loads(line) for line in (out/"updates.jsonl").read_text().splitlines()]
    assert len(traces)==200
    for arm_index,arm in enumerate(module.ARMS):
        for update in range(1,101):
            row=traces[arm_index*100+update-1]
            cycle=(update-1)//5;offset=((update-1)%5)*2
            order=np.random.default_rng(module.SEED+cycle).permutation(10).tolist()
            assert row["arm"]==arm and row["update"]==update
            assert row["cases"]==[p["cases"][i]["id"] for i in order[offset:offset+2]]
            assert row["preclip_norm"]>0 and math.isfinite(row["preclip_norm"])
            c=row["components"]
            loss=c["pixel_mse"] if arm=="pixel_mse" else c["pixel"]+.05*c["color"]+.1*c["vgg"]+.05*c["sobel"]+.1*c["identity"]
            close(row["loss"],loss)
    refs={r["id"]:r for r in p["references"]}
    embeds={ref:np.load(out/"target_embeddings"/(ref+'.npy'),allow_pickle=False) for ref in refs}
    for e in embeds.values():
        assert e.shape==(512,) and np.isfinite(e).all() and np.isclose(np.linalg.norm(e),1.,atol=1e-5)
    metrics={};png_count=0
    for stage in ['baseline']+[f'{arm}_update{n}' for arm in module.ARMS for n in p["snapshots"]]:
        saved=module.read(out/stage/"metrics.json")
        assert len(saved["rows"])==10 and saved["training_only"]
        rebuilt=[]
        for case,row in zip(p["cases"],saved["rows"]):
            for key,value in case.items():
                assert row[key]==value
            ref=refs[case["reference_id"]]
            rgb=np.asarray(Image.open(out/row["prediction"]).convert('RGB'))
            target=np.asarray(Image.open(root/ref["target"]).convert('RGB'))
            support=np.asarray(Image.open(root/ref["observed"]))>0
            input_rgb=np.asarray(Image.open(root/case["input"]).convert('RGB'))
            np.testing.assert_array_equal(rgb[~support],input_rgb[~support])
            values=exported_pixel_metrics(rgb,target,support)
            for key,value in values.items(): close(row[key],value)
            e=np.load(out/row["embedding"],allow_pickle=False)
            assert e.shape==(512,) and np.isfinite(e).all() and np.isclose(np.linalg.norm(e),1.,atol=1e-5)
            cosine=float(np.clip(e@embeds[ref["id"]],-1,1));close(row["ArcFace_observed_fixed"],cosine)
            rebuilt.append({**row,**values,"ArcFace_observed_fixed":cosine});png_count+=1
        close(saved["summary"],aggregate(rebuilt))
        sheet=np.asarray(Image.open(out/saved["preview"]).convert('RGB'))
        for i,row in enumerate(saved["rows"]):
            y=24+i*288+28;ref=refs[row["reference_id"]]
            for j,file in enumerate([root/row["input"],out/row["prediction"],root/ref["target"]]):
                np.testing.assert_array_equal(sheet[y:y+256,j*260+2:j*260+258],np.asarray(Image.open(file).convert('RGB')))
        metrics[stage]=saved["summary"]
    start=torch.load(parent/p["weights"]["start"],map_location='cpu',weights_only=True)
    assert state_hash(start)==module.START_STATE
    model=DGPSynthesizer().eval();model.load_state_dict(start,strict=True)
    buffers={name for name,_ in model.named_buffers()};parameters={name for name,_ in model.named_parameters()}
    changes=[]
    assert [b["arm"] for b in report["branches"]]==module.ARMS
    for branch in report["branches"]:
        arm=branch["arm"]
        assert [s["update"] for s in branch["snapshots"]]==p["snapshots"]
        for snapshot in branch["snapshots"]:
            weights=torch.load(out/snapshot["checkpoint"],map_location='cpu',weights_only=True)
            model.load_state_dict(weights,strict=True)
            assert state_hash(weights)==snapshot["state_hash"]
            assert all(torch.equal(weights[k],start[k]) for k in buffers)
            changed=sum(not torch.equal(weights[k],start[k]) for k in parameters)
            assert changed>0
            changes.append({"arm":arm,"update":snapshot["update"],"changed_parameter_tensors":changed})
        final=metrics[f'{arm}_update100'];base=metrics['baseline']
        ratios={profile:final['dataset/thumbnails128x128/'+profile]['MSE']/base['dataset/thumbnails128x128/'+profile]['MSE'] for profile in p['learnability_groups']}
        close(branch['fixed_final_training_mse_ratios'],ratios)
        assert branch['training_fit_observed']==all(v<=.8 for v in ratios.values())
    assert not list(out.rglob('best.pth'))
    return {"complete":True,"protocol_sha256":PROTOCOL_SHA,"result_sha256":module.sha(out/"results.json"),
        "training_pngs_checked":png_count,"training_embeddings_checked":png_count+10,"update_trace_checked":200,
        "checkpoint_changes":changes,"training_only":True,"local_model_forwards":0,"local_backward_calls":0,
        "local_optimizer_updates":0,"native_reserved_used":False,"production_promoted":False,
        "model_improvement_established":False,"limitation":"Rebuilds training-only arithmetic/states; no recognizer or CUDA gradient replay; fit is not generalization, useful native output or release eligibility"}


def extract(archive,destination):
    digest=__import__('hashlib').sha256(archive.read_bytes()).hexdigest()
    assert Path(str(archive)+'.sha256').read_text().strip().split()==[digest,archive.name]
    assert not destination.exists(),'Preserve existing return'
    with tarfile.open(archive,'r:gz') as stream:
        members=stream.getmembers()
        assert len(members)<3000 and sum(m.size for m in members)<512*1024**2
        for m in members:
            path=Path(m.name)
            assert (m.isfile() or m.isdir()) and not path.is_absolute() and '..' not in path.parts and '\\' not in m.name and ':' not in m.name
        destination.mkdir(parents=True)
        for m in members:
            if m.isfile():
                target=destination/m.name;target.parent.mkdir(parents=True,exist_ok=True)
                with stream.extractfile(m) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--parent-root',type=Path,required=True)
    parser.add_argument('--results',type=Path)
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--extract-to',type=Path)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    if args.archive:
        assert args.extract_to is not None
        extract(args.archive,args.extract_to)
        args.results=args.extract_to/'outputs/cctv_dgp_capacity_v7'
    result=audit(args.root,args.parent_root,args.results)
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    with args.receipt.open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(result)
