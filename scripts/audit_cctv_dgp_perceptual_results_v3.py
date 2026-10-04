"""Audit returned pilot files/cohorts/metrics/checkpoints; no training or backward."""
import argparse
import json
import math
from pathlib import Path, PurePosixPath
import sys
import tarfile

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cctv_dgp_pilot import sha,read,write,verify_bundle,exported_pixel_metrics,aggregate,qualifies,state_hash,grid112,identity_crop,image_tensor

from cctv_dgp_perceptual_training_v3 import (configure_paths, verify_protocol_v3,
    protocol_sha_v3, starting_weights_v3, load_starting_state, START_SHA, START_STATE)
import cctv_dgp_perceptual_training_v3 as recipe_module
from audit_cctv_dgp_normfix_results import save_receipt
verify_bundle = verify_protocol_v3


def require(condition,message):
    if not condition:
        raise ValueError(message)


def same(actual,expected,where="",atol=2e-6):
    if isinstance(expected,dict):
        require(set(actual)==set(expected),"Keys differ: "+where)
        for k,v in expected.items():
            same(actual[k],v,where+"/"+k,atol)
    elif isinstance(expected,list):
        require(len(actual)==len(expected),"Length differs: "+where)
        for i,v in enumerate(expected):
            same(actual[i],v,where+"/"+str(i),atol)
    elif isinstance(expected,float):
        require(isinstance(actual,(float,int)) and math.isfinite(actual) and math.isfinite(expected)
                and abs(actual-expected)<=atol,"Numeric mismatch: "+where)
    else:
        require(actual==expected,"Value differs: "+where)


def safe_result_path(out,name):
    path=PurePosixPath(name)
    require(not path.is_absolute() and '..' not in path.parts and '\\' not in name and ':' not in name,"Unsafe result path")
    result=out/name
    require(result.resolve().is_relative_to(out.resolve()),"Escaped result path")
    return result


def extract_return(archive,dest):
    require(not dest.exists(),"Preserve existing extracted/partial return")
    raw=Path(str(archive)+".sha256").read_bytes().decode('ascii')
    parts=raw.strip().split()
    require(len(parts)==2 and parts[0].lower()==sha(archive) and parts[1].lstrip('*')==archive.name,"Transfer checksum differs")
    allowed={"perceptual_protocol_v3.json","perceptual_protocol_v3.sha256","environment_perceptual_v3.txt","cuda_runtime_before.txt","pilot_perceptual_v3.log"}
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();names=set();total=0
        for m in members:
            name=m.name.rstrip('/')
            safe_result_path(dest,name)
            require(not m.issym() and not m.islnk() and (m.isfile() or m.isdir()),"Linked/special archive member")
            require(name in allowed or name in ('outputs','outputs/cctv_dgp_perceptual_v3') or name.startswith('outputs/cctv_dgp_perceptual_v3/'),"Unexpected returned root")
            require(name not in names,"Duplicate archive member");names.add(name);total+=m.size
        require(total<2*1024**3 and len(members)<12000,"Finite return archive bounds exceeded")
        dest.mkdir(parents=True)
        for m in members:
            target=safe_result_path(dest,m.name.rstrip('/'))
            if m.isdir():
                target.mkdir(parents=True,exist_ok=True)
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(m) as stream,target.open('xb') as output:
                    import shutil
                    shutil.copyfileobj(stream,output)
    return dest/"outputs/cctv_dgp_perceptual_v3"


def unit_embedding(path):
    e=np.load(path,allow_pickle=False)
    require(e.shape==(512,) and np.isfinite(e).all() and abs(float(np.linalg.norm(e))-1)<2e-5,"Invalid normalized recognizer embedding")
    return e


def audit(root,out,verify_recognizer=False):
    root,out=Path(root),Path(out);protocol=verify_bundle(root);report=read(out/"results.json")
    require(report["complete"] and not (out/"failure.json").exists(),"Incomplete/failed return: audit partial evidence separately")
    require(report["protocol_sha256"]==protocol_sha_v3(root),"Returned protocol binding differs")
    require(not report["production_checkpoint_promoted"] and not report["native_reserved_used"] and not report["goal_complete"],"Unauthorized promotion/evaluation claim")
    for name,pin in report["artifacts_sha256"].items():
        require(sha(safe_result_path(out,name))==pin,"Changed returned asset: "+name)
    expected_files=set(report["artifacts_sha256"])
    actual={p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()}
    require(actual-{"results.json","independent_audit.json"}==expected_files,"Unexpected/incomplete returned inventory")
    preflight=read(out/"preflight.json");execution=read(out/"execution.json")
    require(preflight["passed"] and preflight["optimizer_updates"]==0 and not preflight["optimizer_constructed"]
            and preflight["batch_size"]==8 and preflight["model_state_unchanged"] and preflight["identity_input_gradient_mean"]>0,"CUDA preflight record differs")
    require('L4' in preflight["gpu"] and execution["device"]=='cuda' and execution["host"].split('.')[0]=='forensic-dgp-thesis',"VM execution record differs")
    require(execution["protocol_sha256"]==report["protocol_sha256"]==preflight["protocol_sha256"],"Protocol lineage differs")
    require(execution["normalization_running_stats_frozen"] and not execution["ema"] and not execution["amp"],"Optimization policy differs")
    require(report["elapsed_seconds"]<=5400 and report["total_optimizer_updates"]==protocol["expected_total_updates"]
            and report["teacher_states_unchanged"] and report["normalization_buffers_unchanged"],"Finite runtime/update/state record differs")
    require(execution["teacher_states"]==preflight["teacher_state_before"],"Teacher starting fingerprints differ")
    require(preflight['zero_update_backward_calls']==2 and preflight['six_image_eval_state_unchanged']
            and preflight['six_image_eval_state_hash']==START_STATE
            and preflight['starting_checkpoint_sha256']==execution['starting_checkpoint_sha256']==START_SHA,
            'V3 starting/zero-update/tail preflight differs')
    require(preflight['calibration_sha256']==execution['perceptual_calibration_sha256']==sha(recipe_module.BUNDLE_DIR/'perceptual_scales_v3.json'),
            'V3 fixed calibration binding differs')
    for policy in ('postactivation','preactivation'):
        check=preflight['feature_policy_checks'][policy]
        require(check['passed'] and check['feature_policy']==policy and check['optimizer_updates']==0
                and check['model_state_hash']==START_STATE and check['model_state_unchanged']
                and math.isfinite(check['perceptual_input_gradient_mean']) and check['perceptual_input_gradient_mean']>0
                and math.isfinite(check['identity_input_gradient_mean']) and check['identity_input_gradient_mean']>0
                and check['teacher_state_before']==execution['teacher_states'] and not check['optimizer_constructed'],
                'V3 feature/identity gradient preflight differs')
        require(read(out/'preflight_checks'/policy/'preflight.json')==check,'V3 preflight source report differs')
    refs={r["id"]:r for r in protocol["references"]};case_map={c["id"]:c for c in protocol["validation_cases"]}
    reference={r:unit_embedding(out/"reference_embeddings"/(r+".npy")) for r in refs}
    require(report["reference_embedding_files"]==len(refs),"Reference embedding coverage differs")
    targets={r:np.asarray(Image.open(root/ref["target"]).convert('RGB')) for r,ref in refs.items() if ref["role"]=='validation'}
    masks={r:np.asarray(Image.open(root/ref["observed"]))>0 for r,ref in refs.items() if ref["role"]=='validation'}
    original=load_starting_state(root)
    require(state_hash(original)==execution["initial_model_state"]==preflight["model_state_hash"],"Starting checkpoint lineage differs")
    stages=['baseline']+[f"{a['id']}_epoch{e}" for a in protocol['arms'] for e in (1,2)]
    summaries={};png_count=0;float_count=0;cosine_count=0;recognizer_forwards=0
    recognizer=None
    if verify_recognizer:
        import onnxruntime as ort
        torch.set_num_threads(4)
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        recognizer=ort.InferenceSession(str(root/protocol['weights']['arcface']),sess_options=options,providers=['CPUExecutionProvider'])
    def check_recognizer(rgb,refid,expected):
        nonlocal recognizer_forwards
        if recognizer is None:
            return
        mask=torch.from_numpy(masks[refid].copy()).float()[None,None]
        grid=torch.from_numpy(grid112(refs[refid]['matrix112']))[None]
        with torch.no_grad():
            crop=(identity_crop(image_tensor(rgb)[None],mask,grid)*2-1).numpy()
        e=recognizer.run(None,{recognizer.get_inputs()[0].name:crop})[0][0];e=e/np.linalg.norm(e)
        np.testing.assert_allclose(e,expected,rtol=1e-3,atol=3e-4);recognizer_forwards+=1
    preview_refs={case_map[c]['reference_id'] for c in protocol['preview_case_ids']}
    for r in preview_refs:
        check_recognizer(targets[r],r,reference[r])
    for stage in stages:
        metrics=read(out/stage/"metrics.json");require(metrics['complete'],"Incomplete validation stage")
        require(len(metrics['rows'])==len(case_map) and [r['id'] for r in metrics['rows']]==list(case_map),"Fixed validation cohort/order differs")
        rebuilt=[];input_rows=[]
        for row in metrics['rows']:
            case=case_map[row['id']];refid=case['reference_id'];mask=masks[refid];target=targets[refid]
            for k,v in case.items():
                same(row[k],v,'case/'+k)
            rgb=np.asarray(Image.open(safe_result_path(out,row['prediction'])).convert('RGB'))
            low=np.asarray(Image.open(root/case['input']).convert('RGB'))
            require(rgb.shape==(256,256,3) and np.array_equal(rgb[~mask],low[~mask]),"Prediction altered uncaptured context")
            pixel=exported_pixel_metrics(rgb,target,mask)
            # Independent direct masked pixel calculation as well as the SSIM implementation.
            error=rgb[mask].astype(np.float32)/255-target[mask].astype(np.float32)/255
            same(pixel['MSE'],float((error.astype(np.float64)**2).mean()),'direct MSE',2e-9)
            for k,v in pixel.items():
                same(row[k],v,'pixel/'+k)
            emb=unit_embedding(safe_result_path(out,row['embedding']));cosine=float(np.clip(emb@reference[refid],-1,1))
            same(row['ArcFace_observed_fixed'],cosine,'cosine',2e-6);cosine_count+=1
            rebuilt.append({**case,**pixel,'ArcFace_observed_fixed':cosine});png_count+=1
            if row['id'] in protocol['preview_case_ids']:
                floats=np.load(out/stage/'float_preview'/(row['id']+'.npz'),allow_pickle=False)
                network,observed=floats['network_rgb'],floats['observed_rgb']
                require(network.shape==observed.shape==(256,256,3) and np.isfinite(network).all() and np.isfinite(observed).all()
                        and network.min()>=0 and network.max()<=1,"Invalid raw preview")
                np.testing.assert_array_equal(observed,network*mask[...,None]+(low.astype(np.float32)/255)*(~mask[...,None]))
                np.testing.assert_array_equal(rgb,np.floor(observed*255).clip(0,255).astype(np.uint8));float_count+=1
                check_recognizer(rgb,refid,emb)
            if stage=='baseline':
                raw=metrics['input_rows'][len(input_rows)]
                same({k:raw[k] for k in case},case,'input case')
                ip=exported_pixel_metrics(low,target,mask);ie=unit_embedding(safe_result_path(out,raw['embedding']))
                ic=float(np.clip(ie@reference[refid],-1,1))
                for k,v in {**ip,'ArcFace_observed_fixed':ic}.items():
                    same(raw[k],v,'input metrics/'+k)
                input_rows.append({**case,**ip,'ArcFace_observed_fixed':ic});cosine_count+=1
        summaries[stage]=aggregate(rebuilt);same(metrics['summary'],summaries[stage],'summary')
        require(metrics['dgp_batch_forwards']==math.ceil(len(case_map)/8),"Validation forward count differs")
        if stage=='baseline':
            require(len(metrics['input_rows'])==len(case_map),"Input baseline coverage differs")
            same(metrics['input_summary'],aggregate(input_rows),'input summary')
        else:
            require(metrics['input_rows']==[] and metrics['input_summary'] is None,"Unexpected input metric stage")
        for name,pin in metrics['artifact_sha256'].items():
            require(report['artifacts_sha256'].get(name)==pin,"Stage/global artifact binding differs")
        with Image.open(safe_result_path(out,metrics['preview'])) as grid:
            require(grid.size==(492,1964),"Ten-row preview dimensions differ")
    updates=[json.loads(line) for line in (out/'updates.jsonl').read_text().splitlines()]
    require(len(updates)==protocol['expected_total_updates'],"Trace count differs")
    require(all(0<=t['elapsed_total_seconds']<=report['elapsed_seconds'] for t in updates)
            and all(a['elapsed_total_seconds']<=b['elapsed_total_seconds'] for a,b in zip(updates,updates[1:])),"Trace timing differs")
    position=0;changed=[];selections=[]
    require([b['arm'] for b in report['branches']]==protocol['arms'],"Branch coverage/order differs")
    for branch,arm in zip(report['branches'],protocol['arms']):
        aid=arm['id'];baseline_weights=torch.load(out/aid/'baseline.pth',map_location='cpu',weights_only=True)
        require(state_hash(baseline_weights)==state_hash(original),"Branches started at different weights")
        best=summaries['baseline'];best_epoch=0;best_path=f"{aid}/baseline.pth";arm_updates=0
        require([e['epoch'] for e in branch['epochs']]==[1,2] and branch['optimizer_updates']==protocol['updates_per_arm'],"Branch epoch/update counts differ")
        for record in branch['epochs']:
            epoch=record['epoch'];cases=protocol['training_epochs'][str(epoch)]
            for offset in range(0,len(cases),8):
                trace=updates[position];position+=1;arm_updates+=1
                require(trace['feature_policy']==arm['feature_policy'] and trace['arm']==aid and trace['epoch']==epoch and trace['update']==arm_updates
                        and trace['cases']==[c['id'] for c in cases[offset:offset+8]],"Matched update trace differs")
                require(math.isfinite(trace['loss']) and math.isfinite(trace['identity_loss']) and all(math.isfinite(x) for x in trace['components'].values()),"Nonfinite trace")
                expected_loss=trace['components']['pixel']+.05*trace['components']['color']+.1*trace['components']['vgg']+.05*trace['components']['sobel']+arm['lambda_identity']*trace['identity_loss']
                same(trace['loss'],expected_loss,'loss weights',3e-6)
                if not arm['lambda_identity']:
                    require(trace['identity_loss']==0.,"No-identity arm used identity term")
            require(record['updates']==math.ceil(len(cases)/8) and record['cumulative_updates']==arm_updates,"Epoch trace count differs")
            stage=f"{aid}_epoch{epoch}";summary=summaries[stage]
            accepted=qualifies(summary,summaries['baseline'],best)
            require(record['accepted']==accepted and read(out/aid/f'epoch_{epoch}_record.json')==record,"Selection/epoch record differs")
            path=out/record['checkpoint'];require(sha(path)==record['checkpoint_sha256'],"Checkpoint hash differs")
            state=torch.load(path,map_location='cpu',weights_only=True)
            require(set(state)==set(original) and state_hash(state)==record['state_hash'],"Student tensor schema/digest differs")
            different=0
            for name,tensor in original.items():
                value=state[name];require(value.shape==tensor.shape and value.dtype==tensor.dtype and torch.isfinite(value).all(),"Invalid student tensor")
                if name.endswith(('running_mean','running_var','num_batches_tracked')):
                    require(torch.equal(value,tensor),"Frozen running statistic changed")
                elif not torch.equal(value,tensor):
                    different+=1
            require(different>0 and record['buffers_hash']==execution['initial_buffers_hash'],"No trained change/frozen buffer claim differs")
            changed.append({'arm':aid,'epoch':epoch,'changed_parameter_tensors':different})
            if accepted:
                best=summary;best_epoch=epoch;best_path=record['checkpoint']
        selection=read(out/aid/'best_selection.json');same(selection,branch['selection'],'selection')
        require(selection['selected_epoch']==best_epoch and selection['selected_source']==best_path
                and sha(out/aid/'best.pth')==sha(out/best_path)==selection['best_sha256'],"Best/baseline fallback differs")
        selections.append(selection)
    return {'complete':True,'protocol_sha256':protocol_sha_v3(root),'returned_results_sha256':sha(out/'results.json'),
            'png_predictions_checked':png_count,'raw_float_previews_checked':float_count,'embedding_cosines_rebuilt':cosine_count,
            'update_trace_checked':position,'checkpoint_changes':changed,'selections':selections,
            'recognizer_preview_forwards':recognizer_forwards,'local_restoration_forwards':0,'local_optimizer_updates':0,
            'production_promoted':False,'native_visual_review_pending':True,'independent_final_review_pending':True,
            'perceptual_pilot_version':3,'calibration_sha256':sha(recipe_module.BUNDLE_DIR/'perceptual_scales_v3.json')}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'outputs/cctv_dgp_vm_bundle_v1')
    parser.add_argument('--results',type=Path)
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--extract-to',type=Path)
    parser.add_argument('--verify-recognizer',action='store_true')
    parser.add_argument('--bundle-dir',type=Path,default=ROOT)
    parser.add_argument('--parent-return',type=Path)
    parser.add_argument('--receipt',type=Path)
    args=parser.parse_args()
    configure_paths(args.bundle_dir,args.parent_return)
    if args.archive:
        if not args.extract_to:
            parser.error('--archive requires a fresh --extract-to directory')
        out=extract_return(args.archive,args.extract_to)
        require(sha(args.extract_to/'perceptual_protocol_v3.json')==protocol_sha_v3(args.root),'Transferred V3 protocol differs')
    else:
        out=args.results
    if out is None:
        parser.error('Supply --results or --archive')
    report=audit(args.root,out,args.verify_recognizer)
    save_receipt(out,report,args.receipt)
    print(report)
