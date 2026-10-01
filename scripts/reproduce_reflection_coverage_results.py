"""Reproduce every reflection-pilot mask on CPU; no optimizer or fitting."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_reflection_coverage_results import (
    ARCHIVE,EXTRACTION,AUDIT_OUTPUT,PREFIX,ARMS,SIZES,MODEL_SHA,OPTIMIZER_SHA,DATA_PATH,PIXELS_SHA,
    verify_package,validate_run,checkpoint_audit,audit_final_optimizer,recount_supported,
)
from scripts.audit_face_occlusion_results import canonical_member,read_json,read_lines,require,verified_inputs
from scripts.package_face_occlusion_vm import sha
from scripts.train_coverage_vm import real_gate,PARENT_SHA
from scripts.evaluate_coverage_results import binary
from scripts.reproduce_face_occlusion_focus_results import infer_cpu
from detector_replay import retention_passes
from face_occlusion_focus import validate_source,validate_optimizer


def verify_audit_binding(audit,archive_sha,checker_sha):
    require(audit.get('archive_sha256')==archive_sha and audit.get('script_sha256')==checker_sha and
            audit.get('checksum_verified') is True and type(audit.get('saved_masks_recounted')) is int and
            audit['saved_masks_recounted']==4315 and
            audit.get('log_audit',{}).get('total_experiment_updates')==504 and audit.get('promoted') is False,
            'Require complete hash-bound archive/log/mask audit before CPU reproduction')


def cpu_selection(reproductions,histories):
    baseline=reproductions['parent']['scores'];choices={};agreed=True
    require(set(histories)==set(ARMS),'Missing matched CPU arm')
    for arm in ARMS:
        best=baseline['real']['iou'];choices[arm]=[]
        for row in histories[arm]:
            scores=reproductions[f'{arm}/{row["epoch"]}']['scores']
            real=real_gate(scores['real'],baseline['real'],best)
            synthetic=retention_passes(scores['synthetic'],baseline['synthetic']);selected=real and synthetic
            choices[arm].append({'epoch':row['epoch'],'real_gate':real,'synthetic_gate':synthetic,'selected':selected})
            agreed &= (real,synthetic,selected)==(row['real_gate'],row['synthetic_gate'],row['selected'])
            if selected:best=scores['real']['iou']
    return choices,agreed


def infer_fixture_cpu(model,pixels,folder,*,progress=True,label='fixture'):
    require(pixels and all(v.device.type=='cpu' for v in model.state_dict().values()),'CPU model/fixtures required')
    require({p.name for p in folder.glob('*.png')}=={f'{i:04}.png' for i in pixels},'Saved fixture mask membership differs')
    model.requires_grad_(False).eval()
    before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    predictions={};nonexact=[]
    with torch.inference_mode():
        for n,i in enumerate(sorted(pixels),1):
            case=pixels[i];image=case['input']
            require(isinstance(image,torch.Tensor) and image.dtype==torch.uint8 and image.ndim==3 and image.shape[0]==3,
                    'Fixture input must be byte RGB CHW')
            logits=model.detect(image[None].float()/255)
            require(isinstance(logits,torch.Tensor) and logits.shape==(1,1,*image.shape[1:]) and
                    logits.device.type=='cpu' and torch.isfinite(logits).all(),'Invalid CPU fixture logits')
            prediction=(logits.sigmoid()[0,0]>=.5).numpy();saved=binary(folder/f'{i:04}.png')
            require(saved.shape==prediction.shape,'Saved/CPU fixture mask shape differs')
            difference=int(np.count_nonzero(prediction!=saved))
            if difference:nonexact.append({'index':i,'different_pixels':difference})
            predictions[i]=prediction
            if progress and n%100==0:print(label,'training_reflection',n,len(pixels),flush=True)
    require(all(torch.equal(v.cpu(),before[k]) for k,v in model.state_dict().items()) and
            all(p.grad is None for p in model.parameters()),'Read-only fixture inference changed model state/gradients')
    scores=recount_supported(predictions,pixels)
    return {'scores':scores,'mask_reproduction':{'cases':len(pixels),'all_exact':not nonexact,
                'different_pixels':sum(r['different_pixels'] for r in nonexact),'nonexact_cases':nonexact},
            'model_state_unchanged':True,'optimizer_constructed':False,'optimizer_updates_locally':0}


def main():
    from completion_inference import load_completion
    from face_occlusion_adapter import load_adapter,parameter_groups
    from scripts.train_face_occlusion_vm import dependency_versions
    require((AUDIT_OUTPUT/'results.json').is_file(),'Run complete archive/log/mask audit first')
    destination=AUDIT_OUTPUT/'reproduction.json';partial=AUDIT_OUTPUT/'reproduction_partial.json'
    require(not destination.exists() and not partial.exists(),'Preserve completed or partial reproduction evidence')
    audit=read_json(AUDIT_OUTPUT/'results.json')
    verify_audit_binding(audit,sha(ARCHIVE),sha(ROOT/'scripts/audit_reflection_coverage_results.py'))
    inventory=verify_package();_,rows,datasets=verified_inputs();hashes=read_json(AUDIT_OUTPUT/'members.json')
    for name,digest in hashes.items():
        canonical_member(name);require(sha(EXTRACTION/name)==digest,'Extracted evidence changed: '+name)
    returned=EXTRACTION/PREFIX;run=read_json(returned/'run.json');validate_run(run,inventory)
    baseline=read_json(returned/'baseline.json');source=read_json(returned/'source_metrics.json')
    histories={a:read_lines(returned/a/'metrics.jsonl') for a in ARMS}
    pixel_path=ROOT/DATA_PATH/'pixels.pth';require(sha(pixel_path)==PIXELS_SHA,'Frozen fixture pixels changed')
    pixels=torch.load(pixel_path,map_location='cpu',weights_only=True)['pixels']
    sys.path.insert(0,str(ROOT/'outputs/face_extraction_dependencies'));torch.set_num_threads(4)
    report={'archive_sha256':audit['archive_sha256'],'script_sha256':sha(__file__),
            'audit_sha256':sha(AUDIT_OUTPUT/'results.json'),'fixture_cache_sha256':PIXELS_SHA,
            'device':'cpu','torch':str(torch.__version__),
            'dependencies':dependency_versions(ROOT/'outputs/face_extraction_dependencies'),
            'optimizer_updates_locally':0,'optimizer_constructed':False,'promoted':False,
            'checkpoint_checks':{},'final_optimizers':{},'reproductions':{},'fixture_reproductions':{},
            'limitation':'CPU state invariance is checked. Remote original-parent invariance remains an executed-code/log claim; training fixtures do not establish real transfer.'}
    parent_path=ROOT/'outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth'
    require(sha(parent_path)==PARENT_SHA,'Original completion parent changed')
    parent,_=load_completion(parent_path,'cpu')
    report['reproductions']['parent']=infer_cpu(parent,datasets,rows,returned/'baseline_masks',baseline,
                                              ('real','synthetic'),'parent')
    del parent
    require(sha(returned/'initial.pth')==MODEL_SHA and sha(returned/'initial_optimizer.pth')==OPTIMIZER_SHA,
            'Copied source model/moments changed')
    model,initial=load_adapter(returned/'initial.pth','cpu');validate_source(initial)
    source_optimizer=torch.load(returned/'initial_optimizer.pth',map_location='cpu',weights_only=True)
    report['source_optimizer']=validate_optimizer(source_optimizer,parameter_groups(model),MODEL_SHA)
    report['reproductions']['source30']=infer_cpu(model,datasets,rows,returned/'initial_masks',source,SIZES,'source30')
    report['fixture_reproductions']['source30']=infer_fixture_cpu(model,pixels,returned/'initial_masks'/'training_reflection',label='source30')
    del model
    for arm in ARMS:
        for row in histories[arm]:
            epoch=row['epoch'];key=f'{arm}/{epoch}';path=returned/arm/f'epoch_{epoch}.pth'
            model,payload=load_adapter(path,'cpu')
            report['checkpoint_checks'][key]={'sha256':sha(path),**checkpoint_audit(payload,initial,run,row)}
            if epoch==42:
                moments=torch.load(returned/arm/'final_optimizer.pth',map_location='cpu',weights_only=True)
                report['final_optimizers'][arm]=audit_final_optimizer(moments,parameter_groups(model),sha(path),source_optimizer)
                del moments
            folder=returned/arm/f'epoch_{epoch}_masks'
            report['reproductions'][key]=infer_cpu(model,datasets,rows,folder,row,SIZES,key)
            report['fixture_reproductions'][key]=infer_fixture_cpu(model,pixels,folder/'training_reflection',label=key)
            del model,payload
            partial.write_text(json.dumps(report,indent=2)+'\n')
        selected=audit['log_audit']['arms'][arm]['selected_epochs']
        if selected:
            epoch=selected[-1];row=next(r for r in histories[arm] if r['epoch']==epoch)
            best=torch.load(returned/arm/'best_detector.pth',map_location='cpu',weights_only=True)
            checkpoint_audit(best,initial,run,row)
            require(sha(returned/arm/'best_detector.pth')==sha(returned/arm/f'epoch_{epoch}.pth'),
                    'Best bytes differ from eligible candidate')
            report.setdefault('best_detectors',{})[arm]={'epoch':epoch,'state_matches_candidate':True}
            del best
    report['cpu_selection'],report['selection_agrees_with_vm']=cpu_selection(report['reproductions'],histories)
    comparisons=[d for r in report['reproductions'].values() for d in r['mask_reproduction'].values()]
    comparisons.extend(r['mask_reproduction'] for r in report['fixture_reproductions'].values())
    report.update(saved_masks_compared=sum(d['cases'] for d in comparisons),
                  all_saved_masks_exact=all(d['all_exact'] for d in comparisons),
                  different_pixels=sum(d['different_pixels'] for d in comparisons),complete=True)
    require(report['saved_masks_compared']==4315,'Incomplete checkpoint-to-mask reproduction')
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('saved_masks_compared','all_saved_masks_exact','different_pixels',
                                          'selection_agrees_with_vm','optimizer_updates_locally','promoted')}),flush=True)


if __name__=='__main__':main()
