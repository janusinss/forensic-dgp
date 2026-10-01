"""Reproduce every returned matched-pilot mask on CPU; no optimizer constructed."""
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_face_occlusion_focus_results import (
    ARCHIVE,EXTRACTION,AUDIT_OUTPUT,PREFIX,ARMS,SIZES,EPOCHS,MODEL_SHA,OPTIMIZER_SHA,
    verify_package,validate_run,checkpoint_audit,audit_final_optimizer,
)
from scripts.audit_face_occlusion_results import canonical_member,read_json,read_lines,require,subsets,verified_inputs
from scripts.package_face_occlusion_vm import sha
from scripts.train_coverage_vm import real_gate,PARENT_SHA
from scripts.evaluate_coverage_results import binary,recount
from detector_replay import retention_passes
from face_occlusion_focus import validate_source,validate_optimizer


def cpu_selection(reproductions,histories):
    baseline = reproductions['parent']['scores']; choices = {}; agreed = True
    for arm in ARMS:
        best = baseline['real']['iou']; choices[arm] = []
        for row in histories[arm]:
            scores = reproductions[f'{arm}/{row["epoch"]}']['scores']
            real = real_gate(scores['real'],baseline['real'],best)
            synthetic = retention_passes(scores['synthetic'],baseline['synthetic']); selected = real and synthetic
            choices[arm].append({'epoch':row['epoch'],'real_gate':real,'synthetic_gate':synthetic,'selected':selected})
            agreed &= (real,synthetic,selected) == (row['real_gate'],row['synthetic_gate'],row['selected'])
            if selected: best = scores['real']['iou']
    return choices,agreed


def infer_cpu(model,data,rows,folder,logged,saved_domains,label,*,progress=True):
    model.requires_grad_(False).eval()
    before = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    scores = {}; differences = {}; real_predictions = real_truth = None
    with torch.inference_mode():
        for domain,dataset in data.items():
            predictions = []; truth = []; nonexact = []
            for i,(x,m) in enumerate(dataset):
                prediction = (model.detect(x[None]).sigmoid()[0,0] >= .5).numpy()
                target = m[0].numpy().astype(bool); predictions.append(prediction); truth.append(target)
                if domain in saved_domains:
                    saved = binary(folder/domain/f'{i:04}.png')
                    require(saved.shape == prediction.shape, 'Saved/CPU mask shape differs')
                    pixels = int(np.count_nonzero(prediction != saved))
                    if pixels: nonexact.append({'index':i,'different_pixels':pixels})
                if progress and (i+1)%100 == 0: print(label,domain,i+1,len(dataset),flush=True)
            scores[domain] = recount(predictions,truth)
            if domain in saved_domains:
                differences[domain] = {'cases':len(dataset),'all_exact':not nonexact,
                    'different_pixels':sum(r['different_pixels'] for r in nonexact),'nonexact_cases':nonexact}
            if domain == 'real': real_predictions,real_truth = predictions,truth
    subsets(scores,real_predictions,real_truth,rows)
    require(all(torch.equal(v.cpu(),before[k]) for k,v in model.state_dict().items()), 'Read-only inference changed model state')
    deltas = {d:{k:float(v-logged[d][k]) for k,v in metrics.items()} for d,metrics in scores.items()}
    return {'scores':scores,'mask_reproduction':differences,'metric_deltas_cpu_minus_logged':deltas,
            'model_state_unchanged':True}


def main():
    from completion_inference import load_completion
    from face_occlusion_adapter import load_adapter,parameter_groups
    from scripts.train_face_occlusion_vm import dependency_versions
    require((AUDIT_OUTPUT/'results.json').is_file(), 'Run archive/log/mask audit first')
    destination = AUDIT_OUTPUT/'reproduction.json'; require(not destination.exists(), 'Preserve completed reproduction')
    audit = read_json(AUDIT_OUTPUT/'results.json')
    require(sha(ARCHIVE) == audit['archive_sha256'] and
            sha(ROOT/'scripts/audit_face_occlusion_focus_results.py') == audit['script_sha256'],
            'Archive or original checker changed')
    inventory = verify_package(); _,rows,data = verified_inputs(); hashes = read_json(AUDIT_OUTPUT/'members.json')
    for name,digest in hashes.items():
        canonical_member(name); require(sha(EXTRACTION/name) == digest, f'Extracted evidence changed: {name}')
    returned = EXTRACTION/PREFIX; run = read_json(returned/'run.json'); validate_run(run,inventory)
    baseline = read_json(returned/'baseline.json'); source = read_json(returned/'source_metrics.json')
    histories = {a:read_lines(returned/a/'metrics.jsonl') for a in ARMS}
    sys.path.insert(0,str(ROOT/'outputs/face_extraction_dependencies')); torch.set_num_threads(4)
    report = {'archive_sha256':audit['archive_sha256'],'script_sha256':sha(__file__),
              'audit_sha256':sha(AUDIT_OUTPUT/'results.json'),'device':'cpu','torch':str(torch.__version__),
              'dependencies':dependency_versions(ROOT/'outputs/face_extraction_dependencies'),
              'optimizer_updates_locally':0,'optimizer_constructed':False,'promoted':False,
              'checkpoint_checks':{},'final_optimizers':{},'reproductions':{},
              'limitation':'Local original-parent invariance is verified; remote invariance remains an executed-code/log claim'}
    parent_path = ROOT/'outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth'
    require(sha(parent_path) == PARENT_SHA, 'Original completion parent changed')
    parent,_ = load_completion(parent_path,'cpu')
    report['reproductions']['parent'] = infer_cpu(parent,data,rows,returned/'baseline_masks',baseline,('real','synthetic'),'parent')
    del parent
    require(sha(returned/'initial.pth') == MODEL_SHA and sha(returned/'initial_optimizer.pth') == OPTIMIZER_SHA,
            'Copied source bytes changed')
    model,initial = load_adapter(returned/'initial.pth','cpu'); validate_source(initial)
    source_optimizer = torch.load(returned/'initial_optimizer.pth',map_location='cpu',weights_only=True)
    report['source_optimizer'] = validate_optimizer(source_optimizer,parameter_groups(model),MODEL_SHA)
    report['reproductions']['source30'] = infer_cpu(model,data,rows,returned/'initial_masks',source,SIZES,'source30')
    del model
    for arm in ARMS:
        for row in histories[arm]:
            e = row['epoch']; key = f'{arm}/{e}'; path = returned/arm/f'epoch_{e}.pth'
            model,payload = load_adapter(path,'cpu')
            report['checkpoint_checks'][key] = {'sha256':sha(path),**checkpoint_audit(payload,initial,run,row)}
            if e == 40:
                optimizer = torch.load(returned/arm/'final_optimizer.pth',map_location='cpu',weights_only=True)
                report['final_optimizers'][arm] = audit_final_optimizer(optimizer,parameter_groups(model),sha(path),source_optimizer)
                del optimizer
            report['reproductions'][key] = infer_cpu(model,data,rows,returned/arm/f'epoch_{e}_masks',row,SIZES,key)
            del model,payload
            (AUDIT_OUTPUT/'reproduction_partial.json').write_text(json.dumps(report,indent=2)+'\n')
        selected = audit['log_audit']['arms'][arm]['selected_epochs']
        if selected:
            e = selected[-1]; row = next(r for r in histories[arm] if r['epoch'] == e)
            best = torch.load(returned/arm/'best_detector.pth',map_location='cpu',weights_only=True)
            candidate = torch.load(returned/arm/f'epoch_{e}.pth',map_location='cpu',weights_only=True)
            checkpoint_audit(best,initial,run,row)
            require(all(torch.equal(v,candidate['model'][k]) for k,v in best['model'].items()), 'Best state differs from eligible candidate')
            report.setdefault('best_detectors',{})[arm] = {'epoch':e,'state_matches_candidate':True}
            del best,candidate
    report['cpu_selection'],report['selection_agrees_with_vm'] = cpu_selection(report['reproductions'],histories)
    compared = [d for r in report['reproductions'].values() for d in r['mask_reproduction'].values()]
    report.update(saved_masks_compared=sum(d['cases'] for d in compared),
                  all_saved_masks_exact=all(d['all_exact'] for d in compared),
                  different_pixels=sum(d['different_pixels'] for d in compared),complete=True)
    require(report['saved_masks_compared'] == 2915, 'Incomplete checkpoint-to-mask reproduction')
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('saved_masks_compared','all_saved_masks_exact','different_pixels',
                                          'selection_agrees_with_vm','optimizer_updates_locally','promoted')}),flush=True)


if __name__ == '__main__': main()
