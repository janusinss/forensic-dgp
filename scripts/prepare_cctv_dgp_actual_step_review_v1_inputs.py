"""Freeze existing TRAIN data and saved arithmetic proposals; no neural calls."""
from pathlib import Path
import hashlib
import shutil
import time
from datetime import datetime, timezone
import numpy as np
from cctv_dgp_actual_step_review_v1_contract import NAME, FORMAT, UPDATES, PROPOSALS, BUDGETS, read, write, sha, check_proposal

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/cctv_dgp_pcgrad_fit_vm_v41'
RETURN=ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_return'
OUT=ROOT/'outputs'/NAME


def main():
    start=time.monotonic(); assert not OUT.exists(), 'Retain any earlier preparation; no overwrite'
    milestone_path=ROOT/'outputs/cctv_dgp_v41_preservation_trace_milestone/milestone.json'
    milestone=read(milestone_path);closure=read(milestone_path.with_name('independent_closure_audit.json'))
    assert milestone['complete'] and closure['complete'] and closure['milestone_sha256']==sha(milestone_path)
    p=read(SOURCE/'protocol.json');analysis_path=ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_analysis/analysis.json'
    a=read(analysis_path);audit=read(ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_analysis/independent_analysis_audit.json')
    assert a['complete'] and audit['complete'] and a['protocol_sha256']==sha(SOURCE/'protocol.json')
    imported=read(ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_return_import.json')
    assert imported['complete'] and imported['archive_sha256']=='01a6e9198e731d81895c0b64cde21f45ed7eb01be2bb48b4c78a9186726c94dc'
    assert not imported['returned_code_executed']
    cohorts=a['cohorts'];case_ids=set(sum(cohorts.values(),[]));probes=[]
    for update in UPDATES:
        record=read(RETURN/f'outputs/steps/step{update:04d}.json');case_ids.update(record['case_ids'])
    cases=[c for c in p['cases'] if c['id'] in case_ids];refs={c['reference_id'] for c in cases}
    references=[r for r in p['references'] if r['id'] in refs]
    assert len(cases)==145 and len(references)==29 and all(c['role']=='train' for c in cases)
    selected={n for n in p['assets_sha256'] if not n.startswith('data/')}
    for c in cases:selected.update([c['input'],c['target'],c['observed']])
    # Source/native/reduced paths in reference metadata are provenance links,
    # not runtime assets in V41. Retain that metadata without inventing copies.
    OUT.mkdir();bindings={};copied={}
    for name in sorted(selected):
        assert name in p['assets_sha256'] and sha(SOURCE/name)==p['assets_sha256'][name],name
        destination=OUT/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(SOURCE/name,destination)
        assert sha(destination)==p['assets_sha256'][name]
        copied[name]=p['assets_sha256'][name]
        bindings[(SOURCE/name).relative_to(ROOT).as_posix()]=p['assets_sha256'][name]
        assert time.monotonic()-start<300
    for update in UPDATES:
        original=RETURN/f'outputs/steps/step{update:04d}.npz';record_path=original.with_suffix('.json')
        for path in [original,record_path]:assert sha(path)==imported['files_sha256'][path.relative_to(RETURN).as_posix()]
        destination=OUT/f'evidence/steps/step{update:04d}.npz';destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(original,destination);shutil.copyfile(record_path,destination.with_suffix('.json'))
        bindings[original.relative_to(ROOT).as_posix()]=sha(original);bindings[record_path.relative_to(ROOT).as_posix()]=sha(record_path)
        row=a['step_rows'][update-1];assert row['update']==update
        with np.load(original,allow_pickle=False) as arrays:
            g=arrays['components'].astype(np.float64);n=np.linalg.norm(g,axis=1)
            q=np.divide(g,n[:,None],out=np.zeros_like(g),where=n[:,None]>0)
            d=arrays['after'].astype(np.float64)-arrays['before'].astype(np.float64)
            dual=np.array(row['one_fixed_saved_array_cone_proposal']['dual'],dtype=np.float64)
            intended=d-q.T@dual
            assert np.allclose(q@intended,row['one_fixed_saved_array_cone_proposal']['normalized_constraint_dots'],rtol=0,atol=1e-14)
            values=np.stack([arrays['before'],arrays['after'],(arrays['before'].astype(np.float64)+intended).astype(np.float32)])
            proposal_path=OUT/f'evidence/proposals/proposal{update:04d}.npz';proposal_path.parent.mkdir(parents=True,exist_ok=True)
            with proposal_path.open('xb') as stream:np.savez(stream,values=values,dual=dual,intended_delta=intended)
            with np.load(proposal_path,allow_pickle=False) as proposed:arithmetic=check_proposal(arrays,proposed)
        record=read(record_path)
        probes.append({'update':update,'case_ids':record['case_ids'],'step_arrays':destination.relative_to(OUT).as_posix(),
                       'step_record':destination.with_suffix('.json').relative_to(OUT).as_posix(),
                       'proposal_arrays':proposal_path.relative_to(OUT).as_posix(),'arithmetic':arithmetic,
                       'proposal_flat_float32_sha256':[hashlib.sha256(v.tobytes()).hexdigest() for v in values]})
    normalizers=read(RETURN/'outputs/cohort_loss_setup.json')
    for path in [Path(__file__),ROOT/'scripts/cctv_dgp_actual_step_review_v1_contract.py',SOURCE/'protocol.json',analysis_path,
                 ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_analysis/independent_analysis_audit.json',milestone_path,
                 milestone_path.with_name('independent_closure_audit.json'),RETURN/'outputs/cohort_loss_setup.json',
                 ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_return_import.json']:
        bindings[path.relative_to(ROOT).as_posix()]=sha(path)
    draft={'format':FORMAT,'UTC':datetime.now(timezone.utc).isoformat(),'hypothesis':'Compare finite actual/rounded cone parameter proposals on fixed cohorts; first-order feasibility alone does not establish preservation.',
           'parent_V41_protocol_sha256':sha(SOURCE/'protocol.json'),'parent_return_archive_sha256':imported['archive_sha256'],
           'source_evidence_sha256':bindings,'copied_assets_sha256':copied,'cases':cases,'references':references,'cohorts':cohorts,
           'probes':probes,'proposals':PROPOSALS,'parameter_layout':p['parameter_layout'],'initial_states':p['initial_states'],
           'original_checkpoint_sha256':p['original_checkpoint_sha256'],'recognizer_weights_sha256':p['recognizer_weights_sha256'],
           'normalizers':{'feature':normalizers['feature_normalizer'],'interior':normalizers['interior_normalizer']},
           'terms':p['terms'],'retained_capacity_gates':p['retained_capacity_gates'],'budgets':BUDGETS,
           'raw_metric_definition':'float32 RGB versus paired target, same observed support; independent NumPy/skimage metrics, distinct from Torch objective terms',
           'PNG_quantization':'floor(float32 raw*255), outside support copied from input bytes; no display enhancement',
           'post_outcome_TRAIN_failure_probe_selection':True,'fixed_cohorts_unchanged':True,'no_damping_sweep':True,
           'optimizer_updates':0,'gradient_queries':0,'new_trained_checkpoint':False,'native_or_DEV_or_reserved_final_used':False,
           'pretrained_restorer_targets_or_weights_used':False,'automatic_follow_on':False,'app_promotion':False,
           'reference_provenance_paths_are_metadata_only':True,'native_field_means_original_photo_not_native_CCTV':True,
           'manual_VM_execution_required':True,'goal_complete':False,'prepared_only':True}
    write(OUT/'protocol.draft.json',draft)
    write(ROOT/'outputs/cctv_dgp_actual_step_review_v1_inputs_preparation.json',{'complete':True,'cases':145,'references':29,
         'saved_step_probes':10,'proposal_vectors':30,'draft_sha256':sha(OUT/'protocol.draft.json'),
         'new_neural_calls':0,'gradient_queries':0,'optimizer_updates':0,'VM_calls':0,'seconds':time.monotonic()-start,'cap_seconds':300})
    print({'complete':True,'cases':145,'references':29,'proposals':30,'copied_assets':len(copied),'seconds':time.monotonic()-start},flush=True)


if __name__=='__main__':main()
