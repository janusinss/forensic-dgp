"""Preserve the preceding handoff and close the completed mask-only comparison."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/completion_input_footprints_v1_milestone'
PREVIOUS=ROOT/'outputs/cctv_dgp_v32_gradient_return_v33_probe_milestone'
COMPARISON=ROOT/'outputs/completion_input_footprints_comparison_v1_r1'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    assert not OUT.exists();old=read(PREVIOUS/'milestone.json')
    for n,h in old['new_evidence_sha256'].items():assert sha(ROOT/n)==h,n
    assert read(PREVIOUS/'independent_closure_audit.json')['complete']
    for n,h in read(PREVIOUS/'final_readback.json')['evidence_sha256'].items():assert sha(ROOT/n)==h,n
    p,r,a,v=map(read,[COMPARISON/'protocol.json',COMPARISON/'results.json',COMPARISON/'independent_saved_output_audit.json',COMPARISON/'visual_review.json'])
    assert all(q['complete'] for q in [r,a,v]) and a['results_sha256']==v['results_sha256']==sha(COMPARISON/'results.json')
    assert v['all32_eligible_outputs_actually_reviewed'] and v['all8_comparison_pages_viewed_at_original_detail']
    assert not v['app_adoption'] and not v['automatic_quality_qualification'] and not v['assisted_quality_qualification']
    for n,h in {**p['sources_sha256'],**p['app_preservation_sha256']}.items():assert sha(ROOT/n)==h,n
    for n,h in r['artifacts_sha256'].items():assert sha(COMPARISON/n)==h,n
    report=ROOT/'CCTV_DGP_COMPLETION_INPUT_FOOTPRINTS_V1_RESULTS.md';body=report.read_text(encoding='utf-8').split('\n',1)[1]
    addition='\n**Completion milestone - 7 October 2026: input-only footprint ablation audited; full-family quality remains incomplete.**\n'+body+'\nThe complete previous handoff follows. Its V33-unlaunched statements are historical: the human has now supplied a hash-matched return and its independent audit is in progress.\n\n'
    OUT.mkdir();(OUT/'before_docs').mkdir();handoff=ROOT/'PROJECT_HANDOFF.md';before=handoff.read_bytes()
    saved=OUT/'before_docs/PROJECT_HANDOFF.md';saved.write_bytes(before)
    at=before.index(b'\n')+1;handoff.write_bytes(before[:at]+addition.encode('utf-8')+before[at:])
    files=[Path(__file__),ROOT/'scripts/verify_completion_input_footprints_v1_milestone.py',report,handoff,saved,
        PREVIOUS/'milestone.json',PREVIOUS/'independent_closure_audit.json',PREVIOUS/'final_readback.json',
        ROOT/'scripts/inspect_cctv_dgp_v33_probe_status_v1.py',ROOT/'scripts/inspect_cctv_dgp_v33_probe_status_v1_r1.py',
        ROOT/'scripts/record_cctv_dgp_v33_stopped_observation_v1.py']
    files.extend(ROOT/n for n in p['sources_sha256'])
    for folder in ['outputs/completion_input_footprints_v1','outputs/completion_input_footprints_comparison_v1',
        'outputs/completion_input_footprints_comparison_v1_r1','outputs/completion_input_footprints_runtime_v1',
        'outputs/cctv_dgp_v33_probe_status_v1','outputs/cctv_dgp_v33_probe_status_v1_r1']:
        files.extend(q for q in sorted((ROOT/folder).rglob('*')) if q.is_file())
    files.extend(sorted((ROOT/'scripts').glob('*completion_input_footprints*py')))
    for suffix in ['-results.tar.gz','-results.tar.gz.sha256','-export.json']:
        files.append(ROOT/('outputs/cctv-dgp-loss-cone-probe-v33'+suffix))
    evidence={q.relative_to(ROOT).as_posix():sha(q) for q in files}
    milestone={'complete':True,'UTC':datetime.now(timezone.utc).isoformat(),'new_evidence_sha256':evidence,
        'previous_milestone_sha256':sha(PREVIOUS/'milestone.json'),'previous_handoff_path':saved.relative_to(ROOT).as_posix(),
        'document':{'before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':sha(handoff),'addition_bytes':len(addition.encode('utf-8'))},
        'all32_supported_development_outputs_reviewed':True,'completion_forwards':28,'DGP_forwards':0,
        'raw_compositions_verified':28,'exact_empty_bypasses':4,'input_exclusions_retained':4,
        'exact_visible_source_bytes':a['visible_source_bytes_exact'],'exact_protected_bytes':a['protected_source_bytes_exact'],
        'input_only_margin_pixels':2,'source_photo_assistance_on_degraded_pairs':True,
        'mask_segmentation_truth':False,'independent_final_quality_review':False,
        'automatic_quality_qualification':False,'assisted_quality_qualification':False,'app_adopted':False,
        'two_pre_neural_failures_retained':True,'VM_status_observation_is_historical_snapshot':True,
        'V33_return_received':True,'V33_full_independent_audit_pending':True,'local_optimizer_updates':0,
        'training_started_by_agent':False,'native_CCTV_or_reserved_final_used':False,'goal_status':'active','goal_complete':False}
    with (OUT/'milestone.json').open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(milestone,indent=2)+'\n')
    print(json.dumps({'complete':True,'bindings':len(evidence),'milestone_sha256':sha(OUT/'milestone.json')}))


if __name__=='__main__':main()
