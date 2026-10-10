"""Independent milestone readback and preservation audit; zero neural work."""
import hashlib,json,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/completion_input_footprints_v1_milestone'
PREVIOUS=ROOT/'outputs/cctv_dgp_v32_gradient_return_v33_probe_milestone'
C=ROOT/'outputs/completion_input_footprints_comparison_v1_r1'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    started=time.monotonic();m=read(OUT/'milestone.json');assert m['complete']
    for n,h in m['new_evidence_sha256'].items():assert sha(ROOT/n)==h,n
    saved=ROOT/m['previous_handoff_path'];before=saved.read_bytes();after=(ROOT/'PROJECT_HANDOFF.md').read_bytes()
    at=before.index(b'\n')+1;size=m['document']['addition_bytes']
    assert after[:at]==before[:at] and after[at+size:]==before[at:]
    assert sha(saved)==m['document']['before_sha256'] and sha(ROOT/'PROJECT_HANDOFF.md')==m['document']['after_sha256']
    old=read(PREVIOUS/'milestone.json');assert sha(PREVIOUS/'milestone.json')==m['previous_milestone_sha256']
    for n,h in old['new_evidence_sha256'].items():assert sha(saved if n=='PROJECT_HANDOFF.md' else ROOT/n)==h,n
    for n,h in read(PREVIOUS/'final_readback.json')['evidence_sha256'].items():assert sha(saved if n=='PROJECT_HANDOFF.md' else ROOT/n)==h,n
    p,r,a,v=map(read,[C/'protocol.json',C/'results.json',C/'independent_saved_output_audit.json',C/'visual_review.json'])
    assert r['state_before']==r['state_after']==read(ROOT/'outputs/dgp_app_covering_review_v3/results.json')['state_before']['completion']
    assert r['forwards']=={'completion':28,'internal512':28,'DGP':0,'detector':0} and r['seconds']<600
    assert a['exact_outputs']==32 and a['internal512_to256_compositions']==28 and a['input_exclusions_retained']==4
    assert a['visible_source_bytes_exact']==5483088 and a['protected_source_bytes_exact']==1075314
    assert v['all32_eligible_outputs_actually_reviewed'] and v['all8_comparison_pages_viewed_at_original_detail']
    assert not any(v[k] for k in ['automatic_quality_qualification','assisted_quality_qualification','app_adoption','independent_final_review','goal_complete'])
    for n,h in p['app_preservation_sha256'].items():assert sha(ROOT/n)==h,n
    assert read(C/'independent_mask_audit.json')['protected_overlaps']==0
    failed=ROOT/'outputs/completion_input_footprints_comparison_v1'
    assert not (failed/'execution.json').exists() and read(failed/'external_receipt.json')['worker_exit_code']==1
    assert 'ModuleNotFoundError' in (failed/'inference.log').read_text()
    probe=read(ROOT/'outputs/cctv_dgp_v33_probe_status_v1_r1/inspection_review.json')
    assert probe['snapshot_only_not_a_live_future_state_claim'] and not probe['agent_started_VM'] and not probe['agent_started_training']
    archive=ROOT/'outputs/cctv-dgp-loss-cone-probe-v33-results.tar.gz';export=read(ROOT/'outputs/cctv-dgp-loss-cone-probe-v33-export.json')
    assert export['archive_sha256']==sha(archive)=='20a196d6515bc46d2e53c74d1ae4182c28f39812649e32bf44a2c684b56b7c40'
    assert archive.stat().st_size==export['bytes']==1166123827 and export['optimizer_updates']==8
    receipt={'complete':True,'milestone_sha256':sha(OUT/'milestone.json'),'checker_sha256':sha(Path(__file__)),
        'new_bindings_verified':len(m['new_evidence_sha256']),'previous_bindings_preserved':len(old['new_evidence_sha256']),
        'full_previous_handoff_preserved':True,'all32_output_audit_and_actual_development_review_preserved':True,
        'prior_failure_evidence_preserved':True,'all14_app_bindings_unchanged':True,'V33_archive_acquisition_verified':True,
        'V33_full_audit_status_separate':True,'local_model_forwards':0,'local_optimizer_updates':0,
        'app_adoption':False,'independent_final_quality_review':False,'goal_complete':False,'seconds':time.monotonic()-started}
    with (OUT/'independent_closure_audit.json').open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'complete':True,'bindings':receipt['new_bindings_verified'],'seconds':receipt['seconds']}))


if __name__=='__main__':main()
