"""Seal finite manual input-gradient measurement; no neural execution locally."""
import ast
from datetime import datetime,timezone
from pathlib import Path
import shutil
import tarfile

from import_cctv_dgp_appearance_input_response_v1_return import ROOT,OUT,NAME,STEM,SOURCE,BANK,read,sha,write


def text_file(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(value)


def main():
    analysis=OUT/'cctv_dgp_conditioning_response_v1_analysis'
    proof=[analysis/n for n in ['import.json','independent_audit.json','response_review.json','saved_embedding_precision_review.json','visual_review.json','publication/publication_audit.json']]
    assert all(read(f)['complete'] for f in proof)
    assert read(analysis/'visual_review.json')['all20_sheets_and1000_cells_directly_viewed']
    protected=read(analysis/'local_protected_before.json')['files_sha256'];assert all(sha(ROOT/n)==h for n,h in protected.items())
    source=read(SOURCE/'protocol.json');manifest=read(SOURCE/'export_manifest.json');old=read(BANK/'protocol.json')
    response_pin=sha(SOURCE/'protocol.json');bank_pin=sha(BANK/'protocol.json')
    assert response_pin=='e72d76e3f2330420ee93b43cf528d0b49da132f356a30c336a1ec578c8f82f44'
    assert bank_pin=='f64154bd602070edaea61ccbf960ec3028e545c6540a4421ec41f9af21128a12'
    assert manifest['complete_diagnostic'] and not manifest['complete_pilot'] and not manifest['failure_present']
    # export.log is empty in the archive but grows after the VM export finishes.
    # It remains retained; do not require its archived empty bytes at the live VM path.
    response_bindings={n:h for n,h in manifest['files'].items() if n!='export.log'}
    response_bindings['export_manifest.json']=sha(SOURCE/'export_manifest.json')
    bank_bindings=source['parent_required_sha256']
    for folder,bindings in [(SOURCE,response_bindings),(BANK,bank_bindings)]:assert all(sha(folder/n)==h for n,h in bindings.items())
    assert len(response_bindings)==1246 and len(bank_bindings)==5852
    for kind in ['contract','guard','archive']:
        assert read(OUT/f'cctv_dgp_appearance_input_response_v1_{kind}_green.json')['complete']
    assert read(OUT/'cctv_dgp_appearance_input_response_v1_batch_fixture.json')['complete']
    packet=OUT/NAME;assert not packet.exists();packet.mkdir()
    origins={}
    copies=[(ROOT/'scripts/cctv_dgp_appearance_input_response_v1_vm.py','scripts/cctv_dgp_appearance_input_response_v1_vm.py'),
            (ROOT/'scripts/cctv_dgp_appearance_input_response_v1_contract.py','input_projection_contract.py'),
            (SOURCE/'conditioning_scopes.py','input_scopes.py'),(BANK/'cctv_dgp_pilot.py','cctv_dgp_pilot.py'),
            (BANK/'models/identity_loss.py','models/identity_loss.py')]
    for path,name in copies:
        ast.parse(path.read_text(encoding='utf-8'),feature_version=(3,10));target=packet/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target);origins[name]=dict(source_sha256=sha(path),prepared_sha256=sha(target),exact_copy=True)
    text_file(packet/'models/__init__.py','')
    evidence={'parameter_response_audit.json':analysis/'independent_audit.json','response_review.json':analysis/'response_review.json',
        'saved_embedding_precision_review.json':analysis/'saved_embedding_precision_review.json','visual_review.json':analysis/'visual_review.json',
        'source_publication_audit.json':analysis/'publication/publication_audit.json',
        'batch_fixture.json':OUT/'cctv_dgp_appearance_input_response_v1_batch_fixture.json'}
    evidence.update({f'{kind}_green.json':OUT/f'cctv_dgp_appearance_input_response_v1_{kind}_green.json' for kind in ['contract','guard','archive']})
    for name,path in evidence.items():
        (packet/'evidence').mkdir(exist_ok=True);shutil.copyfile(path,packet/'evidence'/name)
    shell=f'''#!/usr/bin/env bash
set -u -o pipefail
cd "$HOME/forensic-dgp/{NAME}" || exit 1
test -n "${{TMUX:-}}" || {{ echo 'Human manual tmux required'; exit 1; }}
test "$#" -eq 1 || exit 1
test ! -e review.log || {{ echo 'Used diagnostic; retain it'; exit 1; }}
PIN="$1"
PY="$HOME/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python"
timeout --signal=TERM --kill-after=30s 480s "$PY" -B -u scripts/cctv_dgp_appearance_input_response_v1_vm.py --root . --response-root ../cctv_dgp_conditioning_response_v1_vm --bank-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha "$PIN" --measure >review.log 2>&1
code=$?
printf '%s\\n' "$code" >review_exit_code.txt
cat review.log
printf 'Diagnostic exit code: %s\\n' "$code"
timeout --signal=TERM --kill-after=30s 210s "$PY" -B -u scripts/cctv_dgp_appearance_input_response_v1_vm.py --root . --response-root ../cctv_dgp_conditioning_response_v1_vm --bank-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha "$PIN" --export >export.log 2>&1
export_code=$?
printf '%s\\n' "$export_code" >export_exit_code.txt
cat export.log
printf 'Export exit code: %s\\n' "$export_code"
exit "$export_code"
'''
    text_file(packet/'scripts/run_input_response.sh',shell)
    groups=[g for g in source['raw_gradient_groups'] if g['metric']=='ArcFace_loss'];assert len(groups)==10
    p=dict(format='frozen-appearance-input-response-v1',name=NAME,UTC=datetime.now(timezone.utc).isoformat(),
        response_protocol_sha256=response_pin,bank_protocol_sha256=bank_pin,
        response_required_sha256=response_bindings,bank_required_sha256=bank_bindings,
        excluded_volatile_VM_paths=dict(response=['export.log'],reason='Archived empty export log grows after export; retained rather than deleted'),
        case_ids=source['case_ids'],reference_blocks=source['reference_blocks'],appearance_groups=groups,
        signed_labels=[r['label'] for r in source['step_plan']],scientific_thresholds=old['scientific_thresholds'],
        original_restorer_checkpoint_sha256=old['original_checkpoint_sha256'],recognizer_weights_sha256=bank_bindings['weights/w600k_r50.onnx'],
        recognizer_state='9ee66c2faefe0b82224c2cea6811073fbf70e8036808e293878109f078ea3963',
        expected_torch='2.9.1+cu129',gradient_repeats=2,maximum_gradient_queries=200,recognizer_forward_cap=40,
        restorer_forward_cap=0,neural_parameter_displacements=0,optimizer_updates=0,accepted_updates=0,full_epochs=0,
        derivative_target='Canonical saved RGB leaf only; all recognizer parameters frozen and no restorer loaded',
        worker_seconds=450,measurement_seconds=420,round_seconds=180,export_seconds=180,
        external_measurement_seconds=480,external_export_seconds=210,termination_grace_seconds=30,
        minimum_free_bytes=3*1024**3,disk_reserve_bytes=1024**3,output_byte_cap=512*1024**2,peak_VRAM_bytes=12*1024**3,peak_RSS_bytes=8*1024**3,
        central_comparison_fraction=.05,central_comparison_is_diagnostic_only=True,
        original_cross_run_elementwise_rtol=1e-4,original_cross_run_elementwise_atol=1e-8,
        original_cross_run_norm_rtol=1e-4,original_cross_run_norm_atol=1e-10,
        existing_parity_failures_remain_binding=True,original_forward_embedding_rtol=1e-6,original_forward_embedding_atol=1e-6,
        original_derivative_scalar_alignment_bound=1e-6,original_independent_scalar_replay_bound=1e-13,
        full_saved_gradient_plan=dict(gradient_batches=40,per_case_RGB_gradients=200,source_profile_pair_repeat_comparisons=60,new_face_images=0),
        component_origins=origins,externally_pretrained_recognizer_is_measurement_only=True,
        observed_crop_affines='Exact frozen2x3 target-to112 matrices; no new detection or alignment',
        precision='Prepared float32 RGB; double cosine loss; float32 input gradients; float64 actual saved RGB displacement dots',
        scientific_metric_and_quality_gates_unchanged=True,no_training_candidate_or_rate_search=True,
        no_new_target_or_initializer=True,no_source_or_final_identity_role_changes=True,
        prior_return_archive_sha256=read(analysis/'import.json')['archive_sha256'],
        timing_estimate=dict(prior_manual_measurement_seconds=366.58775418,prior_gradient_queries=640,prior_recognizer_batches=360,
            estimated_measurement_minutes=[2,5],estimated_with_export_minutes=[3,7],estimate_is_not_new_CUDA_evidence=True),
        native_pixels_used=0,final_identity_pixels=0,app_promotion=False,automatic_follow_on=False,
        failed_run_resume_allowed=False,model_qualified=False,more_epochs_qualified=False,goal_complete=False)
    p['assets_sha256']={f.relative_to(packet).as_posix():sha(f) for f in packet.rglob('*') if f.is_file()}
    write(packet/'protocol.json',p);pin=sha(packet/'protocol.json')
    importer=ROOT/'scripts/import_cctv_dgp_appearance_input_response_v1_return.py';source_text=importer.read_text(encoding='utf-8')
    assert source_text.count("PIN='__PIN__'")==1
    text_file(OUT/'cctv_dgp_appearance_input_response_v1_importer_template.py',source_text)
    importer.write_text(source_text.replace("PIN='__PIN__'",f"PIN='{pin}'",1),encoding='utf-8',newline='\n')
    archive=OUT/(STEM+'-execution.tar.gz')
    with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
        for f in sorted(f for f in packet.rglob('*') if f.is_file()):tar.add(f,arcname=NAME+'/'+f.relative_to(packet).as_posix(),recursive=False)
    text_file(Path(str(archive)+'.sha256'),sha(archive)+'  '+archive.name+'\n')
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(OUT/'cctv_dgp_appearance_input_response_v1_preparation.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        preparer_sha256=sha(__file__),protocol_sha256=pin,archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
        packet_assets=len(p['assets_sha256']),response_file_bindings=len(response_bindings),bank_file_bindings=len(bank_bindings),
        volatile_export_log_retained_and_excluded_from_live_dependency_hashes=True,
        prospective_importer_sha256=sha(importer),prospective_return_auditor_sha256=sha(ROOT/'scripts/audit_cctv_dgp_appearance_input_response_v1_return.py'),
        local_neural_gradient_or_training_calls=0,manual_diagnostic_unrun=True,model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,protocol_sha256=pin,archive_bytes=archive.stat().st_size,packet_assets=len(p['assets_sha256']),manual_unrun=True))


if __name__=='__main__':main()
