"""Freeze a new response measurement; never execute it on a model locally."""
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tarfile

from import_cctv_dgp_raw_conditioning_learning_v1_r1_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, PIN as PREVIOUS_PIN, read, sha, write

NAME='cctv_dgp_conditioning_response_v1_vm'
STEM='cctv-dgp-conditioning-response-v1'


def text_file(path,text):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:f.write(text)


def main():
    for filename in ['failure_forensic_audit.json','visual_review.json','gradient_prediction_review.json','publication/publication_audit.json']:
        assert read(ANALYSIS/filename)['complete']
    for filename in ['contract_green.json','guard_green.json','archive_members_green.json','snapshot_fixture.json']:
        assert read(OUT/('cctv_dgp_conditioning_response_v1_'+filename))['complete']
    prior=read(RETURN/'protocol.json');old=read(PARENT/'protocol.json')
    assert sha(RETURN/'protocol.json')==PREVIOUS_PIN
    assert read(ANALYSIS/'failure_forensic_audit.json')['accepted_trials_verified']==[]
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    assert all(sha(RETURN/n)==h for n,h in prior['assets_sha256'].items())
    parent_bytes=0
    for name,h in prior['parent_required_sha256'].items():
        assert sha(PARENT/name)==h,name;parent_bytes+=(PARENT/name).stat().st_size
    destination=OUT/NAME;assert not destination.exists();destination.mkdir()
    files=['conditioning_contract.py','conditioning_losses.py','conditioning_model.py','conditioning_raw_losses.py','conditioning_scopes.py','readouts.npz']
    origins={}
    for name in files:
        if name=='conditioning_model.py':
            source=(RETURN/name).read_text(encoding='utf-8');assert source.count("NAME = 'cctv_dgp_raw_conditioning_learning_v1_r1_vm'")==1
            text_file(destination/name,source.replace("NAME = 'cctv_dgp_raw_conditioning_learning_v1_r1_vm'",f"NAME = '{NAME}'",1))
        else:shutil.copyfile(RETURN/name,destination/name)
        origins[name]=dict(source_sha256=sha(RETURN/name),prepared_sha256=sha(destination/name),namespace_only_change=name=='conditioning_model.py')
    for source_name,destination_name in [('cctv_dgp_conditioning_response_v1_vm.py','scripts/cctv_dgp_conditioning_response_v1_vm.py'),
                                          ('cctv_dgp_conditioning_response_v1_contract.py','response_contract.py')]:
        path=ROOT/'scripts'/source_name;ast.parse(path.read_text(encoding='utf-8'),feature_version=(3,10))
        (destination/destination_name).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination/destination_name)
    before=RETURN/'outputs/raw_conditioning_learning_v1_r1'
    shutil.copyfile(before/'round00/direction.npy',destination/'saved_direction.npy')
    assert sha(destination/'saved_direction.npy')==sha(before/'round00/direction.npy')
    for filename in ['failure_forensic_audit.json','visual_review.json','gradient_prediction_review.json','original_checker_stop.json']:
        (destination/'evidence').mkdir(exist_ok=True);shutil.copyfile(ANALYSIS/filename,destination/'evidence'/filename)
    shutil.copyfile(RETURN/'protocol.json',destination/'evidence/previous_protocol.json')
    for filename in ['contract_green.json','guard_green.json','archive_members_green.json','snapshot_fixture.json']:
        shutil.copyfile(OUT/('cctv_dgp_conditioning_response_v1_'+filename),destination/'evidence'/filename)
    shell=f'''#!/usr/bin/env bash
set -u -o pipefail
cd "$HOME/forensic-dgp/{NAME}" || exit 1
test -n "${{TMUX:-}}" || {{ echo 'Human manual tmux required'; exit 1; }}
test ! -e review.log || {{ echo 'Used diagnostic; preserve it'; exit 1; }}
test "$#" -eq 1 || exit 1
PIN="$1"
PY="$HOME/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python"
timeout --signal=TERM --kill-after=30s 1230s "$PY" -B -u scripts/cctv_dgp_conditioning_response_v1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha "$PIN" --diagnose >review.log 2>&1
code=$?
printf '%s\\n' "$code" >review_exit_code.txt
cat review.log
printf 'Diagnostic exit code: %s\\n' "$code"
timeout --signal=TERM --kill-after=30s 480s "$PY" -B -u scripts/cctv_dgp_conditioning_response_v1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha "$PIN" --export >export.log 2>&1
export_code=$?
printf '%s\\n' "$export_code" >export_exit_code.txt
cat export.log
printf 'Export exit code: %s\\n' "$export_code"
exit "$export_code"
'''
    text_file(destination/'scripts/run_response.sh',shell)
    copied_keys=['parent_protocol_sha256','parent_required_sha256','case_ids','reference_blocks','raw_gradient_groups','conditioning_names',
                 'conditioning_shapes','conditioning_parameters','initial_model_state','sample_gradient_begins','preserved_scientific_thresholds','all37_learning_names']
    p={k:prior[k] for k in copied_keys}
    base=read(before/'r00_trial02/decision.json')['rate']
    p.update(format='finite-conditioning-response-v1',name=NAME,UTC=datetime.now(timezone.utc).isoformat(),
        previous_protocol_sha256=PREVIOUS_PIN,previous_return_archive_sha256=read(ANALYSIS/'import.json')['archive_sha256'],
        expected_torch=read(before/'environment.json')['torch'],paired_TRAIN_cases=100,gradient_repeats=2,maximum_gradient_queries=640,
        model_forward_cap=200,recognizer_forward_cap=360,temporary_displacements=6,optimizer_updates=0,accepted_updates=0,full_epochs=0,
        new_direction_solve=False,optimizer_constructed=False,base_signed_magnitude=base,
        step_plan=[dict(label=f"{'minus' if sign<0 else 'plus'}{i:02d}",rate=base*(10.**(-i))*sign) for i in range(3) for sign in [-1,1]],
        saved_direction_sha256=sha(destination/'saved_direction.npy'),
        saved_direction_origin='Exact rejected RAW R1 direction; no local new direction or proposal computation',
        maximum_run_relative_parameter_distance=.001,preserved_brightness_limit=.2,preserved_structure_endpoint=.01,
        worker_seconds=1200,measurement_seconds=1170,gradient_round_seconds=600,export_seconds=450,
        external_diagnostic_seconds=1230,external_export_seconds=480,termination_grace_seconds=30,
        minimum_free_bytes=9*1024**3,disk_reserve_bytes=1024**3,output_byte_cap=int(3.5*1024**3),peak_VRAM_bytes=20*1024**3,peak_RSS_bytes=12*1024**3,
        central_comparison_fraction=.05,central_resolution_multiplier=10,
        central_comparison_is_diagnostic_only_and_not_quality_gate=True,
        prior_cross_run_rtol=1e-4,prior_cross_run_atol=1e-8,prior_cross_run_norm_rtol=1e-4,prior_cross_run_norm_atol=1e-10,
        no_prior_cross_run_parity_failure_reclassified_as_passed=True,original_model_checkpoint_sha256=old['original_checkpoint_sha256'],
        external_frozen_prior_sha256=old['standalone_prior_sha256'],prior_is_attributed_pretrained_and_not_own_trained=True,
        component_origins=origins,initial_behavior='Exact original DGP via anchored same-observed-input prior path; tested manually on L4',
        no_optimizer_no_rate_search_no_candidate_selection=True,no_new_initializer_or_training_target=True,
        every_complete_RAW_PNG_snapshot_and_per_reference_record_retained=True,
        full_output_plan=dict(snapshots=8,RAW_outputs=800,PNG_outputs=800,stage_metric_rows=1600,visual_sheets=20,visual_cells=1000),
        projection_basis=dict(previous_manual_seconds=168.5504133100003,previous_gradient_queries=320,previous_snapshots=4,
            new_gradient_queries=640,new_snapshots=8,estimated_measurement_minutes=[6,12],estimated_total_with_export_minutes=[8,15],
            estimate_is_not_a_new_CUDA_measurement=True),
        native_pixels_used=0,final_identity_pixels=0,source_labels_not_ethnicity_or_Zamboanga_claims=True,
        app_promotion=False,automatic_follow_on=False,failed_run_resume_allowed=False,model_qualified=False,goal_complete=False)
    p['assets_sha256']={f.relative_to(destination).as_posix():sha(f) for f in destination.rglob('*') if f.is_file()}
    write(destination/'protocol.json',p);pin=sha(destination/'protocol.json')
    importer=ROOT/'scripts/import_cctv_dgp_conditioning_response_v1_return.py'
    source=importer.read_text(encoding='utf-8');assert source.count("PIN='__PROTOCOL_SHA256__'")==1
    preserved=OUT/'cctv_dgp_conditioning_response_v1_importer_template.py'
    text_file(preserved,source)
    # The local checker is outside the VM packet; seal its exact protocol pin.
    importer.write_text(source.replace("PIN='__PROTOCOL_SHA256__'",f"PIN='{pin}'",1),encoding='utf-8',newline='\n')
    assert set(p['assets_sha256'])=={f.relative_to(destination).as_posix() for f in destination.rglob('*') if f.is_file() and f.name!='protocol.json'}
    archive=OUT/(STEM+'-execution.tar.gz')
    with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
        for f in sorted(f for f in destination.rglob('*') if f.is_file()):tar.add(f,arcname=NAME+'/'+f.relative_to(destination).as_posix(),recursive=False)
    text_file(Path(str(archive)+'.sha256'),sha(archive)+'  '+archive.name+'\n')
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(OUT/'cctv_dgp_conditioning_response_v1_preparation.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        preparer_sha256=sha(__file__),protocol_sha256=pin,archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
        packet_assets=len(p['assets_sha256']),parent_file_bindings=len(p['parent_required_sha256']),parent_bytes=parent_bytes,
        previous_return_protocol_sha256=PREVIOUS_PIN,previous_failed_qualifications_preserved=True,
        saved_rejected_direction_copied_exactly=True,local_checker_sha256=sha(importer),prospective_return_audit_sha256=sha(ROOT/'scripts/audit_cctv_dgp_conditioning_response_v1_return.py'),
        local_importer_template_sha256=sha(preserved),source_files_unchanged_except_new_immutable_namespace=True,
        local_neural_gradient_optimizer_or_weight_proposal_calls=0,manual_diagnostic_unrun=True,
        model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,protocol_sha256=pin,archive_bytes=archive.stat().st_size,packet_assets=len(p['assets_sha256']),manual_unrun=True))


if __name__=='__main__':main()
