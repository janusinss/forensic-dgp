"""Preserve historical evidence and bind V25 closure plus zero-update diagnostic."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/dgp_spatial_features_v25_audit_and_gradient_diagnostic_milestone'
PREVIOUS=ROOT/'outputs/dgp_spatial_features_v25_preparation_milestone/milestone.json'
DOCS=['PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md',
    'CCTV_DGP_DEGRADED_DETAIL_V24_PLAN.md','CCTV_DGP_DEGRADED_DETAIL_V24_VM.md',
    'CCTV_DGP_SPATIAL_FEATURES_V25_PLAN.md','CCTV_DGP_SPATIAL_FEATURES_V25_VM.md']
SOURCES=['scripts/audit_cctv_dgp_spatial_features_v25.py','scripts/audit_cctv_dgp_spatial_features_v25_features.py',
    'scripts/audit_cctv_dgp_spatial_features_v25_execution.py','scripts/verify_cctv_dgp_spatial_features_v25_auditor_source.py',
    'tests/test_cctv_dgp_spatial_features_v25_return_audit.py','scripts/diagnose_cctv_dgp_spatial_features_v25.py',
    'scripts/verify_cctv_dgp_spatial_features_v25_diagnostic.py','scripts/record_cctv_dgp_spatial_features_v25_visual_review.py',
    'scripts/audit_cctv_dgp_spatial_features_v25_loss_v1.py','scripts/verify_cctv_dgp_spatial_features_v25_loss_v1.py',
    'scripts/cctv_dgp_v25_gradient_diagnostic_v1_vm.py','scripts/prepare_cctv_dgp_v25_gradient_diagnostic_v1.py',
    'scripts/verify_cctv_dgp_v25_gradient_diagnostic_v1.py','scripts/audit_cctv_dgp_v25_gradient_diagnostic_v1_preparation.py',
    'tests/test_cctv_dgp_v25_gradient_diagnostic_v1.py','scripts/record_cctv_dgp_v25_closure_gradient_milestone.py',
    'scripts/verify_cctv_dgp_v25_closure_gradient_milestone.py']
LATEST='''**Latest research milestone — 6 October 2026: V25 audited failure; fixed-state gradient diagnostic verified, manual L4 measurement pending.**

The325,762,305-byte V25 return is verified and safely imported; independent replay
checks628 returned files/235 assets,100 raw/PNG pairs, both0/50 snapshots,50 original
DGP CPU forwards and250 frozen feature arrays. All original replay/quality bounds
stay unchanged. The trainer correctly stops at50 updates/51 backwards:
**0.0282235213% delivered degraded structure gain, below the required1%.** Final800
never runs. complete:true packages the retained failure; it does not accept a model.

All ten original-cell sheets/50 paired photographic TRAINING cases are reviewed;
200 exact cells are independently checked. No convincing whole-face gain is visible.
All26 spatial-head tensors change, all five projection gradients are active and
the stopped correction now reaches the output without the old final filter
attenuation. The typical degraded correction is still0.12994 of one byte level.
The saved corrected loss decreases through degraded cases, with a clear-preservation
cost; the V23 clear-reward mismatch does not explain this small saved improvement.
Scalar evidence does not establish GPU gradient competition or optimizer causality.

The next transfer is a **zero-update fixed-state gradient diagnostic**, not a new
training recipe or an unchanged V25 retry. It uses the two saved heads, same50
TRAIN cases, original seven objective terms and250 frozen own-DGP feature arrays.
The15,259-byte packet is independently checked:20 head batches/140 gradient calls/
120 recognizer forwards, zero DGP forwards/optimizer updates. Worker420s, external
480s plus30s grace; export30s internally/60s externally plus10s grace. Require1GiB
free disk and an idle L4; preserve any stop. Actual L4 gradients and independent
returned-matrix audit remain pending before choosing another training recipe.

Thirteen V25 auditor regressions and nine new diagnostic packet/source/matrix
guards pass. Python3.10/actual Windows rejection/Bash syntax/frozen mask and affine
geometry checks pass. Preparation failure evidence is retained. No local gradients,
backwards or optimization and no assistant VM/cloud action occur. The user-selected
own-DGP spatial/feature direction and all visible facial features together remain
in scope. Exact manual upload/install/tmux/launch/download steps are in the new
diagnostic runbook; PuTTY downloads remain three separate remote-source calls.

Original checkpoints/splits/failed gates, previous513 milestone bindings, app22
bindings and concurrent completed VM maintenance are preserved. The app and its
historical34 regressions/bundled inline Playwright checks remain unchanged. No
candidate is promoted. Native CCTV stays unpaired; paired TRAINING metrics remain
separate. No native/reserved-final/new covering pixels, ethnicity or Zamboanga
performance claims are introduced. The previously useful CCTV crop stays usable
despite model softness. Useful native output, candidate app parity/full flow,
insufficient-information handling, all seven automatic/assisted covering families
and independent final review remain required. Goal active/incomplete.

[V25 audited result](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_SPATIAL_FEATURES_V25_RESULTS.md>) ·
[Finite diagnostic design](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V25_GRADIENT_DIAGNOSTIC_V1_PLAN.md>) ·
[Manual diagnostic commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V25_GRADIENT_DIAGNOSTIC_V1_VM.md>) ·
[Milestone](<C:/xampp/htdocs/YEAR 4/Testing/outputs/dgp_spatial_features_v25_audit_and_gradient_diagnostic_milestone/milestone.json>)

Previous bodies below are preserved history. Their pending-return/manual-V25-launch
statements are superseded by the audited V25 failure and diagnostic-only next step.
Old V22–V25 training commands are historical; do not repeat those failed recipes.
'''


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):
    with p.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(v,indent=2,allow_nan=False)+'\n')


def main():
    start=time.monotonic();assert not OUT.exists(),'Preserve previous closure milestone'
    assert sha(PREVIOUS)=='801051639495c408a8ec124ed7d37d70c59a2a6e1243f82b7c91d39abe347c12'
    previous=read(PREVIOUS)
    for name,digest in previous['new_evidence_sha256'].items():assert sha(ROOT/name)==digest,name
    audit=read(ROOT/'outputs/cctv_dgp_spatial_features_v25_independent_audit.json')
    diagnosis=read(ROOT/'outputs/cctv_dgp_spatial_features_v25_diagnostic/results.json')
    visual=read(ROOT/'outputs/cctv_dgp_spatial_features_v25_diagnostic/visual_review.json')
    loss=read(ROOT/'outputs/cctv_dgp_spatial_features_v25_loss_audit_v1/results.json')
    assert audit['complete'] and audit['VM_failure_present'] and not audit['necessary_capacity_pass']
    assert audit['early_structure_stop']['pass'] is False and audit['training_progress_receipt']['updates']==50
    assert diagnosis['complete'] and visual['complete'] and visual['cases_reviewed']==50 and visual['exact_cells']==200
    assert diagnosis['head_exact_layer_traces']==50 and loss['complete'] and loss['optimizer_updates']==0
    for name in ['outputs/cctv_dgp_spatial_features_v25_diagnostic/independent_saved_diagnostic_audit.json',
        'outputs/cctv_dgp_spatial_features_v25_loss_audit_v1/independent_saved_loss_audit.json',
        'outputs/cctv_dgp_spatial_features_v25_return_audit_preparation/independent_source_audit.json']:
        assert read(ROOT/name)['complete'],name
    preparation=ROOT/'outputs/cctv_dgp_v25_gradient_diagnostic_v1_preparation'
    packet=read(preparation/'preparation.json');packet_audit=read(preparation/'independent_packet_audit.json')
    readback=read(preparation/'independent_preparation_readback.json')
    assert packet['complete'] and packet_audit['complete'] and readback['complete'] and readback['Bash_n_exit_code']==0
    assert readback['unsafe_packet_source_matrix_tests_passed']==9 and readback['original_square_erosions_exact']==100
    apppath=ROOT/'outputs/dgp_app_v3_integration_record.json';app=read(apppath)
    for name,digest in {**app['sources_sha256'],**app['evidence_sha256']}.items():assert sha(ROOT/name)==digest,name
    for name in SOURCES:ast.parse((ROOT/name).read_text(encoding='utf-8'),feature_version=(3,10))
    OUT.mkdir();(OUT/'before_docs').mkdir();before={}
    for name in DOCS:
        shutil.copy2(ROOT/name,OUT/'before_docs'/name);before[name]=sha(OUT/'before_docs'/name)
    assert b'VM storage cleanup completed; future VM use preserved' in (OUT/'before_docs/PROJECT_HANDOFF.md').read_bytes()
    locations={name:(OUT/'before_docs'/name).relative_to(ROOT).as_posix() for name in DOCS}
    write(OUT/'before_update_plan.json',{'date':'2026-10-06','before_docs_sha256':before,
        'previous_milestone_sha256':sha(PREVIOUS),'previous513_original_locations':locations,
        'app_record_sha256':sha(apppath),'runner_sha256':sha(Path(__file__)),'actual_gradient_diagnostic_pending':True})
    for name in DOCS:
        assert sha(ROOT/name)==before[name],'Concurrent document update: '+name
        old=(OUT/'before_docs'/name).read_bytes();split=old.index(b'\n')+1
        (ROOT/name).write_bytes(old[:split]+b'\n'+LATEST.encode('utf-8')+b'\n'+old[split:])
        assert (ROOT/name).read_bytes().endswith(old[split:])
    paths={ROOT/name for name in DOCS+SOURCES+['CCTV_DGP_SPATIAL_FEATURES_V25_RESULTS.md',
        'CCTV_DGP_V25_GRADIENT_DIAGNOSTIC_V1_PLAN.md','CCTV_DGP_V25_GRADIENT_DIAGNOSTIC_V1_VM.md',
        'outputs/cctv-dgp-spatial-features-v25-results.tar.gz','outputs/cctv-dgp-spatial-features-v25-results.tar.gz.sha256',
        'outputs/cctv-dgp-spatial-features-v25-export.json','outputs/cctv_dgp_spatial_features_v25_return_import.json',
        'outputs/cctv_dgp_spatial_features_v25_independent_audit.json',
        'outputs/cctv-dgp-v25-gradient-diagnostic-v1-execution.tar.gz','outputs/cctv-dgp-v25-gradient-diagnostic-v1-execution.tar.gz.sha256']}
    for name in ['cctv_dgp_spatial_features_v25_return','cctv_dgp_spatial_features_v25_return_audit_preparation',
        'cctv_dgp_spatial_features_v25_diagnostic','cctv_dgp_spatial_features_v25_loss_audit_v1',
        'cctv_dgp_v25_gradient_diagnostic_v1_vm','cctv_dgp_v25_gradient_diagnostic_v1_preparation']:
        paths.update(p for p in (ROOT/'outputs'/name).rglob('*') if p.is_file())
    paths.update(p for p in OUT.rglob('*') if p.is_file())
    evidence={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}
    degraded=diagnosis['groups']['degraded']
    record={'complete':True,'date':'2026-10-06','scope':'Audited V25 failure and finite manual-L4 zero-update gradient diagnostic; Goal incomplete',
        'new_evidence_sha256':evidence,'previous_milestone_sha256':sha(PREVIOUS),'previous513_original_locations':locations,
        'app_record_sha256':sha(apppath),'actual_V25_return_present':True,'archive_bytes':325762305,
        'archive_sha256':'d32b3d68f5bc4e91858ab05c6cda1efda6109520a6776420cede603ba4d33434',
        'closed_V25_protocol_sha256':audit['protocol_sha256'],'V25_failure_retained':True,'updates':50,'VM_backwards':51,
        'early_structure_gain_percent':100*audit['early_structure_stop']['relative_feature_error_gain'],'minimum_gain_percent':1.,
        'complete_training_or_quality_pass':False,'final800_executed':False,'whole_face_reviewed_cases':50,'reviewed_exact_cells':200,
        'median_degraded_correction_byte_levels':255*degraded['median_saved_correction_RMS'],
        'five_own_DGP_projection_gradients_active':True,'saved_loss_objective_reduction':loss['groups']['all']['objective_reduction'],
        'selected_route':'A','new_training_recipe':False,'gradient_diagnostic_protocol_sha256':packet['protocol_sha256'],
        'gradient_diagnostic_archive_sha256':packet['archive_sha256'],'gradient_diagnostic_archive_bytes':packet['archive_bytes'],
        'actual_L4_gradient_diagnostic_pending':True,'independent_returned_gradient_matrix_audit_pending':True,
        'audit_regressions':13,'diagnostic_guard_regressions':9,'all_original_quality_gates_and_failed_stops_retained':True,
        'local_return_replay_counts':audit['counts'],'local_original_DGP_CPU_forwards':50,'local_exact_head_layer_traces':50,
        'local_saved_loss_counts':loss['counts'],'local_gradient_calls':0,'local_backward_calls':0,'local_optimizer_updates':0,
        'native_or_reserved_used':False,'app_unchanged':True,'app_promotion':False,'VM_actions':False,
        'independent_final_review_complete':False,'goal_status':'active','goal_complete':False,'seconds':time.monotonic()-start}
    write(OUT/'milestone.json',record)
    print(json.dumps({k:v for k,v in record.items() if k not in ['new_evidence_sha256','previous513_original_locations']},indent=2))
    print('Bound files:',len(evidence))


if __name__=='__main__':main()
