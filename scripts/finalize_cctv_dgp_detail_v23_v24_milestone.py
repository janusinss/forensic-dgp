"""Preserve prior evidence/docs and record audited V23 / prepared V24. No VM work."""
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from import_cctv_dgp_detail_skip_v23 import read, require, sha, write, safe

OUT = ROOT/'outputs/dgp_detail_skip_v23_audit_and_degraded_detail_v24_milestone'
PREVIOUS = ROOT/'outputs/dgp_detail_prior_v22_r1_audit_and_skip_v23_milestone/milestone.json'
PREVIOUS_SHA = 'fb1c95d29e159976b6ba1233bc41a946fadefd6759beabb57448f73fab7c1eb8'
OLDER = ROOT/'outputs/dgp_structure_and_detail_pilot_milestone_v21_v22/milestone.json'
APP = ROOT/'outputs/dgp_app_v3_integration_record.json'
APP_SHA = 'e744f3708f248012469a111e204222f66261af4ca28bbaf9521ee4c5464b5e7a'
DOCS = ['PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md','CCTV_DGP_DETAIL_SKIP_V23_VM.md']
STATUS = '''**Current milestone — 6 October 2026: V23 failed run independently audited; V24 objective experiment prepared.**

The downloaded V23 archive matches 76,481,832 bytes/SHA256
`b71639cabd17476d5989dfd4582d47de9b747b8bf2a611232e5f62cc67bf9fd9`.
All 370 returned files, 209 original assets, 100 raw/PNG pairs/metrics, two complete
50-case snapshots, stopped state and source-bound timing/gradient/count receipts
pass independent audit. The L4 completed 50 updates/51 backwards, then correctly
failed its unchanged 1% early gate: delivered degraded feature error worsens
0.00223306%. Nonzero direct gradients and changed tensors are verified. All ten
original-resolution sheets/50 cases show no convincing visible structure gain.
Seven original preservation checks diagnostically fail at 50; the unexecuted
800-update final gate is not evaluated. Close V23 without rerun or app adoption.

The exact old objective is decomposed on both saved raw states, independently
arithmetic-checked, and decreases overall through clear-case gains while the
degraded cohort worsens. Clear cases contribute 101.4223% of its net reduction.
This is a demonstrated loss/goal mismatch, not proof of GPU gradient causality.
V24 changes that objective prospectively: degraded-cohort HF normalizers, no
clear target reward and clear baseline preservation controls. Same 4,613-parameter
head, 50 exposed training cases, 800 updates/80 epochs, optimizer, schedule,
appearance/brightness gates, timing budgets and early stops. No failed gate is
waived; original V22/V23 sources, protocols, archives and stopped states remain.

V24 transfer: 218,041,986 bytes, 221 assets/223 regular members. Independent
archive/source/Python3.10/Bash/Windows guard and cohort contracts pass, with eight
loss invariants. All 50 fixed baseline errors match the prior decomposition;
100 fixed CPU filters, no new head/DGP/recognizer prediction in preparation.
Actual CUDA gradients, learning and capacity are pending. A comparator-only
field exclusion correction and sandbox Bash initialization failure are preserved;
neither changes the pilot or quality conditions. If this distinct objective trial
fails, stop blind head-recipe changes and revisit architecture/data/loss.

Prior 644 and older 560 milestone bindings, all 22 current app bindings and prior
document bodies are verified/preserved. The DGP-led Auto/On/Off application,
design, checkpoints and splits are unchanged; historical 34 regressions and
bundled inline Playwright are not rerun for this evidence/transfer-only milestone.
Manual existing L4/g2-standard-4 under ~/forensic-dgp only: verified transfers and
exact pasteable gcloud/tmux/separate-download commands. No assistant VM/cloud
action or local backward/optimizer update. Earlier maintenance entries are
historical; the user has subsequently returned an L4 run, with no new VM polling.

The user's useful-case feedback still requires clearer visible structure; its
usable input is not relabeled insufficient. No native/reserved/covering/COFW-test
pixels enter V24. Native CCTV remains unpaired; these paired TRAINING metrics are
separate. No ethnicity or local Zamboanga inference. Useful native outputs,
canonical app parity, all seven automatic/assisted covering families and independent
final review remain open. Preserve clear glasses, non-obstructing hair and visible
appearance; show the removal area for optional correction, request a less-covered
crop when evidence is insufficient, and return one plausible estimate with the
original/mask and PNG/optional bundle. No exact hidden identity claim. Goal active.

[V23 audited result](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DETAIL_SKIP_V23_RESULTS.md>) ·
[V24 plan](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DEGRADED_DETAIL_V24_PLAN.md>) ·
[V24 commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DEGRADED_DETAIL_V24_VM.md>)

Previous entries below are preserved history and are superseded by this milestone.
'''
CLOSED = '''**Closed 6 October 2026: V23 return independently audited; do not relaunch.**

The original update-50 structure stop is verified: 50 updates/51 backwards,
delivered degraded feature error 0.00223306% worse, no visible gain across all
50 reviewed cases. Packaging succeeded; model capacity did not pass. Original
sources/gates/partial evidence stay intact. The final800 gate was not executed.

V24 prepares a different training-objective experiment with unchanged head,
data, limits and quality gates. Use [V24 commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DEGRADED_DETAIL_V24_VM.md>).
The old V23 steps below are historical. No automatic retry, assistant VM action
or app checkpoint replacement. [Audited result](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DETAIL_SKIP_V23_RESULTS.md>).
'''


def main():
    started=time.monotonic();require(not OUT.exists(),'Preserve previous milestone');OUT.mkdir();(OUT/'before_docs').mkdir()
    require(sha(PREVIOUS)==PREVIOUS_SHA and sha(APP)==APP_SHA,'Prior milestone/app record differs')
    previous=read(PREVIOUS);older=read(OLDER)
    require(sha(OLDER)==previous['previous_milestone_sha256'],'Older milestone differs')
    mapping={name:'outputs/cctv_dgp_detail_skip_v23_reported_stop_v1/before_docs/'+name for name in DOCS}
    for name,digest in previous['new_evidence_sha256'].items():
        require(sha(safe(ROOT,mapping.get(name,name)))==digest,'Previous644 binding changed: '+name)
    for name,digest in older['new_evidence_sha256'].items():
        require(sha(safe(ROOT,previous['previous_document_locations'].get(name,name)))==digest,'Older560 binding changed: '+name)
    app=read(APP);app_bindings={**app['sources_sha256'],**app['evidence_sha256']}
    for name,digest in app_bindings.items():require(sha(safe(ROOT,name))==digest,'App binding differs: '+name)
    require(len(previous['new_evidence_sha256'])==644 and len(older['new_evidence_sha256'])==560 and len(app_bindings)==22,'Prior counts differ')
    audit=read(ROOT/'outputs/cctv_dgp_detail_skip_v23_independent_audit.json')
    visual=read(ROOT/'outputs/cctv_dgp_detail_skip_v23_diagnostic/visual_review.json')
    loss=read(ROOT/'outputs/cctv_dgp_detail_skip_v23_loss_audit_v1/independent_saved_loss_audit.json')
    v24=read(ROOT/'outputs/cctv_dgp_degraded_detail_v24_preparation/independent_preparation_audit.json')
    tests=read(ROOT/'outputs/cctv_dgp_degraded_detail_v24_preparation/pure_objective_tests.json')
    shell=read(ROOT/'outputs/cctv_dgp_degraded_detail_v24_preparation/shell_and_loss_contract_receipt.json')
    require(audit['complete'] and audit['VM_failure_present'] and not audit['early_structure_stop']['pass'] and
        visual['complete'] and loss['complete'] and v24['complete'] and tests['tests_passed']==8 and shell['Bash_syntax_exit_code']==0,'Missing audit/review/transfer checks')
    docs_before={}
    for name in DOCS:
        shutil.copy2(ROOT/name,OUT/'before_docs'/name);docs_before[name]=sha(OUT/'before_docs'/name)
    plan={'date':'2026-10-06','scope':'Audited V23 failure and new V24 objective transfer only; Goal incomplete',
        'runner_sha256':sha(Path(__file__)),'previous_milestone_sha256':PREVIOUS_SHA,'app_record_sha256':APP_SHA,
        'before_docs_sha256':docs_before,'previous644_document_locations':mapping,'older560_document_locations':previous['previous_document_locations'],
        'local_training_calls':0,'VM_actions':False,'goal_status':'active','goal_complete':False}
    write(OUT/'plan.json',plan)
    write(OUT/'before_update_audit.json',{'complete':True,'previous644_bindings_verified':644,'older560_bindings_verified':560,
        'app22_bindings_verified':22,'documents_preserved':docs_before,'seconds':time.monotonic()-started})
    for name in DOCS:
        old=(OUT/'before_docs'/name).read_bytes();split=old.index(b'\n')+1
        prefix=CLOSED if name=='CCTV_DGP_DETAIL_SKIP_V23_VM.md' else STATUS
        (ROOT/name).write_bytes(old[:split]+b'\n'+prefix.encode('utf-8')+b'\n'+old[split:])
        new=(ROOT/name).read_bytes();require(new.startswith(old[:split]) and new.endswith(old[split:]),'Historical body changed')
    paths=set()
    for folder in ['outputs/cctv_dgp_detail_skip_v23_return','outputs/cctv_dgp_detail_skip_v23_return_audit_preparation',
                   'outputs/cctv_dgp_detail_skip_v23_diagnostic','outputs/cctv_dgp_detail_skip_v23_loss_audit_v1',
                   'outputs/cctv_dgp_degraded_detail_vm_v24','outputs/cctv_dgp_degraded_detail_v24_preparation',
                   'outputs/dgp_detail_skip_v23_audit_and_degraded_detail_v24_milestone']:
        paths.update(f for f in (ROOT/folder).rglob('*') if f.is_file())
    for name in [*DOCS,'CCTV_DGP_DETAIL_SKIP_V23_RESULTS.md','CCTV_DGP_DEGRADED_DETAIL_V24_PLAN.md','CCTV_DGP_DEGRADED_DETAIL_V24_VM.md',
                 'scripts/import_cctv_dgp_detail_skip_v23.py','scripts/audit_cctv_dgp_detail_skip_v23.py',
                 'scripts/audit_cctv_dgp_detail_skip_v23_execution.py','scripts/diagnose_cctv_dgp_detail_skip_v23.py',
                 'scripts/verify_cctv_dgp_detail_skip_v23_diagnostic.py','scripts/audit_cctv_dgp_detail_skip_v23_loss_v1.py',
                 'scripts/verify_cctv_dgp_detail_skip_v23_loss_v1.py','scripts/prepare_cctv_dgp_degraded_detail_v24.py',
                 'scripts/cctv_dgp_degraded_detail_v24.py','scripts/cctv_dgp_degraded_objective_v24.py',
                 'scripts/verify_cctv_dgp_degraded_detail_v24_preparation.py','scripts/finalize_cctv_dgp_detail_v23_v24_milestone.py',
                 'scripts/verify_cctv_dgp_detail_v23_v24_milestone.py','tests/test_cctv_dgp_detail_skip_v23_return_audit.py',
                 'tests/test_cctv_dgp_degraded_objective_v24.py','outputs/cctv_dgp_detail_skip_v23_independent_audit.json',
                 'outputs/cctv_dgp_detail_skip_v23_return_import.json','outputs/cctv-dgp-detail-skip-v23-results.tar.gz',
                 'outputs/cctv-dgp-detail-skip-v23-results.tar.gz.sha256','outputs/cctv-dgp-detail-skip-v23-export.json',
                 'outputs/cctv-dgp-degraded-detail-v24-execution.tar.gz','outputs/cctv-dgp-degraded-detail-v24-execution.tar.gz.sha256']:
        paths.add(ROOT/name)
    bindings={f.relative_to(ROOT).as_posix():sha(f) for f in sorted(paths)}
    result={'complete':True,'date':'2026-10-06','scope':'V23 independently audited failure; V24 objectively distinct finite packet prepared, no Goal acceptance',
        'new_evidence_sha256':bindings,'new_bindings':len(bindings),'previous_milestone_sha256':PREVIOUS_SHA,
        'older_milestone_sha256':sha(OLDER),'app_record_sha256':APP_SHA,'previous644_document_locations':mapping,
        'older560_document_locations':previous['previous_document_locations'],'previous644_bindings_verified':644,'older560_bindings_verified':560,
        'app22_bindings_verified':22,'history_bodies_exact':4,'V23_return_sha256':'b71639cabd17476d5989dfd4582d47de9b747b8bf2a611232e5f62cc67bf9fd9',
        'V23_return_members':370,'V23_updates':50,'V23_backwards_VM':51,'V23_early_gain_percent':audit['early_structure_stop']['relative_feature_error_gain']*100,
        'V23_original_early_gate_pass':False,'V23_final800_gate_evaluated':False,'V23_cases_reviewed':50,'V23_visible_structure_gain':False,
        'V23_fixed_CPU_head_forwards':150,'V23_fixed_CPU_recognizer_forwards':320,
        'V23_loss_mismatch_independently_checked':True,'V23_clear_contribution_fraction':loss['clear_contribution_fraction_of_net_objective_reduction'],
        'V24_protocol_sha256':v24['protocol_sha256'],'V24_archive_sha256':v24['archive_sha256'],'V24_assets':221,'V24_regular_members':223,
        'V24_changes_objective_only':True,'V24_head_data_schedule_gates_budgets_unchanged':True,'V24_loss_invariants_passed':8,
        'V24_fixed_CPU_high_pass_calls':100,'V24_neural_forwards':0,'V24_VM_gradient_training_capacity_quality_verified':False,
        'input_usable_label_preserved':True,'model_changed_in_current_app':False,'app_design_source_changed':False,
        'local_optimizer_updates':0,'local_backward_calls':0,'assistant_VM_cloud_actions':0,
        'native_reserved_covering_COFW_test_pixels_enter_new_pilot':False,'historical_gates_waived':False,
        'historical_app_regressions_or_Playwright_rerun':False,'native_usefulness_qualified':False,'covering_families_qualified':False,
        'canonical_app_parity_complete_for_new_head':False,'independent_final_review_complete':False,
        'execution_preference':'Verified transfer files and exact pasteable VM commands only',
        'goal_status':'active','goal_complete':False,'seconds':time.monotonic()-started,
        'next':'Manual finite V24 on existing L4, then independent returned-source/receipt/output audit and all-case review. Failure closes this head sequence pending architecture/data/loss review.'}
    write(OUT/'milestone.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='new_evidence_sha256'},indent=2))


if __name__=='__main__':main()
