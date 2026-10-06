"""Preserve reported V24 failure/status and documents; no VM/model training work."""
import ast
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from import_cctv_dgp_degraded_detail_v24 import read,require,sha,write

OUT=ROOT/'outputs/cctv_dgp_degraded_detail_v24_reported_stop_v1'
PREP=ROOT/'outputs/cctv_dgp_degraded_detail_v24_return_audit_preparation'
PREVIOUS=ROOT/'outputs/dgp_detail_skip_v23_audit_and_degraded_detail_v24_milestone/milestone.json'
DOCS=['PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md',
      'CCTV_DGP_DEGRADED_DETAIL_V24_VM.md','CCTV_DGP_DEGRADED_DETAIL_V24_PLAN.md']
SOURCES=['scripts/import_cctv_dgp_degraded_detail_v24.py','scripts/audit_cctv_dgp_degraded_detail_v24.py',
         'scripts/audit_cctv_dgp_degraded_detail_v24_execution.py','scripts/audit_cctv_dgp_degraded_detail_v24_cohort.py',
         'tests/test_cctv_dgp_degraded_detail_v24_return_audit.py']
STATUS='''**Latest research status — 6 October 2026: V24 user-reported early structure failure; return audit pending.**

The pasted L4 log reports successful preflight, exact four-case original-DGP
CUDA parity and 50 initial cached-DGP outputs. At update50, delivered degraded
feature error changes from 0.0019965976532523044 to 0.001996531311299121:
0.0033227502% improvement, below the unchanged 1% early requirement. Trainer
exit1 retains that stop. Export exit0 and complete:true confirm failure packaging;
run_results_present:false/failure_present:true. No final800 capacity test executes.

Reported return: 76,492,526 bytes, SHA256
`1d758362366f3e1f17ddf8239b0f41940b6b7c6ebf349a86ed76e0abdf63dd15`.
The three return files are not yet present locally. Actual source/data/gradient/
timing/state/cohort/metric audit and every-case image review remain pending.
Do not infer visible usefulness from the tiny metric improvement or console log.
Use step5 in the V24 runbook: three separate Windows gcloud downloads.
Steps1–4 are historical for this stopped recipe; do not rerun them or relax gates.

V22, V23 and V24 have missed the same early structure requirement. The assumption
that the tested small high-frequency correction on frozen own-DGP outputs can
deliver enough facial structure has not held in these finite pilots. Stop this
head-recipe sequence, audit returned V24 evidence, and review architecture/data/
loss before another recipe. No fourth pilot or training fix is prepared.
The user confirms “all are important”: eyes, nose, mouth, face outline and overall
visible appearance remain in scope together. No region-only acceptance or focus.

The V24 return importer/auditor is prepared with 18 passing tamper/schema/gate
regressions. The unchanged full old-auditor logic and replay tolerances are
checked by AST comparison; new cohort checks independently validate 50 baseline
rows, ten clear controls, forty degraded rows, policy and exact float32 means.
A fixed CPU contract uses100 high-pass calls, no head/DGP/recognizer prediction,
backward or optimization. These are checker tests, not real V24 audit evidence.
An initial auditor-only stale-bundle path failure and its correction are preserved;
the VM runtime, protocol, original checkpoints and all quality gates are unchanged.

Original documents and the concurrent completed-storage-maintenance entry are
preserved before this update. The DGP-led app/design remains unchanged; no new
local training, assistant VM/cloud action, app promotion or browser rerun. Full
native usefulness, canonical app parity, all seven automatic/assisted covering
families and independent final review remain required. Native remains unpaired;
paired TRAINING metrics are separate. No ethnicity or Zamboanga performance claim.
The previously useful CCTV input remains usable despite model softness. Goal active.

[V24 downloads](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DEGRADED_DETAIL_V24_VM.md>) ·
[V24 audit preparation](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_degraded_detail_v24_return_audit_preparation/plan.json>)

Previous entries below are preserved history; V24 pending-training statements
are superseded by this reported stopped run, not by an independent quality pass.
'''


def main():
    started=time.monotonic();require(not OUT.exists(),'Preserve previous status record')
    require(sha(PREVIOUS)=='40f159663d100186dd285ff318680b6201fafb16e38f1b072bb785760ebf94d9','Previous milestone changed')
    previous=read(PREVIOUS);mapping={}
    old_handoff=ROOT/'outputs/cctv_dgp_vm_storage_cleanup_20261006_v1/before_docs_r1/PROJECT_HANDOFF.md'
    require(sha(old_handoff)==previous['new_evidence_sha256']['PROJECT_HANDOFF.md'],'Concurrent research handoff backup differs')
    for name,digest in previous['new_evidence_sha256'].items():
        path=old_handoff if name=='PROJECT_HANDOFF.md' else ROOT/name
        require(sha(path)==digest,'Previous664 binding changed: '+name)
    app=read(ROOT/'outputs/dgp_app_v3_integration_record.json')
    for name,digest in {**app['sources_sha256'],**app['evidence_sha256']}.items():require(sha(ROOT/name)==digest,'App binding changed: '+name)
    testlog=(PREP/'tests_r1.log').read_text(encoding='utf-8-sig')
    require('Ran 18 tests' in testlog and testlog.rstrip().endswith('OK'),'Return-auditor checks have not passed')
    contract=read(PREP/'cohort_CPU_contract_check.json')
    require(contract['complete'] and not contract['real_V24_return_audited'] and
            contract['result']['maximum_independent_baseline_row_difference']==0,'Fixed cohort contract differs')
    for name in SOURCES:ast.parse((ROOT/name).read_text(),feature_version=(3,10))
    OUT.mkdir();(OUT/'before_docs').mkdir()
    before={}
    for name in DOCS:shutil.copy2(ROOT/name,OUT/'before_docs'/name);before[name]=sha(OUT/'before_docs'/name)
    reported={'date':'2026-10-06','scope':'User-pasted VM console evidence only; independent V24 return audit pending',
        'protocol_sha256':'76ab24695a40b411832e2d678c79b0e2a80f6320d4525970b2dd32db2379e114',
        'preflight_reported_pass':True,'reported_initial_head_state':'98d24e4b6c9e7b8d960391c63fb859bf19d6d8979573a16938442f76c3f7885c',
        'reported_recognizer_state':'9ee66c2faefe0b82224c2cea6811073fbf70e8036808e293878109f078ea3963',
        'reported_original_DGP_state':'d89f4e52777b5260e94a3e533aa21a0204cbd79b578137881fc6a4c67f1bb0c3',
        'reported_fresh_DGP_parity_errors':[0.0,0.0,0.0,0.0],'reported_initial_exact_cases':50,
        'reported_updates':50,'reported_baseline_feature_MSE':.0019965976532523044,
        'reported_update50_feature_MSE':.001996531311299121,'relative_gain_from_reported_numbers':1-.001996531311299121/.0019965976532523044,
        'reported_minimum_gain':.01,'reported_early_gate_pass':False,
        'reported_assertion':'No one-percent early structural gain; retain stop','reported_trainer_exit':1,'reported_export_exit':0,
        'reported_export_complete':True,'reported_return_sha256':'1d758362366f3e1f17ddf8239b0f41940b6b7c6ebf349a86ed76e0abdf63dd15',
        'reported_return_bytes':76492526,'reported_export_seconds':5.132257421999839,
        'reported_run_results_present':False,'reported_failure_present':True,
        'user_structure_preference':'all are important','actual_return_audited':False,'visible_V24_quality_reviewed':False,
        'next':'Download and independently audit/review saved V24 evidence; architecture/data/loss discussion before another recipe',
        'goal_complete':False}
    write(OUT/'reported_execution.json',reported)
    write(OUT/'before_update_plan.json',{'date':'2026-10-06','before_docs_sha256':before,'previous_milestone_sha256':sha(PREVIOUS),
        'previous664_handoff_location':old_handoff.relative_to(ROOT).as_posix(),'app_record_sha256':sha(ROOT/'outputs/dgp_app_v3_integration_record.json'),
        'runner_sha256':sha(Path(__file__)),'local_training_or_VM_calls':0,'preserve_concurrent_maintenance':True})
    for name in DOCS:
        require(sha(ROOT/name)==before[name],'Concurrent document update; preserve instead of overwriting: '+name)
        old=(OUT/'before_docs'/name).read_bytes();split=old.index(b'\n')+1
        (ROOT/name).write_bytes(old[:split]+b'\n'+STATUS.encode('utf-8')+b'\n'+old[split:])
        new=(ROOT/name).read_bytes();require(new.startswith(old[:split]) and new.endswith(old[split:]),'Historical body changed')
    preparation={'complete':True,'scope':'Prepared V24 independent return checker; no real returned-data audit yet',
        'sources_sha256':{name:sha(ROOT/name) for name in SOURCES},'original_plan_sha256':sha(PREP/'plan.json'),
        'tests_passed':18,'tests_log_sha256':sha(PREP/'tests_r1.log'),'fixed_CPU_contract_sha256':sha(PREP/'cohort_CPU_contract_check.json'),
        'original_auditor_AST_and_quality_tolerances_exact':True,'new_cohort_scalar_tolerance':2e-9,
        'actual_V24_return_audited':False,'no_new_training_recipe':True,'head_DGP_recognizer_forwards':0,'fixed_CPU_high_pass_calls':100,
        'optimizer_updates':0,'backward_calls':0,'VM_actions':False,'goal_complete':False}
    write(PREP/'preparation_receipt.json',preparation)
    paths=set(ROOT/name for name in DOCS+SOURCES+['scripts/record_cctv_dgp_degraded_detail_v24_reported_stop.py'])
    paths.update(f for f in OUT.rglob('*') if f.is_file());paths.update(f for f in PREP.rglob('*') if f.is_file())
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}
    write(OUT/'status_update_receipt.json',{'complete':True,'scope':'Reported stop, audit-tool preparation and document preservation only',
        'bindings_sha256':bindings,'previous664_bindings_verified':664,'app22_bindings_verified':22,'history_bodies_exact':5,
        'concurrent_storage_maintenance_entry_preserved':True,'training_runtime_protocol_checkpoints_and_gates_unchanged':True,
        'actual_V24_return_audited':False,'no_fourth_training_recipe':True,'user_structure_preference':'all are important',
        'goal_status':'active','goal_complete':False,'local_backward_optimizer_or_VM_calls':0,'seconds':time.monotonic()-started})
    print(json.dumps({'reported_feature_gain_percent':reported['relative_gain_from_reported_numbers']*100,
        'minimum_percent':1,'gate_pass':False,'return_auditor_tests':18,'real_return_audit_pending':True,
        'documents_preserved':5,'bindings':len(bindings),'goal_complete':False},indent=2))


if __name__=='__main__':main()
