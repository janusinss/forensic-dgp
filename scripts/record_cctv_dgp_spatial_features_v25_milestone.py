"""Bind Route-A packet and preserve all previous research and app history."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/dgp_spatial_features_v25_preparation_milestone'
PREVIOUS=ROOT/'outputs/dgp_degraded_detail_v24_audit_and_architecture_milestone/milestone.json'
DOCS=['PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md',
    'CCTV_DGP_DEGRADED_DETAIL_V24_PLAN.md','CCTV_DGP_DEGRADED_DETAIL_V24_VM.md']
SOURCES=['scripts/cctv_dgp_spatial_features_v25.py','scripts/cctv_dgp_spatial_features_v25_prepare.py',
    'scripts/prepare_cctv_dgp_spatial_features_v25.py','scripts/cctv_dgp_spatial_features_v25_vm.py',
    'scripts/verify_cctv_dgp_spatial_features_v25_interface.py','scripts/audit_cctv_dgp_spatial_features_v25_interface.py',
    'scripts/test_cctv_dgp_spatial_features_v25_contracts.py','scripts/verify_cctv_dgp_spatial_features_v25_execution.py',
    'scripts/import_cctv_dgp_spatial_features_v25.py','scripts/audit_cctv_dgp_spatial_features_v25_transfer_preparation.py',
    'tests/test_cctv_dgp_spatial_features_v25_transfer.py','scripts/record_cctv_dgp_spatial_features_v25_milestone.py',
    'scripts/verify_cctv_dgp_spatial_features_v25_milestone.py']
LATEST='''**Latest research milestone — 6 October 2026: selected own-DGP spatial path; V25 transfer verified, manual L4 run pending.**

The downloaded V24 failure is independently audited and all50 exposed training
cases reviewed. Its0.0033227502% delivered structure gain misses the unchanged1%
early stop. Preserve/close V22–V24; no unchanged rerun or failed-gate waiver.
After the architecture discussion, the user selects Route A: revise our DGP's
spatial/feature path. All visible facial features remain required together.

V25 introduces a53,781-parameter decoder using our frozen DGP's five FPN maps
and the full-resolution camera image. Original DGP weights/normalization remain
frozen. It removes the failed final output Gaussian high-pass, retaining observed
RGB mean removal/support masking. This is a new own-model spatial-path capacity
hypothesis, not proven learning, a pretrained restorer replacement or app adoption.

Packet218,140,185 bytes,235 assets/237 regular archive members; SHA256
`7652fa82a95d218de18c31d943114124bfaf9299e6492e8c79da710b59831034`;
protocol`ed43355b1fa0bd768d78e4e2b9ed3bd32cc046fe6221629605899a6a9b2e3175`.
All221 original assets, same50cases/10references/800updates/80epochs, optimizer,
schedule, corrected V24 objective, quality guards and finite timing stops stay
exact. All50 fresh CUDA baseline comparisons, four CPU feature comparisons,
250 frozen feature arrays and actual update2 gradients are prospectively required.
Intermediate CPU/CUDA numerical bounds do not relax quality or qualify the app.

Local interface verification uses4 frozen DGP/5 head forwards, no backward or
optimization: initial fresh CPU baseline is exact and disposable smooth-camera/
feature links reach output. A separate saved-array audit passes. Twelve feature
failure checks and ten safe-return transfer tests pass. Independent archive/
source/Python3.10/Bash/actual Windows transfer and training-host guards pass.
Preparation failures are retained. Real L4 features/gradients/timing/learning and
independent returned-output audit/whole-face review remain pending. complete:true
export packages evidence; it does not certify training success or usefulness.

Manual upload/install/tmux/launch/download commands are frozen in the V25 runbook;
Windows PuTTY downloads use three separate remote-source calls. No assistant VM/
cloud action or local training occurs. Prior434 V24 closure bindings and app22
bindings, original checkpoints/splits/failures and the concurrent completed VM
storage-maintenance entry remain preserved. Historical34 app regressions/bundled
inline Playwright checks are retained; the unchanged app requires no new run.

Native remains unpaired; these paired photograph TRAINING metrics stay separate.
No validation/reserved-final/native/new covering pixels enter V25; no ethnicity
or Zamboanga performance claim. Useful native development output and canonical
local-app inference, insufficient-information handling, all seven automatic/
assisted covering families and independent final review remain required. The
previously useful CCTV crop stays usable despite softness. Goal active/incomplete.

[V25 finite design](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_SPATIAL_FEATURES_V25_PLAN.md>) ·
[V25 manual tmux commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_SPATIAL_FEATURES_V25_VM.md>) ·
[Milestone](<C:/xampp/htdocs/YEAR 4/Testing/outputs/dgp_spatial_features_v25_preparation_milestone/milestone.json>)

Previous bodies below are preserved history. Their pending-return/no-new-packet
statements are superseded by the audited V24 closure and selected V25 preparation;
old V22–V24 launch commands remain historical, not instructions to repeat them.
'''


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):
    with p.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(x,indent=2,allow_nan=False)+'\n')


def main():
    start=time.monotonic();assert not OUT.exists(),'Keep previous milestone'
    assert sha(PREVIOUS)=='9b9ddb27d424a6d00b143728bb52f65212a25ee7d81ce64f029e276c22124b5e'
    previous=read(PREVIOUS)
    for n,d in previous['new_evidence_sha256'].items():assert sha(ROOT/n)==d,n
    prep=ROOT/'outputs/cctv_dgp_spatial_features_v25_preparation'
    packet=read(prep/'preparation.json');audit=read(prep/'independent_execution_audit.json')
    assert packet['complete'] and audit['complete'] and packet['protocol_sha256']==audit['protocol_sha256'] and audit['original_V24_assets_unchanged']==221
    transfer=read(prep/'independent_transfer_preparation_audit.json');assert transfer['complete'] and transfer['transfer_regressions_verified']==10
    interface=read(ROOT/'outputs/cctv_dgp_spatial_features_v25_interface/results.json');assert interface['complete'] and not interface['learned_capacity_or_quality_proven']
    decision=read(ROOT/'outputs/cctv_dgp_post_v24_architecture_review_v1/user_architecture_decision.json');assert decision['selected_route']=='A'
    app_path=ROOT/'outputs/dgp_app_v3_integration_record.json';app=read(app_path)
    for n,d in {**app['sources_sha256'],**app['evidence_sha256']}.items():assert sha(ROOT/n)==d,n
    for n in SOURCES:ast.parse((ROOT/n).read_text(encoding='utf-8'),feature_version=(3,10))
    OUT.mkdir();(OUT/'before_docs').mkdir()
    before={}
    for n in DOCS:
        shutil.copy2(ROOT/n,OUT/'before_docs'/n);before[n]=sha(OUT/'before_docs'/n)
    assert b'VM storage cleanup completed; future VM use preserved' in (OUT/'before_docs/PROJECT_HANDOFF.md').read_bytes()
    originals={n:(OUT/'before_docs'/n).relative_to(ROOT).as_posix() for n in DOCS}
    write(OUT/'before_update_plan.json',{'date':'2026-10-06','before_docs_sha256':before,
        'previous_milestone_sha256':sha(PREVIOUS),'previous434_original_locations':originals,'app_record_sha256':sha(app_path),
        'runner_sha256':sha(Path(__file__)),'VM_training_pending':True})
    for n in DOCS:
        assert sha(ROOT/n)==before[n],'Concurrent document update: '+n
        old=(OUT/'before_docs'/n).read_bytes();split=old.index(b'\n')+1
        (ROOT/n).write_bytes(old[:split]+b'\n'+LATEST.encode('utf-8')+b'\n'+old[split:])
        assert (ROOT/n).read_bytes().endswith(old[split:])
    paths={ROOT/n for n in DOCS+SOURCES+['CCTV_DGP_SPATIAL_FEATURES_V25_PLAN.md','CCTV_DGP_SPATIAL_FEATURES_V25_VM.md',
        'outputs/cctv-dgp-spatial-features-v25-execution.tar.gz','outputs/cctv-dgp-spatial-features-v25-execution.tar.gz.sha256']}
    for folder in ['cctv_dgp_spatial_features_vm_v25','cctv_dgp_spatial_features_v25_interface',
        'cctv_dgp_spatial_features_v25_contract_regressions','cctv_dgp_spatial_features_v25_preparation',
        'cctv_dgp_spatial_features_v25_interface_preparation_failure_01','cctv_dgp_spatial_features_v25_packet_preparation_failure_01',
        'cctv_dgp_spatial_features_v25_execution_checker_failure_01']:
        paths.update(p for p in (ROOT/'outputs'/folder).rglob('*') if p.is_file())
    paths.update(p for p in OUT.rglob('*') if p.is_file())
    new={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}
    record={'complete':True,'date':'2026-10-06','scope':'Selected Route-A finite manual-L4 packet verified; no trained-quality or Goal completion',
        'new_evidence_sha256':new,'previous_milestone_sha256':sha(PREVIOUS),'previous434_original_locations':originals,
        'app_record_sha256':sha(app_path),'selected_route':'A','protocol_sha256':packet['protocol_sha256'],
        'archive_sha256':packet['archive_sha256'],'archive_bytes':packet['archive_bytes'],'parameter_count':53781,
        'previous_failed_recipes_preserved':True,'quality_gates_unchanged':True,'same_exposed_cases_schedule_objective':True,
        'capacity_or_usefulness_pass':False,'actual_V25_return_present':False,'VM_run_pending':True,
        'local_interface_counts':interface['counts'],'local_backward_calls':0,'local_optimizer_updates':0,
        'feature_regressions':12,'transfer_regressions':10,'app_promotion':False,'VM_actions':False,
        'goal_status':'active','goal_complete':False,'seconds':time.monotonic()-start}
    write(OUT/'milestone.json',record);print(json.dumps({k:v for k,v in record.items() if k!='new_evidence_sha256'},indent=2));print('Bound files:',len(new))


if __name__=='__main__':main()
