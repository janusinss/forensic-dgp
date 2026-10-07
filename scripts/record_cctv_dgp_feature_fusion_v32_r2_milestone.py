"""Record the independently checked pre-optimizer R1 failure and R2 routing repair."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_milestone'
PREVIOUS = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r1_milestone'
PACKET = ROOT / 'outputs/cctv_dgp_feature_fusion_vm_v32_r2'
PREP = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_preparation'
FAILED = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r1_failure_v1'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    assert not OUT.exists()
    previous = read(PREVIOUS/'milestone.json')
    for name,digest in previous['new_evidence_sha256'].items(): assert sha(ROOT/name)==digest,name
    old_closure = read(PREVIOUS/'independent_closure_audit.json')
    assert old_closure['complete'] and old_closure['milestone_sha256']==sha(PREVIOUS/'milestone.json')
    audit=read(PREP/'independent_audit_r1/independent_packet_audit.json')
    failure=read(FAILED/'host_access_retry/verification.json')
    p=read(PACKET/'protocol.json')
    assert audit['complete'] and audit['root_regressions_passed']==8 and audit['transfer_dispatch_regressions_passed']==6
    assert audit['unchanged_mathematical_training_behavior'] and failure['complete'] and failure['optimizer_updates']==0
    assert audit['protocol_sha256']==sha(PACKET/'protocol.json') and p['routing_revision']==2
    addition=f'''
**Latest repair milestone - 7 October 2026: R1 stopped before any optimizer; use the verified V32 r2 routing repair.**

The human-launched V32 r1 completed its initial gradient preflight, then failed
at the cache loader's exact directory guard. Its worker required the R1 directory
but the copied cache still required the original V32 directory. This was an agent
packaging error. It is not a learned-structure gate failure and supplies no improved
output evidence. R1 is retained unchanged; do not rerun that broken packet.

A bounded read-only maintenance SSH fetch verifies the original protocol/source
hashes, failure, gradient-preflight receipt, trainer log/exit, supervision and
export manifest/sidecars. The VM archive hash is
599d84dd2b84fdc05e55b774d115951d251e4013ae777d7f023791bdb60f59e2,
169,708,697 bytes. There were zero optimizer updates, no constructed optimizer,
no new checkpoint and no results.json. Export complete:true means packaging only.
The full archive download and independent audit of its saved gradient values
remain pending; the small original receipts are verified separately. The first
sandbox-denied SDK access is preserved; authorized host access succeeded with
zero VM writes, model/gradient calls or training launch by the agent.

The pre-import R1 path mismatch is reproduced with actual guard AST, using only
simulated platform/home metadata. V32 r2 routes both guards to its distinct root
and calls the same exact cache guard during --verify-transfer and before run.
The cache's Linux/exact-root restrictions are preserved. Eight actual guard
regressions and six actual transfer-dispatch regressions pass, including valid
Linux routing, old/base/nested/outside roots, Windows and incorrect parent.
Transfer dependency checks in those regressions are explicit metadata stubs;
they do not establish real CUDA/data preflight or trained usefulness.

The first local transfer test fixture omitted assets_sha256 and failed at the
status print after valid checks. That verifier/error are retained. A separate
audit corrects only the mock metadata field; the R2 packet remains unchanged.
All11 archive members, source/protocol equivalence, Python3.10 syntax, Windows
pre-neural rejection, prospective return schema and the failed1% receipt pass.
The unchanged candidate/training/shell inherit11 prior core regressions, four
exact raw/PNG CPU parity cases and the original read-only Bash parser result.
No new neural, local autograd or optimizer call is performed for this repair.

V32 r2 protocol SHA256: {audit['protocol_sha256']}
V32 r2 execution archive SHA256: {audit['archive_sha256']}
Execution archive:758,053 bytes,11 regular files.

This is the same justified fresh-original fusion/decoder experiment:23 tensors,
978,243 parameters,781 TRAIN references/3,905 cases, frozen schedule, seven losses,
normalizers and AdamW unchanged. All scientific gates remain fixed: maximum800
updates; early50 gain>=1%; final gain>=10%, all17 preservation groups, both source
gains>=0 and mean-only fraction<=20%. Original checkpoint, frozen backbone/head4
and all normalization buffers remain protected. No failed learned recipe or
stopped candidate is resumed. V31's50-update/0.694525% failed gate and every older
checkpoint, split, failure and provenance record remain binding.

Actual new training remains the human's manual upload/SSH/tmux workflow on the
existing L4/g2-standard-4 at ~/forensic-dgp. Require8GiB free and an idle GPU.
Estimate15-35 minutes plus export, with unchanged enforced preflight300s,
cache900s, fit3600s, worker4500s/external4800s+30s, export900s/external930s+30s,
VRAM20GiB and return3GiB limits. This agent has installed/launched no R2 workload.

The original own-DGP app checkpoint/design and actual Windows research-cache
backup remain unchanged. Useful native restoration, all seven completion-family
automatic/assisted quality reviews and independent final review remain pending.
No new development/native/reserved pixels are used. No ethnicity, hidden identity
or Zamboanga performance inference is made. Goal active/incomplete.

[V32 r2 five manual steps and separate R1 failure download commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FEATURE_FUSION_V32_R2_VM.md>)
[Verified original R1 failure receipts](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_feature_fusion_v32_r1_failure_v1/host_access_retry/verification.json>)

The complete prior handoff is preserved below. Its R1 transfer recommendation
and unlaunched wording are historical and superseded by this verified repair.

'''
    OUT.mkdir(); before_dir=OUT/'before_docs';before_dir.mkdir()
    handoff=ROOT/'PROJECT_HANDOFF.md';before=handoff.read_bytes()
    with (before_dir/handoff.name).open('xb') as stream:stream.write(before)
    split=before.index(b'\n')+1
    handoff.write_bytes(before[:split]+addition.encode('utf-8')+before[split:])
    files=[Path(__file__),ROOT/'scripts/verify_cctv_dgp_feature_fusion_v32_r2_milestone.py',
           ROOT/'scripts/prepare_cctv_dgp_feature_fusion_v32_routing_r2.py',
           ROOT/'scripts/verify_cctv_dgp_feature_fusion_v32_routing_r2.py',
           ROOT/'scripts/verify_cctv_dgp_feature_fusion_v32_routing_r2_audit_r1.py',
           ROOT/'scripts/check_cctv_dgp_feature_fusion_v32_root_routing.py',
           ROOT/'scripts/read_cctv_dgp_feature_fusion_v32_r1_failure_v1.py',
           ROOT/'scripts/audit_cctv_dgp_feature_fusion_v32_r2_return.py',
           ROOT/'scripts/cctv_dgp_feature_fusion_v32_r2_return_audit_template.py',
           ROOT/'CCTV_DGP_FEATURE_FUSION_V32_R2_VM.md',
           ROOT/'outputs/cctv-dgp-feature-fusion-v32-r2-execution.tar.gz',
           ROOT/'outputs/cctv-dgp-feature-fusion-v32-r2-execution.tar.gz.sha256',
           handoff,before_dir/handoff.name,PREVIOUS/'milestone.json',
           PREVIOUS/'independent_closure_audit.json',PREVIOUS/'final_readback.json']
    for folder in [PACKET,PREP,FAILED,ROOT/'outputs/cctv_dgp_feature_fusion_v32_routing_regression_v1']:
        files.extend(f for f in sorted(folder.rglob('*')) if f.is_file())
    m={'complete':True,'datetime_UTC':datetime.now(timezone.utc).isoformat(),
       'recorder_sha256':sha(Path(__file__)),'previous_milestone_sha256':sha(PREVIOUS/'milestone.json'),
       'new_evidence_sha256':{f.relative_to(ROOT).as_posix():sha(f) for f in files},
       'previous_document_locations':{'PROJECT_HANDOFF.md':(before_dir/handoff.name).relative_to(ROOT).as_posix()},
       'document':{'name':handoff.name,'before_path':(before_dir/handoff.name).relative_to(ROOT).as_posix(),
                   'before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':sha(handoff),
                   'addition_bytes':len(addition.encode('utf-8')),'full_previous_body_preserved':True},
       'protocol_sha256':audit['protocol_sha256'],'archive_sha256':audit['archive_sha256'],'routing_revision':2,
       'R1_original_failure_receipts_verified':True,'R1_optimizer_updates':0,'R1_full_gradient_audit_pending':True,
       'root_and_transfer_regressions_passed':14,'scientific_behavior_unchanged':True,
       'manual_VM_required':True,'actual_VM_training_started_by_agent':False,
       'local_neural_calls':0,'local_gradient_calls':0,'local_optimizer_updates':0,
       'app_changes':False,'native_or_reserved_used':False,'quality_qualification':False,
       'app_promotion':False,'goal_status':'active','goal_complete':False}
    with (OUT/'milestone.json').open('x',encoding='utf-8',newline='\n') as stream:json.dump(m,stream,indent=2)
    print(json.dumps({'complete':True,'new_bindings':len(m['new_evidence_sha256']),
                      'milestone_sha256':sha(OUT/'milestone.json'),'manual_VM_required':True},indent=2))


if __name__=='__main__':
    main()
