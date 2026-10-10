"""Retain the complete preceding handoff and record audited V34/manual V35 R1."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/cctv_dgp_v34_return_v35_probe_milestone'
PREVIOUS=ROOT/'outputs/cctv_dgp_v33_return_v34_diagnostic_milestone'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    assert not OUT.exists()
    old=read(PREVIOUS/'milestone.json')
    assert read(PREVIOUS/'independent_closure_audit.json')['complete']
    for name,digest in old['new_evidence_sha256'].items():assert sha(ROOT/name)==digest,name
    audit=read(ROOT/'outputs/cctv_dgp_group_guard_grad_v34_independent_audit.json')
    analysis=read(ROOT/'outputs/cctv_dgp_group_guard_grad_v34_analysis_v1_r1/independent_analysis_audit.json')
    prep=read(ROOT/'outputs/cctv_dgp_group_guard_probe_v35_r1_preparation/independent_packet_audit.json')
    assert audit['complete'] and audit['diagnostic_complete'] and audit['members_verified']==230
    assert analysis['complete'] and analysis['group_rows_independently_reassembled']==102
    assert prep['complete'] and prep['all108_projection_rows_verified'] and prep['VM_launches']==0
    bundle=ROOT/'outputs/cctv_dgp_group_guard_probe_v35_r1_vm'
    assert not (bundle/'outputs').exists()
    addition='''
**Research milestone - 8 October 2026: V34 diagnostic audited; V35 R1 finite image probe prepared.**

V34 completed300 per-case original-state non-hinged preservation gradient
queries in52.19seconds on the existing L4. Optimizer/parameter updates, epochs,
backwards and new checkpoints are zero. The independent audit verifies230
archive files and all100 raw outputs: exact parity with audited V33 original
outputs. It separately replays280 frozen CPU outputs within original raw/PNG/
embedding/component tolerances. Audit375.15seconds; no local gradients or
optimizer updates. The original DGP and fixed recognizer states remain intact.

All51 raw source/profile preservation directions are nonzero in each matched
TRAIN subset. The old restoration displacement predicts worsening in12
ArcFace checks per subset, including the aggregate; MSE and SSIM directions
improve. V34 exposes a protected-function conflict absent from the zero-at-
original hinges. These raw batchmatched derivatives are not delivered PNG
preservation, restored identity or native CCTV usefulness.

The independently verified new mathematical direction projects the mean of
the two audited original-state V33 displacements against all102 group guard
rows plus six nonzero existing restoration-loss rows. It retains99.1017% of
parameter-displacement magnitude; this is not a quality gain. Independent
group assembly, full-row KKT and primal checks pass. Float32 copies produce
small positive linear changes, maximum0.000000014073; finite outputs must
still pass the unchanged preservation gates. The mean of displacements is
not an AdamW step on a mean gradient. Both subsets now guide TRAIN design;
unexposed refers only to the first50 V32 updates, not held-out DEV/final.

The first analysis's NumPy-int64 JSON failure and all numeric artifacts are
retained. Serialization-only R1 is verified numerically hash-identical.
The first unrun V35 draft's two stale prospective replay paths are also
retained; R1 restores the validated V27/V28 basis. No recipe or gate changed.

V35 R1 is a new, unrun finite image probe: four disposable directions at
scales1,1/2,1/4,1/8, each reset to original DGP,100 baseline plus400 trial
outputs. Zero optimizer updates, new gradients, committed trajectory updates
or epochs. It creates no checkpoint, resumes no failed50 state and runs no
historical pilot. All17 PNG group, source, brightness and full-corpus1%-at50/
10%-at800 gates remain. A subset probe cannot qualify training capacity.

The9-file12,146,657-byte R1 packet passes independent230 returned/19 local
bindings,108-row geometry, Python3.10/read-only Bash syntax, wrong host/root/
platform, Windows pre-neural and unsafe-return checks. Protocol SHA256:
ba359d8aa6b3cd6c32b3e8f1c5d459b0af40751414d9f55a9261f3055d44b68b.
Archive SHA256:
954d81811e6881612787608086f27cceb96fa86888a49ee6f52537ca4d3b7ca3.
Use only the R1 manual Google Cloud SDK/tmux steps. Require idle running
L4/g2-standard-4 and6GiB free. Estimate2-6minutes plus1-3minutes export;
worker900s/external930s plus30s grace, export300s/external330s plus30s,
allocated VRAM20GiB and uncompressed return768MiB. No VM connection,
start, cleanup or new actual training occurred in this milestone.

The own-DGP-led Auto/override app and all14 current bindings remain unchanged.
No new native, paired DEV or reserved-final pixels were opened. V34/V35 are
paired photographic TRAIN evidence; source labels imply neither ethnicity
nor Zamboanga performance. Native unpaired evidence and synthetic DEV/final
remain separate. Existing functional inline Playwright evidence remains;
negative automatic and assisted covering-family reviews remain binding.
Useful native structure, all seven covering families, independent final
review and the complete DGP-led app scope remain outstanding. Request
clearer/less-covered input when usable structure is insufficient. Goal active.

[V34 findings and retained failures](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_GROUP_GUARD_GRAD_V34_RESULTS.md>)
[V35 R1 five exact manual steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_GROUP_GUARD_PROBE_V35_R1_VM.md>)

The entire previous handoff follows. Its V34-unrun statements are historical
and superseded by this verified return. Its prior failed restoration and
covering-family quality decisions remain current.

'''
    OUT.mkdir();(OUT/'before_docs').mkdir()
    handoff=ROOT/'PROJECT_HANDOFF.md';before=handoff.read_bytes()
    saved=OUT/'before_docs/PROJECT_HANDOFF.md';saved.write_bytes(before)
    at=before.index(b'\n')+1
    handoff.write_bytes(before[:at]+addition.encode('utf-8')+before[at:])
    files=[handoff,saved,ROOT/'CCTV_DGP_GROUP_GUARD_GRAD_V34_RESULTS.md',ROOT/'CCTV_DGP_GROUP_GUARD_PROBE_V35_R1_VM.md',
        ROOT/'CCTV_DGP_GROUP_GUARD_PROBE_V35_VM.md',PREVIOUS/'milestone.json',PREVIOUS/'independent_closure_audit.json',
        ROOT/'outputs/cctv_dgp_group_guard_grad_v34_return_import.json',ROOT/'outputs/cctv_dgp_group_guard_grad_v34_independent_audit.json']
    names=['supervise_cctv_dgp_v34_return_audit_v1.py','analyze_cctv_dgp_group_guard_grad_v34_v1.py',
        'analyze_cctv_dgp_group_guard_grad_v34_v1_r1.py','cctv_dgp_group_guard_cone_v35.py',
        'preserve_cctv_dgp_v34_analysis_failure_v1.py','verify_cctv_dgp_v34_group_analysis_v1_r1.py',
        'cctv_dgp_group_guard_geometry_v35.py','prepare_cctv_dgp_group_guard_probe_v35.py',
        'prepare_cctv_dgp_group_guard_probe_v35_r1.py','prepare_cctv_dgp_v35_prospective_auditor.py',
        'audit_cctv_dgp_group_guard_probe_v35_return.py','audit_cctv_dgp_group_guard_probe_v35_r1_return.py',
        'cctv_dgp_group_guard_probe_v35_vm.py','cctv_dgp_group_guard_probe_v35_r1_vm.py',
        'preserve_cctv_dgp_v35_draft_and_prepare_r1.py','verify_cctv_dgp_group_guard_probe_v35_r1_packet.py',
        'record_cctv_dgp_v34_return_v35_probe_milestone.py','verify_cctv_dgp_v34_return_v35_probe_milestone.py']
    files.extend(ROOT/'scripts'/name for name in names)
    for folder in ['cctv_dgp_group_guard_grad_v34_return','cctv_dgp_group_guard_grad_v34_audit_run_v1',
        'cctv_dgp_group_guard_grad_v34_analysis_v1','cctv_dgp_group_guard_grad_v34_analysis_v1_r1',
        'cctv_dgp_group_guard_probe_v35_vm','cctv_dgp_group_guard_probe_v35_preparation',
        'cctv_dgp_group_guard_probe_v35_r1_vm','cctv_dgp_group_guard_probe_v35_r1_preparation']:
        files.extend(q for q in sorted((ROOT/'outputs'/folder).rglob('*')) if q.is_file())
    files.extend(ROOT/'outputs'/('cctv-dgp-group-guard-grad-v34'+suffix) for suffix in ['-results.tar.gz','-results.tar.gz.sha256','-export.json'])
    for stem in ['cctv-dgp-group-guard-probe-v35','cctv-dgp-group-guard-probe-v35-r1']:
        files.extend(ROOT/'outputs'/(stem+suffix) for suffix in ['-execution.tar.gz','-execution.tar.gz.sha256'])
    evidence={q.relative_to(ROOT).as_posix():sha(q) for q in files}
    milestone={'complete':True,'UTC':datetime.now(timezone.utc).isoformat(),'new_evidence_sha256':evidence,
        'previous_milestone_sha256':sha(PREVIOUS/'milestone.json'),'previous_handoff_path':saved.relative_to(ROOT).as_posix(),
        'document':{'before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':sha(handoff),'addition_bytes':len(addition.encode('utf-8'))},
        'V34_return_verified':True,'V34_case_gradient_queries':300,'V34_parameter_updates':0,'V34_CPU_replays':280,
        'all108_math_constraints_verified':True,'V35_R1_packet_verified':True,'V35_R1_VM_started':False,
        'analysis_serialization_and_unrun_draft_failures_retained':True,'all14_app_bindings_unchanged':True,
        'local_autograd_or_optimizer_calls':0,'new_actual_training':False,'native_or_reserved_final_used':False,
        'covering_family_quality_failures_retained':True,'independent_final_quality_review':False,
        'app_promotion':False,'goal_status':'active','goal_complete':False}
    with (OUT/'milestone.json').open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(milestone,indent=2)+'\n')
    print(json.dumps({'complete':True,'bindings':len(evidence),'milestone_sha256':sha(OUT/'milestone.json'),'V35_R1_VM_started':False}))


if __name__=='__main__':main()
