"""Preserve completed V27 closure and publish only the selected review/manual proof."""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/dgp_original_decoder_review_gradient_v1_r2_milestone'
PREVIOUS=ROOT/'outputs/dgp_feature_skips_v27_audit_milestone/milestone.json'
DOCS=('PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md')


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def verify(bindings,mapping=None):
    for name,digest in bindings.items():
        path=(ROOT/(mapping or {}).get(name,name)).resolve()
        assert path.is_relative_to(ROOT) and sha(path)==digest,name
    return len(bindings)


def main():
    start=time.monotonic();assert not OUT.exists()
    assert sha(PREVIOUS)=='24b85fc75565b3f485a59e8a04c786685569a4cc9dc7aa48163d94b20af3c491'
    previous=read(PREVIOUS);assert verify(previous['new_evidence_sha256'])==668
    assert read(PREVIOUS.with_name('independent_readback.json'))['complete']
    review=read(ROOT/'outputs/cctv_dgp_original_decoder_review_v1/review.json')
    checked=read(ROOT/'outputs/cctv_dgp_original_decoder_review_v1/independent_readback.json')
    prep=ROOT/'outputs/cctv_dgp_original_decoder_gradient_v1_r2_preparation'
    packet=read(prep/'independent_packet_audit.json');tests=read(prep/'test_receipt.json')
    assert review['complete'] and checked['complete'] and review['raw_and_PNG_initial_parity_cases']==50
    assert packet['complete'] and packet['actual_Windows_execution_rejected_before_neural_work']
    assert tests['complete'] and tests['tests_passed']==8 and tests['bash_syntax_exit_code']==0
    app_path=ROOT/'outputs/dgp_app_v3_integration_record.json';app=read(app_path)
    assert sha(app_path)==previous['app_record_sha256']
    assert verify({**app['sources_sha256'],**app['evidence_sha256']})==22
    OUT.mkdir();before=OUT/'before_docs';before.mkdir();mapping={}
    prefix='''**Latest research milestone — 6 October 2026: selected original-decoder review verified; finite zero-update L4 proof R2 ready.**

The user selects review of a separate copy of the original DGP reconstruction
decoder after V25–V27 failed the unchanged early structure gate. V27 is fully
audited and visually reviewed: 0.221088578% gain versus 1%; its failed50/800
endpoint remains closed. No earlier recipe, checkpoint, split or gate changes.

The separate copy uses the unchanged DGPSynthesizer forward and enables only
head1–head4,smooth,smooth2,final:609,219 parameters in14 tensors. The2,703,488
remaining encoder/FPN parameters and every stored buffer stay frozen/evaluation.
All original shared FPN aliases remain internal; no tensor is shared with the
original model. Exact fresh-reference initial raw/PNG parity passes all50 exposed
TRAIN cases. CPU review21.631s:50 original/51 candidate forwards; no derivatives,
backwards, optimization or checkpoint writes. Actual local differentiation and
invalid-input attempts reject before neural work; partial support is preserved.
Independent251-binding source/layout readback passes. Clamp saturation0–3.1993%
of RGB component values is forward evidence, not a derivative or failure cause.

Only the new original-decoder gradient proof V1 R2 is ready for manual execution.
It is not a training pilot:10 five-case batches,70 component gradient queries,
zero optimizer construction/updates/epochs and no new checkpoint. Require finite
connected nonzero improvement gradients at all14 original decoder tensors;
all four initial preservation terms and every gradient must be exactly zero.
Same-batch fresh canonical baseline/candidate output must match exactly. Legacy
cached evidence remains separate. The seven objective formulas, original CPU-
built fixed filter and all retained capacity/preservation gates remain unchanged.
Save all ten7×609,219 matrices and their sum; no automatic follow-on or app promotion.

Unissued V1/R1 drafts are preserved. Source review corrected GPU kernel creation
and an erroneous shell substring deadline before release. R2's full AST/shell
comparison, Python3.10 parsing, actual Windows neural rejection, Bash syntax and
eight malformed scientific-return regressions pass. Synthetic arrays are not VM
evidence. The released13,816-byte/four-file packet uploads no data or weights.
Archive SHA256:f21634d909e880a26b8ae90e0e6393ff32fc7f2758ebb5312faf59707d82900d.
Protocol SHA256:81127e45a205c44ef685646a4f82911d02b2355dfe24c63772c23e62e66a41f2.
Require2GiB free/idle existing L4; worker600s, supervisor660s+30s grace,
export90s/external120s+10s grace; at most20GiB torch-allocated VRAM. Five pasteable
Google Cloud SDK/SSH/tmux steps include separate PuTTY-compatible downloads.
An independent safe importer/full-matrix/CPU forward/vector/cohort audit is
prepared before execution. It never executes returned source or local derivatives.

Actual L4 proof and returned audit remain pending. No new training protocol is
defined before that proof; no further assistant VM/cloud action occurs. Direct
VM cleanup was separately authorized and verified:14 backed-up duplicate archives
removed/1.64GiB; last verified free7.48GiB. Original scientific caches, sources,
checkpoints, outputs, failures and current V27 archives remain protected.

Previous668 research bindings, cleanup60, deeper309/66/692/299/697/513 histories,
three full document bodies and app22 bindings remain intact. The original DGP
remains primary in the existing app; pretrained restorers are comparisons. No
native/reserved-final pixels, ethnicity or local/hidden-identity claim enter this
TRAIN-only diagnostic. Useful native outputs, all visible facial features,
canonical app flow/regressions, all seven automatic/assisted covering families
and independent final review remain required. Goal active/incomplete.

[Original-decoder review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_ORIGINAL_DECODER_REVIEW_V1.md>) ·
[Five manual R2 VM steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_ORIGINAL_DECODER_GRADIENT_V1_R2_VM.md>) ·
[Verified cleanup](<C:/xampp/htdocs/YEAR 4/Testing/VM_STORAGE_CLEANUP_20261006_V2.md>)

Earlier bodies below remain preserved history. Their V27-pending, architecture-
question and prior pilot-launch text is historical; only the new R2 proof is ready.

'''
    for name in DOCS:
        original=(ROOT/name).read_bytes()
        with (before/name).open('xb') as f:f.write(original)
        mapping[name]=(before/name).relative_to(ROOT).as_posix()
        split=original.index(b'\n')+1
        (ROOT/name).write_bytes(original[:split]+b'\n'+prefix.encode('utf-8')+original[split:])
    verify(previous['new_evidence_sha256'],mapping)
    evidence={}
    def bind(path):evidence[path.relative_to(ROOT).as_posix()]=sha(path)
    for name in [*DOCS,'CCTV_DGP_ORIGINAL_DECODER_REVIEW_V1.md','CCTV_DGP_ORIGINAL_DECODER_GRADIENT_V1_R2_VM.md',
        'scripts/cctv_dgp_original_decoder_candidate_v1.py','scripts/review_cctv_dgp_original_decoder_v1.py',
        'scripts/verify_cctv_dgp_original_decoder_review_v1.py','scripts/prepare_cctv_dgp_original_decoder_gradient_v1.py',
        'scripts/cctv_dgp_original_decoder_gradient_v1_vm.py','scripts/prepare_cctv_dgp_original_decoder_gradient_v1_r1.py',
        'scripts/prepare_cctv_dgp_original_decoder_gradient_v1_r2.py','scripts/verify_cctv_dgp_original_decoder_gradient_v1_r1.py',
        'scripts/verify_cctv_dgp_original_decoder_gradient_v1_r2.py',
        'scripts/audit_cctv_dgp_original_decoder_gradient_v1_r1_return.py','scripts/audit_cctv_dgp_original_decoder_gradient_v1_r2_return.py',
        'tests/test_cctv_dgp_original_decoder_gradient_v1_r1_audit.py','tests/test_cctv_dgp_original_decoder_gradient_v1_r2_audit.py',
        'scripts/record_cctv_dgp_original_decoder_review_gradient_v1_r2_milestone.py',
        'scripts/verify_cctv_dgp_original_decoder_review_gradient_v1_r2_milestone.py',
        'outputs/dgp_feature_skips_v27_audit_milestone/milestone.json',
        'outputs/dgp_feature_skips_v27_audit_milestone/independent_readback.json']:
        bind(ROOT/name)
    for stem in ['v1','v1-r1','v1-r2']:
        for suffix in ['.tar.gz','.tar.gz.sha256']:bind(ROOT/('outputs/cctv-dgp-original-decoder-gradient-'+stem+'-execution'+suffix))
    for directory in [ROOT/'outputs/cctv_dgp_original_decoder_review_v1',
        ROOT/'outputs/cctv_dgp_original_decoder_gradient_v1_vm',ROOT/'outputs/cctv_dgp_original_decoder_gradient_v1_preparation',
        ROOT/'outputs/cctv_dgp_original_decoder_gradient_v1_r1_vm',ROOT/'outputs/cctv_dgp_original_decoder_gradient_v1_r1_preparation',
        ROOT/'outputs/cctv_dgp_original_decoder_gradient_v1_r2_vm',prep,before]:
        for path in directory.rglob('*'):
            if path.is_file():bind(path)
    milestone={'complete':True,'date':'2026-10-06','scope':'Selected original-decoder initial review and manual zero-update proof preparation only',
        'new_evidence_sha256':evidence,'previous_milestone_sha256':sha(PREVIOUS),'previous668_original_locations':mapping,
        'original_document_bodies_preserved':3,'app_record_sha256':sha(app_path),'app22_bindings_preserved':True,
        'decoder_original_parameters_enabled_for_future_VM':609219,'decoder_parameter_tensors':14,
        'initial_raw_PNG_parity_cases':50,'local_original_DGP_forwards':50,'local_candidate_forwards':51,
        'local_gradient_calls':0,'local_backward_calls':0,'local_optimizer_updates':0,
        'actual_L4_gradient_proof_pending':True,'prospective_full_return_audit_prepared':True,
        'released_protocol_sha256':packet['protocol_sha256'],'released_archive_sha256':packet['archive_sha256'],
        'released_archive_bytes':13816,'source_return_regressions_passed':8,'unissued_drafts_preserved':True,
        'human_manual_VM_execution_required':True,'VM_actions':False,'new_training_recipe_created':False,
        'native_or_reserved_used':False,'independent_final_review_complete':False,'app_promotion':False,
        'goal_status':'active','goal_complete':False,'seconds':time.monotonic()-start}
    with (OUT/'milestone.json').open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(milestone,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:milestone[k] for k in ['complete','initial_raw_PNG_parity_cases','released_archive_bytes','actual_L4_gradient_proof_pending','goal_status','seconds']},indent=2))


if __name__=='__main__':main()
