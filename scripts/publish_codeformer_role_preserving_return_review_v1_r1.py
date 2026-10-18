"""Publish the immutable R1 zero-update initializer failure readback; no NN."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import stat

ROOT=Path(__file__).resolve().parents[1]
REVIEW=ROOT/'outputs/codeformer_role_preserving_training_v1_r1_return_review'
IMPORTED=ROOT/'outputs/codeformer_role_preserving_training_v1_r1_return'
EXPECTED='075d0fcc9b6dbd35e1ba65050c7837c00f1bd64afb44164a2a45ae991f1ab2e1'
PROTOCOL='4edacae4f1e61f06b21fb1308423398d5b9f842def160c02f74f8edc0975fc40'


def need(value,message):
    if not value:raise ValueError(message)


def regular(path):
    path=Path(path)
    for part in (path,*path.parents):
        if part.exists():
            s=part.lstat();need(not stat.S_ISLNK(s.st_mode)and not getattr(s,'st_file_attributes',0)&1024,'Unlinked evidence paths required')
    s=path.lstat();need(stat.S_ISREG(s.st_mode)and s.st_nlink==1,'Single linked regular evidence required')
    return path


def sha(path):
    path=regular(path);before=path.stat();h=hashlib.sha256()
    with path.open('rb')as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    after=path.stat();need((before.st_size,before.st_mtime_ns,before.st_ino)==
        (after.st_size,after.st_mtime_ns,after.st_ino),'Evidence changed during hash')
    return h.hexdigest()


def load(path):
    path=regular(path);need(path.stat().st_size<=2*1024**2,'Bounded metadata required')
    return json.loads(path.read_bytes())


def main():
    report_path=ROOT/'CCTV_CODEFORMER_ROLE_PRESERVING_TRAINING_V1_R1_RETURN_REVIEW.md'
    receipt_path=REVIEW/'report_readback_v1.json'
    need(not report_path.exists()and not receipt_path.exists(),'Preserve prior/partial R1 report')
    imported=load(REVIEW/'return_import_v1.json');transport=load(REVIEW/'transport_v1.json')
    hq=load(REVIEW/'hq_v1/audit.json');cause=load(REVIEW/'cause_v1/audit.json')
    scope=load(REVIEW/'repair_scope_source_review_v1.json')
    archive=ROOT/'outputs/cctv-codeformer-role-preserving-training-v1-r1-results.tar.gz'
    need(archive.stat().st_size==34174817 and sha(archive)==EXPECTED==imported['archive_sha256']and
        imported['protocol_sha256']==PROTOCOL and imported['regular_files']==467 and
        imported['all32static_files_exact']is True and transport['transport_pass']is True,
        'Exact actual R1 return transport required')
    rel='outputs/role_preserving_training_v1_r1/failure.json';failure_path=IMPORTED/rel
    need(sha(failure_path)==imported['files_sha256'][rel],'Actual imported failure identity differs')
    failure=load(failure_path)
    need(failure['completed_training_updates']==failure['optimizer_constructions_completed']==0 and
        failure['counts']['gradient_queries']==failure['counts']['backward_calls']==0 and
        failure['checkpoints']=={} and failure['last_safe_checkpoint']is None and
        failure['reason']=='ValueError: Matched initializer inference parity failed; actual arrays and context retained',
        'Exact no-training initializer failure required')
    need(hq['pass']is True and hq['selected_TRAIN_references']==100 and
        hq['exact_parent_file_bytes']==hq['canonical_array_payload_hashes']==200,
        'Independent all100HQarray parity required')
    need(cause['complete']is True and cause['archive_sha256']==EXPECTED and
        cause['recorded_worker_difference_metrics_independently_reproduced']is True and
        cause['new_model_created']is False,'Independent actual array audit required')
    comparisons=cause['independent_array_comparisons']
    need(all(comparisons[k]['exact_array_equal']is True and comparisons[k]['exact_NPY_file_bytes_equal']is True
        for k in ('native','raw','latent','tokens'))and
        comparisons['logits']['different_elements']==223123 and comparisons['logits']['elements']==262144 and
        comparisons['logits']['maximum_absolute_difference']==7.62939453125e-06,
        'Exact independently saved-array differences required')
    selection=cause['dictionary_selection_evidence']
    need(selection['all_argmax_indices_equal']is True and selection['tokens']==256 and
        selection['candidate_saved_tokens_equal_candidate_logit_argmax']is True and
        selection['parent_saved_tokens_equal_parent_logit_argmax']is True and
        selection['all256_parent_top2_margins_exceed_twice_corresponding_maximum_logit_error']is True,
        'Independent dictionary selection stability required')
    need(scope['complete']is True and scope['archive_sha256']==EXPECTED and
        scope['actual_runtime_version']=='2.9.1+cu129','Source/runtime cause review required')
    report='''# Role-preserving CodeFormer R1 return — 18 October 2026

**Technical initializer stop: zero new training updates, zero Adam constructions,
zero gradients/backwards and no new checkpoint.** Export completion packaged
the failure evidence; it did not create a trained model or measure improvement.

The actual 34,174,817-byte archive has SHA-256
`075d0fcc9b6dbd35e1ba65050c7837c00f1bd64afb44164a2a45ae991f1ab2e1`.
Protocol: `4edacae4f1e61f06b21fb1308423398d5b9f842def160c02f74f8edc0975fc40`.
All 467 regular files, 32 frozen static files, manifest, gzip/USTAR boundaries
and four companions passed inert readback. The 100 fresh HQ-reference cache
sets contain 200 arrays independently matched to retained parent files and
canonical target payload hashes. Those arrays are cache evidence, not training.

The worker stopped after 59.672 seconds at
`Matched initializer inference parity failed; actual arrays and context retained`.
Its 102 forward reservations cover 100 HQ calls, one frozen parent and one
candidate initializer call. No training-mode witness, gradient, Adam step,
candidate checkpoint or 406-case development quality bank was reached.

## What actually differed

Independent CPU readback of the saved candidate/parent arrays found:

- Native 512 output: exactly equal, including NPY file bytes.
- Delivered 256 RAW output: exactly equal, including NPY file bytes.
- Encoder latent and all 256 selected dictionary indices: exactly equal.
- Intermediate logits: 223,123 of 262,144 values differ; maximum absolute
  difference **0.00000762939453125**, mean absolute difference 0.0000005582493471933958.

All 256 saved indices match both logit argmax arrays. Every parent top-two
logit margin exceeds twice the corresponding maximum observed logit error;
the smallest parent margin is 0.0001087188720703125. These observations show
unchanged hard dictionary selections and reconstructed pixels for this one
saved initializer witness. They do not establish behavior on another input.

All 371 module modes match in evaluation mode, with no-grad FP32 calls and
matching saved before/after RNG hashes. The candidate retains 350 trainable
parameter flags; the frozen parent has zero. Factory fingerprints record the
same 515 tensor values. This return disproves the assumption that those
conditions ensure bit-exact logits across separately allocated objects.

The exact first differing operation and numerical mechanism remain unknown.
No predictor-layer intermediates, same-object repeatability or CUDA kernel
trace was exported. Nondeterministic algorithms were permitted. Neither that
setting nor the different parameter flags proves the cause. The pinned
CodeFormer/PyTorch source review does not support claiming the native
MultiheadAttention fast path caused this difference.

## Prospective numerical test policy

The failed V1 and R1 packages/returns remain immutable. A separate proposed R2
changes the **initializer numerical gate**, not the restoration-quality gates:
permit only float32 intermediate logits within the declared PyTorch 2.9
`assert_close` defaults, absolute tolerance 0.00001 and relative tolerance
0.0000013. In addition, every hard argmax and saved token must remain identical,
and every parent top-two margin must exceed twice its observed maximum logit
error. All native/RAW/latent/token arrays, model state, contexts and RNG evidence
retain exact checks. This is an explicit changed numerical prerequisite;
it is not a measured repair of an identified CUDA operation.

PyTorch documents float32 default comparison tolerances and the limitations
of floating-point reproducibility. Those general documents support declaring
the test policy; they do not identify the cause of this return or qualify
the model. [PyTorch 2.9 assert_close](https://docs.pytorch.org/docs/2.9/testing.html#torch.testing.assert_close),
[PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html).

R2 remains an actual finite restoration-training pilot with unchanged parent,
350/165 trainable/frozen partition, data, objective, rates and quality requirements:
maximum 50 new Adam updates, 200 microbatches, 1,399 neural forwards and 1,854
reservations. It introduces no additional neural diagnostic run or automatic
retry. A newly sealed, independently reviewed manual L4/tmux package is required.
The changed numerical gate and eventual trained outputs remain unverified
until that manual run returns. Do not rerun either failed folder unchanged.

Useful restoration still requires user-visible improvement, appearance
preservation, independent FINAL review and the existing app Auto/override/PNG
flow. Untouched CodeFormer and original DGP/checkpoints/caches/splits/provenance,
source terms and failures remain preserved. Native CCTV stays unpaired and
source-separated; no ethnicity, Zamboanga performance or exact hidden identity
is established. Separate covering/mask review/seven-family completion remains
deferred until restoration acceptance. Storage offload is complete; the full
research goal remains incomplete.

## Evidence

- [Strict transport](outputs/codeformer_role_preserving_training_v1_r1_return_review/transport_v1.json)
- [Independent saved-array cause audit](outputs/codeformer_role_preserving_training_v1_r1_return_review/cause_v1/audit.json)
- [All 100 HQ references](outputs/codeformer_role_preserving_training_v1_r1_return_review/hq_v1/audit.json)
- [Primary source and repair-scope review](outputs/codeformer_role_preserving_training_v1_r1_return_review/repair_scope_source_review_v1.json)
'''
    with report_path.open('x',encoding='utf-8',newline='\n')as stream:stream.write(report)
    names=('return_import_v1.json','transport_v1.json','hq_v1/audit.json','cause_v1/audit.json','repair_scope_source_review_v1.json')
    receipt={'complete':True,'UTC':datetime.now(timezone.utc).isoformat(),'archive_sha256':EXPECTED,
        'archive_bytes':34174817,'protocol_sha256':PROTOCOL,'regular_members':467,
        'failure_file_sha256':sha(failure_path),'evidence_sha256':{name:sha(REVIEW/name)for name in names},
        'actual_failure':failure,'all10saved_parity_arrays_independently_reviewed':True,
        'HQ_arrays_verified':200,'new_training_updates':0,'new_checkpoints':0,
        'report_sha256':sha(report_path),'report_writer_sha256':sha(__file__),
        'initializer_numerical_gate_change_proposed':True,'restoration_quality_gates_changed':False,
        'actual_local_neural_gradient_optimizer_VM_calls':0,'model_qualified':False,'goal_complete':False,
        'scope':'Same owner of importer/report; actual saved-array cause audit is independent. No local neural or VM execution.'}
    with receipt_path.open('x',encoding='utf-8',newline='\n')as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    print(json.dumps({'complete':True,'report':str(report_path),'report_sha256':sha(report_path),
        'actual_new_updates':0,'new_checkpoint':False,'initializer_gate_change_proposed':True}))


if __name__=='__main__':main()
