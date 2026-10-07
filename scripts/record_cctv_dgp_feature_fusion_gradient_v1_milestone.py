"""Release the verified zero-update diagnostic and preserve the preceding handoff."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / 'outputs/cctv_dgp_profile_batches_v31_return_milestone'
OUT = ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_milestone'
PACKET = ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_vm'
VERIFIED = ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_preparation_r1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify(mapping, aliases=None):
    for name, digest in mapping.items():
        path = (ROOT / (aliases or {}).get(name, name)).resolve()
        assert path.is_relative_to(ROOT) and path.is_file() and not path.is_symlink() and sha(path) == digest, name


def main():
    assert not OUT.exists()
    previous = read(PREVIOUS / 'milestone.json')
    verify(previous['new_evidence_sha256'])
    final = read(PREVIOUS / 'final_readback.json')
    assert final['complete'] and final['independent_milestone_pass']
    verify(final['evidence_sha256'])
    closed = read(PREVIOUS / 'independent_closure_audit.json')
    assert closed['complete'] and closed['milestone_sha256'] == sha(PREVIOUS / 'milestone.json')
    checked = read(VERIFIED / 'independent_packet_audit.json')
    proof = read(ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_forward_review/review.json')
    p = read(PACKET / 'protocol.json')
    assert checked['complete'] and checked['regressions_passed'] == 7
    assert checked['protocol_sha256'] == proof['protocol_sha256'] == sha(PACKET / 'protocol.json')
    assert checked['optimizer_updates'] == checked['local_neural_or_gradient_calls'] == 0
    assert proof['complete'] and proof['actual_named_parameter_order_matches_packet']
    assert proof['original_DGP_forwards'] == proof['candidate_DGP_forwards'] == 4
    assert proof['gradient_queries'] == proof['optimizer_updates'] == 0
    assert len(proof['rows']) == 4 and all(row['exact_raw_parity'] and row['exact_PNG_parity'] for row in proof['rows'])
    verify(proof['source_bindings_sha256'])
    verify(p['local_basis_sha256'])
    assert p['selected_tensors'] == 23 and p['fusion_parameters'] == 479616 and p['gradient_queries'] == 280
    assert not p['new_training_recipe_created'] and not checked['VM_execution_started']
    assert not (PACKET / 'outputs').exists()
    failure = read(ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_preparation/bash_syntax.json')
    recovered = read(VERIFIED / 'bash_syntax.json')
    assert not failure['complete'] and "couldn't create signal pipe" in failure['stderr']
    assert recovered['complete'] and recovered['exit_code'] == 0
    assert failure['script_sha256'] == recovered['script_sha256'] == sha(PACKET / 'scripts/run_fusion.sh')
    pin, archive_sha = checked['protocol_sha256'], checked['archive_sha256']
    addition = f'''
**Latest diagnostic milestone - 7 October 2026: original DGP feature-fusion measurement ready for manual VM execution.**

V31 remains closed at 50 updates: 0.694525% structure gain against the unchanged
1% requirement. Its independently audited return, 27,623 files, all50 visual
observations, original/stopped checkpoints and numerical failure remain retained.
All17 delivered preservation groups passing does not waive that stop. The app's
original own-DGP primary checkpoint is unchanged; native usefulness is unqualified.

A distinct zero-update diagnostic measures the original FPN feature-fusion path
alongside the active decoder. The separate copy exposes 11 original fusion
tensors (479,616 parameters) and 12 decoder tensors (498,627 parameters) to
gradient queries without changing their values. Backbone, inactive head4 and
evaluation normalization remain preserved. Two source/profile-matched TRAIN
cohorts each contain ten references with one clear and four degraded views.
The unexposed cohort excludes all50 references used by V31's stopped run.
Original and stopped50 states are observations only; neither is resumed.

The three-file 54,462-byte transfer packet passes archive/source/data checks,
Python3.10 syntax, seven metadata/return-boundary regressions, read-only Bash
syntax and rejection of Windows differentiation before neural imports. Four
CPU forward-only comparisons verify the actual23-tensor order and exact starting
raw/PNG parity. Eight model forwards and zero gradient/optimizer calls occurred.
Gradient connectivity through the new fusion partition is still unmeasured.

Protocol SHA256: {pin}
Execution archive SHA256: {archive_sha}

The worker permits 280 component-gradient queries, zero optimizer updates and
zero epochs, with600s worker/900s external bounds,30s grace,20GiB allocated VRAM,
6GiB required free space and1.5GiB maximum uncompressed return. Export has its
own300s/330s bounds. Any source/state/nonfinite/time failure is retained. No
checkpoint writer, loss/gate relaxation or new training recipe is present.
Actual measurements remain the user's manual existing-L4 tmux workflow. The
agent has not uploaded, installed or launched the new packet on the VM.

The first local Bash parser could not start because the Windows sandbox blocked
its signal pipe. Its failed record/source and passing regression log remain.
The unchanged script passed the same read-only parser outside that sandbox;
this changes no VM recipe or scientific gate.

The prospective local return checker permits only inference and saved-gradient
arithmetic, never returned-code execution or local differentiation. Results must
be audited before choosing any later finite training pilot. No native, development
or reserved-final cases were used by this packet or parity check. The seven
covering families, separate automatic/assisted quality, useful native DGP outputs
and independent final review remain required. The full goal is active/incomplete.

[Five manual upload/tmux/download steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FEATURE_FUSION_GRADIENT_V1_VM.md>)
[V31 audited findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_PROFILE_BATCHES_V31_RESULTS.md>)

The complete previous handoff body follows. Its unreleased-packet statement
describes the prior return milestone, before the distinct diagnostic above.

'''.encode('ascii')
    OUT.mkdir(); (OUT / 'before_docs').mkdir()
    handoff = ROOT / 'PROJECT_HANDOFF.md'
    before = handoff.read_bytes(); backup = OUT / 'before_docs/PROJECT_HANDOFF.md'
    with backup.open('xb') as stream:
        stream.write(before)
    split = before.index(b'\n') + 1
    handoff.write_bytes(before[:split] + addition + before[split:])
    files = [backup, handoff, ROOT / 'SYSTEM_WORKFLOW_AND_GOAL.md', ROOT / 'PRACTICAL_OUTPUT_SCOPE.md',
             ROOT / 'CCTV_DGP_FEATURE_FUSION_GRADIENT_V1_VM.md',
             ROOT / 'outputs/cctv-dgp-feature-fusion-gradient-v1-execution.tar.gz',
             ROOT / 'outputs/cctv-dgp-feature-fusion-gradient-v1-execution.tar.gz.sha256',
             Path(__file__), ROOT / 'scripts/verify_cctv_dgp_feature_fusion_gradient_v1_milestone.py',
             ROOT / 'scripts/prepare_cctv_dgp_feature_fusion_gradient_v1.py',
             ROOT / 'scripts/verify_cctv_dgp_feature_fusion_gradient_v1_packet.py',
             ROOT / 'scripts/review_cctv_dgp_feature_fusion_gradient_v1_forward.py',
             ROOT / 'scripts/cctv_dgp_feature_fusion_gradient_v1_vm.py',
             ROOT / 'scripts/cctv_dgp_feature_fusion_gradient_v1_return_audit_template.py',
             ROOT / 'scripts/audit_cctv_dgp_feature_fusion_gradient_v1_return.py',
             ROOT / 'tests/test_cctv_dgp_feature_fusion_gradient_v1.py']
    for folder in [PACKET, VERIFIED, ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_preparation',
                   ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_forward_review']:
        files += [path for path in folder.rglob('*') if path.is_file()]
    record = {'complete': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Verified manual zero-update feature-fusion diagnostic; no actual VM execution or new training',
              'new_evidence_sha256': {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(set(files))},
              'previous_milestone_sha256': sha(PREVIOUS / 'milestone.json'),
              'previous_document_locations': {'PROJECT_HANDOFF.md': backup.relative_to(ROOT).as_posix()},
              'document': {'name': 'PROJECT_HANDOFF.md', 'before_path': backup.relative_to(ROOT).as_posix(),
                           'before_sha256': sha(backup), 'after_sha256': sha(handoff), 'addition_bytes': len(addition)},
              'protocol_sha256': pin, 'archive_sha256': archive_sha, 'manual_VM_required': True,
              'gradient_queries_bound': 280, 'optimizer_updates': 0, 'actual_VM_execution_started': False,
              'local_original_DGP_forwards': 4, 'local_candidate_DGP_forwards': 4,
              'local_gradient_queries': 0, 'app_changes': False, 'new_training_recipe': False,
              'V31_failure_preserved': True, 'native_or_reserved_used': False,
              'quality_qualification': False, 'app_promotion': False, 'goal_status': 'active', 'goal_complete': False}
    with (OUT / 'milestone.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps({'complete': True, 'new_bindings': len(record['new_evidence_sha256']),
                      'milestone_sha256': sha(OUT / 'milestone.json'), 'goal_complete': False}))


if __name__ == '__main__':
    main()
