"""Record the audited diagnostic and release a finite manual V32 packet."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_milestone'
OUT = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_milestone'
PACKET = ROOT / 'outputs/cctv_dgp_feature_fusion_vm_v32'
RETURN = ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_return'
PREP = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_preparation'


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
    final = read(PREVIOUS / 'final_readback.json'); assert final['complete'] and final['independent_milestone_pass']
    verify(final['evidence_sha256'])
    old_closed = read(PREVIOUS / 'independent_closure_audit.json')
    assert old_closed['complete'] and old_closed['milestone_sha256'] == sha(PREVIOUS / 'milestone.json')
    audit = read(ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_independent_audit.json')
    analysis = read(ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_analysis/analysis.json')
    checked = read(PREP / 'independent_packet_audit.json')
    forward = read(ROOT / 'outputs/cctv_dgp_feature_fusion_v32_forward_review/review.json')
    p = read(PACKET / 'protocol.json')
    assert audit['complete'] and audit['diagnostic_complete'] and audit['CPU_replay']['cases_at_both_states'] == 200
    assert audit['members_verified'] == 756 and audit['optimizer_updates'] == audit['local_gradient_calls'] == 0
    assert analysis['complete'] and analysis['new_finite_fusion_partition_pilot_justified_for_preparation']
    verify(analysis['source_bindings_sha256'])
    assert checked['complete'] and checked['regressions_passed'] == 11 and checked['packet_files_verified'] == 11
    assert checked['protocol_sha256'] == forward['protocol_sha256'] == sha(PACKET / 'protocol.json')
    assert forward['complete'] and forward['all_buffers_unchanged'] and forward['actual_named_parameter_order_matches_packet']
    assert forward['original_DGP_forwards'] == forward['candidate_DGP_forwards'] == 4
    assert forward['gradient_queries'] == forward['optimizer_updates'] == 0
    assert len(forward['rows']) == 4 and all(r['exact_raw_parity'] and r['exact_PNG_parity'] for r in forward['rows'])
    verify(forward['source_bindings_sha256']); verify(p['local_basis_sha256'])
    assert p['selected_parameters'] == 978243 and p['selected_tensors'] == 23
    assert p['optimizer_updates'] == 800 and p['budgets']['minimum_free_disk_bytes'] == 8 * 1024**3
    assert not (PACKET / 'outputs').exists() and not checked['actual_VM_training_started']
    pin, archive_sha = checked['protocol_sha256'], checked['archive_sha256']
    addition = f'''
**Latest research milestone - 7 October 2026: original DGP fusion diagnostic audited; V32 finite pilot ready.**

The human returned the distinct feature-fusion diagnostic. The unchanged
prospective local checker verifies all756 files, 301,298,844 saved gradient
values and200 CPU inference outputs. VM measurement takes40.769s with280
component-gradient queries, zero optimizer updates/backwards/epochs and no new
checkpoint. CPU replay stays within the declared bounds: maximum raw error
2.294778823852539e-6, PNG difference1 byte and vector error5.252659320831299e-7.
No returned code is executed, no gradients are recomputed locally and no native
or reserved-final cases are opened. Both original and stopped states are retained.

All11 original fusion tensors and12 active decoder tensors have nonzero
improvement gradients in both matched TRAIN cohorts at both measured states.
The original fusion-only negative total-objective direction decreases all three
improvement terms in both cohorts. These are first-order observations, not a
prediction of AdamW steps, a unique causal explanation or useful output proof.
Stopped-state preservation terms respond to drift and remain binding. V31 stays
closed at50 updates/0.694525% against its unchanged1% structure requirement.

V32 tests one trainable-partition change on a fresh original-DGP copy: enable
the eight original fusion convolutions (11 tensors/479,616 parameters) alongside
12 active decoder tensors/498,627 parameters. Total978,243 parameters/23 tensors.
Backbone, inactive head4 and every evaluation buffer remain frozen. The full
781-reference/3,905-case TRAIN corpus, paired clear/degraded schedule, original
forward/mean-centering path, seven losses, initial normalizers, AdamW and all
scientific gates are unchanged. No stopped state is resumed or app checkpoint
selected. Source labels remain distinct from ethnicity and native CCTV evidence.

The11-file/754,835-byte packet passes an independent metadata/source/data/transfer
audit,11 meaningful regressions, Python3.10 syntax, read-only Bash syntax and a
Windows rejection before model imports. Four CPU comparisons verify actual23
parameter order, exact starting raw/PNG parity and unchanged buffers even after
train(True). Eight forward calls and zero local gradient/optimizer calls occur.

Protocol SHA256: {pin}
Execution archive SHA256: {archive_sha}

Maximum800 updates (781-batch epoch +19 batches). Stop at50 unless structure gain
reaches1%; final gates require10%, all17 preservation groups, both source gains
>=0 and mean-only fraction<=20%. Require8GiB free for a3GiB uncompressed return,
archive and margin. Enforce preflight300s/cache900s/fit3600s/worker4500s, external
4800s +30s grace, export900s/external930s +30s grace and allocated VRAM20GiB.
Every source, nonfinite, time or numerical stop is retained. Actual V32 training
is the user's manual existing-L4/g2-standard-4 tmux workflow. The agent has not
uploaded, installed or launched it and has deleted no VM or local research file.

An independent return audit and all50 TRAIN preview review are required before
any later fixed520 paired DEV/24 unpaired native DEV pass. Reserved-final pixels
remain unopened. The app still uses its retained original own-DGP checkpoint.
Useful native restoration, separate automatic/assisted covering quality across
all seven families, independent final review and the full goal remain incomplete.
Prior checkpoints, splits, provenance, cache backup and every failed gate remain.

[Audited diagnostic findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FEATURE_FUSION_GRADIENT_V1_RESULTS.md>)
[V32 five manual upload/tmux/download steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FEATURE_FUSION_V32_VM.md>)

The complete previous handoff body follows. Its diagnostic-unrun language
describes the prior packet-release milestone and is superseded above.

'''
    OUT.mkdir(); before_folder = OUT / 'before_docs'; before_folder.mkdir()
    handoff = ROOT / 'PROJECT_HANDOFF.md'; before = handoff.read_bytes()
    (before_folder / handoff.name).write_bytes(before)
    split = before.index(b'\n') + 1
    after = before[:split] + addition.encode('utf-8') + before[split:]
    handoff.write_bytes(after)
    bindings = [Path(__file__), ROOT / 'scripts/verify_cctv_dgp_feature_fusion_v32_milestone.py',
                ROOT / 'scripts/analyze_cctv_dgp_feature_fusion_gradient_v1.py',
                ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_independent_audit.json',
                ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_return_import.json',
                ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_analysis/analysis.json',
                ROOT / 'CCTV_DGP_FEATURE_FUSION_GRADIENT_V1_RESULTS.md', ROOT / 'CCTV_DGP_FEATURE_FUSION_V32_VM.md',
                ROOT / 'outputs/cctv-dgp-feature-fusion-gradient-v1-results.tar.gz',
                ROOT / 'outputs/cctv-dgp-feature-fusion-gradient-v1-results.tar.gz.sha256',
                ROOT / 'outputs/cctv-dgp-feature-fusion-gradient-v1-export.json',
                ROOT / 'outputs/cctv-dgp-feature-fusion-v32-execution.tar.gz',
                ROOT / 'outputs/cctv-dgp-feature-fusion-v32-execution.tar.gz.sha256',
                ROOT / 'PROJECT_HANDOFF.md', before_folder / handoff.name,
                ROOT / 'scripts/prepare_cctv_dgp_feature_fusion_v32.py',
                ROOT / 'scripts/verify_cctv_dgp_feature_fusion_v32_packet.py',
                ROOT / 'scripts/review_cctv_dgp_feature_fusion_v32_forward.py',
                ROOT / 'scripts/cctv_dgp_feature_fusion_v32.py', ROOT / 'scripts/cctv_dgp_feature_fusion_v32_policy.py',
                ROOT / 'scripts/audit_cctv_dgp_feature_fusion_v32_return.py',
                ROOT / 'scripts/cctv_dgp_feature_fusion_v32_return_audit_template.py',
                ROOT / 'tests/test_cctv_dgp_feature_fusion_v32.py',
                PREVIOUS / 'milestone.json', PREVIOUS / 'independent_closure_audit.json', PREVIOUS / 'final_readback.json',
                ROOT / 'SYSTEM_WORKFLOW_AND_GOAL.md', ROOT / 'PRACTICAL_OUTPUT_SCOPE.md']
    for folder in [PACKET, PREP, ROOT / 'outputs/cctv_dgp_feature_fusion_v32_forward_review']:
        bindings.extend(f for f in sorted(folder.rglob('*')) if f.is_file())
    unique = list(dict.fromkeys(bindings))
    m = {'complete': True, 'datetime_UTC': datetime.now(timezone.utc).isoformat(),
         'recorder_sha256': sha(Path(__file__)), 'previous_milestone_sha256': sha(PREVIOUS / 'milestone.json'),
         'previous_document_locations': {'PROJECT_HANDOFF.md': (before_folder / handoff.name).relative_to(ROOT).as_posix()},
         'new_evidence_sha256': {f.relative_to(ROOT).as_posix(): sha(f) for f in unique},
         'document': {'name': handoff.name, 'before_path': (before_folder / handoff.name).relative_to(ROOT).as_posix(),
                      'before_sha256': hashlib.sha256(before).hexdigest(), 'after_sha256': sha(handoff),
                      'addition_bytes': len(addition.encode('utf-8')), 'full_previous_body_preserved': True},
         'diagnostic_protocol_sha256': audit['protocol_sha256'], 'diagnostic_return_archive_sha256': audit['archive_sha256'],
         'diagnostic_complete': True, 'saved_gradient_values_verified': 301298844, 'CPU_return_cases_verified': 200,
         'human_VM_gradient_queries': 280, 'diagnostic_optimizer_updates': 0,
         'protocol_sha256': pin, 'execution_archive_sha256': archive_sha, 'V32_packet_files': 11,
         'V32_finite_updates': 800, 'V32_selected_tensors': 23, 'V32_regressions_passed': 11,
         'V32_original_CPU_forwards': 4, 'V32_candidate_CPU_forwards': 4,
         'new_training_recipe_prepared': True, 'manual_VM_required': True, 'actual_V32_training_started_by_agent': False,
         'local_gradient_queries': 0, 'local_optimizer_updates': 0, 'app_changes': False,
         'native_or_reserved_used': False, 'quality_qualification': False, 'app_promotion': False,
         'goal_status': 'active', 'goal_complete': False}
    with (OUT / 'milestone.json').open('x', encoding='utf-8', newline='\n') as stream: json.dump(m, stream, indent=2)
    print(json.dumps({'complete': True, 'new_bindings': len(m['new_evidence_sha256']), 'milestone_sha256': sha(OUT / 'milestone.json'),
                      'V32_protocol_sha256': pin, 'manual_VM_required': True, 'goal_complete': False}, indent=2))


if __name__ == '__main__':
    main()
