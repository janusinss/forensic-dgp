"""Publish static readiness and the checkpoint-choice caveat without a VM run."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/codeformer_u96_spatial_feasibility_v1_preparation'
PUBLICATION = PREP / 'publication_v1'
DOCS = ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
        'PROJECT_HANDOFF.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md',
        'CCTV_CODEFORMER_U96_SPATIAL_FEASIBILITY_V1_PREPARATION.md')


def pin(path):
    data = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def load(name):
    return json.loads((PREP / name).read_bytes())


def write_new(path, data):
    with path.open('xb') as stream:
        stream.write(data)


def main():
    assert not PUBLICATION.exists() and 'torch' not in sys.modules
    build = load('build_receipt_v1.json')
    audit = load('independent_execution_archive_audit_v1.json')
    guide = load('manual_guide_independent_review_v1.json')
    tests = load('root_runner_source_tests_v1.json')
    source = load('runner_review/new_runner_source_review_v1.json')
    static = load('staged_static_readback_v1.json')
    assert build['complete'] and audit['complete'] and audit['independent_archive_audit_pass']
    assert audit['checks'] == 301 and audit['members'] == 280
    assert guide['complete'] and guide['source_review_pass'] and not guide['unresolved_blockers']
    assert tests['complete'] and tests['all_tests_pass'] and tests['tests'] == 141
    assert source['complete'] and source['source_review_pass'] and not source['unresolved_blockers']
    assert static['complete'] and static['source_review_pass']
    for key in ('archive', 'protocol', 'checksum'):
        assert pin(ROOT / build[key]['path']) == build[key] == audit[key]
    for row in source['sources'] + tests['sources']:
        assert pin(ROOT / row['path']) == row
    guide_path = ROOT / 'CCTV_CODEFORMER_U96_SPATIAL_FEASIBILITY_V1_VM.md'
    assert guide['guide'] == pin(guide_path)
    comparison = PREP / 'historical_gain_comparison_readback_v1.json'
    assert json.loads(comparison.read_bytes())['complete'] is True
    evidence = [PREP / name for name in (
        'build_receipt_v1.json', 'independent_execution_archive_audit_v1.json',
        'manual_guide_independent_review_v1.json', 'root_runner_source_tests_v1.json',
        'runner_review/new_runner_source_review_v1.json', 'staged_static_readback_v1.json',
        'prospective_payload_time_ledger_v1.json', 'historical_gain_comparison_readback_v1.json')]
    prefix = (
        '# 23 October 2026: spatial feasibility packet sealed; checkpoint choice under review\n\n'
        '[Five-step manual guide](CCTV_CODEFORMER_U96_SPATIAL_FEASIBILITY_V1_VM.md) and the distinct '
        'spatial-feasibility execution packet are prepared and UNRUN. Archive40,009,514bytes, SHA256 '
        '`84652e91c859874ece30fceb79b84075d712f1af467b6de69c2ed65ac9b5a8d0`; protocolSHA256 '
        '`6ad2d51f25cc5c7ef6fab95c3ff26f6a646780d5c0e5ba6e118b7d78d16d7f1e`. '
        'The complete runner passed independent source review and141pure/NumPy tests; independent actual '
        'archive audit passed301checks over280members/51,624,775payloadbytes, including complete single-gzip '
        'CRC/ISIZE and tar-tail checks. Staged CLI help and281static bindings passed without importing '
        'Torch/NumPy or installing the two intentionally omitted retained initializer files. The exact '
        'commands and resource/failure wording have independent static review. These are preparation '
        'proofs, not CUDA execution, useful restoration or qualification.\n\n'
        'The prepared diagnostic starts from complete U96, opens only existing ResBlock22/RGBConv24 '
        '(10tensors/75,843scalars;529frozen), and caps218forwards/32component-gradient queries/two '
        'independent FP32 applications from the same initializer. Maximum coordinate displacements '
        '0.00001/0.00005 are not learning rates or epochs; zero optimizer objects/steps. Preserve '
        '16gradientcases,32adaptive TRAIN monitors,24legacy guards and all quality requirements. '
        'Normal completion requires complete539-tensor state witnesses; failure retains existing/partial '
        'bytes with bounded best-effort extra capture. Scientific stop1200s, external1260s plus20s '
        'termination grace, bank phases180s, export300/330s;8GiBinitial/4GiBcontinuing disk,2GiBoutput '
        'and archive caps. Forward settings stay strict; known CUDA bilinear-upsample backward warnings '
        'are recorded in individual warning-only query scopes and flags restored. No bitwise CUDA '
        'gradient repeatability is claimed; unexpected nondeterministic gradient operations stop.\n\n'
        '[Historical gain comparison](outputs/codeformer_u96_spatial_feasibility_v1_preparation/'
        'historical_gain_comparison_readback_v1.json) resolves the user question: prior u64fusion '
        'achieved1.627216%global/1.210135%local PNG gain on the SAME eight adaptive TRAIN monitors and '
        'frozen U96 references; the newer readout route achieved0.103591%/0.131680%. The latter is a '
        'worse independent recipe, not cumulative training or deterioration of the retained u64. '
        'Trainable path, learning rate and objective changed together, so no single-factor cause is '
        'proved. The1.627%TRAIN pass remains verified; later broader saved outputs show preservation '
        'violations in both arms and no qualified successor. U64 remains a retained challenger, U96 '
        'the experimental incumbent. A U96 restart was a controlled alternative, not proof it was the '
        'best starting checkpoint. Readiness of this packet does not select it over a u64-continuation '
        'design. Before the next launch decision, explicitly assess retaining u64 gains while fixing '
        'its failures and compare against both controls; do not automatically launch this or any old guide.\n\n'
        'STATUS: SEALED_UNRUN_DIAGNOSTIC; INITIALIZER_CHOICE_UNDER_REVIEW. No model/Torch/neural/gradient/'
        'optimizer/training/VM call or app/default/checkpoint change occurred in this preparation. '
        'Original u64/U96/R2, all predecessors, splits, failure records and gates remain retained. '
        'Optional reviewed Protect/Apply/bypass stays separate; automatic matching deferred, native/wild '
        'unpaired, reserved FINAL unopened/execution-unfrozen with independent reviewers pending and '
        'later completion deferred. Full goal ACTIVE/incomplete. Sealed instruction copies reflect '
        'their earlier unsealed preparation status and remain immutable historical evidence. Exact '
        'previous document bytes are preserved below.\n\n---\n\n'
    ).encode('utf-8')
    originals = {name: (ROOT / name).read_bytes() for name in DOCS}
    PUBLICATION.mkdir()
    records = []
    for name, before in originals.items():
        snapshot = PUBLICATION / (name + '.before')
        write_new(snapshot, before)
        path = ROOT / name
        temporary = path.with_name(path.name + '.spatial_packet_v1.new')
        write_new(temporary, prefix + before)
        assert path.read_bytes() == before and snapshot.read_bytes() == before
        temporary.replace(path)
        assert path.read_bytes() == prefix + snapshot.read_bytes()
        records.append({'after': pin(path), 'previous_snapshot': pin(snapshot),
                        'prefix_bytes': len(prefix), 'previous_suffix_exact': True})
    receipt = {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
               'status': 'SEALED_UNRUN_DIAGNOSTIC_INITIALIZER_CHOICE_UNDER_REVIEW',
               'previous_goal_turn_classification': 'progress: completed reviewed inert runner and began sealed build',
               'this_goal_turn_classification': 'progress: sealed and independently audited actual execution archive; clarified distinct historical gain recipes and manual guide',
               'documents': records, 'guide': pin(guide_path), 'evidence': [pin(path) for path in evidence],
               'archive': build['archive'], 'protocol': build['protocol'], 'checksum': build['checksum'],
               'publisher': pin(Path(__file__).resolve()), 'model_training_VM_calls': 0,
               'app_changes': False, 'actual_run_selected_or_launched': False, 'goal_complete': False}
    destination = PUBLICATION / 'root_publication_readback_v1.json'
    write_new(destination, (json.dumps(receipt, indent=2, allow_nan=False) + '\n').encode('utf-8'))
    assert 'torch' not in sys.modules
    print(json.dumps({'complete': True, 'documents': len(records), 'receipt': pin(destination),
                      'actual_run_selected_or_launched': False, 'goal_complete': False}))


if __name__ == '__main__':
    main()
