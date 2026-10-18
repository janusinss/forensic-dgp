"""Freshly close actual saved evidence and root review; no model execution."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'outputs/codeformer_bounded_fusion_feasibility_v1_return_review'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    t = read(REVIEW / 'transport_v1.json')
    assert t['complete'] and t['members'] == 2231 and t['payload_bytes'] == 761506959
    assert t['archive_sha256'] == '3e1a0a4bee698b397747ba39d3546cbca3cbd298e9cb450bc1abfc61aa880bb9'
    archive = ROOT / 'outputs/cctv-codeformer-bounded-fusion-feasibility-v1-results.tar.gz'
    assert archive.stat().st_size == 668557079 and sha(archive) == t['archive_sha256']
    imported = Path(t['import_directory'])
    assert imported.resolve() == (ROOT / 'outputs/codeformer_bounded_fusion_feasibility_v1_return').resolve()
    files = {path.relative_to(imported).as_posix() for path in imported.rglob('*') if path.is_file()}
    assert files == set(t['files_sha256']) == set(t['files_bytes'])
    for name in sorted(files):
        path = imported / name
        assert not path.is_symlink() and path.stat().st_size == t['files_bytes'][name]
        assert sha(path) == t['files_sha256'][name]
    packet = ROOT / 'outputs/cctv_codeformer_bounded_fusion_feasibility_v1_vm'
    assert sha(packet / 'protocol.json') == t['protocol_sha256']
    p = read(packet / 'protocol.json')
    for name, digest in p['files_sha256'].items():
        assert sha(packet / name) == digest
    state = read(REVIEW / 'state_v1/state_audit.json')
    pixels = read(REVIEW / 'metrics_v2/metrics_audit.json')
    visual = read(REVIEW / 'visual_review_v1.json')
    science = read(REVIEW / 'independent_scientific_readback_v1.json')
    report = read(REVIEW / 'independent_report_readback_v2.json')
    assert state['complete'] and state['evidence_consistency_pass'] and state['paths_checked'] == 170
    assert state['trace']['events'] == 1130 and state['optimizer_updates'] == state['gradient_queries'] == 0
    assert pixels['complete'] and pixels['saved_output_count'] == 170 and pixels['metric_rows'] == 510
    for name, digest in pixels['files_sha256'].items():
        assert sha(REVIEW / 'metrics_v2' / name) == digest
    assert visual['complete'] and visual['root_personally_viewed_all_planned_sheets']
    assert visual['sheets_reviewed'] == 18 and visual['all_nonblank_source_cells_independently_rechecked'] == 246
    for row in visual['sheets']:
        assert sha(ROOT / row['path']) == row['sha256']
    assert science['approved'] and science['complete'] and not science['blocking_findings']
    assert report['complete'] and report['approved'] and not report['findings']
    report_path = ROOT / 'CCTV_CODEFORMER_BOUNDED_FUSION_FEASIBILITY_V1_RETURN_REVIEW.md'
    assert sha(report_path) == report['report_sha256']
    paths = ['transport_v1.json', 'state_v1/state_audit.json', 'metrics_v2/metrics_audit.json',
        'visual_sheets/manifest.json', 'visual_review_v1.json', 'independent_scientific_readback_v1.json',
        'independent_report_readback_v1.json', 'independent_report_readback_v2.json',
        'adain_parent_reuse_inputs_v1.json', 'scientific_readback_v1.json', 'SCIENTIFIC_READBACK_V1.md',
        'metrics_v1/audit_failure.json', 'metrics_v1/gate_roundoff_diagnosis.json',
        'report_draft1_before_input_policy_readback.md']
    evidence = {str((REVIEW / name).relative_to(ROOT)): sha(REVIEW / name) for name in paths}
    result = {'complete': True, 'status': 'ACTUAL MEASUREMENT AUDITED; FIXED BRIGHTNESS GATE REJECTED',
        'freshly_rehashed_imported_members': 2231, 'freshly_rehashed_payload_bytes': 761506959,
        'freshly_rehashed_original_packet_files': len(p['files_sha256']) + 1,
        'archive_sha256': t['archive_sha256'], 'protocol_sha256': t['protocol_sha256'],
        'evidence_sha256': evidence, 'report_sha256': sha(report_path), 'source_sha256': sha(__file__),
        'root_CPU_fixtures_passed': 30, 'root_CPU_fixture_command':
            'venv/Scripts/python.exe -X utf8 -B -m unittest discover -s tests -p test_codeformer_bounded_fusion_return_*.py',
        'fixed_gate_advanced_to_training': False, 'model_qualified': False, 'goal_complete': False,
        'local_neural_gradient_optimizer_VM_calls': 0,
        'canonical_input_correction': 'All200 retained RGB8/255 float32 inputs match original training pixel hashes; historical continuous-input wording retained as a documentation error, not a recipe change.'}
    with (REVIEW / 'closed_chain_v1.json').open('x', encoding='utf-8', newline='\n') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps({'complete': True, 'members': 2231, 'packet_files': result['freshly_rehashed_original_packet_files'],
        'sheets_reviewed': 18, 'fixed_gate_advanced': False, 'model_qualified': False}))


if __name__ == '__main__':
    main()
