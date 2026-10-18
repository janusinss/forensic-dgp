"""Close the actual saved-return chain without model execution or promotion."""
import hashlib
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'outputs/codeformer_fp32_comparison_v1_return_review'
IMPORTED = ROOT / 'outputs/codeformer_fp32_comparison_v1_return'
PROTOCOL = '8acff86e31c3d135994fdac6ed781374e036150ce1ae2110db82ca0dc5a3aff1'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    start = time.monotonic()
    transport = read(REVIEW / 'transport_v1.json')
    assert transport['complete'] and transport['all_members_hash_verified'] and transport['exact_namespace']
    assert transport['archive_sha256'] == '8b2361147a1d12b7d2bad8a7d3a8965e0e90c077bb1cdbb4ad7378ad8703ef78'
    assert transport['protocol_sha256'] == PROTOCOL and transport['members'] == 6501
    assert Path(transport['import_directory']).resolve() == IMPORTED.resolve()
    export = read(ROOT / 'outputs/cctv-codeformer-fp32-comparison-v1-export.json')
    assert export['complete'] and not export['run_results_present'] and export['failure_present']
    manifest_path = IMPORTED / transport['manifest_member']
    assert sha(manifest_path) == export['manifest_sha256'] == transport['manifest_sha256']
    manifest = read(manifest_path)
    expected = dict(manifest['files_sha256'])
    expected[transport['manifest_member']] = transport['manifest_sha256']
    assert len(expected) == 6501
    actual = {path.relative_to(IMPORTED).as_posix() for path in IMPORTED.rglob('*') if path.is_file()}
    assert actual == set(expected)
    payload = 0
    for name, digest in expected.items():
        path = IMPORTED / name
        assert not path.is_symlink() and path.is_file() and sha(path) == digest, name
        payload += path.stat().st_size
    assert payload == transport['payload_bytes'] == 3120598370
    trajectory = read(REVIEW / 'trajectory_v1/trajectory_audit.json')
    metrics = read(REVIEW / 'metrics_v1/metrics_audit.json')
    guards = read(REVIEW / 'metrics_v1/guards.json')
    visual = read(REVIEW / 'visual_review_v1.json')
    assert trajectory['audit_finished'] and trajectory['evidence_consistency_pass'] and not trajectory['issues']
    assert trajectory['completed_updates'] == 50 and not trajectory['model_qualified'] and not trajectory['resume_authorized']
    assert trajectory['verified_transport']['receipt_sha256'] == sha(REVIEW / 'transport_v1.json')
    assert metrics['complete'] and metrics['protocol_sha256'] == PROTOCOL
    assert metrics['new_FP32_RAW_PNG_pairs'] == metrics['retained_BF16_RAW_PNG_pairs_rechecked'] == 812
    assert metrics['all406_baseline_RAW_exact'] and metrics['all406_baseline_PNG_exact']
    assert not metrics['FP32_clear_guard_pass'] and not metrics['FP32_blur_guard_pass']
    assert metrics['native_reference_metrics_null'] and metrics['reserved_final_pixel_reads'] == 0
    failures = [row for row in guards['clear_case_failures'] if row['state'] == 'FP32_update50']
    assert len(failures) == 3 and all(row['failed_views'] == ['raw', 'native_png'] for row in failures)
    assert visual['complete'] and visual['root_inspected_all_planned_pages'] == 52
    assert visual['case_pairs'] == 406 and visual['returned_outputs_represented'] == 812
    assert not visual['model_qualified'] and not visual['app_adoption']
    assert visual['binding_audit_sha256'] == sha(REVIEW / 'visual_binding_audit_v1.json')
    assert visual['root_notes_sha256'] == sha(REVIEW / 'root_visual_notes_v1.json')
    assert visual['sheet_manifest_sha256'] == sha(REVIEW / 'visual_sheets/manifest.json')
    for page in visual['pages']:
        assert sha(REVIEW / 'visual_sheets' / page['file']) == page['sha256']
    assert sha(visual['critical_figure']['path']) == visual['critical_figure']['sha256']
    approval = read(REVIEW / 'independent_report_readback_v2.json')
    assert approval['complete'] and approval['approved'] and not approval['findings']
    for name, digest in approval['reviewed_files_sha256'].items():
        assert sha(ROOT / name) == digest
    for name, digest in approval['evidence_sha256'].items():
        assert sha(REVIEW / name) == digest
    cross = read(REVIEW / 'independent_source_crossreview_v1.json')
    assert cross['complete'] and cross['approved']
    for row in cross['bindings'].values():
        assert sha(ROOT / row['path']) == row['sha256']
    assert sha(ROOT / 'CCTV_CODEFORMER_FP32_COMPARISON_V1_DESIGN.md') == '44da9c3afa4b0f0209ad762aa0f52c999f83054014bde076e39d46816f8a66fb'
    sources = ['research/codeformer_fp32_comparison_v1/return_archive.py',
        'research/codeformer_fp32_comparison_v1/return_metrics.py',
        'research/codeformer_fp32_comparison_v1/return_trajectory.py',
        'scripts/render_codeformer_fp32_comparison_return_v1.py',
        'scripts/close_codeformer_fp32_visual_review_v1.py']
    receipts = ['transport_v1.json', 'trajectory_v1/trajectory_audit.json',
        'metrics_v1/metrics_audit.json', 'metrics_v1/guards.json', 'metrics_v1/metrics.json',
        'metrics_v1/comparisons.json', 'metrics_v1/visual_manifest.json',
        'metrics_v1/scientific_readback.json', 'visual_review_v1.json', 'visual_binding_audit_v1.json',
        'independent_report_readback_v2.json', 'independent_source_crossreview_v1.json']
    result = {'complete': True, 'scope': 'Strict transport, saved states/data, independent RAW/PNG metrics and root visual review; no model qualification',
        'archive_sha256': transport['archive_sha256'], 'protocol_sha256': PROTOCOL,
        'manifest_sha256': transport['manifest_sha256'], 'payload_members_rehashed': len(expected),
        'payload_bytes_rehashed': payload, 'actual_updates': 50, 'actual_TRAIN_exposures': 200,
        'returned_outputs_independently_checked': 812, 'retained_BF16_outputs_checked': 812,
        'paired_cases': 358, 'native_unpaired_cases': 48, 'visual_pages_root_reviewed': 52,
        'clear_guard_pass': False, 'normal_blur_photometric_guard_pass': False,
        'originals_checkpoints_splits_failures_retained': True,
        'reports_sha256': approval['reviewed_files_sha256'],
        'sources_sha256': {name: sha(ROOT / name) for name in sources},
        'receipts_sha256': {name: sha(REVIEW / name) for name in receipts},
        'prior_BF16_evidence_sha256': {name: sha(ROOT / 'outputs/codeformer_adaptation_v2_return_review' / name)
            for name in ('archive_import_v1.json', 'publication_readback_v1.json', 'publication_static_readback_v1.json')},
        'neural_gradient_optimizer_VM_calls': 0, 'reserved_final_pixel_reads': 0,
        'model_qualified': False, 'unchanged_resume_authorized': False, 'goal_complete': False,
        'source_sha256': sha(__file__), 'seconds': time.monotonic() - start}
    with (REVIEW / 'closed_chain_v1.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({'complete': True, 'members': len(expected), 'quality_stop_retained': True, 'seconds': result['seconds']}))


if __name__ == '__main__':
    main()
