"""Fresh CPU-only closure of one completed AdaIN return audit, never model acceptance.

The report and root visual record must already exist. All copied return members,
the frozen local overlay, resolved immutable dependencies/weights and audit
receipts are freshly hashed. No returned source or checkpoint is executed.
"""
import argparse
from pathlib import Path
import os
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_adain_gradient_isolation_v1.return_archive import (
    PROTOCOL_SHA, ARCHIVE_SHA, MANIFEST_SHA, MANIFEST, no_links, regular, namespace,
    read, sha, require, write_new)

CRITICAL = (
    'transport_v1.json', 'transport_cpu_tests_v1.json', 'transport_checker_diagnosis_v1.json',
    'closure_CPU_tests_v1.json', 'closure_source_before_final_bindings.py', 'all27_CPU_tests_v1.json',
    'metrics_CPU_tests_v1.json', 'decomposition_CPU_tests_v1.json',
    'state_v1/state_audit.json', 'state_v1/independent_findings.json', 'state_v1/STATE_FINDINGS.md',
    'state_v1/CPU_tests_v1.json', 'state_v1/benchmark20_capture_readback_v1.json',
    'metrics_v1/metrics_audit.json', 'metrics_v1/metrics.json', 'metrics_v1/metrics.csv',
    'metrics_v1/guards.json', 'metrics_v1/comparisons.json', 'metrics_v1/witness_pixels.json',
    'metrics_v1/visual_manifest.json', 'metrics_v1/input_audit.json', 'metrics_v1/native_descriptors.json',
    'prior_comparison_binding_v1.json', 'scientific_readback_v1.json', 'SCIENTIFIC_READBACK_V1.md',
    'clear_error_decomposition_v1.json', 'critical_clear_256_bindings.json',
    'independent_scientific_readback_v1.json', 'independent_summary_render_readback_v1.json',
    'next_design_review_v1.json', 'next_design_review_v1.md', 'independent_next_design_readback_v1.json',
    'input_usability_requirement_v1.md', 'input_usability_requirement_v1.json',
    'independent_root_visual_record_v1.json', 'visual_sheets/manifest.json',
    'visual_sheets/INDEX.md', 'independent_report_readback_v1.json',
)


def relative_files(directory):
    found = set()
    for folder, dirs, names in os.walk(directory, followlinks=False):
        for name in dirs: no_links(Path(folder) / name)
        for name in names:
            path = Path(folder) / name; regular(path)
            found.add(path.relative_to(directory).as_posix())
    namespace(found)
    return found


def visual_contract(base, sheetroot, manifest, visual):
    require(visual.get('complete') is True and visual.get('root_personally_viewed_all_planned_sheets') is True and
        visual.get('model_qualified') is False, 'Complete root personal visual review without qualification required')
    sheets = visual.get('sheets')
    require(isinstance(sheets, list) and len(sheets) == len(manifest['pages']), 'Visual review exact page census differs')
    expected = {page['file']: page for page in manifest['pages']}; actual = {}
    for sheet in sheets:
        path = Path(sheet['path']); path = path if path.is_absolute() else base / path
        path = no_links(path).resolve(strict=True)
        name = path.name
        require(name in expected and name not in actual and path == (sheetroot / name).resolve(strict=True),
            'Root viewed page is duplicate or outside planned sheets')
        page = expected[name]
        require(sheet.get('sha256') == page['sha256'] and sheet.get('root_personally_viewed') is True and
            sheet.get('case_ids') == page['case_ids'], 'Root visual image/case evidence differs')
        actual[name] = sheet
    require(set(actual) == set(expected), 'Root visual record is missing planned sheets')
    return len(actual)


def close(report, visual, output, *, seconds=900):
    started = time.monotonic(); review = ROOT / 'outputs/codeformer_adain_gradient_isolation_v1_return_review'
    packet = ROOT / 'outputs/cctv_codeformer_adain_gradient_isolation_v1_vm'
    imported = ROOT / 'outputs/codeformer_adain_gradient_isolation_v1_return'
    report, visual, output = (no_links(Path(path).absolute()) for path in (report, visual, output))
    require(type(seconds) is int and 0 < seconds <= 1800, 'Finite CPU closure cap required')
    require(not output.exists(), 'Retain prior closure receipt')
    for path in (report, visual): regular(path)
    bindings = {}
    def budget(): require(time.monotonic() - started <= seconds, 'Finite local CPU closure cap exceeded')
    def bind(path, expected=None):
        budget(); path = no_links(Path(path).absolute()); regular(path)
        require(path.resolve(strict=True).is_relative_to(ROOT.resolve(strict=True)), 'Evidence outside workspace')
        value = sha(path, budget)
        if expected is not None: require(value == expected, 'Evidence fingerprint differs: ' + str(path))
        key = path.relative_to(ROOT).as_posix()
        if key in bindings: require(bindings[key] == value, 'Evidence changed during closure')
        bindings[key] = value
        return value
    transport = read(review / 'transport_v1.json')
    require(transport['complete'] is True and transport['protocol_sha256'] == PROTOCOL_SHA and
        transport['archive_sha256'] == ARCHIVE_SHA and transport['manifest_sha256'] == MANIFEST_SHA and
        transport['manifest_member'] == MANIFEST and transport['exact_namespace_verified'] is True and
        transport['whole_archive_and_all_member_hashes_verified_before_import'] is True and
        transport['reported_completed_updates'] == 50 and transport['model_qualified'] is False,
        'Pinned completed diagnostic transport required')
    expected = transport['files_sha256']; sizes = transport['files_bytes']; namespace(expected)
    require(len(expected) == transport['members'] == 6558 and set(expected) == set(sizes) and
        relative_files(imported) == set(expected), 'Exact imported return namespace changed')
    for name, digest in expected.items():
        path = imported / name
        require(regular(path).st_size == sizes[name], 'Imported file bytes changed')
        bind(path, digest)
    require(sum(sizes.values()) == transport['payload_bytes'] == 3143169457, 'Imported payload census differs')
    bind(packet / 'protocol.json', PROTOCOL_SHA); p = read(packet / 'protocol.json')
    expected_overlay = (set(p['files_sha256']) - set(p['dependency_files_sha256'])) | {'protocol.json'}
    require(relative_files(packet) == expected_overlay and len(expected_overlay) == 31,
        'Frozen local overlay exact namespace changed')
    for name in expected_overlay - {'protocol.json'}: bind(packet / name, p['files_sha256'][name])
    resolved = transport['immutable_sources_and_assets_freshly_verified']
    require(set(resolved) == set(p['files_sha256']) and len(resolved) == 1939, 'Resolved dependency map differs')
    for name, record in resolved.items():
        require(record['sha256'] == p['files_sha256'][name] and regular(record['resolved_path']).st_size == record['bytes'],
            'Resolved frozen dependency binding differs')
        bind(record['resolved_path'], record['sha256'])
    assets = {}
    for key, record in p['assets'].items():
        pinned = resolved[record['path']]
        require(pinned['bytes'] == record['bytes'] and pinned['sha256'] == record['sha256'], 'Critical original asset differs')
        assets[key] = {name: pinned[name] for name in ('resolved_path', 'sha256', 'bytes')}
    for name in CRITICAL: bind(review / name)
    state, pixel = read(review / 'state_v1/state_audit.json'), read(review / 'metrics_v1/metrics_audit.json')
    require(state['complete'] is state['evidence_consistency_pass'] is True and state['issues'] == [] and
        state['training_finished'] is False and state['resume_authorized'] is False and
        state['completed_updates'] == 50 and pixel['complete'] is True and
        pixel['isolated_clear_guard_pass'] is pixel['isolated_blur_guard_pass'] is False,
        'Do not transform quality failure into audit/model success')
    for record in (state, pixel):
        require(record['model_qualified'] is False and record['protocol_sha256'] == PROTOCOL_SHA,
            'Audit qualification/protocol changed')
        for path, digest in record['source_sha256'].items(): bind(path, digest)
    for name in ('transport_cpu_tests_v1.json', 'state_v1/CPU_tests_v1.json', 'metrics_CPU_tests_v1.json',
                 'decomposition_CPU_tests_v1.json', 'all27_CPU_tests_v1.json'):
        value = read(review / name); require(value['complete'] is True, 'Closed CPU test receipt required')
        for path, digest in value['source_sha256'].items(): bind(ROOT / path, digest)
    for name in ('independent_scientific_readback_v1.json', 'independent_summary_render_readback_v1.json'):
        value = read(review / name)
        require(value['complete'] is True and value['model_qualified'] is False and value['training_authorized'] is False,
            'Independent readback must retain failed quality status')
    independent = read(review / 'independent_scientific_readback_v1.json')
    summary_readback = read(review / 'independent_summary_render_readback_v1.json')
    source_paths = (
        'research/codeformer_adain_gradient_isolation_v1/return_archive.py',
        'research/codeformer_adain_gradient_isolation_v1/return_scientific_crossreview.py',
        'research/codeformer_adain_gradient_isolation_v1/return_error_decomposition.py',
        'scripts/crossreview_codeformer_adain_saved_summary_v1.py',
        'scripts/summarize_codeformer_adain_saved_pixels_v1.py',
        'scripts/render_codeformer_adain_gradient_isolation_return_v1.py',
        'tests/test_codeformer_adain_error_decomposition_v1.py',
        'tests/test_codeformer_adain_gradient_isolation_return_archive_v1.py',
        'tests/test_codeformer_adain_return_closure_v1.py',
        'scripts/close_codeformer_adain_visual_review_v1.py',
        'scripts/verify_codeformer_adain_root_visual_record_v1.py',
    )
    for name in source_paths: bind(ROOT / name)
    bind(ROOT / 'research/codeformer_adain_gradient_isolation_v1/return_archive.py', transport['source_sha256'])
    bind(ROOT / 'research/codeformer_adain_gradient_isolation_v1/return_scientific_crossreview.py', independent['source_sha256'])
    bind(ROOT / 'scripts/crossreview_codeformer_adain_saved_summary_v1.py', summary_readback['source_sha256'])
    decomposition = read(review / 'clear_error_decomposition_v1.json')
    bind(ROOT / 'research/codeformer_adain_gradient_isolation_v1/return_error_decomposition.py', decomposition['source_sha256'])
    for path, digest in decomposition['read_sources_sha256'].items(): bind(path, digest)
    design = read(review / 'next_design_review_v1.json')
    require(design['complete'] is True and design['status'] == 'DESIGN_REVIEW_ONLY_NO_PACKET_UNRUN' and
        design['training_authorized'] is design['packet_prepared'] is design['model_qualified'] is False,
        'Design-only status changed')
    bind(review / 'next_design_review_v1.md', design['markdown_sha256'])
    for path, digest in design['evidence_sha256'].items(): bind(ROOT / path, digest)
    usage = read(review / 'input_usability_requirement_v1.json')
    require(usage['complete'] is True and
        usage['status'] == 'REQUIREMENT_RECORDED_AUTOMATIC_GATE_NOT_IMPLEMENTED_OR_CALIBRATED' and
        usage['automatic_structure_gate_qualified'] is usage['app_changed'] is usage['model_qualified'] is False and
        usage['visible_feature_interpretation']['all_features_need_not_be_visible'] is True and
        usage['visible_feature_interpretation']['covered_mouth_or_nose_alone_not_automatic_rejection'] is True,
        'Prospective usability requirement or actually-visible-feature interpretation changed')
    bind(review / 'input_usability_requirement_v1.md', usage['markdown_sha256'])
    for path, digest in usage['source_and_evidence_sha256'].items(): bind(ROOT / path, digest)
    for kind in ('markdown', 'json'):
        bind(ROOT / usage['prior_revision'][kind + '_path'], usage['prior_revision'][kind + '_sha256'])
    historic_tests = read(review / 'closure_CPU_tests_v1.json')
    require(historic_tests['complete'] is True and historic_tests['CPU_tests'] == 4, 'Retained closure fixture proof differs')
    bind(review / 'closure_source_before_final_bindings.py',
         historic_tests['source_sha256']['scripts/close_codeformer_adain_return_chain_v1.py'])
    alltests = read(review / 'all27_CPU_tests_v1.json')
    require(alltests['complete'] is True and alltests['CPU_tests'] == 27 and alltests['returncode'] == 0,
        'All27 bounded current CPU fixtures must pass')
    sheetroot = review / 'visual_sheets'; manifest = read(sheetroot / 'manifest.json')
    require(manifest['complete'] is True and len(manifest['pages']) == 55, 'Complete55-sheet manifest required')
    viewed = visual_contract(ROOT, sheetroot, manifest, read(visual))
    require(manifest['source_visual_manifest_sha256'] == bind(review / 'metrics_v1/visual_manifest.json') and
        manifest['source_metrics_audit_sha256'] == bind(review / 'metrics_v1/metrics_audit.json'), 'Visual source chain differs')
    for page in manifest['pages']: bind(sheetroot / page['file'], page['sha256'])
    for path in sorted((review / 'root_visual_batches').glob('batch_*.json')): bind(path)
    visual_sha, report_sha = bind(visual), bind(report)
    root_visual = read(visual)
    bind(ROOT / 'scripts/close_codeformer_adain_visual_review_v1.py', root_visual['source_sha256'])
    root_record_check = read(review / 'independent_root_visual_record_v1.json')
    require(root_record_check['complete'] is True and root_record_check['root_visual_review_sha256'] == visual_sha and
        root_record_check['root_only_personal_review_claim_preserved'] is True and
        root_record_check['personal_visual_verdict_by_this_verifier'] is False,
        'Independent root visual-record binding differs')
    bind(ROOT / 'scripts/verify_codeformer_adain_root_visual_record_v1.py', root_record_check['source_sha256'])
    critical_figure = root_visual['critical_figure']
    bind(ROOT / critical_figure['path'], critical_figure['sha256'])
    readback = read(review / 'independent_report_readback_v1.json')
    require(readback.get('complete') is True and readback.get('approved_for_publication') is True and
        readback.get('report_sha256') == report_sha and readback.get('visual_review_sha256') == visual_sha and
        readback.get('model_qualified') is False, 'Independent factual report/visual readback required')
    for path, digest in readback.get('evidence_sha256', {}).items(): bind(ROOT / path, digest)
    bind(__file__)
    result = {'format': 'codeformer-adain-isolation-return-closed-chain-v1', 'complete': True,
        'protocol_sha256': PROTOCOL_SHA, 'archive_sha256': ARCHIVE_SHA, 'manifest_sha256': MANIFEST_SHA,
        'imported_regular_files_freshly_rehashed': len(expected), 'payload_bytes': sum(sizes.values()),
        'exact_imported_namespace': True, 'frozen_local_overlay_regular_files': len(expected_overlay),
        'all1939_immutable_sources_dependencies_freshly_rehashed': True,
        'critical_original_assets_freshly_rehashed': assets, 'root_personally_reviewed_sheets': viewed,
        'report_sha256': report_sha, 'visual_review_sha256': visual_sha,
        'evidence_sha256': bindings, 'seconds': time.monotonic() - started,
        'completed_updates': 50, 'training_status': 'Stopped on unchanged RAW/PNG clear-preservation guard',
        'independent_blur_guard_also_failed': True, 'scientific_audit_consistent_not_model_qualification': True,
        'no_returned_code_or_checkpoint_executed': True, 'neural_gradient_optimizer_VM_calls': 0,
        'new_training_or_resume_authorized': False, 'model_qualified': False, 'overall_goal_complete': False}
    write_new(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT / 'CCTV_CODEFORMER_ADAIN_GRADIENT_ISOLATION_V1_RETURN_REVIEW.md')
    parser.add_argument('--visual', type=Path, default=ROOT / 'outputs/codeformer_adain_gradient_isolation_v1_return_review/visual_review_v1.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/codeformer_adain_gradient_isolation_v1_return_review/closed_chain_v1.json')
    parser.add_argument('--seconds', type=int, default=900)
    args = parser.parse_args(); result = close(args.report, args.visual, args.output, seconds=args.seconds)
    print({key: result[key] for key in ('complete', 'imported_regular_files_freshly_rehashed', 'seconds', 'model_qualified')})
