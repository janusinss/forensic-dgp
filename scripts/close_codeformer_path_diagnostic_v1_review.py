"""Bind the verified return, CPU audits and completed root visual review."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'outputs/codeformer_path_diagnostic_v1_return_review'
RETURN = ROOT / 'outputs/codeformer_path_diagnostic_v1_return'
MANIFEST = 'outputs/path_diagnostic_v1/export_manifest.json'
MANIFEST_SHA = 'be4be94c889b38a0d08346cf9b0c386e889168b9a34d8392a4d25b36dd3b8ea4'
PROTOCOL_SHA = '2d3dcd1dbc3be371de5e10ed9c059e455ce9cc8376bac99f04a1a514acc0d49f'
ARCHIVE_SHA = 'c8ebadef8286ecd56c3080f7ec1311ef209fd3783072a542d46319bdca84b772'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    transport = read(REVIEW / 'transport_v1.json')
    assert transport['complete'] and transport['transport_verified']
    assert Path(transport['destination']).resolve() == RETURN.resolve()
    assert transport['manifest_sha256'] == MANIFEST_SHA
    assert transport['protocol_sha256'] == PROTOCOL_SHA
    assert transport['archive_sha256'] == ARCHIVE_SHA
    assert sha(ROOT / 'outputs/cctv-codeformer-path-diagnostic-v1-results.tar.gz') == ARCHIVE_SHA
    manifest_path = RETURN / MANIFEST
    assert sha(manifest_path) == MANIFEST_SHA
    manifest = read(manifest_path)
    expected = manifest['files_sha256']
    actual = {p.relative_to(RETURN).as_posix() for p in RETURN.rglob('*') if p.is_file()}
    assert actual == set(expected) | {MANIFEST}
    assert len(actual) == transport['members'] == 1243
    for name, digest in expected.items():
        path = RETURN / name
        assert not path.is_symlink() and sha(path) == digest
        assert path.stat().st_size == manifest['files_bytes'][name]
    assert sha(RETURN / 'protocol.json') == PROTOCOL_SHA
    packet = ROOT / 'outputs/cctv_codeformer_path_diagnostic_v1_vm'
    assert sha(packet / 'protocol.json') == PROTOCOL_SHA
    protocol = read(packet / 'protocol.json')
    for name, digest in protocol['files_sha256'].items():
        assert sha(packet / name) == digest
    audits = {'state': REVIEW / 'state_v1/state_audit.json',
              'pixels': REVIEW / 'pixels_v2/pixel_audit.json'}
    state, pixels = (read(audits[k]) for k in ('state', 'pixels'))
    assert state['complete'] and pixels['complete']
    assert state['protocol_sha256'] == pixels['protocol_sha256'] == PROTOCOL_SHA
    assert state['auditor_source_sha256'] == sha(ROOT / 'research/codeformer_path_diagnostic_v1/return_state.py')
    assert pixels['source_sha256'] == sha(ROOT / 'research/codeformer_path_diagnostic_v1/return_pixels.py')
    assert state['optimizer_updates'] == 0 and pixels['neural_or_gradient_or_optimizer_calls'] == 0
    independent = read(REVIEW / 'independent_readback_v1.json')
    visual = read(REVIEW / 'pixels_v2/visual_manifest.json')['conditions']
    assert len(visual) == 104
    by_png = {str(Path(r['native_png']).resolve()): r for r in visual}
    assert len(by_png) == 104
    render_path = REVIEW / 'visual_sheets/render_receipt.json'
    render = read(render_path)
    assert render['complete'] and render['conditions_covered'] == 104 and len(render['sheets']) == 8
    seen, closed_sheets = set(), []
    for sheet in render['sheets']:
        path = Path(sheet['path'])
        assert sha(path) == sheet['sha256']
        kind = 'intervention' if path.name.startswith('interventions') else ('repeat' if path.name.startswith('repeats') else 'path')
        conditions = []
        for binding in sheet['bindings']:
            assert sha(binding['path']) == binding['sha256']
            row = by_png.get(str(Path(binding['path']).resolve()))
            if row and row['condition_kind'] == kind:
                conditions.append(row['condition'])
                seen.add(row['condition'])
        closed_sheets.append({'path': str(path), 'sha256': sheet['sha256'],
                              'conditions': conditions, 'root_visually_reviewed': True})
    assert seen == {r['condition'] for r in visual}
    critical = [
        'paths/original/train_u0001_slot1/E32', 'paths/original/train_u0001_slot1/TBF16',
        'paths/stopped50/train_u0001_slot1/E32',
        'interventions/train_u0001_slot1/original_moments__original_SFT',
        'paths/stopped50/train_u0001_slot3/E32',
        'interventions/train_u0001_slot3/current_moments__original_SFT',
        'paths/stopped50/train_u0024_slot4/E32',
        'interventions/train_u0024_slot4/current_moments__original_SFT']
    by_condition = {r['condition']: r for r in visual}
    full = [{'condition': c, 'path': by_condition[c]['native_png'],
             'sha256': sha(by_condition[c]['native_png']), 'viewed_at_native256': True} for c in critical]
    review = {'complete': True, 'reviewer': 'root', 'date': '2026-10-16',
        'all_planned_conditions': 104, 'sheets': closed_sheets, 'native256_spot_checks': full,
        'notes': [
            'All eight sheets inspected. FP32 columns visually match; repeats add no visible variation, consistent with exact arrays.',
            'Original severe24 output adds coarse facial texture and changes mouth/expression relative to the clean reference; stopped50 looks smoother and closer on this TRAIN example.',
            'Stopped50 brightens dark cases and makes features easier to see; some outputs lose tonal depth and look flat or pale. Brightness alone is not the quality verdict.',
            'Clear eyeglasses remain visible in the sampled paths, with small warmth/smoothness differences; these four TRAIN references cannot settle DEV clear preservation.',
            'Returning original fusion removes part of the useful dark-image exposure; no intervention resolves every sampled content, contrast and photometric issue.'],
        'model_qualified': False, 'user_acceptance_claimed': False,
        'display': 'Saved native PNGs at192px on sheets; eight actual PNGs also viewed at256. No image enhancement.',
        'coverage_note': 'Initial render receipt per-sheet conditions used cumulative tracking. This closed receipt reconstructs exact per-sheet membership from hashed image bindings; all image bytes are unchanged.'}
    save(REVIEW / 'visual_review_v1.json', review)
    bound_paths = [REVIEW / 'transport_v1.json', *audits.values(),
        REVIEW / 'pixels_v2/condition_metrics.json', REVIEW / 'pixels_v2/comparisons.json',
        REVIEW / 'pixels_v2/scientific_readback.json', REVIEW / 'pixels_v2/visual_manifest.json',
        REVIEW / 'state_v1/gradient_components.json', REVIEW / 'state_v1/gradient_cosines.json',
        REVIEW / 'independent_readback_v1.json', REVIEW / 'visual_review_v1.json', render_path,
        ROOT / 'scripts/audit_codeformer_path_diagnostic_v1_transport.py',
        ROOT / 'research/codeformer_path_diagnostic_v1/return_pixels.py',
        ROOT / 'research/codeformer_path_diagnostic_v1/return_state.py']
    save(REVIEW / 'closed_chain_v1.json', {'complete': True, 'archive_sha256': ARCHIVE_SHA,
        'manifest_sha256': MANIFEST_SHA, 'protocol_sha256': PROTOCOL_SHA,
        'imported_namespace_and_hashes_reverified': len(actual), 'packet_assets_reverified': len(protocol['files_sha256']),
        'successful_transport_destination_bound': True, 'scientific_audits_bound_to_import': True,
        'visual_conditions_reviewed': len(seen), 'root_reviewed_sheets': len(closed_sheets),
        'files_sha256': {str(p.relative_to(ROOT)): sha(p) for p in bound_paths},
        'independent_readback_present': bool(independent), 'source_sha256': sha(__file__),
        'optimizer_updates_in_diagnostic': 0, 'local_neural_gradient_optimizer_VM_calls': 0,
        'model_qualified': False, 'app_promoted': False, 'goal_complete': False})
    print(json.dumps({'complete': True, 'files_reverified': len(actual), 'visual_conditions': len(seen)}))


if __name__ == '__main__':
    main()
