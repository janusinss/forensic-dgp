"""Close actually recorded development review; independently check saved pixels only."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import time

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_original_loss_balance_v1_vm'
RETURN = ROOT / 'outputs/cctv_dgp_original_loss_balance_v1_vm_return'
OUT = ROOT / 'outputs/cctv_dgp_original_loss_balance_v1_return_review_v1'
AUDIT = ROOT / 'outputs/cctv_dgp_original_loss_balance_v1_independent_audit.json'
PIN = '8e28ab1de4873bca2aff806a21ae173bc039c9463d5b2acf04848b031791d2d2'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, data):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    start = time.monotonic()
    assert not (OUT / 'visual_review.json').exists(), 'Retain the completed review'
    assert not (OUT / 'gallery_independent_audit.json').exists()
    assert sha(BUNDLE / 'protocol.json') == PIN
    protocol = read(BUNDLE / 'protocol.json')
    result = read(RETURN / 'outputs/results.json')
    audit = read(AUDIT)
    assert audit['complete'] and audit['protocol_sha256'] == PIN
    assert audit['all1000_raw_and_PNG_metrics_recomputed']
    assert audit['all36_stored_row_comparisons_and_failure_decisions_exact']
    assert audit['CPU_replay_cases'] == 200
    assert result['protocol_sha256'] == PIN and result['complete']
    assert result['epochs'] == result['optimizer_updates'] == result['gradient_queries'] == 0
    assert not result['native_or_DEV_or_final_used']
    assert result['states_before'] == result['states_after']
    decisions = [gate for trial in result['trial_summaries']
                 for stages in trial['comparisons'].values() for gate in stages.values()]
    assert len(decisions) == 36 and all(gate['pass'] is False for gate in decisions)
    for name, digest in protocol['local_sources_sha256'].items():
        assert sha(ROOT / name) == digest, name
    gallery = OUT / 'gallery'
    manifest_path = gallery / 'gallery_manifest.json'
    manifest = read(manifest_path)
    assert manifest['protocol_sha256'] == PIN
    assert manifest['results_sha256'] == sha(RETURN / 'outputs/results.json')
    assert manifest['independent_audit_sha256'] == sha(AUDIT)
    assert len(manifest['pages']) == 60
    for name, digest in manifest['source_bindings'].items():
        assert sha(ROOT / name) == digest, name
    cells, models = 0, set()
    for page in manifest['pages']:
        assert sha(gallery / page['file']) == page['sha256']
        with Image.open(gallery / page['file']) as sheet:
            assert sheet.mode == 'RGB' and list(sheet.size) == page['size'] == [1608, 1484]
            assert len(page['cells']) == 30
            for cell in page['cells']:
                with Image.open(ROOT / cell['file']) as source:
                    assert source.size == (256, 256)
                    assert sheet.crop(cell['box']).tobytes() == source.convert('RGB').tobytes()
                cells += 1
                if '_vm_return/outputs/' in cell['file']:
                    models.add(cell['file'])
        assert time.monotonic() - start < 120, 'Finite saved-pixel audit cap'
    assert cells == 1800 and len(models) == 1000
    groups, observation_hashes = [], {}
    for number in range(1, 21):
        path = OUT / 'observations' / f'{number:02d}.json'
        record = read(path)
        assert record['complete'] and record['group_index'] == number
        assert record['viewed_at_original_resolution']
        assert record['all_three_modes_and_three_scales_viewed']
        assert not record['independent_final_reviewer'] and not record['model_qualification']
        assert record['gallery_manifest_sha256'] == sha(manifest_path)
        assert record['record_script_sha256'] == sha(
            ROOT / 'scripts/record_cctv_dgp_original_loss_balance_v1_visual_group.py')
        pages = [page for page in manifest['pages'] if page['file'].startswith(f'{number:02d}-')]
        assert len(pages) == 3 and {page['scope'] for page in pages} == set(protocol['scopes'])
        assert record['pages_sha256'] == {page['file']: page['sha256'] for page in pages}
        assert all(record['case_ids'] == page['case_ids'] for page in pages)
        assert all(record['reference'] == page['reference'] and record['cohort'] == page['cohort']
                   for page in pages)
        assert len(record['note']) >= 30
        observation_hashes[path.relative_to(ROOT).as_posix()] = sha(path)
        groups.append(record)
    assert {cid for group in groups for cid in group['case_ids']} == {
        case['id'] for case in protocol['cases']}
    gallery_audit = {
        'complete': True, 'protocol_sha256': PIN,
        'gallery_manifest_sha256': sha(manifest_path),
        'independent_evidence_audit_sha256': sha(AUDIT),
        'source_bindings_verified': len(manifest['source_bindings']),
        'page_hashes_verified': 60, 'exact256_cells_verified': cells,
        'unique_model_outputs_verified': len(models),
        'observation_bindings': observation_hashes,
        'all20_recorded_reviews_bound_to_exact_pages': True,
        'checker_sha256': sha(Path(__file__)), 'neural_calls': 0,
        'gradient_queries': 0, 'optimizer_updates': 0,
        'seconds': time.monotonic() - start,
    }
    assert gallery_audit['seconds'] < 120
    write(OUT / 'gallery_independent_audit.json', gallery_audit)
    visual = {
        'complete': True, 'reviewer': 'Primary assistant development review',
        'independent_final_reviewer': False, 'protocol_sha256': PIN,
        'results_sha256': sha(RETURN / 'outputs/results.json'),
        'independent_evidence_audit_sha256': sha(AUDIT),
        'gallery_manifest_sha256': sha(manifest_path),
        'gallery_independent_audit_sha256': sha(OUT / 'gallery_independent_audit.json'),
        'all60_pages_viewed_at_original_resolution': True,
        'unique_model_outputs_reviewed': len(models),
        'exact256_cells_independently_verified': cells,
        'page_pixel_compositions_exact': True, 'groups': groups,
        'remaining_group_indices': [],
        'decision': 'All nine trials unqualified. Balanced outputs remain soft, with no convincing useful clarity gain, and preservation still fails. No promotion or epoch continuation.',
        'invalid_assumption': 'Mean first-order identity protection would preserve finite outputs in every source/profile and clear-image group.',
        'native_or_DEV_or_final_used': False, 'neural_calls': 0,
        'gradient_queries': 0, 'optimizer_updates': 0,
        'model_qualification': False, 'goal_complete': False,
        'closed_UTC': datetime.now(timezone.utc).isoformat(),
    }
    write(OUT / 'visual_review.json', visual)
    print({'complete': True, 'reviewed_outputs': 1000, 'pages': 60,
           'exact_pixel_cells': cells, 'all_trials_unqualified': True,
           'seconds': gallery_audit['seconds']}, flush=True)


if __name__ == '__main__':
    main()
