"""Check exact gallery pixels and bind actually recorded development reviews."""
import json
import time
from datetime import datetime, timezone
from PIL import Image
from prepare_cctv_dgp_group_conflicts_v1_visual_review import ROOT, BUNDLE, RETURN, OUT, AUDIT, PIN, read, sha


def write(path, data):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    started = time.monotonic()
    folder = OUT.parent
    assert not (folder / 'visual_review.json').exists()
    assert not (folder / 'gallery_independent_audit.json').exists()
    assert sha(BUNDLE / 'protocol.json') == PIN
    protocol, result, audit = read(BUNDLE / 'protocol.json'), read(RETURN / 'outputs/results.json'), read(AUDIT)
    assert audit['complete'] and audit['all_saved_raw_and_PNG_metrics_recomputed']
    assert audit['all_stored_row_comparisons_and_failure_decisions_exact']
    assert audit['comparisons_verified'] == 12 and audit['CPU_replay_cases'] == 80
    assert result['gradient_queries'] == 160 and result['optimizer_updates'] == result['epochs'] == 0
    assert result['states_before'] == result['states_after'] and not result['native_or_DEV_or_final_used']
    decisions = [gate for trial in result['trial_summaries']
        for stages in trial['comparisons'].values() for gate in stages.values()]
    assert len(decisions) == 12 and all(gate['pass'] is False for gate in decisions)
    for name, digest in protocol['local_sources_sha256'].items():
        assert sha(ROOT / name) == digest, name
    manifest_path = OUT / 'gallery_manifest.json'
    manifest = read(manifest_path)
    assert manifest['protocol_sha256'] == PIN and manifest['results_sha256'] == sha(RETURN / 'outputs/results.json')
    assert manifest['independent_audit_sha256'] == sha(AUDIT) and len(manifest['pages']) == 20
    for name, digest in manifest['source_bindings'].items():
        assert sha(ROOT / name) == digest, name
    cells, model_files, observations, observation_hashes = 0, set(), [], {}
    for number, page in enumerate(manifest['pages'], 1):
        assert sha(OUT / page['file']) == page['sha256'] and page['file'] == f'{number:02d}-common.png'
        with Image.open(OUT / page['file']) as sheet:
            assert sheet.mode == 'RGB' and list(sheet.size) == page['size'] == [1608, 1484]
            assert len(page['cells']) == 30
            for cell in page['cells']:
                with Image.open(ROOT / cell['file']) as source:
                    assert source.size == (256, 256)
                    assert sheet.crop(cell['box']).tobytes() == source.convert('RGB').tobytes()
                cells += 1
                if '_vm_return/outputs/' in cell['file']:
                    model_files.add(cell['file'])
        path = folder / 'observations' / f'{number:02d}.json'
        observation = read(path)
        assert observation['complete'] and observation['group_index'] == number
        assert observation['viewed_at_original_resolution'] and observation['all_three_reset_scales_viewed']
        assert not observation['independent_final_reviewer'] and not observation['model_qualification']
        assert observation['gallery_manifest_sha256'] == sha(manifest_path)
        assert observation['page_sha256'] == page['sha256']
        assert observation['case_ids'] == page['case_ids']
        assert observation['cohort'] == page['cohort'] and observation['reference'] == page['reference']
        assert observation['record_script_sha256'] == sha(ROOT / 'scripts/record_cctv_dgp_group_conflicts_v1_visual_group.py')
        assert len(observation['note']) >= 30
        observations.append(observation)
        observation_hashes[path.relative_to(ROOT).as_posix()] = sha(path)
        assert time.monotonic() - started < 120
    assert cells == 600 and len(model_files) == 400
    assert {cid for observation in observations for cid in observation['case_ids']} == {case['id'] for case in protocol['cases']}
    write(folder / 'gallery_independent_audit.json', {
        'complete': True, 'protocol_sha256': PIN, 'gallery_manifest_sha256': sha(manifest_path),
        'independent_evidence_audit_sha256': sha(AUDIT), 'source_bindings_verified': len(manifest['source_bindings']),
        'page_hashes_verified': 20, 'exact256_cells_verified': cells, 'unique_model_outputs_verified': len(model_files),
        'all20_recorded_reviews_bound_to_exact_pages': True, 'observation_bindings': observation_hashes,
        'subjective_visual_judgments_proved_by_code': False, 'checker_sha256': sha(ROOT / 'scripts/close_cctv_dgp_group_conflicts_v1_visual_review.py'),
        'neural_calls': 0, 'gradient_queries': 0, 'optimizer_updates': 0, 'seconds': time.monotonic() - started})
    write(folder / 'visual_review.json', {
        'complete': True, 'reviewer': 'Primary assistant development review', 'independent_final_reviewer': False,
        'protocol_sha256': PIN, 'results_sha256': sha(RETURN / 'outputs/results.json'),
        'independent_evidence_audit_sha256': sha(AUDIT), 'gallery_manifest_sha256': sha(manifest_path),
        'gallery_independent_audit_sha256': sha(folder / 'gallery_independent_audit.json'),
        'all20_pages_actually_viewed_at_original_resolution': True,
        'unique_model_outputs_reviewed': 400, 'exact256_cells_independently_verified': cells,
        'groups': observations, 'remaining_group_indices': [],
        'decision': 'All three reset trials unqualified. Small steps show no convincing useful detail gain; the largest adds tone/edge artifacts and worsens structure. No promotion or epoch continuation.',
        'invalid_assumption': 'A shared initial derivative descent direction would produce finite, preservation-safe gains across source/profile and cross-cohort faces.',
        'native_or_DEV_or_final_used': False, 'model_qualification': False, 'goal_complete': False,
        'neural_calls': 0, 'gradient_queries': 0, 'optimizer_updates': 0,
        'closed_UTC': datetime.now(timezone.utc).isoformat()})
    print({'complete': True, 'reviewed_outputs': 400, 'pages': 20, 'exact_pixel_cells': cells,
        'all_trials_unqualified': True, 'seconds': time.monotonic() - started}, flush=True)


if __name__ == '__main__':
    main()
