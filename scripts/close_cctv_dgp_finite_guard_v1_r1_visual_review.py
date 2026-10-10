"""Verify all exact gallery cells and bind the actual development observations."""
from datetime import datetime,timezone
from pathlib import Path
import time
from PIL import Image
from prepare_cctv_dgp_finite_guard_v1_r1_visual_review import ROOT,BUNDLE,RETURN,OUT,AUDIT,PIN,sha,read,write


def main():
    started = time.monotonic(); folder = OUT.parent
    assert not (folder/'visual_review.json').exists() and not (folder/'gallery_independent_audit.json').exists()
    p,audit,result = read(BUNDLE/'protocol.json'),read(AUDIT),read(RETURN/'outputs/results.json')
    assert sha(BUNDLE/'protocol.json') == PIN and audit['complete']
    assert audit['all_stored_row_comparisons_and_failure_decisions_exact'] and audit['comparisons_verified'] == 24
    assert result['optimizer_updates'] == 0 and result['states_before'] == result['states_after_restore']
    for name,digest in p['local_sources_sha256'].items(): assert sha(ROOT/name) == digest,name
    manifest_path = OUT/'gallery_manifest.json'; manifest = read(manifest_path)
    assert manifest['protocol_sha256'] == PIN and manifest['results_sha256'] == sha(RETURN/'outputs/results.json')
    assert manifest['independent_audit_sha256'] == sha(AUDIT)
    for name,digest in manifest['source_bindings'].items(): assert sha(ROOT/name) == digest,name
    cells,model_files,observations,bindings = 0,set(),[],{}
    for number,page in enumerate(manifest['pages'],1):
        assert sha(OUT/page['file']) == page['sha256'] and page['file'] == f'{number:02d}-finite.png'
        with Image.open(OUT/page['file']) as sheet:
            assert sheet.mode == 'RGB' and list(sheet.size) == page['size'] == [1608,1484]
            assert len(page['cells']) == 30
            for cell in page['cells']:
                with Image.open(ROOT/cell['file']) as source:
                    assert source.size == (256,256) and sheet.crop(cell['box']).tobytes() == source.convert('RGB').tobytes()
                cells += 1
                if '_vm_return/outputs/' in cell['file']: model_files.add(cell['file'])
        path = folder/'observations'/f'{number:02d}.json'; obs = read(path)
        assert obs['complete'] and obs['group_index'] == number and obs['viewed_at_original_resolution']
        assert obs['all_three_proposals_viewed'] and not obs['independent_final_reviewer'] and not obs['model_qualification']
        assert obs['gallery_manifest_sha256'] == sha(manifest_path) and obs['page_sha256'] == page['sha256']
        assert obs['case_ids'] == page['case_ids'] and obs['cohort'] == page['cohort'] and obs['reference'] == page['reference']
        assert obs['record_script_sha256'] == sha(ROOT/'scripts/record_cctv_dgp_finite_guard_v1_r1_visual_group.py')
        assert len(obs['note']) >= 30
        observations.append(obs); bindings[path.relative_to(ROOT).as_posix()] = sha(path)
        assert time.monotonic()-started < 120
    assert len(observations) == 20 and cells == 600 and len(model_files) == 400
    assert {cid for obs in observations for cid in obs['case_ids']} == {c['id'] for c in p['cases']}
    write(folder/'gallery_independent_audit.json',{'complete':True,'protocol_sha256':PIN,
        'gallery_manifest_sha256':sha(manifest_path),'independent_evidence_audit_sha256':sha(AUDIT),
        'source_bindings_verified':len(manifest['source_bindings']),'page_hashes_verified':20,
        'exact256_cells_verified':cells,'unique_model_outputs_verified':len(model_files),
        'all20_observations_bound_to_exact_pages':True,'observation_bindings':bindings,
        'subjective_visual_judgments_proved_by_code':False,'checker_sha256':sha(Path(__file__)),
        'neural_calls':0,'gradient_queries':0,'optimizer_updates':0,'seconds':time.monotonic()-started})
    write(folder/'visual_review.json',{'complete':True,'reviewer':'Primary assistant development review',
        'independent_final_reviewer':False,'protocol_sha256':PIN,'results_sha256':sha(RETURN/'outputs/results.json'),
        'independent_evidence_audit_sha256':sha(AUDIT),'gallery_manifest_sha256':sha(manifest_path),
        'gallery_independent_audit_sha256':sha(folder/'gallery_independent_audit.json'),
        'all20_pages_actually_viewed_at_original_resolution':True,'unique_model_outputs_reviewed':400,
        'exact256_cells_independently_verified':600,'groups':observations,'remaining_group_indices':[],
        'decision':'Three small proposals remain unqualified; useful new detail not demonstrated. No accepted training change, app adoption or epoch continuation.',
        'native_or_DEV_or_final_used':False,'model_qualification':False,'goal_complete':False,
        'neural_calls':0,'gradient_queries':0,'optimizer_updates':0,'closed_UTC':datetime.now(timezone.utc).isoformat()})
    print({'complete':True,'pages':20,'reviewed_outputs':400,'exact256_cells':600})


if __name__ == '__main__': main()
