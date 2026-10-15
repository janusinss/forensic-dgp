"""Close the result publication using immutable receipts and retained doc bytes."""
from datetime import datetime, timezone
from pathlib import Path
import re
from statistics import median

from PIL import Image

from import_cctv_dgp_conditioning_response_v1_return import ANALYSIS, OUT, PIN, RETURN, ROOT, read, sha, write


def main():
    stage=ANALYSIS/'publication';published=read(stage/'publication.json')
    assert published['complete'] and published['protocol_sha256']==PIN==sha(RETURN/'protocol.json')
    assert published['diagnostic_completed'] and published['original_prospective_return_audit_completed']
    assert not published['original_elementwise_parity_all_pass'] and not published['model_qualified']
    assert published['accepted_updates']==published['optimizer_updates']==0
    assert not published['automatic_VM_launch'] and published['next_diagnostic_is_design_only']
    assert all(sha(ROOT/n)==h for n,h in published['evidence_sha256'].items())
    imported=read(ANALYSIS/'import.json');audit=read(ANALYSIS/'independent_audit.json')
    response=read(ANALYSIS/'response_review.json');precision=read(ANALYSIS/'saved_embedding_precision_review.json')
    visual=read(ANALYSIS/'visual_review.json');png=read(ANALYSIS/'PNG_change_review.json')
    assert sha(OUT/'cctv-dgp-conditioning-response-v1-results.tar.gz')==imported['archive_sha256']==published['return_archive_sha256']
    assert all(v['complete'] and not v['model_qualified'] for v in [imported,audit,response,precision,visual,png])
    assert response['prior_smallest_trial_comparison']['maximum_absolute_score_difference']==0
    run=read(RETURN/'outputs/conditioning_response_v1/run_results.json')
    assert run['counts']==response['diagnostic_counts']==dict(B=200,recognizer=360,gradient_queries=640,temporary_displacements=6,optimizer_updates=0,accepted_updates=0)
    assert run['full_epochs']==run['native_pixels_used']==run['final_identity_pixels']==0
    assert len(audit['completed_snapshots'])==8 and len(audit['central_response_comparisons'])==96
    assert all(r['resolved'] for r in audit['central_response_comparisons'])
    assert sum(r['central_comparison_within_diagnostic_bound'] is True for r in audit['central_response_comparisons'])==76
    for row in response['central_summaries']:
        values=[r for r in audit['central_response_comparisons'] if r['pair']==row['pair'] and r['metric']==row['metric']]
        assert row['groups']==len(values) and row['resolved']==len(values) and row['unresolved']==0
        assert row['within_diagnostic_bound']==sum(r['central_comparison_within_diagnostic_bound'] is True for r in values)
        errors=[abs(r['central_prediction_residual']/r['predicted_two_sided_difference']) for r in values]
        assert row['median_relative_central_error']==median(errors) and row['maximum_relative_central_error']==max(errors)
    repeat=next(r for r in response['original_bound_parity_summaries'] if r['comparison']=='repeat00 versus repeat01')
    assert repeat['groups']==32 and repeat['passing_groups']==31 and repeat['mismatched_elements']==1
    assert all(r['norm_check_pass'] for r in audit['repeat_vector_checks'][1]['original_unchanged_parity_bounds'])
    assert precision['all_frozen_embeddings_exact'] and len(precision['central_comparisons'])==30
    assert [sum(r['saved_double_arithmetic_diagnostic']['central_comparison_within_diagnostic_bound'] is False
                for r in precision['central_comparisons'] if r['pair']==k) for k in range(3)]==[9,5,4]
    for path,h in precision['saved_embedding_file_bindings'].items():assert sha(RETURN/path)==h
    assert visual['all20_sheets_and1000_cells_directly_viewed'] and len(visual['reviewed_sheets'])==20
    assert {r['file'] for r in visual['reviewed_sheets']}=={r['file'] for r in audit['all_planned_visual_sheets']}
    for row in visual['reviewed_sheets']:
        assert sha(ROOT/row['file'])==row['sha256'] and row['all_five_rows_and_ten_columns_reviewed']
        assert row['direct_original_resolution_review'] and row['notes']
        with Image.open(ROOT/row['file']) as image:assert image.size==(2560,1400)
    for point in png['signed_point_recounts']:
        assert point['cases']==len(point['case_records'])==100 and point['cases_with_changed_pixels']==100
        assert point['total_changed_RGB_channels']==sum(r['changed_channels'] for r in point['case_records'])
        assert point['maximum_byte_difference']==max(r['maximum_byte_difference'] for r in point['case_records'])
        for row in point['case_records']:
            assert sha(RETURN/'outputs/conditioning_response_v1/initial/png'/(row['id']+'.png'))==row['initial_png_sha256']
            assert sha(RETURN/'outputs/conditioning_response_v1'/point['label']/'png'/(row['id']+'.png'))==row['signed_png_sha256']
    tiny=next(r for r in png['signed_point_recounts'] if r['label']=='plus02')
    assert tiny['total_changed_RGB_channels']==22124 and tiny['maximum_byte_difference']==1
    documents={};texts=[]
    for name,binding in published['documents'].items():
        old=Path(binding['backup']);after=(ROOT/name).read_bytes()
        assert sha(old)==binding['before_sha256'] and sha(ROOT/name)==binding['after_sha256']
        assert after[binding['prefix_bytes']:]==old.read_bytes() and binding['old_bytes_preserved_as_exact_suffix']
        prefix=after[:binding['prefix_bytes']].decode('utf-8');texts.append(prefix)
        assert 'USED' in prefix and '14 October 2026' in prefix
        documents[name]=dict(history_exact=True,current_sha256=sha(ROOT/name),prefix_bytes=binding['prefix_bytes'])
    assert len(documents)==8
    for name,h in published['new_documents_sha256'].items():
        assert sha(ROOT/name)==h;texts.append((ROOT/name).read_text(encoding='utf-8'))
    result=(ROOT/'CCTV_DGP_CONDITIONING_RESPONSE_V1_RESULTS.md').read_text(encoding='utf-8')
    review=(ROOT/'CCTV_DGP_APPEARANCE_RESPONSE_REVIEW_V1.md').read_text(encoding='utf-8')
    for gain in ['0.346369%','0.361388%','0.040293%','0.053512%','0.004101%','0.006602%']:assert gain in result
    assert '1.72' not in result and imported['archive_sha256'] in result
    assert '1,247' in result and '5,852' in result and '1,600' in result and '1,000' in result
    assert 'more epochs' in review.lower() or 'epoch schedule' in review.lower()
    assert 'not a verified' in review and '200 RGB-input derivative queries' in review
    linked=[]
    for content in texts:
        for raw in re.findall(r'\]\(<([^>]+)>\)',content):
            path=Path(raw);assert path.is_absolute() and path.is_file(),raw;linked.append(raw)
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(stage/'publication_audit.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        publication_sha256=sha(stage/'publication.json'),protocol_sha256=PIN,verified_documents=documents,
        new_documents_sha256=published['new_documents_sha256'],existing_local_links_checked=len(linked),
        full_return_archive_hash_rechecked=True,retained_prospective_audit_and_original_bounds=True,
        saved_score_counts_precision_reviews_and_every_visual_binding_checked=True,
        protected_model_and_app_files_sha256=protected,optimizer_updates=0,accepted_updates=0,
        next_diagnostic_design_only=True,local_neural_or_training_calls=0,
        model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,history_documents=8,new_result_and_design=2,visual_sheets=20,
               diagnostic_complete=True,model_qualified=False,next_diagnostic_design_only=True))


if __name__=='__main__':main()
