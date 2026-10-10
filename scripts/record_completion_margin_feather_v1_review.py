"""Record a complete development review after all eight exact pages are viewed."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/completion_margin_feather_v1'


def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    # Observations are entered only after the implementing assistant has viewed
    # all generated pages; this entrypoint does not create or infer visual claims.
    source = OUT / 'actual_page_review_notes.json'
    assert source.is_file(), 'Actual complete visual observations required'
    p = json.loads((OUT / 'protocol.json').read_text()); r = json.loads((OUT / 'results.json').read_text())
    a = json.loads((OUT / 'independent_saved_output_audit.json').read_text()); notes = json.loads(source.read_text())
    assert a['complete'] and notes['all8_pages_actually_viewed_at_original256_cell_detail']
    assert notes['page_sha256'] == {page['path']:page['sha256'] for page in r['pages']}
    assert [row['id'] for row in notes['rows']] == [c['id'] for c in p['cases']]
    assert len(notes['rows']) == 36 and not notes['independent_final_review']
    receipt = {**notes, 'complete':True, 'protocol_sha256':sha(OUT / 'protocol.json'), 'results_sha256':sha(OUT / 'results.json'),
               'saved_output_audit_sha256':sha(OUT / 'independent_saved_output_audit.json'), 'review_notes_sha256':sha(source),
               'reviewer':'Implementing assistant development review; not independent final human review',
               'saved_raw_neural_outputs_unchanged':True, 'all32_core_estimates_byte_exact':True, 'no_new_masks_or_margin_expansion':True,
               'automatic_outputs':0, 'automatic_quality_qualification':False, 'assisted_quality_qualification':False,
               'app_adoption':False, 'model_forwards':0, 'gradient_calls':0, 'optimizer_updates':0, 'goal_complete':False}
    with (OUT / 'visual_review.json').open('x', encoding='utf-8', newline='\n') as stream: stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete':True, 'reviewed_outputs':32, 'app_adoption':False}))


if __name__ == '__main__': main()
