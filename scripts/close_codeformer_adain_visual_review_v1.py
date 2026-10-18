"""Bind the root's completed saved-sheet observations; CPU files only."""
import hashlib
import json
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'outputs/codeformer_adain_gradient_isolation_v1_return_review'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def main():
    manifest_path = REVIEW / 'visual_sheets/manifest.json'
    manifest = read(manifest_path)
    assert manifest['complete'] and len(manifest['pages']) == 55
    batches = sorted((REVIEW / 'root_visual_batches').glob('batch_*.json'))
    seen = []
    notes = []
    for p in batches:
        b = read(p)
        assert b['reviewer'] == 'root' and b['visual_acceptance'] is False
        seen.extend(b['pages'])
        notes.append({'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'pages': b['pages'], 'notes': b['notes']})
    assert sorted(seen) == list(range(1,56)) and len(seen) == len(set(seen))
    sheets = []
    for n,p in enumerate(manifest['pages'],1):
        f = REVIEW / 'visual_sheets' / p['file']
        assert sha(f) == p['sha256'] and n in seen
        sheets.append({'path': str(f.relative_to(ROOT)), 'sha256': sha(f), 'case_ids': p['case_ids'], 'root_personally_viewed': True})
    critical = REVIEW / 'critical_clear_256.png'
    bindings_path = REVIEW / 'critical_clear_256_bindings.json'
    bindings = read(bindings_path)
    assert len(bindings['rows']) == 15
    with Image.open(critical) as im:
        figure = np.asarray(im.convert('RGB'))
    for row in bindings['rows']:
        p = Path(row['path'])
        assert sha(p) == row['sha256']
        with Image.open(p) as im:
            arr = np.asarray(im.convert('RGB'))
        x1,y1,x2,y2 = row['box']
        assert arr.shape == (256,256,3) and np.array_equal(figure[y1:y2,x1:x2],arr)
    guards_path = REVIEW / 'metrics_v1/guards.json'
    result = {
        'complete': True, 'reviewer': 'root', 'root_personally_viewed_all_planned_sheets': True,
        'model_qualified': False, 'visual_acceptance': False, 'goal_complete': False,
        'review_scope': 'All55 saved sheets personally viewed by root, plus all3 failed clear cases in a separate original-detail256 figure; no neural execution or output changes.',
        'sheets': sheets, 'observation_batches': notes,
        'visual_manifest_sha256': sha(manifest_path), 'guards_sha256': sha(guards_path),
        'critical_figure': {'path': str(critical.relative_to(ROOT)), 'sha256': sha(critical), 'bindings_sha256': sha(bindings_path), 'exact256_cells_verified':15, 'root_personally_viewed_original_detail': True, 'case_ids': bindings['ids'], 'notes':'Same three clear failures as prior FP32. Trained columns are nearly identical, with brightness/smoothing changes; HQ27440 is also brighter. Tone drift and residual changes remain. The figure was created before independent metric recomputation; its original caption is retained, while guards.json is authoritative.'},
        'display_limit': 'Page overview may be resized by the viewing tool. All saved cells are exact256 pixels without enhancement/resampling. Critical figure viewed at original detail.',
        'limits': ['No independent final visual or user acceptance claim.', 'Paired synthetic target comparisons do not establish native CCTV recovery.', 'Native QMUL and ChokePoint remain separately reported unpaired DEV evidence; no ethnicity, exact identity or Zamboanga claim.', 'FINAL pixels and the separate completion model remain untouched.'],
        'source_sha256': sha(__file__)
    }
    out = REVIEW / 'visual_review_v1.json'
    with out.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,indent=2); stream.write('\n')
    print(json.dumps({'complete':True,'root_sheets_seen':55,'critical_exact256_cells':15,'model_qualified':False,'sha256':sha(out)}))

if __name__ == '__main__':
    main()
