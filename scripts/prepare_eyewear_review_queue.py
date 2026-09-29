"""Prepare source-only training eyewear review, never automatic pixel labels."""
import json,hashlib
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw


def main():
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    root=Path('outputs/training_diversity_audit')
    roles=json.loads((root/'source_roles_v2.json').read_text())['records']
    candidate_path=root/'candidate_sources.json';candidate=json.loads(candidate_path.read_text())
    screen=json.loads((root/'duplicate_screen.json').read_text())
    assert screen['candidate_manifest_sha256']==sha(candidate_path)
    assert not screen['decoded_matches'] and not screen['review_flags']
    split_path=Path('outputs/downloaded_phase4/outputs/phase4_with_progress/split.json')
    assert sha(split_path)==candidate['split_sha256']
    split=json.loads(split_path.read_text());train={p.replace('\\','/') for p in split['train']}
    held={p.replace('\\','/') for p in split['validation']};assert train.isdisjoint(held)
    # Previously identified eyewear, plus explicit control/reflection candidates
    # selected by source appearance only (no checkpoint scores used).
    positives=[13,49,84,106,119,171,177,216,218,310,348,374]
    controls=[14,46,52,157,207,208,212,217,219,224,230,232,240,241,249]
    selected=[r for r in roles if r['index'] in set(positives+controls)]
    assert len(selected)==27
    real_path=Path('dataset/detector_glare_review_v3/manifest.json')
    benchmark=Path('outputs/downloaded_completion_benchmark/outputs/completion_pretrained_vm/manifest.json')
    assert sha(real_path)==candidate['v3_manifest_sha256'] and sha(benchmark)==candidate['benchmark_manifest_sha256']
    excluded={sha(p) for p in sorted(held)}
    real=json.loads(real_path.read_text())['records']
    excluded.update(r[k] for r in real for k in ('image_sha256','source_sha256') if r.get(k))
    excluded.update(c['source_sha256'] for c in json.loads(benchmark.read_text())['cases'])
    out=Path('outputs/eyewear_review_v1');out.mkdir(exist_ok=False);records=[]
    for r in selected:
        assert r['path'] in train and r['path'] not in held
        assert sha(r['path'])==r['sha256'] and r['sha256'] not in excluded
        records.append({k:r[k] for k in ('index','path','sha256','source','width','height')}|
                       {'partition':'train','training_enabled':False,'mask':None,
                        'proposal':'opaque_or_reflective_eyewear' if r['index'] in positives else 'transparent_or_reflection_review',
                        'grouping_status':'image hash only; identity grouping not certified'})
    for start in range(0,len(records),9):
        subset=records[start:start+9];canvas=Image.new('RGB',(1152,420*((len(subset)+2)//3)),'white');d=ImageDraw.Draw(canvas)
        for j,r in enumerate(subset):
            with Image.open(r['path']) as im:
                rgb=ImageOps.exif_transpose(im).convert('RGB');tile=ImageOps.contain(rgb,(384,384))
            x=j%3*384;y=j//3*420
            d.text((x+3,y+2),f"{r['index']} {Path(r['path']).name} ({r['width']}x{r['height']})",fill='black')
            d.text((x+3,y+16),r['proposal'],fill='black');canvas.paste(tile,(x+(384-tile.width)//2,y+32))
        canvas.save(out/f'page_{start//9}.png')
    report={'partition':'train','training_enabled':False,'records':records,
            'candidate_manifest_sha256':sha(candidate_path),'split_sha256':sha(split_path),
            'source_role_manifest_sha256':sha(root/'source_roles_v2.json'),
            'duplicate_screen_sha256':sha(root/'duplicate_screen.json'),
            'exact_excluded_hashes':len(excluded),'selection':'Prior source-only eyewear roles and visual control candidates; no model scores',
            'limitations':['Whole-image duplicate screen is heuristic, not identity-disjoint proof','No pixel masks or clear labels assigned','Source images may have other occlusions; inspect before annotation']}
    (out/'queue.json').write_text(json.dumps(report,indent=2));print('27 training-only candidates, 3 review pages; training disabled')


if __name__=='__main__':main()
