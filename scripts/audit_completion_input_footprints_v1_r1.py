"""Independent saved-mask geometry/support recount; no Torch/model imports."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/completion_input_footprints_comparison_v1_r1'
DRAFT = ROOT/'outputs/completion_input_footprints_v1/mask_draft_v1_r1'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def binary(path):
    with Image.open(path) as im:
        assert im.mode=='L' and im.size==(256,256)
        a=np.array(im)
    assert set(np.unique(a))<={0,255}; return a!=0


def polygon_trace(values):
    image=Image.new('L',(256,256));d=ImageDraw.Draw(image)
    for points in values:
        assert len(points)>=3 and all(len(v)==2 and all(isinstance(x,int) and 0<=x<256 for x in v) for v in points)
        d.polygon(list(map(tuple,points)),fill=255)
    return np.array(image)!=0


def main():
    started=time.monotonic();p,d=read(OUT/'protocol.json'),read(DRAFT/'annotations.json')
    for name,h in {**p['sources_sha256'],**p['app_preservation_sha256']}.items(): assert sha(ROOT/name)==h,name
    pairs,rows={},[]
    for c in p['cases']:
        pairs.setdefault(c['base_id'],[]).append(c)
        if c['rejected']:
            assert c['input_review'] in ['out_of_scope','needs_clearer'] and 'masks' not in c
            continue
        arrays={k:binary(ROOT/path) for k,path in c['masks'].items()}
        v=d['prepared'][c['base_id']];s=v['annotation']
        core,face,protected,mask=[arrays[k] for k in ['core','face','protected','removal']]
        if core.any():
            assert np.array_equal(face,polygon_trace(s['face_eligibility_polygons']))
            assert np.array_equal(core,polygon_trace(s['covering_polygons']) & face)
            assert np.array_equal(protected,polygon_trace(s['protected_visible_polygons']))
            distance=distance_transform_edt(~core)
            expected=core|((distance<=2)&face&~protected)
            assert np.array_equal(mask,expected),'Independent Euclidean distance margin differs'
            assert float(distance[mask&~core].max(initial=0))<=2
        else:
            assert not mask.any() and not face.any() and protected.all()
            assert c['base_id'] in ['08_uncovered','09_clear_glasses']
        assert not (mask&protected).any() and not (mask&~face).any() and (mask[core]).all()
        assert int(core.sum())==v['core_pixels'] and int(mask.sum())==v['removal_pixels']
        assert int((mask&~core).sum())==v['margin_added_pixels']
        old=binary(ROOT/c['reviewed']);automatic=binary(ROOT/c['automatic'])
        assert int(old.sum())==v['old_pixels']
        assert int((mask&~old).sum())==v['added_from_old_pixels']
        assert int((old&~mask).sum())==v['removed_from_old_pixels']
        overlap={'proposal_pixels':int(automatic.sum()),'assisted_pixels':int(mask.sum()),
                 'missed_assisted_pixels':int((mask&~automatic).sum()),'extra_proposal_pixels':int((automatic&~mask).sum())}
        union=int((mask|automatic).sum());overlap['approximate_assisted_overlap']=int((mask&automatic).sum())/union if union else 1.
        rows.append({'id':c['id'],'family':c['family'],'condition':c['condition'],'core_pixels':int(core.sum()),
            'removal_pixels':int(mask.sum()),'margin_pixels':int((mask&~core).sum()),'protected_overlap_pixels':0,
            'added_from_old_pixels':v['added_from_old_pixels'],'removed_from_old_pixels':v['removed_from_old_pixels'],
            'automatic_comparison':overlap,'reference_is_approximate_operator_footprint_not_segmentation_truth':True})
        assert time.monotonic()-started<120
    assert len(rows)==32 and len(pairs)==18
    for base,cases in pairs.items():
        assert len(cases)==2 and cases[0]['rejected']==cases[1]['rejected']
        if not cases[0]['rejected']: assert cases[0]['masks']==cases[1]['masks']
    failure=read(ROOT/'outputs/completion_input_footprints_v1/draft_preparation_failure_v1/failure.json')
    assert not failure['complete'] and not failure['draft_output_created'] and failure['new_generator_forwards']==0
    assert failure['source_sha256']==sha(ROOT/'scripts/draft_completion_input_footprints_v1.py')
    assert failure['source_sha256']==sha(ROOT/'outputs/completion_input_footprints_v1/draft_preparation_failure_v1/draft_source_before_correction.py')
    # Actual original/degraded visual review is retained, rather than claiming
    # that a geometry assertion establishes mask or hidden-face truth.
    visual=read(OUT/'input_visual_review.json');assert visual['complete'] and visual['all6_draft_input_pages_actually_viewed_at_original_detail']
    assert not visual['new_generator_outputs_seen'] and not visual['quality_qualification']
    assert 'torch' not in sys.modules
    receipt={'complete':True,'protocol_sha256':sha(OUT/'protocol.json'),'checker_sha256':sha(Path(__file__)),
        'seconds':time.monotonic()-started,'cap_seconds':120,'source_bindings':len(p['sources_sha256']),
        'eligible_masks_recounted':32,'unique_masks':16,'nonempty_masks':14,'empty_masks':2,'exclusions_retained':4,
        'all_margin_pixels_within2_Euclidean_pixels':True,'protected_overlaps':0,'outside_face_overlaps':0,
        'matched_degraded_pairs_original_photo_assistance':True,'rows':rows,
        'first_glare_frame_conflict_preserved':True,'model_forwards':0,'optimizer_updates':0,
        'automatic_quality_qualification':False,'assisted_quality_qualification':False,
        'mask_segmentation_truth':False,'independent_final_quality_review':False,'goal_complete':False}
    with (OUT/'independent_mask_audit.json').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:receipt[k] for k in ['complete','seconds','eligible_masks_recounted','protected_overlaps','model_forwards']}))


if __name__ == '__main__': main()
