"""Record direct full-resolution development inspection, never hidden-identity accuracy."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/completion_mat_mirror_comparison_v1'


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


# Entered after viewing each of the eight1024x1148 PNG pages at original detail.
# These are qualitative development observations, not labels for hidden anatomy.
OBSERVATIONS={
 '00_cloth_mask':(
  'A coherent lower-face estimate is present, but the nose and cheek texture have uneven shading; new beard appearance is only an estimate. The current baseline is smoother.',
  'Nose/cheek shading is more patchy and less coherent than the current baseline; added texture does not establish clearer useful anatomy.'),
 '01_pink_mask':(
  'Lower-face estimate is plausible but changes estimated expression; a conspicuous pale/pink neck-edge remnant remains near the removal boundary.',
  'Lower face is smooth, with a conspicuous pale/pink chin/neck join. No clear improvement over the baseline.'),
 '03_dark_sunglasses':(
  'Both eyes are estimated as closed lids, with coarse brow/bridge blending. Closed eyes are not a hidden-accuracy error; this supplies no demonstrated useful eye-structure gain.',
  'Closed-lid estimate and blurred brow/bridge blend remain. Neither candidate can establish the actual hidden eye state.'),
 '04_sunglasses':(
  'Open-eye estimate is plausible and less rectangular in its transition than the current baseline; hidden eye appearance is unknown.',
  'A glasses-like dark rim pattern is generated over the removed area, despite complete opaque-eyewear support. Removal is not consistently satisfied.'),
 '05_white_glare':(
  'Bright, indistinct lens/eye regions remain inside the requested glare-removal support; visible clear frames stay retained. The current baseline exposes more legible eye structure.',
  'Lens/eye estimates remain pale and indistinct. No clear glare-removal or eye-structure improvement.'),
 '06_mirrored_glare':(
  'A lowered-lid eye estimate with dark brow/eye makeup-like texture is plausible in isolation but supplies no verified hidden eye state or consistent structural advantage.',
  'Eye estimate is more open, but bridge/eye shading is uneven. Estimates differ substantially between original and degraded versions.'),
 '07_hand_over_mask':(
  'The central covering is replaced by a plausible unsmiling lower face; patch texture and lower hand/cheek transition remain uneven.',
  'Central hand/mask estimate is smooth. The lower-face join is coarse and no broad improvement over the current baseline is established.'),
 '08_uncovered':(
  'Empty reviewed mask bypasses the generator; input is unchanged.',
  'Empty reviewed mask bypasses the generator; degraded input is unchanged.'),
 '09_clear_glasses':(
  'Empty reviewed mask preserves the complete source, including clear frames and non-obstructing hair.',
  'Empty reviewed mask preserves every source pixel, including clear frames and non-obstructing hair.'),
 'val_18_hand_eyes':(
  'Estimated eyelid/brow area contains orange finger-like arcs and broken hand/cheek joins inside and around the patch. Visible lower face remains exact.',
  'Eye area is estimated with closed lids; broad palm/finger support remains outside the frozen footprint. Boundary remnants persist; hidden lid state is unknown.'),
 'val_25_hand_mouth':(
  'Lower-face estimate has coherent mouth/nose structure, but stronger dark lip and patch shading. This is plausible anatomy, not recovered identity.',
  'Estimated mouth/nose are legible but tinted and uneven against the visible grayscale source. No consistent appearance advantage.'),
 'val_362_hair_eye':(
  'Obstructing hair-like strands still occupy the right-eye removal region; the requested removal is unresolved. Visible left eye and non-obstructing hair remain exact.',
  'The right eye remains obscured/indistinct with generated hair-like texture. No useful completion gain; visible support is unchanged.'),
 'val_244_knit_scarf':(
  'Nose/upper-mouth estimate is coarse, with dark broken nostril/philtrum texture and an uneven cheek/scarf join; the current baseline is more coherent.',
  'Nose/mouth patch is heavily mottled and less coherent than the current baseline. Peripheral retained scarf is outside the frozen removal support.'),
 'val_336_scarf_gloves':(
  'A plausible smiling mouth replaces the central scarf/glove patch, but the nose join is irregular and orange knitted texture remains at the edge. The baseline also fails this boundary.',
  'Mouth/nose estimate is blurry with orange edge remnants; neither result cleanly resolves the covering boundary.'),
 'val_6_flower_mouth':(
  'Flower is removed and a plausible open-lip lower-face estimate is present. Soft boundary transitions remain; hidden mouth appearance is unknown.',
  'Flower is removed with a smooth lower-face estimate and less beard-like texture than the baseline. Estimated appearance remains uncertain.'),
 'val_7_leaf_eye':(
  'A plausible eye and bridge are estimated. Hand/fingers remain outside the reviewed leaf footprint and are exactly retained, not a generator-copy failure inside the mask.',
  'Eye estimate is plausible but soft. A retained finger/hand edge remains outside the reviewed footprint; hidden eye accuracy is unavailable.')}


def main():
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    r=json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    audit=json.loads((OUT/'independent_saved_output_audit.json').read_text(encoding='utf-8'))
    assert audit['complete'] and audit['results_sha256']==sha(OUT/'results.json') and r['requests']==32
    sheets=[]
    for page in range(1,9):
        path=OUT/'preview'/(f'page_{page:02d}.png')
        sheets.append({'path':path.relative_to(OUT).as_posix(),'sha256':sha(path),
                       'actually_viewed':True,'requested_detail':'original','source_pixels':[1024,1148]})
    rows=[]
    for case in p['cases']:
        original=case['condition']=='original_photo'
        suffix='_native' if original else '_degraded'
        assert case['id'].endswith(suffix)
        key=case['id'][:-len(suffix)];assert key in OBSERVATIONS
        rows.append({'id':case['id'],'family':case['family'],'condition':case['condition'],
                     'mode':'assisted','observation':OBSERVATIONS[key][0 if original else 1],
                     'hidden_accuracy':None,'visible_preservation':'Independently verified source copy outside frozen mask.'})
    review={'complete':True,'recorder_sha256':sha(Path(__file__)),
            'reviewer':'Implementing Codex assistant; direct development review, not independent final examiner.',
            'protocol_sha256':sha(OUT/'protocol.json'),'results_sha256':sha(OUT/'results.json'),
            'independent_saved_output_audit_sha256':sha(OUT/'independent_saved_output_audit.json'),
            'all32_cases_reviewed':True,'all8_sheets_viewed_at_original_detail':True,'sheets':sheets,'rows':rows,
            'decision':'Do not promote this converted MAT FFHQ512 candidate: it lacks consistent useful improvement across the agreed covering families.',
            'necessary_next_actions':['Retain all32 estimates and the pinned conversion/source limitations.',
                                      'Correct input-observed covering-footprint misses before judging removal outside support.',
                                      'Keep generated anatomy defects separate from footprint defects; do not rerun unchanged or select a favorable seed.',
                                      'A different weight/processing recipe needs explicit justification and a newly frozen full-family comparison.'],
            'automatic_results_generated':False,'automatic_quality_qualification':False,'assisted_quality_qualification':False,
            'MAT_original_author_model_failure_claim':False,'source_pretraining_overlap_unknown':True,
            'native_CCTV_or_reserved_final_used':False,'hidden_reference_metrics':None,
            'app_changes':False,'training':False,'goal_complete':False}
    with (OUT/'visual_review.json').open('x',encoding='utf-8',newline='\n') as stream:json.dump(review,stream,indent=2)
    print(json.dumps({'complete':True,'cases_reviewed':32,'sheets_viewed':8,'promoted':False,
                      'visual_review_sha256':sha(OUT/'visual_review.json'),'goal_complete':False},indent=2))


if __name__=='__main__':main()
