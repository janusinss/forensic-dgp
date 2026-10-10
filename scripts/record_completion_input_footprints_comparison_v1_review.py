"""Actual full-gallery assistant development review, without new inference."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/completion_input_footprints_comparison_v1_r1'

OBSERVATIONS={
 '00_cloth_mask':{'original':'Lower face is plausible in both versions; revised output has conspicuous dark material at the right cheek/ear join and a small left object edge. The earlier broader completion is cleaner.',
    'degraded':'A lower-face estimate remains plausible, with the same side-edge/cheek join limitation and softer visible context.',
    'decision':'Partial estimate; mask/face-boundary uncertainty and join remain.'},
 '01_pink_mask':{'original':'Central cloth is replaced by a plausible mouth/chin estimate, with unverified invented facial hair and a lower-boundary remnant. Visible eyes/braids remain exact.',
    'degraded':'Central lower face is estimated; the visible crop remains soft and the lower edge/invented facial hair remain. No hidden reference establishes identity accuracy.',
    'decision':'Partial estimate; hidden appearance and boundary quality unverified.'},
 '03_dark_sunglasses':{'original':'Revised result contains dark eyewear across the newly generated eye support. The earlier broader-mask result exposes plausible eyes instead.',
    'degraded':'Opaque dark eyewear also appears in the revised estimated support; this fails the intended ocular covering removal.',
    'decision':'Fail covering removal; generated object-like appearance inside target is distinct from retained source outside it.'},
 '04_sunglasses':{'original':'Revised output contains tinted frames/lenses over the eyes. Earlier broader mask produced visible eye estimates with fewer eyewear remnants.',
    'degraded':'The revised output again retains/regenerates eyewear appearance around estimated eyes; smile and surrounding input remain exact.',
    'decision':'Fail intended covering removal; no useful uniform improvement.'},
 '05_white_glare':{'original':'Clear frames are preserved, but bright reflected appearance remains prominent in the estimated eye areas; the older full-lens footprint gave more readable ocular estimates.',
    'degraded':'The revised patches provide little readable eye improvement over the soft source. Frame preservation alone does not meet the glare-removal objective.',
    'decision':'Fail useful glare estimate; conservative reflection boundary is ambiguous at256px.'},
 '06_mirrored_glare':{'original':'Opaque mirrored colour disappears and eyes are readable, but new clear-eyewear appearance remains/generated around them. The older estimate removed eyewear more completely.',
    'degraded':'Eyes become readable, with similar clear-eyewear appearance and soft source context. No exact hidden identity claim.',
    'decision':'Partial ocular estimate; new accessory appearance/removal scope remains unresolved.'},
 '07_hand_over_mask':{'original':'Central mouth/face estimate is plausible, but a white mask-side strip and abrupt hand/face join remain. Exterior fingers are intentionally preserved.',
    'degraded':'Central lower-face estimate remains possible; white edge and retained hand boundary remain conspicuous, with source softness unchanged.',
    'decision':'Partial central estimate; mixed covering/join remains.'},
 '08_uncovered':{'original':'Exact original bypass; no removal or completion.', 'degraded':'Exact degraded-input bypass; no restoration is requested in this Off-only ablation.',
    'decision':'Functional empty-mask control passes; not a restoration-quality claim.'},
 '09_clear_glasses':{'original':'Exact original bypass, including transparent glasses and hairstyle.', 'degraded':'Exact degraded-input bypass, including clear-glasses appearance.',
    'decision':'Functional clear-glasses protection passes.'},
 'val_18_hand_eyes':{'original':'Eyes are estimated but narrow and asymmetric, with conspicuous finger/skin-like lines near the upper bridge and hand joins. Exposed nose/mouth, head-top glasses and exterior hands remain exact.',
    'degraded':'Both eyes are estimated more softly; hand/face contacts and nose-bridge joins remain artificial. The visible open mouth is preserved.',
    'decision':'Fail consistent anatomical/join quality; exterior hands are retained by the frozen face-overlap policy.'},
 'val_25_hand_mouth':{'original':'Mouth/nose estimates are plausible; revised lower contour meets unchanged palms/wrists abruptly. Older broad-hand removal gave a smoother face boundary.',
    'degraded':'Mouth/nose estimate remains plausible but source hands outside the approximate jaw and a dark lower-face join remain noticeable.',
    'decision':'Partial estimate; join and approximate hidden-jaw boundary remain.'},
 'val_362_hair_eye':{'original':'Revised prediction still contains hair-like material over the right eye. Older broader footprint yielded a readable right-eye estimate; observed left gaze/nose/lips remain exact in revised result.',
    'degraded':'Hair-like texture also obscures the revised right-eye estimate. Source assistance does not resolve this prediction limitation.',
    'decision':'Fail obstructing-hair removal within estimated facial support.'},
 'val_244_knit_scarf':{'original':'A rough nose/mouth estimate appears with a substantial invented beard; the retained scarf below the approximate face boundary produces a noticeable join.',
    'degraded':'Rough lower-face estimate remains, with invented beard and scarf/jaw seam. Hidden contour and actual appearance are unknown.',
    'decision':'Partial estimate; hidden appearance and scarf join remain unqualified.'},
 'val_336_scarf_gloves':{'original':'Nose/mouth are more coherent than the older patterned result, but orange/thread-like texture and glove/scarf joins remain at the lower facial support.',
    'degraded':'The revised mouth/nose are somewhat cleaner than the prior patterned result; fabric colour and the retained glove/face boundary still limit usefulness.',
    'decision':'Local improvement with unresolved mixed-object texture/join; insufficient for full-family qualification.'},
 'val_6_flower_mouth':{'original':'Rose-like petal/colour remnants occur near the revised mouth/chin and facial edge. Earlier broader footprint was cleaner; visible eyes/hair remain exact.',
    'degraded':'Central estimate is softer, with a small petal-edge/colour remnant near the lower facial boundary. No hidden-reference accuracy claim.',
    'decision':'Fail consistently clean object removal/joins.'},
 'val_7_leaf_eye':{'original':'A left-eye estimate appears, but green/brown leaf-like staining occurs on the revised forehead/cheek. Exterior leaf/hand remain intentionally visible.',
    'degraded':'A left-eye estimate is plausible but green/brown tone at the estimated facial edge remains. The older broader result is cleaner.',
    'decision':'Partial eye estimate; colour/material remnant and join remain.'},
}


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    p,r,a=read(OUT/'protocol.json'),read(OUT/'results.json'),read(OUT/'independent_saved_output_audit.json')
    assert r['complete'] and a['complete'] and a['results_sha256']==sha(OUT/'results.json')
    old=read(ROOT/'outputs/dgp_app_covering_review_v3/results.json')
    assert r['state_before']==r['state_after']==old['state_before']['completion']
    assert len(OBSERVATIONS)==16 and len(r['pages'])==8
    # These eight pages were actually displayed at original detail, all32 cells.
    for page in r['pages']:assert page['sha256']==sha(OUT/page['path'])
    rows=[]
    for c in p['cases']:
        if c['rejected']:
            rows.append({'id':c['id'],'input_review':c['input_review'],'excluded_before_generation':True,'no_new_quality_claim':True})
            continue
        v=OBSERVATIONS[c['base_id']];condition='original' if c['condition']=='original_photo' else 'degraded'
        rows.append({'id':c['id'],'family':c['family'],'condition':c['condition'],'excluded_before_generation':False,
            'observed_comparison':v[condition],'development_decision':v['decision'],
            'hidden_reference':None,'independent_final_quality_review':False})
    record={'complete':True,'recorded_UTC':datetime.now(timezone.utc).isoformat(),'recorder_sha256':sha(Path(__file__)),
        'protocol_sha256':sha(OUT/'protocol.json'),'results_sha256':sha(OUT/'results.json'),
        'independent_saved_output_audit_sha256':sha(OUT/'independent_saved_output_audit.json'),
        'input_visual_review_sha256':sha(OUT/'input_visual_review.json'),'pages':r['pages'],'rows':rows,
        'all32_eligible_outputs_actually_reviewed':True,'all8_comparison_pages_viewed_at_original_detail':True,
        'reviewer':'Implementing assistant development visual review, not independent final review',
        'same_frozen_completion_state_as_original_route':True,'four_input_exclusions_preserved':True,
        'automatic_proposals':'Saved input-bound proposals have empty misses and false marks; overlap is against approximate operator masks, not segmentation truth; no new automatic output.',
        'automatic_quality_qualification':False,'assisted_quality_qualification':False,'app_adoption':False,
        'mask_geometry_compliance_not_quality_success':True,'cohort_is_exposed_photographs_not_native_CCTV':True,
        'unknown_pretraining_overlap':True,'mask_boundaries_and_hidden_jaw_are_approximate':True,
        'no_output_based_reannotation_or_seed_search':True,'new_hidden_metrics':None,
        'finding':'Tighter final removal support preserves visible appearance but does not establish useful completion. Remaining predicted eyewear/hair and joins require a conditioning diagnosis.',
        'causal_limit':'The mask changes both network context and compositing support. This comparison does not independently isolate which of those changes causes each visual difference.',
        'next_step':'Review network conditioning separately from final delivered support; retain the failed historical uniform context6 comparison and do not enlarge the delivered mask to erase clear features.',
        'model_forwards_during_review':0,'optimizer_updates':0,'app_changes':False,'independent_final_review':False,'goal_complete':False}
    with (OUT/'visual_review.json').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'complete':True,'actually_reviewed':32,'pages':8,'automatic_qualified':False,'assisted_qualified':False,'app_adoption':False}))


if __name__=='__main__':main()
