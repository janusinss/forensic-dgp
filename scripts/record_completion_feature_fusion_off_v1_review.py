"""Record the actual eight-sheet all-case development inspection; no generation."""
from datetime import datetime, timezone
from pathlib import Path
from completion_feature_fusion_off_v1_common import ROOT, OUT, sha, read, write, verify_bindings


def main():
    p, r, a = [read(OUT/name) for name in ['protocol.json', 'results.json', 'independent_saved_output_audit_r1.json']]
    assert a['complete'] and a['results_sha256'] == sha(OUT/'results.json')
    assert a['delivered_outputs'] == 32 and a['all160_page_cells_exact']
    verify_bindings(p)
    # These separate notes follow actual inspection of every original256 cell.
    notes = {
        '00_cloth_mask': (
            'The lower face is estimated but the w0 nose/mouth and cheek are redder and more textured than w1. Its cheek support edge and copied strap remain conspicuous.',
            'The w0 nose/mouth has reddish texture and an abrupt cheek transition against the blurred input; source strap/material at the boundary remains.'),
        '01_pink_mask': (
            'The mouth/chin estimate has more tooth and beard detail than w1, with a bright nose/mouth and abrupt lower join. The unknown facial hair itself is not a correctness failure without a hidden reference.',
            'The central estimate becomes redder and brighter than w1 and has a visible transition against the soft copied eyes/cheeks. Hidden facial hair cannot be validated.'),
        '03_dark_sunglasses': (
            'Estimated eyes remain inside a conspicuous pale blue-gray lens-shaped patch. The bridge and rounded support outline remain obvious; w0 does not resolve this defect.',
            'The pale lens-shaped eye patch and sharp bridge/support outline remain against the blurred visible face. Eye identity and accuracy cannot be judged from a hidden reference.'),
        '04_sunglasses': (
            'Estimated eyes are plausible, while thin lens/rim outlines and a generated-versus-copied eye-region join remain. The w0 setting does not demonstrate complete removal quality.',
            'The eyes are more distinct than the input lenses, but a thin rim/bridge contour and local sharpness mismatch remain. This Off test does not restore visible blur.'),
        '05_white_glare': (
            'The small estimated eyes still sit within cloudy and reflective lens areas; clear frames are copied. A convincing clear-eye result is not established by w0.',
            'Clouded gray lens regions and bright residual reflection remain; estimated eyes are weakly resolved. No full strong-glare qualification follows.'),
        '06_mirrored_glare': (
            'Green lenses are replaced by estimated eyes, but the w0 dark eye patch, retained frame/bridge and abrupt lens boundary remain. Visible smile and hairstyle are exact copies.',
            'Estimated eyes replace the green lens material with a darker patch and thin framing. The support transition remains distinct from the soft copied face.'),
        '07_hand_over_mask': (
            'The w0 central nose/mouth contains conspicuous red patterned patches and irregular anatomy absent from the smoother w1 estimate. Copied exterior hand and cheek seam also remain.',
            'The central estimate is reddish and coarse with a cheek/hand seam; less severe than its original-photo counterpart but still no useful whole-path qualification.'),
        '08_uncovered': (
            'Exact original-photo bypass preserves all visible features and ordinary hair.',
            'Exact synthetic-degraded input bypass; Off makes no visible-restoration claim.'),
        '09_clear_glasses': (
            'Exact bypass preserves transparent lenses, frames, visible appearance and ordinary hair.',
            'Exact bypass preserves clear glasses and ordinary hair at the supplied degraded quality.'),
        'val_18_hand_eyes': (
            'Estimated eyes remain surrounded by a large pale/yellow patch with a central bridge notch and sharp hand-to-face edges. Original mouth and glasses atop the head stay copied.',
            'The estimated eyes are sharp relative to the blurred face, with a large pale support patch and retained central finger-like notch. The joins remain unqualified.'),
        'val_25_hand_mouth': (
            'The nose/mouth estimate is plausible but texture and tone differ from the visible upper face; the lower jaw meets the copied hands at a stepped contact edge.',
            'A plausible mouth/nose estimate has coarse tonal and texture changes and a conspicuous lower hand/jaw join. Exterior hands are intentionally copied.'),
        'val_362_hair_eye': (
            'The estimated eye is more distinct than in w1, but the narrow generated cheek/eye strip has a visible join with the copied hair. Gaze/hidden-eye correctness is not known.',
            'An eye is estimated in the strip; its sharpness and the generated cheek/hair boundary differ from the soft input. This local change does not establish hair-family readiness.'),
        'val_244_knit_scarf': (
            'The w0 nose and adjacent skin are markedly redder and brighter than the copied upper face and w1. Estimated beard appearance is unknown; the face-to-retained-neck-scarf contour remains approximate.',
            'The w0 nose/upper jaw is redder and has a strong upper support boundary against the soft dark visible face. Retained neck scarf and approximate contour remain.'),
        'val_336_scarf_gloves': (
            'The w0 central face has conspicuous orange/red and overbright nose/mouth patches. Red material-like lower texture and copied glove/scarf joins are prominent.',
            'The w0 patch is strongly red/orange with a bright nose/mouth and a sharp upper edge. Scarf/glove joins and lower reddish texture remain.'),
        'val_6_flower_mouth': (
            'The central lips/chin are plausible but become brighter/redder than w1, with a visible contour transition. Small peripheral flower tips are copied outside final support; w0 cannot remove them.',
            'Estimated lips/chin are much sharper and brighter than the soft face, with a cool/pale contour patch and a retained red side tip. This is an exposed photographic diagnostic, not hidden-face accuracy.'),
        'val_7_leaf_eye': (
            'The estimated eye/forehead lacks the original leaf texture, but the thin mixed leaf-to-face join remains; exterior leaf and hand stay copied. No hidden-eye correctness can be established.',
            'The central leaf texture is replaced, with a thin forehead/cheek transition against the blurred face and retained exterior leaf/hand. Whole-object-family readiness remains unproven.'),
    }
    rows = []
    for c in p['cases']:
        row = {'id': c['id'], 'condition': c['condition'], 'family': c['family'], 'excluded_before_generation': c['rejected']}
        if c['rejected']:
            row.update({'input_review': c['input_review'], 'decision': 'Retain the unchanged prospective input-only exclusion', 'new_output_viewed': False})
        else:
            base, suffix = c['id'].rsplit('_', 1)
            assert suffix in ['native', 'degraded']
            note = notes[base][int(suffix == 'degraded')]
            bypass = next(entry for entry in r['rows'] if entry['id'] == c['id'])['empty_bypass']
            row.update({'new_output_and_w1_baseline_actually_viewed': True, 'observed_comparison': note,
                        'empty_bypass': bypass, 'visible_outside_final_and_protected_bytes_exact': True,
                        'development_decision': 'Empty control passes' if bypass else 'Retain diagnostic only; no app adoption',
                        'hidden_reference': None, 'hidden_accuracy_decision': None, 'independent_final_quality_review': False})
        rows.append(row)
    assert len(rows) == 36 and len(notes) == 16
    write(OUT/'visual_review.json', {
        'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(), 'reviewer': 'Implementing assistant development inspection; not independent final review',
        'protocol_sha256': sha(OUT/'protocol.json'), 'results_sha256': sha(OUT/'results.json'),
        'independent_saved_output_audit_r1_sha256': sha(OUT/'independent_saved_output_audit_r1.json'),
        'record_source_sha256': sha(Path(__file__)), 'all32_outputs_and32_w1_baselines_actually_viewed': True,
        'all8_pages_actually_viewed_at_original256_cell_detail': True, 'pages': r['pages'], 'rows': rows,
        'quality_criteria_unchanged': p['quality_criteria'],
        'finding': 'Removing direct encoder-feature fusion leaves central glare/eye-support defects and introduces or strengthens red/bright lower-face and scarf patches. Exact source preservation does not establish useful completion; w0 is not adopted.',
        'same_input_encoder_codes_in_matched_case': True, 'no_weight_or_seed_search': True,
        'all4_controls_exact': True, 'all4_input_exclusions_preserved': True, 'no_output_based_reannotation': True,
        'all7_covering_families_remain_in_scope': True, 'no_delivered_mask_expansion': True,
        'scope_limit': 'Same TWO historical assisted masks, same final support, exposed photographs and reused original-photo assistance for synthetic pairs; no single-mask automatic quality evidence',
        'unknown_pretraining_overlap': True, 'source_is_exposed_photographs_not_native_CCTV': True,
        'hidden_ground_truth': None, 'hidden_metrics': None, 'new_automatic_outputs': 0,
        'automatic_quality_qualification': False, 'assisted_quality_qualification': False, 'app_adoption': False,
        'independent_final_review': False, 'model_forwards_during_review': 0, 'gradient_calls': 0,
        'optimizer_updates': 0, 'VM_calls': 0, 'goal_complete': False})
    print({'complete': True, 'outputs_actually_reviewed': 32, 'baseline_cells': 32, 'pages': 8, 'app_adoption': False}, flush=True)


if __name__ == '__main__': main()
