"""Record actual all-case development review and saved-pixel origin windows."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/completion_conditioning_union_v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rgb(path):
    with Image.open(path) as im:
        return np.array(im.convert('RGB'))


def binary(path):
    with Image.open(path) as im:
        return np.array(im.convert('L')) != 0


def write(path, data):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    p, r, audit = read(OUT / 'protocol.json'), read(OUT / 'results.json'), read(OUT / 'independent_saved_output_audit.json')
    assert audit['complete'] and audit['results_sha256'] == sha(OUT / 'results.json')
    # Each note follows actual native256 inspection of the eight saved pages.
    notes = {
        '00_cloth_mask_native': ('Central lower face remains plausible; union context changes its mouth/jaw estimate. Small source strap/material pixels and the hard peripheral support boundary remain.', 'plausible central estimate; peripheral joins remain'),
        '00_cloth_mask_degraded': ('Central cloth texture is removed with a softer plausible lower face. The retained mask/strap edge and generated-versus-blurred cheek transition remain visible.', 'plausible rough estimate; joining remains unqualified'),
        '01_pink_mask_native': ('Estimated mouth/chin remain plausible, including generated facial hair; this is unknown hidden appearance, not recovered anatomy. The union changes little in overall usefulness and retains a visible peripheral transition.', 'plausible central estimate; not hidden-identity evidence'),
        '01_pink_mask_degraded': ('A smoother lower-face estimate retains visible eyes and hairstyle exactly outside final support. Its facial hair differs from the baseline; no hidden reference decides which appearance is correct.', 'plausible rough estimate; whole-path quality unqualified'),
        '03_dark_sunglasses_native': ('The regenerated dark lenses are largely replaced by visible estimated eyes. Faint lens/frame-like outlines and a conspicuous pale support boundary remain.', 'material-remnant improvement; boundary remains'),
        '03_dark_sunglasses_degraded': ('Estimated eyes replace the earlier dark-lens pattern, with a lighter rounded frame/outline still apparent at the bridge and support boundary.', 'material-remnant improvement; boundary remains'),
        '04_sunglasses_native': ('The earlier tinted eyewear appearance is reduced; the new central eyes are plausible. Faint rim-like traces and the generated eye-region transition remain.', 'central removal improvement; not broad family qualification'),
        '04_sunglasses_degraded': ('The central opaque/tinted effect is reduced and eyes are estimated. Thin surrounding outlines remain; the unchanged blurred visible face is separate from this Off completion test.', 'central removal improvement; boundary remains'),
        '05_white_glare_native': ('The clear frames remain, but the estimated lens regions still look gray/clouded and bright reflective material remains around their edges. No convincing clear-eye result is established.', 'strong glare remains unqualified'),
        '05_white_glare_degraded': ('The estimated lenses retain a bright/gray patch and weakly resolved eyes. Broad hidden context does not consistently clear this glare example.', 'strong glare remains unqualified'),
        '06_mirrored_glare_native': ('Opaque green lens material is replaced by more distinct estimated eyes than the baseline. A thin frame/bridge-like line remains; the visible smile and hairstyle are copied.', 'central lens-material improvement; hidden eye identity unknown'),
        '06_mirrored_glare_degraded': ('Estimated eyes replace green lenses; thin framing and an eye-region transition remain. This does not establish ordinary-frame removal or restoration of visible blur.', 'central improvement; complete workflow unqualified'),
        '07_hand_over_mask_native': ('The new central nose/mouth is cleaner than the earlier red/patterned estimate. A conspicuous light diagonal boundary near the cheek and retained exterior hand still limit the join.', 'central estimate improvement; hand/face join remains'),
        '07_hand_over_mask_degraded': ('The lower face is less patterned, but light patches near the bridge/cheek and the hand-to-face boundary remain visible.', 'central estimate improvement; hand/face join remains'),
        '08_uncovered_native': ('Exact original bypass, including visible facial structure and ordinary hair.', 'empty-mask control passes'),
        '08_uncovered_degraded': ('Exact degraded-input bypass; Off makes no claim of restored visible structure.', 'empty-mask control passes'),
        '09_clear_glasses_native': ('Exact original bypass preserves clear lenses, frames, facial appearance and hairstyle.', 'clear-glasses control passes'),
        '09_clear_glasses_degraded': ('Exact degraded-input bypass preserves clear glasses and hair. No completion or visible restoration is requested.', 'clear-glasses control passes'),
        'val_18_hand_eyes_native': ('The estimated eyes are clearer and less narrow than the baseline, but the fingertip-like bridge notch and hand/face transitions remain conspicuous. Original mouth and head-top glasses remain copied.', 'eye estimate improvement; mixed source/generation join remains'),
        'val_18_hand_eyes_degraded': ('Estimated eyes are more distinct than the prior ghostlike result. The central finger-like notch and sharp transitions against the soft visible face still limit usefulness.', 'eye estimate improvement; mixed source/generation join remains'),
        'val_25_hand_mouth_native': ('The estimated nose/mouth is more coherent; the lower jaw ends at a stepped hand/contact boundary. Exterior hands remain intentionally copied.', 'plausible central estimate; jaw/hand join remains'),
        'val_25_hand_mouth_degraded': ('The mouth/nose estimate is smoother, with a conspicuous lower hand/jaw boundary still present. Soft visible eyes remain unchanged.', 'plausible central estimate; jaw/hand join remains'),
        'val_362_hair_eye_native': ('An estimated eye replaces much of the earlier regenerated obstruction. Its gaze/placement relative to the exposed eye and the adjacent hair/cheek join remain uncertain; ordinary exterior hairstyle is unchanged.', 'hair-remnant improvement; gaze/join still needs review'),
        'val_362_hair_eye_degraded': ('An eye is exposed in the estimated strip rather than mainly hair, but its gaze/placement and the soft-to-generated cheek/hair transition remain unqualified.', 'hair-remnant improvement; gaze/join still needs review'),
        'val_244_knit_scarf_native': ('A plausible central nose/mouth/beard estimate remains; the exterior neck scarf is deliberately retained. The lower face/scarf contour is approximate, with no hidden-reference appearance claim.', 'plausible central estimate; single exposed source pair'),
        'val_244_knit_scarf_degraded': ('The central face remains plausible and broadly similar to the baseline. The retained neck scarf and generated jaw meet at an approximate boundary.', 'plausible rough estimate; whole-family generalization unproven'),
        'val_336_scarf_gloves_native': ('The nose/mouth is smoother than the earlier coarse orange texture, but reddish lower-face material and the glove/scarf joins remain. Visible upper face is exact.', 'central texture improvement; lower joins remain'),
        'val_336_scarf_gloves_degraded': ('The central mouth/nose is less patterned, while the reddish lower boundary and mixed clothing/face join remain visible.', 'central texture improvement; lower joins remain'),
        'val_6_flower_mouth_native': ('The large generated petal/colored patches are reduced and the central lips/chin are cleaner. Small red peripheral tips remain at the cheek/face boundary.', 'generated-remnant improvement; peripheral tips remain'),
        'val_6_flower_mouth_degraded': ('The central mouth/chin is less stained and a plausible rough estimate; a small red tip remains near the side boundary.', 'generated-remnant improvement; peripheral tip remains'),
        'val_7_leaf_eye_native': ('The newly estimated eye/forehead is much less green/brown than the baseline. Exterior leaf/hand is retained, with a thin mixed leaf-to-face join still visible.', 'generated-staining improvement; exterior join remains'),
        'val_7_leaf_eye_degraded': ('The central eye/forehead staining is reduced. A thin leaf/face transition and mismatch with the blurred visible cheek remain.', 'generated-staining improvement; exterior join remains'),
    }
    rows = []
    for c in p['cases']:
        if c['rejected']:
            rows.append({'id': c['id'], 'condition': c['condition'], 'input_review': c['input_review'],
                         'excluded_before_generation': True, 'decision': 'prior input-only exclusion preserved; zero neural request'})
            continue
        observation, decision = notes[c['id']]
        rows.append({'id': c['id'], 'condition': c['condition'], 'family': c['family'],
                     'excluded_before_generation': False, 'observed_comparison': observation,
                     'development_decision': decision, 'visible_outside_final_and_protected_bytes_exact': True,
                     'mask_source': 'assisted; original-photo annotation reused for synthetic pair',
                     'hidden_reference': None, 'independent_final_quality_review': False})
    assert len(notes) == 32 and len(rows) == 36
    # Output-selected processing windows are NOT revised masks or semantic truth.
    windows = [
        ('val_18_hand_eyes', 'upper-hand central bridge notch', [124, 129, 141, 164]),
        ('val_6_flower_mouth', 'left peripheral flower tip', [47, 204, 67, 226]),
        ('val_6_flower_mouth', 'right peripheral flower tip', [201, 202, 219, 228]),
        ('05_white_glare', 'left lens/glare support', [64, 89, 105, 125]),
    ]
    diagnostic_rows = []
    for c in p['cases']:
        if c['rejected']:
            continue
        for base_id, label, rect in windows:
            if c['base_id'] != base_id:
                continue
            source, final = rgb(ROOT / c['input']), binary(ROOT / c['masks']['removal'])
            estimate, baseline = rgb(OUT / 'images' / (c['id'] + '.png')), rgb(ROOT / c['baseline_output'])
            x1, y1, x2, y2 = rect
            support = final[y1:y2, x1:x2]
            new, old, src = [a[y1:y2, x1:x2] for a in [estimate, baseline, source]]
            np.testing.assert_array_equal(new[~support], src[~support])
            np.testing.assert_array_equal(old[~support], src[~support])
            diagnostic_rows.append({'id': c['id'], 'label': label, 'rectangle_xyxy': rect,
                                    'window_pixels': int(support.size), 'generated_support_pixels': int(support.sum()),
                                    'copied_support_pixels': int((~support).sum()), 'copied_pixels_exact_in_both_outputs': True,
                                    'changed_estimated_pixels': int(np.any(new != old, axis=-1)[support].sum()),
                                    'inside_mask_pixels_coincidentally_equal_source': int(np.all(new == src, axis=-1)[support].sum())})
    assert len(diagnostic_rows) == 8
    write(OUT / 'pixel_support_diagnostic.json', {'complete': True, 'protocol_sha256': sha(OUT / 'protocol.json'),
          'results_sha256': sha(OUT / 'results.json'), 'selection': 'Post-output development inspection of four processing windows on three exposed source photographs and their degraded pairs',
          'not_covering_labels_or_hidden_face_truth': True, 'no_mask_or_annotation_changed': True,
          'scope_counts_not_covering_miss_rates': True, 'rectangles_include_ordinary_face_and_covering': True,
          'rows': diagnostic_rows, 'model_forwards': 0, 'gradient_calls': 0, 'optimizer_updates': 0,
          'hidden_metrics': None, 'app_changes': False})
    write(OUT / 'visual_review.json', {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
          'protocol_sha256': sha(OUT / 'protocol.json'), 'results_sha256': sha(OUT / 'results.json'),
          'independent_saved_output_audit_sha256': sha(OUT / 'independent_saved_output_audit.json'),
          'pixel_support_diagnostic_sha256': sha(OUT / 'pixel_support_diagnostic.json'),
          'reviewer': 'Implementing assistant development review; not independent final human review',
          'all32_outputs_and_same_final_baselines_actually_viewed': True,
          'all8_pages_actually_viewed_at_original256_cell_detail': True, 'pages': r['pages'], 'rows': rows,
          'quality_criteria_unchanged': p['quality_criteria'],
          'finding': 'Keeping final support fixed while changing frozen network context substantially reduces several regenerated object-like patterns. Strong glare, source-boundary fragments and joins remain; this is a causal conditioning observation, not all-family readiness.',
          'scope_limit': 'Union uses TWO historical assisted masks. No single-mask automatic rule, new automatic estimate, degraded-only annotation or app-ready selection policy is validated.',
          'same_frozen_completion_state': True, 'all4_controls_exact': True, 'all4_input_exclusions_preserved': True,
          'no_output_based_reannotation_or_seed_search': True, 'no_delivered_mask_expansion': True,
          'unknown_pretraining_overlap': True, 'source_is_exposed_photographs_not_native_CCTV': True,
          'original_photo_assistance_reused_for_synthetic_pairs': True, 'hidden_ground_truth': None, 'hidden_metrics': None,
          'new_automatic_outputs': 0, 'automatic_quality_qualification': False, 'assisted_quality_qualification': False,
          'app_adoption': False, 'independent_final_review': False, 'model_forwards_during_review': 0,
          'gradient_calls': 0, 'optimizer_updates': 0, 'goal_complete': False})
    print(json.dumps({'complete': True, 'actually_reviewed_outputs': 32, 'baseline_output_cells': 32,
                      'pages': 8, 'support_windows': 8, 'app_adoption': False}), flush=True)


if __name__ == '__main__':
    main()
