"""Input-only manual polygon draft; no generator or training imports.

Annotations are approximate development removal areas, not publisher labels or
hidden anatomy. Source photographs/paired degradations were inspected before
these traces. Preserve this draft even if its visual review calls for revision.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/completion_input_footprints_v1'
OUT = PARENT / 'mask_draft_v1_r1'
PARENT_SHA = '84032b200aa6108f5e920b1a3124aefe39fd9ff307599ee2431b9b6f16585cd9'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + '\n')


def polys(values):
    image = Image.new('L', (256, 256)); draw = ImageDraw.Draw(image)
    for value in values:
        draw.polygon([tuple(point) for point in value], fill=255)
    return np.array(image) != 0


def rectangle(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def spec(face, objects, protected, observation, uncertainty):
    return {'face_eligibility_polygons': [face], 'covering_polygons': objects,
            'protected_visible_polygons': protected, 'input_observation': observation,
            'boundary_uncertainty': uncertainty}


# Coordinates are manually traced from the frozen256px inputs. No generated
# output, diagnostic ROI, pixel colour classifier or old-mask dilation is used.
ANNOTATIONS = {
    '00_cloth_mask': spec(
        [(51,36),(157,31),(199,63),(214,112),(210,167),(208,204),(184,239),(134,255),(77,255),(53,243),(29,204),(27,149),(43,113)],
        [[(32,138),(43,144),(62,134),(82,122),(103,126),(125,136),(146,146),(168,152),(193,147),(198,158),(192,208),(177,225),(150,242),(126,251),(93,253),(58,242),(47,219),(38,185)],
         [(193,149),(207,129),(220,109),(223,111),(214,129),(199,158)]],
        [rectangle(49,97,167,119), [(216,120),(229,107),(247,108),(249,137),(233,159),(215,171)]],
        'Patterned opaque mask front and a thin strap cross the lower facial area. Eyes, eyebrow structure, exposed ear and hair remain visible.',
        'Hidden cheek/chin contour is approximate. The strap outside the face and exposed ear are retained; no reconstructed contour label is asserted.'),
    '01_pink_mask': spec(
        [(61,52),(97,31),(153,36),(191,64),(211,109),(213,168),(205,199),(184,227),(157,247),(117,253),(81,239),(61,213),(48,174),(49,121)],
        [[(49,139),(67,132),(89,132),(112,136),(145,135),(176,130),(205,125),(212,146),(213,164),(208,182),(198,205),(183,224),(156,240),(118,245),(85,235),(63,214),(54,190),(47,171)]],
        [rectangle(65,96,191,123)],
        'Pink cloth covers nose and lower face; lower patterned cloth belongs to the covering. Eyes, brows and braids are visible.',
        'Hidden jaw is approximate; preserve side braids outside the facial overlap.'),
    '03_dark_sunglasses': spec(
        [(46,73),(111,60),(185,64),(239,91),(250,123),(246,188),(220,224),(178,245),(126,253),(80,246),(48,218),(24,180),(19,120)],
        [[(34,85),(53,82),(84,86),(112,90),(123,106),(137,105),(152,91),(186,88),(216,87),(237,97),(246,112),(242,135),(229,152),(198,160),(163,161),(143,147),(132,129),(120,127),(105,148),(84,156),(53,154),(35,146),(26,130),(28,111)]],
        [rectangle(83,168,176,245)],
        'Opaque sunglasses and front rim hide both eyes. Helmet/hood and lower facial features are visible.',
        'Upper rim and small temple joins are approximate at256px; helmet/hood outside the facial obstruction are not removal targets.'),
    '04_sunglasses': spec(
        [(72,69),(106,44),(148,48),(176,76),(195,103),(201,148),(186,189),(165,222),(137,244),(111,244),(87,224),(64,190),(53,145),(58,101)],
        [[(60,102),(88,103),(110,106),(121,111),(133,106),(169,100),(187,100),(195,110),(195,130),(185,142),(161,147),(142,142),(131,127),(124,122),(118,137),(105,146),(77,148),(64,142),(57,125),(57,111)]],
        [rectangle(82,164,171,219)],
        'Both dark lenses, bridge and opaque front rim obscure eyes; lower face and smile remain readable.',
        'Thin rim ends are approximate; hair outside the facial region remains.'),
    '05_white_glare': spec(
        [(63,32),(134,19),(177,36),(201,76),(207,123),(201,170),(180,212),(139,237),(91,230),(61,195),(40,147),(36,98)],
        [[(84,92),(94,90),(104,93),(109,102),(109,114),(98,117),(86,111),(83,103)],
         [(157,104),(168,105),(178,109),(184,119),(180,132),(166,132),(158,124)]],
        [[(73,84),(87,81),(111,84),(128,91),(136,103),(131,118),(121,126),(99,130),(78,123),(70,110),(70,91),
          (74,91),(75,108),(82,119),(99,125),(119,121),(128,114),(130,103),(125,95),(110,89),(87,86)],
         [(146,97),(154,96),(176,101),(198,110),(203,122),(198,139),(187,144),(163,141),(149,132),(143,113),
          (148,112),(153,129),(165,136),(186,139),(193,135),(198,122),(194,114),(176,106),(156,101),(149,102)],
         rectangle(83,158,163,217)],
        'Strong white reflection obscures parts of the lenses; clear frames and non-glare visible appearance are retained.',
        'Reflected-eye boundaries are soft at256px. This is a conservative glare-only trace, not a claim that all lens pixels are opaque or a segmentation truth.'),
    '06_mirrored_glare': spec(
        [(73,61),(126,40),(168,58),(192,95),(198,142),(185,190),(162,221),(125,237),(97,224),(76,197),(59,151),(60,103)],
        [[(65,117),(78,113),(104,114),(119,118),(122,128),(117,148),(109,157),(91,162),(74,156),(66,143),(61,130)],
         [(138,120),(149,114),(172,113),(188,117),(195,129),(190,148),(182,158),(166,161),(149,155),(140,141)],
         [(119,119),(139,116),(142,128),(121,129)]],
        [rectangle(86,174,169,206)],
        'Opaque mirrored lenses, front rim and bridge hide the eyes; visible hair and lower face are retained.',
        'This mixed glare/sunglasses source is not ordinary transparent eyewear. Temple joins outside the facial region remain.'),
    '07_hand_over_mask': spec(
        [(105,59),(162,55),(191,79),(209,115),(210,157),(199,187),(180,214),(153,234),(124,232),(99,215),(86,187),(81,139),(84,99)],
        [[(82,130),(102,126),(126,130),(148,134),(171,137),(196,122),(213,115),(209,147),(203,162),(199,173),(190,194),(174,215),(154,227),(129,232),(102,220),(89,202),(84,162)]],
        [rectangle(92,109,179,123), rectangle(214,123,226,165)],
        'White mask and hand jointly cover the lower face. Visible eyes, hair and ear are not obstructed.',
        'Hidden jaw and hand/face boundary are approximate. Fingers and wrist outside the intended face remain intact.'),
    'val_18_hand_eyes': spec(
        [(88,57),(120,42),(155,45),(181,63),(194,104),(197,150),(183,189),(163,223),(139,239),(118,238),(95,219),(75,186),(60,143),(61,100)],
        [[(56,106),(75,92),(99,85),(111,85),(119,80),(123,82),(127,89),(121,96),(129,102),(128,109),(118,114),(124,120),(119,127),(115,138),(110,150),(92,159),(79,177),(66,170),(60,151),(55,134)],
         [(131,88),(141,84),(149,79),(158,80),(186,88),(203,104),(204,139),(195,174),(182,180),(163,162),(146,150),(139,138),(143,128),(136,123),(140,116),(130,109),(131,100)]],
        [[(127,124),(134,126),(138,143),(134,153),(121,154),(118,147),(122,135)],
         rectangle(102,169,160,215)],
        'Hands/fingers obscure both eye regions and parts of the cheeks. Central exposed nose and open mouth remain; glasses resting on the head do not obstruct the face.',
        'Finger edges are approximate; hands outside the facial overlap, hairstyle, head-top glasses and earrings are retained.'),
    'val_25_hand_mouth': spec(
        [(58,101),(65,73),(91,54),(126,48),(156,55),(177,76),(191,108),(193,149),(182,184),(167,213),(143,231),(115,236),(91,224),(74,196),(62,156)],
        [[(109,120),(120,121),(126,135),(131,130),(136,132),(144,127),(154,133),(172,157),(187,191),(188,236),(174,255),(57,255),(53,212),(60,176),(73,157),(92,139)]],
        [rectangle(77,106,103,126), rectangle(146,106,176,125)],
        'Cupped hands cover nose/lower face; both eyes and hair are visible. Removal stops at the approximate face boundary.',
        'Hidden chin is not known; palms/wrists beyond the estimated facial overlap remain, instead of removing all hands to the image bottom.'),
    'val_362_hair_eye': spec(
        [(95,66),(129,58),(163,68),(178,92),(192,128),(204,175),(205,211),(194,243),(168,255),(132,252),(107,230),(89,199),(79,157),(81,113)],
        [[(170,70),(181,87),(194,105),(197,131),(205,163),(211,193),(206,224),(193,249),(185,252),(187,226),(181,207),(181,192),(182,182),(176,168),(169,157),(162,140),(160,117),(161,95)]],
        [rectangle(86,102,140,134), [(134,120),(147,119),(154,134),(155,155),(145,168),(130,166)],
         [(121,177),(148,174),(167,179),(172,187),(171,198),(154,206),(130,200),(121,189)]],
        'Dense curls obstruct the right eye and cross the right cheek. Left eye/gaze, visible nose/mouth and ordinary hair outside the face remain.',
        'Strands and hidden right contour are approximate; this does not authorize removing the complete hairstyle or inventing the visible left gaze.'),
    'val_244_knit_scarf': spec(
        [(86,75),(121,65),(150,73),(167,98),(182,136),(191,175),(183,203),(166,230),(137,246),(101,245),(75,225),(58,192),(59,154),(68,113)],
        [[(61,141),(79,126),(94,123),(111,131),(119,133),(136,140),(161,150),(180,165),(183,184),(176,220),(155,236),(130,247),(84,244),(60,220),(51,190)]],
        [[(69,105),(87,102),(96,111),(94,121),(77,124),(69,119)],
         [(109,103),(131,103),(146,112),(147,125),(134,130),(115,123)]],
        'Knit scarf covers nose/lower face; eyes look upward and remain visible. Afro and clothing beyond the face are preserved.',
        'Scarf hides the entire jaw: facial eligibility is an explicit approximate estimate, not hidden-anatomy truth or a target for deleting all clothing.'),
    'val_336_scarf_gloves': spec(
        [(70,72),(112,44),(150,47),(183,74),(204,115),(207,159),(197,190),(181,217),(157,238),(125,245),(94,235),(71,212),(49,179),(43,139),(48,100)],
        [[(42,142),(64,143),(88,146),(110,150),(130,151),(151,148),(176,139),(209,141),(224,158),(230,190),(221,223),(191,246),(135,255),(62,251),(33,208)]],
        [rectangle(61,115,107,137), rectangle(148,114,188,134), rectangle(115,122,137,140)],
        'Scarf and gloves obscure the lower face. The orange knit edge within the face is included; eyes, brows and visible nose bridge are retained.',
        'Gloves and scarf beyond the hidden jaw are not removal targets. Facial outline and thread-level edges are approximate.'),
    'val_6_flower_mouth': spec(
        [(91,62),(121,44),(155,49),(176,76),(188,114),(188,158),(183,189),(171,223),(148,245),(119,248),(96,230),(78,199),(72,154),(73,113)],
        [[(84,167),(88,151),(103,145),(116,149),(131,146),(146,150),(158,145),(172,150),(180,161),(178,176),(183,185),(185,199),(174,203),(172,222),(160,224),(157,238),(146,243),(132,235),(114,236),(104,233),(94,223),(83,218),(82,206),(77,197),(78,181)]],
        [rectangle(76,104,116,133), rectangle(141,103,182,131)],
        'Rose petals cover lower nose/mouth/chin. Visible eyes and blue hair remain; petals outside the approximate face are retained.',
        'Petal tips and covered chin are approximate; no hidden facial reference exists.'),
    'val_7_leaf_eye': spec(
        [(82,55),(104,42),(149,36),(180,54),(195,97),(190,147),(179,194),(162,232),(136,249),(109,246),(85,221),(71,193),(66,152),(66,115),(74,78)],
        [[(92,76),(111,111),(122,141),(120,172),(114,191),(101,203),(85,211),(66,213),(38,198),(13,186),(11,174),(18,139),(33,115),(48,106),(66,100),(77,87)],
         [(39,196),(55,188),(73,192),(82,207),(97,211),(105,225),(105,249),(98,255),(18,255),(14,220)]],
        [rectangle(142,103,180,132), [(128,144),(137,149),(141,164),(135,169),(123,167)],
         [(119,184),(135,181),(150,187),(155,194),(145,203),(123,204),(112,199)]],
        'Leaf hides the left eye/cheek; fingers overlap the lower-left facial area. Right eye, visible nose/lips and surrounding blue hair are retained.',
        'Leaf/hand portions outside the approximate face remain by design. Covered left outline is not known; broad deletion of background or all fingers is not justified.'),
}


def margin(core):
    result = core.copy()
    # Euclidean radius2: thirteen integer offsets. No square/chebyshev dilation.
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dx * dx + dy * dy > 4:
                continue
            y0, y1 = max(0, dy), min(256, 256 + dy)
            x0, x1 = max(0, dx), min(256, 256 + dx)
            result[y0:y1, x0:x1] |= core[y0-dy:y1-dy, x0-dx:x1-dx]
    return result


def main():
    assert not OUT.exists(), 'Retain any earlier mask draft'
    assert sha(PARENT / 'protocol.json') == PARENT_SHA
    p = json.loads((PARENT / 'protocol.json').read_text())
    for name, digest in {**p['sources_sha256'], **p['app_preservation_sha256']}.items():
        assert sha(ROOT / name) == digest, name
    started = time.monotonic()
    prepared, mask_arrays, observations = {}, {}, []
    for c in p['cases']:
        if c['condition'] != 'original_photo': continue
        base = c['id'][:-7]
        if c['input_review'] != 'usable':
            observations.append({'base_id': base, 'decision': c['input_review'],
                                 'new_mask_created': False, 'note': c['input_only_note']})
            continue
        if base in ['08_uncovered', '09_clear_glasses']:
            s = {'face_eligibility_polygons': [], 'covering_polygons': [], 'protected_visible_polygons': [],
                 'input_observation': 'No obstructing covering; preserve the whole image, including ordinary clear glasses/hair.',
                 'boundary_uncertainty': 'Empty mask control, no completion or context margin required.'}
            core = face = np.zeros((256,256), bool); protected = np.ones((256,256), bool)
        else:
            s = ANNOTATIONS[base]; face = polys(s['face_eligibility_polygons'])
            material = polys(s['covering_polygons']); core = material & face
            protected = polys(s['protected_visible_polygons'])
            # Reject a conflicting trace; do not hide conflicts by subtracting
            # observed features from the core after tracing.
            assert not (core & protected).any(), (base, int((core & protected).sum()))
        removal = core | (margin(core) & face & ~protected)
        assert not (removal & protected).any() and not (removal & ~face).any()
        old = np.array(Image.open(ROOT / c['reviewed']).convert('L')) != 0
        paths = {kind: 'masks/' + base + '_' + kind + '.png' for kind in ['core','face','protected','removal']}
        prepared[base] = {'annotation': s, 'paths': paths, 'core_pixels': int(core.sum()),
                          'removal_pixels': int(removal.sum()), 'margin_added_pixels': int((removal & ~core).sum()),
                          'old_pixels': int(old.sum()), 'added_from_old_pixels': int((removal & ~old).sum()),
                          'removed_from_old_pixels': int((old & ~removal).sum())}
        mask_arrays[base] = dict(core=core, face=face, protected=protected, removal=removal)
        observations.append({'base_id': base, 'decision': 'usable', 'new_mask_created': True, **s})
    assert len(prepared) == 16 and len(ANNOTATIONS) == 14 and len(observations) == 18
    OUT.mkdir(); (OUT / 'masks').mkdir(); (OUT / 'atlases').mkdir()
    for base, arrays in mask_arrays.items():
        for kind, array in arrays.items(): Image.fromarray(array.astype(np.uint8) * 255).save(OUT / prepared[base]['paths'][kind])
    write(OUT / 'annotations.json', {'format': 'manual-input-only-removal-footprints-draft-v1',
          'parent_protocol_sha256': PARENT_SHA, 'source_sha256': sha(Path(__file__)),
          'original_draft_source_sha256': sha(ROOT / 'scripts/draft_completion_input_footprints_v1.py'),
          'original_preparation_failure_sha256': sha(PARENT / 'draft_preparation_failure_v1/failure.json'),
          'written_UTC': datetime.now(timezone.utc).isoformat(), 'observations': observations,
          'all18_source_and_degraded_pairs_actually_viewed': True, 'six_source_only_atlases_viewed_at_original_detail': True,
          'additional_individual_input_reviews': ['00_cloth_mask','01_pink_mask','03_dark_sunglasses','04_sunglasses','05_white_glare',
              '06_mirrored_glare','07_hand_over_mask','val_18_hand_eyes','val_25_hand_mouth','val_362_hair_eye',
              'val_244_knit_scarf','val_336_scarf_gloves','val_6_flower_mouth','val_7_leaf_eye'],
          'draft': True, 'new_output_used_for_annotation': False, 'generator_forwards': 0,
          'mask_margin_radius': 2, 'margin_metric': 'Euclidean disk clipped to face eligibility and excluding explicitly protected visible support',
          'geometry_source': 'Original-photo assistance reused for corresponding synthetic degraded photo',
          'approximate_face_boundary_not_hidden_anatomy_truth': True, 'publisher_label': False,
          'model_based_segmentation_truth': False, 'previous_exclusions_retained': 4,
          'all7_covering_families_and2_clear_controls': True, 'prepared': prepared})
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 13)
    cases = [c for c in p['cases'] if c['condition'] == 'original_photo']; pages = []
    for begin in range(0,18,3):
        page = Image.new('RGB', (1280,918), 'white'); draw = ImageDraw.Draw(page); rows = []
        for k,label in enumerate(['Original source256','Old assisted mask','New removal draft','New core / protected','Degraded / new mask']):
            draw.text((k*256+4,7),label,font=font,fill='black')
        for i,c in enumerate(cases[begin:begin+3]):
            base = c['id'][:-7]; degraded_case = next(a for a in p['cases'] if a['id'] == base + '_degraded')
            src = np.array(Image.open(ROOT / c['input']).convert('RGB')); degraded = np.array(Image.open(ROOT / degraded_case['input']).convert('RGB'))
            old = np.array(Image.open(ROOT / c['reviewed']).convert('L')) != 0
            if base in mask_arrays:
                arrays = mask_arrays[base]; removal = arrays['removal']; core = arrays['core']; protected = arrays['protected']
            else:
                removal = core = np.zeros((256,256),bool); protected = np.ones((256,256),bool)
            def overlay(source, selected):
                a = source.astype(np.float64); a[selected] = .55*a[selected] + .45*np.array([16,185,129]); return np.floor(a+.5).astype(np.uint8)
            diagnostic = overlay(src, core); diagnostic[protected] = np.floor(.72*src[protected] + .28*np.array([41,99,230])+.5).astype(np.uint8)
            y = 30+i*296
            for k,arr in enumerate([src,overlay(src,old),overlay(src,removal),diagnostic,overlay(degraded,removal)]):
                page.paste(Image.fromarray(arr),(k*256,y))
            draw.text((4,y+260),base+' | '+c['family']+' | '+c['input_review'],font=font,fill='black')
            rows.append({'base_id':base,'row':i})
        path = OUT/'atlases'/f'page_{begin//3+1:02d}.png';page.save(path)
        pages.append({'path':path.relative_to(OUT).as_posix(),'sha256':sha(path),'rows':rows})
    write(OUT/'draft_receipt.json', {'complete':True,'annotations_sha256':sha(OUT/'annotations.json'),
          'artifacts_sha256':{q.relative_to(OUT).as_posix():sha(q) for q in sorted(OUT.rglob('*')) if q.is_file()},
          'pages':pages,'eligible_pairs':16,'new_nonempty_masks':14,'clear_masks':2,'excluded_pairs':2,
          'generator_forwards':0,'optimizer_updates':0,'app_changes':False,
          'seconds':time.monotonic()-started,'cap_seconds':120,'mask_visual_review_pending':True})
    print(json.dumps({'complete':True,'eligible_pairs':16,'nonempty':14,'empty_controls':2,'input_only_pages':6,'new_generator_calls':0}))


if __name__ == '__main__': main()
