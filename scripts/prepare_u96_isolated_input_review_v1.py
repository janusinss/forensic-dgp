"""Prepare anonymous degraded-only review; no model, training or result reads."""
from pathlib import Path
import hashlib
import json
import random
import shutil
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/codeformer_u96_context_study_v1'
OUT = ROOT / 'scratch/u96_isolated_degraded_input_review_v1'


def pin(path):
    raw = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


def write(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    assert not OUT.exists(), 'Never overwrite a prior review packet'
    plan = json.loads((BASE / 'comparison_plan_v1.json').read_bytes())
    criteria_path = ROOT / 'outputs/codeformer_filter_pairs_v1_preparation/input_only_review_criteria_v1.json'
    criteria = json.loads(criteria_path.read_bytes())
    inputs = [row for row in plan['cases'] if row['role'] != 'clear']
    assert len(inputs) == 16
    random.Random(2026102103).shuffle(inputs)
    OUT.mkdir(parents=True)
    mapping = []
    for index, row in enumerate(inputs, 1):
        source = ROOT / row['input']['path']
        assert pin(source) == row['input']
        anonymous = OUT / f'input_{index:02d}.png'
        shutil.copyfile(source, anonymous)
        mapping.append({'anonymous_id': f'I{index:02d}', 'id': row['id'],
                        'role': row['role'], 'reference_id': row['reference_id'],
                        'original_input': row['input'], 'anonymous_input': pin(anonymous)})
    sheets = []
    for start in range(0, 16, 4):
        sheet = Image.new('RGB', (4 * 272, 296), '#e6e2dc')
        draw = ImageDraw.Draw(sheet)
        for slot, row in enumerate(mapping[start:start+4]):
            draw.text((slot*272+8, 8), row['anonymous_id'], fill='black')
            with Image.open(ROOT / row['anonymous_input']['path']) as image:
                assert image.size == (256, 256) and image.mode == 'RGB'
                sheet.paste(image, (slot*272+8, 32))
        path = OUT / f'inputs_{start//4+1:02d}.png'
        sheet.save(path)
        sheets.append(pin(path))
    write(OUT / 'review_rules.json', {
        'purpose': 'Ancillary degraded-input-only admission review. Not training or FINAL evaluation.',
        'observation': criteria['observation'], 'decisions': criteria['decisions'],
        'prohibited_decision_evidence': criteria['prohibited_decision_evidence'],
        'do_not_open_other_project_files': True,
        'only_review_anonymous_inputs_and_rules': True,
        'sheets': sheets})
    write(BASE / 'isolated_input_review_map_v1.json', {
        'complete': True, 'review_stage': 'post-freeze ancillary review while inference was running',
        'reason': 'Earlier root sheets included clear controls beside degraded counterparts. Those labels were not strictly input-only.',
        'frozen_plan_changed': False, 'outputs_or_metrics_used_for_packet': False,
        'criteria': pin(criteria_path), 'rules': pin(OUT/'review_rules.json'),
        'sheets': sheets, 'cases': mapping,
        'clean_references_or_model_outputs_in_review_packet': False,
        'independent_FINAL_claim': False})
    print(json.dumps({'complete': True, 'degraded_inputs': 16,
                      'sheets': 4, 'neural_calls': 0, 'training_updates': 0,
                      'packet': str(OUT)}))


if __name__ == '__main__':
    main()
