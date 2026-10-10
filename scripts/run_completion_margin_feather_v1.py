"""Generate32 display variants from saved pixels with unchanged core estimates."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from completion_margin_feather_v1 import feather

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'outputs/completion_conditioning_union_v1'
OUT = ROOT / 'outputs/completion_margin_feather_v1'


def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))


def rgb(path):
    with Image.open(path) as im:
        assert im.size == (256, 256); return np.array(im.convert('RGB'))


def binary(path):
    with Image.open(path) as im:
        assert im.mode == 'L' and im.size == (256, 256); v = np.array(im)
    assert set(np.unique(v)) <= {0, 255}; return v != 0


def edge_jump(source, output, final, margin):
    totals, count = 0., 0
    for axis in [0, 1]:
        a = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
        b = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
        boundary = (final[a] != final[b]) & (margin[a] | margin[b])
        values = np.abs(output[a].astype(np.float64) - output[b].astype(np.float64)).mean(axis=-1)
        totals += float(values[boundary].sum()); count += int(boundary.sum())
    return {'edge_pairs':count, 'mean_abs_channel_jump_255':totals / count if count else None,
            'processing_continuity_only_not_quality':True}


def main():
    started = time.monotonic(); p = read(OUT / 'protocol.json')
    assert not (OUT / 'results.json').exists() and not (OUT / 'images').exists()
    for name, digest in {**p['sources_sha256'], **p['app_preservation_sha256']}.items(): assert sha(ROOT / name) == digest, name
    for folder in ['images', 'weights', 'pages']: (OUT / folder).mkdir()
    rows = []
    for c in p['cases']:
        assert time.monotonic() - started < p['cap_seconds']
        if c['rejected']:
            assert c['input_review'] in ['out_of_scope', 'needs_clearer']
            rows.append({'id':c['id'], 'rejected_before_processing':True, 'input_review':c['input_review']}); continue
        source, old = rgb(ROOT / c['input']), rgb(OLD / 'images' / (c['id'] + '.png'))
        core, final, protected = [binary(ROOT / c['masks'][name]) for name in ['core', 'removal', 'protected']]
        assert not (final & protected).any()
        output, weights = feather(source, old, core, final)
        assert np.array_equal(output[~final], source[~final]) and np.array_equal(output[protected], source[protected])
        assert np.array_equal(output[core], old[core])
        Image.fromarray(output).save(OUT / 'images' / (c['id'] + '.png'))
        np.save(OUT / 'weights' / (c['id'] + '.npy'), weights, allow_pickle=False)
        margin = final & ~core
        rows.append({'id':c['id'], 'condition':c['condition'], 'family':c['family'], 'rejected_before_processing':False,
                     'empty_bypass':not bool(final.any()), 'core_pixels':int(core.sum()), 'margin_pixels':int(margin.sum()),
                     'changed_pixels':int(np.any(output != old, axis=-1).sum()), 'changed_core_pixels':0,
                     'source_changed_pixels_outside_final':0, 'protected_changed_pixels':0,
                     'old_boundary_jump':edge_jump(source, old, final, margin), 'new_boundary_jump':edge_jump(source, output, final, margin)})
    assert len(rows) == 36 and sum(r['rejected_before_processing'] for r in rows) == 4
    assert sum(r.get('empty_bypass', False) for r in rows) == 4
    font, pages = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 13), []
    for condition in ['original_photo', 'synthetic_degraded_photo']:
        cases = [c for c in p['cases'] if not c['rejected'] and c['condition'] == condition]
        assert len(cases) == 16
        for first in range(0, 16, 4):
            page = Image.new('RGB', (1280, 1214), 'white'); draw = ImageDraw.Draw(page)
            for k, text in enumerate(['Input256', 'Unchanged core/removal', 'Current hard composite', 'Two-pixel margin feather', 'Feather weight (white=1)']):
                draw.text((k * 256 + 4, 7), text, font=font, fill='black')
            entries = []
            for row, c in enumerate(cases[first:first + 4]):
                source = rgb(ROOT / c['input']); core, final = [binary(ROOT / c['masks'][name]) for name in ['core', 'removal']]
                overlay = source.astype(np.float64); overlay[core] = .55 * overlay[core] + .45 * np.array([16, 185, 129])
                overlay[final & ~core] = .55 * overlay[final & ~core] + .45 * np.array([240, 140, 32])
                weight = np.load(OUT / 'weights' / (c['id'] + '.npy'), allow_pickle=False)
                grey = np.floor(weight * np.float32(255) + np.float32(.5)).astype(np.uint8)
                cells = [source, np.floor(overlay + .5).astype(np.uint8), rgb(OLD / 'images' / (c['id'] + '.png')),
                         rgb(OUT / 'images' / (c['id'] + '.png')), np.repeat(grey[..., None], 3, axis=-1)]
                y = 30 + row * 296
                for k, cell in enumerate(cells): page.paste(Image.fromarray(cell), (k * 256, y))
                draw.text((4, y + 260), c['id'] + ' | ' + c['family'], font=font, fill='black'); entries.append({'id':c['id'], 'row':row})
            path = 'pages/' + condition + '_' + str(first // 4 + 1) + '.png'; page.save(OUT / path)
            pages.append({'path':path, 'sha256':sha(OUT / path), 'entries':entries})
    artifacts = {q.relative_to(OUT).as_posix():sha(q) for folder in ['images', 'weights', 'pages'] for q in sorted((OUT / folder).glob('*'))}
    total = sum((OUT / name).stat().st_size for name in artifacts)
    assert total <= p['artifact_cap_bytes'] and time.monotonic() - started < p['cap_seconds']
    assert not any(name in sys.modules for name in ['torch', 'pretrained_completion', 'dgp_face_workflow_v3'])
    result = {'complete':True, 'protocol_sha256':sha(OUT / 'protocol.json'), 'seconds':time.monotonic() - started,
              'rows':rows, 'pages':pages, 'artifacts_sha256':artifacts, 'artifact_bytes':total, 'expected_outputs':32,
              'model_forwards':0, 'gradient_calls':0, 'optimizer_updates':0, 'new_masks':False, 'app_changes':False,
              'automatic_quality_qualification':False, 'assisted_quality_qualification':False, 'app_adoption':False,
              'visual_review_pending':True, 'independent_final_review':False, 'goal_complete':False}
    with (OUT / 'results.json').open('x', encoding='utf-8', newline='\n') as stream: stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete':True, 'outputs':32, 'artifact_bytes':total, 'seconds':result['seconds'], 'model_forwards':0}))


if __name__ == '__main__': main()
