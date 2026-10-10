"""Source-bound helpers for an input/proposal-only detector trace."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/automatic_proposal_score_v1'
BUDGETS = {'worker_seconds': 180, 'external_seconds': 210, 'terminate_seconds': 20,
           'artifact_bytes': 64*1024**2, 'detector_forwards': 36, 'audit_detector_forwards': 2}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''): h.update(chunk)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def rgb(path):
    with Image.open(path) as source:
        assert source.size == (256, 256) and source.mode == 'RGB'
        return np.array(source)


def binary(path):
    with Image.open(path) as source:
        assert source.size == (256, 256) and source.mode == 'L'; mask = np.array(source)
    assert np.isin(mask, [0, 255]).all()
    return mask == 255


def overlay(image, mask, color):
    result = image.astype(np.float64)
    result[mask] = .55*result[mask]+.45*np.asarray(color)
    return np.floor(result+.5).astype(np.uint8)


def stats(values, region):
    selected = values[region].astype(np.float64)
    if not selected.size: return {'pixels': 0, 'minimum': None, 'maximum': None,
        'mean': None, 'q10': None, 'median': None, 'q90': None, 'fraction_ge_0_5': None}
    return {'pixels': int(selected.size), 'minimum': float(selected.min()), 'maximum': float(selected.max()),
        'mean': float(selected.mean()), 'q10': float(np.quantile(selected, .1)), 'median': float(np.median(selected)),
        'q90': float(np.quantile(selected, .9)), 'fraction_ge_0_5': float((selected >= .5).mean())}


def regions(case):
    if not case.get('masks'): return {}
    masks = {key: binary(ROOT/name) for key, name in case['masks'].items()}
    return {'assisted_core': masks['core'], 'visible_face': masks['face'] & ~masks['removal'],
            'protected_appearance': masks['protected'], 'outside_face': ~masks['face']}


def state_sha(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def bindings(protocol):
    assert protocol['format'] == 'automatic-proposal-score-trace-v1' and protocol['budgets'] == BUDGETS
    assert len(protocol['cases']) == 36 and len({c['id'] for c in protocol['cases']}) == 36
    assert sum(c['rejected'] for c in protocol['cases']) == 4
    assert protocol['threshold'] == .5 and protocol['margin_pixels_256'] == 3
    assert protocol['completion_forwards'] == protocol['DGP_forwards'] == protocol['optimizer_updates'] == 0
    assert not protocol['native_or_final_used'] and not protocol['app_adoption'] and not protocol['goal_complete']
    for name, digest in protocol['sources_sha256'].items(): assert sha(ROOT/name) == digest, name
