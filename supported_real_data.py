"""Opt-in reviewed real masks with explicit source and supervision support.

Reads image/mask/valid triples only. No models, optimizers, augmentation or resizing.
The distinct supported_records schema makes legacy pair-only loaders fail early.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
from PIL import Image
import torch

FORMAT = 'dgp-supported-real-masks-v1'
FIELDS = ('image', 'mask', 'valid', 'source_valid')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def data_path(root, value):
    require(isinstance(value, str) and value and '\\' not in value,
            'Portable root-relative PNG path required')
    relative = PurePosixPath(value)
    require(relative.as_posix() == value and not relative.is_absolute()
            and not PureWindowsPath(value).drive and relative.suffix.lower() == '.png'
            and all(p not in ('.', '..') for p in value.split('/')), 'Data path is not canonical')
    path = (root/value).resolve()
    require(path.is_relative_to(root) and path.is_file(), 'Data missing or outside manifest directory')
    return path


def binary(path):
    with Image.open(path) as image:
        a = np.array(image)
    require(a.ndim == 2 and a.dtype == np.uint8 and np.isin(a, [0, 255]).all(),
            'Binary grayscale byte PNG required')
    return a.astype(bool)


def validate_arrays(image, mask, valid, source_valid, size, kind):
    require(isinstance(image, np.ndarray) and image.dtype == np.uint8 and image.shape == (size, size, 3)
            and all(isinstance(a, np.ndarray) and a.dtype == np.bool_ and a.shape == (size, size)
                    for a in (mask, valid, source_valid)), 'Fixed RGB/binary support dimensions required')
    require(kind in ('covered', 'uncovered') and valid.any() and not (valid & ~source_valid).any()
            and not (mask & ~valid).any() and bool(mask.any()) == (kind == 'covered'),
            'Target kind disagrees, or a positive/valid pixel lies outside supervision/source support')
    require(np.all(image[~source_valid] == 96), 'Source padding must be neutral; do not invent input pixels')


def read_item(row, size):
    for key in FIELDS:
        require(sha(row[key+'_path']) == row[key+'_sha256'], 'Data changed after loading: '+key)
    with Image.open(row['image_path']) as image:
        rgb = np.array(image.convert('RGB'))
    mask, valid, source = (binary(row[key+'_path']) for key in ('mask', 'valid', 'source_valid'))
    validate_arrays(rgb, mask, valid, source, size, row['kind'])
    return rgb, mask, valid, source


def load_supported_manifest(path):
    path = Path(path).resolve(); root = path.parent
    data = json.loads(path.read_text(encoding='utf-8'))
    require(data.get('format') == FORMAT and 'records' not in data
            and data.get('dataset_reviewed') is True
            and data.get('support_semantics') == 'valid_one_is_supervised'
            and type(data.get('training_recipe_ready')) is bool
            and type(data.get('size')) is int and data['size'] >= 32, 'Unsupported reviewed-data format or support semantics')
    serialized = data.get('supported_records')
    require(isinstance(serialized, list) and serialized, 'Nonempty supported records required')
    seen = set(); owners = {'group': {}, 'source_sha256': {}}; rows = []
    for record in serialized:
        require(isinstance(record, dict) and record.get('reviewed') is True
                and record.get('support_required') is True
                and record.get('split') in ('train', 'validation', 'test')
                and record.get('kind') in ('covered', 'uncovered')
                and isinstance(record.get('group'), str) and bool(record['group'].strip())
                and digest(record.get('source_sha256')), 'Reviewed group/split/kind/support declaration required')
        row = dict(record)
        for key in FIELDS:
            require(digest(record.get(key+'_sha256')), 'Missing SHA256: '+key)
            file = data_path(root, record.get(key))
            require(sha(file) == record[key+'_sha256'], 'Reviewed artifact hash changed: '+key)
            row[key+'_path'] = str(file)
        require(row['image_sha256'] not in seen, 'Duplicate exact training/held-out image')
        seen.add(row['image_sha256'])
        for key in owners:
            value = row[key]
            require(owners[key].get(value, row['split']) == row['split'], 'Cross-split group or source leakage')
            owners[key][value] = row['split']
        read_item(row, data['size']); rows.append(row)
    require(all({r['kind'] for r in rows if r['split'] == split} == {'covered', 'uncovered'}
                for split in ('train', 'validation', 'test')), 'Each fixed split requires covered and clear cases')
    if 'counts' in data:
        actual = {split: dict(Counter(r['kind'] for r in rows if r['split'] == split))
                  for split in ('train', 'validation', 'test')}
        require(actual == data['counts'], 'Declared supported membership differs')
    return data, rows


class SupportedMasks(torch.utils.data.Dataset):
    """Three tensors; callers must use valid in every supervised reduction."""
    def __init__(self, path, *, split):
        require(split in ('train', 'validation', 'test'), 'Explicit supported split required')
        self.metadata, rows = load_supported_manifest(path)
        self.rows = [r for r in rows if r['split'] == split]
        self.size = self.metadata['size']

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        rgb, mask, valid, _ = read_item(self.rows[index], self.size)
        return (torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()/255,
                torch.from_numpy(mask.astype('float32'))[None],
                torch.from_numpy(valid.astype('float32'))[None])
