"""Artifact helpers for one frozen completion architecture diagnostic."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/completion_conditioning_union_v1'
OUT = ROOT/'outputs/completion_feature_fusion_off_v1'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False)+'\n')


def rgb(path):
    with Image.open(path) as im:
        assert im.size == (256, 256) and im.mode == 'RGB'
        return np.array(im)


def binary(path):
    with Image.open(path) as im:
        assert im.mode == 'L' and im.size == (256, 256); array = np.array(im)
    assert set(np.unique(array)) <= {0, 255}
    return array != 0


def overlay(source, mask, colour):
    values = source.astype(np.float64)
    values[mask] = .55*values[mask]+.45*np.asarray(colour)
    return np.floor(values+.5).astype(np.uint8)


def verify_bindings(p):
    for name, digest in {**p['sources_sha256'], **p['app_preservation_sha256']}.items():
        assert sha(ROOT/name) == digest, name
