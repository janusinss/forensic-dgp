"""Prospective V2 paired photographic data; pure CPU, no neural dependency.

Implements design a7105b577863e22c0bc35ce12d310234a510eec41c8115ebeec2e9b46bb511b9.
Only provenance/verified direct-1024 derivation helpers are reused from immutable
V1. V1 profile generation and degradation are never invoked. Normal/dark members
share independent SHA-keyed PCG64 noise fields. Camera ranges are declared
synthetic stress proxies, not a fitted native CCTV distribution. No crop, warp,
covering simulation, glasses removal, alignment or acceptance gate is applied.

The clear control still traverses inverse/forward sRGB and floor quantization,
as prospectively declared; floating round-trip identity to RGB8 is not assumed.
DEV pixels must be loaded from the saved V1 manifest by the worker, never newly
generated here. Only the exact historical TRAIN namespace can enter this path.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
from numbers import Integral
import re
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import __version__ as PILLOW_VERSION

from research.codeformer_adaptation_v1 import data as audited


Reference = audited.Reference
PairedExample = audited.PairedExample
load_reference_manifest = audited.load_reference_manifest
derive_clean_target = audited.derive_clean_target
file_sha256 = audited.file_sha256
pixel_sha256 = audited.pixel_sha256

SEED = 20261015
MAX_UPDATES = 600
TRAIN_COUNT = 391
A_EDGES = (24, 28, 32, 40)
B_VARIANTS = ('clear256', 'clear128', 'clear64', 'light256', 'light128', 'light64')
CAMERA_POLICY = 'paired-linear-light-keyed-fields-v2'
DESIGN_SHA256 = 'a7105b577863e22c0bc35ce12d310234a510eec41c8115ebeec2e9b46bb511b9'


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _train_id(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r'tr_ffhq_[0-9]{5}', value) is None:
        raise ValueError('Only historical FFHQ TRAIN IDs are permitted')


def _update(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= int(value) <= MAX_UPDATES:
        raise ValueError('V2 update must be an integer from 1 through 600')
    return int(value)


def cycle_spec(update: int) -> dict[str, Any]:
    """One-based update; all 24 A-edge/B-variant combinations repeat 25 times."""
    update = _update(update)
    u = update - 1
    variant = B_VARIANTS[(u // 4) % 6]
    family = 'clear' if variant.startswith('clear') else 'light'
    return {'update': update, 'zero_based_update': u, 'a_edge': A_EDGES[u % 4],
            'b_variant': variant, 'b_family': family,
            'b_edge': int(variant[len(family):])}


def _seed_record(reference_id: str, pair_role: str, purpose: str, update: int) -> dict[str, Any]:
    key = (f'codeformer-adaptation-v2|{SEED}|{update - 1}|'
           f'{reference_id}|{pair_role}|{purpose}')
    return {'key': key,
            'seed64': int.from_bytes(hashlib.sha256(key.encode('utf-8')).digest()[:8], 'little')}


def camera_parameters(reference_id: str, pair_role: str, update: int) -> dict[str, Any]:
    """Draw the frozen shared pair recipe without decoding any source pixels."""
    _train_id(reference_id)
    spec = cycle_spec(update)
    if pair_role not in ('A', 'B'):
        raise ValueError('Pair role must be A or B')
    seeds = {purpose: _seed_record(reference_id, pair_role, purpose, spec['update'])
             for purpose in ('parameters', 'linear_noise', 'rgb_noise')}
    rng = np.random.Generator(np.random.PCG64(seeds['parameters']['seed64']))
    p = {'camera_policy': CAMERA_POLICY, 'update': spec['update'],
         'zero_based_update': spec['zero_based_update'], 'pair_role': pair_role,
         'gaussian_sigma_256': 0., 'jpeg_quality': None,
         'shot_coefficient': 0., 'read_sigma_linear': 0., 'rgb_noise_sigma_255': 0.,
         'seeds': seeds, 'numpy_bit_generator': 'PCG64',
         'normal_exposure_linear': 1., 'motion_kernel_size_256': 0}
    if pair_role == 'A':
        p.update(family='severe', variant='severe' + str(spec['a_edge']), long_edge=spec['a_edge'])
        # Draw order is part of the prospectively pinned recipe.
        p['gaussian_sigma_256'] = float(rng.uniform(.6, 2.))
        p['jpeg_quality'] = int(rng.integers(35, 66))
        p['shot_coefficient'] = float(rng.uniform(.0004, .0012))
        p['read_sigma_linear'] = float(rng.uniform(.001, .003))
        p['dark_exposure_linear'] = float(rng.uniform(.15, .65))
    else:
        p.update(family=spec['b_family'], variant=spec['b_variant'], long_edge=spec['b_edge'])
        if p['family'] == 'light':
            p['gaussian_sigma_256'] = float(rng.uniform(.5, 1.5))
            p['rgb_noise_sigma_255'] = float(rng.uniform(1, 12))
            p['jpeg_quality'] = int(rng.integers(35, 96))
        p['dark_exposure_linear'] = float(rng.uniform(.2, .65))
    p['image_eligible'] = p['long_edge'] >= 32
    p['image_eligibility_policy'] = 'generated observation edge >=32; no other exclusion'
    return p


def _noise_fields(parameters: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    shape = (parameters['long_edge'], parameters['long_edge'], 3)
    fields = []
    for purpose in ('linear_noise', 'rgb_noise'):
        rng = np.random.Generator(np.random.PCG64(parameters['seeds'][purpose]['seed64']))
        fields.append(np.ascontiguousarray(rng.standard_normal(shape), dtype=np.float64))
    return fields[0], fields[1]


def _observation_base(rgb8: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    rgb = rgb8.astype(np.float32) / np.float32(255)
    if parameters['gaussian_sigma_256'] > 0:
        rgb = cv2.GaussianBlur(rgb, (0, 0), parameters['gaussian_sigma_256'],
                               borderType=cv2.BORDER_REPLICATE)
    edge = parameters['long_edge']
    if edge < 256:
        rgb = cv2.resize(rgb, (edge, edge), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb, dtype=np.float32)


def _render_camera(base: np.ndarray, parameters: dict[str, Any], exposure: float,
                   linear_noise: np.ndarray, rgb_noise: np.ndarray) -> np.ndarray:
    """Exact declared camera, including the sRGB path for normal clear inputs."""
    linear = np.where(base <= np.float32(.04045), base / np.float32(12.92),
                      ((base + np.float32(.055)) / np.float32(1.055)) ** np.float32(2.4))
    signal = np.float32(exposure) * linear
    deviation = np.sqrt(np.float32(parameters['shot_coefficient']) * signal
                        + np.float32(parameters['read_sigma_linear'] ** 2))
    noisy = np.clip(signal + linear_noise * deviation, 0., 1.)
    rgb = np.where(noisy <= .0031308, 12.92 * noisy,
                   1.055 * noisy ** (1 / 2.4) - .055)
    rgb = rgb + rgb_noise * (parameters['rgb_noise_sigma_255'] / 255)
    observed8 = np.floor(np.clip(rgb, 0., 1.) * 255).astype(np.uint8)
    quality = parameters['jpeg_quality']
    if quality is not None:
        ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(observed8, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ValueError('V2 JPEG encode failed; no fallback permitted')
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None or decoded.shape != observed8.shape:
            raise ValueError('V2 JPEG decode failed or changed geometry')
        observed8 = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    if parameters['long_edge'] < 256:
        observed8 = cv2.resize(observed8, (256, 256), interpolation=cv2.INTER_CUBIC)
    if observed8.shape != (256, 256, 3) or observed8.dtype != np.uint8:
        raise ValueError('Invalid V2 observation geometry')
    return np.ascontiguousarray(observed8)


def _make_matched_pair(reference: Reference, pair_role: str,
                       update: int) -> tuple[PairedExample, PairedExample]:
    parameters = camera_parameters(reference.id, pair_role, update)
    # The shared immutable helper hashes the same bytes it decodes and verifies
    # both original1024 geometry and direct512 target pixels before returning.
    base256, clean512_rgb8 = audited._verified_derivations(reference)
    clean512 = np.ascontiguousarray(clean512_rgb8.astype(np.float32) / np.float32(255))
    base = _observation_base(base256, parameters)
    linear_noise, rgb_noise = _noise_fields(parameters)
    noise = {'shape': list(linear_noise.shape), 'dtype': 'float64',
             'linear_noise_sha256': pixel_sha256(linear_noise),
             'rgb_noise_sha256': pixel_sha256(rgb_noise)}
    pair_id = f'v2_u{update:04d}_{pair_role}_{reference.id}'
    result = []
    for offset, member in enumerate(('normal', 'dark')):
        exposure = 1. if member == 'normal' else parameters['dark_exposure_linear']
        observed8 = _render_camera(base, parameters, exposure, linear_noise, rgb_noise)
        prepared = np.ascontiguousarray(observed8.astype(np.float32) / np.float32(255))
        slot = (1 if pair_role == 'A' else 3) + offset
        p = copy.deepcopy(parameters)
        p.update(pair_id=pair_id, member=member, slot=slot, exposure_linear=exposure,
                 behavior_anchor=(pair_role == 'B' and member == 'normal'),
                 noise_fields=copy.deepcopy(noise))
        record = {'format': 'codeformer-adaptation-matched-pair-v2',
                  'reference_id': reference.id, 'role': reference.role,
                  'source_path': reference.source_relative_path,
                  'source_sha256': reference.source_sha256,
                  'reference_manifest_sha256': reference.manifest_sha256,
                  'profile': parameters['family'] + '_' + member,
                  'pair_id': pair_id, 'pair_role': pair_role, 'member': member,
                  'slot': slot, 'update': update, 'parameters': p,
                  'input256_rgb8_sha256': pixel_sha256(observed8),
                  'input256_float32_sha256': pixel_sha256(prepared),
                  'clean512_rgb8_sha256': reference.clean512_rgb8_sha256,
                  'clean512_float32_sha256': pixel_sha256(clean512),
                  'geometry': {'source': [1024, 1024], 'input': [256, 256],
                               'target': [512, 512], 'source_to_input': 'Pillow BILINEAR',
                               'source_to_target': 'Pillow LANCZOS', 'alignment_changed': False,
                               'crop': False, 'covering_simulation': False},
                  'runtime': {'numpy': np.__version__, 'Pillow': PILLOW_VERSION,
                              'OpenCV': cv2.__version__},
                  'attribution': reference.attribution}
        result.append(PairedExample(prepared, clean512, record))
    return result[0], result[1]


def make_training_update(ref_a: Reference, ref_b: Reference,
                          update: int) -> tuple[PairedExample, PairedExample, PairedExample, PairedExample]:
    """Materialize exactly slots A-normal/A-dark/B-normal/B-dark, TRAIN only.

    RGB arrays are contiguous HWC float32[0,1], input256/target512. Pair fields
    are shared, not re-sampled through conditional branches. No GT/DEV sampling
    or file mutation occurs. This does not execute a model or optimizer.
    """
    update = _update(update)
    for reference in (ref_a, ref_b):
        if not isinstance(reference, Reference) or reference.role != 'train':
            raise ValueError('V2 training pairs require audited TRAIN references')
        _train_id(reference.id)
    if (ref_a.id == ref_b.id or ref_a.source_sha256 == ref_b.source_sha256
            or ref_a.source_path == ref_b.source_path):
        raise ValueError('Each V2 update needs two distinct photographic TRAIN references')
    return (*_make_matched_pair(ref_a, 'A', update), *_make_matched_pair(ref_b, 'B', update))


@dataclass(frozen=True)
class Selection:
    update: int
    id_a: str
    id_b: str
    rotation_events: tuple[dict[str, Any], ...]
    state_before_sha256: str
    state_after_sha256: str
    _state_after: dict[str, Any] = field(repr=False)

    def to_record(self) -> dict[str, Any]:
        """Compact trajectory entry; full state belongs only in checkpoints."""
        return {'update': self.update, 'id_a': self.id_a, 'id_b': self.id_b,
                'rotation_events': copy.deepcopy(list(self.rotation_events)),
                'state_before_sha256': self.state_before_sha256,
                'state_after_sha256': self.state_after_sha256}


class TrainSampler:
    """Finite deterministic photographic sampler with explicit commit boundary.

    Preview is pure/idempotent. The worker must commit only after successful
    optimizer.step and retain the previous committed state for interruptions.
    This class cannot launch training or automatically resume a failed recipe.
    """
    def __init__(self, train_ids: Sequence[str], seed: int = SEED):
        if isinstance(seed, bool) or not isinstance(seed, Integral) or int(seed) != SEED:
            raise ValueError('The prospective V2 sampler seed is fixed at 20261015')
        ids = sorted(train_ids)
        if len(ids) != TRAIN_COUNT or len(set(ids)) != TRAIN_COUNT:
            raise ValueError('The V2 sampler requires exactly 391 distinct TRAIN IDs')
        for rid in ids:
            _train_id(rid)
        self._ids = ids
        rng = np.random.Generator(np.random.PCG64(SEED))
        self._state = {'format': 'codeformer-train-sampler-v2', 'seed': SEED,
                       'reference_ids_sha256': _digest(ids), 'completed_updates': 0,
                       'next_update': 1, 'selection_count': 0, 'permutation_index': -1,
                       'permutation': [], 'cursor': 0,
                       'rng_state': copy.deepcopy(rng.bit_generator.state),
                       'rotation_history': []}

    def state_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._state)

    def preview_update(self) -> Selection:
        state = copy.deepcopy(self._state)
        update = state['completed_updates'] + 1
        _update(update)
        before = _digest(state)
        rng = np.random.Generator(np.random.PCG64(SEED))
        rng.bit_generator.state = copy.deepcopy(state['rng_state'])
        events = []

        def draw(avoid: str | None = None) -> str:
            if state['cursor'] == len(state['permutation']):
                state['permutation'] = rng.permutation(self._ids).tolist()
                state['cursor'] = 0
                state['permutation_index'] += 1
                if avoid is not None and state['permutation'][0] == avoid:
                    amount = 1
                    while state['permutation'][amount] == avoid:
                        amount += 1
                    state['permutation'] = state['permutation'][amount:] + state['permutation'][:amount]
                    event = {'permutation_index': state['permutation_index'],
                             'left_rotation': amount, 'avoided_id': avoid, 'update': update}
                    events.append(event)
                    state['rotation_history'].append(event)
            value = state['permutation'][state['cursor']]
            state['cursor'] += 1
            return value

        id_a = draw()
        id_b = draw(avoid=id_a)
        if id_a == id_b:
            raise ValueError('Sampler produced identical A/B references')
        state.update(completed_updates=update, next_update=update + 1,
                     selection_count=2 * update, rng_state=copy.deepcopy(rng.bit_generator.state))
        return Selection(update, id_a, id_b, tuple(events), before, _digest(state), state)

    def commit(self, selection: Selection) -> None:
        if not isinstance(selection, Selection) or selection != self.preview_update():
            raise ValueError('Commit requires the exact next pending sampler selection')
        self._state = copy.deepcopy(selection._state_after)
