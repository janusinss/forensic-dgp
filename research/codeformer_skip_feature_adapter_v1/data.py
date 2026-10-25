"""Inert, prospective 55-source skip-feature pilot data and finite sampler.

Historical admission flags stay unchanged; the new explicit allowlist alone
permits the three specified TRAIN roles in a manually launched sealed packet.
"""
from collections import Counter
import copy
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat

SEED = 20261019
DARK_LINEAR_PROFILE = 'linear_srgb_exposure0p35'
HELD = ('tr_ffhq_21374', 'tr_ffhq_45780')
UPDATES = 110
PROFILES = {'filter': 'gaussian_2p4', 'clear': 'clear', 'dark': DARK_LINEAR_PROFILE}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name,
            'Portable relative path required')
    p = PurePosixPath(name)
    require(not p.is_absolute() and p.as_posix() == name and
            all(v not in ('', '.', '..') for v in p.parts), 'Canonical relative path required')
    return p


def regular(root, name):
    root = Path(root).absolute(); path = root.joinpath(*canonical(name).parts)
    require(path.resolve(strict=True).is_relative_to(root.resolve(strict=True)), 'Packet path escaped root')
    for current in (path, *path.parents):
        st = current.lstat()
        require(not stat.S_ISLNK(st.st_mode) and not getattr(st, 'st_file_attributes', 0) & 1024,
                'Linked/reparse packet path refused')
    st = path.stat()
    require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1, 'Single-link regular packet file required')
    return path


def read_json(path):
    path = Path(path); require(path.stat().st_size <= 4 * 1024**2, 'Bounded metadata required')
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, 'Duplicate metadata field'); value[key] = item
        return value
    return json.loads(path.read_bytes(), object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite metadata')))


def verified_bytes(root, pin):
    require(set(pin) >= {'path', 'bytes', 'sha256'}, 'Complete physical file pin required')
    path = regular(root, pin['path']); before = path.stat()
    require(before.st_size == pin['bytes'] and before.st_size <= 4 * 1024**2, 'Bounded image size differs')
    payload = path.read_bytes(); after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns) and
            hashlib.sha256(payload).hexdigest() == pin['sha256'], 'Physical pixel file changed')
    return payload


def rgb8(root, pin, expected=None):
    import numpy as np
    from PIL import Image
    with Image.open(io.BytesIO(verified_bytes(root, pin))) as image:
        require(image.format == 'PNG' and image.mode == 'RGB' and image.size == (256, 256),
                'Exact prepared RGB PNG256 required')
        array = np.asarray(image).copy()
    require(array.dtype == np.uint8 and array.shape == (256, 256, 3), 'RGB8 geometry differs')
    if expected is not None:
        require(hashlib.sha256(array.tobytes()).hexdigest() == expected, 'Decoded RGB8 pixels changed')
    return array


def schedule_for(order):
    require(type(order) is list and len(order) == len(set(order)) == 55 and
            not any(r in HELD for r in order), 'Exact 55 eligible source order required')
    rows = []
    for u in range(UPDATES):
        slots = []
        for number, role, rid in ((1, 'filter', order[2*u % 55]),
                                  (2, 'filter', order[(2*u+1) % 55]),
                                  (3, 'clear', order[u % 55]), (4, 'dark', order[u % 55])):
            slots.append({'slot': number, 'role': role, 'reference_id': rid,
                          'profile': PROFILES[role], 'id': rid + '__' + PROFILES[role]})
        rows.append({'update': u+1, 'slots': slots})
    return {'format': 'codeformer-skip-feature-finite-schedule-v1', 'seed': SEED,
            'max_updates': UPDATES, 'anchor_source_order': list(order), 'updates': rows,
            'microbatches': 440, 'role_exposures_per_source': {'filter': 4, 'clear': 2, 'dark': 2},
            'automatic_launch_authorized': False}


class FiniteSampler:
    def __init__(self, schedule):
        require(schedule == schedule_for(schedule['anchor_source_order']), 'Frozen finite schedule differs')
        self.schedule = copy.deepcopy(schedule); self.committed = 0

    def state(self):
        return {'completed_updates': self.committed, 'completed_microbatches': self.committed * 4,
                'seed': SEED}

    def preview(self):
        require(self.committed < UPDATES, 'Finite110 schedule exhausted; no extension')
        return copy.deepcopy(self.schedule['updates'][self.committed])

    def commit(self, row):
        require(row == self.preview(), 'Partial/stale/altered update cannot commit')
        self.committed += 1


def _local_metadata(root, protocol):
    root = Path(root); spec = protocol['data']
    bundle = read_json(regular(root, spec['bundle_path']))
    schedule = read_json(regular(root, spec['schedule_path']))
    admission = read_json(regular(root, spec['admission_path']))
    order = schedule['anchor_source_order']; require(schedule == schedule_for(order), 'Finite schedule changed')
    cases = bundle['case_by_id']
    require(bundle['format'] == 'codeformer-skip-feature-training-bundle-v1' and
            len(cases) == 165 and all(r['split'] == 'TRAIN' for r in cases.values()), 'Exact165 TRAIN roles required')
    expected = {(role, rid+'__'+profile) for rid in order for role, profile in PROFILES.items()}
    actual = {(r['active_role'], r['id']) for r in cases.values()}
    require(actual == expected and admission['complete'] is True and admission['root_approved'] is True and
            {(r['role'], r['id']) for r in admission['admitted_role_cases']} == expected and
            admission['old_flags_unmodified'] is True and admission['automatic_launch_authorized'] is False,
            'Separate exact prospective admission required')
    for u in schedule['updates']:
        for slot in u['slots']:
            row = cases[slot['id']]
            require(row['reference_id'] == slot['reference_id'] and row['profile'] == PROFILES[slot['role']],
                    'Schedule case/source/role join differs')
    evaluation_bundle = read_json(regular(root, protocol['evaluation_path']))
    evaluation = evaluation_bundle['cases']
    require(len(evaluation) == len({r['id'] for r in evaluation}) == 478 and
            Counter(r['cohort'] for r in evaluation) == {'historical_DEV406': 406, 'retained_filter72_mixed_roles': 72},
            'Full478 historical comparison scope required')
    require(len(evaluation_bundle['refused_controls']) == 24 and evaluation_bundle['reserved_FINAL_used'] is False,
            '24 no-forward refusals and reserved FINAL must remain')
    require(sum(r['evidence'] == 'native_unpaired' for r in evaluation) == 48 and
            all(r.get('target256', r.get('target')) is None for r in evaluation if r['evidence'] == 'native_unpaired'),
            '48 native CCTV inputs remain unpaired')
    auxiliary = read_json(regular(root, protocol['auxiliary_path']))['cases']
    require(len(auxiliary) == len({r['id'] for r in auxiliary}) == 18 and
            len({r['reference_id'] for r in auxiliary}) == 6 and
            Counter(r['input_only_decision'] for r in auxiliary) == {'readable': 16, 'borderline': 2} and
            all(r['optimization_permitted_now'] is False and r['split'] == 'TRAIN' for r in auxiliary),
            'Fixed18 TRAIN-only auxiliary probes required; no optimization')
    guards = protocol['guard_cases']
    require(len(guards) == 24 and Counter(r['role'] for r in guards) == {'filter': 8, 'clear': 8, 'dark': 8} and
            {r['reference_id'] for r in guards} == set(order[:8]) and all(r['id'] in cases for r in guards),
            'Frozen first8-source TRAIN guards differ')
    require(set(protocol['witness_cases']) == set(PROFILES) and
            all(protocol['witness_cases'][role] == order[0]+'__'+profile for role, profile in PROFILES.items()),
            'Fixed three role witnesses differ')
    return cases, schedule, evaluation, auxiliary
