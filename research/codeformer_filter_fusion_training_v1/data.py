"""Finite filter/clear/dark data contract; imports perform no neural work.

Historical pixels and decisions are immutable. A schedule is prospective until
the root's separate admission and dark-input review have both been sealed.
No camera rendering, checkpoint loading, seed search, or implicit resume exists.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat

SEED = 20261018
UPDATES = 157
PADDING_ID = 'tr_ffhq_03212__global_mosaic_8'
PROFILE_COUNTS = {'gaussian_2p4': 57, 'anisotropic_h5_v0p7': 57,
                  'anisotropic_h0p7_v5': 53, 'repeated_rows_4_blend0p75': 57,
                  'global_mosaic_8': 12, 'local_gaussian_3p0': 52,
                  'local_mosaic_8': 25}
ROLES = ('filter_a', 'filter_b', 'clear', 'dark')
RID = re.compile(r'tr_ffhq_[0-9]{5}')
DARK_LINEAR_EXPOSURE = .35
DARK_LINEAR_PROFILE = 'linear_srgb_exposure0p35'


def linear_dark_proxy(rgb):
    """Fixed brightness-only retention proxy, RGB8 input/output, no geometry.

    This is not the historical noisy/blurred lowlight_lr32 camera. Parameters
    are frozen before new rendering; no image-dependent choice is performed.
    """
    import numpy as np
    require(isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and
            rgb.ndim == 3 and rgb.shape[-1] == 3, 'RGB8 array required')
    value = rgb.astype(np.float64) / 255.
    linear = np.where(value <= .04045, value / 12.92, ((value + .055) / 1.055) ** 2.4)
    linear = linear * DARK_LINEAR_EXPOSURE
    out = np.where(linear <= .0031308, linear * 12.92,
                   1.055 * linear ** (1. / 2.4) - .055)
    return np.ascontiguousarray(np.floor(np.clip(out, 0., 1.) * 255. + .5), dtype=np.uint8)


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def relative(name):
    require(isinstance(name, str) and name and '\\' not in name,
            'Portable nonempty POSIX relative path required')
    path = PurePosixPath(name)
    require(not path.is_absolute() and ':' not in name and
            all(p not in ('', '.', '..') for p in name.split('/')),
            'Unsafe path')
    return path


def verified_bytes(root, pin):
    """Hash the exact regular singly-linked file buffer subsequently decoded.

    Ancestor symlinks and Windows reparse points are rejected. This is an I/O
    race guard, not an OS-level immutable-file or hostile-actor guarantee.
    """
    name = str(relative(pin['path']))
    root = Path(root).absolute()
    for ancestor in (root, *root.parents):
        info = ancestor.lstat()
        require(not ancestor.is_symlink() and not
                getattr(info, 'st_file_attributes', 0) & 0x400,
                'Linked/reparse data root or ancestor rejected')
    require(root.is_dir(), 'Existing data root required')
    path = root
    for component in relative(name).parts:
        path = path / component
        info = path.lstat()
        require(not path.is_symlink() and not
                getattr(info, 'st_file_attributes', 0) & 0x400,
                'Linked/reparse data path rejected')
    before = path.stat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and
            before.st_size == pin['bytes'], 'Regular exact singly-linked file required')
    payload = path.read_bytes()
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns),
            'File changed during read')
    require(hashlib.sha256(payload).hexdigest() == pin['sha256'], 'Data hash mismatch')
    return payload


def rgb8(root, pin, expected_pixel_sha256, size=256):
    """Decode only the verified buffer, never a second unverified file read."""
    import numpy as np
    from PIL import Image
    with Image.open(io.BytesIO(verified_bytes(root, pin))) as image:
        require(image.mode == 'RGB' and image.size == (size, size), 'Exact RGB geometry required')
        image.load()
        pixels = np.asarray(image).copy()
    require(hashlib.sha256(pixels.tobytes()).hexdigest() == expected_pixel_sha256,
            'RGB8 pixel digest mismatch')
    return pixels


def validate_holds(document):
    require(document.get('format') == 'codeformer-filter-train-v2-conservative-candidate-holds-v1'
            and document.get('complete') is True, 'Completed frozen hold ledger required')
    rows = document['all512_cases']
    require(len(rows) == len({r['id'] for r in rows}) == 512, 'All512 historical observations retained')
    require(all(r.get('optimization_permitted_now') is False for r in rows),
            'Historical ledger must remain unmodified/non-authorizing')
    selected = [copy.deepcopy(r) for r in rows if r['candidate_for_finite_packet_preparation']]
    require(len(selected) == 370 and len({r['reference_id'] for r in selected}) == 57,
            'Conservative370/57 cohort differs')
    clear = [r for r in selected if r['profile'] == 'clear']
    filtered = [r for r in selected if r['profile'] != 'clear']
    require(len(clear) == 57 and Counter(r['profile'] for r in filtered) == PROFILE_COUNTS,
            'Conservative profile census differs')
    require(all(RID.fullmatch(r['reference_id']) and r['root_input_decision'] == 'readable'
                and r['root_source_clear_decision'] == 'readable' and not r['hold_reasons']
                for r in selected), 'Held, unreadable or non-TRAIN candidate entered cohort')
    require(all(r['second_observer_global_decision'] == 'readable' for r in filtered
                if r['profile'] == 'global_mosaic_8'), 'Global mosaic needs two readable decisions')
    require(PADDING_ID == min(r['id'] for r in filtered if r['profile'] == 'global_mosaic_8'),
            'Fixed global padding rule differs')
    return selected


def _order(domain, value):
    return hashlib.sha256(f'codeformer-filter-fusion-v1|{SEED}|{domain}|{value}'.encode()).hexdigest()


def filter_sequence(filtered):
    """Weighted-fair deterministic interleave; every admitted case once plus pad.

    At each position choose the remaining profile with the smallest emitted /
    total ratio (cross multiplication; no floating rounding). Ties and order
    inside a profile use a single fixed SHA256-seeded order, never quality.
    """
    require(Counter(r['profile'] for r in filtered) == PROFILE_COUNTS,
            'Exact313 filter candidates required')
    require(len({r['id'] for r in filtered}) == 313, 'Unique filter IDs required')
    groups = defaultdict(list)
    for row in filtered:
        groups[row['profile']].append({'id': row['id'], 'reference_id': row['reference_id'],
                                      'profile': row['profile'], 'padding_duplicate': False})
    padded = copy.deepcopy(next(r for r in groups['global_mosaic_8'] if r['id'] == PADDING_ID))
    padded['padding_duplicate'] = True
    groups['global_mosaic_8'].append(padded)
    for profile, rows in groups.items():
        rows.sort(key=lambda r: (_order('filter', r['id']), r['id'], r['padding_duplicate']))
    consumed = {p: 0 for p in groups}
    sequence = []
    for _ in range(314):
        remaining = [p for p in groups if consumed[p] < len(groups[p])]
        best = remaining[0]
        for profile in remaining[1:]:
            left, right = consumed[profile] * len(groups[best]), consumed[best] * len(groups[profile])
            if left < right or (left == right and _order('profile', profile) < _order('profile', best)):
                best = profile
        sequence.append(copy.deepcopy(groups[best][consumed[best]]))
        consumed[best] += 1
    return sequence


def build_schedule(holds, dark_joins):
    """Prospective exact628 slots; does not itself grant training admission."""
    rows = validate_holds(holds)
    filters = filter_sequence([r for r in rows if r['profile'] != 'clear'])
    clear_by_ref = {r['reference_id']: r['id'] for r in rows if r['profile'] == 'clear'}
    require(len(dark_joins) == 57 and len({r['reference_id'] for r in dark_joins}) == 57
            and set(clear_by_ref) == {r['reference_id'] for r in dark_joins},
            'Same-reference dark joins required for all57 anchors')
    require(all(r['historical_role'] == 'train' and r['profile'] == 'lowlight_lr32'
                for r in dark_joins), 'Only audited historical TRAIN lowlight anchors allowed')
    dark_by_ref = {r['reference_id']: r['id'] for r in dark_joins}
    anchors = sorted(clear_by_ref, key=lambda rid: (_order('anchor', rid), rid))
    updates = []
    for index in range(UPDATES):
        ref = anchors[index % len(anchors)]
        slots = []
        for number, role in enumerate(ROLES, 1):
            if number <= 2:
                item = copy.deepcopy(filters[2 * index + number - 1])
            else:
                item = {'id': clear_by_ref[ref] if number == 3 else dark_by_ref[ref],
                        'reference_id': ref, 'profile': 'clear' if number == 3 else 'lowlight_lr32',
                        'padding_duplicate': False}
            item.update(slot=number, role=role)
            slots.append(item)
        updates.append({'update': index + 1, 'slots': slots})
    return {'format': 'codeformer-filter-fusion-finite-schedule-v1', 'seed': SEED,
            'max_updates': UPDATES, 'slot_roles': list(ROLES), 'updates': updates,
            'filter_slots': 314, 'clear_slots': 157, 'dark_slots': 157,
            'unique_filter_candidates': 313, 'padding_duplicate_id': PADDING_ID,
            'padding_duplicate_count': 1, 'anchor_source_order': anchors,
            'clear_and_dark_same_reference_each_update': True,
            'automatic_training_or_resume_authorized': False}


class FiniteSampler:
    """Pure preview, explicit commit only after a successful optimizer update."""
    def __init__(self, schedule):
        require(schedule.get('format') == 'codeformer-filter-fusion-finite-schedule-v1'
                and schedule.get('seed') == SEED and schedule.get('max_updates') == UPDATES
                and len(schedule['updates']) == UPDATES, 'Exact finite schedule required')
        require([r['update'] for r in schedule['updates']] == list(range(1, UPDATES + 1))
                and all([s['role'] for s in r['slots']] == list(ROLES) and
                        [s['slot'] for s in r['slots']] == [1, 2, 3, 4]
                        for r in schedule['updates']), 'Update/slot order differs')
        self.schedule = copy.deepcopy(schedule)
        self.schedule_sha256 = digest(schedule)
        self._state = {'format': 'codeformer-filter-fusion-sampler-state-v1',
                       'seed': SEED, 'schedule_sha256': self.schedule_sha256,
                       'committed_updates': 0, 'committed_microbatches': 0}

    def state(self):
        return copy.deepcopy(self._state)

    def preview(self):
        count = self._state['committed_updates']
        require(count < UPDATES, 'Finite schedule exhausted; no extension')
        before = self.state()
        after = {**before, 'committed_updates': count + 1, 'committed_microbatches': 4 * (count + 1)}
        return {**copy.deepcopy(self.schedule['updates'][count]),
                'state_before': before, 'state_after': after,
                'state_before_sha256': digest(before), 'state_after_sha256': digest(after)}

    def commit(self, preview):
        require(preview == self.preview(), 'Stale, modified or partial update cannot commit')
        self._state = copy.deepcopy(preview['state_after'])


def require_admission(admission, holds_sha256, dark_review_sha256, schedule):
    """Prospective worker must call this before permitting optimizer creation."""
    require(admission.get('format') == 'codeformer-filter-fusion-explicit-admission-v1'
            and admission.get('complete') is True and admission.get('root_approved') is True,
            'Explicit completed root pilot admission required')
    require(admission.get('candidate_holds_sha256') == holds_sha256 and
            admission.get('dark_input_review_sha256') == dark_review_sha256 and
            admission.get('schedule_sha256') == digest(schedule), 'Admission dependency mismatch')
    actual = {(s['role'], s['id']) for u in schedule['updates'] for s in u['slots']}
    expected = {(r['role'], r['id']) for r in admission['admitted_role_cases']}
    require(actual == expected and len(admission['admitted_role_cases']) == len(expected),
            'Explicit admitted role-case census differs')
    require(admission.get('held512_ledger_unchanged') is True and
            admission.get('historical_DEV_FINAL_native_excluded') is True and
            admission.get('automatic_launch_authorized') is False,
            'Preserved scope/manual-launch boundary missing')
    return True
