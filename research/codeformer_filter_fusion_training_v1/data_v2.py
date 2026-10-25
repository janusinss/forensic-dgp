"""Prospective151-update sampler after two documented whole-source scope holds.

The earlier157-update sampler and all historical labels remain immutable. This
module performs no neural work, optimizer work, pixel generation or launch.
"""
from collections import Counter
import copy

from .data import PADDING_ID, ROLES, SEED, digest, require

UPDATES = 151
HELD_SOURCES = ('tr_ffhq_21374', 'tr_ffhq_45780')
PROFILE_COUNTS = {'gaussian_2p4': 55, 'anisotropic_h5_v0p7': 55,
                  'anisotropic_h0p7_v5': 51, 'repeated_rows_4_blend0p75': 55,
                  'global_mosaic_8': 11, 'local_gaussian_3p0': 50,
                  'local_mosaic_8': 24}


def revised_schedule(old, linear_rows):
    """Keep frozen order as a subsequence; no fresh shuffle/quality selection."""
    require(old.get('max_updates') == 157 and old.get('seed') == SEED and
            len(old['updates']) == 157, 'Exact historical157 schedule required')
    filters = [copy.deepcopy(s) for u in old['updates'] for s in u['slots'][:2]
               if s['reference_id'] not in HELD_SOURCES]
    unique = [s for s in filters if not s['padding_duplicate']]
    require(len(filters) == 302 and len(unique) == len({s['id'] for s in filters}) == 301 and
            Counter(s['profile'] for s in unique) == PROFILE_COUNTS and
            [s['id'] for s in filters if s['padding_duplicate']] == [PADDING_ID] and
            Counter(s['id'] for s in filters)[PADDING_ID] == 2,
            'Exact301 unique filter cases plus original fixed padding required')
    anchors = [r for r in old['anchor_source_order'] if r not in HELD_SOURCES]
    require(len(anchors) == len(set(anchors)) == 55, 'Exact55 original anchor sources required')
    dark_by_ref = {r['reference_id']: r for r in linear_rows if r['reference_id'] not in HELD_SOURCES}
    require(set(anchors) == set(dark_by_ref), 'All55 same-reference linear-dark anchors required')
    updates = []
    for index in range(UPDATES):
        slots = []
        for number, row in enumerate(filters[2 * index:2 * index + 2], 1):
            row.update(slot=number, role=ROLES[number - 1])
            slots.append(row)
        rid = anchors[index % len(anchors)]
        for number, role, identifier, profile in [
                (3, 'clear', rid + '__clear', 'clear'),
                (4, 'dark', dark_by_ref[rid]['id'], dark_by_ref[rid]['profile'])]:
            slots.append({'slot': number, 'role': role, 'id': identifier,
                          'reference_id': rid, 'profile': profile, 'padding_duplicate': False})
        updates.append({'update': index + 1, 'slots': slots})
    return {'format': 'codeformer-filter-fusion-finite-schedule-v2', 'seed': SEED,
            'max_updates': UPDATES, 'slot_roles': list(ROLES), 'updates': updates,
            'filter_slots': 302, 'unique_filter_candidates': 301, 'clear_slots': 151,
            'dark_slots': 151, 'padding_duplicate_count': 1, 'padding_duplicate_id': PADDING_ID,
            'anchor_source_order': anchors, 'whole_source_scope_holds': list(HELD_SOURCES),
            'previous157_schedule_canonical_sha256': digest(old),
            'filter_order_policy': 'original frozen314 sequence minus12 held-source cases; original fixed padding retained',
            'clear_and_dark_same_reference_each_update': True,
            'automatic_training_or_resume_authorized': False}


class FiniteSampler:
    """Pure preview; commit only after the complete four-slot optimizer update."""
    def __init__(self, schedule):
        require(schedule.get('format') == 'codeformer-filter-fusion-finite-schedule-v2'
                and schedule.get('seed') == SEED and schedule.get('max_updates') == UPDATES
                and len(schedule['updates']) == UPDATES, 'Exact151 schedule required')
        require([r['update'] for r in schedule['updates']] == list(range(1, UPDATES + 1))
                and all([s['role'] for s in r['slots']] == list(ROLES) and
                        [s['slot'] for s in r['slots']] == [1, 2, 3, 4]
                        for r in schedule['updates']), 'Update/slot order differs')
        filters = [s for u in schedule['updates'] for s in u['slots'][:2]]
        unique = [s for s in filters if not s['padding_duplicate']]
        counts = Counter(s['id'] for s in filters)
        require(len(filters) == 302 and len(unique) == len(counts) == 301 and
                Counter(s['profile'] for s in unique) == PROFILE_COUNTS and
                [s['id'] for s in filters if s['padding_duplicate']] == [PADDING_ID] and
                {k: v for k, v in counts.items() if v != 1} == {PADDING_ID: 2} and
                not any(s['reference_id'] in HELD_SOURCES for u in schedule['updates'] for s in u['slots']),
                'Held source, padding or incomplete unique coverage in schedule')
        require(all(u['slots'][2]['reference_id'] == u['slots'][3]['reference_id']
                    for u in schedule['updates']), 'Samephoto clear/dark pairing required')
        self.schedule = copy.deepcopy(schedule)
        self.schedule_sha256 = digest(schedule)
        self._state = {'format': 'codeformer-filter-fusion-sampler-state-v2',
                       'seed': SEED, 'schedule_sha256': self.schedule_sha256,
                       'committed_updates': 0, 'committed_microbatches': 0}

    def state(self):
        return copy.deepcopy(self._state)

    def preview(self):
        count = self._state['committed_updates']
        require(count < UPDATES, 'Finite151 schedule exhausted; no extension')
        before = self.state()
        after = {**before, 'committed_updates': count + 1, 'committed_microbatches': 4 * (count + 1)}
        return {**copy.deepcopy(self.schedule['updates'][count]), 'state_before': before,
                'state_after': after, 'state_before_sha256': digest(before),
                'state_after_sha256': digest(after)}

    def commit(self, preview):
        require(preview == self.preview(), 'Stale, modified or partial update cannot commit')
        self._state = copy.deepcopy(preview['state_after'])


def require_admission(admission, holds_sha256, dark_review_sha256, schedule):
    require(admission.get('format') == 'codeformer-filter-fusion-explicit-admission-v2'
            and admission.get('complete') is True and admission.get('root_approved') is True,
            'Explicit completed root pilot admission required')
    require(admission.get('candidate_holds_sha256') == holds_sha256 and
            admission.get('dark_input_review_sha256') == dark_review_sha256 and
            admission.get('schedule_sha256') == digest(schedule), 'Admission dependency mismatch')
    actual = {(s['role'], s['id']) for u in schedule['updates'] for s in u['slots']}
    expected = {(r['role'], r['id']) for r in admission['admitted_role_cases']}
    require(actual == expected and len(admission['admitted_role_cases']) == len(expected),
            'Explicit admitted role-case census differs')
    require(admission.get('held_sources') == list(HELD_SOURCES) and
            admission.get('held512_ledger_unchanged') is True and
            admission.get('historical_DEV_FINAL_native_excluded') is True and
            admission.get('automatic_launch_authorized') is False,
            'Preserved scope/manual-launch boundary missing')
    FiniteSampler(schedule)
    return True
