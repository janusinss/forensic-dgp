"""Pure metadata schedule: one reference's clear and four degraded views per batch."""
PROFILES = ['clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24']


def validate_profile_batches(cases, batches):
    for batch in batches:
        assert len(batch) == len(set(batch)) == 5
        assert all(isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(cases) for i in batch)
        rows = [cases[i] for i in batch]
        assert all(row['role'] == 'train' for row in rows), 'TRAIN only'
        assert len({row['source_person_or_reference'] for row in rows}) == 1, 'Clear control and degradations must share a reference'
        assert len({row['source'] for row in rows}) == 1
        assert [row['profile'] for row in rows] == PROFILES, 'Exactly one clear and four declared degraded views'


def make_profile_batches(cases, old_batches, first_epoch_updates, total_updates):
    assert len(cases) == first_epoch_updates * 5 and len(old_batches) == total_updates
    assert sorted(i for batch in old_batches[:first_epoch_updates] for i in batch) == list(range(len(cases)))
    grouped = {}
    for index, case in enumerate(cases):
        assert case['role'] == 'train'
        profiles = grouped.setdefault(case['source_person_or_reference'], {})
        assert case['profile'] not in profiles
        profiles[case['profile']] = index
    assert len(grouped) == first_epoch_updates and all(set(p) == set(PROFILES) for p in grouped.values())
    def first_occurrences(batches):
        seen, result = set(), []
        for batch in batches:
            for i in batch:
                ref = cases[i]['source_person_or_reference']
                if ref not in seen: seen.add(ref); result.append(ref)
        return result
    first = first_occurrences(old_batches[:first_epoch_updates])
    second = first_occurrences(old_batches[first_epoch_updates:])[:total_updates - first_epoch_updates]
    assert len(first) == first_epoch_updates and len(second) == total_updates - first_epoch_updates
    refs = first + second
    batches = [[grouped[ref][profile] for profile in PROFILES] for ref in refs]
    validate_profile_batches(cases, batches)
    assert len(batches) == total_updates
    assert sorted(i for batch in batches[:first_epoch_updates] for i in batch) == list(range(len(cases)))
    assert len(set(i for batch in batches[first_epoch_updates:] for i in batch)) == (total_updates - first_epoch_updates) * 5
    return batches, refs
