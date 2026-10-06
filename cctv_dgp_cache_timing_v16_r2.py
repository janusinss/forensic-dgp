"""Source/role-stratified cache projection; startup and warmups counted once."""
import collections
import math

TRAIN_PER_SOURCE = 10
VALIDATION_PER_SOURCE = 5
SAMPLE_REFERENCES = 30
METHOD = 'source-role-steady-reference-rate-startup-warmups-once-v16-r2'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def cache_order(plan):
    refs = plan['references']
    require(len(refs) == 885 and len({ref['id'] for ref in refs}) == 885, 'Invalid cache cohort')
    sources = sorted({ref['source'] for ref in refs})
    require(len(sources) == 2 and {ref['role'] for ref in refs} == {'train', 'validation'}, 'Invalid source/roles')
    sampled = []
    for role, count in [('train', TRAIN_PER_SOURCE), ('validation', VALIDATION_PER_SOURCE)]:
        for source in sources:
            group = sorted((ref for ref in refs if (ref['source'], ref['role']) == (source, role)),
                           key=lambda ref: ref['id'])
            require(len(group) >= count, 'Insufficient cache timing group')
            sampled.extend(group[:count])
    chosen = {ref['id'] for ref in sampled}
    require(len(chosen) == SAMPLE_REFERENCES, 'Cache timing sample differs')
    return sampled + [ref for ref in refs if ref['id'] not in chosen]


def projection(plan, measured, elapsed_seconds, initialization_seconds):
    d = plan['design']
    require(d['cache_timing_at_reference'] == SAMPLE_REFERENCES and len(measured) == SAMPLE_REFERENCES,
            'Require30 source/role timing references')
    require(all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x >= 0
                for x in [elapsed_seconds, initialization_seconds]), 'Invalid initialization/elapsed timing')
    ordered = cache_order(plan)
    groups = collections.defaultdict(list)
    totals = collections.Counter((ref['source'], ref['role']) for ref in plan['references'])
    seen = set()
    for row, ref in zip(measured, ordered[:SAMPLE_REFERENCES]):
        require(row['id'] == ref['id'] and row['source'] == ref['source'] and row['role'] == ref['role'] and
                row['cases'] == 5, 'Timing reference/source/role/profile coverage differs')
        seconds = row['seconds']
        require(isinstance(seconds, (int, float)) and not isinstance(seconds, bool) and math.isfinite(seconds) and seconds > 0,
                'Invalid reference timing')
        key = (ref['source'], ref['role'])
        require(row['warmup'] is (key not in seen), 'Warmup assignment differs')
        seen.add(key); groups[key].append(row)
    work = math.fsum(row['seconds'] for row in measured)
    require(initialization_seconds + work <= elapsed_seconds + 1e-5, 'Timing components exceed elapsed clock')
    strata = []
    for key in sorted(totals):
        rows = groups[key]
        require(len(rows) == (TRAIN_PER_SOURCE if key[1] == 'train' else VALIDATION_PER_SOURCE), 'Unbalanced timing sample')
        steady = [row['seconds'] for row in rows if not row['warmup']]
        rate = math.fsum(steady) / len(steady)
        remaining = totals[key] - len(rows)
        strata.append({'source': key[0], 'role': key[1], 'sample_references': len(rows),
                       'steady_references': len(steady), 'warmup_seconds': rows[0]['seconds'],
                       'steady_seconds_per_reference': rate, 'remaining_references': remaining,
                       'remaining_seconds': remaining * rate})
    remaining = math.fsum(group['remaining_seconds'] for group in strata)
    projected = elapsed_seconds + remaining * d['timing_safety_factor'] + 30
    return {'method': METHOD, 'references': SAMPLE_REFERENCES, 'seconds': elapsed_seconds,
            'initialization_seconds': initialization_seconds, 'sample_work_seconds': work,
            'sample_overhead_seconds': max(0., elapsed_seconds - initialization_seconds - work),
            'startup_scaled': False, 'warmups_scaled': False, 'strata': strata,
            'reference_measurements': measured, 'estimated_remaining_seconds': remaining,
            'timing_safety_factor': d['timing_safety_factor'], 'reserve_seconds': 30,
            'projected_seconds': projected, 'cap_seconds': d['cache_cap_seconds'],
            'passed': projected <= d['cache_cap_seconds']}


def verify_cache_timing(plan, recorded):
    rebuilt = projection(plan, recorded['reference_measurements'], recorded['seconds'], recorded['initialization_seconds'])
    # Same pinned inputs and deterministic arithmetic must survive JSON serialization exactly.
    require(recorded == rebuilt, 'Recorded cache projection differs from independent arithmetic')
    return rebuilt
