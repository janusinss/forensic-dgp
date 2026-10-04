"""Finite train-cohort V14 recipe and no-neural verification helpers."""
import random
from pathlib import Path

from cctv_dgp_face_code_fit_v12 import (require, sha, read, write, safe,
                                       validate_cohort, require_vm, SOURCES)

FORMAT = 'cctv-dgp-direct-code-statistics-capacity-v14'
PLAN = 'direct_code_protocol_v14.json'
PARENT_PROTOCOL = '06f056ea4b1b04e6d8c631e5e78df129345c78279230c7946f6b5deb2e226846'
PARENT_RESULTS = '95fb1993e93aba1972ae89d024df1ad8d50574d1fefd7eacf5dc977b6c9c4796'
CONTROL_PROTOCOL = '093cf67b630de4ed74d0c2fa62fc928a69c2c2464e5b9ecfd23d22a8a0a603ac'
CONTROL_RESULTS = '367ffef001155a4a551baeb1a03576a515f78f75f3366218a13f97a7788e5be6'
DESIGN = {'seed': 20261004, 'epochs': 40, 'updates': 1000, 'exposures': 2000,
          'batch_size': 2, 'optimizer': 'Adam', 'learning_rate': .001,
          'betas': [.9, .999], 'lambda_code_ce': 1., 'lambda_mean_mse': 1.,
          'lambda_logstd_mse': 1., 'gradient_clip_norm': 1., 'snapshots': [0, 300, 1000],
          'statistics_modes': ['observed', 'none', 'predicted'], 'fidelity': 0.,
          'trainable_parameters': 2619808, 'use_amp': False,
          'trainer_cap_seconds': 600, 'supervisor_cap_seconds': 900,
          'peak_vram_cap_bytes': 20 * 1024**3, 'timing_update': 25,
          'selection': 'None: diagnostic adapters only; no best.pth or app promotion.'}


def schedule(p):
    validate_cohort(p)
    steps = []
    for epoch in range(1, DESIGN['epochs'] + 1):
        rng = random.Random(DESIGN['seed'] + epoch)
        lists = []
        for source in SOURCES:
            ids = sorted(c['id'] for c in p['cases'] if c['source'] == source)
            rng.shuffle(ids)
            lists.append(ids)
        steps.extend({'epoch': epoch, 'case_ids': list(pair)} for pair in zip(*lists))
    return steps


def verify(root, parent, expected_sha, *, include_weights=True):
    root, parent = Path(root), Path(parent)
    require(sha(root / PLAN) == expected_sha ==
            (root / 'direct_code_protocol_v14.sha256').read_text().strip(), 'V14 protocol differs')
    p = read(root / PLAN)
    require(p['format'] == FORMAT and p['design'] == DESIGN and
            p['parent_protocol_sha256'] == PARENT_PROTOCOL and
            p['parent_results_sha256'] == PARENT_RESULTS and
            p['control_protocol_sha256'] == CONTROL_PROTOCOL and
            p['control_results_sha256'] == CONTROL_RESULTS, 'V14 design/lineage differs')
    validate_cohort(p)
    require(read(root / 'schedule_v14.json')['steps'] == schedule(p), 'V14 schedule differs')
    require(sha(parent / 'face_code_fit_protocol_v12.json') == PARENT_PROTOCOL,
            'Parent protocol differs')
    parent_plan = read(parent / 'face_code_fit_protocol_v12.json')
    require(parent_plan['assets_sha256'] == p['parent_assets_sha256'] and
            p['references'] == parent_plan['references'] and p['cases'] == parent_plan['cases'],
            'Parent assets/cohort differ')
    for name, pin in p['sources_sha256'].items():
        require(sha(safe(root, name)) == pin, 'V14 execution source differs: ' + name)
    require(sha(root / 'schedule_v14.json') == p['schedule_sha256'], 'Schedule fingerprint differs')
    for name, pin in p['parent_assets_sha256'].items():
        if not include_weights and name in parent_plan['weights'].values():
            continue
        require(sha(safe(parent, name)) == pin, 'Parent source/asset differs: ' + name)
    return p
