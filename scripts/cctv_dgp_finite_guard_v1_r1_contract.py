"""Frozen mechanics-study contract. Imports no neural libraries."""
import hashlib
import json
from pathlib import Path

NAME = 'cctv_dgp_finite_guard_v1_r1_vm'
STEM = 'cctv-dgp-finite-guard-v1-r1'
STATE = 'd89f4e52777b5260e94a3e533aa21a0204cbd79b578137881fc6a4c67f1bb0c3'
CHECKPOINT = '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'
PROFILES = ['clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24']
FRACTIONS = [2e-5, 1e-5, 5e-6]
BUDGETS = {'cache_seconds': 120, 'gradient_seconds_per_state': 240, 'solver_seconds_per_state': 120,
    'trial_seconds_per_state': 180, 'worker_seconds': 1800, 'external_seconds': 1830,
    'kill_grace_seconds': 30, 'export_seconds': 360, 'export_external_seconds': 390,
    'minimum_free_disk_bytes': 7 * 1024 ** 3, 'disk_reserve_bytes': 512 * 1024 ** 2,
    'return_uncompressed_bytes': 3 * 1024 ** 3, 'peak_vram_bytes': 20 * 1024 ** 3,
    'original_forward_calls': 20, 'candidate_forward_calls': 260, 'recognizer_forward_calls': 280}
PARITY = {'MSE_abs': 1e-12, 'SSIM_abs': 3e-5, 'landmark_structure_abs': 1e-10}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 ** 2), b''): digest.update(chunk)
    return digest.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def definitions(cases, cohorts):
    by_id = {case['id']: case for case in cases}
    groups = []
    for cohort in cohorts:
        chosen = [by_id[cid] for cid in cohort['case_ids']]
        sources = sorted({c['source'] for c in chosen})
        assert len(sources) == 2
        for source in sources:
            for profile in PROFILES:
                selected = [c['id'] for c in chosen if c['source'] == source and c['profile'] == profile]
                assert len(selected) == 5
                for metric in ['MSE', 'SSIM_loss', 'ArcFace_loss']:
                    groups.append({'cohort': cohort['name'], 'source': source, 'profile': profile,
                        'metric': metric, 'case_ids': selected, 'cases': 5, 'batch_contributions': 5})
            selected = [c['id'] for c in chosen if c['source'] == source and c['profile'] != 'clear']
            assert len(selected) == 20
            groups.append({'cohort': cohort['name'], 'source': source, 'profile': 'degraded',
                'metric': 'landmark_structure', 'case_ids': selected, 'cases': 20, 'batch_contributions': 5})
    assert len(groups) == 64
    return groups


def validate(p):
    assert p['format'] == 'own-DGP-finite-recertified-mechanics-v1'
    assert p['original_checkpoint_sha256'] == CHECKPOINT and p['original_state'] == STATE
    assert p['budgets'] == BUDGETS and p['loss_surrogate_parity'] == PARITY
    assert p['relative_displacement_fractions'] == FRACTIONS
    assert p['maximum_gradient_queries'] == 960 and p['maximum_trial_variants'] == 9
    assert p['maximum_accepted_parameter_changes'] == 3 and p['epochs'] == 0
    assert p['accepted_changes_are_actual_training'] and p['former_cross_cohort_used_for_fitting']
    assert p['previous_mean_shift_anchor'] == 'preceding accepted raw output; original anchor retained separately'
    for key in ['native_or_DEV_or_final_used', 'model_qualification', 'automatic_follow_on', 'app_promotion', 'goal_complete', 'resume_permitted']:
        assert p[key] is False
    assert p['manual_tmux_required'] and p['no_failed_recipe_resume']
    assert p['scientific_thresholds'] == {'early_structure_gain': .01, 'final_structure_gain': .1,
        'MSE_regression_tolerance': 1e-12, 'SSIM_ArcFace_regression_tolerance': 1e-6,
        'brightness_fraction_maximum': .2}
    assert p['CPU_replay_tolerances'] == {'raw_max_abs': 3e-6, 'PNG_byte_max': 1, 'embedding_max_abs': 5e-5}
    assert len(p['cases']) == 100 and len(p['references']) == 20
    by_id = {c['id']: c for c in p['cases']}
    assert len(by_id) == 100 and all(c['role'] == 'train' for c in by_id.values())
    assert len(p['cohorts']) == 2 and [c['id'] for c in p['cases']] == [cid for co in p['cohorts'] for cid in co['case_ids']]
    assert all(len(co['case_ids']) == 50 for co in p['cohorts'])
    assert set(p['cohorts'][0]['case_ids']).isdisjoint(p['cohorts'][1]['case_ids'])
    for co in p['cohorts']:
        cases = [by_id[cid] for cid in co['case_ids']]
        for begin in range(0, 50, 5):
            assert [c['profile'] for c in cases[begin:begin + 5]] == PROFILES
            assert len({c['source_person_or_reference'] for c in cases[begin:begin + 5]}) == 1
    assert p['group_losses'] == definitions(p['cases'], p['cohorts'])
    assert p['minimum_normalized_descent_cosine'] == 1e-7
    assert p['direction_reconstruction_arithmetic_atol'] == 1e-12
    assert p['scale_weight_partition'] == 'decoder_control'
    assert len(p['parameter_layout']) == 158
    end = 0
    for entry in p['parameter_layout']:
        assert entry['start'] == end and entry['end'] - entry['start'] == entry['elements']
        assert entry['partition'] in ['feature_only', 'decoder_control']
        end = entry['end']
    assert end == 1996035


def verified_assets(root, pin):
    root = Path(root).resolve()
    assert sha(root / 'protocol.json') == pin
    p = read(root / 'protocol.json'); validate(p)
    declared = set(p['assets_sha256']) | {'protocol.json'}
    for name, digest in p['assets_sha256'].items():
        path = root / name
        assert path.resolve().is_relative_to(root) and path.is_file() and not path.is_symlink()
        assert sha(path) == digest, name
    logs = {'diagnostic.log', 'diagnostic_exit_code.txt', 'supervisor_receipt.json', 'export_manifest.json'}
    for path in root.rglob('*'):
        assert not path.is_symlink()
        if path.is_file():
            name = path.relative_to(root).as_posix()
            assert name in declared or name.startswith('outputs/') or name in logs, name
    return p
