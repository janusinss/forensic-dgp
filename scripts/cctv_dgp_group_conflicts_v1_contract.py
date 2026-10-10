"""Prospective group-loss diagnostic contract; no neural imports or execution."""
import hashlib
import json
from pathlib import Path

NAME = 'cctv_dgp_group_conflicts_v1_vm'
STEM = 'cctv-dgp-group-conflicts-v1'
STATE = 'd89f4e52777b5260e94a3e533aa21a0204cbd79b578137881fc6a4c67f1bb0c3'
CHECKPOINT = '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'
PROFILES = ['clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24']
FRACTIONS = [1e-5, 1e-4, 1e-3]
TERMS = ['g' + format(i, '02d') for i in range(32)]
SCOPES = ['common']
BUDGETS = {'cache_seconds': 120, 'gradient_seconds': 480, 'solver_seconds': 120,
    'trial_seconds': 300, 'worker_seconds': 1500, 'external_seconds': 1530,
    'kill_grace_seconds': 30, 'export_seconds': 300, 'export_external_seconds': 330,
    'minimum_free_disk_bytes': 4 * 1024 ** 3, 'disk_reserve_bytes': 512 * 1024 ** 2,
    'return_uncompressed_bytes': int(1.5 * 1024 ** 3), 'peak_vram_bytes': 20 * 1024 ** 3,
    'original_forward_calls': 20, 'candidate_forward_calls': 90,
    'recognizer_forward_calls': 110}
PARITY = {'MSE_abs': 1e-12, 'SSIM_abs': 3e-5, 'landmark_structure_abs': 1e-10}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def definitions(cases):
    sources = sorted({case['source'] for case in cases})
    assert len(sources) == 2
    groups = []
    for source in sources:
        for profile in PROFILES:
            selected = [c['id'] for c in cases if c['source'] == source and c['profile'] == profile]
            assert len(selected) == 5
            for metric in ['MSE', 'SSIM_loss', 'ArcFace_loss']:
                groups.append({'source': source, 'profile': profile, 'metric': metric,
                               'case_ids': selected, 'cases': 5, 'batch_contributions': 5})
        selected = [c['id'] for c in cases if c['source'] == source and c['profile'] != 'clear']
        assert len(selected) == 20
        groups.append({'source': source, 'profile': 'degraded', 'metric': 'landmark_structure',
                       'case_ids': selected, 'cases': 20, 'batch_contributions': 5})
    assert len(groups) == 32
    return groups


def validate(protocol):
    assert protocol['format'] == 'own-DGP-source-profile-gradient-conflicts-v1'
    assert protocol['original_checkpoint_sha256'] == CHECKPOINT and protocol['original_state'] == STATE
    assert protocol['budgets'] == BUDGETS and protocol['loss_surrogate_parity'] == PARITY
    assert protocol['relative_displacement_fractions'] == FRACTIONS
    assert protocol['gradient_queries'] == 160 and protocol['maximum_trial_variants'] == 3
    assert protocol['optimizer_updates'] == protocol['epochs'] == protocol['committed_trajectory_updates'] == 0
    for key in ['native_or_DEV_or_final_used', 'model_qualification', 'automatic_follow_on', 'app_promotion', 'goal_complete']:
        assert protocol[key] is False
    assert protocol['manual_tmux_required'] and protocol['no_failed_recipe_resume']
    assert protocol['scientific_thresholds'] == {'early_structure_gain': .01, 'final_structure_gain': .1,
        'MSE_regression_tolerance': 1e-12, 'SSIM_ArcFace_regression_tolerance': 1e-6,
        'brightness_fraction_maximum': .2}
    assert protocol['CPU_replay_tolerances'] == {'raw_max_abs': 3e-6, 'PNG_byte_max': 1, 'embedding_max_abs': 5e-5}
    assert len(protocol['cases']) == 100 and len(protocol['references']) == 20
    by_id = {case['id']: case for case in protocol['cases']}
    assert len(by_id) == 100 and all(case['role'] == 'train' for case in by_id.values())
    assert len(protocol['cohorts']) == 2
    first, second = protocol['cohorts']
    assert len(first['case_ids']) == len(second['case_ids']) == 50
    assert set(first['case_ids']).isdisjoint(second['case_ids'])
    assert set(by_id) == set(first['case_ids']) | set(second['case_ids'])
    for cohort in protocol['cohorts']:
        cases = [by_id[cid] for cid in cohort['case_ids']]
        for begin in range(0, 50, 5):
            assert [case['profile'] for case in cases[begin:begin + 5]] == PROFILES
            assert len({case['source_person_or_reference'] for case in cases[begin:begin + 5]}) == 1
    assert protocol['group_losses'] == definitions([by_id[cid] for cid in first['case_ids']])
    assert protocol['minimum_normalized_descent_cosine'] == 1e-7
    assert protocol['direction_reconstruction_arithmetic_atol'] == 1e-12
    assert protocol['scale_weight_partition'] == 'decoder_control'
    assert len(protocol['parameter_layout']) == 158
    end = 0
    for entry in protocol['parameter_layout']:
        assert entry['start'] == end and entry['end'] - entry['start'] == entry['elements']
        assert entry['partition'] in ['feature_only', 'decoder_control']
        end = entry['end']
    assert end == 1996035


def verified_assets(root, pin):
    root = Path(root).resolve()
    assert sha(root / 'protocol.json') == pin
    protocol = read(root / 'protocol.json')
    validate(protocol)
    declared = set(protocol['assets_sha256']) | {'protocol.json'}
    for name, digest in protocol['assets_sha256'].items():
        path = root / name
        assert path.resolve().is_relative_to(root) and path.is_file() and not path.is_symlink()
        assert sha(path) == digest, name
    allowed_logs = {'diagnostic.log', 'diagnostic_exit_code.txt', 'supervisor_receipt.json', 'export_manifest.json'}
    for path in root.rglob('*'):
        assert not path.is_symlink()
        if path.is_file():
            name = path.relative_to(root).as_posix()
            assert name in declared or name.startswith('outputs/') or name in allowed_logs, name
    return protocol
