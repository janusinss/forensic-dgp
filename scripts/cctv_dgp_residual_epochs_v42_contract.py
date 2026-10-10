"""Pure finite study contract. Original scientific gates are imported unchanged."""
from pathlib import Path
import tarfile

from frozen_capacity_contract import (sha, read, write, PROFILES, METRICS, erode,
    feature_support, exported_pixel_metrics, detail_metric, groups, capacity)

NAME = 'cctv_dgp_residual_epochs_vm_v42'
STEM = 'cctv-dgp-residual-epochs-v42'
UPDATES = 3905
SNAPSHOTS = [0, 50, 781, 1562, 3905]
BUDGETS = {'cache_seconds': 900, 'fit_seconds': 6300, 'worker_seconds': 7200,
           'external_seconds': 7230, 'kill_grace_seconds': 30,
           'export_seconds': 900, 'export_external_seconds': 930,
           'minimum_free_disk_bytes': 8 * 1024**3,
           'disk_reserve_bytes': 512 * 1024**2,
           'export_uncompressed_bytes': int(3.5 * 1024**3),
           'peak_vram_bytes': 20 * 1024**3}
LOSS_WEIGHTS = {'observed_correction': 1., 'landmark_correction': 1.,
                'RGB_pyramid_correction': .25, 'clear_correction_multiplier': 4.,
                'degraded_correction_multiplier': 1.25,
                'identity_absolute': .1, 'identity_regression': 5.,
                'pixel_regression': 2., 'SSIM_regression': 5.}


def validate_schedule(cases, schedule):
    assert len(cases) == len({c['id'] for c in cases}) == 3905
    assert all(c['role'] == 'train' and c['profile'] in PROFILES for c in cases)
    assert schedule['seed'] == 420042 and schedule['epochs'] == 5
    batches = schedule['batches']
    assert len(batches) == UPDATES
    for ids in batches:
        assert len(ids) == len(set(ids)) == 5
        assert all(type(i) is int and 0 <= i < 3905 for i in ids)
        selected = [cases[i] for i in ids]
        assert len({c['source_person_or_reference'] for c in selected}) == 1
        assert [c['profile'] for c in selected] == PROFILES
    for epoch in range(5):
        assert sorted(i for b in batches[epoch*781:(epoch+1)*781] for i in b) == list(range(3905))
    assert len({tuple(tuple(b) for b in batches[e*781:(e+1)*781]) for e in range(5)}) == 5


def validate_protocol(p):
    assert p['format'] == 'own-DGP-direct-residual-supervision-finite-epochs-v42'
    assert p['updates'] == UPDATES and p['snapshots'] == SNAPSHOTS
    assert p['budgets'] == BUDGETS and p['loss_weights'] == LOSS_WEIGHTS
    assert p['decoder_parameters'] == 17952 and p['decoder_tensors'] == 57
    assert len(p['references']) == 781 and len(p['preview_case_ids']) == 50
    assert p['early_gain'] == .01 and p['final_gain'] == .1
    assert p['original_checkpoint_sha256'] == '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'
    assert p['optimizer'] == {'type': 'AdamW', 'learning_rate': .0003,
         'weight_decay': .01, 'gradient_clip_norm': 1., 'AMP': False, 'EMA': False}
    assert p['scheduler'] == {'type': 'MultiStepLR', 'milestones_updates': [1562], 'gamma': .3}
    assert p['normalizer_floor'] == .001 and p['timing_safety_factor'] == 1.25
    assert p['native_or_reserved_used'] is False and p['automatic_follow_on'] is False
    assert p['app_promotion'] is False and p['goal_complete'] is False


def verified_assets(root, p, pin):
    root = Path(root).resolve()
    assert sha(root/'protocol.json') == pin
    validate_protocol(p)
    validate_schedule(p['cases'], read(root/'schedule.json'))
    for name, digest in p['assets_sha256'].items():
        path = root/name
        assert path.resolve().is_relative_to(root) and path.is_file() and not path.is_symlink()
        assert sha(path) == digest, name


def safe_return_members(members, p):
    result = []; seen = set(); total = 0
    for m in members:
        assert m.isfile() and not m.issym() and not m.islnk()
        parts = m.name.split('/')
        assert len(parts) >= 2 and parts[0] == NAME+'_return'
        name = '/'.join(parts[1:])
        assert name and all(part not in ['', '.', '..'] for part in parts)
        assert '\\' not in name and ':' not in name and name not in seen
        assert name not in p['assets_sha256']
        assert name == 'protocol.json' or name == 'export_manifest.json' or \
            name == 'supervisor_receipt.json' or name in ['trainer.log', 'trainer_exit_code.txt'] or \
            name.startswith('outputs/')
        total += m.size
        assert total <= BUDGETS['export_uncompressed_bytes'] + 32 * 1024**2
        seen.add(name); result.append((m, name))
    assert len(seen) <= 100000
    assert {'protocol.json', 'export_manifest.json', 'supervisor_receipt.json',
            'trainer.log', 'trainer_exit_code.txt'} <= seen
    return result, total
