"""Prospective finite factorial calibration; pure transfer contract."""
import hashlib
import json
from pathlib import Path

NAME = 'cctv_dgp_multiscale_calibration_vm_v1'
STEM = 'cctv-dgp-multiscale-calibration-v1'
STATE = 'd89f4e52777b5260e94a3e533aa21a0204cbd79b578137881fc6a4c67f1bb0c3'
CHECKPOINT = '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'
INITIAL = '0ca17ba45611fe472503bd13a7f3fb925b7408806631b2967fee150861a1593e'
RECOGNIZER = '9ee66c2faefe0b82224c2cea6811073fbf70e8036808e293878109f078ea3963'
PROFILES = ['clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24']
WEIGHTS = [.2, 1., 1., 1.]
NORMALIZERS = [.022182490369457405, .30685945008702126, .5253697948823282, .0017099954417771745]
SIGNALS = ['MSE_all', 'SSIM_loss_all', 'ArcFace_loss_all', 'HF_degraded',
           'MSE_clear', 'SSIM_loss_clear', 'ArcFace_loss_clear', 'MSE_blur', 'SSIM_loss_blur',
           'ArcFace_loss_blur', 'HF_blur', 'observed_mean_R', 'observed_mean_G', 'observed_mean_B']
BUDGETS = {'worker_seconds': 1800, 'external_seconds': 1830, 'kill_grace_seconds': 30,
    'gradient_seconds': 600, 'arm_fit_seconds': 120, 'snapshot_seconds': 120,
    'export_seconds': 600, 'export_external_seconds': 630,
    'peak_vram_bytes': 20 * 1024**3, 'minimum_free_disk_bytes': 7 * 1024**3,
    'disk_reserve_bytes': 1024**3, 'return_uncompressed_bytes': 2684354560,
    'original_forward_calls': 44, 'candidate_forward_calls': 712, 'recognizer_forward_calls': 400,
    'maximum_gradient_queries': 280, 'maximum_backwards': 120, 'maximum_optimizer_updates': 12}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, d):
    with Path(p).open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(d, indent=2, allow_nan=False) + '\n')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def arms():
    return [{'id': f'{part}_lr{rate:g}_pool{pool}', 'partition': part, 'lr': rate, 'pool': pool,
             'optimizer_updates': 1, 'accumulated_reference_batches': 10, 'fit_exposures': 50}
            for part in ['deep3', 'decoder15'] for rate in [1e-5, 1e-4, 1e-3] for pool in [0, 1]]


def validate(p):
    assert p['format'] == 'own-DGP-multiscale-first-step-calibration-v1'
    assert p['original_checkpoint_sha256'] == CHECKPOINT and p['original_state'] == STATE
    assert p['initial_candidate_state'] == INITIAL and p['recognizer_state'] == RECOGNIZER
    assert p['budgets'] == BUDGETS and p['arms'] == arms()
    assert p['reconstruction_weights'] == WEIGHTS and p['normalizers'] == NORMALIZERS
    assert p['signals'] == SIGNALS and p['regression_barrier_coefficient'] == 5
    assert p['optimizer'] == {'name': 'Adam', 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 1e-5, 'gradient_clip_L2': 1.}
    assert p['partitions'] == {'deep3': {'tensors': 3, 'elements': 147456}, 'decoder15': {'tensors': 15, 'elements': 609219}}
    cs = p['cases']; assert len(cs) == 100 and len({c['id'] for c in cs}) == 100
    assert len(p['references']) == 20 and all(r['role'] == 'train' for r in p['references'])
    assert all(c['role'] == 'train' for c in cs)
    for i in range(0, 100, 5):
        batch = cs[i:i+5]
        assert [c['profile'] for c in batch] == PROFILES
        assert len({c['source_person_or_reference'] for c in batch}) == 1
    assert p['pools'] == [list(range(0, 50, 5)), list(range(50, 100, 5))]
    for indices in p['pools']:
        sources = [cs[i]['source'] for i in indices]
        assert len(set(sources)) == 2 and all(sources.count(s) == 5 for s in set(sources))
    assert len(p['native_development']) == 24
    assert all(c['role'] == 'development' and c['input_review']['input_review'] == 'usable' for c in p['native_development'])
    assert len({c['id'] for c in p['native_development']}) == 24
    assert p['scientific_thresholds'] == {'early_structure_gain': .01, 'final_structure_gain': .1,
        'MSE_regression_tolerance': 1e-12, 'SSIM_ArcFace_regression_tolerance': 1e-6, 'brightness_fraction_maximum': .2}
    assert p['complete_epochs'] == 0 and p['original_epoch_references'] == 781
    assert p['gradient_shape'] == [20, 14, 609219] and p['gradient_dtype'] == 'float32'
    assert p['one_step_does_not_qualify_model'] and p['manual_tmux_required']
    assert p['paired_TRAIN_and_unpaired_native_reported_separately']
    assert p['all_seven_completion_families_still_required']
    for k in ['app_promotion', 'goal_complete', 'failed_state_resume_allowed', 'automatic_follow_on', 'reserved_final_pixels']:
        assert p[k] is False or (k == 'reserved_final_pixels' and p[k] == 0)


def verify(root, pin):
    root = Path(root).resolve(); assert sha(root / 'protocol.json') == pin
    p = read(root / 'protocol.json'); validate(p)
    for n, d in p['assets_sha256'].items():
        f = root / n
        assert f.is_file() and not f.is_symlink() and f.resolve().is_relative_to(root) and sha(f) == d, n
    allowed = {'protocol.json', 'trainer.log', 'trainer_exit_code.txt', 'supervisor_receipt.json',
               'export_manifest.json', 'export.log', 'export_exit_code.txt'}
    for f in root.rglob('*'):
        assert not f.is_symlink()
        if f.is_file():
            n = f.relative_to(root).as_posix()
            assert n in p['assets_sha256'] or n in allowed or n.startswith('outputs/'), n
    return p
