"""Frozen two-route finite comparison; transfer validation makes no neural calls."""
import hashlib
import json
from pathlib import Path

NAME = 'cctv_dgp_bank_comparison_v1_vm'
STEM = 'cctv-dgp-bank-comparison-v1'
CHECKPOINT = '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'
STATE = 'd89f4e52777b5260e94a3e533aa21a0204cbd79b578137881fc6a4c67f1bb0c3'
RECOGNIZER = '9ee66c2faefe0b82224c2cea6811073fbf70e8036808e293878109f078ea3963'
PRIOR = '05f5d33d79b32a3355cae3ede30e7ee06a90e56c60b1c2efe4ddd0d0e5a2959f'
MEAN = 'b27864667b6aa563059deeebb4d863d619bb9d8185e9b217bc92e7466b4d1574'
PROFILES = ['clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24']
NORMALIZERS = [.022182490369457405, .30685945008702126, .5253697948823282, .0017099954417771745]
ANCHORS = [.0006738122269695833, .007011396786714702]
THRESHOLDS = {'early_structure_gain': .01, 'final_structure_gain': .1,
    'MSE_regression_tolerance': 1e-12, 'SSIM_ArcFace_regression_tolerance': 1e-6,
    'brightness_fraction_maximum': .2}
BUDGETS = {'worker_seconds': 5400, 'external_seconds': 5430, 'kill_grace_seconds': 30,
    'preflight_seconds': 300, 'arm_fit_seconds': 600, 'snapshot_seconds': 1200,
    'export_seconds': 600, 'export_external_seconds': 630,
    'peak_vram_bytes': 20 * 1024**3, 'minimum_free_disk_bytes': 16 * 1024**3,
    'disk_reserve_bytes': 1024**3, 'return_uncompressed_bytes': 6 * 1024**3,
    'original_forward_calls': 1000, 'A_forward_calls': 1200, 'B_forward_calls': 1300,
    'recognizer_forward_calls': 3100, 'prior_fixture_forward_calls': 1, 'maximum_gradient_queries': 6,
    'maximum_backwards': 500, 'maximum_optimizer_updates': 100}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(data, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def schedule(cases):
    # Same 250 distinct TRAIN references for both routes: 125 from each source.
    sources = sorted({c['source'] for c in cases})
    assert len(sources) == 2
    buckets = [[i for i in range(0, len(cases), 5) if cases[i]['source'] == s] for s in sources]
    # SHA order is deterministic and independent of model outcomes.
    buckets = [sorted(b, key=lambda i: hashlib.sha256(('bank-comparison-v1|' + cases[i]['reference_id']).encode()).hexdigest())[:125]
               for b in buckets]
    order = [index for pair in zip(*buckets) for index in pair]
    return [order[i:i+5] for i in range(0, 250, 5)]


def validate(p):
    assert p['format'] == 'own256-conditioned-frozen-generative-bank-comparison-v1'
    assert p['original_checkpoint_sha256'] == CHECKPOINT and p['original_state'] == STATE
    assert p['recognizer_state'] == RECOGNIZER and p['standalone_prior_sha256'] == PRIOR
    assert p['mean_style_sha256'] == MEAN and p['budgets'] == BUDGETS
    assert p['normalizers'] == NORMALIZERS and p['active_anchor_normalizers'] == ANCHORS
    assert p['scientific_thresholds'] == THRESHOLDS
    assert p['reconstruction_weights'] == [.2, 1., 1., 1.] and p['regression_barrier_coefficient'] == 5
    assert p['active_clear_blur_anchor_coefficients'] == [1., 1.]
    assert p['optimizer'] == {'name': 'Adam', 'lr': .0001, 'betas': [.9, .999], 'eps': 1e-8,
        'weight_decay': 1e-5, 'gradient_clip_L2': 1., 'scheduler': 'constant LambdaLR'}
    assert p['routes'] == {'A': {'tensors': 15, 'elements': 609219}, 'B': {'tensors': 37, 'elements': 3364643}}
    assert len(p['cases']) == 3905 and len(p['references']) == 781
    assert all(c['role'] == 'train' for c in p['cases']) and all(r['role'] == 'train' for r in p['references'])
    assert len({c['id'] for c in p['cases']}) == 3905
    assert len({r['id'] for r in p['references']}) == 781
    for i in range(0, 3905, 5):
        cs = p['cases'][i:i+5]
        assert [c['profile'] for c in cs] == PROFILES
        assert len({c['source_person_or_reference'] for c in cs}) == 1
    assert p['schedule'] == schedule(p['cases']) and p['maximum_updates_per_route'] == 50
    assert p['full_epochs_per_route'] == 0 and p['fitting_reference_exposures_per_route'] == 250
    assert p['fitting_image_exposures_per_route'] == 1250
    assert len(p['preview_case_ids']) == 100 and set(p['preview_case_ids']) <= {c['id'] for c in p['cases']}
    assert len(p['native_development']) == 24
    assert all(c['role'] == 'development' and c['input_review']['input_review'] == 'usable' for c in p['native_development'])
    assert p['reserved_final_pixels'] == 0 and p['manual_tmux_required']
    assert p['independent_routes_continue_after_quality_stop'] and p['full_3905_TRAIN_raw_PNG_gates']
    assert p['B_disabled_ablation_TRAIN_cases'] == 100 and p['B_disabled_ablation_native_cases'] == 24
    assert p['initial_conditioning_gradients_expected_zero'] and p['B_post_update1_conditioning_gradient_check_required']
    for key in ['app_promotion', 'goal_complete', 'automatic_follow_on', 'failed_run_resume_allowed']:
        assert p[key] is False


def verify(root, pin):
    root = Path(root).resolve()
    assert sha(root / 'protocol.json') == pin
    p = read(root / 'protocol.json'); validate(p)
    for name, expected in p['assets_sha256'].items():
        f = root / name
        assert f.is_file() and not f.is_symlink() and f.resolve().is_relative_to(root) and sha(f) == expected, name
    allowed = {'protocol.json', 'trainer.log', 'trainer_exit_code.txt', 'supervisor_receipt.json',
               'export_manifest.json', 'export.log', 'export_exit_code.txt'}
    for f in root.rglob('*'):
        assert not f.is_symlink()
        if f.is_file():
            name = f.relative_to(root).as_posix()
            assert name in p['assets_sha256'] or name in allowed or name.startswith('outputs/'), name
    return p
