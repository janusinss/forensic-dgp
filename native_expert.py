"""Bounded native-coverage expert contracts; no model/optimizer on import."""
import random
import sys

import torch

MODEL_SHA = 'a51f20e8fa12cee7dec574195debc5ada9b8cfeeb289b12bad11a6d39a9acf25'
OPTIMIZER_SHA = '8075fde6c2d383d1923a6e69d5798882b0620ca435d20a18640eb49a34243935'
PARENT_SHA = 'c2d4cd180226413af63bcfc2a3e17461cfa95f87aff9a8998b893fc3afc42e93'
NATIVE_COVERING = (171, 216, 348, 374)
EPOCHS, STEPS, SEED = 6, 56, 42


def require(condition, message):
    if not condition: raise ValueError(message)


def require_vm_gpu():
    if sys.platform != 'linux' or not torch.cuda.is_available():
        raise RuntimeError('Actual training requires the Linux VM CUDA device; no local training')


def fixed_schedule(real_rows, fixture_rows):
    require(len(real_rows) == 83 and all(r.get('split') == 'train' for r in real_rows),
            'Require the exact supported training split, never held-out rows')
    positive = [i for i, r in enumerate(real_rows) if r.get('kind') == 'covered']
    clear = [i for i, r in enumerate(real_rows) if r.get('kind') == 'uncovered']
    require(len(positive) == 51 and len(clear) == 32, 'Supported training strata differ')
    require(len(fixture_rows) == 280 and [r['case_id'] for r in fixture_rows] == list(range(280)),
            'Require the existing ordered280-case fixture manifest')
    fixture_clear = [r['case_id'] for r in fixture_rows if r['style'] == 'clear']
    fixture_positive = [r['case_id'] for r in fixture_rows if r['style'] != 'clear']
    require(len(fixture_positive) == 224 and len(fixture_clear) == 56, 'Fixture strata differ')
    rng = random.Random(SEED); schedule = []
    for _ in range(EPOCHS):
        pools = [list(p) for p in (positive, clear, fixture_positive, fixture_clear)]
        for pool in pools: rng.shuffle(pool)
        a, b, c, d = pools
        schedule.append([{'real': [a[(2*i) % len(a)], a[(2*i+1) % len(a)], b[i % len(b)]],
                          'fixture': c[4*i:4*i+4] + [d[i]]} for i in range(STEPS)])
    return schedule


def counters(epoch):
    require(type(epoch) is int and 1 <= epoch <= EPOCHS, 'Epoch outside the fixed budget')
    updates = epoch * STEPS
    return {'epoch': 42 + epoch, 'optimizer_updates': 882 + updates,
            'additional_epoch': 32 + epoch, 'fresh_optimizer_updates': 672 + updates,
            'experiment_epoch': epoch, 'experiment_updates': updates}


def validate_source(payload):
    require(payload.get('format') == 'dgp-face-occlusion-adapter-v1'
            and payload.get('target') == 'covered_region_is_one'
            and payload.get('architecture') == 'resnet18-unet'
            and payload.get('smp_version') == '0.5.0'
            and payload.get('initialization') == 'pretrained'
            and payload.get('reflection_arm') == 'reflective', 'Wrong fixed source architecture/arm')
    for key, expected in {'epoch': 42, 'optimizer_updates': 882, 'additional_epoch': 32,
                          'fresh_optimizer_updates': 672, 'experiment_updates': 252}.items():
        require(type(payload.get(key)) is int and payload[key] == expected, 'Source counter differs: ' + key)


def validate_source_optimizer(payload, groups, model_sha):
    require(set(payload) == {'state', 'model_sha256', 'experiment_updates',
                            'optimizer_state_step', 'cumulative_model_updates'}
            and payload['model_sha256'] == model_sha, 'Source42 optimizer schema/model binding differs')
    for key, expected in {'experiment_updates': 252, 'optimizer_state_step': 672,
                          'cumulative_model_updates': 882}.items():
        require(type(payload[key]) is int and payload[key] == expected, 'Optimizer counter differs: ' + key)
    state = payload['state']
    require(set(state) == {'state', 'param_groups'} and len(groups) == len(state['param_groups']) == 2,
            'Require encoder and decoder/head groups')
    parameters, ids = [], []
    for actual, expected, lr in zip(state['param_groups'], groups, (1e-5, 1e-4)):
        require(expected['lr'] == actual.get('lr') == lr and actual.get('betas') == (.9, .999)
                and actual.get('eps') == 1e-8 and actual.get('weight_decay') == 1e-4
                and actual.get('decoupled_weight_decay') is True
                and all(actual.get(k) is False for k in ('amsgrad', 'maximize', 'capturable', 'differentiable'))
                and all(actual.get(k) is None for k in ('foreach', 'fused'))
                and len(actual['params']) == len(expected['params']), 'Optimizer hyperparameters/group shape differ')
        ids.extend(actual['params']); parameters.extend(expected['params'])
    require(all(type(i) is int for i in ids) and ids == list(range(len(parameters)))
            and set(state['state']) == set(ids), 'Optimizer parameter IDs differ')
    for i, parameter in zip(ids, parameters):
        moment = state['state'][i]
        require(set(moment) == {'step', 'exp_avg', 'exp_avg_sq'}
                and isinstance(moment['step'], torch.Tensor) and moment['step'].numel() == 1
                and torch.isfinite(moment['step']).all() and moment['step'].item() == 672,
                'Source moment step differs')
        require(all(isinstance(moment[k], torch.Tensor) and moment[k].shape == parameter.shape
                    and moment[k].dtype == parameter.dtype and torch.isfinite(moment[k]).all()
                    for k in ('exp_avg', 'exp_avg_sq')) and (moment['exp_avg_sq'] >= 0).all(),
                'Invalid source moments')
    return {'parameter_states': len(parameters), 'parameter_elements': sum(p.numel() for p in parameters),
            'restored_step': 672, 'cumulative_model_updates': 882}


def expert_loss(real_logits, real_mask, real_valid, fixture_logits, fixture_mask, fixture_valid):
    from reflection_coverage import supported_segmentation_loss
    require(real_logits.shape[0] == 3 and fixture_logits.shape[0] == 5, 'Fixed3-real/5-fixture batch required')
    real = supported_segmentation_loss(real_logits, real_mask, real_valid, .25, .1)
    fixture = supported_segmentation_loss(fixture_logits, fixture_mask, fixture_valid, .25, .1)
    return .5 * real + .5 * fixture, real, fixture


def fit_decision(before, after):
    def retained(group):
        a, b = after[group], before[group]
        return a['iou'] >= b['iou'] and all(a[k] <= b[k] for k in
            ('missed_fraction', 'visible_false_positive', 'empty_mask_cases', 'negative_false_positive_cases'))
    expected = {str(i) for i in NATIVE_COVERING}
    require(set(before['native_cases']) == set(after['native_cases']) == expected, 'All four native cases required')
    checks = {
        'native_coverage_without_added_fp': after['native_real']['iou'] > before['native_real']['iou']
            and after['native_real']['fp'] <= before['native_real']['fp']
            and all(after['native_cases'][i]['tp'] > before['native_cases'][i]['tp'] for i in expected),
        'original_real_fit_retained': retained('original_real'),
        'reflection_fixture_fit_retained': retained('reflection'),
        'old_and_new_lens_recovery': after['lens']['new'] > before['lens']['new']
            and after['lens']['old'] >= before['lens']['old'],
        'clear_controls_remain_empty': after['all_real']['negative_false_positive_cases'] == 0
            and after['reflection']['negative_false_positive_cases'] == 0,
    }
    return {'checks': checks, 'passes': all(checks.values()),
            'training_fit_only': True, 'development_eligible': False, 'promoted': False}
