"""Synthetic metadata/PNG/CPU-tensor fixtures; no model or optimizer instance."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import shutil
import unittest
import uuid

import numpy as np
from PIL import Image
import torch

from research.codeformer_adaptation_v2 import data, worker
from research.codeformer_adaptation_v2 import return_trajectory as audit


def protocol():
    value = deepcopy(worker.DEFAULT_PROTOCOL)
    value['files_sha256'] = {'references.json': 'f' * 64}
    return value


def references():
    return [{'id': 'tr_ffhq_%05d' % i, 'historical_role': 'train',
             'source_path': 'data/sources/tr_ffhq_%05d.png' % i,
             'source_sha256': hashlib.sha256(('owned source%d' % i).encode()).hexdigest(),
             'proposed_512_rgb_sha256': 'c' * 64} for i in range(391)]


def history_fixture(count=2):
    p, refs, rows = protocol(), references(), []
    by_id = {row['id']: row for row in refs}
    sampler = data.TrainSampler(sorted(by_id))
    for update in range(1, count + 1):
        selection = sampler.preview_update()
        samples = []
        for slot in range(1, 5):
            identifier = selection.id_a if slot <= 2 else selection.id_b
            role, member = ('A' if slot <= 2 else 'B'), ('normal' if slot % 2 else 'dark')
            params = data.camera_parameters(identifier, role, update)
            params.update(member=member, slot=slot, behavior_anchor=(slot == 3),
                pair_id='owned_pair%d_%s' % (update, role),
                exposure_linear=1. if member == 'normal' else params['dark_exposure_linear'],
                noise_fields={'owned synthetic pair field': role})
            record = {'reference_id': identifier, 'role': 'train', 'source_path': by_id[identifier]['source_path'],
                'source_sha256': by_id[identifier]['source_sha256'], 'reference_manifest_sha256': 'f' * 64,
                'clean512_rgb8_sha256': 'c' * 64, 'clean512_float32_sha256': 'd' * 64,
                'update': update, 'slot': slot, 'pair_role': role, 'member': member,
                'pair_id': params['pair_id'], 'parameters': params,
                'profile': params['family'] + '_' + member}
            sample = {'id': identifier, 'slot': slot, 'profile': record['profile'],
                'image_eligible': params['image_eligible'], 'pair_record': record,
                'behavior_anchor': {'synthetic': True} if slot == 3 else None}
            terms = {'token_ce': .2, 'latent_mse': .3, 'coarse_rgb_l1': .1}
            if sample['image_eligible']:
                terms.update(pixel_l1_rgb=.1, lpips_vgg=.2)
            if slot == 3:
                terms.update(behavior_rgb_l1=.1, behavior_token_kl=.2)
            sample['terms'] = terms
            sample['weighted_terms'] = {name: value * p['training']['loss_weights'][name] for name, value in terms.items()}
            sample['total_before_accumulation_division'] = sum(sample['weighted_terms'].values())
            samples.append(sample)
        rows.append({'update': update, 'microbatches': samples, 'TRAIN_selection': selection.to_record(),
            'group_lrs': {name: audit.lr_at(update, p['training'][base], p['training'])
                          for name, base in [('encoder_token', 'encoder_token_lr'), ('fusion', 'fusion_lr')]},
            'optimization_seconds': update * 2., 'optimizer_committed': True})
        sampler.commit(selection)
    return p, refs, rows


def tiny_checkpoint(update=1):
    p = protocol()
    original = {name + '.weight': torch.zeros(2) for name in audit.TRAINABLE + audit.FROZEN}
    model = {name: value.clone() + (update * .01 if name.split('.')[0] in audit.TRAINABLE else 0)
             for name, value in original.items()}
    groups, moments = [], {}
    index = 0
    for name, base_key, prefixes in [('encoder_token', 'encoder_token_lr', audit.TRAINABLE[:-1]),
                                     ('fusion', 'fusion_lr', ('fuse_convs_dict',))]:
        ids = []
        for prefix in prefixes:
            ids.append(index)
            if update:
                moments[index] = {'step': torch.tensor(float(update)), 'exp_avg': torch.zeros(2),
                                  'exp_avg_sq': torch.ones(2) * .01}
            index += 1
        groups.append({'name': name, 'params': ids, 'base_lr': p['training'][base_key],
            'lr': audit.lr_at(update, p['training'][base_key], p['training']), 'eps': 1e-8,
            'betas': (.9, .99), 'weight_decay': 0.})
    np_state = np.random.RandomState(5).get_state()
    counts = {name: 0 for name in p['training']['call_caps']}
    counts.update(optimizer_updates=update, backward_calls=4 * update, optimizer_step_attempts=update)
    sampler = {'owned': 'synthetic sampler state', 'completed_updates': update}
    state = {'format': 'codeformer-adaptation-resumable-state-v2', 'protocol_sha256': audit.PROTOCOL_SHA256,
        'completed_updates': update, 'schedule': {'completed_updates': update, 'protocol_training': p['training']},
        'EMA': False, 'no_automatic_resume_supported': True, 'model': model,
        'optimizer': {'param_groups': groups, 'state': moments}, 'sampler': sampler,
        'rng': {'python': random.Random(5).getstate(),
            'numpy': {'algorithm': np_state[0], 'keys': np_state[1].tolist(), 'position': np_state[2],
                      'has_gauss': np_state[3], 'cached_gaussian': np_state[4]},
            'torch_cpu': torch.get_rng_state(), 'torch_cuda': [torch.tensor([1, 2], dtype=torch.uint8)]},
        'counts': counts, 'resume_safe': True, 'partial_microbatches_discarded': 0}
    return p, original, sampler, state


class ReturnTrajectoryTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / 'scratch'
        scratch.mkdir(exist_ok=True)
        self.root = scratch / ('v2_return_trajectory_' + uuid.uuid4().hex)
        self.root.mkdir()

    def tearDown(self):
        exact = self.root.resolve(strict=True)
        self.assertEqual(exact.parent.name, 'scratch')
        self.assertTrue(exact.name.startswith('v2_return_trajectory_'))
        self.assertFalse(exact.is_symlink())
        shutil.rmtree(exact)

    def test_source_python310_and_no_neural_execution_api(self):
        source = Path(audit.__file__).read_text()
        tree = ast.parse(source, feature_version=(3, 10))
        self.assertNotIn('.backward(', source)
        self.assertNotIn('torch.optim.', source)
        self.assertNotIn('load_state_dict(', source)
        for call in [node for node in ast.walk(tree) if isinstance(node, ast.Call) and
                     isinstance(node.func, ast.Attribute) and node.func.attr == 'load' and
                     isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch']:
            keywords = {node.arg: node.value for node in call.keywords}
            self.assertTrue(ast.literal_eval(keywords['weights_only']))
            self.assertEqual(ast.literal_eval(keywords['map_location']), 'cpu')

    def test_sampler_four_slots_eligibility_and_lr_replayed(self):
        p, refs, rows = history_fixture()
        report, states, expected = audit.audit_history(rows, refs, p)
        self.assertEqual(report['committed_exposures'], 8)
        self.assertEqual(report['detail_LPIPS_exposures'], 4)
        self.assertEqual(states[2]['completed_updates'], 2)
        self.assertEqual(len(expected), 8)
        rows[1]['TRAIN_selection']['id_a'] = rows[0]['TRAIN_selection']['id_a']
        with self.assertRaisesRegex(audit.AuditError, 'sampler replay'):
            audit.audit_history(rows, refs, p)

    def test_weighted_losses_reject_missing_or_changed_terms(self):
        p, _, rows = history_fixture(1)
        sample = rows[0]['microbatches'][2]
        self.assertGreater(audit.check_loss(sample, p['training']['loss_weights']), 0)
        sample['weighted_terms']['coarse_rgb_l1'] += .01
        with self.assertRaises(audit.AuditError):
            audit.check_loss(sample, p['training']['loss_weights'])

    def test_inactive_noise_warning_has_zero_contribution_active_mismatch_stops(self):
        def bound_parameters(role):
            parameters = data.camera_parameters('tr_ffhq_00001', role, 1)
            linear, rgb = data._noise_fields(parameters)
            parameters['noise_fields'] = {'shape': list(linear.shape), 'dtype': 'float64',
                'linear_noise_sha256': hashlib.sha256(linear.tobytes()).hexdigest(),
                'rgb_noise_sha256': hashlib.sha256(rgb.tobytes()).hexdigest()}
            return parameters, linear
        inactive, linear = bound_parameters('B')
        inactive['noise_fields']['linear_noise_sha256'] = '0' * 64
        report = audit.check_noise_fields(inactive, 'owned inactive fixture', {'numpy': 'different fixture version'})
        self.assertEqual(report['exact_hash_matches'], 1)
        warning = report['unused_field_replay_warnings'][0]
        self.assertFalse(warning['exact_cross_runtime_array_hash_verified'])
        self.assertTrue(all(value == 0 for value in warning['coefficients'].values()))
        signal = np.linspace(0, 1, linear.size).reshape(linear.shape)
        contribution = linear * np.sqrt(inactive['shot_coefficient'] * signal + inactive['read_sigma_linear']**2)
        self.assertTrue(np.equal(contribution, 0).all())
        active, _ = bound_parameters('A')
        active['noise_fields']['linear_noise_sha256'] = '0' * 64
        with self.assertRaisesRegex(audit.AuditError, 'ACTIVE'):
            audit.check_noise_fields(active, 'owned active fixture', {'numpy': 'different fixture version'})

    def test_producer_normal_dark_hash_bindings_remain_exact(self):
        p, refs, rows = history_fixture(1)
        rows[0]['microbatches'][1]['pair_record']['parameters']['noise_fields'] = {'changed producer binding': True}
        with self.assertRaisesRegex(audit.AuditError, 'share fields'):
            audit.audit_history(rows, refs, p)

    def test_history_recovers_only_committed_pending_prefix_and_keeps_tail(self):
        path = self.root / 'training.jsonl'
        _, _, rows = history_fixture(2)
        path.write_text(json.dumps(rows[0]) + '\n{"partial":', encoding='utf-8')
        recovered, warnings = audit.read_history(path, {'pending_step_record': rows[1]})
        self.assertEqual(len(recovered), 2)
        self.assertEqual(len(warnings), 2)
        bad = deepcopy(rows[1]); bad['update'] = 3
        with self.assertRaises(audit.AuditError):
            audit.read_history(path, {'pending_step_record': bad})

    def test_path_escape_rejected_before_read(self):
        for name in ['../escape', '/absolute', 'C:/outside', 'a\\b']:
            with self.assertRaises(audit.AuditError):
                audit.safe_file(self.root, name)

    def test_checkpoint_baseline_frozen_adam_rng_and_changes_cpu(self):
        p, original, sampler, state = tiny_checkpoint(1)
        path = self.root / 'owned_cpu_checkpoint.pt'
        torch.save(state, path)
        loaded = torch.load(path, map_location='cpu', weights_only=True)
        report = audit.check_checkpoint(loaded, original, p, sampler, 1)
        self.assertEqual(report['Adam_step_histogram'], {1: 6})
        self.assertEqual(report['module_changes']['generator']['changed_tensors'], 0)
        self.assertEqual(report['module_changes']['encoder']['changed_tensors'], 1)
        self.assertFalse(report['resume_authorized'])
        loaded['model']['generator.weight'][0] = 1.
        with self.assertRaisesRegex(audit.AuditError, 'Frozen'):
            audit.check_checkpoint(loaded, original, p, sampler, 1)

    def test_initial_state_exact_and_no_optimizer_moments(self):
        p, original, sampler, state = tiny_checkpoint(0)
        report = audit.check_checkpoint(state, original, p, sampler, 0)
        self.assertEqual(report['Adam_parameter_states'], 0)
        state['model']['encoder.weight'][0] = .1
        with self.assertRaisesRegex(audit.AuditError, 'Update0'):
            audit.check_checkpoint(state, original, p, sampler, 0)

    def test_bad_adam_moments_and_rng_are_explicit_failures(self):
        p, original, sampler, state = tiny_checkpoint()
        state['optimizer']['state'][0]['exp_avg_sq'][0] = -.1
        with self.assertRaisesRegex(audit.AuditError, 'negative'):
            audit.check_checkpoint(state, original, p, sampler, 1)
        state['optimizer']['state'][0]['exp_avg_sq'][0] = .1
        state['rng']['torch_cuda'].append(torch.tensor([3], dtype=torch.uint8))
        with self.assertRaisesRegex(audit.AuditError, 'Single-L4'):
            audit.check_checkpoint(state, original, p, sampler, 1)

    def test_tree_digest_covers_optimizer_and_rng_values(self):
        _, _, _, state = tiny_checkpoint()
        before = audit.tree_digest(state)
        state['optimizer']['state'][0]['exp_avg'][0] = .1
        self.assertNotEqual(before, audit.tree_digest(state))

    def test_training_pngs_and_teacher_arrays_bound_without_model(self):
        p, refs, rows = history_fixture(1)
        imported = self.root
        out = imported / p['outputs']; out.mkdir(parents=True)
        expected = {}
        frozen = {'behavior_teacher': 'b' * 64}
        for sample in rows[0]['microbatches']:
            record = sample['pair_record']; slot = sample['slot']
            path = out / 'training_inputs/update_0001' / ('slot_%d' % slot); path.mkdir(parents=True)
            pixels = np.full((256, 256, 3), slot * 20, dtype=np.uint8)
            floats = np.ascontiguousarray(pixels.astype(np.float32) / np.float32(255))
            Image.fromarray(pixels).save(path / 'input.png')
            linear, rgb = data._noise_fields(record['parameters'])
            record['parameters']['noise_fields'] = {'shape': list(linear.shape), 'dtype': 'float64',
                'linear_noise_sha256': hashlib.sha256(linear.tobytes()).hexdigest(),
                'rgb_noise_sha256': hashlib.sha256(rgb.tobytes()).hexdigest()}
            record.update(input256_rgb8_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                input256_float32_sha256=hashlib.sha256(floats.tobytes()).hexdigest(),
                saved_input_path=(path / 'input.png').relative_to(imported).as_posix(),
                saved_input_sha256=audit.sha256(path / 'input.png'))
            (path / 'pair_record.json').write_text(json.dumps(record))
            expected[(1, slot)] = sample
        anchor_dir = out / 'behavior_anchors/update_0001'; anchor_dir.mkdir(parents=True)
        np.save(anchor_dir / 'delivered256.npy', np.full((1, 3, 256, 256), .5, dtype=np.float32))
        np.save(anchor_dir / 'logits.npy', np.zeros((1, 256, 1024), dtype=np.float32))
        record = expected[(1, 3)]['pair_record']
        anchor = {'update': 1, 'slot': 3, 'pair_record': record, 'reference_id': record['reference_id'],
            'input256_float32_sha256': record['input256_float32_sha256'], 'teacher_state_sha256': frozen['behavior_teacher'],
            'teacher_FP32_eval_frozen': True, 'teacher_is_not_clean_identity_truth': True}
        for label, name in [('delivered', 'delivered256.npy'), ('logits', 'logits.npy')]:
            anchor[label + '_path'] = (anchor_dir / name).relative_to(imported).as_posix()
            anchor[label + '_sha256'] = audit.sha256(anchor_dir / name)
        (anchor_dir / 'record.json').write_text(json.dumps(anchor))
        expected[(1, 3)]['behavior_anchor'] = anchor
        report = audit.audit_inputs_anchors(imported, out, refs, expected, frozen, 1, p)
        self.assertEqual(report['verified_TRAIN_input_records'], 4)
        self.assertEqual(report['verified_teacher_anchor_updates'], [1])
        altered = np.full((1, 3, 256, 256), float('nan'), dtype=np.float32)
        np.save(anchor_dir / 'delivered256.npy', altered)
        anchor['delivered_sha256'] = audit.sha256(anchor_dir / 'delivered256.npy')
        (anchor_dir / 'record.json').write_text(json.dumps(anchor))
        expected[(1, 3)]['behavior_anchor'] = anchor
        with self.assertRaisesRegex(audit.AuditError, 'finiteness'):
            audit.audit_inputs_anchors(imported, out, refs, expected, frozen, 1, p)


if __name__ == '__main__':
    unittest.main()
