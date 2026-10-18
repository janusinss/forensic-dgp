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
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from research.codeformer_adaptation_v2 import data
from research.codeformer_fp32_comparison_v1 import worker
from research.codeformer_fp32_comparison_v1 import return_trajectory as audit


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
    state = {'format': 'codeformer-fp32-comparison-state-v1', 'protocol_sha256': audit.PROTOCOL_SHA256,
        'completed_updates': update, 'schedule': {'completed_updates': update, 'protocol_training': p['training']},
        'EMA': False, 'no_automatic_resume_supported': True, 'model': model,
        'optimizer': {'param_groups': groups, 'state': moments}, 'sampler': sampler,
        'rng': {'python': random.Random(5).getstate(),
            'numpy': {'algorithm': np_state[0], 'keys': np_state[1].tolist(), 'position': np_state[2],
                      'has_gauss': np_state[3], 'cached_gaussian': np_state[4]},
            'torch_cpu': torch.get_rng_state(), 'torch_cuda': [torch.tensor([1, 2], dtype=torch.uint8)]},
        'counts': counts, 'resume_safe': True, 'partial_microbatches_discarded': 0}
    return p, original, sampler, state


class FP32ReturnTrajectoryTests(unittest.TestCase):
    def setUp(self):
        scratch = Path.cwd() / 'scratch'; scratch.mkdir(exist_ok=True)
        self.root = scratch / ('fp32-return-fixture-' + uuid.uuid4().hex)
        self.root.mkdir()

    def tearDown(self):
        exact = self.root.resolve(strict=True)
        self.assertEqual(exact.parent, (Path.cwd() / 'scratch').resolve(strict=True))
        self.assertFalse(exact.is_symlink()); shutil.rmtree(exact)

    def test_no_neural_execution_and_all_loads_CPU_weights_only(self):
        source = Path(audit.__file__).read_text()
        tree = ast.parse(source, feature_version=(3, 10))
        for forbidden in ('.backward(', 'torch.optim.', 'load_state_dict(', 'torch.autograd.', 'CodeFormer('):
            self.assertNotIn(forbidden, source)
        for call in [node for node in ast.walk(tree) if isinstance(node, ast.Call) and
                     isinstance(node.func, ast.Attribute) and node.func.attr == 'load' and
                     isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch']:
            keys = {item.arg: item.value for item in call.keywords}
            self.assertTrue(ast.literal_eval(keys['weights_only']))
            self.assertEqual(ast.literal_eval(keys['map_location']), 'cpu')

    def test_fifty_history_exact_sampler_LR_eligibility(self):
        p, refs, rows = history_fixture(50)
        report, states, records = audit.audit_history(rows, refs, p)
        self.assertEqual(report['committed_updates'], 50)
        self.assertEqual(report['committed_exposures'], 200)
        self.assertEqual(report['detail_LPIPS_exposures'], 148)
        self.assertEqual(len(records), 200)
        self.assertEqual(states[50]['completed_updates'], 50)
        rows[2]['microbatches'][0]['image_eligible'] = False
        with self.assertRaises(audit.AuditError): audit.audit_history(rows, refs, p)

    def test_loss_scalar_reduction_tolerance_is_not_quality_gate(self):
        p, _, rows = history_fixture(1)
        sample = rows[0]['microbatches'][2]
        sample['total_before_accumulation_division'] = float(np.float32(sum(sample['weighted_terms'].values())))
        audit.check_loss(sample, p['training']['loss_weights'])
        sample['weighted_terms']['behavior_rgb_l1'] += .01
        with self.assertRaises(audit.AuditError): audit.check_loss(sample, p['training']['loss_weights'])

    def test_call_order_full_2057_sequence_refuses_mutation(self):
        p = protocol(); rows, counts = audit.expected_call_events(p)
        self.assertEqual(len(rows), 2057)
        self.assertEqual(counts['optimizer_updates'], 50)
        self.assertEqual(counts['candidate_training_forwards'], 200)
        self.assertEqual(counts['snapshot_forwards'], 812)
        self.assertEqual(counts['perceptual_training_forwards'], 148)
        out = self.root / 'outputs/fp32_comparison_v1'; (out / 'call_trace').mkdir(parents=True)
        for row in rows:
            row['seconds'] = row['event'] / 10
            (out / 'call_trace' / f"{row['event']:05d}.json").write_text(json.dumps(row))
        terminal = {'completed_updates': 50, 'counts': counts, 'last_call_reservation': rows[-1]}
        self.assertEqual(audit.audit_calls(self.root, out, p, terminal)['exact_call_reservations_verified'], 2057)
        path = out / 'call_trace/01000.json'
        changed = json.loads(path.read_text()); changed['counts']['optimizer_updates'] += 1
        path.write_text(json.dumps(changed))
        with self.assertRaises(audit.AuditError): audit.audit_calls(self.root, out, p, terminal)

    def test_checkpoint_original_frozen_Adam_commit_RNG(self):
        p, original, sampler, state = tiny_checkpoint(1)
        report = audit.check_checkpoint(state, original, p, sampler, 1)
        self.assertEqual(report['Adam_step_histogram'], {1: 6})
        self.assertEqual(report['module_changes']['generator']['changed_tensors'], 0)
        state['optimizer']['state'][0]['exp_avg_sq'][0] = -.1
        with self.assertRaises(audit.AuditError): audit.check_checkpoint(state, original, p, sampler, 1)

    def test_safe_initial_checkpoint_cannot_hide_partial_Adam_state(self):
        p, original, sampler, state = tiny_checkpoint(0)
        audit.check_checkpoint(state, original, p, sampler, 0)
        state['optimizer']['state'][0] = {'step': torch.tensor(0.), 'exp_avg': torch.zeros(2), 'exp_avg_sq': torch.zeros(2)}
        with self.assertRaises(audit.AuditError): audit.check_checkpoint(state, original, p, sampler, 0)

    def test_tree_digest_covers_optimizer_rng_and_sampler(self):
        _, _, _, state = tiny_checkpoint(1)
        before = audit.tree_digest(state)
        state['sampler']['completed_updates'] = 2
        self.assertNotEqual(audit.tree_digest(state), before)

    def test_history_pending_must_be_committed_and_contiguous(self):
        _, _, rows = history_fixture(2)
        path = self.root / 'history.jsonl'
        path.write_text(json.dumps(rows[0]) + '\n{"partial":')
        actual, warnings = audit.read_history(path, {'pending_step_record': rows[1]})
        self.assertEqual(len(actual), 2); self.assertEqual(len(warnings), 2)
        rows[1]['update'] = 4
        with self.assertRaises(audit.AuditError): audit.read_history(path, {'pending_step_record': rows[1]})

    def test_NPY_trailing_bytes_wrong_dtype_and_tokens_rejected(self):
        path = self.root / 'tokens.npy'; np.save(path, np.zeros(256, np.int64), allow_pickle=False)
        audit.load_array(path, (256,), np.int64)
        with path.open('ab') as stream: stream.write(b'trailing')
        with self.assertRaises(audit.AuditError): audit.load_array(path, (256,), np.int64)
        np.save(path, np.zeros(256, np.float32), allow_pickle=False)
        with self.assertRaises(audit.AuditError): audit.load_array(path, (256,), np.int64)

    def test_dtypes_no_autocast_or_BF16_substitution(self):
        value = {'logits_dtype': 'torch.float32', 'probabilities_dtype': 'torch.float32',
                 'native_output_dtype': 'torch.float32', 'autocast_enabled': False,
                 'actual_native_softmax_and_selected_indices_captured': True}
        audit.dtype_evidence(value)
        value['logits_dtype'] = 'torch.bfloat16'
        with self.assertRaises(audit.AuditError): audit.dtype_evidence(value)

    def test_path_and_duplicate_json_rejected_before_read(self):
        for name in ('../escape', '/absolute', 'C:/escape', 'a\\b'):
            with self.assertRaises(audit.AuditError): audit.safe_file(self.root, name)
        path = self.root / 'duplicate.json'; path.write_text('{"x":1,"x":2}')
        with self.assertRaises(audit.AuditError): audit.read_json(path)

    def test_transport_receipt_binds_exact_destination_and_manifest(self):
        manifest = self.root / audit.MANIFEST_MEMBER
        manifest.parent.mkdir(parents=True); manifest.write_text('{}')
        receipt = {'complete': True, 'archive_sha256': audit.ARCHIVE_SHA256,
            'protocol_sha256': audit.PROTOCOL_SHA256, 'manifest_member': audit.MANIFEST_MEMBER,
            'manifest_sha256': audit.MANIFEST_SHA256, 'all_members_hash_verified': True,
            'all_members_verified_before_destination_created': True, 'exact_namespace': True,
            'members': 6501, 'import_directory': str(self.root.resolve())}
        path = self.root / 'transport.json'; path.write_text(json.dumps(receipt))
        with patch.object(audit, 'sha256', return_value=audit.MANIFEST_SHA256):
            self.assertEqual(audit.verified_transport(self.root, path)['members'], 6501)
            receipt['all_members_verified_before_destination_created'] = False
            path.write_text(json.dumps(receipt))
            with self.assertRaises(audit.AuditError): audit.verified_transport(self.root, path)


if __name__ == '__main__':
    unittest.main()


