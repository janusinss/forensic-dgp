"""CPU-only independent data/finite-budget checks; no framework imports.

Reads the exact16 frozen TRAIN records and derives loss eligibility/counts.
Does not execute returned Python, a model, optimizer, gradient or VM call.
"""
import ast
from collections import Counter
import hashlib
import json
import operator
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
RETURN = ROOT / 'outputs/codeformer_adain_gradient_isolation_v1_return'
RUN = RETURN / 'outputs/adain_gradient_isolation_v1'
UPDATES = (1, 6, 15, 24)
COMPONENTS = ('token_ce', 'latent_mse', 'pixel_l1_rgb', 'lpips_vgg',
              'coarse_rgb_l1', 'behavior_rgb_l1', 'behavior_token_kl')
WEIGHTS = dict(zip(COMPONENTS, (.5, 1., 1., 1., 4., 2., .1)))


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def selected():
    return [load(RUN / f'training_inputs/update_{u:04d}/slot_{s}/pair_record.json')
            for u in UPDATES for s in range(1, 5)]


def term_names(record):
    p = record['parameters']
    result = ['token_ce', 'latent_mse', 'coarse_rgb_l1']
    if p['long_edge'] >= 32:
        result.extend(('pixel_l1_rgb', 'lpips_vgg'))
    if record['slot'] == 3:
        result.extend(('behavior_rgb_l1', 'behavior_token_kl'))
    return result


def static_assignment(path, name):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    matches = [n.value for n in tree.body if isinstance(n, ast.Assign) and
               any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    if len(matches) != 1:
        raise AssertionError(f'One static {name} definition required')
    def literal(node):
        if isinstance(node, ast.BinOp):
            operators = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                         ast.FloorDiv: operator.floordiv, ast.Pow: operator.pow}
            return operators[type(node.op)](literal(node.left), literal(node.right))
        if isinstance(node, ast.Dict):
            return {literal(k): literal(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.List):
            return [literal(v) for v in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(literal(v) for v in node.elts)
        return ast.literal_eval(node)
    return literal(matches[0])


class FrozenDataSelection(unittest.TestCase):
    def test_exact_train_namespace_and_frozen_pairing(self):
        rows = selected()
        self.assertEqual(len(rows), 16)
        self.assertEqual(len({r['reference_id'] for r in rows}), 8)
        history = [json.loads(x) for x in (RUN / 'training.jsonl').read_text(encoding='utf-8').splitlines()]
        export = load(next(RUN.glob('export_manifest_*.json')))['files_sha256']
        for r in rows:
            self.assertEqual(r['role'], 'train')
            self.assertRegex(r['reference_id'], r'^tr_ffhq_[0-9]{5}$')
            self.assertEqual(r, history[r['update'] - 1]['microbatches'][r['slot'] - 1]['pair_record'])
            self.assertEqual(r['geometry']['input'], [256, 256])
            self.assertEqual(r['geometry']['target'], [512, 512])
            path = RUN / f"training_inputs/update_{r['update']:04d}/slot_{r['slot']}/input.png"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, r['saved_input_sha256'])
            self.assertEqual(digest, export[path.relative_to(RETURN).as_posix()])
        for u in UPDATES:
            rr = [r for r in rows if r['update'] == u]
            for a, b in ((rr[0], rr[1]), (rr[2], rr[3])):
                self.assertEqual((a['member'], b['member']), ('normal', 'dark'))
                self.assertEqual(a['reference_id'], b['reference_id'])
                self.assertEqual(a['clean512_rgb8_sha256'], b['clean512_rgb8_sha256'])
                self.assertEqual(a['clean512_float32_sha256'], b['clean512_float32_sha256'])
                self.assertEqual(a['parameters']['noise_fields'], b['parameters']['noise_fields'])

    def test_prospective_selection_preserves_coverage_limits(self):
        rows = selected()
        a = [r for r in rows if r['slot'] == 1]
        b = [r for r in rows if r['slot'] == 3]
        self.assertEqual([r['parameters']['long_edge'] for r in a], [24, 28, 32, 40])
        self.assertEqual([r['parameters']['variant'] for r in b],
                         ['clear256', 'clear128', 'light256', 'light64'])
        self.assertTrue(all(r['parameters']['shot_coefficient'] > 0 and
                            r['parameters']['read_sigma_linear'] > 0 and
                            not r['parameters']['behavior_anchor'] for r in a))
        self.assertTrue(all(r['parameters']['behavior_anchor'] for r in b))
        self.assertFalse({'clear64', 'light128'} & {r['parameters']['variant'] for r in b})
        self.assertTrue(all(r['parameters']['image_eligible'] ==
                            (r['parameters']['long_edge'] >= 32) for r in rows))
        self.assertEqual(sum(r['parameters']['image_eligible'] for r in rows), 12)

    def test_loss_queries_derive_from_eligibility(self):
        rows = selected()
        counts = Counter(term for r in rows for term in term_names(r))
        self.assertEqual(counts, Counter(token_ce=16, latent_mse=16, coarse_rgb_l1=16,
                                        pixel_l1_rgb=12, lpips_vgg=12,
                                        behavior_rgb_l1=4, behavior_token_kl=4))
        self.assertEqual(sum(counts.values()), 80)
        self.assertEqual(2 * sum(counts.values()) + 2 * len(rows) + 1, 193)
        self.assertEqual(len(rows) * 10 * 2, 320)
        self.assertEqual(2 * (1 + len(COMPONENTS) + 1), 18)


class FiniteBudgetArithmetic(unittest.TestCase):
    def test_worker_static_caps_match_independent_derivation(self):
        caps = static_assignment(ROOT / 'research/codeformer_loss_response_v1/worker.py', 'EXPECTED_CAPS')
        expected = {'candidate_source_forwards': 32, 'candidate_probe_forwards': 320,
                    'restoration_repeat_forwards': 32, 'behavior_teacher_forwards': 4,
                    'initializer_parity_candidate_forwards': 2,
                    'initializer_parity_benchmark_forwards': 2,
                    'HQ_teacher_forwards': 8, 'perceptual_source_forwards': 24,
                    'perceptual_prerequisite_forwards': 1,
                    'component_gradient_queries': 160, 'full_gradient_queries': 32,
                    'perceptual_prerequisite_gradient_queries': 1, 'optimizer_step_attempts': 18,
                    'disposable_optimizer_steps': 18, 'committed_training_updates': 0,
                    'backward_calls': 0}
        self.assertEqual(caps, expected)
        self.assertEqual(caps, static_assignment(
            ROOT / 'research/codeformer_loss_response_v1/contract.py', 'COUNTS'))
        self.assertEqual(sum(v for k, v in caps.items() if k.endswith('_forwards')), 425)
        self.assertEqual(sum(v for k, v in caps.items() if 'gradient_queries' in k), 193)
        self.assertEqual(32 + 320 + 32 + 4 + 4, 392)

    def test_contract_limits_cover_actual_worker_consumers(self):
        contract = ROOT / 'research/codeformer_loss_response_v1/contract.py'
        worker = ROOT / 'research/codeformer_loss_response_v1/worker.py'
        limits = static_assignment(contract, 'LIMITS')
        optimizer = static_assignment(contract, 'OPTIMIZER')
        tree = ast.parse(worker.read_text(encoding='utf-8'))
        consumers = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Subscript) or not isinstance(node.slice, ast.Constant):
                continue
            value = node.value
            if isinstance(value, ast.Name) and value.id == 'limits' or \
                    isinstance(value, ast.Attribute) and value.attr == 'limits':
                consumers.add(node.slice.value)
        self.assertFalse(consumers - set(limits), 'Runtime limit name missing from canonical contract')
        self.assertEqual(limits['RSS_bytes'], 8 * 1024**3)
        self.assertEqual(limits['CUDA_reserved_bytes'], 22 * 1024**3)
        self.assertEqual(limits['required_free_bytes'], 14 * 1024**3)
        self.assertEqual(limits['reserve_free_bytes'], 4 * 1024**3)
        self.assertEqual(limits['result_payload_bytes'], 4 * 1024**3)
        self.assertEqual(limits['archive_bytes'], 9 * 1024**3 // 2)
        self.assertEqual(limits['initialization_seconds'] + limits['source_seconds'] +
                         limits['probe_seconds'], limits['worker_seconds'])
        self.assertEqual(limits['worker_seconds'], 1800)
        self.assertEqual(limits['export_seconds'], 600)
        self.assertEqual(limits['preflight_seconds'], 300)
        self.assertEqual(optimizer['gradient_accumulation_denominator'], 16)
        self.assertEqual(optimizer['fixed_group_rates'], {'encoder_token': 1e-5, 'fusion': 2e-5})
        self.assertFalse(optimizer['chained_steps'])
        self.assertTrue(optimizer['None_is_not_zero'])

    def test_sparse_vectors_and_storage_envelope(self):
        census = load(RUN / 'initialization.json')['census']['groups']
        encoder = census['encoder']['parameters']
        predictor = sum(census[k]['parameters'] for k in
                        ('position_emb', 'feat_emb', 'ft_layers', 'idx_pred_layer'))
        fusion = census['fuse_convs_dict']['parameters']
        parameters = encoder + predictor + fusion
        self.assertEqual(parameters, 59788608)
        sparse_components = 2 * 4 * (2 * (encoder + predictor) + encoder + 4 * fusion)
        self.assertEqual(sparse_components, 1362499072)
        outputs = 320 * (512 * 512 * 3 + 256 * 256 * 3) * 4
        self.assertEqual(outputs, 1258291200)
        # Bound assumptions are explicit: no per-record fullgradient vectors,
        # uncompressed NPY, <=200KiB each PNG, and additional source/metadata.
        aggregate_full = 2 * 4 * parameters
        png_and_tokens = 320 * (200 * 1024 + 256 * 8 + 1024)
        source_vectors_and_arrays = 32 * (512 * 512 * 3 * 4 + 256 * 256 * 3 * 4 +
                                         256 * 1024 * 4 + 256 * 16 * 16 * 4)
        headers_metadata = 128 * 1024**2
        conservative_payload = sparse_components + outputs + aggregate_full + \
            png_and_tokens + source_vectors_and_arrays + headers_metadata
        self.assertLess(conservative_payload, 4 * 1024**3)
        self.assertGreater(32 * 4 * parameters, 4 * 1024**3)
        #14GiB after install covers capped payload+archive+reserve and one
        # temporary largest-vector write, without copying parent checkpoints.
        self.assertLess(4 * 1024**3 + 9 * 1024**3 // 2 + 4 * 1024**3 +
                        4 * parameters + 128 * 1024**2, 14 * 1024**3)

    def test_framework_imports_absent_from_this_audit(self):
        tree = ast.parse(Path(__file__).read_text(encoding='utf-8'))
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(x.name.split('.')[0] for x in node.names)
            if isinstance(node, ast.ImportFrom):
                modules.add((node.module or '').split('.')[0])
        self.assertFalse(modules & {'torch', 'tensorflow', 'jax', 'research', 'cv2'})


if __name__ == '__main__':
    unittest.main()
