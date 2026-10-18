"""Saved-state CPU fixtures; no neural/model/gradient execution."""
import ast
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np

from research.codeformer_bounded_fusion_feasibility_v1 import return_state as s

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def fixture_directory():
    scratch = ROOT / 'scratch'; scratch.mkdir(exist_ok=True)
    root = scratch / ('bounded-state-fixture-' + uuid.uuid4().hex)
    root.mkdir()
    try:
        yield root
    finally:
        checked = root.resolve(strict=True)
        assert checked.parent == scratch.resolve(strict=True) and not root.is_symlink()
        shutil.rmtree(checked)


def protocol():
    return s.c.read_json(ROOT / 'outputs/cctv_codeformer_bounded_fusion_feasibility_v1_vm/protocol.json')


def trace_fixture(p):
    counts = {k: 0 for k in p['call_caps']}; rows = []
    for i, (kind, context) in enumerate(s.expected_events(p), 1):
        counts[kind] += 1
        rows.append({'event': i, 'before_call': kind, 'context': context,
                     'counts': dict(counts), 'elapsed_seconds': i / 100})
    return rows


def stats_fixture(site='32', gate=.25):
    base = np.ones((1, *s.SITES[site]), np.float32); other = base + np.float32(.2)
    replay = s.canary_replay(base, other, gate); used = replay['used_delta']
    flat = lambda a: a.reshape(-1).tolist()
    difference = other - base
    row = {'site': site, 'gate': gate, 'dtype': 'torch.float32', 'block_evaluations': 2,
           'nonfinite_count': 0, 'local_cap_with_FP32_roundoff_pass': True,
           'D': flat(replay['scale']),
           'difference_absmax': flat(np.max(np.abs(difference), (-2, -1))),
           'difference_RMS': flat(np.sqrt(np.mean(difference**2, (-2, -1), dtype=np.float32))),
           'used_delta_absmax': flat(np.max(np.abs(used), (-2, -1))),
           'used_delta_RMS': flat(np.sqrt(np.mean(used**2, (-2, -1), dtype=np.float32))),
           'tanh_abs_ge_095_fraction': flat(np.mean(np.abs(np.tanh(difference / replay['scale'])) >= .95,
                                                 (-2, -1), dtype=np.float32))}
    return base, other, used, row


class SavedStateAuditTests(unittest.TestCase):
    def test_independent_complete170_960_plan_and_order(self):
        p = protocol(); self.assertEqual(len(s.plan_rows(p)), 170)
        result = s.check_trace(trace_fixture(p), p)
        self.assertEqual(result['events'], 1130)
        self.assertEqual(result['counts']['fusion_block_evaluations'], 960)
        self.assertEqual(result['counts']['optimizer_updates'], 0)
        self.assertEqual(len(s.expected_namespace(p)), 2231)

    def test_trace_rejects_role_order_counter_and_time_drift(self):
        p = protocol(); original = trace_fixture(p)
        changes = []
        row = deepcopy(original); row[2]['context']['role'] = 'wrong'; changes.append(row)
        row = deepcopy(original); row[2], row[3] = row[3], row[2]; changes.append(row)
        row = deepcopy(original); row[2]['counts']['optimizer_updates'] = 1; changes.append(row)
        row = deepcopy(original); row[2]['elapsed_seconds'] = -1.; changes.append(row)
        row = deepcopy(original); row[2]['counts']['optimizer_updates'] = False; changes.append(row)
        changes.append(original[:-1])
        for index, rows in enumerate(changes):
            with self.subTest(index=index), self.assertRaises(ValueError): s.check_trace(rows, p)

    def test_observed_gate_has_no_metadata_input_and_center_support(self):
        a = np.full((256, 256, 3), 128, np.uint8); b = a.copy(); b[:32] = 0
        self.assertEqual(s.observed_gate(a, s.c.GATE), s.observed_gate(b, s.c.GATE))
        with self.assertRaises(ValueError): s.observed_gate(a.astype(np.float32), s.c.GATE)

    def test_metadata_gate_roundoff_requires_identical_float32_multiplier(self):
        self.assertTrue(s.gate_match(.5 + 1e-15, .5))
        self.assertFalse(s.gate_match(.50001, .5))
        self.assertFalse(s.gate_match(True, 1.))
        self.assertTrue(s.gate_match(None, None))
        self.assertFalse(s.gate_match(.25, None))

    def test_exact_content_schema_and_actual_tokens(self):
        tokens = np.zeros(256, np.int64)
        def cap(shape): return {'shape': list(shape), 'dtype': 'float32', 'sha256': 'a' * 64}
        row = {'encoder_skips': {key: cap((1, *shape)) for key, shape in s.SITES.items()},
               'pre_generator': cap((1, 256, 16, 16)), 'latent': cap((1, 256, 16, 16)),
               'pre_generator_mean': cap((1, 256, 1, 1)), 'pre_generator_std': cap((1, 256, 1, 1)),
               'logits': cap((1, 256, 1024)), 'selected_tokens_sha256': s.digest_array(tokens)}
        s.check_content(row, tokens)
        bad = deepcopy(row); bad['encoder_skips']['32']['dtype'] = 'bfloat16'
        with self.assertRaises(ValueError): s.check_content(bad, tokens)
        with self.assertRaises(ValueError): s.check_content(row, tokens + 1)

    def test_channel_bound_and_canary_frozen_replay(self):
        base, other, used, row = stats_fixture()
        s.channel_stats(row, '32', .25)
        result = s.replay_canary(base, other, used, .25, row)
        self.assertTrue(result['frozen_feature_replay_pass'])
        self.assertEqual(result['maximum_replay_error'], 0.)

    def test_false_gate_route_rejected_by_metadata_and_feature_array(self):
        base, other, used, row = stats_fixture()
        wrong = s.canary_replay(base, other, 1.)['used_delta']
        with self.assertRaises(ValueError): s.replay_canary(base, other, wrong, .25, row)
        bad = deepcopy(row); bad['gate'] = 1.
        with self.assertRaises(ValueError): s.channel_stats(bad, '32', .25)

    def test_channel_cap_fraction_and_zero_delta_not_trusted_from_booleans(self):
        _, _, _, row = stats_fixture()
        for name, value in [('used_delta_absmax', 1.), ('tanh_abs_ge_095_fraction', .3), ('D', float('nan'))]:
            bad = deepcopy(row); bad[name][0] = value
            with self.subTest(name=name), self.assertRaises(ValueError): s.channel_stats(bad, '32', .25)
        with self.assertRaises(ValueError): s.channel_stats(row, '32', .25, zero=True)

    def test_saved_used_delta_maxima_are_exact_not_tolerance_only(self):
        base, other, used, row = stats_fixture()
        bad = deepcopy(row); bad['used_delta_absmax'][0] = float(np.nextafter(np.float32(row['used_delta_absmax'][0]), np.float32(1)))
        with self.assertRaises(ValueError): s.replay_canary(base, other, used, .25, bad)

    def test_parent_raw_hashes_reuse_prior_cpu_digest_without_tensor_load(self):
        with fixture_directory() as root:
            a = root / 'original.pth'; b = root / 'stopped.pt'
            a.write_bytes(b'original fixture'); b.write_bytes(b'stopped fixture')
            assets = {label: {'bytes': path.stat().st_size, 'sha256': s.c.sha256(path)}
                      for label, path in [('original', a), ('stopped', b)]}
            result = s.parent_state_audit(a, b, assets)
            self.assertFalse(result['original']['tensors_loaded_again'])
            self.assertEqual(result['stopped']['prior_closed_CPU_state_audit_sha256'], s.PRIOR_SHA)
            b.write_bytes(b'changed')
            with self.assertRaises(ValueError): s.parent_state_audit(a, b, assets)

    def test_strict_closed_transport_required_before_packet_or_parents(self):
        with fixture_directory() as root:
            packet = root / 'packet'; packet.mkdir(); imported = root / 'imported'; imported.mkdir()
            receipt = root / 'transport.json'
            s.c.write_new_json(receipt, {'complete': True, 'archive_sha256': s.ARCHIVE_SHA,
                'protocol_sha256': s.PROTOCOL_SHA, 'manifest_sha256': s.MANIFEST_SHA, 'import_directory': str(imported)})
            with self.assertRaisesRegex(ValueError, 'Strict import'):
                s.audit(imported, packet, receipt, root / 'absent-prior', root / 'absent-original', root / 'absent-stopped')

    def test_failure_receipt_is_preserved_and_rerun_refused(self):
        with fixture_directory() as root, redirect_stdout(io.StringIO()):
            out = root / 'receipt'
            self.assertEqual(s.run(root / 'missing', root, root / 'missing-json', root / 'missing-prior',
                                   root / 'missing-original', root / 'missing-stopped', out), 1)
            saved = s.c.read_json(out / 'state_audit.json')
            self.assertFalse(saved['evidence_consistency_pass'])
            with self.assertRaises(ValueError):
                s.run(root, root, root, root, root, root, out)

    def test_auditor_source_has_no_model_framework_gradient_or_cuda_call(self):
        tree = ast.parse(Path(s.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse(any(a.name.split('.')[0] in ('torch', 'third_party') for a in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertFalse(node.module and node.module.split('.')[0] in ('torch', 'third_party'))
            if isinstance(node, ast.Call):
                final = node.func.attr if isinstance(node.func, ast.Attribute) else ''
                self.assertNotIn(final, {'backward', 'grad', 'cuda', 'load_state_dict', 'step'})


if __name__ == '__main__':
    unittest.main()
