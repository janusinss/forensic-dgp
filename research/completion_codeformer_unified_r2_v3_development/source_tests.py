"""Source and CPU tensor/gradient verification tests for Unified R2 CodeFormer completion v3."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for pkg in ('research', 'research.completion_codeformer_unified_r2_v3'):
    if pkg not in sys.modules:
        mod = types.ModuleType(pkg)
        mod.__path__ = []
        sys.modules[pkg] = mod

contracts = load(
    'research.completion_codeformer_unified_r2_v3.contracts', HERE / 'contracts.py'
)
objectives = load(
    'research.completion_codeformer_unified_r2_v3.unified_objectives',
    HERE / 'unified_objectives.py',
)
budget_module = load(
    'research.completion_codeformer_unified_r2_v3.optimized_budget',
    HERE / 'optimized_budget.py',
)
terminal_module = load(
    'research.completion_codeformer_unified_r2_v3.terminal_admission',
    HERE / 'terminal_admission.py',
)
TORCH_IMPORTED_AT_LOAD = any(
    n == 'torch' or n.startswith('torch.') for n in sys.modules
)


class UnifiedR2V3Tests(unittest.TestCase):
    def source(self):
        return (HERE / 'cctv_completion_codeformer_unified_r2_v3_vm.py').read_text(
            encoding='utf-8'
        )

    def test_no_torch_imported_at_module_load(self):
        self.assertFalse(TORCH_IMPORTED_AT_LOAD)

    def test_python310_all_sources(self):
        for p in HERE.glob('*.py'):
            ast.parse(p.read_text(encoding='utf-8'), feature_version=(3, 10))

    def test_dual_optimizer_constructors(self):
        tree = ast.parse(self.source())
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == 'Adam'
        ]
        self.assertEqual(len(calls), 2)

    def test_no_teacher_or_oracle_forward(self):
        src = self.source()
        self.assertNotIn('oracle_indices=', src)
        self.assertNotIn('hq_vqgan', src)
        self.assertNotIn('teacher_encoder', src)
        self.assertNotIn('teacher_quantizer', src)

    def test_exact_497_tensor_partition(self):
        state = json.loads(
            (
                ROOT
                / 'outputs/completion_codeformer_same_codebook_probe_v1_return/'
                'cctv_completion_codeformer_same_codebook_probe_v1_results/outputs/state_before.json'
            ).read_text(encoding='utf-8')
        )['tensors']
        code_names = [n for n in state if contracts.code_route_name(n)]
        skip_names = [n for n in state if contracts.skip_encoder_name(n)]
        bot_names = [n for n in state if contracts.bottleneck_encoder_name(n)]
        pred_names = [n for n in state if contracts.predictor_route_name(n)]
        fusion_names = [n for n in state if contracts.fusion_route_name(n)]
        prior_names = [n for n in state if contracts.frozen_prior_name(n)]
        self.assertEqual(len(state), 497)
        self.assertEqual(len(code_names), 278)
        self.assertEqual(len(skip_names), 94)
        self.assertEqual(len(bot_names), 70)
        self.assertEqual(len(pred_names), 114)
        self.assertEqual(set(skip_names) | set(bot_names) | set(pred_names), set(code_names))
        self.assertEqual(len(fusion_names), 54)
        self.assertEqual(len(prior_names), 165)
        self.assertEqual(set(code_names) & set(fusion_names), set())
        self.assertEqual(
            set(code_names) | set(fusion_names) | set(prior_names), set(state)
        )

    def test_dual_192fit_schedules_and_union(self):
        p1 = HERE / 'balanced128_subset.json'
        p2 = HERE / 'orthogonal_balanced128_subset.json'
        self.assertEqual(
            hashlib.sha256(p1.read_bytes()).hexdigest(), contracts.BRANCH1_SUBSET_SHA
        )
        self.assertEqual(
            hashlib.sha256(p2.read_bytes()).hexdigest(), contracts.SUBSET_SHA
        )
        d1 = json.loads(p1.read_text(encoding='utf-8'))
        d2 = json.loads(p2.read_text(encoding='utf-8'))
        rows1 = d1['cases_in_reference_role_order']
        rows2 = d2['cases_in_reference_role_order']
        by_id1 = {r['id']: r for r in rows1}
        by_id2 = {r['id']: r for r in rows2}
        self.assertEqual(len(rows1), 128)
        self.assertEqual(len(rows2), 128)
        self.assertEqual(len(set(by_id1) | set(by_id2)), 192)
        self.assertEqual(len(set(by_id1) & set(by_id2)), 64)
        flat1 = sum(d1['schedule'], [])
        flat2 = sum(d2['schedule'], [])
        self.assertEqual(len(flat1), len(set(flat1)))
        self.assertEqual(len(flat2), len(set(flat2)))
        self.assertEqual(len(d1['schedule']), 32)
        self.assertEqual(len(d2['schedule']), 32)
        self.assertEqual(set(by_id1), set(flat1))
        self.assertEqual(set(by_id2), set(flat2))
        self.assertEqual(len({r['reference_id'] for r in rows1}), 32)
        self.assertEqual(len({r['reference_id'] for r in rows2}), 32)
        for batch in d1['schedule']:
            self.assertEqual(len({by_id1[cid]['reference_id'] for cid in batch}), 4)
        for batch in d2['schedule']:
            self.assertEqual(len({by_id2[cid]['mask_kind'] for cid in batch}), 4)
            self.assertEqual(len({by_id2[cid]['reference_id'] for cid in batch}), 4)
        for g in contracts.KINDS:
            self.assertEqual(sum(r['mask_kind'] == g for r in rows1), 32)
            self.assertEqual(
                sum(r['role'] == 'fit_anchor' and r['mask_kind'] == g for r in rows2),
                16,
            )
            self.assertEqual(
                sum(r['role'] == 'fit_augmented' and r['mask_kind'] == g for r in rows2),
                16,
            )

    def test_split_cosine_lr_schedules(self):
        b1_lrs = [contracts.code_branch1_lr_for_cycle(c) for c in range(1, 33)]
        code_lrs = [contracts.code_lr_for_cycle(c) for c in range(1, 33)]
        fusion_lrs = [contracts.fusion_lr_for_cycle(c) for c in range(1, 33)]
        self.assertAlmostEqual(b1_lrs[0], 8.0e-6, places=12)
        self.assertAlmostEqual(b1_lrs[-1], 2.0e-6, places=12)
        self.assertAlmostEqual(code_lrs[0], 4.0e-6, places=12)
        self.assertAlmostEqual(code_lrs[-1], 1.0e-6, places=12)
        self.assertAlmostEqual(fusion_lrs[0], 1.0e-5, places=12)
        self.assertAlmostEqual(fusion_lrs[-1], 3.0e-6, places=12)
        for a, b in zip(b1_lrs, b1_lrs[1:]):
            self.assertGreater(a, b)
        for a, b in zip(code_lrs, code_lrs[1:]):
            self.assertGreater(a, b)
        for a, b in zip(fusion_lrs, fusion_lrs[1:]):
            self.assertGreater(a, b)

    def gray_fixture(self):
        return {
            'records': [
                {
                    'id': str(i),
                    'style': 'GRAY',
                    'mask_kind': contracts.KINDS[i % 4],
                    'GRAY_raw_mean_channel_spread': 0.02,
                    'GRAY_PNG_mean_channel_spread': 5.0,
                }
                for i in range(8)
            ]
        }

    def test_gray_gate_all8macro4geometry(self):
        baseline = self.gray_fixture()
        now = deepcopy(baseline)
        self.assertTrue(
            contracts.compare_gray(now, baseline)[
                'all_GRAY_macro_geometry_chroma_nonworse'
            ]
        )
        now['records'][0]['GRAY_PNG_mean_channel_spread'] = 6.0
        r = contracts.compare_gray(now, baseline)
        self.assertFalse(r['all_GRAY_macro_geometry_chroma_nonworse'])
        self.assertEqual(
            {v['group'] for v in r['nonworse_failures']},
            {'macro', 'geometry/lower_face'},
        )

    def test_gray_gate_missing_case_refused(self):
        now = self.gray_fixture()
        now['records'].pop()
        with self.assertRaises(ValueError):
            contracts.gray_style(now)

    def test_planned_counts(self):
        c = contracts.CONTRACT
        self.assertEqual(sum(contracts.TYPED_CAPS.values()), 640)
        self.assertEqual(c['planned_network_forwards'], 640)
        self.assertEqual(c['hard_forward_cap'], 668)
        self.assertEqual(c['backward_calls_newphase'], 384)
        self.assertEqual(c['selected_fit_union_cases'], 192)
        self.assertEqual(c['fusion_moment_calibration_steps'], 64)
        self.assertEqual(
            c['code_optimizer_steps_newphase'] + c['fusion_optimizer_steps_newphase'],
            96,
        )
        self.assertEqual(c['saved_generated_NPZ_max'], 388)
        self.assertEqual(c['total_NPZ_cap'], 468)
        self.assertEqual(
            c['cooperative_stop_seconds'] + c['save_reserve_seconds'],
            c['worker_seconds'],
        )
        self.assertEqual(c['fusion_loss_weights']['delivered256_core_L1'], 0.0)
        self.assertEqual(c['fusion_loss_weights']['tiny_mask_area_tau256'], 300.0)

    def test_budget_exact_owned_extents_seek_and_overwrite_refusal(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            b = budget_module.OutputBudget(
                out, 0, disk_free=lambda: 20 * 1024 ** 3, clock=lambda: 1
            )
            p = out / 'FIT_logits/a.npz'
            with b.open(p) as stream:
                stream.write(b'abcd')
                stream.seek(1)
                stream.write(b'X')
            self.assertEqual(b.total(), 4)
            self.assertTrue(b.assert_ledger_consistent())
            with self.assertRaises(ValueError):
                b.open(p)
            with self.assertRaises(ValueError):
                b.open(out.parent / 'escape.npz')

    def test_budget_terminal_cap_before_write(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            b = budget_module.OutputBudget(
                out, 0, disk_free=lambda: 20 * 1024 ** 3, clock=lambda: 1
            )
            with b.open(
                out / 'terminal_training_state.pth.partial', terminal=True
            ) as stream:
                with self.assertRaises(ValueError):
                    b.before_entry_write(
                        stream.entry,
                        contracts.CONTRACT['terminal_bytes'] + 1,
                        terminal=True,
                    )

    def test_terminal_internal_caps_and_namespace(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            def pack(name, extra=(), pickle_size=1):
                p = root / name
                with zipfile.ZipFile(p, 'w', compression=zipfile.ZIP_STORED) as z:
                    z.writestr('archive/data.pkl', b'x' * pickle_size)
                    z.writestr('archive/data/0', b'1234')
                    z.writestr('archive/version', b'3')
                    for item in extra:
                        z.writestr(item, b'x')
                return p

            self.assertTrue(
                terminal_module.admit_terminal_zip(
                    pack('ok.partial'), guard=lambda: None
                )['all_member_CRC_pass']
            )
            for path in (
                pack('unsafe.partial', ['archive/../escape']),
                pack('script.partial', ['archive/run.py']),
                pack('pickle.partial', pickle_size=1024 ** 2 + 1),
                pack(
                    'entries.partial',
                    ['archive/data/' + str(i) for i in range(1, 1855)],
                ),
            ):
                with self.assertRaises(ValueError):
                    terminal_module.admit_terminal_zip(path, guard=lambda: None)
                self.assertTrue(path.exists())

    def test_pytorch_cpu_objectives_and_gradients(self):
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            return
        torch.manual_seed(29)
        mask256 = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
        mask256[:, :, 96:224, 64:192] = 1.0
        mask512 = F.interpolate(mask256, size=(512, 512), mode='nearest')

        w16_norm, m16, r16, v16 = objectives.spatial_support_weights(mask256, F=F)
        self.assertEqual(tuple(w16_norm.shape), (1, 1, 16, 16))
        self.assertAlmostEqual(float(w16_norm.mean().item()), 1.0, places=6)
        inner_ring16, core16, outer_ring16, vis16 = objectives.decompose_16x16_regions(mask256, F=F)
        ir_w = float(w16_norm[inner_ring16 > 0.5].mean().item())
        core_w = float(w16_norm[core16 > 0.5].mean().item())
        or_w = float(w16_norm[outer_ring16 > 0.5].mean().item())
        vis_w = float(w16_norm[vis16 > 0.5].mean().item())
        self.assertGreater(ir_w, or_w)
        self.assertGreater(or_w, vis_w)
        self.assertGreater(vis_w, core_w)

        w16_b1, _, _, _ = objectives.spatial_support_weights_hole_ring(mask256, F=F)
        self.assertEqual(tuple(w16_b1.shape), (1, 1, 16, 16))
        self.assertAlmostEqual(float(w16_b1.mean().item()), 1.0, places=6)

        logits_b1 = torch.randn((1, 256, 512), dtype=torch.float32, requires_grad=True)
        lq_b1 = torch.randn((1, 256, 16, 16), dtype=torch.float32, requires_grad=True)
        target_codes = torch.randint(0, 512, (1, 256), dtype=torch.long)
        target_latent = torch.randn((1, 256, 16, 16), dtype=torch.float32)
        b1_loss, b1_diag, _ = objectives.spatial_code_latent_loss_branch1(
            logits_b1,
            lq_b1,
            target_codes,
            target_latent,
            mask256,
            role_weight=1.25,
            latent_mse_weight=1.00,
            F=F,
        )
        b1_loss.backward()
        self.assertTrue(torch.isfinite(b1_loss).item())
        self.assertGreater(float(logits_b1.grad.norm().item()), 0.0)
        self.assertGreater(float(lq_b1.grad.norm().item()), 0.0)
        self.assertAlmostEqual(b1_diag['latent_mse_weight'], 1.00, places=6)

        logits = torch.randn((1, 256, 512), dtype=torch.float32, requires_grad=True)
        lq_feat = torch.randn((1, 256, 16, 16), dtype=torch.float32, requires_grad=True)
        code_loss, code_diag, _ = objectives.spatial_code_latent_loss(
            logits,
            lq_feat,
            target_codes,
            target_latent,
            mask256,
            role_weight=1.05,
            latent_mse_weight=0.10,
            F=F,
        )
        code_loss.backward()
        self.assertTrue(torch.isfinite(code_loss).item())
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(lq_feat.grad)
        self.assertGreater(float(logits.grad.norm().item()), 0.0)
        self.assertGreater(float(lq_feat.grad.norm().item()), 0.0)
        self.assertGreater(code_diag['hole_16x16_tokens'], 0)
        self.assertAlmostEqual(code_diag['latent_mse_weight'], 0.10, places=6)

        raw512 = torch.randn((1, 3, 512, 512), dtype=torch.float32, requires_grad=True)
        source256 = torch.rand((1, 3, 256, 256), dtype=torch.float32)
        gray256 = source256.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        gray512 = F.interpolate(
            gray256, size=(512, 512), mode='bilinear', align_corners=False
        )
        fusion_loss, out256, fusion_diag = objectives.unified_fusion_loss(
            raw512,
            gray256,
            mask256,
            mask512,
            gray256,
            gray512,
            role_weight=0.80,
            style='GRAY',
            F=F,
            area_tau256=300.0,
        )
        fusion_loss.backward()
        self.assertTrue(torch.isfinite(fusion_loss).item())
        self.assertTrue(
            torch.equal(out256 * (1.0 - mask256), gray256 * (1.0 - mask256))
        )
        self.assertAlmostEqual(fusion_diag['chroma_weight'], 0.25, places=6)
        self.assertGreater(fusion_diag['delivered256_chroma_L1'], 0.0)
        self.assertNotIn('delivered256_core_L1', fusion_diag)
        m256_sum = float(mask256.sum().item())
        self.assertAlmostEqual(
            fusion_diag['area_scale'], m256_sum / (m256_sum + 300.0), places=6
        )
        self.assertIsNotNone(raw512.grad)
        self.assertGreater(float(raw512.grad.norm().item()), 0.0)

        x_synth = torch.randn((128, 8), dtype=torch.float32)
        w_true = torch.randn((8, 4), dtype=torch.float32)
        b_true = torch.randn((4,), dtype=torch.float32)
        y_synth = x_synth @ w_true + b_true
        w_est, b_est = objectives.solve_ridge_delta(x_synth, y_synth, 1e-4, torch=torch)
        self.assertTrue(torch.isfinite(w_est).all().item() and torch.isfinite(b_est).all().item())
        self.assertLess(float((w_est - w_true).abs().max().item()), 1e-3)

        p1 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        a1 = torch.tensor([1.0, 1.0, 1.0])
        trust = objectives.anchor_trust_penalty([p1], [a1], coef=1e-2)
        trust.backward()
        self.assertAlmostEqual(
            float(trust.item()), 0.5 * 1e-2 * (0.0 + 1.0 + 4.0), places=7
        )
        self.assertTrue(
            torch.allclose(p1.grad, 1e-2 * (p1.detach() - a1), atol=1e-7)
        )


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(UnifiedR2V3Tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    record = {
        'format': 'unified-r2-v3-source-and-cpu-preparation-tests-v1',
        'complete': result.wasSuccessful(),
        'tests': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'torch_imported_at_module_load': TORCH_IMPORTED_AT_LOAD,
        'pytorch_cpu_executed': 'torch' in sys.modules,
        'VM_calls': 0,
        'UTC': datetime.now(timezone.utc).isoformat(),
    }
    target = HERE / 'source_preparation_tests.json'
    target.write_bytes((json.dumps(record, indent=2) + '\n').encode('utf-8'))
    raise SystemExit(0 if result.wasSuccessful() else 1)
