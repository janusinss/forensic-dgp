"""Source and CPU tensor/gradient verification tests for Unified R2 CodeFormer completion v2."""
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


for pkg in ('research', 'research.completion_codeformer_unified_r2_v2'):
    if pkg not in sys.modules:
        mod = types.ModuleType(pkg)
        mod.__path__ = []
        sys.modules[pkg] = mod

contracts = load(
    'research.completion_codeformer_unified_r2_v2.contracts', HERE / 'contracts.py'
)
objectives = load(
    'research.completion_codeformer_unified_r2_v2.unified_objectives',
    HERE / 'unified_objectives.py',
)
budget_module = load(
    'research.completion_codeformer_unified_r2_v2.optimized_budget',
    HERE / 'optimized_budget.py',
)
terminal_module = load(
    'research.completion_codeformer_unified_r2_v2.terminal_admission',
    HERE / 'terminal_admission.py',
)
TORCH_IMPORTED_AT_LOAD = any(
    n == 'torch' or n.startswith('torch.') for n in sys.modules
)


class UnifiedR2V2Tests(unittest.TestCase):
    def source(self):
        return (HERE / 'cctv_completion_codeformer_unified_r2_v2_vm.py').read_text(
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
        fusion_names = [n for n in state if contracts.fusion_route_name(n)]
        prior_names = [n for n in state if contracts.frozen_prior_name(n)]
        self.assertEqual(len(state), 497)
        self.assertEqual(len(code_names), 278)
        self.assertEqual(len(fusion_names), 54)
        self.assertEqual(len(prior_names), 165)
        self.assertEqual(set(code_names) & set(fusion_names), set())
        self.assertEqual(
            set(code_names) | set(fusion_names) | set(prior_names), set(state)
        )

    def test_orthogonal_balanced_fixed128_schedule(self):
        p = HERE / 'orthogonal_balanced128_subset.json'
        self.assertEqual(
            hashlib.sha256(p.read_bytes()).hexdigest(), contracts.SUBSET_SHA
        )
        data = json.loads(p.read_text(encoding='utf-8'))
        rows = data['cases_in_reference_role_order']
        by_id = {r['id']: r for r in rows}
        flat = sum(data['schedule'], [])
        self.assertEqual(len(rows), 128)
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual(len(data['schedule']), 32)
        self.assertEqual(set(by_id), set(flat))
        self.assertEqual(len({r['reference_id'] for r in rows}), 32)
        for batch in data['schedule']:
            self.assertEqual(len({by_id[cid]['mask_kind'] for cid in batch}), 4)
            self.assertEqual(len({by_id[cid]['reference_id'] for cid in batch}), 4)
        for g in contracts.KINDS:
            self.assertEqual(
                sum(r['role'] == 'fit_anchor' and r['mask_kind'] == g for r in rows),
                16,
            )
            self.assertEqual(
                sum(r['role'] == 'fit_augmented' and r['mask_kind'] == g for r in rows),
                16,
            )

    def test_split_cosine_lr_schedules(self):
        code_lrs = [contracts.code_lr_for_cycle(c) for c in range(1, 33)]
        fusion_lrs = [contracts.fusion_lr_for_cycle(c) for c in range(1, 33)]
        self.assertAlmostEqual(code_lrs[0], 4e-6, places=12)
        self.assertAlmostEqual(code_lrs[-1], 1e-6, places=12)
        self.assertAlmostEqual(fusion_lrs[0], 8e-6, places=12)
        self.assertAlmostEqual(fusion_lrs[-1], 2e-6, places=12)
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
        self.assertEqual(sum(contracts.TYPED_CAPS.values()), 512)
        self.assertEqual(c['planned_network_forwards'], 512)
        self.assertEqual(c['hard_forward_cap'], 540)
        self.assertEqual(c['backward_calls_newphase'], 256)
        self.assertEqual(
            c['code_optimizer_steps_newphase'] + c['fusion_optimizer_steps_newphase'],
            64,
        )
        self.assertEqual(c['saved_generated_NPZ_max'], 388)
        self.assertEqual(c['total_NPZ_cap'], 468)
        self.assertEqual(
            c['cooperative_stop_seconds'] + c['save_reserve_seconds'],
            c['worker_seconds'],
        )

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
        eroded16 = 1.0 - F.max_pool2d(1.0 - m16, kernel_size=3, stride=1, padding=1)
        inner_ring16 = (m16 - eroded16).clamp(0.0, 1.0)
        core16 = eroded16.clamp(0.0, 1.0)
        ir_w = float(w16_norm[inner_ring16 > 0.5].mean().item())
        or_w = float(w16_norm[r16 > 0.5].mean().item())
        vis_w = float(w16_norm[v16 > 0.5].mean().item())
        core_w = float(w16_norm[core16 > 0.5].mean().item())
        self.assertGreater(ir_w, or_w)
        self.assertGreater(or_w, vis_w)
        self.assertGreater(vis_w, core_w)

        logits = torch.randn((1, 256, 512), dtype=torch.float32, requires_grad=True)
        lq_feat = torch.randn((1, 256, 16, 16), dtype=torch.float32, requires_grad=True)
        target_codes = torch.randint(0, 512, (1, 256), dtype=torch.long)
        target_latent = torch.randn((1, 256, 16, 16), dtype=torch.float32)
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
            role_weight=0.95,
            style='GRAY',
            F=F,
        )
        fusion_loss.backward()
        self.assertTrue(torch.isfinite(fusion_loss).item())
        self.assertTrue(
            torch.equal(out256 * (1.0 - mask256), gray256 * (1.0 - mask256))
        )
        self.assertAlmostEqual(fusion_diag['chroma_weight'], 0.70, places=6)
        self.assertGreater(fusion_diag['delivered256_chroma_L1'], 0.0)
        self.assertGreater(fusion_diag['delivered256_core_L1'], 0.0)
        self.assertIsNotNone(raw512.grad)
        self.assertGreater(float(raw512.grad.norm().item()), 0.0)

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
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(UnifiedR2V2Tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    record = {
        'format': 'unified-r2-v2-source-and-cpu-preparation-tests-v1',
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
