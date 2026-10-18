"""Pure standard-library contract tests: no Torch, state, model or optimizer."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'research/codeformer_role_preserving_training_v1_r2/contract.py'
SPEC = importlib.util.spec_from_file_location('role_preserving_contract_CPU_fixture', SOURCE)
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)


def protocol_fixture():
    """Synthetic inert hash maps; no real parent/model/pixel files are opened."""
    refs = ['tr_ffhq_%05d' % i for i in range(100)]
    names = set(c.HISTORICAL_RUNTIME_FILES) | {'protocol.json', 'references.json',
        'comparison_cases.json', 'perceptual/manifest.json', 'training_parity_v1.json',
        'hq_cache_parity_v1.json', 'validation_saved_v1.json'}
    names.update(x['path'] for x in c.ASSETS.values())
    zero = {'sha256': '0'*64, 'bytes': 1}
    catalog = {n: deepcopy(zero) for n in names}
    catalog['protocol.json'] = {'sha256': c.PARENT_PROTOCOL_SHA, 'bytes': 1}
    for row in c.ASSETS.values():
        catalog[row['path']] = {k: row[k] for k in ('sha256', 'bytes')}
    for row in c.PARENT_CONTROLS.values():
        catalog[row.get('path', row.get('manifest'))]['sha256'] = row['sha256']
    deps = deepcopy(catalog)
    for i in range(2011-len(catalog)):
        catalog['unused_metadata/f%04d.txt' % i] = deepcopy(zero)
    retained = {'retained/%04d/raw.npy' % i: deepcopy(zero) for i in range(1624)}
    extra = {n: deepcopy(zero) for n in c.expected_extra_returned_paths(refs)}
    extra.update(deepcopy(c.FIXED_RETURNED_DEPENDENCIES))
    sources = {n: '0'*64 for n in c.REQUIRED_SOURCE_FILES | {'data_bundle.json'}}
    sizes = dict.fromkeys(sources, 1)
    sources['research/__init__.py'] = hashlib.sha256(b'').hexdigest()
    sizes['research/__init__.py'] = 0
    sources[c.DESIGN_DOCUMENT] = c.DESIGN_DOCUMENT_SHA256
    old = {'synthetic': 'immutable provenance only'}
    p = {'format': c.FORMAT, 'module': c.MODULE, 'vm_name': c.VM_NAME,
        'parent_dir': c.PARENT_NAME, 'parent_protocol_sha256': c.PARENT_PROTOCOL_SHA,
        'outputs': c.OUTPUTS, 'export_prefix': c.PREFIX,
        'training': deepcopy(c.TRAINING), 'objective': deepcopy(c.OBJECTIVE),
        'evaluation': deepcopy(c.EVALUATION), 'gradient_route': deepcopy(c.GRADIENT_ROUTE),
        'initializer_parity': deepcopy(c.INITIALIZER_PARITY),
        'perceptual_prerequisite': deepcopy(c.PERCEPTUAL_PREREQUISITE),
        'state_policy': deepcopy(c.STATE_POLICY), 'design': deepcopy(c.DESIGN),
        'design_sha256': c.DESIGN_SHA256, 'assets': deepcopy(c.ASSETS),
        'model_digests': deepcopy(c.MODEL_DIGESTS), 'call_caps': deepcopy(c.COUNTS),
        'limits': deepcopy(c.LIMITS), **deepcopy(c.PARENT_CONTROLS),
        'status': 'PREPARED_UNRUN', 'manual_tmux_required': True,
        'numerical_initializer_gate_changed': True,
        'numerical_initializer_policy': c.INITIALIZER_PARITY['policy'],
        'new_training_recipe': True, 'actual_candidate_checkpoints_required': True,
        'data_manifest': 'references.json', 'comparison_cases': 'comparison_cases.json',
        'data_bundle_file': 'data_bundle.json', 'historical_parent_protocol': old,
        'historical_dependency_bindings': catalog, 'dependencies': deps,
        'retained_output_dependencies': retained, 'extra_returned_dependencies': extra,
        'selected_train_reference_ids': refs, 'files_sha256': sources, 'files_bytes': sizes,
        'design_document': {'path': c.DESIGN_DOCUMENT, 'sha256': c.DESIGN_DOCUMENT_SHA256}}
    for name in ('automatic_training', 'automatic_VM_launch', 'automatic_resume', 'model_qualified',
            'automatic_app_promotion', 'quality_gates_changed', 'FINAL_used', 'old_optimizer_loaded',
            'historical_RNG_loaded', 'cached_gradient_reuse', 'tone_head_used'):
        p[name] = False
    pins = {'PARENT_PROTOCOL_CANONICAL_SHA': c.canonical_sha256(old),
            'DEPENDENCIES_SHA256': c.canonical_sha256(catalog),
            'RETAINED_OUTPUTS_SHA256': c.canonical_sha256(retained)}
    return p, pins


class RoleContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary_parent = ROOT/'scratch'
        cls.temporary_parent.mkdir(exist_ok=True)

    def test_import_surface_is_standard_library_only(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'), feature_version=(3, 10))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(x.name.split('.')[0] for x in node.names)
            if isinstance(node, ast.ImportFrom):
                imported.append(node.module.split('.')[0])
        self.assertLessEqual(set(imported), {'hashlib', 'json', 'os', 'pathlib', 'platform', 're', 'stat'})
        self.assertFalse({'torch', 'numpy', 'PIL'} & set(imported))

    def test_role_term_graph_excludes_normal_clean_terms(self):
        self.assertEqual(c.role_term_names('normal', True), ('behavior_rgb_l1', 'behavior_token_kl'))
        self.assertEqual(c.role_term_names('normal', False), c.NORMAL_TERMS)
        self.assertEqual(len(c.role_term_names('dark', True)), 7)
        self.assertEqual(len(c.role_term_names('dark', False)), 5)
        with self.assertRaises(ValueError):
            c.role_term_names('normal', 1)

    def test_source_schedule_census_and_no_51(self):
        eligible = sum(slot == 4 or (u-1) % 4 >= 2 for u in range(1, 51) for slot in (2, 4))
        self.assertEqual(eligible, 74)
        self.assertEqual(c.COUNTS['perceptual_training_forwards'], eligible)
        self.assertEqual(c.COUNTS['initializer_parity_perceptual_forwards'], 1)
        self.assertEqual(c.COUNTS['initializer_training_candidate_forwards'], 2)
        self.assertEqual(sum(c.COUNTS[n] for n in c.FORWARD_COUNTERS), 1399)
        self.assertEqual(sum(c.COUNTS.values()), 1854)
        self.assertEqual(c.learning_rate(1, 1e-5), 1e-5/50)
        self.assertEqual(c.learning_rate(50, 2e-5), 2e-5)
        for update in (0, 51, True):
            with self.assertRaises(ValueError):
                c.learning_rate(update, 1e-5)

    def test_r2_numerical_contract_keeps_actual_training_and_quality_recipe(self):
        unchanged={'training':{k:v for k,v in c.TRAINING.items() if k != 'call_caps'},
            **{name:getattr(c,name) for name in ('OBJECTIVE','EVALUATION','ASSETS','MODEL_DIGESTS')}}
        self.assertEqual(c.canonical_sha256(unchanged),
            '00cd91c2a280d0d887e64d84713c81012f10be1f104fde5614447b3dbd6526eb')
        self.assertEqual(c.VM_NAME,'cctv_codeformer_role_preserving_training_v1_r2_vm')
        self.assertEqual(c.PREFIX,'cctv-codeformer-role-preserving-training-v1-r2')
        self.assertEqual(c.OUTPUTS,'outputs/role_preserving_training_v1_r2')
        self.assertEqual(c.LOCK_NAME,'.role_preserving_training_v1_r2.lock')
        self.assertTrue(c.INITIALIZER_PARITY['candidate_parent_all_modules_eval'])
        self.assertFalse(c.INITIALIZER_PARITY['candidate_parent_grad_enabled'])
        self.assertFalse(c.INITIALIZER_PARITY['candidate_parent_inference_mode'])
        self.assertFalse(c.GRADIENT_ROUTE['eval_initializer_capture_reused_for_training_forward_parity'])
        self.assertTrue(c.GRADIENT_ROUTE['training_unscoped_candidate_capture_reused_for_forward_parity'])
        self.assertEqual(c.INITIALIZER_PARITY['policy'],'initializer-numerical-contract-v2')
        self.assertTrue(c.INITIALIZER_PARITY['numerical_initializer_gate_changed'])
        self.assertFalse(c.INITIALIZER_PARITY['restoration_quality_gates_changed'])
        self.assertEqual(c.INITIALIZER_PARITY['exact_arrays'],['native','raw','latent','tokens'])
        self.assertEqual((c.INITIALIZER_PARITY['logits_rtol'],c.INITIALIZER_PARITY['logits_atol']),(1.3e-6,1e-5))

    def test_synthetic_protocol_positive_and_role_weight_mutations(self):
        p, pins = protocol_fixture()
        with mock.patch.multiple(c, **pins):
            self.assertIs(c.verify_protocol(p), p)
            for area, field, bad in [('objective', 'normal_clean_supervision', True),
                    ('training', 'resume', True), ('evaluation', 'mean_regression_limit', .02),
                    ('gradient_route', 'witness_update', 1)]:
                changed = deepcopy(p); changed[area][field] = bad
                with self.subTest(area=area, field=field), self.assertRaises(ValueError):
                    c.verify_protocol(changed)
            changed = deepcopy(p); changed['objective']['weights']['behavior_rgb_l1'] = 20.
            with self.assertRaises(ValueError):
                c.verify_protocol(changed)

    def test_flags_and_baseline_cannot_be_relaxed(self):
        p, pins = protocol_fixture()
        with mock.patch.multiple(c, **pins):
            for flag in ('FINAL_used', 'old_optimizer_loaded', 'cached_gradient_reuse', 'tone_head_used'):
                changed = deepcopy(p); changed[flag] = True
                with self.subTest(flag=flag), self.assertRaises(ValueError):
                    c.verify_protocol(changed)
            changed = deepcopy(p); changed['evaluation']['normal_guard_baseline'] = 'candidate_snapshot0'
            with self.assertRaises(ValueError):
                c.verify_protocol(changed)
            changed = deepcopy(p); changed['numerical_initializer_gate_changed'] = False
            with self.assertRaises(ValueError):
                c.verify_protocol(changed)
            changed = deepcopy(p); changed['numerical_initializer_policy'] = 'legacy-exact'
            with self.assertRaises(ValueError):
                c.verify_protocol(changed)
            changed = deepcopy(p); changed['initializer_parity']['logits_atol'] = 1e-4
            with self.assertRaises(ValueError):
                c.verify_protocol(changed)

    def test_historical_metadata_catalog_is_not_full_physical_dependency_set(self):
        p, pins = protocol_fixture()
        self.assertEqual(len(p['historical_dependency_bindings']), 2011)
        self.assertLess(len(p['dependencies']), 2011)
        with mock.patch.multiple(c, **pins):
            changed = deepcopy(p); changed['dependencies'] = deepcopy(changed['historical_dependency_bindings'])
            with self.assertRaises(ValueError):
                c.verify_protocol(changed)
            changed = deepcopy(p); changed['dependencies']['references.json']['sha256'] = '1'*64
            with self.assertRaises(ValueError):
                c.verify_protocol(changed)

    def test_exact_HQ100_and_runtime_bindings(self):
        p, pins = protocol_fixture()
        self.assertEqual(len(p['extra_returned_dependencies']), 304)
        with mock.patch.multiple(c, **pins):
            changed = deepcopy(p); changed['selected_train_reference_ids'][0] = 'val_ffhq_00000'
            with self.assertRaises(ValueError):
                c.verify_protocol(changed)
            changed = deepcopy(p); changed['extra_returned_dependencies'].pop(next(iter(c.FIXED_RETURNED_DEPENDENCIES)))
            with self.assertRaises(ValueError):
                c.verify_protocol(changed)

    def test_static_namespace_current_design_and_payload_caps(self):
        p, pins = protocol_fixture()
        self.assertEqual(hashlib.sha256((ROOT/c.DESIGN_DOCUMENT).read_bytes()).hexdigest(), c.DESIGN_DOCUMENT_SHA256)
        with mock.patch.multiple(c, **pins):
            for name, bad in [('research/__init__.py', '1'*64), (c.DESIGN_DOCUMENT, '1'*64)]:
                changed = deepcopy(p); changed['files_sha256'][name] = bad
                with self.subTest(name=name), self.assertRaises(ValueError):
                    c.verify_protocol(changed)
            changed = deepcopy(p); changed['files_bytes']['data_bundle.json'] = c.LIMITS['source_packet_bytes']
            with self.assertRaises(ValueError):
                c.verify_protocol(changed)

    def test_duplicate_nonfinite_and_path_escapes_refused(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_parent) as directory:
            path = Path(directory)/'x.json'
            for text in ('{"a":1,"a":2}', '{"a":NaN}'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    c.read_json(path)
            for text in ('../x', '/x', 'a\\x', 'a/../x', 'CON.txt', 'a:stream', 'x.'):
                with self.subTest(text=text), self.assertRaises(ValueError):
                    c.safe_path(directory, text, must_exist=False)
            c.write_new_json(path.with_name('new.json'), {'inert': True})
            with self.assertRaises(FileExistsError):
                c.write_new_json(path.with_name('new.json'), {'inert': False})

    def test_hardlinked_file_refused(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_parent) as directory:
            source = Path(directory)/'a'; link = Path(directory)/'b'
            source.write_bytes(b'inert')
            try:
                os.link(source, link)
            except OSError as error:
                self.skipTest(str(error))
            with self.assertRaises(ValueError):
                c.safe_path(directory, 'b')

    def test_windows_guard_precedes_any_root_lookup(self):
        with mock.patch.object(c.platform, 'system', return_value='Windows'), \
                mock.patch.object(c, 'no_links', side_effect=AssertionError('must not inspect root')):
            with self.assertRaisesRegex(ValueError, 'Manual NVIDIA L4'):
                c.manual_root('Z:/not-a-VM')

    def test_checkpoint_slot_sizes_fit_declared_admission_scope(self):
        slots = c.LIMITS['initial_checkpoint_file_bytes'] + c.LIMITS['trained_checkpoint_file_bytes']
        self.assertEqual(slots, 1216*1024**2)
        self.assertEqual(len(c.CHECKPOINT_PATHS), 3)
        self.assertTrue(c.STATE_POLICY['checkpoint50_and_actual_stopped_mutually_exclusive'])
        self.assertTrue(c.STATE_POLICY['checkpoint50_written_only_after_all_snapshot50_calls_complete'])
        self.assertTrue(c.STATE_POLICY['no_new_reservations_RNG_mutations_or_model_operations_after_checkpoint50_before_alias_terminal'])
        # Admission arithmetic is an envelope, not a claimed full output ledger.
        self.assertLessEqual(c.LIMITS['result_payload_bytes']+c.LIMITS['archive_bytes']+
            c.LIMITS['checkpoint_temporary_bytes']+c.LIMITS['reserve_free_bytes'], c.LIMITS['initial_free_bytes'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
