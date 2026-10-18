"""Independent CPU-only exact-data/features/contract fixtures; no framework imports."""
from copy import deepcopy
import hashlib
import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock
import uuid

import numpy as np
from PIL import Image

from research.codeformer_tone_control_training_v1 import contract as c, data, budget
from scripts import prepare_codeformer_tone_control_training_v1 as builder

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'outputs/codeformer_normal_tone_training_v1_preparation/stage_v1.json'


class ToneDataCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stage = c.read_json(STAGE)
        cls.local = cls.stage['local_dependency_resolution']
        cls.references_document = c.read_json(cls.local['references.json'])
        cls.references = data._validate_references(cls.references_document['references'])
        cls.parity = c.read_json(cls.local['training_parity_v1.json'])
        cls.records = {(r['update'], r['slot']): c.read_json(cls.local[r['pair_record_path']])
                       for r in cls.parity['cases']}
        cls.refsha = cls.stage['dependencies']['references.json']['sha256']
        cls.oldroot = ROOT / 'outputs/codeformer_adain_gradient_isolation_v1_return'
        imported = c.read_json(ROOT / 'outputs/codeformer_adain_gradient_isolation_v1_return_review/transport_v1.json')
        manifest = c.read_json(cls.oldroot / imported['manifest_member'])
        cls.retained = {name: {'sha256': digest, 'bytes': (cls.oldroot / name).stat().st_size}
            for name, digest in manifest['files_sha256'].items()
            if any(name.startswith('outputs/adain_gradient_isolation_v1/snapshot_' + u + '/')
                   for u in ('0000', '0050')) and name.endswith(('/raw.npy', '/native.png'))}

    def validate(self, document=None, records=None, refs=None):
        records = self.records if records is None else records
        return data.validate_training_data(self.parity if document is None else document,
            self.references if refs is None else refs,
            lambda row: records[(row['update'], row['slot'])], self.refsha)

    def test_real200_data_order_and_roles_without_old_optimizer_recipe(self):
        rows, updates = self.validate()
        self.assertEqual(len(rows), 200)
        self.assertEqual(len(updates), 50)
        # Historical LR/callcap metadata must not constrain the distinct new head recipe.
        altered = deepcopy(self.parity)
        for row in altered['updates']:
            row['group_lrs'] = {'irrelevant_historical_optimizer': 123}
        altered['call_caps'] = {'not_a_new_controller_constraint': -1}
        self.assertEqual(len(self.validate(altered)[0]), 200)

    def test_role_order_source_and_stale_pixel_joins_refused(self):
        for field, value in [('source_path', 'data/sources/other.png'), ('role', 'validation'),
                             ('member', 'dark'), ('slot', 2), ('reference_manifest_sha256', '0' * 64),
                             ('input256_float32_sha256', '0' * 64)]:
            changed = deepcopy(self.records)
            changed[(1, 1)][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(records=changed)
        altered = deepcopy(self.parity)
        altered['cases'][0]['update'] = True
        with self.assertRaises(ValueError):
            self.validate(altered)
        altered = deepcopy(self.parity)
        altered['cases'][0], altered['cases'][1] = altered['cases'][1], altered['cases'][0]
        with self.assertRaises(ValueError):
            self.validate(altered)

    def test_TRAIN_DEV_reference_overlap_and_unapproved_roles_refused(self):
        rows = deepcopy(self.references_document['references'])
        tr = next(row for row in rows if row['historical_role'] == 'train')
        va = next(row for row in rows if row['historical_role'] == 'validation')
        va['source_sha256'] = tr['source_sha256']
        with self.assertRaises(ValueError):
            data._validate_references(rows)
        rows = deepcopy(self.references_document['references'])
        rows[0]['historical_role'] = 'final'
        with self.assertRaises(ValueError):
            data._validate_references(rows)

    def test_support_normalized_features_plane_order_empty_cells_and_padding(self):
        x = np.zeros((256, 256, 3), np.float32)
        mask = np.zeros((256, 256), bool)
        mask[:16, :32] = True
        x[:16, :32] = [.125, .5, .75]
        f = data.features(x, mask).reshape(4, 8, 8)
        np.testing.assert_array_equal(f[:, 0, 0], np.asarray([.125, .5, .75, .5], np.float32))
        np.testing.assert_array_equal(f[:, 0, 1], np.zeros(4, np.float32))
        x[~mask] = .9375
        np.testing.assert_array_equal(data.features(x, mask), f.reshape(256))
        self.assertEqual(f.dtype, np.float32)
        self.assertTrue(f.flags.c_contiguous)

    def test_feature_refusals_and_mask_binary_geometry(self):
        x = np.full((256, 256, 3), .5, np.float32)
        for mask in (np.zeros((256, 256), bool), np.ones((256, 256), np.uint8), np.ones((128, 128), bool)):
            with self.assertRaises(ValueError):
                data.features(x, mask)
        for value in (float('nan'), 1.01, -.01):
            bad = x.copy()
            bad[0, 0, 0] = value
            with self.assertRaises(ValueError):
                data.features(bad, np.ones((256, 256), bool))
        folder = ROOT / 'scratch' / ('tone_data_mask_' + uuid.uuid4().hex)
        folder.mkdir(parents=True)
        for value in (0, 1):
            path = folder / ('mask%d.png' % value)
            Image.fromarray(np.full((256, 256), value, np.uint8)).save(path)
            with self.assertRaises(ValueError):
                data.load_mask(path)

    def test_retained_RGB8_division_matches_all200_original_tensor_hashes(self):
        for row in self.parity['cases']:
            pixels = data.load_rgb(self.local[row['input_path']])
            floats = data.input_float32(pixels)
            self.assertEqual(data.pixel_sha256(pixels), row['input256_rgb8_sha256'])
            self.assertEqual(data.pixel_sha256(floats), row['input256_float32_sha256'])
            self.assertTrue(floats.flags.c_contiguous)

    def test_root_and_path_refusals_precede_framework_import(self):
        before = set(sys.modules)
        with mock.patch.object(c.platform, 'system', return_value='Windows'), self.assertRaises(ValueError):
            c.manual_root(ROOT)
        with self.assertRaises(ValueError):
            c.safe_path(ROOT, '../escape')
        with self.assertRaises(ValueError):
            c.safe_path(ROOT, 'C:/escape')
        with mock.patch.object(c.platform, 'system', return_value='Linux'), self.assertRaises(ValueError):
            c.manual_root(ROOT)
        self.assertNotIn('torch', set(sys.modules) - before)

    def test_design_counter_asset_and_type_mutations_refused(self):
        self.assertEqual(sum(c.COUNTS.values()), 3090)
        self.assertEqual(c.CENSUS['all_neural_forward_calls'], 2638)
        self.assertEqual(c.DESIGN['controller']['parameter_count'], 8257)
        self.assertEqual(c.ASSETS['base']['sha256'], '43c0f313365e13849ddcd852f401511e4c7ae52cbac55fdd98a6b032f06ef37b')
        # Protocol false flags must be actual bools, not JSON integer aliases.
        self.assertFalse(c.equal({'flag': False}, {'flag': 0}))
        for name in ('final/../escape', 'a\\b', 'a//b', 'CON.txt', 'x:drive', 'a.'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                c.safe_relative(name)

    def protocol_fixture(self):
        files = {name: hashlib.sha256(b'opaque source fixture never executed').hexdigest()
                 for name in c.REQUIRED_SOURCE_FILES}
        sizes = dict.fromkeys(files, 42)
        files['research/__init__.py'] = hashlib.sha256(b'').hexdigest()
        sizes['research/__init__.py'] = 0
        files['research/codeformer_tone_control_training_v1/design.json'] = c.DESIGN_SHA256
        sizes['research/codeformer_tone_control_training_v1/design.json'] = c.DESIGN_BYTES
        doc = 'instructions/CCTV_CODEFORMER_TONE_CONTROL_TRAINING_V1_DESIGN.md'
        files[doc], sizes[doc] = hashlib.sha256(b'design fixture').hexdigest(), 14
        return {'format': c.FORMAT, 'module': c.MODULE, 'parent_dir': c.PARENT_NAME,
            'parent_protocol_sha256': c.PARENT_PROTOCOL_SHA, 'outputs': c.OUTPUTS, 'export_prefix': c.PREFIX,
            'design': deepcopy(c.DESIGN), 'design_sha256': c.DESIGN_SHA256, 'call_caps': deepcopy(c.COUNTS),
            'limits': deepcopy(c.LIMITS), 'assets': deepcopy(c.ASSETS), **deepcopy(c.PARENT_CONTROLS),
            'data_manifest': 'references.json', 'comparison_cases': 'comparison_cases.json',
            'historical_parent_protocol': c.read_json(self.local['protocol.json']),
            'dependencies': deepcopy(self.stage['dependencies']), 'retained_output_dependencies': deepcopy(self.retained),
            'parent_parity_outputs': c.parent_parity_paths(), 'files_sha256': files, 'files_bytes': sizes,
            'design_document': {'path': doc, 'sha256': files[doc]}, 'status': 'PREPARED_UNRUN',
            'manual_tmux_required': True, 'new_training_recipe': True, 'actual_candidate_checkpoints_required': True,
            **dict.fromkeys(('automatic_training', 'automatic_resume', 'model_qualified', 'automatic_app_promotion',
                'quality_gates_changed', 'FINAL_used', 'old_optimizer_loaded', 'cached_gradient_reuse'), False)}

    def test_exact_protocol_mutations_refuse_prior_recipe_stale_sources_and_counts(self):
        nominal = self.protocol_fixture()
        self.assertIs(c.verify_protocol(nominal), nominal)
        for target, key, value in [('call_caps', 'head_backward_calls', 201),
            ('limits', 'process_RSS_bytes', c.LIMITS['process_RSS_bytes'] + 1),
            ('design', 'namespace', 'old_recipe'), ('files_sha256', 'research/__init__.py', '0' * 64),
            ('historical_parent_protocol', 'format', 'mutated_old_provenance')]:
            altered = deepcopy(nominal)
            altered[target][key] = value
            with self.subTest(target=target, key=key), self.assertRaises(ValueError):
                c.verify_protocol(altered)
        for key, value in [('manual_tmux_required', 1), ('quality_gates_changed', 0),
                           ('automatic_training', True), ('design_sha256', '0' * 64)]:
            altered = deepcopy(nominal)
            altered[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                c.verify_protocol(altered)
        altered = deepcopy(nominal)
        first = next(iter(altered['retained_output_dependencies']))
        altered['retained_output_dependencies'][first]['bytes'] += 1
        with self.assertRaises(ValueError):
            c.verify_protocol(altered)

    def test_builder_source_only_synthetic_stage_and_refuses_existing_stage(self):
        # This isolated fixture creates source text only, never the real final stage.
        root = ROOT / 'scratch' / ('tone_builder_fixture_' + uuid.uuid4().hex)
        (root / 'outputs').mkdir(parents=True)
        for name in c.REQUIRED_SOURCE_FILES - {'research/__init__.py'}:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / name).read_bytes() if name.endswith('/design.json') else
                             b'raise RuntimeError("opaque source fixture; never execute")\n')
        # A nonempty local initializer is provenance only. Executing this text
        # would fail the fixture; the staged transport initializer must be empty.
        local_init = b'raise RuntimeError("local research initializer must never be imported by staging")\n'
        (root / 'research/__init__.py').write_bytes(local_init)
        for suffix in ('data', 'model', 'transport', 'worker'):
            path = root / ('tests/test_codeformer_tone_control_' + suffix + '_v1.py')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'# opaque test source only\n')
        for name in ('AGENTS.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
                     'PROJECT_HANDOFF.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md',
                     'CCTV_CODEFORMER_TONE_CONTROL_TRAINING_V1_DESIGN.md'):
            (root / name).write_bytes(b'bounded synthetic document\n')
        stage, receipt = root / 'outputs' / c.VM_NAME, root / 'outputs/fixture_receipt.json'
        with mock.patch.object(builder, 'ROOT', root), mock.patch.object(builder, 'dependency_bindings',
            return_value=(self.stage['dependencies'], self.retained, {}, c.read_json(self.local['protocol.json']))):
            result = builder.prepare(stage, receipt)
            self.assertTrue(result['stage_only_not_sealed'])
            c.verify_protocol(c.read_json(stage / 'protocol.json'))
            self.assertEqual((root / 'research/__init__.py').read_bytes(), local_init)
            self.assertEqual((stage / 'research/__init__.py').read_bytes(), b'')
            stage_receipt = c.read_json(receipt)
            self.assertIs(stage_receipt['current_sources']['research/__init__.py'], False)
            synthetic = stage_receipt['synthetic_sources']['research/__init__.py']
            self.assertEqual(synthetic['local_reference_sha256'], hashlib.sha256(local_init).hexdigest())
            self.assertEqual(synthetic['local_reference_bytes'], len(local_init))
            self.assertEqual(synthetic['packet_sha256'], hashlib.sha256(b'').hexdigest())
            self.assertEqual(synthetic['packet_bytes'], 0)
            self.assertIs(synthetic['copied_or_executed'], False)
            self.assertFalse(any(p.suffix in ('.pt', '.pth', '.npy', '.npz', '.png', '.gz') for p in stage.rglob('*')))
            with self.assertRaises(ValueError):
                builder.prepare(stage, root / 'outputs/other_receipt.json')

    def test_source_resource_ledger_fits_payload_files_archive_and_disk_without_measurement_claims(self):
        row = budget.resource_ledger()
        self.assertTrue(row['complete'])
        self.assertFalse(row['measured_peak_resources'])
        self.assertEqual(row['operation_reservations'], 3090)
        self.assertEqual(row['full_neural_forwards'], 2638)
        self.assertEqual(row['file_categories']['TRAIN_metadata_records'], 200)
        self.assertEqual(row['file_categories']['history_attempt_and_committed_records'], 100)
        self.assertEqual(row['regular_files_upper'], 8617)
        self.assertLess(row['payload_upper_bytes'], c.LIMITS['result_payload_bytes'])
        self.assertLess(row['regular_files_upper'], 9000)
        self.assertLess(row['archive']['default_zlib_gzip_upper_bytes'], c.LIMITS['archive_bytes'])
        self.assertGreater(row['disk']['admission_margin_bytes'], 0)
        self.assertEqual(row['units']['RGB256_NPY_bound_bytes'], 786688)
        self.assertLessEqual(row['prerequisite_source_detail']['all25files_upper_bytes'], 2 * 1024 ** 2)
        for key in ('result_payload_bytes', 'regular_files', 'archive_bytes', 'initial_free_bytes_after_install'):
            altered = dict(c.LIMITS)
            altered[key] = 1
            with self.subTest(key=key), mock.patch.object(c, 'LIMITS', altered), self.assertRaises(ValueError):
                budget.resource_ledger()
        import random
        for length in (0, 1, 16383, 65536):
            payload = random.Random(length).randbytes(length)
            self.assertLessEqual(len(gzip.compress(payload, mtime=0)), budget.gzip_upper_bound(length))

    def test_checkpoint_slot_budget_preserves_partial_without_full_duplicate(self):
        a, b, stopped = c.CHECKPOINT_PATHS
        cap = c.DESIGN['state_and_return']['checkpoint_write_cap_bytes_each']
        partial = b + '.tmp-' + 'a' * 32
        row = budget.checkpoint_slot_accounting({a: cap, partial: cap, stopped: cap})
        self.assertEqual(row['physical_slot_files'], 3)
        self.assertEqual(row['slot_bytes'], 3 * cap)
        self.assertEqual(row['temporary_bytes'], cap)
        for files in ({b: cap, partial: 1}, {a: cap + 1}, {c.OUTPUTS + '/other.pt': 1},
                      {partial: -1}, {b + '.tmp-nothex': 1}):
            with self.subTest(files=files), self.assertRaises(ValueError):
                budget.checkpoint_slot_accounting(files)

    def test_full_real_index_all406_masks_targets_inputs_and_retained_outputs(self):
        p = {'dependencies': self.stage['dependencies'], 'retained_output_dependencies': self.retained,
             'data_manifest': 'references.json', 'comparison_cases': 'comparison_cases.json', **c.PARENT_CONTROLS}
        def resolve(name):
            return self.local[name] if name in self.local else self.oldroot / name
        index = data.load_index(self.oldroot, p, resolver=resolve)
        self.assertEqual((len(index['train']), len(index['dev'])), (200, 406))
        self.assertTrue(all(row['target'] is None and row['observed'].all() for row in index['train']))
        self.assertEqual(sum(row['target'] is None for row in index['dev']), 48)
        comparisons = c.read_json(self.local['comparison_cases.json'])['cases']
        expected_partial = sum(not data.load_mask(self.local[row['observed_path']]).all() for row in comparisons)
        self.assertEqual(sum(not row['observed'].all() for row in index['dev']), expected_partial)
        self.assertGreater(expected_partial, 0)
        self.assertEqual(sum(len(v) * 2 for v in index['retained_outputs'].values()), 1624)
        self.assertTrue(index['binding_receipt']['complete'])


if __name__ == '__main__':
    unittest.main()
