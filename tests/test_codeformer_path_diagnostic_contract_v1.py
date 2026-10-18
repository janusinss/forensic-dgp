"""Owned synthetic CPU transfer fixtures; no real model/gradient/VM calls."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

import numpy as np
from PIL import Image

from research.codeformer_path_diagnostic_v1 import contract as c

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('diagnostic_preparer_fixture', ROOT / 'scripts/prepare_codeformer_path_diagnostic_v1.py')
PREPARER = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(PREPARER)


def sha_pixels(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class DiagnosticContractTests(unittest.TestCase):
    def setUp(self):
        self.parent = (ROOT / 'scratch').resolve(); self.parent.mkdir(exist_ok=True)
        self.workspace = self.parent / ('path_diagnostic_contract_' + uuid.uuid4().hex)
        self.workspace.mkdir()
        self.root = self.workspace / c.PACKAGE_DIR; self.root.mkdir()
        self.external = self.workspace / c.PARENT_DIR; self.external.mkdir()
        self.assets = {}
        self.parent_files = {}
        for name, original in c.ASSETS.items():
            relative = original['path']; path = self.external / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = ('owned nonmodel external fixture ' + name).encode(); path.write_bytes(payload)
            self.assets[name] = {'path': relative, 'sha256': c.sha256(path), 'bytes': len(payload)}
            self.parent_files[relative] = c.sha256(path)
        (self.external / 'protocol.json').write_text('{"owned_parent_fixture":true}\n')
        self.parent_pin = c.sha256(self.external / 'protocol.json')
        self.spec = deepcopy(c.SPEC); self.spec['assets'] = self.assets
        self.spec['parent_protocol_sha256'] = self.parent_pin
        self.patches = [patch.object(c, 'ASSETS', self.assets), patch.object(c, 'SPEC', self.spec),
                        patch.object(c, 'PARENT_PROTOCOL_SHA256', self.parent_pin)]
        for p in self.patches:
            p.start(); self.addCleanup(p.stop)
        files = {}
        def put(relative, payload):
            p = self.root / relative; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(payload)
            files[relative] = c.sha256(p); return relative
        put('instructions/DIAGNOSTIC_DESIGN.md', b'Owned finite zero-update fixture\n')
        put('NOTICE.md', b'Owned fixture source notice\n')
        for relative in c.REQUIRED_SOURCES:
            put(relative, b'"""Owned non-executable fixture source."""\n')
        target = np.zeros((512, 512, 3), np.float32)
        clean_by_id = {}
        for rid in sorted(set(c.REFERENCE_IDS)):
            relative = f'data/clean512/{rid}.npy'; path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True); np.save(path, target)
            files[relative] = c.sha256(path); clean_by_id[rid] = relative
        cases = []
        for index, ((update, slot), rid) in enumerate(zip(c.POSITIONS, c.REFERENCE_IDS)):
            identifier = c.case_id(update, slot)
            inp = f'data/cases/{identifier}/input.png'; path = self.root / inp
            path.parent.mkdir(parents=True, exist_ok=True)
            pixels = np.full((256, 256, 3), 90 + index, np.uint8); Image.fromarray(pixels).save(path)
            files[inp] = c.sha256(path)
            pair = {'role': 'train', 'reference_id': rid, 'update': update, 'slot': slot,
                'member': 'normal' if slot % 2 else 'dark', 'pair_role': 'A' if slot <= 2 else 'B',
                'saved_input_sha256': files[inp], 'input256_rgb8_sha256': sha_pixels(pixels),
                'input256_float32_sha256': sha_pixels(pixels.astype(np.float32) / np.float32(255)),
                'clean512_float32_sha256': sha_pixels(target),
                'parameters': {'image_eligible': update == 24 or slot >= 3}}
            record = put(f'data/cases/{identifier}/pair_record.json', json.dumps(pair).encode())
            row = {'id': identifier, 'update': update, 'slot': slot, 'reference_id': rid,
                'image_eligible': pair['parameters']['image_eligible'], 'input_path': inp, 'input_sha256': files[inp],
                'clean512_path': clean_by_id[rid], 'clean512_sha256': files[clean_by_id[rid]],
                'pair_record_path': record, 'pair_record_sha256': files[record], 'behavior': None}
            if slot == 3:
                behavior = {}
                for label, shape in [('delivered', (1, 3, 256, 256)), ('logits', (1, 256, 1024))]:
                    relative = f'data/behavior/update_{update:04d}/{label}.npy'; path = self.root / relative
                    path.parent.mkdir(parents=True, exist_ok=True); np.save(path, np.zeros(shape, np.float32))
                    files[relative] = c.sha256(path); behavior[label + '_path'] = relative; behavior[label + '_sha256'] = files[relative]
                row['behavior'] = behavior
            cases.append(row)
        self.protocol = deepcopy(self.spec)
        self.protocol.update(files_sha256=files, parent_files_sha256=self.parent_files, cases=cases,
            instruction={'path': 'instructions/DIAGNOSTIC_DESIGN.md', 'sha256': files['instructions/DIAGNOSTIC_DESIGN.md']},
            export_files=['instructions/DIAGNOSTIC_DESIGN.md', 'NOTICE.md'],
            optimizer_updates=0, no_automatic_training=True)
        self.pin = self.save_protocol()

    def tearDown(self):
        resolved = self.workspace.resolve()
        self.assertEqual(resolved.parent, self.parent)
        self.assertTrue(resolved.name.startswith('path_diagnostic_contract_'))
        shutil.rmtree(resolved)

    def save_protocol(self):
        (self.root / 'protocol.json').write_text(json.dumps(self.protocol), encoding='utf-8')
        return c.sha256(self.root / 'protocol.json')

    def test_exact_transfer_and_external_parent_hashes(self):
        before = {p.relative_to(self.external).as_posix(): c.sha256(p) for p in self.external.rglob('*') if p.is_file()}
        result = c.verify_transfer(self.root, self.pin)
        self.assertEqual(result, self.protocol)
        self.assertEqual(before, {p.relative_to(self.external).as_posix(): c.sha256(p) for p in self.external.rglob('*') if p.is_file()})
        self.assertEqual(result['call_caps']['optimizer_updates'], 0)

    def test_protocol_mutation_caps_roles_and_order_refused(self):
        for change in ('optimizer', 'caps', 'case_order', 'role', 'parent'):
            changed = deepcopy(self.protocol)
            if change == 'optimizer': changed['optimizer_updates'] = 1
            elif change == 'caps': changed['call_caps']['gradient_queries'] = 65
            elif change == 'case_order': changed['cases'].reverse()
            elif change == 'role': changed['cases'][0]['reference_id'] = 'va_ffhq_16572'
            elif change == 'parent': changed['parent_dir'] = '../other'
            with self.subTest(change=change), self.assertRaises(ValueError):
                c.validate_protocol(changed)

    def test_actual_input_pixel_mismatch_refused_even_with_updated_file_hash(self):
        row = self.protocol['cases'][0]; path = self.root / row['input_path']
        Image.fromarray(np.full((256, 256, 3), 199, np.uint8)).save(path)
        row['input_sha256'] = self.protocol['files_sha256'][row['input_path']] = c.sha256(path)
        record_path = self.root / row['pair_record_path']; pair = c.read_json(record_path)
        pair['saved_input_sha256'] = row['input_sha256']; record_path.write_text(json.dumps(pair))
        row['pair_record_sha256'] = self.protocol['files_sha256'][row['pair_record_path']] = c.sha256(record_path)
        with self.assertRaisesRegex(ValueError, 'exact V2 TRAIN float32 tensor'):
            c.verify_transfer(self.root, self.save_protocol(), verify_parent=False)

    def test_clean_target_pixel_mismatch_and_trailing_bytes_refused(self):
        row = self.protocol['cases'][0]; path = self.root / row['clean512_path']
        np.save(path, np.full((512, 512, 3), .25, np.float32))
        for case in self.protocol['cases']:
            if case['clean512_path'] == row['clean512_path']:
                case['clean512_sha256'] = c.sha256(path)
        self.protocol['files_sha256'][row['clean512_path']] = c.sha256(path)
        with self.assertRaisesRegex(ValueError, 'exact saved TRAIN target'):
            c.verify_transfer(self.root, self.save_protocol(), verify_parent=False)
        with path.open('ab') as f: f.write(b'junk')
        for case in self.protocol['cases']:
            if case['clean512_path'] == row['clean512_path']: case['clean512_sha256'] = c.sha256(path)
        self.protocol['files_sha256'][row['clean512_path']] = c.sha256(path)
        with self.assertRaisesRegex(ValueError, 'trailing clean-target'):
            c.verify_transfer(self.root, self.save_protocol(), verify_parent=False)

    def test_parent_asset_and_protocol_mutation_refused(self):
        path = self.external / self.assets['stopped_checkpoint']['path']; path.write_bytes(b'changed checkpoint fixture')
        with self.assertRaisesRegex(ValueError, 'External parent dependency differs'):
            c.verify_transfer(self.root, self.pin)
        (self.external / 'protocol.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Parent protocol SHA256 differs'):
            c.verify_transfer(self.root, self.pin)

    def test_unlisted_file_traversal_reserved_names_and_duplicate_json_refused(self):
        (self.root / 'unlisted.py').write_text('raise AssertionError("must never run")')
        with self.assertRaisesRegex(ValueError, 'namespace differs'):
            c.verify_transfer(self.root, self.pin, verify_parent=False)
        for relative in ('../escape', 'data/a/../b', 'C:/outside', 'data/CON.txt', 'data/a.', 'a\\b'):
            with self.subTest(path=relative), self.assertRaises(ValueError): c.bounded(self.root, relative, must_exist=False)
        (self.root / 'dup.json').write_text('{"a":1,"a":2}')
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON'):
            c.read_json(self.root / 'dup.json')

    def test_behavior_geometry_and_nonfinite_refused(self):
        row = self.protocol['cases'][2]; path = self.root / row['behavior']['logits_path']
        np.save(path, np.full((1, 256, 1024), np.nan, np.float32))
        row['behavior']['logits_sha256'] = self.protocol['files_sha256'][row['behavior']['logits_path']] = c.sha256(path)
        with self.assertRaisesRegex(ValueError, 'anchor dtype/geometry/nonfinite'):
            c.verify_transfer(self.root, self.save_protocol(), verify_parent=False)

    def test_write_new_receipt_never_replaces_existing(self):
        path = self.root / 'owned.json'; c.write_new_json(path, {'complete': True})
        with self.assertRaises(FileExistsError): c.write_new_json(path, {'complete': False})
        self.assertTrue(c.read_json(path)['complete'])

    def test_preparer_missing_prerequisite_stops_before_staging(self):
        # This is an independent empty synthetic workspace, never the real stage.
        target = self.workspace / 'empty_owned_workspace'; target.mkdir()
        with self.assertRaises(ValueError): PREPARER.stage_inputs(target)
        self.assertFalse((target / 'outputs' / c.PACKAGE_DIR).exists())
        self.assertFalse((target / PREPARER.EVIDENCE).exists())

    def test_seal_requires_reviewed_sources_before_any_payload_change(self):
        readiness = self.workspace / 'not_approved.json'
        readiness.write_text('{"complete":true,"approved_for_protocol_freeze":false}')
        before = c.sha256(readiness)
        with self.assertRaisesRegex(ValueError, 'reviewed-source approval'):
            PREPARER.seal_sources(self.workspace, 'not_approved.json', before, 'absent.md', '0' * 64)
        self.assertEqual(c.sha256(readiness), before)

    def test_all_seal_sources_and_instruction_are_preflighted_before_copy(self):
        # A late missing source or stale instruction must not leave half-sealed
        # executable sources beside the previously approved input stage.
        for scenario in ('late_missing_source', 'stale_instruction', 'unreviewed_design'):
            with self.subTest(scenario=scenario):
                target = self.workspace / scenario; target.mkdir()
                packet = target / 'outputs' / c.PACKAGE_DIR; packet.mkdir(parents=True)
                (packet / 'input.fixture').write_bytes(b'owned unchanged staged bytes')
                staged = {'complete': True, 'files_sha256': {'input.fixture': c.sha256(packet / 'input.fixture')}}
                c.write_new_json(target / PREPARER.EVIDENCE / 'stage_inputs.json', staged)
                sources = {}
                for index, relative in enumerate(c.REQUIRED_SOURCES):
                    source = target / relative; source.parent.mkdir(parents=True, exist_ok=True)
                    payload = b'"""Owned seal preflight fixture, never executed."""\n'
                    sources[relative] = hashlib.sha256(payload).hexdigest()
                    if scenario != 'late_missing_source' or index < len(c.REQUIRED_SOURCES) - 1:
                        source.write_bytes(payload)
                instruction = target / 'instruction.md'; instruction.write_bytes(b'Owned design fixture\n')
                reviewed = {'complete': True, 'approved_for_protocol_freeze': True,
                            'design_sha256': ('f' * 64 if scenario == 'unreviewed_design' else c.sha256(instruction)),
                            'training_authorized': False, 'source_sha256': sources}
                c.write_new_json(target / 'approved.json', reviewed)
                instruction_pin = '0' * 64 if scenario == 'stale_instruction' else c.sha256(instruction)
                with self.assertRaises((ValueError, OSError)):
                    PREPARER.seal_sources(target, 'approved.json', c.sha256(target / 'approved.json'),
                                          'instruction.md', instruction_pin)
                self.assertEqual(PREPARER.exact_files(packet), {'input.fixture'})
                self.assertEqual(c.sha256(packet / 'input.fixture'), staged['files_sha256']['input.fixture'])


if __name__ == '__main__':
    unittest.main()
