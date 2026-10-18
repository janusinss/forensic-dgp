"""Finite CPU fixtures for the final personal-visual evidence boundary."""
from copy import deepcopy
from pathlib import Path
import unittest
import uuid

from scripts.close_codeformer_adain_return_chain_v1 import visual_contract

ROOT = Path(__file__).resolve().parents[1]


class ClosureVisualBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.folder = ROOT / 'scratch' / ('adain-closure-boundary-' + uuid.uuid4().hex)
        self.folder.mkdir(parents=True)
        self.pages = []
        for index in (1, 2):
            path = self.folder / ('page%d.png' % index)
            path.write_bytes(b'opaque metadata fixture - no image decoding here')
            self.pages.append({'file': path.name, 'sha256': str(index) * 64, 'case_ids': ['case%d' % index]})
        self.manifest = {'pages': self.pages}
        self.visual = {'complete': True, 'root_personally_viewed_all_planned_sheets': True,
            'model_qualified': False, 'sheets': [{'path': str(self.folder / page['file']),
                'sha256': page['sha256'], 'case_ids': page['case_ids'], 'root_personally_viewed': True}
                for page in self.pages]}

    def test_exact_personal_page_record_is_accepted_without_model_qualification(self):
        self.assertEqual(visual_contract(ROOT, self.folder, self.manifest, self.visual), 2)

    def test_missing_duplicate_or_wrong_case_pages_fail(self):
        variants = [dict(self.visual, sheets=self.visual['sheets'][:1]),
                    dict(self.visual, sheets=[self.visual['sheets'][0]] * 2)]
        changed = deepcopy(self.visual); changed['sheets'][0]['case_ids'] = ['other']; variants.append(changed)
        for variant in variants:
            with self.assertRaises(ValueError): visual_contract(ROOT, self.folder, self.manifest, variant)

    def test_personal_review_flags_and_no_qualification_are_required(self):
        for field, value in (('complete', False), ('root_personally_viewed_all_planned_sheets', False), ('model_qualified', True)):
            changed = deepcopy(self.visual); changed[field] = value
            with self.assertRaises(ValueError): visual_contract(ROOT, self.folder, self.manifest, changed)
        changed = deepcopy(self.visual); changed['sheets'][0]['root_personally_viewed'] = False
        with self.assertRaises(ValueError): visual_contract(ROOT, self.folder, self.manifest, changed)

    def test_image_hash_and_exact_planned_namespace_are_required(self):
        changed = deepcopy(self.visual); changed['sheets'][0]['sha256'] = 'f' * 64
        with self.assertRaises(ValueError): visual_contract(ROOT, self.folder, self.manifest, changed)
        outside = self.folder / 'other'; outside.mkdir(); (outside / self.pages[0]['file']).write_bytes(b'other')
        changed = deepcopy(self.visual); changed['sheets'][0]['path'] = str(outside / self.pages[0]['file'])
        with self.assertRaises(ValueError): visual_contract(ROOT, self.folder, self.manifest, changed)


if __name__ == '__main__': unittest.main()
