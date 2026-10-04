"""Role, schedule and actual local-training rejection for the new VM diagnostic."""
import importlib.util
from pathlib import Path
import unittest

import cctv_dgp_face_code_fit_v12 as f


def fixture():
    refs = [{'id': f'{source}_{i}', 'source': source, 'role': 'train'} for source in f.SOURCES for i in range(5)]
    cases = [{'id': ref['id'] + '_' + profile, 'reference_id': ref['id'], 'source': ref['source'], 'profile': profile}
             for ref in refs for profile in f.PROFILES]
    return {'references': refs, 'cases': cases, 'native_used': False, 'validation_used': False,
            'native_reserved_used': False, 'production_promotion': False}


class Boundaries(unittest.TestCase):
    def test_every_balanced_epoch_uses_each_case_once(self):
        from collections import Counter
        p = fixture(); steps = f.schedule(p); cases = {c['id']: c for c in p['cases']}
        self.assertEqual(len(steps), 300)
        for epoch in range(1, 13):
            rows = [s for s in steps if s['epoch'] == epoch]
            self.assertEqual(len(rows), 25)
            self.assertEqual(Counter(c for s in rows for c in s['case_ids']), Counter(cases.keys()))
            for row in rows:
                self.assertEqual([cases[c]['source'] for c in row['case_ids']], f.SOURCES)
        p['cases'].reverse(); p['references'].reverse()
        self.assertEqual(steps, f.schedule(p))

    def test_validation_role_and_reserved_use_rejected(self):
        p = fixture(); p['references'][0]['role'] = 'validation'
        with self.assertRaisesRegex(ValueError, 'training references'): f.schedule(p)
        p = fixture(); p['native_reserved_used'] = True
        with self.assertRaisesRegex(ValueError, 'scope'): f.schedule(p)

    def test_repeated_profile_and_misassigned_source_rejected(self):
        p = fixture(); p['cases'][0]['profile'] = 'compound_lr24'
        with self.assertRaisesRegex(ValueError, 'profile'): f.schedule(p)
        p = fixture(); p['cases'][0]['source'] = f.SOURCES[1]
        with self.assertRaisesRegex(ValueError, 'source'): f.schedule(p)

    def test_actual_trainer_rejects_local_before_output_or_model(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/train_cctv_dgp_face_code_fit_v12.py'
        spec = importlib.util.spec_from_file_location('face_code_trainer_test', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with self.assertRaisesRegex(RuntimeError, 'only on'): module.train(Path('.').resolve(), '0' * 64)
        self.assertIsNone(module.CONTEXT['out']); self.assertIsNone(module.CONTEXT['model'])
        self.assertEqual(module.CONTEXT['updates'], 0); self.assertEqual(module.CONTEXT['backwards'], 0)


if __name__ == '__main__': unittest.main()
