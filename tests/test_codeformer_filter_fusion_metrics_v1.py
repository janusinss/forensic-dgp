import copy
import importlib.util
from pathlib import Path
import unittest
import numpy as np

P = Path(__file__).resolve().parents[1] / 'research/codeformer_filter_fusion_training_v1/metrics.py'
S = importlib.util.spec_from_file_location('fusion_metrics', P)
M = importlib.util.module_from_spec(S); S.loader.exec_module(M)


def banks():
    out = {}
    for prefix, count, profile, source in [('clear',61,'clear','paired control'), ('blur',53,'blur','FFHQ validation'),
        ('low',53,'lowlight','FFHQ validation'), ('compound',53,'compound','FFHQ validation'),
        ('other',138,'other','other'), ('native',48,'native','camera')]:
        for i in range(count):
            out[prefix+str(i)] = {'source':source, 'profile':profile,
                'evidence':'native_unpaired' if prefix=='native' else 'paired_synthetic',
                'metrics':{v:{'mae':.1,'ssim':.8,'luminance_bias':.1,'coarse_rgb_mae':.1}
                           for v in ('raw','native_png')}}
    return out


def guards():
    return {role + str(i): {'role': role, 'split': 'TRAIN',
        'behavior_distance': {v: 0.0 for v in ('raw', 'native_png')},
        'metrics': {v: {'mae': .1, 'adjacent_rgb_error': .1} for v in ('raw', 'native_png')}}
        for role in ('clear', 'dark', 'filter') for i in range(8)}


class MetricsFixtures(unittest.TestCase):
    def test_native_cannot_have_reference_or_clean_score(self):
        a=np.zeros((256,256,3),np.float32);p=a.astype(np.uint8);s=np.ones((256,256),bool)
        self.assertTrue(all(v is None for row in M.metrics(a,p,None,s,'native_unpaired').values() for v in row.values()))
        with self.assertRaises(ValueError):M.metrics(a,p,a,s,'native_unpaired')

    def test_raw_and_floor_png_are_separate(self):
        a=np.full((256,256,3),.5,np.float32);p=np.floor(a*255).astype(np.uint8);s=np.ones((256,256),bool)
        x=M.metrics(a,p,a,s,'paired_synthetic')
        self.assertEqual(x['raw']['mae'],0);self.assertIsNone(x['raw']['psnr'])
        self.assertGreater(x['native_png']['mae'],0)
        self.assertEqual(x['raw']['ssim'],1)

    def test_partial_support_has_no_invented_regional_ssim(self):
        a=np.zeros((256,256,3),np.float32);s=np.zeros((256,256),bool);s[:10]=True
        x=M.metrics(a,a.astype(np.uint8),a,s,'paired_synthetic')['raw']
        self.assertIsNone(x['ssim']);self.assertIsNone(x['coarse_rgb_mae']);self.assertEqual(x['mae'],0)

    def test_bias_absolute_gate_retains_signed_diagnostic(self):
        a=np.zeros((256,256,3),np.float32);t=np.full_like(a,.25);s=np.ones((256,256),bool)
        r=M.metrics(a,a.astype(np.uint8),t,s,'paired_synthetic')['raw']
        self.assertAlmostEqual(r['luminance_bias'],.25);self.assertAlmostEqual(r['signed_luminance_bias'],-.25)

    def test_legacy_gate_census_and_boundaries(self):
        b=banks();self.assertTrue(M.preservation(b,b)['pass'])
        with self.assertRaises(ValueError):M.preservation(dict(list(b.items())[:-1]),b)
        c=copy.deepcopy(b);c['clear0']['metrics']['raw']['mae']+=.030001
        self.assertFalse(M.preservation(c,b)['pass'])
        self.assertTrue(M.preservation(b,b)['quality_gate_thresholds_changed'] is False)

    def test_inherited_failure_recorded_without_false_qualification(self):
        b=banks();initial=copy.deepcopy(b);initial['clear0']['metrics']['raw']['mae']+=.05
        r=M.interim(initial,b,initial)
        self.assertTrue(r['pass']);self.assertFalse(r['qualification']['pass'])
        worse=copy.deepcopy(initial);worse['clear0']['metrics']['raw']['mae']+=1e-8
        self.assertFalse(M.interim(worse,b,initial)['pass'])

    def test_passing_case_cannot_cross_original_limit(self):
        b=banks();c=copy.deepcopy(b);c['blur0']['metrics']['native_png']['coarse_rgb_mae']+=.031
        self.assertFalse(M.interim(c,b,b)['pass'])

    def test_dark_requires_both_mae_and_ssim(self):
        b=banks();c=copy.deepcopy(b)
        self.assertTrue(M.dark_retention(c,b)['pass'])
        c['low0']['metrics']['native_png']['ssim']-=.001
        self.assertFalse(M.dark_retention(c,b)['pass'])

    def test_filter_does_not_relabel_train_as_dev(self):
        with self.assertRaises(ValueError):M.filter_summary([])

    def test_train_guard_has_no_qualification_claim(self):
        b = guards(); r = M.guard_interim(b, b)
        self.assertTrue(r['pass']); self.assertFalse(r['model_qualified'])
        c = copy.deepcopy(b); c['clear0']['split'] = 'DEV'
        with self.assertRaises(ValueError): M.guard_interim(c, b)

    def test_train_guard_preserves_raw_png_separation(self):
        b = guards(); c = copy.deepcopy(b)
        c['dark0']['behavior_distance']['native_png'] = .031
        self.assertFalse(M.guard_interim(c, b)['pass'])
        self.assertTrue(M.guard_interim(c, b)['groups']['dark']['views']['raw']['behavior_distance']['pass'])

    def test_train_guard_does_not_hide_inherited_behavior_drift(self):
        b = guards(); b['clear0']['behavior_distance']['raw'] = .04
        self.assertTrue(M.guard_interim(b, b)['pass'])
        c = copy.deepcopy(b); c['clear0']['behavior_distance']['raw'] += 1e-8
        self.assertFalse(M.guard_interim(c, b)['pass'])

    def test_train_filter_guard_checks_mean_and_worst(self):
        b = guards(); c = copy.deepcopy(b)
        for k in c:
            if c[k]['role'] == 'filter': c[k]['metrics']['raw']['adjacent_rgb_error'] += .010001
        self.assertFalse(M.guard_interim(c, b)['pass'])
        c = copy.deepcopy(b); c['filter0']['metrics']['native_png']['mae'] += .030001
        self.assertFalse(M.guard_interim(c, b)['pass'])


if __name__=='__main__':unittest.main()
