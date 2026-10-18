"""Known-answer frozen factorial/region diagnostics; no neural modules."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('path_isolation_audit',ROOT/'scripts/audit_codeformer_filter_path_isolation_v1.py')
a=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(a)


class PathIsolationMetricFixtures(unittest.TestCase):
    def test_additive_settings_have_zero_signed_pixel_interaction(self):
        base=np.full((9,9,3),.2,np.float64)
        outputs={'w0_adain0':base,'w0_adain1':base+.1,'w1_adain0':base+.2,'w1_adain1':base+.3}
        maps=a.contrast_maps(outputs)
        self.assertTrue(np.allclose(maps['AdaIN_effect_at_w0'],.1,rtol=0,atol=1e-15))
        self.assertTrue(np.allclose(maps['fusion_effect_at_AdaINoff'],.2,rtol=0,atol=1e-15))
        self.assertLess(float(np.max(np.abs(maps['interaction_Y11_Y10_Y01_Y00']))),1e-15)

    def test_interaction_keeps_negative_two_without_clipping(self):
        z=np.zeros((9,9,3),np.float64);one=np.ones_like(z)
        maps=a.contrast_maps({'w0_adain0':z,'w0_adain1':one,'w1_adain0':one,'w1_adain1':z})
        stats=a.signed_map_statistics(maps['interaction_Y11_Y10_Y01_Y00'],np.ones((9,9),bool))
        self.assertEqual(stats['minimum'],-2);self.assertEqual(stats['maximum'],-2)
        self.assertEqual(stats['mean_absolute_RGB'],2);self.assertFalse(stats['clipping_applied'])

    def test_metric_interaction_is_not_metric_of_pixel_interaction(self):
        z=np.zeros((9,9,3),np.float64);q=np.full_like(z,.25)
        outputs={'w0_adain0':z,'w0_adain1':q,'w1_adain0':q,'w1_adain1':q*2}
        r=a.factorial_diagnostics(outputs,z,{'full':np.ones((9,9),bool)})['full']
        pixel=r['setting_contrasts']['interaction_Y11_Y10_Y01_Y00']['signed_pixel_map']
        metric=r['setting_contrasts']['interaction_Y11_Y10_Y01_Y00']['metric_contrast_not_metric_of_interaction_map']
        self.assertEqual(pixel['maximum_absolute_RGB'],0)
        self.assertEqual(metric['mse'],.125)

    def test_exact_local_footprint_inside_outside_and_partial_SSIM_null(self):
        foot=np.zeros((256,256),np.uint8);foot[64:192,64:192]=255
        regions=a.region_masks(a.CASE_IDS[2],foot)
        self.assertEqual([int(v.sum()) for v in regions.values()],[65536,16384,49152])
        clean=np.full((256,256,3),.25,np.float64)
        outputs={s:clean.copy() for s in a.SETTINGS}
        r=a.factorial_diagnostics(outputs,clean,regions)
        self.assertIsNone(r['inside_corruption_footprint']['paired_setting_metrics']['w0_adain0']['ssim'])
        self.assertIsNone(r['outside_corruption_footprint']['paired_setting_metrics']['w0_adain0']['ssim'])
        self.assertEqual(r['full_observed']['paired_setting_metrics']['w0_adain0']['ssim'],1)

    def test_global_profiles_never_gain_an_undeclared_ROI(self):
        full=np.full((256,256),255,np.uint8)
        self.assertEqual(set(a.region_masks(a.CASE_IDS[4],full)),{'full_observed'})
        with self.assertRaisesRegex(ValueError,'Exact local'):
            a.region_masks(a.CASE_IDS[3],full)

    def test_pair_mask_does_not_cross_local_boundary(self):
        values=np.zeros((3,3,3),np.float64);values[1,1]=1
        support=np.zeros((3,3),bool);support[1,1]=True
        stats=a.signed_map_statistics(values,support)
        self.assertEqual(stats['adjacent_x_pairs'],0);self.assertEqual(stats['adjacent_y_pairs'],0)
        self.assertIsNone(stats['adjacent_signed_map_gradient_mae'])

    def test_four_conditions_and_same_geometry_required(self):
        z=np.zeros((9,9,3),np.float64)
        with self.assertRaisesRegex(ValueError,'Exact four'):
            a.contrast_maps({'w0_adain0':z})
        wrong={s:z for s in a.SETTINGS};wrong['w1_adain1']=z[:8]
        with self.assertRaisesRegex(ValueError,'Same fixed'):
            a.contrast_maps(wrong)

    def test_nonfinite_signed_or_normalized_arrays_rejected(self):
        z=np.zeros((9,9,3),np.float64);bad=z.copy();bad[0,0,0]=np.nan
        with self.assertRaises(ValueError):a.signed_map_statistics(bad,np.ones((9,9),bool))
        outputs={s:z for s in a.SETTINGS};outputs['w1_adain1']=bad
        with self.assertRaises(ValueError):a.contrast_maps(outputs)

    def test_global_colour_offset_remains_diagnostic_not_structure_label(self):
        z=np.full((9,9,3),.2,np.float64)
        outputs={s:z.copy() for s in a.SETTINGS};outputs['w1_adain1']=z+.1
        r=a.factorial_diagnostics(outputs,z,{'full':np.ones((9,9),bool)})['full']
        m=r['paired_setting_metrics']['w1_adain1']
        self.assertAlmostEqual(m['global_RGB_offset_energy_share'],1)
        self.assertLess(abs(m['mean_centered_residual_mae']),1e-14)
        self.assertEqual(m['adjacent_RGB_residual_gradient_mae'],0)
        self.assertFalse(r['acceptance_gate_or_optimal_setting_selected'])

    def test_no_neural_module_imported(self):
        self.assertFalse(any(name in sys.modules for name in ('torch','tensorflow','jax')))


def valid_readout():
    tokens=list(range(256))
    return {'logits':{'shape':[1,256,1024],'dtype':'torch.float32','array_sha256':'0'*64},
        'lq_feature':{'shape':[1,256,16,16],'dtype':'torch.float32','array_sha256':'1'*64},
        'recomputed_tokens':{'shape':[1,256],'dtype':'int64',
            'array_sha256':hashlib.sha256(np.asarray(tokens,np.int64).reshape(1,256).tobytes()).hexdigest(),
            'indices':tokens,'rule':'Exact released F.softmax(returned_logits,dim2) then torch.topk(k1,dim2); squeeze trailing singleton',
            'topk_native_shape':[1,256,1],'actual_internal_codebook_lookup_captured':False},
        'top2_probability_margin':{'minimum':.01,'mean':.02,'maximum':.04,'exact_zero_margin_token_count':0},
        'readout_only_no_extra_neural_forward':True}


def valid_observer(setting):
    value=a.SETTING_VALUES[setting];n=1 if value['w']==1 else 0
    return {'top_level_calls':1,'top_level_settings':[{'w':value['w'],'adain':value['adain'],'detach_16':True}],
        'fusion_calls':dict.fromkeys(a.FUSIONS,n),'fusion_order':list(a.FUSIONS) if n else []}


def valid_journal():
    resources={'process_RSS_bytes':1000,'available_RAM_bytes':2*1024**3,'disk_free_bytes':2*1024**3}
    journal=[];evidence={};proofs={};call=0
    for arm in a.ARMS:
        proof={'arm':arm,'test_proof':'fixed'};proofs[arm]=proof
        journal.append({'event':'model_loaded_frozen','arm':arm,'state_sha256':a.MODELS[arm][2],'resources':resources})
        plan=[(a.CASE_IDS[0],'w1_adain1','fresh_clear_parity')]+[
            (key,setting,'new_factorial') for key in a.CASE_IDS for setting in a.SETTINGS[:3]]
        for number,(key,setting,kind) in enumerate(plan):
            call+=1;observer=valid_observer(setting);pin={'path':str(call),'sha256':'2'*64,'bytes':100}
            base={'call':call,'arm':arm,'id':key,'kind':kind}
            journal.append(dict(base,event='forward_reserved',setting=a.SETTING_VALUES[setting],seconds=float(call)))
            journal.append(dict(base,event='forward_returned',setting=setting,observer=observer,resources=resources))
            journal.append(dict(base,event='output_committed',setting=setting,observer=observer,resources=resources,evidence=pin))
            evidence[call]=dict(base,setting=a.SETTING_VALUES[setting],observer=observer,_pin=pin)
            if number==0:
                journal.append({'event':'baseline_parity_passed','arm':arm,'id':a.CASE_IDS[0],'before_interventions':True})
        journal.append({'event':'model_finished_frozen','arm':arm,'proof':proof})
    journal.append({'event':'complete','progress':{'reserved':48,'returned':48,'committed':48,'in_flight':None,'arm':'r2'},
        'resources':resources})
    return journal,evidence,proofs


class SavedArtifactContractFixtures(unittest.TestCase):
    def test_floor_policy_rejects_rounding_instead_of_floor(self):
        raw=np.full((256,256,3),.5,np.float32)
        floor=np.full((256,256,3),127,np.uint8);a.check_floor(raw,floor)
        with self.assertRaisesRegex(ValueError,'floor'):
            a.check_floor(raw,np.full_like(floor,128))

    def test_all_four_hook_conditions_and_no_false_bypass(self):
        for setting in a.SETTINGS:a.observed_setting(valid_observer(setting),setting)
        wrong=valid_observer('w0_adain0');wrong['fusion_calls']['32']=1
        with self.assertRaisesRegex(ValueError,'bypass'):
            a.observed_setting(wrong,'w0_adain0')
        wrong=valid_observer('w1_adain1');wrong['fusion_order'].reverse()
        with self.assertRaisesRegex(ValueError,'order'):
            a.observed_setting(wrong,'w1_adain1')

    def test_hook_boolean_counts_and_changed_detach_rejected(self):
        wrong=valid_observer('w1_adain1');wrong['top_level_calls']=True
        with self.assertRaises(ValueError):a.observed_setting(wrong,'w1_adain1')
        wrong=valid_observer('w0_adain0');wrong['top_level_settings'][0]['detach_16']=False
        with self.assertRaises(ValueError):a.observed_setting(wrong,'w0_adain0')

    def test_exact_token_fingerprint_geometry_and_recomputed_claim(self):
        value=valid_readout();a.readout_signature(value)
        for mutate in ('token','shape','dtype','capture'):
            wrong=copy.deepcopy(value)
            if mutate=='token':wrong['recomputed_tokens']['indices'][0]=1023
            if mutate=='shape':wrong['logits']['shape']=[256,1024]
            if mutate=='dtype':wrong['lq_feature']['dtype']='torch.float16'
            if mutate=='capture':wrong['recomputed_tokens']['actual_internal_codebook_lookup_captured']=True
            with self.assertRaises(ValueError,msg=mutate):a.readout_signature(wrong)

    def test_exact_zero_margin_tie_is_recorded_without_epsilon(self):
        value=valid_readout();value['top2_probability_margin']={'minimum':0.,'mean':.01,'maximum':.03,'exact_zero_margin_token_count':1}
        a.readout_signature(value)
        wrong=copy.deepcopy(value);wrong['top2_probability_margin']['exact_zero_margin_token_count']=0
        with self.assertRaisesRegex(ValueError,'margin'):a.readout_signature(wrong)
        wrong=copy.deepcopy(value);wrong['top2_probability_margin']['mean']=float('nan')
        with self.assertRaises(ValueError):a.readout_signature(wrong)

    def test_full154_event48_call_journal_and100_samples(self):
        journal,evidence,proofs=valid_journal()
        self.assertEqual(len(journal),154)
        self.assertEqual(len(a.validate_journal(journal,evidence,proofs)),100)

    def test_swapped_return_commit_or_missing_call_rejected(self):
        journal,evidence,proofs=valid_journal();journal[2],journal[3]=journal[3],journal[2]
        with self.assertRaisesRegex(ValueError,'Ordered'):a.validate_journal(journal,evidence,proofs)
        journal,evidence,proofs=valid_journal();del evidence[48]
        with self.assertRaisesRegex(ValueError,'48'):a.validate_journal(journal,evidence,proofs)

    def test_false_commit_evidence_and_late_parity_rejected(self):
        journal,evidence,proofs=valid_journal();journal[3]['evidence']={'path':'other','sha256':'2'*64,'bytes':100}
        with self.assertRaisesRegex(ValueError,'join'):a.validate_journal(journal,evidence,proofs)
        journal,evidence,proofs=valid_journal();journal[4]['before_interventions']=False
        with self.assertRaisesRegex(ValueError,'before'):a.validate_journal(journal,evidence,proofs)

    def test_resource_boundary_failure_and_unclosed_inflight_rejected(self):
        journal,evidence,proofs=valid_journal();journal[-1]['resources']=dict(journal[-1]['resources'],available_RAM_bytes=768*1024**2-1)
        with self.assertRaisesRegex(ValueError,'resource'):a.validate_journal(journal,evidence,proofs)
        journal,evidence,proofs=valid_journal();journal[-1]['progress']['in_flight']={'call':48}
        with self.assertRaisesRegex(ValueError,'close'):a.validate_journal(journal,evidence,proofs)

    def test_frozen_helper_prefix_exact_and_no_worker_import(self):
        prefix=ROOT/'outputs/codeformer_filter_path_isolation_v1_preparation/metric_helper_source_at_freeze_v1.py'
        self.assertTrue((ROOT/'scripts/audit_codeformer_filter_path_isolation_v1.py').read_bytes().startswith(prefix.read_bytes()))
        self.assertEqual(hashlib.sha256(prefix.read_bytes()).hexdigest(),a.HELPER_PREFIX_SHA)
        self.assertFalse(any(k in sys.modules for k in ('torch','tensorflow','jax','trusted_previous_frozen_helper')))

    def test_large_inert_stream_pin_checks_bytes_without_checkpoint_load(self):
        with tempfile.TemporaryDirectory(prefix='path-inert-pin-',dir=ROOT/'outputs') as folder:
            folder=Path(folder);self.assertTrue(folder.resolve().is_relative_to(ROOT/'outputs'))
            path=folder/'not-a-checkpoint.bin';payload=b'x'*(9*1024**2);path.write_bytes(payload)
            pin={'path':path.relative_to(ROOT).as_posix(),'absolute_path':str(path),
                'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
            self.assertIsNone(a.workspace_pin(pin,load=False))
            with self.assertRaisesRegex(ValueError,'changed'):
                a.workspace_pin(dict(pin,sha256='0'*64),load=False)
            with self.assertRaisesRegex(ValueError,'inert'):
                a.workspace_pin(pin,load=True)
            with self.assertRaisesRegex(ValueError,'workspace'):
                a.workspace_pin(dict(pin,absolute_path=str(path.with_name('other'))),load=False)
        self.assertFalse(any(k in sys.modules for k in ('torch','tensorflow','jax')))


if __name__=='__main__':unittest.main()
