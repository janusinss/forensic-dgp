"""Independent saved counterfactual compositions, metrics and limited CPU replay."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'outputs/cctv_dgp_bank_comparison_v1_r4_B_return'
PROBE=ROOT/'outputs/cctv_dgp_bank_feature_DC_probe_v1'
ANALYSIS=ROOT/'outputs/cctv_dgp_bank_feature_DC_probe_v1_analysis'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()


def image(path,mode='RGB'):
    with Image.open(path) as item:
        assert item.size==(256,256)
        return np.asarray(item.convert(mode)).copy()


def main():
    start=time.monotonic();receipt=ANALYSIS/'independent_audit.json'
    assert not receipt.exists()
    plan=read(PROBE/'plan.json');result=read(PROBE/'results.json');manifest=read(PROBE/'manifest.json')
    supervisor=read(ROOT/'outputs/cctv_dgp_bank_feature_DC_probe_v1_execution/supervisor_receipt.json')
    assert all(row['complete'] for row in [plan,result,manifest,supervisor])
    assert supervisor['exit_code']==0 and not supervisor['timeout']
    assert result['seconds']<plan['maximum_seconds']==900 and supervisor['external_cap_seconds']==930
    assert result['plan_sha256']==sha(PROBE/'plan.json')
    assert result['worker_sha256']==supervisor['worker_sha256']==sha(ROOT/'scripts/probe_cctv_dgp_bank_feature_DC_v1.py')
    assert {file.relative_to(PROBE).as_posix() for file in PROBE.rglob('*') if file.is_file()}==set(manifest['files_sha256'])|{'manifest.json'}
    for name,expected in manifest['files_sha256'].items():assert sha(PROBE/name)==expected,name
    assert sum(file.stat().st_size for file in PROBE.rglob('*') if file.is_file())<=plan['maximum_output_bytes']
    for name,expected in result['parent_evidence_files_sha256'].items():assert sha(ROOT/name)==expected
    counts=result['counts']
    assert counts['optimizer_updates']==counts['gradient_queries']==counts['backwards']==0
    assert counts['observation_and_bank_batches']==26 and counts['reconstruction_calls']==52 and counts['recognizer_batches']==20
    assert result['model_state_unchanged'] and result['all_weights_and_buffers_frozen'] and result['original_CUDA_outputs_preserved']
    p=read(PARENT/'protocol.json');refs={row['id']:row for row in p['references']}
    paired=[case for case in p['cases'] if case['id'] in p['preview_case_ids']]
    assert len(paired)==100 and len(p['native_development'])==24 and p['reserved_final_pixels']==0
    sys.path.insert(0,str(PARENT))
    from cctv_dgp_head4_lossless_v1 import unpack
    from frozen_capacity_contract import feature_support,exported_pixel_metrics,detail_metric
    from frozen_raw_metrics import pixel_metrics,detail_float,mean_only,review_groups
    variant_rows={};checked=0;native_checked=0
    replay_by_id={row['id']:row for row in result['CPU_CUDA_replay']}
    assert len(replay_by_id)==124
    for label in plan['variants']:
        metrics=read(PROBE/label/'metrics.json');assert metrics['complete']
        assert [row['id'] for row in metrics['rows']]==[case['id'] for case in paired]
        fresh=[]
        for begin in range(0,100,5):
            cs=paired[begin:begin+5];ids=[case['id'] for case in cs]
            original_begin=next(i for i,case in enumerate(p['cases']) if case['id']==cs[0]['id'])
            filename=f'b{original_begin//5:03d}.npz'
            baseline,_=unpack(PARENT/'outputs/bank_comparison_v1/baseline/packs'/filename,ids)
            saved_B,_=unpack(PARENT/'outputs/bank_comparison_v1/B50/packs'/filename,ids,baseline=baseline)
            for i,case in enumerate(cs):
                a=np.load(PROBE/label/'paired'/(case['id']+'.npy'),allow_pickle=False)
                row=metrics['rows'][begin+i]
                camera=image(PARENT/case['input']);target=image(PARENT/refs[case['source_person_or_reference']]['target'])
                mask=image(PARENT/refs[case['source_person_or_reference']]['observed'],'L')>0
                assert a.dtype==np.float32 and a.shape==(256,256,3) and np.isfinite(a).all() and 0<=a.min()<=a.max()<=1
                assert np.array_equal(a[~mask],(camera.astype(np.float32)/np.float32(255))[~mask])
                png=np.floor(a*np.float32(255)).astype(np.uint8);png[~mask]=camera[~mask]
                assert np.array_equal(png,image(PROBE/label/'paired'/(case['id']+'.png')))
                assert hashlib.sha256(a.tobytes()).hexdigest()==row['raw_RGB_float32_sha256']
                assert hashlib.sha256(png.tobytes()).hexdigest()==row['PNG_RGB_sha256']
                support=feature_support(mask,case['landmarks5_canvas_xy'])
                mean_raw,mean_png,shift=mean_only(a,baseline[i],camera,mask)
                rm=pixel_metrics(a,target,mask);pm=exported_pixel_metrics(png,target,mask)
                emb=np.asarray(row['embedding'],dtype=np.float32)
                assert emb.shape==(3,512) and np.isfinite(emb).all() and np.allclose(np.linalg.norm(emb,axis=1),1,rtol=0,atol=1e-5)
                rm.update(ArcFace_observed_fixed=float(emb[0]@emb[2]),landmark_high_frequency_MSE=detail_float(a,target,support),
                    constant_mean_shift_only_MSE=pixel_metrics(mean_raw,target,mask)['MSE'])
                pm.update(ArcFace_observed_fixed=float(emb[1]@emb[2]),landmark_high_frequency_MSE=detail_metric(png,target,support),
                    constant_mean_shift_only_MSE=exported_pixel_metrics(mean_png,target,mask)['MSE'])
                for stage,new in [('raw',rm),('png',pm)]:
                    for key,value in new.items():
                        old=row[stage][key]
                        if value is None or isinstance(value,bool):assert value==old
                        else:assert np.isclose(value,old,rtol=0,atol=2e-7 if key=='ArcFace_observed_fixed' else 1e-13),(label,case['id'],key)
                assert np.allclose(shift,row['postclip_mean_RGB_shift'],rtol=0,atol=1e-14)
                fresh.append({**row,'raw':rm,'png':pm})
                if label==plan['variants'][0]:
                    error=float(np.abs(a-saved_B[i]).max())
                    assert error==replay_by_id[case['id']]['maximum_raw_error']
                    assert np.allclose(a,saved_B[i],rtol=1e-4,atol=1e-4)
                checked+=1
        for stage in ['raw','png']:
            groups=review_groups(fresh,stage)
            assert set(groups)==set(metrics['groups'][stage])
            for group,values in groups.items():
                for key,value in values.items():
                    assert np.isclose(value,metrics['groups'][stage][group][key],rtol=0,atol=2e-7 if key=='ArcFace_observed_fixed' else 1e-13)
        variant_rows[label]=metrics
        assert [row['id'] for row in metrics['native_unpaired_rows']]==[case['id'] for case in p['native_development']]
        for case,row in zip(p['native_development'],metrics['native_unpaired_rows']):
            camera=image(PARENT/case['input']);mask=image(PARENT/case['observed'],'L')>0
            a=np.load(PROBE/label/'native'/(case['id']+'.npy'),allow_pickle=False)
            assert a.dtype==np.float32 and a.shape==(256,256,3) and np.isfinite(a).all() and 0<=a.min()<=a.max()<=1
            assert np.array_equal(a[~mask],(camera.astype(np.float32)/np.float32(255))[~mask])
            png=np.floor(a*np.float32(255)).astype(np.uint8);png[~mask]=camera[~mask]
            assert np.array_equal(png,image(PROBE/label/'native'/(case['id']+'.png')))
            assert hashlib.sha256(a.tobytes()).hexdigest()==row['raw_RGB_float32_sha256']
            assert hashlib.sha256(png.tobytes()).hexdigest()==row['PNG_RGB_sha256']
            assert row['clean_reference'] is None and row['paired_accuracy_not_available']
            if label==plan['variants'][0]:
                old=np.load(PARENT/'outputs/bank_comparison_v1/B50/native'/(case['id']+'.npy'),allow_pickle=False)
                assert float(np.abs(a-old).max())==replay_by_id[case['id']]['maximum_raw_error']
                assert np.allclose(a,old,rtol=1e-4,atol=1e-4)
            native_checked+=1

    # Two independently expressed CPU spot replays: one fixed TRAIN reference
    # per source. Use float64 spatial sums for the centering diagnostic rather
    # than the worker's float32 mean; allow only small forward roundoff.
    import torch
    from cctv_dgp_bank_comparison_v1_model import ConditionedBankDGP
    from cctv_dgp_generative_bank_v1 import load_prior
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_pilot import state_hash,FixedObservedIdentity,grid112
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    frozen,_=load_frozen_dgp_restorer(PARENT/'weights/dgp_v2.pth',expected_sha256=p['original_checkpoint_sha256'])
    prior=load_prior(PARENT/'prior_source',PARENT/'bank_implementation',p['standalone_prior_sha256'])
    model=ConditionedBankDGP(frozen.net,prior,torch.from_numpy(np.load(PARENT/'initializers/mean_style.npy',allow_pickle=False)))
    state=torch.load(PARENT/'outputs/bank_comparison_v1/state_B50/full_state.pth',map_location='cpu',weights_only=True)
    model.load_state_dict(state['model'],strict=True);model.eval().requires_grad_(False)
    identity=FixedObservedIdentity(PARENT/'weights/w600k_r50.onnx','cpu')
    before=state_hash(model);assert all(not item.requires_grad for item in model.parameters())
    selection=[];sources=set()
    for begin in range(0,100,5):
        if paired[begin]['source'] not in sources:
            sources.add(paired[begin]['source']);selection.append(paired[begin:begin+5])
    assert len(selection)==2
    neural=[];recognizer_replays=[]
    with torch.inference_mode():
        for cs in selection:
            x=torch.from_numpy(np.stack([image(PARENT/case['input']) for case in cs]).astype(np.float32)/np.float32(255)).permute(0,3,1,2)
            mm=torch.from_numpy(np.stack([image(PARENT/refs[case['source_person_or_reference']]['observed'],'L')>0 for case in cs]))[:,None]
            maps=model.current.observe(x);features=model.conditioned_features(maps)
            extras={size:model.fusions[size](value) for size,value in features.items()}
            adjusted={size:(value.double()-value.double().sum(dim=(-2,-1),keepdim=True)/(value.shape[-1]*value.shape[-2])).float()
                      for size,value in extras.items()}
            for label,value in zip(plan['variants'],[extras,adjusted]):
                predicted=torch.where(mm,model.current.reconstruct(x,maps,value),x).permute(0,2,3,1).numpy()
                recorded=np.stack([np.load(PROBE/label/'paired'/(case['id']+'.npy'),allow_pickle=False) for case in cs])
                assert np.allclose(predicted,recorded,rtol=0,atol=1e-5),'Independent centered-feature replay differs'
                neural.append(dict(label=label,source=cs[0]['source'],reference=cs[0]['reference_id'],
                    maximum_absolute_error=float(np.abs(predicted-recorded).max()),atol=1e-5,rtol=0))
            raw_sets=[np.stack([np.load(PROBE/label/'paired'/(case['id']+'.npy'),allow_pickle=False) for case in cs])
                      for label in plan['variants']]
            png_sets=[np.stack([image(PROBE/label/'paired'/(case['id']+'.png')) for case in cs])
                      for label in plan['variants']]
            target=np.stack([image(PARENT/refs[case['source_person_or_reference']]['target']) for case in cs])
            arrays=[raw_sets[0],png_sets[0].astype(np.float32)/np.float32(255),
                    raw_sets[1],png_sets[1].astype(np.float32)/np.float32(255),target.astype(np.float32)/np.float32(255)]
            grids=torch.from_numpy(np.stack([grid112(refs[case['source_person_or_reference']]['matrix112']) for case in cs]))
            vec=identity.embedding(torch.cat([torch.from_numpy(array).permute(0,3,1,2) for array in arrays]),
                torch.cat([mm.float()]*5),torch.cat([grids]*5)).numpy()
            for label,slots in zip(plan['variants'],[[0,1,4],[2,3,4]]):
                rebuilt=np.stack([vec[index*len(cs):(index+1)*len(cs)] for index in slots],1)
                index={row['id']:row for row in variant_rows[label]['rows']}
                saved=np.asarray([index[case['id']]['embedding'] for case in cs],dtype=np.float32)
                assert np.allclose(rebuilt,saved,rtol=2e-4,atol=2e-4)
                recognizer_replays.append(dict(label=label,reference=cs[0]['reference_id'],
                    maximum_absolute_error=float(np.abs(rebuilt-saved).max()),atol=2e-4,rtol=2e-4))
    assert state_hash(model)==before
    # Diagnostic comparisons do not run or redefine the full acceptance gate.
    effects={}
    old,new=[variant_rows[label] for label in plan['variants']]
    for stage in ['raw','png']:
        effects[stage]={}
        for group,before_group in old['groups'][stage].items():
            after=new['groups'][stage][group]
            effects[stage][group]=dict(cases=before_group['cases'],
                MSE_change=after['MSE']-before_group['MSE'],SSIM_change=after['SSIM']-before_group['SSIM'],
                observed_recognizer_similarity_change=after['ArcFace_observed_fixed']-before_group['ArcFace_observed_fixed'],
                relative_structure_gain=1-after['landmark_high_frequency_MSE']/before_group['landmark_high_frequency_MSE'])
    for name,expected in manifest['files_sha256'].items():assert sha(PROBE/name)==expected
    for name,expected in result['parent_evidence_files_sha256'].items():assert sha(ROOT/name)==expected
    ANALYSIS.mkdir(exist_ok=True)
    record=dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        results_sha256=sha(PROBE/'results.json'),manifest_sha256=sha(PROBE/'manifest.json'),
        plan_sha256=sha(PROBE/'plan.json'),supervisor_sha256=sha(ROOT/'outputs/cctv_dgp_bank_feature_DC_probe_v1_execution/supervisor_receipt.json'),
        file_bindings_checked=len(manifest['files_sha256']),paired_compositions_checked=checked,
        native_unpaired_compositions_checked=native_checked,CPU_CUDA_replays_checked=124,
        independent_CPU_spot_replays=neural,independent_recognizer_spot_replays=recognizer_replays,diagnostic_effects=effects,
        same_CPU_before_and_after=True,original_CUDA_raw_and_PNG_not_replaced=True,
        fresh_inference_and_recognizer_replays_not_training=True,local_gradient_queries=0,local_optimizer_updates=0,
        source_checkpoint_and_model_state_unchanged=True,full_3905_counterfactual_not_tested=True,
        no_scientific_gate_redefinition=True,native_reference_accuracy_claimed=False,
        trained_candidate_qualification=False,app_promotion=False,visual_review_pending=True,
        independent_final_review_pending=True,goal_complete=False,seconds=time.monotonic()-start)
    with receipt.open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(dict(complete=True,paired_compositions=checked,native_unpaired_compositions=native_checked,
        CPU_spot_replays=len(neural),trained_candidate_qualified=False),flush=True)


if __name__=='__main__':main()
