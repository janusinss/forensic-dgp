"""Finite CPU inference diagnostic of spatially constant bank-feature changes.

Uses the retained rejected B50 state without training or resuming it. Compare
the same CPU state with original fusion versus zero spatial mean of each bank
fusion plane. This is a counterfactual, never model qualification or display
processing. Raw CUDA B50 stays authoritative and unchanged.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r4_B_return'
ANALYSIS = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r4_B_analysis'
OUT = ROOT / 'outputs/cctv_dgp_bank_feature_DC_probe_v1'
PIN = '8c906e8b1a3b9534872313414321a2c780560d9ea5ab50fec3e34232813cad85'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    assert not OUT.exists(), 'Retain the first diagnostic and any failure'
    p = read(PARENT/'protocol.json')
    audit = read(ANALYSIS/'independent_audit.json')
    matched = read(ANALYSIS/'matched_AB_comparison.json')
    diagnosis = read(ANALYSIS/'saved_learning_diagnostic.json')
    assert all(row['complete'] for row in [audit,matched,diagnosis])
    assert sha(PARENT/'protocol.json') == audit['protocol_sha256'] == PIN
    assert audit['optimizer_updates_in_VM']==50 and len(audit['limited_CPU_replays'])==2
    assert audit['CPU_CUDA_prior_fixtures_rechecked']==6 and not audit['model_qualified']
    assert matched['optimizer_updates_each']==50 and p['reserved_final_pixels']==0
    original_manifest = read(PARENT/'export_manifest.json')['files']
    bindings={}

    def bind(path):
        actual=sha(path)
        if path.is_relative_to(PARENT):
            name=path.relative_to(PARENT).as_posix()
            expected=PIN if name=='protocol.json' else original_manifest[name]
            assert actual==expected,name
        bindings[path.relative_to(ROOT).as_posix()]=actual

    for name in ['independent_audit.json','matched_AB_comparison.json','saved_learning_diagnostic.json']:
        bind(ANALYSIS/name)
    bind(PARENT/'protocol.json')
    for name in ['cctv_dgp_bank_comparison_v1_model.py','cctv_dgp_generative_bank_v1.py',
                 'cctv_dgp_pilot.py','dgp_frozen_inference_v2.py','frozen_capacity_contract.py',
                 'frozen_raw_metrics.py','cctv_dgp_head4_lossless_v1.py']:
        bind(PARENT/name)
    for name in original_manifest:
        if name.startswith(('bank_implementation/','prior_source/','dgp_source/')):
            bind(PARENT/name)
    cases=[case for case in p['cases'] if case['id'] in p['preview_case_ids']]
    assert len(cases)==100
    references=list(dict.fromkeys(case['source_person_or_reference'] for case in cases))
    assert len(references)==20 and len(p['native_development'])==24
    by_source={}
    for reference in references:
        case=next(case for case in cases if case['source_person_or_reference']==reference)
        by_source.setdefault(case['source'],[]).append(reference)
    assert len(by_source)==2
    visual_references=[reference for values in by_source.values() for reference in values[:2]]
    OUT.mkdir()
    plan=dict(complete=True,protocol_sha256=PIN,parent_import_sha256=sha(ANALYSIS/'import.json'),
        parent_state_sha256=original_manifest['outputs/bank_comparison_v1/state_B50/full_state.pth'],
        variants=['original_B50_CPU','B50_bank_feature_DC_removed_CPU'],
        single_changed_operation='subtract spatial mean of each projected bank feature plane before decoder fusion',
        projection_dimensions=[16,32,64,128,256],input_only_operation=True,mean_uses_no_clean_target=True,
        photographic_TRAIN_cases=100,photographic_references=20,native_DEV_unpaired_cases=24,
        final_identities=0,maximum_observation_and_conditioned_bank_batches=26,
        maximum_reconstruction_calls=52,maximum_recognizer_batches=20,maximum_seconds=900,
        maximum_output_bytes=512*1024**2,minimum_disk_reserve_bytes=2*1024**3,
        CPU_threads=4,optimizer_updates=0,gradient_queries=0,backwards=0,
        visual_reference_ids=visual_references,native_visual_case_ids=[case['id'] for case in p['native_development'][:4]],
        planned_visual_sheets=5,visual_selection_not_final_qualification=True,
        bank_on_off_subset_not_full_cohort=True,failed_state_resume=False,
        model_qualified=False,app_promotion=False,goal_complete=False)
    write(OUT/'plan.json',plan)
    start=time.monotonic()
    counts=dict(observation_and_bank_batches=0,reconstruction_calls=0,recognizer_batches=0,
                optimizer_updates=0,gradient_queries=0,backwards=0)
    try:
        import shutil
        import numpy as np
        from PIL import Image
        import torch
        sys.path.insert(0,str(PARENT))
        from cctv_dgp_bank_comparison_v1_model import ConditionedBankDGP
        from cctv_dgp_generative_bank_v1 import load_prior
        from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
        from cctv_dgp_pilot import state_hash,FixedObservedIdentity,grid112
        from frozen_capacity_contract import feature_support,exported_pixel_metrics,detail_metric
        from frozen_raw_metrics import pixel_metrics,detail_float,deliver,mean_only,review_groups
        from cctv_dgp_head4_lossless_v1 import unpack
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        assert not torch.cuda.is_available(),'CPU-only diagnostic required'
        import platform
        import importlib.metadata
        write(OUT/'environment.json',dict(Python=sys.version,Torch=torch.__version__,
            NumPy=np.__version__,ONNXRuntime=importlib.metadata.version('onnxruntime'),
            host=platform.node(),platform=platform.platform(),device='CPU',
            external_prior_frozen=True,parent_checkpoint_failed=True,
            original_training_not_resumed=True,threads=4,optimizer_updates=0,gradient_queries=0))
        for name in ['weights/dgp_v2.pth','weights/w600k_r50.onnx','initializers/mean_style.npy',
                     'outputs/bank_comparison_v1/state_B50/full_state.pth']:
            bind(PARENT/name)
        frozen,_=load_frozen_dgp_restorer(PARENT/'weights/dgp_v2.pth',expected_sha256=p['original_checkpoint_sha256'])
        prior=load_prior(PARENT/'prior_source',PARENT/'bank_implementation',p['standalone_prior_sha256'])
        model=ConditionedBankDGP(frozen.net,prior,torch.from_numpy(np.load(PARENT/'initializers/mean_style.npy',allow_pickle=False)))
        state=torch.load(PARENT/'outputs/bank_comparison_v1/state_B50/full_state.pth',map_location='cpu',weights_only=True)
        model.load_state_dict(state['model'],strict=True);model.eval().requires_grad_(False)
        expected=read(PARENT/'outputs/bank_comparison_v1/state_B50/receipt.json')['model_state']
        assert state_hash(model)==expected and all(not value.requires_grad for value in model.parameters())
        identity=FixedObservedIdentity(PARENT/'weights/w600k_r50.onnx','cpu')
        refs={row['id']:row for row in p['references']}
        variants=plan['variants'];rows={label:[] for label in variants};native={label:[] for label in variants}
        replay=[];feature_means=[]
        for label in variants:
            (OUT/label/'paired').mkdir(parents=True);(OUT/label/'native').mkdir()

        def clock():
            assert time.monotonic()-start<plan['maximum_seconds'],'Finite900s inference stop'
            assert shutil.disk_usage(OUT).free>=plan['minimum_disk_reserve_bytes'],'Local2GiB reserve stop'
            assert sum(file.stat().st_size for file in OUT.rglob('*') if file.is_file())<=plan['maximum_output_bytes']
            assert counts['observation_and_bank_batches']<=26 and counts['reconstruction_calls']<=52 and counts['recognizer_batches']<=20
            assert counts['optimizer_updates']==counts['gradient_queries']==counts['backwards']==0

        def pixels(path,mode='RGB'):
            bind(path)
            with Image.open(path) as image:
                assert image.size==(256,256)
                return np.asarray(image.convert(mode)).copy()

        def infer(cs,is_native):
            clock()
            cameras=[pixels(PARENT/case['input']) for case in cs]
            masks=[pixels(PARENT/(case['observed'] if is_native else refs[case['source_person_or_reference']]['observed']),'L')>0 for case in cs]
            x=torch.from_numpy(np.stack(cameras).astype(np.float32)/np.float32(255)).permute(0,3,1,2)
            mask=torch.from_numpy(np.stack(masks))[:,None]
            maps=model.current.observe(x)
            features=model.conditioned_features(maps)
            extras={size:model.fusions[size](value) for size,value in features.items()}
            centered={size:value-value.mean(dim=(-2,-1),keepdim=True) for size,value in extras.items()}
            counts['observation_and_bank_batches']+=1
            values=[]
            for option in [extras,centered]:
                values.append(torch.where(mask,model.current.reconstruct(x,maps,option),x))
                counts['reconstruction_calls']+=1
            raw=[value.permute(0,2,3,1).numpy().copy() for value in values]
            pngs=[[deliver(a,camera,mm) for a,camera,mm in zip(value,cameras,masks)] for value in raw]
            for i,case in enumerate(cs):
                feature_means.append(dict(id=case['id'],native_unpaired=is_native,
                    projected_mean_L2={size:float(value[i].mean(dim=(-2,-1)).double().norm()) for size,value in extras.items()},
                    centered_absolute_mean_max={size:float(value[i].mean(dim=(-2,-1)).abs().max()) for size,value in centered.items()}))
            if is_native:
                saved_paths=[PARENT/'outputs/bank_comparison_v1/B50/native'/(case['id']+'.npy') for case in cs]
                for path in saved_paths:bind(path)
                saved=[np.load(path,allow_pickle=False) for path in saved_paths]
                target8=None;embeddings=None
            else:
                begin=next(i for i,case in enumerate(p['cases']) if case['id']==cs[0]['id'])
                ids=[case['id'] for case in cs]
                baseline_path=PARENT/'outputs/bank_comparison_v1/baseline/packs'/f'b{begin//5:03d}.npz'
                result_path=PARENT/'outputs/bank_comparison_v1/B50/packs'/f'b{begin//5:03d}.npz'
                bind(baseline_path);bind(result_path)
                baseline,_=unpack(baseline_path,ids)
                saved,_=unpack(result_path,ids,baseline=baseline)
                target8=[pixels(PARENT/refs[case['source_person_or_reference']]['target']) for case in cs]
                target=torch.from_numpy(np.stack(target8).astype(np.float32)/np.float32(255)).permute(0,3,1,2)
                grids=torch.from_numpy(np.stack([grid112(refs[case['source_person_or_reference']]['matrix112']) for case in cs]))
                # One recognizer invocation per reference batch, with five sets.
                tensor_pngs=[torch.from_numpy(np.stack(value).astype(np.float32)/np.float32(255)).permute(0,3,1,2) for value in pngs]
                vectors=identity.embedding(torch.cat([values[0],tensor_pngs[0],values[1],tensor_pngs[1],target]),
                    torch.cat([mask.float()]*5),torch.cat([grids]*5)).numpy()
                counts['recognizer_batches']+=1
                embeddings=[np.stack([vectors[j*len(cs):(j+1)*len(cs)] for j in [0,1,4]],1),
                            np.stack([vectors[j*len(cs):(j+1)*len(cs)] for j in [2,3,4]],1)]
            for i,case in enumerate(cs):
                maximum=float(np.abs(raw[0][i]-saved[i]).max())
                assert np.allclose(raw[0][i],saved[i],rtol=1e-4,atol=1e-4),'CPU replay exceeds existing parity bound'
                replay.append(dict(id=case['id'],native_unpaired=is_native,maximum_raw_error=maximum,
                    raw_atol=1e-4,raw_rtol=1e-4,
                    original_CUDA_PNG_replaced=False))
                for j,label in enumerate(variants):
                    a=raw[j][i];png=pngs[j][i];mm=masks[i];camera=cameras[i]
                    assert a.dtype==np.float32 and np.isfinite(a).all() and 0<=a.min()<=a.max()<=1
                    assert np.array_equal(a[~mm],(camera.astype(np.float32)/np.float32(255))[~mm])
                    folder=OUT/label/('native' if is_native else 'paired')
                    np.save(folder/(case['id']+'.npy'),a,allow_pickle=False)
                    Image.fromarray(png).save(folder/(case['id']+'.png'))
                    row=dict(id=case['id'],raw_RGB_float32_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                             PNG_RGB_sha256=hashlib.sha256(png.tobytes()).hexdigest())
                    if is_native:
                        row.update(clean_reference=None,paired_accuracy_not_available=True)
                        native[label].append(row)
                    else:
                        support=feature_support(mm,case['landmarks5_canvas_xy'])
                        mean_raw,mean_png,shift=mean_only(a,baseline[i],camera,mm)
                        rm=pixel_metrics(a,target8[i],mm);pm=exported_pixel_metrics(png,target8[i],mm)
                        emb=embeddings[j][i]
                        rm.update(ArcFace_observed_fixed=float(emb[0]@emb[2]),landmark_high_frequency_MSE=detail_float(a,target8[i],support),
                                  constant_mean_shift_only_MSE=pixel_metrics(mean_raw,target8[i],mm)['MSE'])
                        pm.update(ArcFace_observed_fixed=float(emb[1]@emb[2]),landmark_high_frequency_MSE=detail_metric(png,target8[i],support),
                                  constant_mean_shift_only_MSE=exported_pixel_metrics(mean_png,target8[i],mm)['MSE'])
                        row.update(source=case['source'],profile=case['profile'],raw=rm,png=pm,embedding=emb.tolist(),
                                   postclip_mean_RGB_shift=shift.tolist())
                        rows[label].append(row)
            clock()

        with torch.inference_mode():
            for begin in range(0,len(cases),5):
                infer(cases[begin:begin+5],False)
                print(dict(paired_cases=begin+5,of=100,seconds=time.monotonic()-start),flush=True)
            for begin in range(0,24,4):
                infer(p['native_development'][begin:begin+4],True)
                print(dict(native_unpaired_cases=begin+4,of=24,seconds=time.monotonic()-start),flush=True)
        assert state_hash(model)==expected and all(not value.requires_grad for value in model.parameters())
        for label in variants:
            assert len(rows[label])==100 and len(native[label])==24
            write(OUT/label/'metrics.json',dict(complete=True,label=label,rows=rows[label],
                groups={stage:review_groups(rows[label],stage) for stage in ['raw','png']},
                native_unpaired_rows=native[label],full_cohort_capacity_not_tested=True,model_qualified=False))
        for name,expected_hash in bindings.items():
            assert sha(ROOT/name)==expected_hash
        result=dict(complete=True,plan_sha256=sha(OUT/'plan.json'),worker_sha256=sha(__file__),
            protocol_sha256=PIN,parent_evidence_files_sha256=bindings,counts=counts,
            paired_TRAIN_cases=100,native_DEV_unpaired_cases=24,CPU_CUDA_replay=replay,
            feature_plane_means=feature_means,model_state_unchanged=True,all_weights_and_buffers_frozen=True,
            original_CUDA_outputs_preserved=True,processing_not_trained_candidate=True,
            qualification=False,all_seven_completion_families_pending=True,independent_final_review=False,
            independent_counterfactual_audit_pending=True,visual_review_pending=True,goal_complete=False,
            seconds=time.monotonic()-start)
        write(OUT/'results.json',result)
    except BaseException as error:
        write(OUT/'failure.json',dict(complete=False,type=type(error).__name__,message=str(error),
            traceback=traceback.format_exc(),counts=counts,seconds=time.monotonic()-start,
            optimizer_updates=0,failed_artifacts_retained=True,goal_complete=False))
        raise
    finally:
        write(OUT/'manifest.json',dict(complete=True,files_sha256={file.relative_to(OUT).as_posix():sha(file)
              for file in sorted(OUT.rglob('*')) if file.is_file()},model_qualified=False,goal_complete=False))


if __name__=='__main__':
    main()
