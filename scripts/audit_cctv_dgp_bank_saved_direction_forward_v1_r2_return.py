"""Audit all finite proposal arrays, metrics and sheets without model inference."""
import ast
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import convolve1d
from skimage.metrics import structural_similarity
import torch

from import_cctv_dgp_bank_saved_direction_forward_v1_r2_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, PIN, read, sha, write


def image(path, mode='RGB'):
    with Image.open(path) as im:
        assert im.size == (256,256)
        return np.asarray(im.convert(mode)).copy()


def support(mask, landmarks):
    patch=np.zeros((256,256),bool)
    for x,y in np.floor(np.asarray(landmarks)).astype(int):
        patch[max(0,y-12):min(256,y+12),max(0,x-12):min(256,x+12)]=True
    patch &= cv2.erode(mask.astype(np.uint8),np.ones((13,13),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    assert patch.any()
    return patch


def metrics(actual, target8, mask, patch, vector, target_vector, mean_actual, prediction8=None):
    target=target8.astype(np.float32)/np.float32(255)
    error=actual-target
    mse=float(np.square(error[mask]).astype(np.float64).mean())
    _,ssmap=structural_similarity(target,actual,data_range=1,channel_axis=-1,win_size=7,full=True)
    interior=cv2.erode(mask.astype(np.uint8),np.ones((7,7),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    z=np.arange(-6,7,dtype=np.float64)
    kernel=np.exp(-.5*np.square(z/2));kernel/=kernel.sum()
    luma=np.array([.299,.587,.114])
    def high(a):
        return a-convolve1d(convolve1d(a,kernel,axis=0,mode='reflect'),kernel,axis=1,mode='reflect')
    detail_rgb=actual.astype(np.float64) if prediction8 is None else prediction8.astype(np.float64)/255
    detail=float(np.square(high((detail_rgb*luma).sum(2))-high((target8.astype(np.float64)/255*luma).sum(2)))[patch].mean())
    return dict(MSE=mse,SSIM=float(ssmap[interior].astype(np.float64).mean()),
        PSNR=float(-10*np.log10(mse)) if mse else None,perfect_match=mse==0,
        MAE=float(abs(error[mask]).astype(np.float64).mean()),
        ArcFace_observed_fixed=float(vector@target_vector),landmark_high_frequency_MSE=detail,
        constant_mean_shift_only_MSE=float(np.square((mean_actual-target)[mask]).astype(np.float64).mean()))


def delivered(raw, camera, mask):
    result=np.floor(raw*np.float32(255)).astype(np.uint8)
    result[~mask]=camera[~mask]
    return result


def compare_groups(before, after):
    assert set(before)==set(after)
    failures=[]
    for key,a in before.items():
        b=after[key]
        assert a['cases']==b['cases']
        for metric in ['MSE','SSIM','ArcFace_observed_fixed']:
            bad=b[metric]>a[metric]+1e-12 if metric=='MSE' else b[metric]<a[metric]-1e-6
            if bad:
                failures.append(dict(group=key,metric=metric,baseline=a[metric],candidate=b[metric]))
    a,b=before['degraded'],after['degraded']
    gain=1-b['landmark_high_frequency_MSE']/a['landmark_high_frequency_MSE']
    source={key:1-after[key]['landmark_high_frequency_MSE']/value['landmark_high_frequency_MSE']
            for key,value in before.items() if key.endswith('/degraded')}
    brightness=max(0,a['MSE']-b['constant_mean_shift_only_MSE'])/max(a['MSE']-b['MSE'],1e-12)
    if brightness>.2:
        failures.append(dict(group='degraded',metric='brightness_gain_fraction',candidate=brightness,maximum=.2))
    return dict(relative_feature_gain=gain,minimum_gain=.01,source_feature_gains=source,
        brightness_gain_fraction=brightness,preservation_failures=failures,
        pass_all_preview_requirements=not failures and gain>=.01 and all(v>=0 for v in source.values()),
        preview_pass_is_not_full_TRAIN_or_native_or_model_qualification=True)


def grouped(rows, stage):
    result={}
    for row in rows:
        src,profile=row['source'],row['profile']
        keys={'all','clear' if profile=='clear' else 'degraded',src+'/all',
              src+('/clear' if profile=='clear' else '/degraded'),src+'/'+profile}
        for key in keys:
            result.setdefault(key,[]).append(row[stage])
    names=['MSE','SSIM','ArcFace_observed_fixed','landmark_high_frequency_MSE','constant_mean_shift_only_MSE']
    return {key:dict(cases=len(values),**{name:float(np.mean([v[name] for v in values])) for name in names})
            for key,values in sorted(result.items())}


def state_hash(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        a=value.detach().cpu().contiguous()
        digest.update(name.encode()+b'\0'+str(a.dtype).encode()+b'\0')
        digest.update(str(tuple(a.shape)).encode()+b'\0'+a.numpy().tobytes())
    return digest.hexdigest()


def main():
    stamp=time.monotonic()
    receipt=ANALYSIS/'independent_audit.json'
    assert not receipt.exists()
    imported=read(ANALYSIS/'import.json')
    assert imported['complete'] and imported['protocol_sha256']==PIN
    p,old=read(RETURN/'protocol.json'),read(PARENT/'protocol.json')
    assert sha(RETURN/'protocol.json')==PIN
    manifest=read(RETURN/'export_manifest.json')
    assert manifest['complete_diagnostic'] and not manifest['failure_present']
    assert {f.relative_to(RETURN).as_posix() for f in RETURN.rglob('*') if f.is_file()}==set(manifest['files'])|{'export_manifest.json'}
    assert all(sha(RETURN/name)==digest for name,digest in manifest['files'].items())
    assert all(sha(PARENT/name)==digest for name,digest in p['parent_required_sha256'].items())
    assert not list(RETURN.rglob('*.pth'))
    assert p['preserved_scientific_thresholds']==old['scientific_thresholds']
    out=RETURN/'outputs/saved_direction_forward_v1'
    run=read(out/'run_results.json')
    counts=dict(B=100,recognizer=160,original=0,gradient_queries=0,backwards=0,
                optimizer_updates=0,temporary_parameter_proposals=4)
    assert run['complete'] and run['counts']==counts and not run['model_qualified']
    assert run['candidate_initial_state_restored'] and run['original_and_recognizer_unchanged']
    assert not run['checkpoint_created'] and run['four_temporary_model_displacements_are_not_optimizer_updates']
    assert run['native_pixels_used']==run['final_identity_pixels']==0
    assert run['more_epochs_not_qualified']
    assert 0<run['seconds']<p['worker_seconds'] and 0<run['peak_VRAM']<=p['peak_VRAM_bytes']
    assert (RETURN/'review_exit_code.txt').read_text(encoding='utf-8').strip()=='0'
    env=read(out/'environment.json')
    assert env['GPU']=='NVIDIA L4' and env['host']=='forensic-dgp-thesis' and env['manual_tmux']
    assert not env['optimizer_constructed'] and not env['gradient_enabled']
    begins=[i for i in range(0,3905,5) if old['cases'][i]['id'] in p['case_ids']]
    assert len(begins)==20
    assert [(r['proposal'],r['begin']) for r in run['records']]==[(prop['label'],i) for prop in p['proposals'] for i in begins]
    assert [r['begin'] for r in run['controls']]==begins
    projection=read(out/'timing_projection.json')
    assert projection['projection_seconds']<projection['cap_seconds']==p['measurement_seconds']
    assert projection['measured_B_batches']==22 and projection['candidate_reference_batches']==2
    assert np.isclose(projection['projection_seconds'],run['records'][1]['seconds']/22*100*1.5,rtol=0,atol=1e-7)
    torch.set_grad_enabled(False)
    initial=torch.load(PARENT/'initializers/initial_B.pth',map_location='cpu',weights_only=True)
    assert state_hash(initial)==p['initial_model_state']
    with np.load(RETURN/'directions.npz',allow_pickle=False) as archive:
        assert len(archive.files)==74
        directions={label:[archive[f'{label}_p{i:02d}'] for i in range(37)] for label in ['ordinary','constrained']}
    sys.path.insert(0,str(PARENT))
    from cctv_dgp_head4_lossless_v1 import unpack
    references={r['id']:r for r in old['references']}
    base_metrics=read(PARENT/'outputs/bank_comparison_v1/baseline/metrics.json')
    expected_metrics={r['id']:r for r in base_metrics['rows']}
    cases={c['id']:c for c in old['cases']}
    cache={}
    baseline_rows=[]
    for serial,begin in enumerate(begins):
        cs=old['cases'][begin:begin+5]
        ids=[c['id'] for c in cs]
        raw,emb=unpack(PARENT/f'outputs/bank_comparison_v1/baseline/packs/b{begin//5:03d}.npz',ids)
        control=read(out/f'control_b{begin//5:03d}.json')
        assert control==run['controls'][serial] and control['case_ids']==ids
        assert control['exact_raw_replay'] and control['maximum_absolute_error']==control['changed_values']==0
        assert control['actual_RGB_float32_sha256']==control['expected_RGB_float32_sha256']==hashlib.sha256(raw.tobytes()).hexdigest()
        cameras,targets,masks,patches=[],[],[],[]
        for i,c in enumerate(cs):
            ref=references[c['source_person_or_reference']]
            camera,target,mask=image(PARENT/c['input']),image(PARENT/ref['target']),image(PARENT/ref['observed'],'L')>0
            patch=support(mask,c['landmarks5_canvas_xy'])
            png=delivered(raw[i],camera,mask)
            assert np.array_equal(png,image(PARENT/'outputs/bank_comparison_v1/baseline/previews'/(c['id']+'.png')))
            row=dict(id=c['id'],source=c['source'],profile=c['profile'])
            row['raw']=metrics(raw[i],target,mask,patch,emb[i,0],emb[i,2],raw[i])
            row['png']=metrics(png.astype(np.float32)/255,target,mask,patch,emb[i,1],emb[i,2],png.astype(np.float32)/255,prediction8=png)
            for stage in ['raw','png']:
                for key,value in row[stage].items():
                    other=expected_metrics[c['id']][stage][key]
                    if value is None or isinstance(value,bool):
                        assert value==other
                    else:
                        assert np.isclose(value,other,rtol=0,atol=1e-13),(c['id'],stage,key,value,other)
            baseline_rows.append(row)
            cameras.append(camera);targets.append(target);masks.append(mask);patches.append(patch)
        cache[begin]=(cs,raw,emb,cameras,targets,masks,patches)
    groups_before={stage:grouped(baseline_rows,stage) for stage in ['raw','png']}
    results={}
    for proposal in p['proposals']:
        label=proposal['label']
        parameter=read(out/label/'parameter_receipt.json')
        state=dict(initial)
        for i,name in enumerate(p['learning_names']):
            value=initial[name].numpy()+np.float32(proposal['rate'])*directions[proposal['direction']][i]
            assert hashlib.sha256(value.tobytes()).hexdigest()==parameter['selected_parameter_value_sha256'][name]
            state[name]=torch.from_numpy(value)
        assert state_hash(state)==parameter['model_state'] and parameter['proposal']==proposal
        assert not parameter['checkpoint_saved'] and parameter['optimizer_updates']==0
        rows=[]
        entries=[r for r in run['records'] if r['proposal']==label]
        for entry in entries:
            begin=entry['begin']
            cs,baseline,old_emb,cameras,targets,masks,patches=cache[begin]
            assert entry['case_ids']==[c['id'] for c in cs] and entry['state']==parameter['model_state']
            raw_path,emb_path=out/entry['raw_file'],out/entry['embeddings_file']
            assert raw_path.resolve().is_relative_to(out/label) and emb_path.resolve().is_relative_to(out/label)
            assert sha(raw_path)==entry['raw_sha256'] and sha(emb_path)==entry['embeddings_sha256']
            raw=np.load(raw_path,allow_pickle=False)
            assert raw.dtype==np.float32 and raw.shape==(5,256,256,3) and np.isfinite(raw).all()
            assert 0<=raw.min()<=raw.max()<=1
            with np.load(emb_path,allow_pickle=False) as archive:
                assert set(archive.files)=={'raw','png'}
                raw_emb,png_emb=archive['raw'],archive['png']
            assert all(a.dtype==np.float32 and a.shape==(15,512) and np.isfinite(a).all()
                       and np.allclose(np.linalg.norm(a,axis=1),1,rtol=0,atol=1e-5) for a in [raw_emb,png_emb])
            for actual,index in [(raw_emb,0),(png_emb,1)]:
                assert np.allclose(actual[5:10],old_emb[:,index],rtol=1e-6,atol=1e-6)
                assert np.allclose(actual[10:15],old_emb[:,2],rtol=1e-6,atol=1e-6)
            maximum=float(max(abs(raw_emb[10:15]-old_emb[:,2]).max(),abs(png_emb[10:15]-old_emb[:,2]).max()))
            assert maximum==entry['maximum_target_embedding_error']
            for i,c in enumerate(cs):
                mask,camera,target=masks[i],cameras[i],targets[i]
                assert np.array_equal(raw[i][~mask],(camera.astype(np.float32)/255)[~mask])
                png=image(out/label/'png'/(c['id']+'.png'))
                assert np.array_equal(png,delivered(raw[i],camera,mask))
                shift=(raw[i]-baseline[i])[mask].astype(np.float64).mean(0)
                mean=np.clip(baseline[i].astype(np.float64)+shift,0,1).astype(np.float32)
                mean[~mask]=camera[~mask].astype(np.float32)/255
                mean_png=delivered(mean,camera,mask)
                row=dict(id=c['id'],source=c['source'],profile=c['profile'],
                    raw=metrics(raw[i],target,mask,patches[i],raw_emb[i],raw_emb[10+i],mean),
                    png=metrics(png.astype(np.float32)/255,target,mask,patches[i],png_emb[i],png_emb[10+i],mean_png.astype(np.float32)/255,prediction8=png),
                    postclip_mean_RGB_shift=shift.tolist(),raw_RGB_sha256=hashlib.sha256(raw[i].tobytes()).hexdigest(),
                    PNG_RGB_sha256=hashlib.sha256(png.tobytes()).hexdigest())
                rows.append(row)
        assert len(rows)==100
        groups={stage:grouped(rows,stage) for stage in ['raw','png']}
        gates={stage:compare_groups(groups_before[stage],groups[stage]) for stage in ['raw','png']}
        results[label]=dict(rows=rows,groups=groups,gates=gates,all_preview_requirements_pass=all(g['pass_all_preview_requirements'] for g in gates.values()),
            native_generalization_and_model_qualification=False)
        print(dict(proposal_checked=label,output_cases=100,preview_raw_PNG_pass=results[label]['all_preview_requirements_pass']),flush=True)
    logged=[ast.literal_eval(line) for line in (RETURN/'review.log').read_text(encoding='utf-8').splitlines()]
    assert len(logged)==80
    for serial,(entry,line) in enumerate(zip(run['records'],logged,strict=True)):
        assert line['proposal']==entry['proposal'] and line['reference_batches']==serial%20+1 and line['of']==20
        assert line['seconds']==entry['seconds'] and 0<line['seconds']<=run['seconds']
    metric_receipt=ANALYSIS/'finite_metrics.json'
    write(metric_receipt,dict(complete=True,run_results_sha256=sha(out/'run_results.json'),
        baseline_rows=baseline_rows,baseline_groups=groups_before,proposals=results,
        TRAIN_only=True,native_pixels_used=0,final_identity_pixels_used=0,model_qualified=False))
    sheets=ANALYSIS/'sheets'
    sheets.mkdir()
    sheet_rows=[]
    labels=[a['label'] for a in p['proposals']]
    for begin in begins:
        cs,baseline,emb,cameras,targets,masks,patches=cache[begin]
        canvas=Image.new('RGB',(256*8,5*284+40),(20,20,20))
        draw=ImageDraw.Draw(canvas)
        titles=['Input','Retained DGP','Ordinary 1e-4','Ordinary 1e-5','Constrained 1e-4','Constrained 1e-5','Clean TRAIN target','Observed mask']
        for column,title in enumerate(titles):
            draw.text((column*256+6,8),title,fill='white')
        cell_hashes=[]
        for i,c in enumerate(cs):
            cells=[cameras[i],delivered(baseline[i],cameras[i],masks[i])]
            cells += [image(out/label/'png'/(c['id']+'.png')) for label in labels]
            cells += [targets[i],np.repeat((masks[i].astype(np.uint8)*255)[...,None],3,axis=2)]
            y=40+i*284
            for column,cell in enumerate(cells):
                canvas.paste(Image.fromarray(cell),(column*256,y))
                cell_hashes.append(dict(case=c['id'],column=column,RGB_sha256=hashlib.sha256(cell.tobytes()).hexdigest()))
            draw.text((6,y+259),c['id']+' '+c['profile'],fill='white')
        file=sheets/f'b{begin//5:03d}.png'
        canvas.save(file)
        with Image.open(file) as saved:
            a=np.asarray(saved.convert('RGB'))
        for i,c in enumerate(cs):
            for column in range(8):
                value=a[40+i*284:40+i*284+256,column*256:(column+1)*256]
                assert hashlib.sha256(value.tobytes()).hexdigest()==cell_hashes[i*8+column]['RGB_sha256']
        sheet_rows.append(dict(file=file.relative_to(ROOT).as_posix(),sha256=sha(file),cells=cell_hashes))
    assert len(sheet_rows)==20
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        import_sha256=sha(ANALYSIS/'import.json'),protocol_sha256=PIN,run_results_sha256=sha(out/'run_results.json'),
        files_checked=len(manifest['files'])+1,parent_hashes_checked=5852,counts_recomputed=counts,
        VM_seconds=run['seconds'],VM_peak_VRAM=run['peak_VRAM'],exact_control_case_replays=100,
        disposable_parameter_value_and_state_hashes_recomputed=4,
        proposed_state_bytes_are_reconstructed_not_neural_quality_replayed=True,
        finite_raw_case_outputs_checked=400,exact_floor_PNGs_checked=400,
        independent_raw_and_PNG_metric_rows=800,all17_source_and_profile_groups_checked=True,
        finite_metrics_sha256=sha(metric_receipt),comparison_sheets=sheet_rows,
        all20_sheets_and800_cells_require_visual_review=True,visual_review_complete=False,
        new_neural_derivatives_or_optimizer_updates=0,no_local_model_parameter_displacement=True,
        no_local_neural_gradient_or_optimizer_calls=True,no_agent_VM_connection_or_launch=True,
        preserved_original_checkpoint_app_examples_unchanged=True,
        full_TRAIN_native_final_and_completion_qualification_pending=True,more_epochs_not_qualified=True,
        model_qualified=False,app_promotion=False,goal_complete=False,seconds=time.monotonic()-stamp))


if __name__=='__main__':
    main()
