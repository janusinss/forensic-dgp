"""Independent saved-array audit of the missing frozen bank-only composition."""
from datetime import datetime, timezone
import hashlib
import sys
import time
import numpy as np

from import_cctv_dgp_bank_path_ablation_v1_return import ROOT, RETURN, PARENT, ANALYSIS as PATH_ANALYSIS, read, sha, write
from audit_cctv_dgp_bank_path_ablation_v1_return import image
from audit_cctv_dgp_bank_comparison_return_v1_portable import equal_metrics, grouped, gate
from probe_cctv_dgp_bank_partition_v1_frozen import DEST


def main():
    start=time.monotonic(); receipt=DEST/'independent_audit.json'; assert not receipt.exists()
    run=read(DEST/'run_results.json'); assert run['complete'] and run['no_new_checkpoint']
    assert not (DEST/'failure.json').exists() and not list(DEST.rglob('*.pth'))
    manifest=read(DEST/'file_manifest.json')['files']
    assert {f.relative_to(DEST).as_posix() for f in DEST.rglob('*') if f.is_file()}==set(manifest)|{'file_manifest.json'}
    for name,expected in manifest.items(): assert sha(DEST/name)==expected
    assert run['counts']==dict(B=50,recognizer=20,original=0,gradient_queries=0,backwards=0,optimizer_updates=0)
    assert len(run['control_parity'])==6 and all(r['cases']==5 and r['raw_maximum_absolute_error']<1e-4 for r in run['control_parity'])
    assert 0<run['seconds']<900 and sum((DEST/n).stat().st_size for n in manifest)<256*1024**2
    p,old=read(RETURN/'protocol.json'),read(PARENT/'protocol.json')
    for name,expected in p['parent_required_sha256'].items(): assert sha(PARENT/name)==expected
    for name,expected in read(RETURN/'export_manifest.json')['files'].items(): assert sha(RETURN/name)==expected
    sys.path.insert(0,str(PARENT))
    from cctv_dgp_head4_lossless_v1 import unpack
    from frozen_capacity_contract import feature_support,exported_pixel_metrics,detail_metric
    from frozen_raw_metrics import pixel_metrics,detail_float,deliver,mean_only
    helpers=['cctv_dgp_head4_lossless_v1.py','frozen_capacity_contract.py','frozen_raw_metrics.py']
    assert all(sha(PARENT/n)==p['parent_required_sha256'][n] for n in helpers)
    refs={r['id']:r for r in old['references']}
    selected=[i for i in range(0,3905,5) if old['cases'][i]['id'] in p['case_ids']]
    ids=[c['id'] for begin in selected for c in old['cases'][begin:begin+5]]
    folder=DEST/'bank_only'; metrics=read(folder/'metrics.json')
    assert metrics['complete'] and metrics['model_state']==run['frozen_model_state_unchanged']
    assert [r['id'] for r in metrics['rows']]==ids
    recorded={r['id']:r for r in metrics['rows']}; rows=[]
    for begin in selected:
        cases=old['cases'][begin:begin+5]; block_ids=[c['id'] for c in cases]
        base,base_emb=unpack(PARENT/'outputs/bank_comparison_v1/baseline/packs'/f'b{begin//5:03d}.npz',block_ids)
        raw,emb=unpack(folder/'packs'/f'b{begin//5:03d}.npz',block_ids)
        assert np.allclose(np.linalg.norm(emb,axis=2),1,rtol=0,atol=1e-5)
        assert np.array_equal(emb[:,2],base_emb[:,2])
        for i,c in enumerate(cases):
            ref=refs[c['source_person_or_reference']]
            camera,target=image(PARENT/c['input']),image(PARENT/ref['target']); observed=image(PARENT/ref['observed'],'L')>0
            support=feature_support(observed,c['landmarks5_canvas_xy']); a=raw[i]
            assert np.array_equal(a[~observed],(camera.astype(np.float32)/np.float32(255))[~observed])
            png=deliver(a,camera,observed); assert np.array_equal(png,image(folder/'previews'/(c['id']+'.png')))
            mr,mp,shift=mean_only(a,base[i],camera,observed)
            rm,pm=pixel_metrics(a,target,observed),exported_pixel_metrics(png,target,observed)
            rm.update(ArcFace_observed_fixed=float(emb[i,0]@emb[i,2]),landmark_high_frequency_MSE=detail_float(a,target,support),
                constant_mean_shift_only_MSE=pixel_metrics(mr,target,observed)['MSE'])
            pm.update(ArcFace_observed_fixed=float(emb[i,1]@emb[i,2]),landmark_high_frequency_MSE=detail_metric(png,target,support),
                constant_mean_shift_only_MSE=exported_pixel_metrics(mp,target,observed)['MSE'])
            row=recorded[c['id']]; equal_metrics(rm,row['raw']); equal_metrics(pm,row['png'])
            assert np.allclose(shift,row['postclip_mean_RGB_shift'],rtol=0,atol=1e-14)
            assert hashlib.sha256(a.tobytes()).hexdigest()==row['raw_RGB_float32_sha256']
            assert hashlib.sha256(png.tobytes()).hexdigest()==row['PNG_RGB_sha256']
            rows.append(dict(id=c['id'],source=c['source'],profile=c['profile'],raw=rm,png=pm))
    rebuilt={stage:grouped(rows,stage) for stage in ['raw','png']}
    for stage,groups in rebuilt.items():
        assert set(groups)==set(metrics['groups'][stage])
        for name,values in groups.items(): equal_metrics(values,metrics['groups'][stage][name])
    assert [r['id'] for r in metrics['native_unpaired_rows']]==p['native_case_ids']
    for c,row in zip(old['native_development'],metrics['native_unpaired_rows']):
        camera,mask=image(PARENT/c['input']),image(PARENT/c['observed'],'L')>0
        raw=np.load(folder/'native'/(c['id']+'.npy'),allow_pickle=False)
        assert raw.dtype==np.float32 and raw.shape==(256,256,3) and np.isfinite(raw).all() and 0<=raw.min()<=raw.max()<=1
        assert np.array_equal(raw[~mask],(camera.astype(np.float32)/np.float32(255))[~mask])
        png=deliver(raw,camera,mask); assert np.array_equal(png,image(folder/'native'/(c['id']+'.png')))
        assert row['unpaired_no_clean_metrics']
        assert hashlib.sha256(raw.tobytes()).hexdigest()==row['raw_RGB_float32_sha256']
        assert hashlib.sha256(png.tobytes()).hexdigest()==row['PNG_RGB_sha256']
    prior=read(PATH_ANALYSIS/'independent_audit.json')
    groups={'initial':prior['baseline_groups'],'joint':prior['variant_groups']['enabled'],
        'decoder_only':prior['variant_groups']['disabled'],'bank_only':rebuilt}
    gates={label:{stage:dict(gate(groups['initial'][stage],values),only_100_TRAIN_preview_cases=True,model_qualified=False)
        for stage,values in result.items()} for label,result in groups.items() if label!='initial'}
    for name,expected in read(PATH_ANALYSIS/'local_protected_before.json')['files_sha256'].items(): assert sha(ROOT/name)==expected
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        run_results_sha256=sha(DEST/'run_results.json'),manifest_sha256=sha(DEST/'file_manifest.json'),
        prior_full_saved_array_audit_sha256=sha(PATH_ANALYSIS/'independent_audit.json'),
        new_paired_raw_PNG_cases_independently_checked=100,new_native_compositions_checked=24,
        reused_control_paired_cases=300,reused_native_control_cases=72,variant_groups=groups,partial_preview_gate_diagnostics=gates,
        original_scientific_thresholds_unchanged=True,new_variant_CPU_saved_controls_L4=True,
        limited_CPU_CUDA_control_parity_not_full=True,all_fixed_target_embeddings_identical=True,
        fixed_weight_cross_compositions_not_independently_trained_arms=True,full_3905_gate_tested=False,
        local_neural_forwards=0,local_gradient_queries=0,local_optimizer_updates=0,
        independent_final_review_pending=True,visual_review_pending=True,model_qualified=False,
        all_seven_completion_families_pending=True,goal_complete=False,seconds=time.monotonic()-start))
    print({label:{s:dict(gain_percent=100*g['gain'],failures=len(g['failures']),brightness_percent=100*g['brightness'])
        for s,g in result.items()} for label,result in gates.items()},flush=True)


if __name__=='__main__': main()
