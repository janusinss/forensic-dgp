"""Prospective saved-pixel/metric checker, with four fixed CPU embedding replays."""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_vm'
RETURN = ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_vm_return'
OUT = ROOT/'outputs/cctv_dgp_finite_guard_quantization_v1'
PIN = 'efdd62759213136114a56f0aaa256278753cac8bac4bc6c6a52f1f8b7550598f'
sys.path.insert(0,str(BUNDLE))
from cctv_dgp_finite_guard_v1_r1_contract import read,write,sha
from frozen_capacity_contract import capacity,feature_support,exported_pixel_metrics,detail_metric
from frozen_raw_metrics import deliver,mean_only,review_groups


def pixels(path, mode='RGB'):
    with Image.open(path) as image:
        assert image.size == (256,256)
        return np.asarray(image.convert(mode)).copy()


def independent_round(value, camera, observed):
    scaled = value.astype(np.float64)*255
    integer = np.floor(scaled).astype(np.int32)
    rounded = (integer+(scaled-integer >= .5)).astype(np.uint8)
    rounded[~observed] = camera[~observed]
    return rounded


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--protocol-sha',required=True); args = parser.parse_args()
    started = time.monotonic(); assert sha(OUT/'protocol.json') == args.protocol_sha
    assert not (OUT/'independent_audit.json').exists()
    q,p = read(OUT/'protocol.json'),read(BUNDLE/'protocol.json')
    assert sha(BUNDLE/'protocol.json') == PIN == q['parent_protocol_sha256']
    assert q['policies'] == ['floor','nearest'] and q['baseline_and_candidate_same_policy']
    assert q['scientific_thresholds'] == p['scientific_thresholds']
    assert q['scientific_thresholds']['early_structure_gain'] == .01
    for name,digest in q['source_bindings'].items(): assert sha(ROOT/name) == digest,name
    manifest = read(OUT/'artifact_manifest.json')
    assert manifest['complete'] and manifest['protocol_sha256'] == args.protocol_sha
    assert len(manifest['files_sha256']) == 806
    assert {f.relative_to(OUT).as_posix() for f in OUT.rglob('*') if f.is_file()} == set(manifest['files_sha256'])|{'artifact_manifest.json'}
    for name,digest in manifest['files_sha256'].items(): assert sha(OUT/name) == digest,name
    result = read(OUT/'results.json'); original = read(RETURN/'outputs/results.json')
    assert result['complete'] and result['protocol_sha256'] == args.protocol_sha
    assert result['seconds'] < q['worker_seconds'] and result['worker_recognizer_forwards'] == 80
    assert result['new_PNGs'] == 400 and result['PNG_metrics_recomputed'] == 800
    for key in ['DGP_forwards','completion_forwards','gradient_queries','parameter_updates','epochs']: assert result[key] == q[key] == 0
    for key in ['raw_outputs_changed','scientific_thresholds_changed','new_images_visually_qualified',
                'native_or_DEV_or_final_used','app_adoption','model_qualification','goal_complete']: assert result[key] is False
    assert original['optimizer_updates'] == 0 and original['states_before'] == original['states_after_restore']
    rows_by_label,groups_by_label,max_previous,max_byte = {},{},0.,0
    for label in q['variants']:
        receipt = read(OUT/label/'metrics.json'); assert receipt['complete'] and receipt['variant'] == label
        assert [r['id'] for r in receipt['rows']] == q['cases'] == [c['id'] for c in p['cases']]
        computed = []
        for c,saved in zip(p['cases'],receipt['rows']):
            assert time.monotonic()-started < q['audit_seconds']
            cid = c['id']; camera,target = pixels(BUNDLE/c['input']),pixels(BUNDLE/c['target'])
            observed = pixels(BUNDLE/c['observed'],'L') > 0
            raw = np.load(RETURN/'outputs'/label/(cid+'.npy'),allow_pickle=False)
            baseline = np.load(RETURN/'outputs/baseline'/(cid+'.npy'),allow_pickle=False)
            floor,rounded = pixels(RETURN/'outputs'/label/(cid+'.png')),pixels(OUT/label/(cid+'.png'))
            assert np.array_equal(deliver(raw,camera,observed),floor)
            assert np.array_equal(independent_round(raw,camera,observed),rounded)
            assert np.array_equal(floor[~observed],camera[~observed]) and np.array_equal(rounded[~observed],camera[~observed])
            difference = np.abs(floor.astype(np.int16)-rounded.astype(np.int16)); max_byte = max(max_byte,int(difference.max()))
            assert max_byte <= 1
            vectors = np.load(OUT/label/(cid+'_embeddings.npy'),allow_pickle=False)
            assert vectors.dtype == np.float32 and vectors.shape == (3,512) and np.isfinite(vectors).all()
            assert np.allclose(np.sum(vectors.astype(np.float64)**2,axis=1),1,rtol=0,atol=1e-5)
            previous = np.load(RETURN/'outputs'/label/(cid+'_embeddings.npy'),allow_pickle=False)
            max_previous = max(max_previous,float(np.abs(vectors[[0,2]]-previous[[1,2]]).max()))
            support = feature_support(observed,c['landmarks5_canvas_xy'])
            mean_raw,_,_ = mean_only(raw,baseline,camera,observed)
            fresh = {'id':cid,'source':c['source'],'profile':c['profile']}
            for name,index,png in [('floor',0,floor),('nearest',1,rounded)]:
                metrics = exported_pixel_metrics(png,target,observed)
                mean_png = deliver(mean_raw,camera,observed) if name == 'floor' else independent_round(mean_raw,camera,observed)
                metrics.update({'ArcFace_observed_fixed':float(vectors[index]@vectors[2]),
                    'landmark_high_frequency_MSE':detail_metric(png,target,support),
                    'constant_mean_shift_only_MSE':exported_pixel_metrics(mean_png,target,observed)['MSE']})
                assert saved[name] == metrics; fresh[name] = metrics
            scaled = raw[observed].astype(np.float64)*255
            quant = {'observed_components':int(scaled.size),
                'changed_components':int(np.count_nonzero(floor[observed] != rounded[observed])),
                'maximum_policy_byte_difference':int(difference.max())}
            for name,png in [('floor',floor),('nearest',rounded)]:
                delta = png[observed].astype(np.float64)-scaled
                quant[name] = {'signed_error_byte_sum':float(delta.sum()),
                    'absolute_error_byte_sum':float(np.abs(delta).sum()),
                    'maximum_absolute_error_bytes':float(np.abs(delta).max())}
            assert saved['quantization'] == quant
            assert quant['nearest']['maximum_absolute_error_bytes'] <= .5
            assert quant['nearest']['absolute_error_byte_sum'] <= quant['floor']['absolute_error_byte_sum']
            fresh['quantization'] = quant; assert fresh == saved; computed.append(fresh)
        groups = {policy:{co['name']:review_groups([{**r,'png':r[policy]} for r in computed if r['id'] in co['case_ids']], 'png')
            for co in p['cohorts']} for policy in q['policies']}
        assert groups == receipt['groups']; rows_by_label[label],groups_by_label[label] = computed,groups
    comparisons = {}; historical_decisions_match = True
    for trial in original['trial_summaries']:
        label = trial['variant']; comparisons[label] = {}
        for policy in q['policies']:
            comparisons[label][policy] = {}
            for co in p['cohorts']:
                name = co['name']; fresh = capacity(groups_by_label['baseline'][policy][name],groups_by_label[label][policy][name],.01)
                comparisons[label][policy][name] = fresh
                if policy == 'floor':
                    keys = lambda d:{(r['group'],r['metric']) for r in d['preservation_failures']}
                    old = trial['comparisons'][name]['png']
                    historical_decisions_match &= keys(old) == keys(fresh) and old['pass'] == fresh['pass']
    assert comparisons == result['comparisons']
    assert result['all_original_floor_failure_decisions_reproduced'] == historical_decisions_match
    quantization = {}
    for label,rows in rows_by_label.items():
        components = sum(r['quantization']['observed_components'] for r in rows)
        quantization[label] = {'observed_components':components,
            'fraction_changed_components':sum(r['quantization']['changed_components'] for r in rows)/components,
            **{policy:{'mean_signed_error_bytes':sum(r['quantization'][policy]['signed_error_byte_sum'] for r in rows)/components,
                'mean_absolute_error_bytes':sum(r['quantization'][policy]['absolute_error_byte_sum'] for r in rows)/components}
                for policy in q['policies']}}
    assert quantization == result['quantization']
    assert max_previous == result['maximum_original_VM_embedding_replay_error'] and max_previous <= q['embedding_replay_max_abs']
    import torch
    from cctv_dgp_pilot import FixedObservedIdentity,grid112,state_hash
    torch.set_num_threads(q['CPU_threads']); identity = FixedObservedIdentity(BUNDLE/'weights/w600k_r50.onnx','cpu')
    assert state_hash(identity) == p['recognizer_state'] == result['recognizer_state_before_and_after']
    by_id = {c['id']:c for c in p['cases']}; references = {r['id']:r for r in p['references']}
    rgb = lambda value:torch.from_numpy(value.astype(np.float32)/np.float32(255)).permute(2,0,1)
    worst,calls = 0.,0
    with torch.no_grad():
        for label,cid in zip(q['variants'],q['audit_replay_cases']):
            assert time.monotonic()-started < q['audit_seconds']
            c = by_id[cid]; mask = pixels(BUNDLE/c['observed'],'L') > 0
            images = [pixels(RETURN/'outputs'/label/(cid+'.png')),pixels(OUT/label/(cid+'.png')),pixels(BUNDLE/c['target'])]
            grid = torch.from_numpy(grid112(references[c['source_person_or_reference']]['matrix112']))[None].repeat(3,1,1,1)
            support = torch.from_numpy(mask.astype(np.float32))[None,None].repeat(3,1,1,1)
            vector = identity.embedding(torch.stack([rgb(value) for value in images]),support,grid).numpy().copy(); calls += 1
            saved = np.load(OUT/label/(cid+'_embeddings.npy'),allow_pickle=False)
            worst = max(worst,float(np.abs(vector-saved).max()))
    assert calls == q['audit_recognizer_forwards'] == 4 and worst <= q['embedding_replay_max_abs']
    assert state_hash(identity) == p['recognizer_state'] and all(not v.requires_grad and v.grad is None for v in identity.parameters())
    for name,digest in q['source_bindings'].items(): assert sha(ROOT/name) == digest,name
    assert sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file()) <= q['maximum_output_bytes']
    write(OUT/'independent_audit.json',{'complete':True,'protocol_sha256':args.protocol_sha,
        'checker_sha256':sha(Path(__file__)),'source_bindings_verified':len(q['source_bindings']),
        'artifact_bindings_verified':806,'new_PNGs_independently_recomputed':400,'all_PNG_metric_rows_verified':800,
        'quality_decisions_verified':12,'maximum_policy_byte_difference':max_byte,
        'original_floor_failure_decisions_reproduced':bool(historical_decisions_match),
        'all_source_profile_mean_shift_controls_verified':True,'baseline_and_proposal_policy_matched':True,
        'quantization_errors_independently_recomputed':True,'recognizer_state_unchanged':True,
        'CPU_replay_cases':4,'maximum_CPU_embedding_replay_error':worst,
        'local_DGP_forwards':0,'local_completion_forwards':0,'local_gradient_queries':0,'local_parameter_updates':0,
        'new_images_visually_qualified':False,'model_qualification':False,'goal_complete':False,
        'seconds':time.monotonic()-started})
    print({'complete':True,'PNG_rows':800,'quality_decisions':12,'CPU_replays':4,
        'maximum_embedding_error':worst,'seconds':time.monotonic()-started},flush=True)


if __name__ == '__main__': main()
