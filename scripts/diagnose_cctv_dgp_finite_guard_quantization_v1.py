"""Fixed saved-pixel conversion experiment; CPU inference only, no DGP or learning."""
import argparse
from datetime import datetime, timezone
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
sys.path.insert(0, str(BUNDLE))
from cctv_dgp_finite_guard_v1_r1_contract import read, write, sha
from frozen_raw_metrics import deliver, mean_only, review_groups
from frozen_capacity_contract import feature_support, exported_pixel_metrics, detail_metric, capacity


def pixels(path, mode='RGB'):
    with Image.open(path) as image:
        assert image.size == (256,256)
        return np.asarray(image.convert(mode)).copy()


def nearest(raw, camera, mask):
    assert raw.dtype == np.float32 and raw.shape == (256,256,3)
    assert np.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 1
    value = np.floor(raw.astype(np.float64)*255 + .5).astype(np.uint8)
    value[~mask] = camera[~mask]
    return value


def quantization_row(raw, floor, rounded, mask):
    scaled = raw[mask].astype(np.float64)*255
    result = {'observed_components':int(scaled.size),
        'changed_components':int(np.count_nonzero(floor[mask] != rounded[mask])),
        'maximum_policy_byte_difference':int(np.abs(floor.astype(np.int16)-rounded.astype(np.int16)).max())}
    for name, value in [('floor',floor),('nearest',rounded)]:
        error = value[mask].astype(np.float64)-scaled
        result[name] = {'signed_error_byte_sum':float(error.sum()),
            'absolute_error_byte_sum':float(np.abs(error).sum()),
            'maximum_absolute_error_bytes':float(np.abs(error).max())}
    assert result['maximum_policy_byte_difference'] <= 1
    assert result['nearest']['maximum_absolute_error_bytes'] <= .5
    assert result['nearest']['absolute_error_byte_sum'] <= result['floor']['absolute_error_byte_sum']
    return result


def bound_sources(q):
    for name, digest in q['source_bindings'].items():
        assert sha(ROOT/name) == digest, name


def prepare():
    assert not OUT.exists(), 'Retain every previous preparation and failure'
    assert sha(BUNDLE/'protocol.json') == PIN
    p = read(BUNDLE/'protocol.json'); result = read(RETURN/'outputs/results.json')
    audit = ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_independent_audit.json'
    visual = ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_return_review_v1/visual_review.json'
    assert read(audit)['complete'] and read(visual)['complete']
    assert result['optimizer_updates'] == result['epochs'] == 0
    assert result['states_before'] == result['states_after_restore']
    labels = ['baseline']+[t['variant'] for t in result['trial_summaries']]
    assert labels == ['baseline','proposal_s01_2em05','proposal_s01_1em05','proposal_s01_5em06']
    bindings = {}
    def bind(path, expected=None):
        path = Path(path); digest = sha(path)
        if expected is not None: assert digest == expected, path
        bindings[path.relative_to(ROOT).as_posix()] = digest
    for name, digest in p['local_sources_sha256'].items(): bind(ROOT/name, digest)
    for name, digest in p['assets_sha256'].items(): bind(BUNDLE/name, digest)
    bind(BUNDLE/'protocol.json', PIN)
    export = read(RETURN/'export_manifest.json')
    for label in labels:
        for case in p['cases']:
            for suffix in ['.npy','.png','_embeddings.npy']:
                name = 'outputs/'+label+'/'+case['id']+suffix
                bind(RETURN/name, export['files_sha256'][name])
        name = 'outputs/'+label+'/metrics.json'; bind(RETURN/name, export['files_sha256'][name])
    for path in [RETURN/'protocol.json',RETURN/'outputs/results.json',RETURN/'export_manifest.json',audit,visual,
                 visual.parent/'gallery_independent_audit.json',Path(__file__),
                 ROOT/'scripts/audit_cctv_dgp_finite_guard_quantization_v1.py',
                 ROOT/'scripts/supervise_cctv_dgp_finite_guard_quantization_v1.py',
                 ROOT/'CCTV_DGP_FINITE_GUARD_QUANTIZATION_V1_DESIGN.md']:
        bind(path)
    choices = [('TRAIN_gradient','dataset/asian_faces','lowlight_lr32'),
               ('TRAIN_cross_cohort','dataset/thumbnails128x128','blur_lr24'),
               ('TRAIN_gradient','dataset/thumbnails128x128','clear'),
               ('TRAIN_cross_cohort','dataset/asian_faces','compound_lr24')]
    replay = []
    for cohort, source, profile in choices:
        ids = next(c['case_ids'] for c in p['cohorts'] if c['name'] == cohort)
        replay.append(next(c['id'] for c in p['cases'] if c['id'] in ids and c['source'] == source and c['profile'] == profile))
    OUT.mkdir()
    write(OUT/'protocol.json', {'format':'saved-pixel-quantization-diagnostic-v1',
        'prepared_UTC':datetime.now(timezone.utc).isoformat(),'parent_protocol_sha256':PIN,
        'source_bindings':bindings,'variants':labels,'cases':[c['id'] for c in p['cases']],
        'policies':['floor','nearest'],'nearest_policy':'floor(float64(raw)*255+0.5)',
        'baseline_and_candidate_same_policy':True,'camera_padding_exact':True,
        'scientific_thresholds':p['scientific_thresholds'],
        'embedding_replay_max_abs':p['CPU_replay_tolerances']['embedding_max_abs'],
        'worker_seconds':600,'external_worker_seconds':630,'audit_seconds':300,'external_audit_seconds':330,
        'maximum_output_bytes':512*1024**2,'CPU_threads':4,'worker_recognizer_forwards':80,
        'audit_recognizer_forwards':4,'audit_replay_cases':replay,
        'DGP_forwards':0,'completion_forwards':0,'gradient_queries':0,'parameter_updates':0,'epochs':0,
        'native_or_DEV_or_final_used':False,'new_images_visually_qualified':False,
        'app_adoption':False,'model_qualification':False,'goal_complete':False})
    print({'prepared':True,'source_bindings':len(bindings),'cases':100,
        'protocol_sha256':sha(OUT/'protocol.json'),'neural_calls':0},flush=True)


def run(pin):
    started = time.monotonic(); assert sha(OUT/'protocol.json') == pin
    assert not (OUT/'results.json').exists() and not (OUT/'failure.json').exists()
    q = read(OUT/'protocol.json'); bound_sources(q)
    p = read(BUNDLE/'protocol.json'); prior = read(RETURN/'outputs/results.json')
    assert q['parent_protocol_sha256'] == PIN and q['scientific_thresholds'] == p['scientific_thresholds']
    import torch
    from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash
    torch.set_num_threads(q['CPU_threads'])
    identity = FixedObservedIdentity(BUNDLE/'weights/w600k_r50.onnx','cpu')
    assert state_hash(identity) == p['recognizer_state']
    references = {r['id']:r for r in p['references']}; labels = q['variants']
    cache = {}
    for c in p['cases']:
        camera,target,mask = pixels(BUNDLE/c['input']),pixels(BUNDLE/c['target']),pixels(BUNDLE/c['observed'],'L') > 0
        cache[c['id']] = (camera,target,mask,feature_support(mask,c['landmarks5_canvas_xy']),
            grid112(references[c['source_person_or_reference']]['matrix112']))
    calls,worst,all_rows,all_groups = 0,0.,{},{}
    rgb = lambda v: torch.from_numpy(v.astype(np.float32)/np.float32(255)).permute(2,0,1)
    with torch.no_grad():
        for label in labels:
            folder = OUT/label; folder.mkdir()
            old = read(RETURN/'outputs'/label/'metrics.json'); rows = []
            for begin in range(0,100,5):
                assert time.monotonic()-started < q['worker_seconds']
                batch = p['cases'][begin:begin+5]; images,masks,grids,data = [],[],[],[]
                for c in batch:
                    cid = c['id']; camera,target,mask,support,grid = cache[cid]
                    raw = np.load(RETURN/'outputs'/label/(cid+'.npy'),allow_pickle=False)
                    base = np.load(RETURN/'outputs/baseline'/(cid+'.npy'),allow_pickle=False)
                    floor = pixels(RETURN/'outputs'/label/(cid+'.png')); rounded = nearest(raw,camera,mask)
                    assert np.array_equal(floor,deliver(raw,camera,mask))
                    assert np.array_equal(rounded[~mask],camera[~mask])
                    mraw,_,_ = mean_only(raw,base,camera,mask)
                    data.append((c,floor,rounded,mraw,raw))
                for choice in range(3):
                    for c,floor,rounded,mraw,raw in data:
                        camera,target,mask,support,grid = cache[c['id']]
                        images.append(rgb([floor,rounded,target][choice]))
                        masks.append(torch.from_numpy(mask.astype(np.float32))[None]); grids.append(torch.from_numpy(grid))
                vectors = identity.embedding(torch.stack(images),torch.stack(masks),torch.stack(grids)).numpy().copy()
                calls += 1; assert vectors.dtype == np.float32 and vectors.shape == (15,512)
                for offset,(c,floor,rounded,mraw,raw) in enumerate(data):
                    cid = c['id']; camera,target,mask,support,grid = cache[cid]
                    saved = np.stack([vectors[offset],vectors[offset+5],vectors[offset+10]])
                    np.save(folder/(cid+'_embeddings.npy'),saved,allow_pickle=False)
                    Image.fromarray(rounded).save(folder/(cid+'.png'))
                    previous = np.load(RETURN/'outputs'/label/(cid+'_embeddings.npy'),allow_pickle=False)
                    worst = max(worst,float(np.abs(saved[[0,2]]-previous[[1,2]]).max()))
                    row = {'id':cid,'source':c['source'],'profile':c['profile'],
                        'quantization':quantization_row(raw,floor,rounded,mask)}
                    for name,index,png in [('floor',0,floor),('nearest',1,rounded)]:
                        metrics = exported_pixel_metrics(png,target,mask)
                        metrics.update({'ArcFace_observed_fixed':float(saved[index]@saved[2]),
                            'landmark_high_frequency_MSE':detail_metric(png,target,support),
                            'constant_mean_shift_only_MSE':exported_pixel_metrics(
                                deliver(mraw,camera,mask) if name == 'floor' else nearest(mraw,camera,mask),target,mask)['MSE']})
                        row[name] = metrics
                    rows.append(row)
                if (begin+5) % 25 == 0:
                    print({'variant':label,'converted_cases':begin+5,'of':100,'recognizer_forwards':calls},flush=True)
            grouped = {name:{co['name']:review_groups([
                {**r,'png':r[name]} for r in rows if r['id'] in co['case_ids']], 'png') for co in p['cohorts']}
                for name in q['policies']}
            write(folder/'metrics.json',{'complete':True,'variant':label,'rows':rows,'groups':grouped})
            all_rows[label],all_groups[label] = rows,grouped
    comparisons = {}; original_floor_matches = True
    for trial in prior['trial_summaries']:
        label = trial['variant']; comparisons[label] = {}
        for name in q['policies']:
            comparisons[label][name] = {}
            for co in p['cohorts']:
                cname = co['name']; comparison = capacity(all_groups['baseline'][name][cname],all_groups[label][name][cname],.01)
                comparisons[label][name][cname] = comparison
                if name == 'floor':
                    keys = lambda d:{(r['group'],r['metric']) for r in d['preservation_failures']}
                    original = trial['comparisons'][cname]['png']
                    original_floor_matches &= keys(comparison) == keys(original) and comparison['pass'] == original['pass']
    assert calls == q['worker_recognizer_forwards'] and worst <= q['embedding_replay_max_abs']
    assert state_hash(identity) == p['recognizer_state']
    assert all(not v.requires_grad and v.grad is None for v in identity.parameters())
    bound_sources(q)
    quantization = {}
    for label,rows in all_rows.items():
        count = sum(r['quantization']['observed_components'] for r in rows)
        quantization[label] = {'observed_components':count,
            'fraction_changed_components':sum(r['quantization']['changed_components'] for r in rows)/count,
            **{name:{'mean_signed_error_bytes':sum(r['quantization'][name]['signed_error_byte_sum'] for r in rows)/count,
                'mean_absolute_error_bytes':sum(r['quantization'][name]['absolute_error_byte_sum'] for r in rows)/count}
                for name in q['policies']}}
    assert sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file()) <= q['maximum_output_bytes']
    result = {'complete':True,'protocol_sha256':pin,'parent_protocol_sha256':PIN,'comparisons':comparisons,
        'quantization':quantization,'all_original_floor_failure_decisions_reproduced':bool(original_floor_matches),
        'maximum_original_VM_embedding_replay_error':worst,'recognizer_state_before_and_after':p['recognizer_state'],
        'worker_recognizer_forwards':calls,'new_PNGs':400,'PNG_metrics_recomputed':800,
        'raw_outputs_changed':False,'raw_VM_decisions_retained':True,'baseline_and_candidate_same_policy':True,
        'scientific_thresholds_changed':False,'new_images_visually_qualified':False,
        'DGP_forwards':0,'completion_forwards':0,'gradient_queries':0,'parameter_updates':0,'epochs':0,
        'native_or_DEV_or_final_used':False,'app_adoption':False,'model_qualification':False,
        'goal_complete':False,'seconds':time.monotonic()-started}
    assert result['seconds'] < q['worker_seconds']; write(OUT/'results.json',result)
    manifest = {f.relative_to(OUT).as_posix():sha(f) for f in OUT.rglob('*') if f.is_file()}
    write(OUT/'artifact_manifest.json',{'complete':True,'protocol_sha256':pin,'files_sha256':manifest})
    print({'complete':True,'new_PNGs':400,'PNG_metrics':800,'recognizer_forwards':calls,
        'floor_decisions_reproduced':bool(original_floor_matches),'seconds':time.monotonic()-started},flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--protocol-sha'); args = parser.parse_args()
    if args.prepare: prepare()
    else:
        assert args.protocol_sha
        try: run(args.protocol_sha)
        except Exception:
            import traceback
            if OUT.exists() and not (OUT/'failure.json').exists():
                write(OUT/'failure.json',{'complete':False,'protocol_sha256':args.protocol_sha,
                    'traceback':traceback.format_exc(),'training_updates':0,'goal_complete':False})
            raise


if __name__ == '__main__': main()
