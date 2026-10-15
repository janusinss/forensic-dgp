"""Local CPU inference-only missing cross-composition; never train or resume."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch

from import_cctv_dgp_bank_path_ablation_v1_return import ROOT, RETURN, PARENT, ANALYSIS as PATH_ANALYSIS, read, sha, write
from audit_cctv_dgp_bank_path_ablation_v1_return import image
from cctv_dgp_bank_partition_v1 import validate_states, compose

DEST = ROOT/'outputs/cctv_dgp_bank_partition_v1_analysis'


def main():
    started = time.monotonic()
    assert not DEST.exists(), 'Immutable frozen diagnostic; no unchanged rerun'
    assert shutil.disk_usage(ROOT).free >= 2*1024**3
    assert platform.system() == 'Windows'
    for name in ['independent_audit.json', 'limited_frozen_replay.json', 'visual_review.json', 'partition_state_audit.json']:
        assert read(PATH_ANALYSIS/name)['complete']
    p, old = read(RETURN/'protocol.json'), read(PARENT/'protocol.json')
    for name, expected in p['parent_required_sha256'].items(): assert sha(PARENT/name) == expected
    state_audit = read(PATH_ANALYSIS/'partition_state_audit.json')
    assert sha(ROOT/'scripts/cctv_dgp_bank_partition_v1.py') == state_audit['helper_sha256']
    sys.path.insert(0, str(PARENT))
    from cctv_dgp_bank_comparison_v1_model import ConditionedBankDGP
    from cctv_dgp_generative_bank_v1 import load_prior
    from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash, buffer_hash
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_head4_lossless_v1 import pack, unpack
    from frozen_capacity_contract import feature_support, exported_pixel_metrics, detail_metric
    from frozen_raw_metrics import pixel_metrics, detail_float, deliver, mean_only, review_groups
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.set_grad_enabled(False)
    original, provenance = load_frozen_dgp_restorer(PARENT/'weights/dgp_v2.pth', expected_sha256=old['original_checkpoint_sha256'])
    identity = FixedObservedIdentity(PARENT/'weights/w600k_r50.onnx', 'cpu')
    prior = load_prior(PARENT/'prior_source', PARENT/'bank_implementation', old['standalone_prior_sha256'])
    model = ConditionedBankDGP(original.net, prior, torch.from_numpy(np.load(PARENT/'initializers/mean_style.npy', allow_pickle=False)))
    initial = torch.load(PARENT/'initializers/initial_B.pth', map_location='cpu', weights_only=True)
    stored = torch.load(PARENT/'outputs/bank_comparison_v1/state_B50/full_state.pth', map_location='cpu', weights_only=True)
    assert stored['updates'] == 50 and not stored['manual_failed_resume_allowed']
    trained = stored['model']; del stored
    partitions = validate_states(initial, trained, model.learning_names())
    counts = dict(B=0, recognizer=0, original=0, gradient_queries=0, backwards=0, optimizer_updates=0)
    model.register_forward_hook(lambda *_: counts.__setitem__('B', counts['B']+1))
    identity.encoder.register_forward_hook(lambda *_: counts.__setitem__('recognizer', counts['recognizer']+1))
    model.eval().requires_grad_(False); identity.eval().requires_grad_(False)
    DEST.mkdir(); folder = DEST/'bank_only'
    for child in ['packs', 'previews', 'native']: (folder/child).mkdir(parents=True)
    refs = {r['id']:r for r in old['references']}
    selected = [i for i in range(0, 3905, 5) if old['cases'][i]['id'] in p['case_ids']]
    parent_out = PARENT/'outputs/bank_comparison_v1'
    def tensor(a):
        return torch.from_numpy(np.asarray(a).astype(np.float32)/np.float32(255)).permute(0,3,1,2)
    def clock():
        assert time.monotonic()-started < 900, 'Finite15-minute frozen CPU stop'
        assert shutil.disk_usage(ROOT).free >= 1024**3, 'Local disk reserve stop'
        assert sum(f.stat().st_size for f in DEST.rglob('*') if f.is_file()) <= 256*1024**2
        assert counts['B'] <= 50 and counts['recognizer'] <= 20 and not torch.is_grad_enabled()
    parity, rows, native = [], [], []
    try:
        with torch.inference_mode():
            # Small fresh same-CPU control comparisons bound CPU/CUDA differences.
            for variant in ['initial', 'joint', 'decoder_only']:
                model.load_state_dict(compose(initial, trained, partitions, variant), strict=True)
                before, buffers = state_hash(model), buffer_hash(model)
                assert before == state_audit['variants'][variant]['model_state']
                for begin in p['parity_begins']:
                    clock(); block = old['cases'][begin:begin+5]; ids = [c['id'] for c in block]
                    cameras = [image(PARENT/c['input']) for c in block]
                    masks = np.stack([image(PARENT/refs[c['source_person_or_reference']]['observed'], 'L') > 0 for c in block])
                    x = tensor(cameras)
                    actual = torch.where(torch.from_numpy(masks)[:,None], model(x), x).permute(0,2,3,1).numpy()
                    control = parent_out/'baseline' if variant == 'initial' else RETURN/'outputs/path_ablation_v1'/('enabled' if variant == 'joint' else 'disabled')
                    expected, _ = unpack(control/'packs'/f'b{begin//5:03d}.npz', ids)
                    assert np.allclose(actual, expected, rtol=1e-4, atol=1e-4), (variant, 'CPU/CUDA control parity')
                    parity.append(dict(variant=variant, begin=begin, cases=5, source=block[0]['source'],
                        raw_maximum_absolute_error=float(np.abs(actual-expected).max())))
                assert state_hash(model) == before and buffer_hash(model) == buffers
            model.load_state_dict(compose(initial, trained, partitions, 'bank_only'), strict=True)
            before, buffers = state_hash(model), buffer_hash(model)
            assert before == state_audit['variants']['bank_only']['model_state']
            for begin in selected:
                clock(); block = old['cases'][begin:begin+5]; ids = [c['id'] for c in block]
                cameras = [image(PARENT/c['input']) for c in block]
                targets = [image(PARENT/refs[c['source_person_or_reference']]['target']) for c in block]
                masks = [image(PARENT/refs[c['source_person_or_reference']]['observed'], 'L') > 0 for c in block]
                support = [feature_support(m,c['landmarks5_canvas_xy']) for m,c in zip(masks,block)]
                x = tensor(cameras); mm = torch.from_numpy(np.stack(masks))[:,None]
                pred = torch.where(mm, model(x), x); raw = pred.permute(0,2,3,1).numpy().copy()
                pngs = [deliver(a,c,m) for a,c,m in zip(raw,cameras,masks)]
                grids = torch.from_numpy(np.stack([grid112(refs[c['source_person_or_reference']]['matrix112']) for c in block]))
                vectors = identity.embedding(torch.cat([pred,tensor(pngs),tensor(targets)]),
                    torch.cat([mm.float()]*3),torch.cat([grids]*3)).numpy().copy()
                base, base_emb = unpack(parent_out/'baseline/packs'/f'b{begin//5:03d}.npz',ids)
                assert np.allclose(vectors[10:15],base_emb[:,2],rtol=2e-4,atol=2e-4)
                # Keep the exact same saved target vectors for all four comparisons.
                emb = np.stack([vectors[:5],vectors[5:10],base_emb[:,2]],axis=1)
                pack(folder/'packs'/f'b{begin//5:03d}.npz',raw,ids,emb)
                for i,c in enumerate(block):
                    a,png,target,m = raw[i],pngs[i],targets[i],masks[i]
                    mr,mp,shift = mean_only(a,base[i],cameras[i],m)
                    rm,pm = pixel_metrics(a,target,m),exported_pixel_metrics(png,target,m)
                    rm.update(ArcFace_observed_fixed=float(emb[i,0]@emb[i,2]),landmark_high_frequency_MSE=detail_float(a,target,support[i]),
                        constant_mean_shift_only_MSE=pixel_metrics(mr,target,m)['MSE'])
                    pm.update(ArcFace_observed_fixed=float(emb[i,1]@emb[i,2]),landmark_high_frequency_MSE=detail_metric(png,target,support[i]),
                        constant_mean_shift_only_MSE=exported_pixel_metrics(mp,target,m)['MSE'])
                    Image.fromarray(png).save(folder/'previews'/(c['id']+'.png'))
                    rows.append(dict(id=c['id'],source=c['source'],profile=c['profile'],raw=rm,png=pm,
                        postclip_mean_RGB_shift=shift.tolist(),raw_RGB_float32_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                        PNG_RGB_sha256=hashlib.sha256(png.tobytes()).hexdigest()))
                print(dict(frozen_bank_only_cases=len(rows),of=100,seconds=time.monotonic()-started),flush=True)
            for c in old['native_development']:
                clock(); camera=image(PARENT/c['input']); m=image(PARENT/c['observed'],'L')>0; x=tensor([camera])
                raw=torch.where(torch.from_numpy(m)[None,None],model(x),x)[0].permute(1,2,0).numpy().copy()
                png=deliver(raw,camera,m)
                np.save(folder/'native'/(c['id']+'.npy'),raw,allow_pickle=False); Image.fromarray(png).save(folder/'native'/(c['id']+'.png'))
                native.append(dict(id=c['id'],raw_RGB_float32_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                    PNG_RGB_sha256=hashlib.sha256(png.tobytes()).hexdigest(),unpaired_no_clean_metrics=True))
            assert state_hash(model)==before and buffer_hash(model)==buffers
        assert state_hash(original.net)==old['original_state'] and state_hash(identity)==old['recognizer_state']
        assert all(not m.training for net in [model,original,identity] for m in net.modules())
        assert all(not a.requires_grad and a.grad is None for net in [model,original,identity] for a in net.parameters())
        assert counts==dict(B=50,recognizer=20,original=0,gradient_queries=0,backwards=0,optimizer_updates=0)
        assert len(rows)==100 and len(native)==24
        write(folder/'metrics.json',dict(complete=True,variant='bank_only',rows=rows,native_unpaired_rows=native,
            groups={stage:review_groups(rows,stage) for stage in ['raw','png']},model_state=before,
            source_and_profile_metrics_TRAIN_only=True,full_3905_gate_not_tested=True,model_qualified=False))
        for name,expected in p['parent_required_sha256'].items(): assert sha(PARENT/name)==expected
        for name,expected in read(PATH_ANALYSIS/'local_protected_before.json')['files_sha256'].items(): assert sha(ROOT/name)==expected
        clock()
        write(DEST/'run_results.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),worker_sha256=sha(__file__),
            partition_state_audit_sha256=sha(PATH_ANALYSIS/'partition_state_audit.json'),counts=counts,
            frozen_model_state_unchanged=before,frozen_buffers_unchanged=buffers,control_parity=parity,
            fresh_missing_variant_paired_cases=100,fresh_missing_variant_native_cases=24,
            reused_initial_joint_decoder_controls=True,saved_control_source_GPU='NVIDIA L4',new_variant_device='CPU',
            all_fixed_target_embedding_rows_reused=100,CPU_CUDA_parity_is_limited_not_full=True,
            counterfactual_not_independently_trained_arms=True,no_new_checkpoint=True,
            seconds=time.monotonic()-started,worker_cap_seconds=900,output_cap_bytes=256*1024**2,
            minimum_free_bytes=2*1024**3,reserve_bytes=1024**3,model_qualified=False,app_promotion=False))
        write(DEST/'file_manifest.json',dict(complete=True,files={f.relative_to(DEST).as_posix():sha(f) for f in DEST.rglob('*') if f.is_file()}))
        print(dict(complete=True,counts=counts,seconds=time.monotonic()-started),flush=True)
    except BaseException as exc:
        write(DEST/'failure.json',dict(complete=False,type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),
            counts=counts,seconds=time.monotonic()-started,optimizer_updates=0,prior_quality_failure_retained=True))
        raise


if __name__=='__main__': main()
