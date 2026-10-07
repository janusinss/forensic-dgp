"""Finite existing-L4 differentiation proof for the separate original decoder.

Zero optimizer construction, updates, epochs or checkpoint writes. Keep every
partial result and original V27 failure. No automatic follow-on experiment.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import shutil
import signal
import sys
import tarfile
import time
import traceback
from types import MethodType


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify(root, pin):
    assert sha(root / 'protocol.json') == pin, 'Decoder diagnostic protocol changed'
    p = read(root / 'protocol.json')
    assert p['format'] == 'own-DGP-original-decoder-zero-update-gradient-proof-v1'
    assert p['optimizer_updates'] == p['epochs'] == 0
    assert p['decoder_parameters'] == 609219 and p['decoder_parameter_tensors'] == 14
    assert p['batches'] == 10 and p['component_gradient_calls'] == 70
    for name, digest in p['assets_sha256'].items():
        path = (root / name).resolve()
        assert path.is_relative_to(root) and path.is_file() and not path.is_symlink() and sha(path) == digest, name
    return p


def parent_check(parent, p):
    assert sha(parent / 'protocol.json') == p['closed_V27_protocol_sha256']
    base = read(parent / 'protocol.json')
    assert base['format'] == 'dgp-direct-feature-skips-capacity-v27'
    assert len(base['cases']) == 50 and all(c['role'] == 'train' for c in base['cases'])
    for name, digest in base['assets_sha256'].items():
        path = (parent / name).resolve()
        assert path.is_relative_to(parent) and path.is_file() and sha(path) == digest, 'Original V27 asset changed: ' + name
    for name, digest in p['closed_V27_evidence_sha256'].items():
        assert sha(parent / name) == digest, 'Original V27 failed evidence changed: ' + name
    failure = read(parent / 'outputs/failure.json')
    early = read(parent / 'outputs/early_structure_stop.json')
    assert failure['updates'] == 50 and failure['backwards'] == 51
    assert failure['cause'] == 'No one-percent early structural gain; retain stop'
    assert early['minimum'] == .01 and not early['pass'] and not (parent / 'outputs/results.json').exists()
    assert base['prospective_gates'] == p['retained_capacity_gates']
    return base


def vm_scope(root, parent):
    assert sys.platform == 'linux', 'Existing Linux VM only; no local gradients'
    scope = (Path.home() / 'forensic-dgp').resolve()
    assert root.is_relative_to(scope) and parent == scope / 'cctv_dgp_feature_skips_vm_v27'
    assert root.name == 'cctv_dgp_original_decoder_gradient_v1_vm'


def original_functions(parent):
    """Compile only the verified original guard and fixed-filter/loss definitions."""
    import os
    import platform
    import subprocess
    import urllib.request
    tree = ast.parse((parent / 'scripts/cctv_dgp_feature_skips_v27_vm.py').read_text(encoding='utf-8'))
    guard = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'require_vm')
    namespace = {'sys':sys, 'platform':platform, 'Path':Path, 'urllib':__import__('urllib'), 'os':os, 'subprocess':subprocess}
    exec(compile(ast.Module(body=[guard], type_ignores=[]), '<pinned-V27-existing-L4-guard>', 'exec'), namespace)
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    functions = [next(n for n in run.body if isinstance(n, ast.FunctionDef) and n.name == name)
                 for name in ['mean', 'feature_errors', 'ssim']]
    head_tree = ast.parse((parent / 'cctv_dgp_feature_skips_v27.py').read_text(encoding='utf-8'))
    head = next(n for n in head_tree.body if isinstance(n, ast.ClassDef) and n.name == 'SpatialFeatureHead')
    filters = [next(n for n in head.body if isinstance(n, ast.FunctionDef) and n.name == name) for name in ['blur', 'high']]
    return namespace['require_vm'], functions, filters


def run(root, parent, p, pin):
    vm_scope(root, parent)  # Before importing any model or creating outputs.
    assert not (root / 'outputs').exists(), 'Preserve prior or partial diagnostic; no resume'
    start = time.monotonic()
    progress = {'reference_DGP_forwards':0, 'candidate_DGP_forwards':0, 'recognizer_forwards':0,
                'component_gradient_calls':0, 'optimizer_updates':0, 'epochs':0}
    out = root / 'outputs'
    out.mkdir()
    try:
        assert shutil.disk_usage(root).free >= p['budgets']['minimum_free_disk_bytes'], 'Need2GiB free; no deletion by this diagnostic'
        base = parent_check(parent, p)
        guard, definitions, filters = original_functions(parent)
        guard(root, idle=True)
        import numpy as np
        from PIL import Image
        import torch
        from torch.nn import functional as F
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        sys.path.insert(0, str(parent))
        # Candidate definition is shipped separately; original implementation remains pinned.
        sys.path.insert(0, str(root))
        from cctv_dgp_original_decoder_candidate_v1 import OriginalDecoderCandidate
        from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash
        from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
        from dgp_face_restoration import as_tensor
        from cctv_dgp_degraded_objective_v24 import cohort_normalizers
        from cctv_dgp_batchmatched_identity_v26 import objective_terms
        def clock():
            torch.cuda.synchronize()
            assert time.monotonic() - start < p['budgets']['worker_seconds'], 'Decoder proof worker cap600 seconds'
            assert torch.cuda.max_memory_allocated() <= p['budgets']['peak_vram_bytes'], 'Decoder proof allocated VRAM cap20GiB'
        original, provenance = load_frozen_dgp_restorer(parent / 'weights/dgp_v2.pth',
            expected_sha256=base['assets_sha256']['weights/dgp_v2.pth'], device='cuda')
        original_state = state_hash(original.net)
        assert original_state == p['original_DGP_state']
        candidate = OriginalDecoderCandidate(original.net)
        assert state_hash(candidate.net) == original_state
        identity = FixedObservedIdentity(parent / 'weights/w600k_r50.onnx', 'cuda')
        identity_state = state_hash(identity)
        assert identity_state == p['frozen_recognizer_state']
        for model, key in [(original.net,'reference_DGP_forwards'), (candidate.net,'candidate_DGP_forwards'), (identity.encoder,'recognizer_forwards')]:
            model.register_forward_hook(lambda *_args, key=key: progress.__setitem__(key, progress[key]+1))
        named = [(name, v) for name, v in candidate.net.named_parameters() if v.requires_grad]
        parameters = [v for _, v in named]
        layout = []; offset = 0
        for name, v in named:
            layout.append({'name':name,'shape':list(v.shape),'start':offset,'end':offset+v.numel()})
            offset += v.numel()
        assert layout == p['parameter_layout'] and offset == 609219 and len(parameters) == 14
        refs = {ref['id']:ref for ref in base['references']}
        items = []
        for case in base['cases']:
            with Image.open(parent / case['input']) as im: camera = np.asarray(im.convert('RGB')).copy()
            with Image.open(parent / case['target']) as im: target = np.asarray(im.convert('RGB')).copy()
            with Image.open(parent / case['observed']) as im: support = np.asarray(im).copy() > 0
            assert camera.shape == target.shape == (256,256,3) and support.shape == (256,256) and support.any()
            def erode(radius):
                size = 2*radius+1; padded = np.pad(support.astype(np.int64), radius)
                summed = np.pad(padded.cumsum(0).cumsum(1), ((1,0),(1,0)))
                return summed[size:,size:]-summed[:-size,size:]-summed[size:,:-size]+summed[:-size,:-size] == size*size
            interior = erode(6); feature = np.zeros((256,256), bool)
            for point in case['landmarks5_canvas_xy']:
                xx, yy = np.floor(point).astype(int)
                feature[max(0,yy-12):min(256,yy+12),max(0,xx-12):min(256,xx+12)] = True
            feature &= interior
            assert feature.any()
            t = lambda value: torch.from_numpy(np.asarray(value).copy()).cuda()
            items.append({'case':case, 'camera':camera, 'target8':target, 'mask8':support,
                'x':as_tensor(camera,'cuda'), 'target':as_tensor(target,'cuda'),
                'mask':t(support.astype(np.float32))[None,None],
                'feature':t(feature.astype(np.float32))[None,None], 'interior':t(interior.astype(np.float32))[None,None],
                'valid7':t(erode(3).astype(np.float32))[None,None],
                'grid':t(grid112(refs[case['source_person_or_reference']]['matrix112']))[None]})
        initial = []
        baseline_folder = out / 'initial_baseline'; baseline_folder.mkdir()
        for begin in range(0,50,5):
            clock(); group = items[begin:begin+5]
            x = torch.cat([item['x'] for item in group]); mask = torch.cat([item['mask'] for item in group])
            with torch.no_grad():
                fresh = torch.where(mask.bool(), original(x), x).detach().clone()
                targets = identity.embedding(torch.cat([item['target'] for item in group]), mask, torch.cat([item['grid'] for item in group]))
            assert not torch.is_inference(fresh) and not fresh.requires_grad
            for slot, item in enumerate(group):
                item['base'] = fresh[slot:slot+1].clone()
                item['truth'] = targets[slot:slot+1].detach().clone()
                cid = item['case']['id']
                raw = item['base'][0].permute(1,2,0).cpu().numpy().copy()
                cached = np.load(parent / item['case']['raw_dgp'],allow_pickle=False)
                error = float(np.abs(raw[item['mask8']] - cached[item['mask8']]).max())
                assert error <= p['historical_CUDA_cache_tolerance'], 'Fresh canonical/legacy compatibility failed: ' + cid
                delivered = np.where(item['mask8'][...,None], np.floor(raw*np.float32(255)), item['camera']).astype(np.uint8)
                np.save(baseline_folder / (cid+'.npy'), raw, allow_pickle=False)
                Image.fromarray(delivered).save(baseline_folder / (cid+'.png'))
                np.save(baseline_folder / (cid+'_target_embedding.npy'), item['truth'][0].cpu().numpy().copy(), allow_pickle=False)
                initial.append({'id':cid,'historical_CUDA_cache_maximum_error':error,
                    'raw_file_sha256':sha(baseline_folder/(cid+'.npy')), 'PNG_file_sha256':sha(baseline_folder/(cid+'.png'))})
        # Preserve the exact fixed high-pass implementations without constructing the failed head.
        ns = {'torch':torch, 'F':F}
        exec(compile(ast.Module(body=filters,type_ignores=[]),'<pinned-V27-fixed-high-pass>', 'exec'),ns)
        class FixedFilter: pass
        fixed = FixedFilter()
        z = torch.arange(-6,7,dtype=torch.float32,device='cuda')
        fixed.kernel = torch.exp(-.5*(z/2).square()); fixed.kernel /= fixed.kernel.sum()
        fixed.reflect_indices = torch.cat((torch.arange(5,-1,-1),torch.arange(256),torch.arange(255,249,-1))).cuda()
        fixed.blur = MethodType(ns['blur'],fixed); fixed.high = MethodType(ns['high'],fixed)
        normalizers, cohort = cohort_normalizers(items, fixed)
        write(out / 'cohort_loss_setup.json', cohort)
        ns.update({'head':fixed,'identity':identity,'normalizers':normalizers})
        exec(compile(ast.Module(body=definitions,type_ignores=[]),'<pinned-V27-unchanged-loss-fields>', 'exec'),ns)
        total = np.zeros((7,609219), dtype=np.float64)
        scalar_values = np.zeros(7,dtype=np.float64)
        rows = []
        keys = ['x','base','target','mask','feature','interior','valid7','grid','truth','degraded_weight','clear_weight']
        gradient_folder = out / 'gradients'; gradient_folder.mkdir()
        for begin in range(0,50,5):
            clock(); group = items[begin:begin+5]
            b = {key:torch.cat([item[key] for item in group]) for key in keys}
            # The inference wrapper is not invoked for the candidate graph.
            prediction = candidate(b['x'],b['mask'])
            assert prediction.requires_grad and not torch.is_inference(prediction), 'Candidate graph was suppressed'
            assert torch.equal(prediction.detach(),b['base']), 'Initial same-batch raw output must exactly equal retained DGP'
            terms = objective_terms(b,prediction,identity,ns['mean'],ns['feature_errors'],ns['ssim'],normalizers)
            assert list(terms) == p['terms']
            matrix = np.empty((7,609219),dtype=np.float64)
            batch_values = []
            for i,name in enumerate(p['terms']):
                clock(); scalar = terms[name].mean()/10
                pieces = torch.autograd.grad(scalar,parameters,retain_graph=i<6,create_graph=False,allow_unused=False)
                progress['component_gradient_calls'] += 1
                assert len(pieces) == 14 and all(bool(torch.isfinite(g).all()) for g in pieces)
                if name in p['initial_preservation_terms_exact_zero']:
                    assert float(scalar.detach()) == 0 and all(torch.count_nonzero(g) == 0 for g in pieces), 'Initial preservation value/all14 gradients must remain exactly zero: ' + name
                matrix[i] = torch.cat([g.detach().reshape(-1).double() for g in pieces]).cpu().numpy()
                value = float(scalar.detach()); scalar_values[i] += value; batch_values.append(value)
            total += matrix
            path = gradient_folder / ('batch'+str(begin//5)+'.npy')
            np.save(path,matrix,allow_pickle=False)
            row = {'batch':begin//5,'ids':[item['case']['id'] for item in group],
                   'component_values':batch_values,'component_norms':np.linalg.norm(matrix,axis=1).tolist(),
                   'gradient_array_sha256':sha(path),'initial_raw_and_PNG_parity_exact':True,
                   'initial_all14_preservation_gradients_exact_zero':True}
            rows.append(row)
            assert state_hash(original.net) == state_hash(candidate.net) == original_state and state_hash(identity) == identity_state
            assert all(v.grad is None for v in candidate.parameters())
            assert all(not v.requires_grad and v.grad is None for v in candidate.net.fpn.parameters())
            print(json.dumps({'batch':begin//5+1,'of':10,'component_norms':row['component_norms'],'gradient_queries':progress['component_gradient_calls'],'seconds':time.monotonic()-start}),flush=True)
        norms = np.linalg.norm(total,axis=1)
        gram = total @ total.T
        denominator = norms[:,None]*norms[None,:]
        cosines = np.divide(gram,denominator,out=np.zeros_like(gram),where=denominator>0)
        partitions = {}
        for parameter in layout:
            block = total[:,parameter['start']:parameter['end']]
            partitions[parameter['name']] = {'component_norms':np.linalg.norm(block,axis=1).tolist(),
                'improvement_gradient_norm':float(np.linalg.norm(block[:3].sum(0))),
                'total_gradient_norm':float(np.linalg.norm(block.sum(0)))}
        np.save(out / 'gradient_components.npy',total,allow_pickle=False)
        write(out / 'gradient_summary.json',{'complete':True,'terms':p['terms'],'parameter_layout':layout,
            'batches':rows,'component_values':scalar_values.tolist(),'objective':float(scalar_values.sum()),
            'component_norms':norms.tolist(),'component_gram':gram.tolist(),'component_cosines':cosines.tolist(),
            'per_parameter_gradients':partitions,'gradient_array_sha256':sha(out/'gradient_components.npy')})
        assert all(row['improvement_gradient_norm'] > 0 for row in partitions.values()), 'All14 original decoder tensors need finite nonzero improvement gradients before any optimizer'
        assert progress == {'reference_DGP_forwards':10,'candidate_DGP_forwards':10,'recognizer_forwards':20,
                            'component_gradient_calls':70,'optimizer_updates':0,'epochs':0}
        assert all(not v.requires_grad and v.grad is None for v in original.parameters())
        assert all(not v.requires_grad and v.grad is None for v in identity.parameters())
        assert state_hash(original.net) == state_hash(candidate.net) == original_state and state_hash(identity) == identity_state
        parent_check(parent,p); clock()
        write(out / 'results.json',{'complete':True,'protocol_sha256':pin,'seconds':time.monotonic()-start,
            'scope':'Initial original-decoder forward/differentiation proof only; no restoration gain or training acceptance',
            'parameter_layout':layout,'decoder_parameters':609219,'decoder_parameter_tensors':14,
            'fresh_canonical_initial_rows':initial,'raw_and_PNG_parity_exact_cases':50,
            'normalization':'Pinned as_tensor on CUDA; fresh same-batch baseline; legacy cached raw compatibility bounded separately',
            'all14_improvement_gradients_nonzero':True,'initial_preservation_gradients_exact_zero_all_batches':True,
            'DGP_state_before_after':original_state,'recognizer_state_before_after':identity_state,
            'original_DGP_provenance':provenance, **progress,
            'peak_allocated_VRAM_bytes':torch.cuda.max_memory_allocated(),'original_V27_failure_retained':True,
            'optimizer_constructed':False,'new_checkpoint_created':False,'automatic_follow_on':False,
            'independent_audit_pending':True,'native_or_reserved_used':False,'app_promotion':False,'goal_complete':False})
    except BaseException as exc:
        write(out / 'failure.json',{'complete':False,'protocol_sha256':pin,'seconds':time.monotonic()-start,
            **progress,'cause':str(exc),'traceback':traceback.format_exc(),'resume_permitted':False,
            'optimizer_constructed':False,'new_checkpoint_created':False,'app_promotion':False,'goal_complete':False})
        raise


def export(root,parent,p,pin):
    vm_scope(root,parent)
    start = time.monotonic()
    name = 'cctv-dgp-original-decoder-gradient-v1-results.tar.gz'
    destination = Path.home() / name
    assert not destination.exists() and not Path(str(destination)+'.sha256').exists(), 'Preserve previous export'
    files = [root/'protocol.json'] + [root/name for name in p['assets_sha256']]
    files += [path for path in (root/'outputs').rglob('*') if path.is_file()] if (root/'outputs').exists() else []
    files += [root/name for name in ['trainer.log','trainer_exit_code.txt','supervisor_receipt.json'] if (root/name).exists()]
    assert all(not path.is_symlink() and path.resolve().is_relative_to(root) for path in files)
    assert sum(path.stat().st_size for path in files) <= p['budgets']['maximum_export_uncompressed_bytes']
    write(root/'export_manifest.json',{'complete':True,'protocol_sha256':pin,
        'files_sha256':{path.relative_to(root).as_posix():sha(path) for path in files},'quality_acceptance_not_implied':True})
    files.append(root/'export_manifest.json')
    with tarfile.open(destination,'x:gz',compresslevel=3) as tar:
        for path in files:
            assert time.monotonic()-start < p['budgets']['export_seconds'], 'Decoder proof export cap90 seconds'
            tar.add(path,arcname='cctv_dgp_original_decoder_gradient_v1_return/'+path.relative_to(root).as_posix(),recursive=False)
    digest = sha(destination)
    with Path(str(destination)+'.sha256').open('x',encoding='ascii',newline='\n') as f:
        f.write(digest+'  '+name+'\n')
    receipt = {'complete':True,'archive_sha256':digest,'bytes':destination.stat().st_size,'seconds':time.monotonic()-start,
        'training_success_not_implied':True,'optimizer_updates':0,'run_results_present':(root/'outputs/results.json').exists(),
        'failure_present':(root/'outputs/failure.json').exists()}
    write(Path.home()/'cctv-dgp-original-decoder-gradient-v1-export.json',receipt)
    print(json.dumps(receipt))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--protocol-sha',required=True)
    parser.add_argument('--parent',type=Path,default=Path.home()/'forensic-dgp/cctv_dgp_feature_skips_vm_v27')
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ['verify-transfer','run','export','record-supervision']:
        modes.add_argument('--'+mode,action='store_true')
    parser.add_argument('--supervisor-start',type=float)
    parser.add_argument('--supervisor-end',type=float)
    parser.add_argument('--trainer-exit',type=int)
    a = parser.parse_args(); root = a.root.resolve(); parent = a.parent.resolve(); p = verify(root,a.protocol_sha)
    if a.verify_transfer:
        base = parent_check(parent,p)
        print(json.dumps({'complete':True,'packet_assets':len(p['assets_sha256']),'original_assets':len(base['assets_sha256']),
                          'cases':50,'optimizer_updates':0,'neural_or_gradient_calls':0})); return
    if a.record_supervision:
        vm_scope(root,parent)
        assert all(v is not None and math.isfinite(v) for v in [a.supervisor_start,a.supervisor_end])
        assert a.supervisor_end >= a.supervisor_start and a.trainer_exit is not None
        elapsed = a.supervisor_end-a.supervisor_start
        write(root/'supervisor_receipt.json',{'complete':True,'protocol_sha256':a.protocol_sha,'seconds':elapsed,
            'cap_seconds':660,'kill_grace_seconds':30,'within_external_bound':elapsed<=690,'trainer_exit_code':a.trainer_exit,
            'optimizer_updates':0,'quality_acceptance_not_implied':True}); return
    if a.export:
        export(root,parent,p,a.protocol_sha); return
    vm_scope(root,parent)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Decoder proof worker cap600 seconds')))
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('Decoder proof external deadline')))
    signal.alarm(600)
    run(root,parent,p,a.protocol_sha)


if __name__ == '__main__':
    main()
