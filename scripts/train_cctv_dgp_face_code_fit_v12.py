"""VM-only 300-update, training-only DGP face-code conditioner fitting diagnostic."""
import argparse
import json
from pathlib import Path
import sys
import time

CONTEXT = {'out': None, 'model': None, 'updates': 0, 'backwards': 0}


def train(root, expected_sha):
    sys.path.insert(0, str(root))
    from cctv_dgp_face_code_fit_v12 import require_vm, verify, read, write, sha, require
    require_vm(root)  # Before model construction, output creation or any backward.
    p = verify(root, expected_sha); design = p['design']
    import numpy as np
    from PIL import Image
    import torch
    from torch.nn import functional as F
    from dgp_face_code_conditioner_v11 import DGPFaceCodePrior, load_teacher, teacher_codes
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from pretrained_face_restoration import load_face_restorer
    from cctv_dgp_pilot import (FixedObservedIdentity, grid112, state_hash, exported_pixel_metrics)
    from face_prior_grid_v10 import render_grid

    torch.set_num_threads(4); torch.manual_seed(design['seed'])
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic(); deadline = started + design['trainer_cap_seconds']
    out = root / 'outputs/cctv_dgp_face_code_fit_v12'
    require(not out.exists(), 'Preserve previous/partial run; no resume')
    out.mkdir(parents=True); CONTEXT['out'] = out
    dgp, _ = load_frozen_dgp_restorer(root / p['weights']['dgp'], expected_sha256=p['assets_sha256'][p['weights']['dgp']], device='cuda')
    baseline, prior_provenance = load_face_restorer(root / p['weights']['prior'], device='cuda')
    teacher = load_teacher(root / p['weights']['teacher'], p['assets_sha256'][p['weights']['teacher']], device='cuda')
    model = DGPFaceCodePrior(dgp.net, baseline.net).cuda(); CONTEXT['model'] = model
    identity = FixedObservedIdentity(root / p['weights']['arcface'], 'cuda')
    refs = {r['id']: r for r in p['references']}; cases = {c['id']: c for c in p['cases']}
    frozen = {'dgp': model.dgp, 'prior': model.prior, 'teacher': teacher, 'recognizer': identity}
    frozen_states = {name: state_hash(net) for name, net in frozen.items()}
    initial_state = state_hash(model.conditioner)
    require(sum(v.numel() for v in model.conditioner.parameters()) == design['trainable_parameters'], 'Conditioner size differs')
    artifacts = {}; counts = {}
    modules = {'dgp': model.dgp, 'conditioner': model.conditioner,
               'prior_encoder': model.prior.encoder.blocks[0], 'prior_transformer_head': model.prior.idx_pred_layer,
               'prior_generator': model.prior.generator.blocks[0], 'teacher_encoder': teacher.encoder,
               'teacher_quantizer': teacher.quantize, 'recognizer': identity.encoder}
    handles = []
    for key, module in modules.items():
        counts[key] = 0
        def hook(_, __, ___, key=key): counts[key] += 1
        handles.append(module.register_forward_hook(hook))
    def clock():
        torch.cuda.synchronize()
        if time.monotonic() > deadline:
            raise TimeoutError('V12 trainer exceeded600 seconds')
        require(torch.cuda.max_memory_allocated() <= design['peak_vram_cap_bytes'], 'Peak VRAM exceeded20 GiB')
    def bind(name): artifacts[name] = sha(out / name)
    def save_array(name, array):
        (out / name).parent.mkdir(parents=True, exist_ok=True); np.save(out / name, array, allow_pickle=False); bind(name)
    def image(path):
        with Image.open(root / path) as im:
            require(im.mode == 'RGB' and im.size == (256, 256), 'Require exact256 RGB PNG')
            return np.asarray(im).copy()
    def tensor(rgb): return torch.from_numpy(rgb.astype(np.float32) / 255).permute(2, 0, 1)[None].cuda()
    support = {}; clean = {}; code_targets = {}; feature_targets = {}; target_embeddings = {}; grids = {}; inputs = {}
    loading_seconds = time.monotonic() - started
    write(out / 'execution.json', {'protocol_sha256': expected_sha, 'torch': torch.__version__, 'device': 'cuda',
          'gpu': torch.cuda.get_device_name(0), 'loading_seconds': loading_seconds, 'frozen_state_hashes': frozen_states,
          'initial_conditioner_state_hash': initial_state, 'prior_provenance': prior_provenance, 'design': design,
          'native_used': False, 'validation_used': False, 'production_promoted': False})
    try:
        with torch.no_grad():
            for ref in p['references']:
                clock(); rid = ref['id']; clean[rid] = image(ref['target'])
                mask = np.asarray(Image.open(root / ref['observed'])) > 0
                support[rid] = torch.from_numpy(mask.astype(np.float32))[None, None].cuda()
                grids[rid] = torch.from_numpy(grid112(ref['matrix112']))[None].cuda()
                codes, _ = teacher_codes(teacher, tensor(clean[rid]))
                code_targets[rid] = codes
                feature_targets[rid] = model.prior.quantize.get_codebook_feat(codes, [1, 16, 16, 256])
                target_embeddings[rid] = identity.embedding(tensor(clean[rid]), support[rid], grids[rid])[0].cpu().numpy()
                save_array('teacher/' + rid + '_codes.npy', codes.cpu().numpy())
                save_array('teacher/' + rid + '_features.npy', feature_targets[rid].cpu().numpy())
                save_array('teacher/' + rid + '_embedding.npy', target_embeddings[rid])
            for c in p['cases']: inputs[c['id']] = tensor(image(c['input']))
        def objective(ids, capture=False):
            batch = torch.cat([inputs[c] for c in ids], 0)
            rid = [cases[c]['reference_id'] for c in ids]
            codes = torch.cat([code_targets[k] for k in rid], 0)
            targets = torch.cat([feature_targets[k] for k in rid], 0)
            mask = torch.cat([support[k] for k in rid], 0)
            token_mask = F.interpolate(mask, (16, 16), mode='nearest').flatten(1)
            logits, features = model.training_codes(batch)
            ce_tokens = F.cross_entropy(logits.permute(0, 2, 1), codes, reduction='none')
            feature_error = (features - targets).square().mean(1).flatten(1)
            ce = (ce_tokens * token_mask).sum() / token_mask.sum()
            mse = (feature_error * token_mask).sum() / token_mask.sum()
            accuracy = ((logits.argmax(2) == codes).float() * token_mask).sum() / token_mask.sum()
            loss = design['lambda_code_ce'] * ce + design['lambda_feature_mse'] * mse
            require(torch.isfinite(loss).item(), 'Nonfinite face-code objective')
            metrics = {'code_ce': ce.item(), 'feature_mse': mse.item(), 'code_accuracy': accuracy.item()}
            if capture:
                metrics['_logits'] = logits.detach()
                metrics['_features'] = features.detach()
            return loss, metrics

        schedule = read(root / 'schedule_v12.json')['steps']
        first = inputs[schedule[0]['case_ids'][0]]
        with torch.inference_mode():
            original = baseline(first, fidelity=1.0); zero = model(first, fidelity=1.0)
            deviation = (original - zero).abs().max().item()
            require(deviation <= 2e-6 and torch.equal(torch.floor(original * 255), torch.floor(zero * 255)), 'CUDA zero-conditioner parity failed')
        model.enable_vm_training(root)
        require(all(not param.requires_grad for net in frozen.values() for param in net.parameters()), 'Frozen parameter enabled')
        # Zero-update backward preflight. Only zero-initialized projection receives
        # a nonzero first gradient; trunk becomes reachable after an actual update.
        preflight_state = state_hash(model.conditioner)
        loss, metrics = objective(schedule[0]['case_ids']); loss.backward(); CONTEXT['backwards'] += 1
        norms = {name: (None if param.grad is None else param.grad.norm().item()) for name, param in model.conditioner.named_parameters()}
        require(all(v is not None and np.isfinite(v) for v in norms.values()), 'Missing/nonfinite conditioner gradient')
        require(norms['projection.weight'] > 0 and norms['projection.bias'] > 0, 'Projection gradient does not reach our conditioner')
        require(all(param.grad is None for net in frozen.values() for param in net.parameters()), 'Frozen pretrained parameter received gradient')
        require(state_hash(model.conditioner) == preflight_state and
                {name: state_hash(net) for name, net in frozen.items()} == frozen_states, 'Preflight changed weights')
        for param in model.conditioner.parameters(): param.grad = None
        del loss
        clock(); write(out / 'cuda_preflight.json', {'complete': True, 'loss': metrics, 'gradient_norms': norms,
              'conditioner_state_unchanged': True, 'frozen_states_unchanged': True, 'optimizer_updates': 0,
              'backward_calls': 1, 'zero_conditioner_max_float_deviation': deviation,
              'peak_vram_bytes': torch.cuda.max_memory_allocated(), 'seconds': time.monotonic()-started})
        print('V12 CUDA gradient/VRAM preflight passed: one backward, zero updates', flush=True)

        def snapshot(update):
            clock(); prefix = 'update' + str(update); rows = []
            checkpoint = prefix + '/conditioner.pth'; (out / checkpoint).parent.mkdir(parents=True)
            torch.save({k: v.detach().cpu().clone() for k, v in model.conditioner.state_dict().items()}, out / checkpoint); bind(checkpoint)
            with torch.no_grad():
                for c in p['cases']:
                    clock(); cid = c['id']; rid = c['reference_id']; _, code_metrics = objective([cid], capture=True)
                    logits_name = prefix + '/code_probes/' + cid + '_logits.npy'
                    features_name = prefix + '/code_probes/' + cid + '_features.npy'
                    save_array(logits_name, code_metrics.pop('_logits').cpu().numpy())
                    save_array(features_name, code_metrics.pop('_features').cpu().numpy())
                    for fidelity in design['fidelities']:
                        arm = 'w' + str(int(fidelity)); prediction = model(inputs[cid], fidelity=fidelity)
                        raw = prediction[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
                        require(np.isfinite(raw).all() and 0 <= raw.min() <= raw.max() <= 1, 'Invalid render float')
                        raw_name = prefix + '/' + arm + '/raw/' + cid + '.npy'; save_array(raw_name, raw)
                        low = image(c['input']); observed = support[rid][0, 0].cpu().numpy() > 0
                        png = np.floor(raw * 255).astype(np.uint8); png[~observed] = low[~observed]
                        png_name = prefix + '/' + arm + '/images/' + cid + '.png'; (out / png_name).parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(png).save(out / png_name); bind(png_name)
                        vec = identity.embedding(tensor(png), support[rid], grids[rid])[0].cpu().numpy()
                        vec_name = prefix + '/' + arm + '/embeddings/' + cid + '.npy'; save_array(vec_name, vec)
                        rows.append({**{k: c[k] for k in ['id', 'reference_id', 'source', 'profile']}, 'fidelity': fidelity,
                                     **code_metrics, **exported_pixel_metrics(png, clean[rid], observed),
                                     'code_logits': logits_name, 'code_features': features_name,
                                     'ArcFace_observed_fixed': float(np.clip(vec @ target_embeddings[rid], -1, 1)),
                                     'prediction': png_name, 'raw': raw_name, 'embedding': vec_name})
            write(out / (prefix + '/metrics.json'), {'update': update, 'checkpoint': checkpoint,
                  'conditioner_state_hash': state_hash(model.conditioner), 'rows': rows}); bind(prefix + '/metrics.json')
            print(f'snapshot {update}:100 training-only renders elapsed={time.monotonic()-started:.1f}s', flush=True)
            return {'update': update, 'checkpoint': checkpoint, 'metrics': prefix + '/metrics.json'}

        snapshots = [snapshot(0)]
        optimizer = torch.optim.Adam(model.trainable_parameters(), lr=design['learning_rate'], betas=tuple(design['betas']))
        traces = []; training_start = time.monotonic(); timing = None
        for i, step in enumerate(schedule, 1):
            clock(); optimizer.zero_grad(set_to_none=True)
            loss, metrics = objective(step['case_ids']); loss.backward(); CONTEXT['backwards'] += 1
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), design['gradient_clip_norm']).item()
            require(np.isfinite(gradient_norm), 'Nonfinite gradient norm')
            require(all(v.grad is None for net in frozen.values() for v in net.parameters()), 'Frozen gradient detected')
            optimizer.step(); CONTEXT['updates'] = i
            traces.append({'update': i, 'epoch': step['epoch'], 'case_ids': step['case_ids'], **metrics,
                           'loss': loss.item(), 'gradient_norm_before_clip': gradient_norm,
                           'seconds': time.monotonic()-started})
            with (out / 'update_trace.jsonl').open('a' if i > 1 else 'x', encoding='utf-8', newline='\n') as trace:
                trace.write(json.dumps(traces[-1], allow_nan=False) + '\n')
            del loss
            if i == design['timing_update']:
                clock(); elapsed = time.monotonic() - training_start
                # Two remaining snapshots use the measured baseline snapshot budget.
                baseline_seconds = traces[0]['seconds'] - read(out/'cuda_preflight.json')['seconds']
                projected = (time.monotonic()-started) + elapsed / i * (300-i) + 2*baseline_seconds + 20
                timing = {'update': i, 'training_seconds': elapsed, 'projected_total_seconds': projected, 'cap_seconds': 600}
                write(out/'timing.json',timing)
                require(projected <= 600, 'Projected finite fit exceeds600 seconds; stop')
            if i % 25 == 0 or i == 1: print(f'V12 update{i}/300 loss={traces[-1]["loss"]:.5f} elapsed={time.monotonic()-started:.1f}s', flush=True)
            if i in design['snapshots'][1:]: snapshots.append(snapshot(i))
        clock(); after = {name: state_hash(net) for name, net in frozen.items()}
        require(after == frozen_states, 'Frozen prior/DGP/teacher/recognizer changed')
        require(state_hash(model.conditioner) != initial_state, 'Our conditioner did not change')
        require(CONTEXT['updates'] == 300 and CONTEXT['backwards'] == 301, 'Actual update/backward counts differ')
        expected_counts = {'dgp': 752, 'conditioner': 752, 'prior_encoder': 753,
                           'prior_transformer_head': 753, 'prior_generator': 302,
                           'teacher_encoder': 10, 'teacher_quantizer': 10, 'recognizer': 310}
        require(counts == expected_counts, 'Forward count audit failed')
        write(out/'update_trace.json', {'steps': traces}); bind('update_trace.json'); bind('update_trace.jsonl')
        write(out/'neural_execution_receipt.json', {'complete': True, 'frozen_before': frozen_states, 'frozen_after': after,
              'counts': counts, 'expected_counts': expected_counts, 'backward_calls': 301, 'optimizer_updates': 300,
              'initial_conditioner_state_hash': initial_state, 'final_conditioner_state_hash': state_hash(model.conditioner),
              'peak_vram_bytes': torch.cuda.max_memory_allocated(), 'seconds_before_grids': time.monotonic()-started})
        # Five ten-reference/profile grids per fidelity. Input, initial, two learned
        # snapshots and clean target; all cells are original256 pixels.
        grids_saved = []
        for fidelity in design['fidelities']:
            arm = 'w' + str(int(fidelity))
            for profile in p['profiles']:
                rows = []
                for ref in p['references']:
                    c = next(c for c in p['cases'] if c['reference_id'] == ref['id'] and c['profile'] == profile)
                    cells = [image(c['input'])]
                    for update in design['snapshots']:
                        with Image.open(out / f'update{update}/{arm}/images/{c["id"]}.png') as im: cells.append(np.asarray(im).copy())
                    cells.append(clean[ref['id']]); rows.append({'id': c['id'], 'images': cells})
                name = 'grids/' + arm + '_' + profile + '_10_rows.png'; (out / name).parent.mkdir(exist_ok=True)
                render_grid(['camera_input','initial_prior','conditioner_100','conditioner_300','clean_training_target'],rows).save(out/name)
                bind(name); grids_saved.append(name)
        clock(); verify(root,expected_sha)
        results = {'complete': True, 'protocol_sha256': expected_sha, 'snapshots': snapshots, 'grids': grids_saved,
                   'artifacts_sha256': artifacts, 'optimizer_updates': 300, 'backward_calls': 301,
                   'training_exposures': 600, 'seconds': time.monotonic()-started, 'counts': counts,
                   'peak_vram_bytes': torch.cuda.max_memory_allocated(), 'native_used': False, 'validation_used': False,
                   'native_reserved_used': False, 'production_promoted': False, 'checkpoint_selected': False,
                   'best_checkpoint_created': False, 'independent_final_review_pending': True,
                   'limitation': 'Fitting ten training references only; no held-out generalization or useful CCTV upgrade established.'}
        write(out/'results.json',results); print({'complete':True,'updates':300,'seconds':results['seconds'],'production_promoted':False},flush=True)
    finally:
        for handle in handles: handle.remove()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--expected-protocol-sha',required=True)
    args=parser.parse_args();root=args.root.resolve()
    try: train(root,args.expected_protocol_sha)
    except Exception as error:
        out=CONTEXT['out']
        if out is not None:
            from cctv_dgp_face_code_fit_v12 import write
            if CONTEXT['model'] is not None:
                import torch
                path=out/'partial_conditioner.pth'
                if not path.exists():torch.save({k:v.detach().cpu().clone() for k,v in CONTEXT['model'].conditioner.state_dict().items()},path)
            if not (out/'failure.json').exists():write(out/'failure.json',{'complete':False,'error_type':type(error).__name__,'error':str(error),
                'updates_recorded':CONTEXT['updates'],'backwards_recorded':CONTEXT['backwards'],'resume_permitted':False,'production_promoted':False})
        raise
