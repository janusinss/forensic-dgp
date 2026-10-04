"""Finite local inference: zero-conditioner parity and clean-code capacity only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image

import cctv_dgp_face_prior_v10 as inherited

OUT = ROOT / 'outputs/dgp_face_code_prior_checks_v11'
TEACHER = 'outputs/codeformer_teacher_pretrained_v1/vqgan_code1024.pth'
TEACHER_SHA = '4d1c6741b3cffcbdc2cd1a12b2c3c2442282e042d5de66909cb643d4fa31b20f'
DGP = 'outputs/cctv_dgp_mixed_return_v9/outputs/cctv_dgp_mixed_v9/checkpoints/baseline.pth'
DGP_SHA = '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'
CAP = 180


def prepare():
    inherited.require(not OUT.exists(), 'Preserve prior/partial checks')
    p = inherited.read(inherited.MIXED / 'mixed_protocol_v9.json')
    inherited.require(inherited.sha(inherited.MIXED / 'mixed_protocol_v9.json') ==
                      '6cd9ac8d5f3bf27be773979c2f628e33f5d3cf09748452eeab91a65b05b01c70', 'Parent V9 changed')
    candidates = [r for r in p['references'] if r['role'] == 'train']
    references = []
    for source in ['dataset/asian_faces', 'dataset/thumbnails128x128']:
        rows = sorted([r for r in candidates if r['source'] == source],
                      key=lambda r: hashlib.sha256(('face-code-v11:' + r['id']).encode()).hexdigest())
        references.extend(rows[:5])
    inherited.require(len(references) == 10 and all(r['role'] == 'train' for r in references), 'Training-only capacity cohort')
    assets = {TEACHER: TEACHER_SHA, DGP: DGP_SHA,
              inherited.CF_WEIGHT: inherited.PINS[inherited.CF_WEIGHT],
              inherited.IDENTITY_WEIGHT: inherited.PINS[inherited.IDENTITY_WEIGHT]}
    for name in ['dgp_face_code_conditioner_v11.py', 'scripts/check_dgp_face_code_prior_v11.py',
                 'tests/test_dgp_face_code_conditioner_v11.py', 'outputs/dgp_face_code_conditioner_v11_contract_tests.json',
                 'outputs/codeformer_teacher_pretrained_v1/acquisition.json',
                 'outputs/codeformer_teacher_component_diagnostic_v11.json',
                 'pretrained_face_restoration.py', 'dgp_frozen_inference_v2.py', 'dgp_face_restoration.py',
                 'cctv_dgp_frozen_norm.py', 'cctv_dgp_pilot.py', 'cctv_dgp_face_prior_v10.py',
                 'face_prior_grid_v10.py', 'outputs/cctv_dgp_mixed_vm_v9_r2/mixed_protocol_v9.json']:
        assets[name] = inherited.sha(ROOT / name)
    for folder in ['models', 'third_party/codeformer']:
        for path in (ROOT / folder).glob('*'):
            if path.is_file():
                assets[path.relative_to(ROOT).as_posix()] = inherited.sha(path)
    refs = []
    for r in references:
        files = {}
        for key in ['target', 'observed']:
            path = inherited.MIXED / r[key]
            inherited.require(inherited.sha(path) == p['assets_sha256'][r[key]], 'Training target changed')
            files[key] = path.relative_to(ROOT).as_posix()
            assets[files[key]] = inherited.sha(path)
        refs.append({**r, 'files': files})
    plan = {'format': 'dgp-face-code-interface-and-capacity-v11', 'date': '2026-10-04',
            'assets_sha256': assets, 'references': refs, 'frozen_before_outputs': True,
            'selection': 'Five training references/source, ascending SHA256(face-code-v11:reference_id); no output filtering.',
            'architecture': 'Frozen retained V2 DGP RGB plus raw RGB -> our 455072-parameter convolutional residual at original prior encoder16x16; frozen CodeFormer transformer/codebook/renderer.',
            'zero_conditioner': 'Two artificial RGB inputs, fidelity w1, same crop/normalization/renderer; maximum float deviation <= 2e-6 and exactly identical floor PNGs; no newly generated dataset input.',
            'teacher_policy': 'Official clean VQGAN encoder/quantizer; teacher code indices rendered by frozen CodeFormer generator at w0 without target AdaIN or fidelity skip features. Oracle uses clean target and is NOT restoration.',
            'teacher_compatibility': 'Release weights differ slightly; all tensors allclose rtol/atol1e-5, code cosine min0.9999993, 1024/1024 nearest-index agreement. Preserve both weights unchanged.',
            'capacity_arms': ['codeformer_clean_w1', 'teacher_vq_reconstruction', 'clean_code_oracle_w0'],
            'decision': 'Inspect both-source target layout/expression against the oracle. Capacity can justify a VM-only conditioning diagnostic, not adoption or restored identity. Reject demonstrated severe prior representation mismatch before training.',
            'budget': {'cpu_threads': 4, 'runtime_after_loading_seconds': CAP, 'backward_calls': 0, 'optimizer_updates': 0,
                       'dgp_forwards': 2, 'conditioner_forwards': 2, 'prior_encoder_forwards': 14,
                       'prior_generator_forwards': 24, 'teacher_encoder_forwards': 10, 'teacher_quantizer_forwards': 10,
                       'teacher_generator_forwards': 10, 'recognizer_forwards': 40},
            'native_used': False, 'validation_used': False, 'native_reserved_used': False,
            'training': False, 'production_promotion': False, 'independent_final_review_pending': True}
    for name, pin in assets.items():
        inherited.require(inherited.sha(inherited.safe(ROOT, name)) == pin, 'Changed asset: ' + name)
    tests = inherited.read(ROOT / 'outputs/dgp_face_code_conditioner_v11_contract_tests.json')
    inherited.require(tests['complete'], 'Require passing interface tests')
    OUT.mkdir(); inherited.write(OUT / 'frozen_plan.json', plan)
    (OUT / 'frozen_plan.sha256').write_text(inherited.sha(OUT / 'frozen_plan.json') + '\n', encoding='ascii')
    print({'prepared': True, 'plan_sha256': inherited.sha(OUT / 'frozen_plan.json'),
           'training_references': 10, 'validation_or_native_used': False, 'optimizer_updates': 0})


def verify():
    inherited.require(inherited.sha(OUT / 'frozen_plan.json') == (OUT / 'frozen_plan.sha256').read_text().strip(), 'Plan fingerprint differs')
    p = inherited.read(OUT / 'frozen_plan.json')
    inherited.require(p['format'] == 'dgp-face-code-interface-and-capacity-v11' and not p['training'] and
                      not p['native_used'] and not p['validation_used'] and not p['native_reserved_used'], 'Scope changed')
    inherited.require(len(p['references']) == 10 and all(r['role'] == 'train' for r in p['references']), 'Role differs')
    for name, pin in p['assets_sha256'].items():
        inherited.require(inherited.sha(inherited.safe(ROOT, name)) == pin, 'Changed frozen asset: ' + name)
    return p


def state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run():
    p = verify(); out = OUT / 'run'
    inherited.require(not out.exists(), 'Preserve prior/partial inference; no resume')
    import torch
    from dgp_face_code_conditioner_v11 import (DGPFaceCodePrior, load_teacher, teacher_codes,
                                              from_prior_canvas)
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from pretrained_face_restoration import load_face_restorer
    from cctv_dgp_pilot import FixedObservedIdentity
    from face_prior_grid_v10 import render_grid
    from scripts.recover_cctv_dgp_face_prior_v10 import pixels

    torch.set_num_threads(4); torch.manual_seed(20261004)
    loading = time.monotonic()
    dgp, _ = load_frozen_dgp_restorer(ROOT / DGP, expected_sha256=DGP_SHA)
    baseline, provenance = load_face_restorer(ROOT / inherited.CF_WEIGHT)
    teacher = load_teacher(ROOT / TEACHER, TEACHER_SHA)
    model = DGPFaceCodePrior(dgp.net, baseline.net)
    identity = FixedObservedIdentity(ROOT / inherited.IDENTITY_WEIGHT, device='cpu')
    models = {'dgp': dgp.net, 'prior': baseline.net, 'teacher': teacher,
              'conditioner': model.conditioner, 'recognizer': identity}
    before = {name: state_hash(net) for name, net in models.items()}
    counts = {k: 0 for k in p['budget'] if k.endswith('_forwards')}
    modules = {'dgp_forwards': dgp.net, 'conditioner_forwards': model.conditioner,
               'prior_encoder_forwards': baseline.net.encoder.blocks[0],
               'prior_generator_forwards': baseline.net.generator.blocks[0],
               'teacher_encoder_forwards': teacher.encoder, 'teacher_quantizer_forwards': teacher.quantize,
               'teacher_generator_forwards': teacher.generator, 'recognizer_forwards': identity}
    handles = []
    for key, module in modules.items():
        def hook(_, __, ___, key=key): counts[key] += 1
        handles.append(module.register_forward_hook(hook))
    loading_seconds = time.monotonic() - loading
    out.mkdir(); started = time.monotonic(); artifacts = {}
    inherited.write(out / 'execution.json', {'plan_sha256': inherited.sha(OUT / 'frozen_plan.json'),
                    'torch': torch.__version__, 'device': 'cpu', 'cpu_threads': 4, 'loading_seconds': loading_seconds,
                    'before_state_hashes': before, 'training': False, 'local_backward_calls': 0, 'local_optimizer_updates': 0,
                    'prior_provenance': provenance})
    def clock(): inherited.require(time.monotonic() - started <= CAP, 'Exceeded180-second inference cap')
    def bind(name): artifacts[name] = inherited.sha(out / name)
    def raw(net_float): return net_float[0].detach().cpu().permute(1, 2, 0).contiguous().numpy().astype(np.float32)
    try:
        parity = []
        for i in range(2):
            clock()
            axis = torch.linspace(0, 1, 256)
            xx = axis[None, :].repeat(256, 1); yy = axis[:, None].repeat(1, 256)
            rgb = torch.stack([xx, yy, (xx + yy) / 2])[None] if i == 0 else torch.stack([
                .5 + .35 * torch.sin(xx * 8), .5 + .35 * torch.cos(yy * 6), .3 + .6 * xx * yy])[None]
            original = baseline(rgb, fidelity=1.0); candidate = model(rgb, fidelity=1.0)
            difference = (original - candidate).abs().max().item()
            same_png = torch.equal(torch.floor(original * 255), torch.floor(candidate * 255))
            inherited.require(difference <= 2e-6 and same_png, 'Zero-conditioner baseline parity failed')
            inherited.require(not candidate.requires_grad and not any(v.requires_grad for v in model.parameters()), 'Unexpected local gradient graph')
            parity.append({'artificial_case': i, 'max_abs_float_difference': difference, 'floor_png_identical': same_png})
        rows = []
        for r in p['references']:
            clock(); target = inherited.rgb(ROOT / r['files']['target'])
            observed = np.asarray(Image.open(ROOT / r['files']['observed'])) > 0
            clean = torch.from_numpy(target.astype(np.float32) / 255).permute(2, 0, 1)[None]
            matrix = torch.tensor(r['matrix112'], dtype=torch.float32)[None]
            with torch.inference_mode():
                codes, quantized = teacher_codes(teacher, clean)
                outputs = {'codeformer_clean_w1': baseline(clean, fidelity=1.0),
                           'teacher_vq_reconstruction': from_prior_canvas(teacher.generator(quantized)),
                           'clean_code_oracle_w0': model.clean_code_oracle(codes)}
                gt_vec = identity(clean, matrix)[0].cpu().numpy()
                name = 'latents/' + r['id'] + '.npz'; (out / name).parent.mkdir(exist_ok=True)
                np.savez(out / name, codes=codes.cpu().numpy(), quantized=quantized.cpu().numpy(), target_embedding=gt_vec); bind(name)
                row = {'id': r['id'], 'source': r['source'], 'role': 'train', 'target': r['files']['target'],
                       'latents': name, 'arms': {}}
                for arm, value in outputs.items():
                    value_float = raw(value); png = inherited.png(value_float, target, observed)
                    raw_name = f'raw/{r["id"]}_{arm}.npy'; png_name = f'images/{r["id"]}_{arm}.png'
                    for directory in ['raw', 'images', 'embeddings']: (out / directory).mkdir(exist_ok=True)
                    np.save(out / raw_name, value_float); Image.fromarray(png).save(out / png_name)
                    delivered = torch.from_numpy(png.astype(np.float32) / 255).permute(2, 0, 1)[None]
                    vec = identity(delivered, matrix)[0].cpu().numpy(); vec_name = f'embeddings/{r["id"]}_{arm}.npy'; np.save(out / vec_name, vec)
                    for name in [raw_name, png_name, vec_name]: bind(name)
                    row['arms'][arm] = {**pixels(png, target, observed), 'ArcFace_observed_fixed': float(np.clip(vec @ gt_vec, -1, 1)),
                                        'raw': raw_name, 'prediction': png_name, 'embedding': vec_name}
                row['teacher_prior_png_max_abs'] = int(np.abs(np.asarray(Image.open(out / row['arms']['teacher_vq_reconstruction']['prediction'])).astype(np.int16) -
                                                             np.asarray(Image.open(out / row['arms']['clean_code_oracle_w0']['prediction'])).astype(np.int16)).max())
                rows.append(row)
            print(f'capacity {len(rows)}/10 elapsed={time.monotonic()-started:.2f}s', flush=True)
        clock(); after = {name: state_hash(net) for name, net in models.items()}
        inherited.require(before == after, 'Frozen model/conditioner state changed')
        inherited.require(counts == {k: p['budget'][k] for k in counts}, 'Forward counts differ')
        # Write terminal neural proof BEFORE grid rendering, unlike the failed V10 runner.
        inherited.write(out / 'neural_execution_receipt.json', {'complete': True, 'before_state_hashes': before,
                        'after_state_hashes': after, 'counts': counts, 'zero_conditioner_parity': parity,
                        'seconds_before_rendering': time.monotonic() - started, 'local_backward_calls': 0,
                        'local_optimizer_updates': 0, 'native_used': False, 'validation_used': False})
        grids = []
        for source, name in [('dataset/asian_faces', 'asian_capacity_5_rows.png'),
                             ('dataset/thumbnails128x128', 'ffhq_capacity_5_rows.png')]:
            cells = [{'id': row['id'], 'images': [inherited.rgb(ROOT / row['target'])] +
                      [inherited.rgb(out / row['arms'][a]['prediction']) for a in p['capacity_arms']]}
                     for row in rows if row['source'] == source]
            render_grid(['clean_training_reference'] + p['capacity_arms'], cells).save(out / name); bind(name); grids.append(name)
        clock(); verify()
        result = {'complete': True, 'plan_sha256': inherited.sha(OUT / 'frozen_plan.json'), 'rows': rows,
                  'grids': grids, 'artifacts_sha256': artifacts, 'state_hashes_unchanged': before == after,
                  'counts': counts, 'zero_conditioner_parity': parity, 'seconds': time.monotonic() - started,
                  'native_used': False, 'validation_used': False, 'native_reserved_used': False,
                  'local_backward_calls': 0, 'local_optimizer_updates': 0, 'production_promoted': False,
                  'limitation': 'Clean-code oracle sees clean training targets; representation capacity only, not restored CCTV or generalization.'}
        inherited.write(out / 'results.json', result); print({'complete': True, 'seconds': result['seconds'], 'counts': counts})
    except Exception as error:
        inherited.write(out / 'failure.json', {'complete': False, 'error_type': type(error).__name__, 'error': str(error),
                        'counts': counts, 'seconds': time.monotonic() - started, 'local_optimizer_updates': 0,
                        'resume_permitted': False})
        raise
    finally:
        for handle in handles: handle.remove()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.prepare: prepare()
    elif args.run: run()
    else: parser.error('Choose --prepare or --run')
