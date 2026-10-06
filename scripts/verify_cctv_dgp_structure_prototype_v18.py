"""Finite local fresh-image interface proof; zero training or backward calls."""
import importlib.util
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/cctv_dgp_face_code_fit_vm_v12_r2'
R2 = ROOT / 'outputs/cctv_dgp_broader_codes_vm_v16_r2'
RETURN = ROOT / 'outputs/cctv_dgp_broader_codes_failure_return_v16_r2/outputs/broader_codes_v16_r2'
MIXED = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'
OUT = ROOT / 'outputs/cctv_dgp_structure_prototype_v18'
PIN = '4f64cf2204458f8fadcab1824fc4bc293c65803e123fdebb304f7b5bd3a502a0'
COUNTS = {'dgp': 0, 'prior_encoder': 0, 'prior_classifier': 0, 'r2_head': 0,
          'prior_generator': 0, 'residual_head': 0, 'unused_v11': 0}


def run():
    sys.path.insert(0, str(PARENT))
    sys.path.insert(0, str(R2))
    import cctv_dgp_broader_codes_v16 as v
    p = v.verify(R2, PARENT, MIXED,
        ROOT / 'outputs/cctv_dgp_generalization_return_v15/outputs/generalization_v15', PIN)
    full = v.read(ROOT / 'outputs/cctv_dgp_broader_codes_v16_r2_audit_recovery_1/local_full_audit.json')
    v.require(full['complete'] and full['results_sha256'] == v.sha(RETURN / 'results.json'), 'Require full R2 audit')
    checkpoint = RETURN / 'epoch8/conditioner.pth'
    v.require(v.sha(checkpoint) == '3ef704e70f68d633ac7624eb47f79cfada189341dfdc58bad08444595d7d6757',
              'Audited R2 classifier checkpoint differs')
    v.require(not OUT.exists(), 'Preserve existing prototype; no repeat/overwrite')
    OUT.mkdir()
    for path in [Path(__file__), ROOT / 'dgp_structure_conditioner_v18.py']:
        (OUT / path.name).write_bytes(path.read_bytes())
    spec = importlib.util.spec_from_file_location('structure_v18', ROOT / 'dgp_structure_conditioner_v18.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import numpy as np
    from PIL import Image
    import torch
    from dgp_broader_code_conditioner_v16 import DGPBroaderCodePrior
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from pretrained_face_restoration_portable_v12 import load_face_restorer
    from cctv_dgp_pilot import state_hash
    torch.set_num_threads(4)
    torch.manual_seed(20261005)
    refs = {r['id']: r for r in p['references']}
    ids = [p['train_preview_reference_ids'][0], p['train_preview_reference_ids'][5]]
    cases = [next(c for c in p['training_cases'] if c['reference_id'] == rid and c['profile'] == profile)
             for rid, profile in zip(ids, ['clear', 'blur_lr24'])]
    v.require(all(refs[rid]['role'] == 'train' for rid in ids), 'Training-only prototype differs')
    v.write(OUT / 'protocol.json', {'format': 'dgp-spatial-residual-interface-v18', 'date': '2026-10-05',
        'parent_protocol_sha256': PIN, 'r2_results_sha256': full['results_sha256'],
        'r2_checkpoint_sha256': v.sha(checkpoint), 'case_ids': [c['id'] for c in cases],
        'role': 'train', 'device': 'cpu', 'cap_seconds': 300, 'external_cap_seconds': 360,
        'module_sha256': v.sha(OUT / 'dgp_structure_conditioner_v18.py'),
        'runner_sha256': v.sha(OUT / Path(__file__).name), 'feature_interface': 'prior.generator.blocks[12]:N256x64x64',
        'statistics': 'none', 'fidelity': 0, 'prior_RGB_replacement': False, 'training': False,
        'native_used': False, 'validation_used': False, 'production_promoted': False})
    start = time.monotonic()
    pp = v.read(PARENT / 'face_code_fit_protocol_v12.json')
    dgp, dp = load_frozen_dgp_restorer(PARENT / pp['weights']['dgp'],
        expected_sha256=pp['assets_sha256'][pp['weights']['dgp']], device='cpu')
    prior, cp = load_face_restorer(PARENT / pp['weights']['prior'], 'cpu')
    model = DGPBroaderCodePrior(dgp.net, prior.net).eval()
    model.conditioner.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    head = module.DGPStructureResidualHead().eval()
    frozen = {'dgp': model.core.dgp, 'prior': model.core.prior,
              'r2_classifier': model.conditioner, 'unused_v11': model.core.conditioner, 'residual_head': head}
    before = {name: state_hash(net) for name, net in frozen.items()}
    v.require(all(not net.training and not any(x.requires_grad for x in net.parameters())
                  for net in frozen.values()), 'Inference-only/frozen parameters differ')
    handles = []
    for name, net in [('dgp', model.core.dgp), ('prior_encoder', model.core.prior.encoder.blocks[0]),
        ('prior_classifier', model.core.prior.idx_pred_layer), ('r2_head', model.conditioner),
        ('prior_generator', model.core.prior.generator.blocks[0]), ('residual_head', head),
        ('unused_v11', model.core.conditioner)]:
        def count(_a, _b, _c, key=name): COUNTS[key] += 1
        handles.append(net.register_forward_hook(count))
    rows, artifacts = [], {}
    with torch.inference_mode():
        for case in cases:
            v.require(time.monotonic() - start <= 300, 'Prototype300s neural limit exceeded')
            camera = v.rgb(MIXED / case['input'])
            image = torch.from_numpy(camera.copy()).permute(2, 0, 1).float()[None] / 255
            base, features, base_logits = model.frozen_inputs(image)
            logits = model.conditioner(image, base, features, base_logits)
            feature64 = module.prior_features64(model.core.prior, logits)
            actual = head(image, base, feature64)
            ablated = head(image, base, feature64, prior_ablation=True)
            v.require(torch.equal(actual, base) and torch.equal(ablated, base), 'Initial DGP pixel parity failed')
            arrays = {'dgp_base': base[0].permute(1, 2, 0).numpy().copy(),
                      'output': actual[0].permute(1, 2, 0).numpy().copy(),
                      'prior64': feature64[0].numpy().copy()}
            saved = {}
            for name, value in arrays.items():
                filename = case['id'] + '_' + name + '.npy'
                with (OUT / filename).open('xb') as stream: np.save(stream, value, allow_pickle=False)
                artifacts[filename] = v.sha(OUT / filename)
                saved[name] = filename
            ref = refs[case['reference_id']]
            with Image.open(MIXED / ref['observed']) as mask:
                support = np.asarray(mask) > 0
            filename = case['id'] + '_output.png'
            Image.fromarray(v.png(arrays['output'], camera, support)).save(OUT / filename)
            artifacts[filename] = v.sha(OUT / filename)
            rows.append({'id': case['id'], 'role': 'train', 'source': case['source'],
                         'exact_dgp_raw_parity': True, 'exact_zero_prior_raw_parity': True,
                         'files': saved, 'PNG': filename})
            print({'case': case['id'], 'exact_DGP_parity': True, 'seconds': time.monotonic() - start}, flush=True)
    after = {name: state_hash(net) for name, net in frozen.items()}
    v.require(before == after and COUNTS == {'dgp': 2, 'prior_encoder': 2, 'prior_classifier': 2,
        'r2_head': 2, 'prior_generator': 2, 'residual_head': 4, 'unused_v11': 0}, 'Frozen state/counts differ')
    for handle in handles: handle.remove()
    v.require(time.monotonic() - start <= 300, 'Prototype300s neural limit exceeded')
    v.write(OUT / 'results.json', {'complete': True, 'protocol_sha256': v.sha(OUT / 'protocol.json'),
        'module_sha256': v.sha(OUT / 'dgp_structure_conditioner_v18.py'), 'rows': rows,
        'trainable_capacity_parameters': sum(x.numel() for x in head.parameters()), 'counts': COUNTS,
        'frozen_before': before, 'frozen_after': after, 'provenance': {'dgp': dp, 'prior': cp},
        'artifacts_sha256': artifacts, 'torch': torch.__version__, 'seconds': time.monotonic() - start,
        'backward_calls': 0, 'optimizer_updates': 0, 'validation_used': False,
        'native_used': False, 'native_reserved_used': False, 'production_promoted': False,
        'scope': 'Fresh-image interface and initial parity only. Zero-residual output is unchanged DGP, not trained benefit or useful restoration.'})
    print({'complete': True, 'trainable_capacity_parameters': sum(x.numel() for x in head.parameters()),
           'counts': COUNTS, 'seconds': time.monotonic() - start}, flush=True)


if __name__ == '__main__':
    try:
        run()
    except BaseException as error:
        if OUT.is_dir() and not (OUT / 'failure.json').exists():
            import json
            with (OUT / 'failure.json').open('x', encoding='utf-8') as stream:
                json.dump({'complete': False, 'error_type': type(error).__name__, 'error': str(error),
                           'counts': COUNTS, 'backward_calls': 0, 'optimizer_updates': 0,
                           'resume_permitted': False}, stream, indent=2)
        raise
