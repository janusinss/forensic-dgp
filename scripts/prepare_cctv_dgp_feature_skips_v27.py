"""One prospective spatial-path change; preparation performs no neural work."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'outputs/cctv_dgp_batchmatched_identity_vm_v26'
NEW = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'
OUT = ROOT / 'outputs/cctv_dgp_feature_skips_v27_preparation'
NAME = 'cctv-dgp-feature-skips-v27'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def once(text, old, new):
    assert text.count(old) == 1, 'Frozen source anchor differs: ' + old[:80]
    return text.replace(old, new, 1)


def head_source(text):
    anchor = "        self.register_buffer('reflect_indices', torch.cat((torch.arange(5, -1, -1), torch.arange(256), torch.arange(255, 249, -1))))"
    text = once(text, anchor, anchor + '''
        # Append after all original initialization: inherited tensor bytes stay exact.
        self.feature_skips = nn.ModuleList([nn.Conv2d(channels, 3, 1) for channels in [64, 128, 128, 128, 128]])
        for layer in self.feature_skips:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
''')
    old = "        q = .05 * torch.tanh(self.direct(observed) + self.tail(full))"
    new = '''        # Five independent observation-conditioned shortcuts. No target/statistics fit.
        residuals = []
        for layer, feature in zip(self.feature_skips, fpn):
            support = F.interpolate(mask, size=feature.shape[-2:], mode='area')
            mass = support.sum((2, 3), keepdim=True)
            centered = feature - (feature * support).sum((2, 3), keepdim=True) / mass
            scale = ((centered.square() * support).sum((2, 3), keepdim=True) / mass).sqrt().clamp_min(.05)
            residuals.append(F.interpolate(layer(centered / scale), size=(256, 256), mode='bilinear', align_corners=False))
        feature_skip = sum(residuals) / (5 ** .5)
        q = .05 * torch.tanh(self.direct(observed) + self.tail(full) + feature_skip)'''
    return once(text, old, new)


def preparation_source(text):
    text = once(text, 'from cctv_dgp_spatial_features_v25 import SpatialFeatureHead, frozen_fpn_features, tensor_receipt',
        'from cctv_dgp_feature_skips_v27 import SpatialFeatureHead, frozen_fpn_features, tensor_receipt')
    text = text.replace('53781', '55524')
    anchor = '    original_head = state_hash(head)'
    text = once(text, anchor, anchor + '''
    prior = torch.load(root.parent / 'cctv_dgp_batchmatched_identity_vm_v26/outputs/update0/head.pth', map_location='cpu', weights_only=True)
    shared = head.state_dict()
    extra = {name for name in shared if name.startswith('feature_skips.')}
    assert len(extra) == 10 and set(shared) == set(prior) | extra
    assert all(torch.equal(shared[name].cpu(), value) for name, value in prior.items()), 'Original V26 initialization changed'
    assert all(torch.count_nonzero(shared[name]) == 0 for name in extra), 'New shortcuts must initialize exactly zero'
''')
    text = once(text, "        'feature_preflight_file': evidence_name, 'all_feature_tensors_ordinary_detached': True}",
        "        'feature_preflight_file': evidence_name, 'all_feature_tensors_ordinary_detached': True,\n        'shared_V26_initial_tensors_exact': len(prior), 'new_feature_skip_tensors_exact_zero': 10}")
    return text


def runtime_source(text):
    for old, new in [
        ('dgp-spatial-batchmatched-identity-capacity-v26', 'dgp-direct-feature-skips-capacity-v27'),
        ('scripts/cctv_dgp_batchmatched_identity_v26_vm.py', 'scripts/cctv_dgp_feature_skips_v27_vm.py'),
        ('scripts/run_v26.sh', 'scripts/run_v27.sh'),
        ('cctv_dgp_batchmatched_identity_v26_return', 'cctv_dgp_feature_skips_v27_return'),
        ('cctv-dgp-batchmatched-identity-v26', NAME),
        ('cctv_dgp_spatial_features_v25_prepare', 'cctv_dgp_feature_skips_v27_prepare'),
        ('cctv_dgp_batchmatched_identity_v26_preflight', 'cctv_dgp_feature_skips_v27_preflight'),
        ('cctv_dgp_spatial_features_v25.py', 'cctv_dgp_feature_skips_v27.py'),
        ('scripts/install_v26.py', 'scripts/install_v27.py'), ('V26 update', 'V27 update')]:
        text = text.replace(old, new)
    anchor = "        assert direct_gradient > 0, 'Nonzero full-resolution bypass gradient required'"
    text = once(text, anchor, anchor + '''
        skip_gradients = {str(index): float(layer.weight.grad.double().square().sum()) for index,layer in enumerate(head.feature_skips)}
        assert all(value > 0 for value in skip_gradients.values()), 'All five direct DGP-feature readouts must have nonzero gradients before any optimizer'
''')
    text = once(text, "            'optimizer_constructed': False, 'direct_gradient_sum_squares': direct_gradient,",
        "            'optimizer_constructed': False, 'direct_gradient_sum_squares': direct_gradient,\n            'per_feature_skip_weight_gradient_sum_squares': skip_gradients,")
    return text


def main():
    started = time.monotonic()
    assert not NEW.exists() and not OUT.exists(), 'Preserve earlier preparation'
    gradient_path = ROOT / 'outputs/cctv_dgp_v26_gradient_diagnostic_v1_independent_audit.json'
    gradient = read(gradient_path)
    assert gradient['complete'] and gradient['VM_optimizer_updates_verified'] == 0
    assert gradient['corrected_initial_identity_exact_zero_verified']
    path_analysis = ROOT / 'outputs/cctv_dgp_v26_gradient_paths_v1/analysis.json'
    checked = read(path_analysis.with_name('independent_readback.json'))
    assert checked['complete'] and checked['analysis_sha256'] == sha(path_analysis)
    original = read(OLD / 'protocol.json')
    assert sha(OLD / 'protocol.json') == 'f9ccf86ee8e87ca87d43e849df7c0deae8e511e0db1a175b86ab279ec6964994'
    assert len(original['assets_sha256']) == 240
    for name, digest in original['assets_sha256'].items():
        assert sha(OLD / name) == digest, name
    identity = (OLD / 'cctv_dgp_batchmatched_identity_v26_preflight.py').read_text(encoding='utf-8')
    identity = identity.replace('len(parameters)==26', 'len(parameters)==36').replace('torch.zeros(53781', 'torch.zeros(55524').replace('all26_', 'all36_')
    jobs = {
        'cctv_dgp_feature_skips_v27.py': head_source((OLD / 'cctv_dgp_spatial_features_v25.py').read_text(encoding='utf-8')),
        'cctv_dgp_feature_skips_v27_prepare.py': preparation_source((OLD / 'cctv_dgp_spatial_features_v25_prepare.py').read_text(encoding='utf-8')),
        'cctv_dgp_feature_skips_v27_preflight.py': identity,
        'scripts/cctv_dgp_feature_skips_v27_vm.py': runtime_source((OLD / 'scripts/cctv_dgp_batchmatched_identity_v26_vm.py').read_text(encoding='utf-8')),
        'scripts/run_v27.sh': (OLD / 'scripts/run_v26.sh').read_text(encoding='utf-8').replace('scripts/cctv_dgp_batchmatched_identity_v26_vm.py', 'scripts/cctv_dgp_feature_skips_v27_vm.py'),
        'scripts/install_v27.py': (ROOT / 'scripts/install_cctv_dgp_feature_skips_v27_vm.py').read_text(encoding='utf-8'),
    }
    for name, text in jobs.items():
        if name.endswith('.py'):
            ast.parse(text, feature_version=(3, 10))
    NEW.mkdir(); inherited = {}
    for name, digest in original['assets_sha256'].items():
        destination = NEW / name; destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(OLD / name, destination)
        assert sha(destination) == digest
        inherited[name] = {'source': name, 'sha256': digest}
    transfer = {}
    for name, text in jobs.items():
        destination = NEW / name; destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('x', encoding='utf-8', newline='\n') as handle:
            handle.write(text)
        transfer[name] = sha(destination)
    cache_root = ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_return'
    p = copy.deepcopy(original)
    p['format'] = 'dgp-direct-feature-skips-capacity-v27'
    p['purpose'] = 'Test one direct own-DGP feature shortcut hypothesis after independently audited V26 corrected gradients; no application or native/final acceptance'
    p['hypothesis'] = 'The six-stage randomly initialized decoder attenuates feature influence relative to direct RGB. Independent normalized zero-initialized readouts from all five frozen DGP scales may supply visible structure while preserving the original bounded residual and controls.'
    p['difference_from_closed_recipes'] = 'Add five 1x1-to-RGB observation-only normalized feature shortcuts, averaged by sqrt(5), before the unchanged .05 tanh and support-mean removal. Keep every original learned decoder/RGB tensor at its exact V26 initialization. No loss/schedule/optimizer/gate/encoder change.'
    p['loss_alignment_evidence'] = 'Corrected identity value/all36 gradients must remain exactly zero before an optimizer. The verified V26 preservation vector opposes but does not cancel its improvement vector. No preservation term or margin changes.'
    p['assets_sha256'] = {**original['assets_sha256'], **transfer}
    p['inherited_assets'] = inherited
    p['transfer_assets_sha256'] = transfer
    p['closed_V26_protocol_sha256'] = sha(OLD / 'protocol.json')
    p['closed_V26_early_failure_sha256'] = sha(cache_root / 'outputs/early_structure_stop.json')
    p['closed_V26_initial_head_sha256'] = sha(cache_root / 'outputs/update0/head.pth')
    p['closed_V26_initial_head_state'] = '6f397b5f90b5a8595a8c7928b50f33703b10f3f0302b2a1a858291cdf4ec29e1'
    p['design']['trainable_parameters'] = 55524
    p['design']['direct_feature_skip_parameters'] = 1743
    p['design']['direct_feature_skip_channel_RMS_floor'] = .05
    p['design']['direct_feature_skip_scale'] = '1/sqrt(5)'
    p['identity_preflight_policy']['all26_matched_parameter_gradients_exactly_zero'] = False
    del p['identity_preflight_policy']['all26_matched_parameter_gradients_exactly_zero']
    p['identity_preflight_policy']['all36_matched_parameter_gradients_exactly_zero'] = True
    p['thin_install_policy'] = {'existing_parent': '~/forensic-dgp/cctv_dgp_batchmatched_identity_vm_v26',
        'original_assets_verified_and_hardlinked': 240, 'seconds_cap': 60,
        'minimum_free_after_links_bytes': 3 * 1024 ** 3, 'original_V26_failure_must_remain': True,
        'original_files_changed': False, 'model_or_gradient_calls': 0, 'copy_fallback_permitted': False}
    p['prospective_return_audit']['initial_identity_preflight_required'] = True
    p['prospective_return_audit']['all36_parameter_groups_and_new_skip_gradients_required'] = True
    p['prospective_return_audit']['cache_and_original_encoder_guards_unchanged'] = True
    p['whole_face_requirement'] = ['eyes', 'nose', 'mouth', 'face outline', 'overall visible appearance']
    p['local_basis_sha256'] = {path.relative_to(ROOT).as_posix(): sha(path) for path in [gradient_path, path_analysis,
        path_analysis.with_name('independent_readback.json'), ROOT / 'CCTV_DGP_V26_GRADIENT_DIAGNOSTIC_V1_RESULTS.md',
        ROOT / 'outputs/cctv_dgp_post_v24_architecture_review_v1/user_architecture_decision.json']}
    p['actual_VM_processing_proof_pending'] = True
    p['actual_native_useful_outputs_pending'] = True
    write(NEW / 'protocol.json', p)
    archive = ROOT / 'outputs' / (NAME + '-execution.tar.gz')
    assert not archive.exists() and not archive.with_name(archive.name + '.sha256').exists()
    with tarfile.open(archive, 'x:gz', compresslevel=6) as tar:
        for name in ['protocol.json', *sorted(transfer)]:
            tar.add(NEW / name, arcname=NEW.name + '/' + name, recursive=False)
    digest = sha(archive)
    with archive.with_name(archive.name + '.sha256').open('x', encoding='ascii', newline='\n') as handle:
        handle.write(digest + '  ' + archive.name + '\n')
    OUT.mkdir()
    record = {'complete': True, 'date': '2026-10-06', 'protocol_sha256': sha(NEW / 'protocol.json'),
        'archive_sha256': digest, 'archive_bytes': archive.stat().st_size, 'regular_archive_members': 7,
        'original_assets': 240, 'new_assets': 6, 'assets': 246, 'data_or_weights_reuploaded': False,
        'model_difference': 'Five normalized direct feature readouts only; 1743 added parameters',
        'original_shared_initialization_required_exact': True, 'same_corrected_objective_optimizer_schedule_and_scientific_gates': True,
        'finite_updates': 800, 'finite_epochs': 80, 'unchanged_early_1_percent_and_final_10_percent_structure_gates': True,
        'all_five_skip_weight_gradients_required_before_optimizer': True,
        'preflight_seconds': 300, 'fit_seconds': 1500, 'worker_seconds': 1800,
        'supervisor_seconds': 2100, 'external_grace_seconds': 30,
        'original_failed_recipes_and_inputs_unchanged': True, 'new_tests_pending': True,
        'local_model_calls': 0, 'local_gradient_calls': 0, 'local_optimizer_updates': 0,
        'VM_actions': False, 'app_promotion': False, 'goal_complete': False, 'seconds': time.monotonic() - started}
    write(OUT / 'preparation.json', record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
