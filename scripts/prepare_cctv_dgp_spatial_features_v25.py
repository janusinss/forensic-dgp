"""Prepare a distinct own-DGP feature/spatial capacity pilot after Route A selection."""
import ast
import copy
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from import_cctv_dgp_degraded_detail_v24 import read, require, sha, write

OLD = ROOT / 'outputs/cctv_dgp_degraded_detail_vm_v24'
NEW = ROOT / 'outputs/cctv_dgp_spatial_features_vm_v25'
EVIDENCE = ROOT / 'outputs/cctv_dgp_spatial_features_v25_preparation'


def runtime_source(original):
    text = original.replace('dgp-degraded-detail-cohort-capacity-v24', 'dgp-spatial-feature-capacity-v25')
    text = text.replace('scripts/cctv_dgp_degraded_detail_v24.py', 'scripts/cctv_dgp_spatial_features_v25_vm.py')
    text = text.replace('cctv_dgp_degraded_detail_v24_return', 'cctv_dgp_spatial_features_v25_return')
    text = text.replace('cctv-dgp-degraded-detail-v24', 'cctv-dgp-spatial-features-v25')
    text = text.replace('scripts/run_v24.sh', 'scripts/run_v25.sh').replace('V24 update', 'V25 update')
    tree = ast.parse(text)
    old_prepare = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'prepare')
    lines = text.splitlines(keepends=True)
    replacement = '''def prepare(root, p):
    require_vm(root, idle=True)
    from cctv_dgp_spatial_features_v25_prepare import prepare_features
    return prepare_features(root, p, require_vm)
'''
    text = ''.join(lines[:old_prepare.lineno-1]) + replacement + ''.join(lines[old_prepare.end_lineno:])
    old_batch = """    def batch(ids):
        return {key: torch.cat([items[i][key] for i in ids]) for key in ['x','base','mask','target','feature','interior','valid7','grid','truth','base_cosine','degraded_weight','clear_weight']}
"""
    new_batch = """    def batch(ids):
        values = {key: torch.cat([items[i][key] for i in ids]) for key in ['x','base','mask','target','feature','interior','valid7','grid','truth','base_cosine','degraded_weight','clear_weight']}
        values['fpn'] = tuple(torch.cat([items[i]['fpn'][index] for i in ids]) for index in range(5))
        return values
"""
    require(text.count(old_batch) == 1, 'Original batch differs'); text = text.replace(old_batch, new_batch)
    text = text.replace("head(b['x'],b['base'],b['mask'])", "head(b['x'],b['base'],b['mask'],b['fpn'])")
    text = text.replace("head(item['x'],item['base'],item['mask'])", "head(item['x'],item['base'],item['mask'],item['fpn'])")
    anchor = "    write(out/'cohort_loss_setup.json', loss_setup)"
    extra = '''
    cache = out / 'frozen_DGP_features'; cache.mkdir()
    feature_bindings = {}
    for item in items:
        for index, value in enumerate(item['fpn']):
            name = item['case']['id'] + '_fpn' + str(index) + '.npy'
            np.save(cache / name, value[0].detach().cpu().numpy().copy(), allow_pickle=False)
            feature_bindings[name] = sha(cache / name)
    write(out/'frozen_DGP_features.json', {'complete': True, 'cases': 50, 'feature_arrays': 250,
        'files_sha256': feature_bindings, 'CUDA_feature_rows': preflight['frozen_feature_cache_rows'],
        'DGP_state_unchanged': preflight['DGP_state_unchanged'], 'optimizer_constructed': False,
        'scope': 'Frozen inference-only own-DGP features from camera inputs, no target feature conditioning'})
'''
    require(text.count(anchor) == 1, 'Cohort receipt anchor differs'); text = text.replace(anchor, anchor + extra)
    anchor = "            torch.nn.utils.clip_grad_norm_(head.parameters(),1);optimizer.step();progress['updates']=update"
    proof = '''            if update == 2:
                projection_gradients = {str(index): float(layer.weight.grad.double().square().sum()) for index,layer in enumerate(head.projections)}
                assert all(value > 0 for value in projection_gradients.values()), 'Nonzero gradients through all five own-DGP feature projections required'
                write(out/'feature_path_gradient_update2.json', {'complete': True, 'update_before_optimizer': 2,
                    'per_projection_gradient_sum_squares': projection_gradients,
                    'frozen_feature_tensors_have_no_gradients': all(not value.requires_grad and value.grad is None for item in items for value in item['fpn']),
                    'DGP_state_unchanged': preflight['DGP_state_unchanged']})
'''
    require(text.count(anchor) == 1, 'Gradient proof anchor differs'); text = text.replace(anchor, proof + anchor)
    anchor = "    files+=sorted(root.glob('preflight_*.json'))"
    extra = "\n    files+=sorted(root.glob('feature_preflight_*.json'))\n    files+=[root/name for name in ['cctv_dgp_spatial_features_v25.py','cctv_dgp_spatial_features_v25_prepare.py','cctv_dgp_degraded_objective_v24.py']]"
    require(text.count(anchor) == 1, 'Export anchor differs'); text = text.replace(anchor, anchor + extra)
    ast.parse(text, feature_version=(3, 10))
    return text


def prepare():
    started = time.monotonic()
    require(not NEW.exists() and not EVIDENCE.exists(), 'Preserve previous V25 preparation')
    require(sha(OLD / 'protocol.json') == '76ab24695a40b411832e2d678c79b0e2a80f6320d4525970b2dd32db2379e114', 'V24 source changed')
    old = read(OLD / 'protocol.json')
    for name, digest in old['assets_sha256'].items():
        require(sha(OLD / name) == digest, 'V24 asset changed: ' + name)
    decision = read(ROOT / 'outputs/cctv_dgp_post_v24_architecture_review_v1/user_architecture_decision.json')
    require(decision['selected_route'] == 'A', 'Human architecture discussion/selection required')
    audit = read(ROOT / 'outputs/cctv_dgp_degraded_detail_v24_independent_audit.json')
    visual = read(ROOT / 'outputs/cctv_dgp_degraded_detail_v24_diagnostic/visual_review.json')
    interface_path = ROOT / 'outputs/cctv_dgp_spatial_features_v25_interface/results.json'
    interface = read(interface_path)
    interface_check = read(interface_path.with_name('independent_saved_interface_audit.json'))
    require(audit['complete'] and not audit['early_structure_stop']['pass'] and visual['complete'], 'V24 closure required')
    require(interface['complete'] and interface_check['complete'] and interface_check['results_sha256'] == sha(interface_path)
            and interface['trainable_design_parameters'] == 53781, 'Audited new forward interface required')
    require(interface['source_bindings_sha256']['scripts/cctv_dgp_spatial_features_v25.py'] == sha(ROOT / 'scripts/cctv_dgp_spatial_features_v25.py'), 'Tested head source changed')
    NEW.mkdir(); assets = {}
    for name, digest in old['assets_sha256'].items():
        revised = 'lineage/v24_original_' + Path(name).name if name in ['scripts/cctv_dgp_degraded_detail_v24.py', 'scripts/run_v24.sh'] else name
        dest = NEW / revised; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(OLD / name, dest)
        require(sha(dest) == digest, 'Copied original asset differs'); assets[revised] = digest
    new_source = runtime_source((OLD / 'scripts/cctv_dgp_degraded_detail_v24.py').read_text(encoding='utf-8'))
    source_path = ROOT / 'scripts/cctv_dgp_spatial_features_v25_vm.py'
    with source_path.open('x', encoding='utf-8', newline='\n') as stream: stream.write(new_source)
    names = [('scripts/cctv_dgp_spatial_features_v25_vm.py', source_path),
             ('cctv_dgp_spatial_features_v25.py', ROOT / 'scripts/cctv_dgp_spatial_features_v25.py'),
             ('cctv_dgp_spatial_features_v25_prepare.py', ROOT / 'scripts/cctv_dgp_spatial_features_v25_prepare.py')]
    for name, file in names:
        (NEW / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, NEW / name); assets[name] = sha(NEW / name)
    shell = (OLD / 'scripts/run_v24.sh').read_text(encoding='utf-8').replace('scripts/cctv_dgp_degraded_detail_v24.py', 'scripts/cctv_dgp_spatial_features_v25_vm.py')
    with (NEW / 'scripts/run_v25.sh').open('x', encoding='utf-8', newline='\n') as stream: stream.write(shell)
    assets['scripts/run_v25.sh'] = sha(NEW / 'scripts/run_v25.sh')
    lineage = [ROOT / 'outputs/cctv_dgp_degraded_detail_v24_independent_audit.json',
               ROOT / 'outputs/cctv_dgp_degraded_detail_v24_return/outputs/failure.json',
               ROOT / 'outputs/cctv_dgp_degraded_detail_v24_return/outputs/early_structure_stop.json',
               ROOT / 'outputs/cctv_dgp_degraded_detail_v24_diagnostic/results.json',
               ROOT / 'outputs/cctv_dgp_degraded_detail_v24_diagnostic/visual_review.json',
               ROOT / 'outputs/cctv_dgp_degraded_detail_v24_loss_audit_v1/independent_saved_loss_audit.json',
               ROOT / 'outputs/cctv_dgp_post_v24_architecture_review_v1/review.json',
               ROOT / 'outputs/cctv_dgp_post_v24_architecture_review_v1/user_architecture_decision.json',
               interface_path, interface_path.with_name('independent_saved_interface_audit.json')]
    for index, file in enumerate(lineage):
        name = 'lineage/v25_basis_' + str(index).zfill(2) + '_' + file.name
        shutil.copy2(file, NEW / name); assets[name] = sha(NEW / name)
    p = copy.deepcopy(old); p['format'] = 'dgp-spatial-feature-capacity-v25'; p['assets_sha256'] = assets
    p['purpose'] = 'Necessary exposed-training capacity of our feature-conditioned own-DGP spatial path; no app/native/final acceptance'
    p['hypothesis'] = 'Use the frozen own-DGP multiscale representation and camera detail to learn spatial structure without the failed final Gaussian high-pass. Retain observed RGB mean removal, V24 degraded/clear objective and original preservation gates.'
    p['difference_from_closed_recipes'] = 'A new53,781-parameter own-DGP feature decoder and mean-preserving spatial correction, not the V22-V24 filtered RGB head, V18 pretrained-feature decoder, a gate waiver or a longer unchanged retry. User selected Route A after three failures.'
    p['design']['trainable_parameters'] = 53781
    p['design']['architecture'] = 'Five frozen own-DGP FPN maps64/128/128/128/128 at128/64/32/16/8 projected to16ch; four24ch coarse-to-fine3x3 decoder convolutions; full256 camera/DGP/HF/support13->16 branch;40->24 fusion;24->3 3x3 tail plus13->3 3x3 direct bypass. SiLU, bilinear align_cornersFalse; both RGB tails zero. No target/source/profile/identity input.'
    p['design']['correction'] = 'q=.05*tanh(direct(observed)+tail(fused_own_features)); subtract observed per-channel q mean; mask; clip cachedDGP+correction. No final output Gaussian high-pass; original support/padding. Appearance/brightness guards unchanged.'
    p['design']['frozen_components'] = ['Retained ownDGP checkpoint and five corrected normalization layers, inference-only feature provider', 'ArcFace recognizer and shared target alignment']
    p['feature_evidence'] = 'Export all250 detached ordinary float32 input-conditioned FPN arrays and fingerprints before optimizer construction. All50 fresh CUDA raw comparisons<=2e-6; four fixed VM CPU comparisons<=1e-5 raw/5e-5 features before any gradients. Fixed numerical compatibility only; quality tolerances unchanged. Verify allfive projection gradients at actual update2.'
    p['normalization_scope'] = old['normalization_scope'] + ' All50 feature-extraction DGP inputs use the declared inherited CUDA-scalar convention; camera conditioning stays canonical float32. VM CPU byte-scalar compatibility is separately bounded and not canonical app parity.'
    p['all_facial_features_required_together'] = ['eyes', 'nose', 'mouth', 'face outline', 'overall visible appearance']
    p['visual_review_requirement'] = 'All50 exact input/base/candidate/target cells; whole-face feature and appearance review mandatory before any broader experiment. Five-patch capacity gate alone cannot accept outline or useful restoration.'
    p['closed_v24_protocol_sha256'] = sha(OLD / 'protocol.json')
    p['closed_v24_return_sha256'] = '1d758362366f3e1f17ddf8239b0f41940b6b7c6ebf349a86ed76e0abdf63dd15'
    p['loss_alignment_evidence'] = 'Historical V23 loss mismatch is closed. Independently audited V24 saved-loss decomposition confirms degraded-reward/clear-preservation alignment, but V24 misses the original structure stop. V25 preserves that corrected objective exactly; no claim of proven GPU gradient causality.'
    p['no_further_blind_attempt'] = 'No automatic follow-on; retain any failed feature/gradient/time/structure/preservation gate. Diagnose this distinct path before another recipe; capacity/packet completion is not Goal completion.'
    write(NEW / 'protocol.json', p); pin = sha(NEW / 'protocol.json')
    with (NEW / 'protocol.sha256').open('x', encoding='ascii', newline='\n') as stream: stream.write(pin + '  protocol.json\n')
    archive = ROOT / 'outputs/cctv-dgp-spatial-features-v25-execution.tar.gz'
    with tarfile.open(archive, 'x:gz', compresslevel=3) as stream:
        for file in sorted(NEW.rglob('*')):
            if file.is_file():
                info = stream.gettarinfo(str(file), NEW.name + '/' + file.relative_to(NEW).as_posix())
                info.mtime = 0; info.uid = info.gid = 0; info.uname = info.gname = ''
                info.mode = 0o755 if file.name == 'run_v25.sh' else 0o644
                with file.open('rb') as payload: stream.addfile(info, payload)
    digest = sha(archive)
    with Path(str(archive) + '.sha256').open('x', encoding='ascii', newline='\n') as stream: stream.write(digest + '  ' + archive.name + '\n')
    EVIDENCE.mkdir()
    receipt = {'complete': True, 'date': '2026-10-06', 'scope': 'Distinct Route-A finite transfer; actual VM feature/gradient/timing/learning/quality pending',
        'protocol_sha256': pin, 'archive_sha256': digest, 'archive_bytes': archive.stat().st_size, 'assets': len(assets),
        'parameters': 53781, 'same50_cases_same800_updates_same80epochs': True, 'quality_gates_unchanged': True,
        'objective_and_schedule_unchanged_from_V24': True, 'user_selected_route': 'A',
        'seconds': time.monotonic() - started, 'neural_calls': 0, 'backward_calls': 0, 'optimizer_updates': 0,
        'VM_actions': False, 'app_promotion': False, 'goal_complete': False}
    write(EVIDENCE / 'preparation.json', receipt); print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    prepare()
