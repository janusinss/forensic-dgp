"""Prepare a thin fixed-state L4 measurement of corrected V26; no local gradients."""
import ast
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cctv_dgp_v25_gradient_diagnostic_v1_vm import sha, read, write

NEW = ROOT / 'outputs/cctv_dgp_v26_gradient_diagnostic_v1_vm'
OUT = ROOT / 'outputs/cctv_dgp_v26_gradient_diagnostic_v1_preparation'
PARENT = ROOT / 'outputs/cctv_dgp_batchmatched_identity_vm_v26'
RETURN = ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_return'
NAME = 'cctv-dgp-v26-gradient-diagnostic-v1'
TERMS = ['degraded_landmark_detail', 'degraded_observed_detail', 'degraded_pixel',
         'clear_baseline_anchor', 'pixel_regression', 'SSIM_regression', 'ArcFace_regression']


def corrected_worker(old):
    pairs = [
        ('cctv_dgp_v25_gradient_diagnostic_v1', 'cctv_dgp_v26_gradient_diagnostic_v1'),
        ('cctv-dgp-v25-gradient-diagnostic-v1', 'cctv-dgp-v26-gradient-diagnostic-v1'),
        ('closed_V25', 'closed_V26'), ('own-DGP-V25', 'own-DGP-V26'),
        ('scripts/cctv_dgp_spatial_features_v25_vm.py', 'scripts/cctv_dgp_batchmatched_identity_v26_vm.py'),
        ('cctv_dgp_spatial_features_vm_v25', 'cctv_dgp_batchmatched_identity_vm_v26'),
        ("base['format']=='dgp-spatial-feature-capacity-v25'", "base['format']=='dgp-spatial-batchmatched-identity-capacity-v26'"),
        ('V25', 'V26'), ("'recognizer_forwards':120", "'recognizer_forwards':70"),
    ]
    for before, after in pairs:
        assert before in old, before
        old = old.replace(before, after)
    legacy = "                item['base_cosine']=(identity.embedding(item['base']*item['mask']+item['x']*(1-item['mask']),item['mask'],item['grid'])*item['truth']).sum(1)\n"
    assert legacy in old
    old = old.replace(legacy, '').replace("assert progress['recognizer_forwards']==100", "assert progress['recognizer_forwards']==50")
    old = old.replace("'grid','truth','base_cosine','degraded_weight'", "'grid','truth','degraded_weight'")
    selection = "        functions += [next(n for n in helper.body if isinstance(n,ast.FunctionDef) and n.name==name) for name in ['assemble_terms','objective_terms']]"
    replacement = "        functions += [next(n for n in helper.body if isinstance(n,ast.FunctionDef) and n.name=='assemble_terms')]\n        corrected=ast.parse((parent/'cctv_dgp_batchmatched_identity_v26.py').read_text())\n        functions += [next(n for n in corrected.body if isinstance(n,ast.FunctionDef) and n.name==name) for name in ['batchmatched_scores','objective_terms']]"
    assert selection in old
    old = old.replace(selection, replacement)
    checkpoint = "                    assert len(pieces)==len(parameters) and all(torch.isfinite(g).all() for g in pieces)"
    assert checkpoint in old
    old = old.replace(checkpoint, checkpoint + "\n                    if update==0 and name=='ArcFace_regression':\n                        assert float(scalar.detach())==0 and all(torch.count_nonzero(g)==0 for g in pieces),'Initial corrected identity value/all26 gradients must stay exactly zero'")
    proof_check = "    cache=read(parent/'outputs/frozen_DGP_features.json')"
    assert proof_check in old
    old = old.replace(proof_check, "    proof=read(parent/p['identity_proof_file'])\n    assert proof['complete'] and proof['cases']==50 and proof['gradient_calls']==20\n    assert proof['batchmatched_component_value']==proof['batchmatched_component_gradient_norm']==0\n    assert len(proof['rows'])==10 and all(r['all26_matched_gradient_tensors_exactly_zero'] and r['exact_reference_prediction_cosines'] for r in proof['rows'])\n" + proof_check)
    return old


def main():
    started = time.monotonic()
    assert not NEW.exists() and not OUT.exists(), 'Preserve previous diagnostic preparation'
    auditpath = ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_independent_audit.json'
    audit = read(auditpath)
    lossfolder = ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_loss_audit_v1'
    loss, losscheck = read(lossfolder / 'results.json'), read(lossfolder / 'independent_saved_loss_audit.json')
    visualpath = ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_diagnostic/visual_review.json'
    assert audit['complete'] and audit['VM_failure_present'] and not audit['early_structure_stop']['pass']
    assert audit['identity_preflight_audit']['batchmatched_component_gradient_norm'] == 0
    assert loss['complete'] and losscheck['complete'] and losscheck['results_sha256'] == sha(lossfolder / 'results.json')
    assert read(visualpath)['complete'] and read(visualpath)['cases_reviewed'] == 50
    assert sha(PARENT / 'protocol.json') == audit['protocol_sha256'] == sha(RETURN / 'protocol.json')
    base = read(PARENT / 'protocol.json')
    for name, digest in base['assets_sha256'].items():
        assert sha(PARENT / name) == digest, name
    for name, digest in loss['source_bindings_sha256'].items():
        assert sha(ROOT / name) == digest, name
    basis = ROOT / 'scripts/cctv_dgp_v25_gradient_diagnostic_v1_vm.py'
    worker = corrected_worker(basis.read_text(encoding='utf-8'))
    ast.parse(worker, feature_version=(3, 10))
    NEW.mkdir()
    (NEW / 'scripts').mkdir()
    source = ROOT / 'scripts/cctv_dgp_v26_gradient_diagnostic_v1_vm.py'
    with source.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(worker)
    shutil.copy2(source, NEW / 'scripts' / source.name)
    shell_basis = ROOT / 'outputs/cctv_dgp_v25_gradient_diagnostic_v1_vm/scripts/run_gradient.sh'
    shell = shell_basis.read_text(encoding='utf-8').replace('cctv_dgp_v25_gradient_diagnostic_v1', 'cctv_dgp_v26_gradient_diagnostic_v1')
    with (NEW / 'scripts/run_gradient.sh').open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(shell)
    inputs = ['outputs/failure.json', 'outputs/early_structure_stop.json', 'outputs/cohort_loss_setup.json',
        'outputs/frozen_DGP_features.json', 'outputs/one_batch_gradient_preflight.json',
        'outputs/feature_path_gradient_update2.json', 'outputs/execution_receipt.json',
        'outputs/stopped_head.pth', 'supervisor_receipt.json', 'trainer_exit_code.txt', 'installation_receipt.json']
    proof = sorted(RETURN.glob('batchmatched_identity_preflight_*.json'))
    assert len(proof) == 1
    inputs += [p.relative_to(RETURN).as_posix() for pattern in [
        'batchmatched_identity_preflight_*.json', 'preflight_*.json', 'feature_preflight_*.json'] for p in RETURN.glob(pattern)]
    for update in [0, 50]:
        inputs += ['outputs/update' + str(update) + '/' + name for name in ['head.pth', 'metrics.json']]
        inputs += ['outputs/update' + str(update) + '/' + c['id'] + '.npy' for c in base['cases']]
    bindings = {name: sha(RETURN / name) for name in inputs}
    p = {'format': 'own-DGP-V26-saved-state-gradient-diagnostic-v1', 'date': '2026-10-06',
        'purpose': 'Measure corrected saved-state structure/preservation gradients before choosing another recipe; no training or quality acceptance',
        'assets_sha256': {name: sha(NEW / name) for name in ['scripts/' + source.name, 'scripts/run_gradient.sh']},
        'closed_V26_protocol_sha256': audit['protocol_sha256'],
        'closed_V26_return_sha256': '9ea6afad52b70a27681630b8b09f96dfe560f6756664b627037198fe9e47ae53',
        'closed_V26_inputs_sha256': bindings, 'identity_proof_file': proof[0].name,
        'snapshots': [0, 50], 'head_parameters': 53781, 'head_batches': 20, 'component_gradient_calls': 140,
        'recognizer_forwards': 70, 'DGP_forwards': 0, 'optimizer_updates': 0,
        'head_states': {'0': audit['VM_initial_state'], '50': audit['stopped_head_state']},
        'frozen_recognizer_state': '9ee66c2faefe0b82224c2cea6811073fbf70e8036808e293878109f078ea3963',
        'terms': TERMS, 'cohort_policy': 'Original fifty exposed TRAIN cases, states0/50, ten sequential batches of five each. Each scalar mean/10 gives full fifty-case component gradient.',
        'source_policy': 'Compile unchanged own-DGP head and original VM guard/three metrics; original V24 assemble_terms plus corrected V26 batchmatched_scores/objective_terms. No optimizer or training launcher is imported.',
        'initial_corrected_identity_exact_zero_each_batch_and_all26_gradients': True,
        'hypothesis': 'Small residual penalties can oppose structure through large active gradients; compare actual corrected fixed states and feature/direct decoder parameter groups before any learning change.',
        'frozen_policy': 'No optimizer, backward(), updates, new checkpoints, input/feature changes, gate changes or automatic follow-on. All local work is saved arithmetic/inference only.',
        'timing_limits': {'worker_seconds': 420, 'external_seconds': 480, 'external_kill_grace_seconds': 30,
            'export_seconds': 30, 'external_export_seconds': 60, 'export_kill_grace_seconds': 10,
            'free_disk_bytes': 1024 ** 3, 'allocated_VRAM_bytes': 20 * 1024 ** 3},
        'prospective_return_checks': {'head_raw_parity_maximum': 2e-6, 'gradient_shape': [7, 53781],
            'gradient_dtype': 'float64', 'finite_gradients': True, 'head_and_frozen_states_unchanged': True,
            'exact_saved_raw_and_cache_bindings': True, 'counts_and_timing_checked': True,
            'original_V26_failure_retained': True, 'saved_gradient_arithmetic_only_local': True,
            'local_gradient_calls': 0, 'local_optimizer_updates': 0,
            'CPU_saved_loss_scalar_tolerances': {'nonidentity_terms': 2e-5, 'ArcFace_regression': 5e-4},
            'scalar_tolerance_scope': 'Retain prior fixed CPU/CUDA scalar replay bounds; identity bound uses original cosine5e-5 twice times coefficient5. These are no quality threshold waivers.'},
        'limitations': 'Fixed-state gradients do not reconstruct AdamW moments/trajectory, establish causal sufficiency, qualify native CCTV or prove generalization. TRAIN source labels are not ethnicity. Whole-face usefulness remains required.',
        'local_basis_sha256': {path.relative_to(ROOT).as_posix(): sha(path) for path in [auditpath, visualpath,
            lossfolder / 'results.json', lossfolder / 'independent_saved_loss_audit.json',
            ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_diagnostic/independent_saved_diagnostic_audit.json',
            ROOT / 'outputs/cctv_dgp_post_v24_architecture_review_v1/user_architecture_decision.json',
            basis, shell_basis]},
        'manual_existing_L4_only': True, 'new_training_recipe': False, 'app_promotion': False, 'goal_complete': False}
    write(NEW / 'protocol.json', p)
    pin = sha(NEW / 'protocol.json')
    with (NEW / 'protocol.sha256').open('x', encoding='ascii', newline='\n') as handle:
        handle.write(pin + '  protocol.json\n')
    archive = ROOT / 'outputs' / (NAME + '-execution.tar.gz')
    with tarfile.open(archive, 'x:gz', compresslevel=3) as tar:
        for file in sorted(NEW.rglob('*')):
            if file.is_file():
                info = tar.gettarinfo(str(file), NEW.name + '/' + file.relative_to(NEW).as_posix())
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ''
                info.mode = 0o755 if file.suffix == '.sh' else 0o644
                with file.open('rb') as handle:
                    tar.addfile(info, handle)
    digest = sha(archive)
    with Path(str(archive) + '.sha256').open('x', encoding='ascii', newline='\n') as handle:
        handle.write(digest + '  ' + archive.name + '\n')
    OUT.mkdir()
    receipt = {'complete': True, 'date': '2026-10-06', 'protocol_sha256': pin, 'archive_sha256': digest,
        'archive_bytes': archive.stat().st_size, 'archive_members': 4, 'assets': 2,
        'parent_saved_input_bindings': len(bindings), 'builder_sha256': sha(Path(__file__)),
        'data_or_weights_reuploaded': False, 'new_training_recipe': False, 'optimizer_updates': 0,
        'local_neural_calls': 0, 'local_gradient_calls': 0, 'VM_actions': False,
        'seconds': time.monotonic() - started, 'app_promotion': False, 'goal_complete': False}
    write(OUT / 'preparation.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
