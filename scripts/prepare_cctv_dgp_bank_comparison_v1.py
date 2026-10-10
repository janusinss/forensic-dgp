"""Freeze and hash a new finite manual comparison. No Torch or neural calls."""
import ast
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cctv_dgp_bank_comparison_v1_contract import (NAME, STEM, CHECKPOINT, STATE, RECOGNIZER,
    PRIOR, MEAN, NORMALIZERS, ANCHORS, THRESHOLDS, BUDGETS, schedule, read, write, sha, verify)


def prepare():
    out = ROOT / 'outputs'; dest = out/NAME
    archive = out/(STEM+'-execution.tar.gz')
    assert not dest.exists() and not archive.exists(), 'Prepared packets are immutable'
    proof = out/'cctv_dgp_conditioned_bank_initialization_v1'
    audited = out/'cctv_dgp_conditioned_bank_initialization_v1_independent_audit.json'
    a = read(audited); result = read(proof/'results.json')
    assert a['complete'] and a['results_sha256'] == sha(proof/'results.json')
    assert a['manifest_sha256'] == sha(proof/'manifest.json')
    assert a['saved_raw_PNG_compositions_checked'] == 124 and a['sheet_cells_checked'] == 66
    assert a['local_gradient_APIs_refused'] == 2 and result['optimizer_updates'] == 0
    visual = out/'cctv_dgp_conditioned_bank_initialization_v1_visual_review.json'
    v = read(visual); assert v['complete'] and v['sheets_viewed_at_original_resolution'] == 3 and v['cells_inspected'] == 66
    for f, expected in read(proof/'plan.json')['source_bindings'].items():
        assert sha(ROOT/f) == expected
    for row in v['sheets']:
        assert sha(row['path']) == row['sha256']
    parent = out/'cctv_dgp_head4_capacity_vm_v1'; p0 = read(parent/'protocol.json')
    native_parent = out/'cctv_dgp_multiscale_calibration_vm_v1'; np0 = read(native_parent/'protocol.json')
    source = out/'cctv_dgp_generative_bank_source_v1'
    implementation = out/'cctv_dgp_generative_bank_probe_v1/implementation'
    dest.mkdir(); copies = {}; local_sources = {}

    def copy(src, name, expected=None):
        src = Path(src); f = dest/name
        assert src.is_file() and not src.is_symlink() and not f.exists()
        digest = sha(src)
        if expected is not None:
            assert digest == expected, str(src)
        f.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, f)
        assert sha(f) == digest
        copies[name] = digest; local_sources[src.relative_to(ROOT).as_posix()] = digest

    helpers = ['cctv_dgp_frozen_norm.py', 'cctv_dgp_group_conflicts_v1_losses.py', 'cctv_dgp_head4_lossless_v1.py',
        'cctv_dgp_pilot.py', 'dgp_face_restoration.py', 'dgp_frozen_inference_v2.py', 'frozen_capacity_contract.py',
        'frozen_definitions.py', 'frozen_raw_metrics.py', 'weights/dgp_v2.pth', 'weights/w600k_r50.onnx']
    helpers += [n for n in p0['assets_sha256'] if n.startswith('models/') and n.endswith('.py')]
    inputs = {c['input'] for c in p0['cases']} | {r[k] for r in p0['references'] for k in ['target', 'observed']}
    for n in helpers + sorted(inputs):
        copy(parent/n, n, p0['assets_sha256'][n])
    for c in np0['native_development']:
        for key in ['input', 'observed', 'native_crop']:
            copy(native_parent/c[key], c[key], np0['assets_sha256'][c[key]])
    for n, expected in np0['assets_sha256'].items():
        if n.startswith('provenance/native/'):
            copy(native_parent/n, n, expected)
    for stem in ['cctv_dgp_bank_comparison_v1_contract', 'cctv_dgp_bank_comparison_v1_model', 'cctv_dgp_generative_bank_v1']:
        copy(ROOT/'scripts'/(stem+'.py'), stem+'.py')
    for stem in ['cctv_dgp_bank_comparison_v1_vm', 'supervise_cctv_dgp_bank_comparison_v1', 'audit_cctv_dgp_bank_comparison_return_v1']:
        copy(ROOT/'scripts'/(stem+'.py'), 'scripts/'+stem+'.py')
    for f in sorted(implementation.iterdir()):
        if f.is_file():
            copy(f, 'bank_implementation/'+f.name)
    acquisition = read(source/'acquisition.json')
    for n, binding in acquisition['bindings'].items():
        copy(source/n, 'prior_source/'+n, binding['sha256'])
    copy(source/'acquisition.json', 'prior_source/acquisition.json')
    for n in ['initial_A.pth', 'initial_B.pth', 'mean_style.npy']:
        copy(proof/n, 'initializers/'+n, read(proof/'manifest.json')[n])
    for n in ['call0_z.npy', 'call0_raw512.npy'] + ['seed0_feature'+str(s)+'.npy' for s in [16, 32, 64, 128, 256]]:
        copy(out/'cctv_dgp_generative_bank_probe_v1'/n, 'prior_fixtures/'+n)
    evidence = [parent/'protocol.json', proof/'plan.json', proof/'results.json', audited, visual,
        ROOT/'CCTV_DGP_GENERATIVE_PRIOR_METHOD_REVIEW_V2.md', ROOT/'SYSTEM_WORKFLOW_AND_GOAL.md',
        ROOT/'PRACTICAL_OUTPUT_SCOPE.md', ROOT/'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md',
        ROOT/'CCTV_DGP_MULTISCALE_CALIBRATION_V1_RESULTS.md', ROOT/'CCTV_DGP_POST_MULTISCALE_TRAINING_REVIEW.md',
        out/'cctv_dgp_multiscale_active_anchor_v1/plan.json',
        out/'cctv_dgp_generative_bank_probe_v1_independent_audit.json']
    for i, f in enumerate(evidence):
        copy(f, 'provenance/'+f'{i:02d}_'+f.name)
    cases = p0['cases']; selected = schedule(cases)
    protocol = {'format': 'own256-conditioned-frozen-generative-bank-comparison-v1',
        'UTC': datetime.now(timezone.utc).isoformat(), 'original_checkpoint_sha256': CHECKPOINT,
        'original_state': STATE, 'recognizer_state': RECOGNIZER, 'standalone_prior_sha256': PRIOR,
        'mean_style_sha256': MEAN, 'cases': cases, 'references': p0['references'],
        'canonical_target_RGB_sha256': p0['canonical_target_RGB_sha256'], 'source_mapping': p0['source_mapping'],
        'native_development': np0['native_development'], 'preview_case_ids': [c['id'] for c in np0['cases']],
        'schedule': selected, 'maximum_updates_per_route': 50, 'fitting_reference_exposures_per_route': 250,
        'fitting_image_exposures_per_route': 1250, 'full_epochs_per_route': 0,
        'fraction_of_781_reference_epoch_per_route': 250/781, 'routes': {'A': {'tensors': 15, 'elements': 609219},
            'B': {'tensors': 37, 'elements': 3364643}}, 'normalizers': NORMALIZERS,
        'active_anchor_normalizers': ANCHORS, 'active_clear_blur_anchor_coefficients': [1., 1.],
        'reconstruction_weights': [.2, 1., 1., 1.], 'regression_barrier_coefficient': 5,
        'scientific_thresholds': THRESHOLDS, 'optimizer': {'name': 'Adam', 'lr': .0001, 'betas': [.9, .999],
            'eps': 1e-8, 'weight_decay': 1e-5, 'gradient_clip_L2': 1., 'scheduler': 'constant LambdaLR'},
        'budgets': BUDGETS, 'manual_tmux_required': True, 'independent_routes_continue_after_quality_stop': True,
        'global_preflight_or_resource_failure_stops_comparison': True, 'full_3905_TRAIN_raw_PNG_gates': True,
        'B_disabled_ablation_TRAIN_cases': 100, 'B_disabled_ablation_native_cases': 24,
        'initial_conditioning_gradients_expected_zero': True, 'B_post_update1_conditioning_gradient_check_required': True,
        'original_forward_and_bank_feature_path_resolution': 256, 'standalone_prior_native_checkpoint_resolution': 512,
        'preflight_generated_seed_test_uses_full512_prior_once': True, 'new_conditioning_seed': 20261010,
        'mean_style_seed': 20261010, 'mean_style_Z_samples': 256,
        'external_generator_not_own_trained': True, 'pretrained_restoration_encoder_used': False,
        'pretrained_prior_FFHQ_person_overlap_unexcluded': True, 'source_terms_and_historical_overlap_limits_inherited': True,
        'source_labels_not_ethnicity_or_native_capture_claims': True,
        'reference_auxiliary_native_paths_ancestral_metadata_only': True,
        'native_CCTV_unpaired_and_synthetic_TRAIN_paired_separate': True, 'reserved_final_pixels': 0,
        'app_promotion': False, 'goal_complete': False, 'automatic_follow_on': False, 'failed_run_resume_allowed': False,
        'all_seven_completion_families_required_separately': True, 'longer_epoch_study_not_qualified': True,
        'visual_review_plan': {'paired_preview_sheets': 20, 'paired_ROWS_each': 5,
            'paired_COLUMNS': ['input', 'target', 'baseline', 'A50', 'B50', 'B50_bank_disabled'],
            'native_sheets': 6, 'native_ROWS_each': 4,
            'native_COLUMNS': ['input', 'baseline', 'A50', 'B50', 'B50_bank_disabled'],
            'all_planned_available_sheets_must_be_viewed': True, 'partial_or_failed_missing_outputs_declared': True,
            'all_visible_features_and_appearance': True}, 'assets_sha256': copies, 'local_sources': local_sources}
    shell = '#!/usr/bin/env bash\nset -euo pipefail\ncd "$(dirname "$0")/.."\n: "${TMUX:?Launch manually inside tmux}"\ntest "$#" -eq 1\nexec python -B -u scripts/supervise_cctv_dgp_bank_comparison_v1.py --root . --protocol-sha "$1"\n'
    f = dest/'scripts/run_bank_comparison.sh'; f.write_text(shell, encoding='utf-8', newline='\n')
    copies['scripts/run_bank_comparison.sh'] = sha(f)
    write(dest/'protocol.json', protocol); pin = sha(dest/'protocol.json'); verify(dest, pin)
    for f in dest.rglob('*.py'):
        ast.parse(f.read_text(encoding='utf-8-sig'), filename=str(f))
    with archive.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz', compresslevel=1) as tar:
            for f in sorted(dest.rglob('*')):
                if f.is_file():
                    tar.add(f, arcname=NAME+'/'+f.relative_to(dest).as_posix(), recursive=False)
    digest = sha(archive)
    Path(str(archive)+'.sha256').write_text(digest+'  '+archive.name+'\n', encoding='ascii', newline='\n')
    record = {'complete': True, 'protocol_sha256': pin, 'archive_sha256': digest, 'bytes': archive.stat().st_size,
        'packet_assets': len(copies), 'local_gradient_queries': 0, 'local_optimizer_updates': 0, 'neural_calls': 0,
        'manual_VM_run_pending': True, 'standalone_GPU_and_learning_checks_pending': True,
        'model_qualified': False, 'app_promotion': False, 'goal_complete': False}
    write(out/'cctv_dgp_bank_comparison_v1_preparation.json', record)
    print(record, flush=True)


if __name__ == '__main__':
    prepare()
