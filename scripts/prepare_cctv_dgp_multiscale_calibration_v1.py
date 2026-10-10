"""Build and bind a new manual finite diagnostic. No neural/training imports."""
import ast
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cctv_dgp_multiscale_calibration_contract_v1 import NAME, STEM, STATE, INITIAL, CHECKPOINT, RECOGNIZER, BUDGETS, WEIGHTS, NORMALIZERS, SIGNALS, arms, read, write, sha, verify


def prepare():
    out = ROOT / 'outputs'; dest = out / NAME; cp = out / 'cctv_dgp_head4_capacity_vm_v1'
    archive = out / (STEM + '-execution.tar.gz')
    assert not dest.exists() and not archive.exists(), 'Do not replace a prepared packet'
    parity = read(out / 'cctv_dgp_multiscale_calibration_v1_parity/plan.json')
    pr = read(out / 'cctv_dgp_multiscale_calibration_v1_parity/results.json')
    pa = read(out / 'cctv_dgp_multiscale_calibration_v1_parity/independent_audit.json')
    assert pr['complete'] and pa['complete'] and pr['states_before']['candidate'] == INITIAL
    assert pa['worker_results_sha256'] == sha(out / 'cctv_dgp_multiscale_calibration_v1_parity/results.json')
    for n, digest in parity['source_sha256'].items():
        assert sha(ROOT / n) == digest
    learning = read(out / 'cctv_dgp_head4_learning_signal_review_v1/independent_audit.json')
    assert learning['complete']
    source = read(cp / 'protocol.json'); cases = parity['cases']; references = parity['references']
    native_root = out / 'cctv_chokepoint_native_development_v1'
    frozen = read(native_root / 'frozen_subset.json'); reviewed = read(native_root / 'input_review.json')
    assert reviewed['subset_sha256'] == sha(native_root / 'frozen_subset.json') and reviewed['reviewed_before_model_outputs']
    reviews = {r['id']: r for r in reviewed['rows']}
    native = [dict(c) for c in frozen['cases'] if c['role'] == 'development']
    assert len(native) == 24 and all(reviews[c['id']]['input_review'] == 'usable' for c in native)
    dest.mkdir(); (dest / 'scripts').mkdir(); copies = {}; local_sources = {}

    def copy(src, name):
        src = Path(src); f = dest / name
        assert src.is_file() and not src.is_symlink() and not f.exists()
        f.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, f)
        digest = sha(src); assert sha(f) == digest
        copies[name] = digest; local_sources[src.relative_to(ROOT).as_posix()] = digest

    helpers = ['cctv_dgp_frozen_norm.py', 'cctv_dgp_group_conflicts_v1_losses.py', 'cctv_dgp_head4_lossless_v1.py',
        'cctv_dgp_pilot.py', 'dgp_face_restoration.py', 'dgp_frozen_inference_v2.py',
        'frozen_capacity_contract.py', 'frozen_definitions.py', 'frozen_raw_metrics.py']
    helpers += [n for n in source['assets_sha256'] if n.startswith('models/') and n.endswith('.py')]
    helpers += ['weights/dgp_v2.pth', 'weights/w600k_r50.onnx']
    for n in helpers:
        assert sha(cp / n) == source['assets_sha256'][n]; copy(cp / n, n)
    for stem in ['cctv_dgp_multiscale_calibration_contract_v1', 'cctv_dgp_multiscale_calibration_model_v1']:
        copy(ROOT / 'scripts' / (stem + '.py'), stem + '.py')
    for stem in ['cctv_dgp_multiscale_calibration_vm_v1', 'supervise_cctv_dgp_multiscale_calibration_v1',
                 'audit_cctv_dgp_multiscale_calibration_return_v1']:
        copy(ROOT / 'scripts' / (stem + '.py'), 'scripts/' + stem + '.py')
    data = {c['input'] for c in cases}
    data |= {r[k] for r in references for k in ['target', 'observed']}
    for n in sorted(data):
        assert sha(cp / n) == source['assets_sha256'][n]; copy(cp / n, n)
    for c in native:
        c['input_review'] = reviews[c['id']]
        for key in ['input', 'observed', 'native_crop']:
            old = c[key]; assert sha(native_root / old) == c[key + '_sha256']
            name = 'native_development/' + old; copy(native_root / old, name); c[key] = name
    for n in ['input_review.json', 'frozen_subset.json', 'selection_plan.json', 'LICENSE_SOURCE.html', 'DERIVATIVE_NOTICE.txt']:
        if (native_root / n).exists():
            copy(native_root / n, 'provenance/native/' + n)
    proof_paths = [cp / 'protocol.json', ROOT / 'CCTV_DGP_CURRENT_TRAINING_IMPROVEMENT_PLAN.md',
        out / 'cctv_dgp_head4_capacity_v1_independent_audit.json',
        out / 'cctv_dgp_head4_learning_signal_review_v1/results.json',
        out / 'cctv_dgp_head4_learning_signal_review_v1/independent_audit.json',
        out / 'cctv_dgp_multiscale_calibration_v1_parity/plan.json',
        out / 'cctv_dgp_multiscale_calibration_v1_parity/results.json',
        out / 'cctv_dgp_multiscale_calibration_v1_parity/independent_audit.json',
        out / 'cctv_dgp_full_training_coverage_v1/results.json',
        out / 'cctv_dgp_full_training_coverage_v1/independent_audit.json']
    for f in proof_paths:
        copy(f, 'provenance/' + f.name.replace('results.json', f.parent.name + '_results.json').replace('independent_audit.json', f.parent.name + '_independent_audit.json'))
    # The 1%/10% requirements and original source/split provenance are unchanged.
    p = {'format': 'own-DGP-multiscale-first-step-calibration-v1', 'UTC': datetime.now(timezone.utc).isoformat(),
        'original_state': STATE, 'original_checkpoint_sha256': CHECKPOINT, 'initial_candidate_state': INITIAL,
        'recognizer_state': RECOGNIZER, 'cases': cases, 'references': references, 'native_development': native,
        'canonical_target_RGB_sha256': {r['id']: source['canonical_target_RGB_sha256'][r['id']] for r in references},
        'pools': [list(range(0, 50, 5)), list(range(50, 100, 5))], 'arms': arms(),
        'partitions': {'deep3': {'tensors': 3, 'elements': 147456}, 'decoder15': {'tensors': 15, 'elements': 609219}},
        'reconstruction_weights': WEIGHTS, 'normalizers': NORMALIZERS, 'signals': SIGNALS, 'regression_barrier_coefficient': 5,
        'optimizer': {'name': 'Adam', 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 1e-5, 'gradient_clip_L2': 1.},
        'gradient_shape': [20, 14, 609219], 'gradient_dtype': 'float32', 'budgets': BUDGETS,
        'scientific_thresholds': source['scientific_thresholds'], 'complete_epochs': 0, 'original_epoch_references': 781,
        'one_step_does_not_qualify_model': True, 'manual_tmux_required': True,
        'paired_TRAIN_and_unpaired_native_reported_separately': True, 'all_seven_completion_families_still_required': True,
        'exposure': 'two historically exposed TRAIN pools; not held-out validation; native24 development only',
        'ancestral_overlap_unconfirmed': True, 'native_captured_country_unspecified': True,
        'source_mapping': source['source_mapping'], 'source_roles_provenance_terms_inherited': True,
        'reference_auxiliary_paths_are_ancestral_metadata_and_not_used': True,
        'completed_epoch_study_1_2_5': False, 'automatic_follow_on': False, 'app_promotion': False,
        'goal_complete': False, 'failed_state_resume_allowed': False, 'reserved_final_pixels': 0,
        'visual_review_plan': {'paired_pages': 40, 'native_pages': 24, 'total_pages': 64,
            'paired_rows': 'one reference/all5 profiles; input,target,baseline and3rates; each partition',
            'native_rows': 'four cases/page; input,baseline and3rates; each partition andpool',
            'all_planned_pages_must_be_viewed': True, 'all_visible_features_and_appearance': True,
            'native_no_clean_target_metrics': True}, 'assets_sha256': copies, 'local_sources': local_sources}
    shell = '#!/usr/bin/env bash\nset -euo pipefail\ncd "$(dirname "$0")/.."\n: "${TMUX:?Launch manually inside tmux}"\ntest "$#" -eq 1\nexec python -B -u scripts/supervise_cctv_dgp_multiscale_calibration_v1.py --root . --protocol-sha "$1"\n'
    f = dest / 'scripts/run_multiscale.sh'; f.write_text(shell, encoding='utf-8', newline='\n'); p['assets_sha256']['scripts/run_multiscale.sh'] = sha(f)
    write(dest / 'protocol.json', p); pin = sha(dest / 'protocol.json'); verify(dest, pin)
    for f in dest.rglob('*.py'):
        ast.parse(f.read_text(encoding='utf-8-sig'), filename=str(f))
    with archive.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz', compresslevel=6) as tar:
            for f in sorted(dest.rglob('*')):
                if f.is_file():
                    tar.add(f, arcname=(Path(NAME) / f.relative_to(dest)).as_posix(), recursive=False)
    digest = sha(archive)
    with Path(str(archive) + '.sha256').open('x', encoding='ascii', newline='\n') as f:
        f.write(digest + '  ' + archive.name + '\n')
    write(out / 'cctv_dgp_multiscale_calibration_v1_preparation.json', {'complete': True, 'protocol_sha256': pin,
        'archive_sha256': digest, 'bytes': archive.stat().st_size, 'packet_assets': len(copies),
        'local_optimizer_updates': 0, 'local_gradient_queries': 0, 'neural_calls': 0,
        'manual_run_pending': True, 'model_qualification': False, 'goal_complete': False})
    print({'prepared': True, 'protocol_sha256': pin, 'archive_sha256': digest, 'bytes': archive.stat().st_size}, flush=True)


if __name__ == '__main__':
    prepare()
