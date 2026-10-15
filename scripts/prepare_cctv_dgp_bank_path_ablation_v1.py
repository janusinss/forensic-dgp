"""Seal a thin frozen-inference diagnostic against independently audited return."""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs'
PARENT = OUT/'cctv_dgp_bank_photometric_guard_v1_return'
ANALYSIS = OUT/'cctv_dgp_bank_photometric_guard_v1_analysis'
NAME = 'cctv_dgp_bank_path_ablation_v1_vm'
STEM = 'cctv-dgp-bank-path-ablation-v1'
PIN = 'f64154bd602070edaea61ccbf960ec3028e545c6540a4421ec41f9af21128a12'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    dest = OUT/NAME
    archive = OUT/(STEM+'-execution.tar.gz')
    receipt = OUT/'cctv_dgp_bank_path_ablation_v1_preparation.json'
    assert not any(path.exists() for path in [dest, archive, receipt])
    sources = [ANALYSIS/name for name in ['import.json', 'execution_audit.json', 'independent_audit.json',
        'matched_guard_comparison.json', 'visual_review.json', 'saved_bank_attribution.json', 'user_development_feedback.json']]
    assert all(read(path)['complete'] for path in sources)
    audit = read(ANALYSIS/'independent_audit.json')
    assert audit['optimizer_updates_in_VM'] == 50 and audit['saved_paired_raw_PNG_cases_checked'] == 7910
    assert read(ANALYSIS/'visual_review.json')['viewed_sheets'] == 26
    assert not all(g['pass'] for g in audit['metric_and_gate_decisions'].values())
    ablation = read(ANALYSIS/'saved_bank_attribution.json')
    assert ablation['paired_TRAIN_ablation_cases'] == 100 and ablation['native_DEV_unpaired_cases'] == 24
    # A weaker pooled result with the bank enabled motivates scale attribution,
    # not selection of the disabled diagnostic as a trained qualified model.
    group = ablation['groups']['all|degraded']['stages']['raw']
    assert group['values']['B50']['MSE'] > group['values']['B50_bank_disabled']['MSE']
    assert group['values']['B50']['landmark_high_frequency_MSE'] > group['values']['B50_bank_disabled']['landmark_high_frequency_MSE']
    parent_protocol = read(PARENT/'protocol.json')
    assert sha(PARENT/'protocol.json') == PIN
    manifest = read(PARENT/'export_manifest.json')['files']
    required = dict(parent_protocol['assets_sha256'])
    required['protocol.json'] = PIN
    selected = [i for i in range(0, 3905, 5) if parent_protocol['cases'][i]['id'] in parent_protocol['preview_case_ids']]
    for label in ['baseline', 'B50', 'B50_bank_disabled']:
        for begin in selected:
            name = f'outputs/bank_comparison_v1/{label}/packs/b{begin//5:03d}.npz'
            required[name] = manifest[name]
        for case in parent_protocol['native_development']:
            for suffix in ['.npy', '.png']:
                name = f"outputs/bank_comparison_v1/{label}/native/{case['id']}{suffix}"
                required[name] = manifest[name]
    for name in ['outputs/bank_comparison_v1/state_B50/full_state.pth',
                 'outputs/bank_comparison_v1/state_B50/receipt.json',
                 'outputs/bank_comparison_v1/outcome_B.json', 'outputs/bank_comparison_v1/environment.json']:
        required[name] = manifest[name]
    for name, expected in required.items():
        assert sha(PARENT/name) == expected, name
    sources_by_source = sorted({parent_protocol['cases'][i]['source'] for i in selected})
    parity_begins = [next(i for i in selected if parent_protocol['cases'][i]['source'] == s) for s in sources_by_source]
    assert len(parity_begins) == 2
    dest.mkdir(); (dest/'scripts').mkdir()
    worker = ROOT/'scripts/cctv_dgp_bank_path_ablation_v1_vm.py'
    shutil.copyfile(worker, dest/'scripts'/worker.name)
    shell = """#!/usr/bin/env bash
set -uo pipefail
test -n "${TMUX:-}" || exit 1
test $# -eq 1 || exit 1
test ! -e outputs || { echo 'Immutable used root; no unchanged rerun'; exit 1; }
test ! -e review.log || exit 1
command -v timeout >/dev/null || exit 1
python -B -u scripts/cctv_dgp_bank_path_ablation_v1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha "$1" --verify-transfer || exit 1
timeout --signal=TERM --kill-after=30 1230 python -B -u scripts/cctv_dgp_bank_path_ablation_v1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha "$1" --run 2>&1 | tee review.log
statuses=("${PIPESTATUS[@]}")
diagnostic_exit=${statuses[0]}
printf '%s\\n' "$diagnostic_exit" > review_exit_code.txt
printf 'Diagnostic exit code: %s\\n' "$diagnostic_exit"
timeout --signal=TERM --kill-after=30 330 python -B -u scripts/cctv_dgp_bank_path_ablation_v1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha "$1" --export 2>&1 | tee export.log
statuses=("${PIPESTATUS[@]}")
export_exit=${statuses[0]}
printf '%s\\n' "$export_exit" > export_exit_code.txt
printf 'Export exit code: %s\\n' "$export_exit"
test "$export_exit" -eq 0 || exit "$export_exit"
exit "$diagnostic_exit"
"""
    (dest/'scripts/run_paths.sh').write_text(shell, encoding='utf-8', newline='\n')
    p = dict(format='frozen-rejected-B-bank-readout-attribution-v1', name=NAME,
        parent_name='cctv_dgp_bank_photometric_guard_v1_vm', parent_protocol_sha256=PIN,
        parent_required_sha256=required, assets_sha256={f.relative_to(dest).as_posix():sha(f) for f in dest.rglob('*') if f.is_file()},
        case_ids=parent_protocol['preview_case_ids'], native_case_ids=[c['id'] for c in parent_protocol['native_development']],
        parity_begins=parity_begins, paired_cases=100, native_cases=24, variants=['enabled', 'disabled', 'drop16', 'drop32', 'drop64', 'drop128', 'drop256'],
        rejected_B50_model_state=read(PARENT/'outputs/bank_comparison_v1/state_B50/receipt.json')['model_state'],
        original_checkpoint_sha256=parent_protocol['original_checkpoint_sha256'],
        preserved_scientific_thresholds=parent_protocol['scientific_thresholds'],
        optimizer_updates=0, gradient_queries=0, backwards=0, full_epochs=0, final_identity_pixels=0,
        worker_seconds=1200, external_seconds=1230, kill_grace_seconds=30, inference_seconds=900,
        export_seconds=300, export_external_seconds=330, peak_VRAM_bytes=20*1024**3,
        minimum_free_bytes=4*1024**3, disk_reserve_bytes=1024**3, output_byte_cap=1024**3,
        model_forward_cap=312, recognizer_forward_cap=140, original_forward_cap=0,
        paired_variant_compositions=700, native_variant_compositions=168,
        frozen_rejected_checkpoint_measurement_only=True, no_training_checkpoint_created=True,
        same_L4_enabled_disabled_raw_and_embedding_replay_exact_required=True,
        zero_all_readout_equals_disabled_parity_required=True, manual_tmux_required=True,
        conditioned_prior_unchanged=True, no_global_bank_feature_centering=True,
        no_pixel_oracle_or_degradation_profile_in_model_inference=True,
        ablation_not_independently_trained_architecture_comparison=True,
        full_3905_case_quality_gate_not_claimed=True, no_strength_or_variant_app_selection=True,
        native_unpaired=True, independent_final_review_pending=True,
        all_seven_completion_families_pending=True, app_promotion=False,
        automatic_follow_on=False, failed_run_resume_allowed=False, goal_complete=False,
        evidence_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in sources})
    write(dest/'protocol.json', p); pin = sha(dest/'protocol.json')
    spec = importlib.util.spec_from_file_location('prepared_bank_path_validator', worker)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.verify(dest, PARENT, pin)
    tree = ast.parse(worker.read_text(encoding='utf-8'))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert not any(isinstance(n.func, ast.Attribute) and n.func.attr in {'backward', 'step', 'enable_vm_learning'} for n in calls)
    assert not any(isinstance(n.func, ast.Attribute) and n.func.attr in {'grad', 'Adam', 'AdamW', 'SGD'} for n in calls)
    with tarfile.open(archive, 'x:gz', compresslevel=1) as tar:
        for f in sorted(dest.rglob('*')):
            if f.is_file():
                tar.add(f, arcname=NAME+'/'+f.relative_to(dest).as_posix(), recursive=False)
    digest = sha(archive)
    with Path(str(archive)+'.sha256').open('x', encoding='ascii', newline='\n') as stream:
        stream.write(digest+'  '+archive.name+'\n')
    write(receipt, dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), name=NAME, stem=STEM,
        protocol_sha256=pin, archive_sha256=digest, archive_bytes=archive.stat().st_size,
        packet_assets=len(p['assets_sha256']), parent_required_assets=len(required),
        zero_neural_data_and_source_verification=True, local_model_calls=0,
        local_gradient_queries=0, local_optimizer_updates=0, manual_VM_diagnostic_pending=True,
        no_agent_VM_launch=True, full_goal_incomplete=True))
    print(dict(complete=True, protocol_sha256=pin, archive_bytes=archive.stat().st_size, manual_VM_run_pending=True))


if __name__ == '__main__':
    main()
