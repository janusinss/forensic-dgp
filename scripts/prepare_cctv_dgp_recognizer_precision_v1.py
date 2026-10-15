"""Seal a finite manual precision diagnostic; hash/file operations only."""
import ast
from datetime import datetime, timezone
import shutil
import subprocess
import sys
import tarfile

from import_cctv_dgp_recognizer_precision_v1_return import ROOT, OUT, INPUT, SOURCE, BANK, NAME, STEM, read, sha, write
from cctv_dgp_recognizer_precision_v1_contract import select_references


def text_file(path, value, encoding='utf-8'):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding=encoding, newline='\n') as f:
        f.write(value)


def main():
    analysis = OUT / 'cctv_dgp_appearance_input_response_v1_analysis'
    proofs = [analysis / n for n in ['independent_audit.json', 'boundary_review.json', 'publication/publication_audit.json']]
    fixture = OUT / 'cctv_dgp_recognizer_precision_v1_fixture.json'; proofs.append(fixture)
    assert all(read(f)['complete'] for f in proofs)
    assert read(fixture)['return_auditor_sha256'] == sha(ROOT / 'scripts/audit_cctv_dgp_recognizer_precision_v1_return.py')
    guard = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/check_cctv_dgp_recognizer_precision_v1_guard.py')],
                           capture_output=True, text=True, timeout=30)
    assert guard.returncode == 0, guard.stdout + guard.stderr
    protected = read(analysis / 'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / n) == h for n, h in protected.items())
    source = read(INPUT / 'protocol.json'); old = read(BANK / 'protocol.json')
    inputs = {n: h for n, h in read(INPUT / 'export_manifest.json')['files'].items() if n != 'export.log'}
    inputs['export_manifest.json'] = sha(INPUT / 'export_manifest.json')
    response = source['response_required_sha256']; bank = source['bank_required_sha256']
    for folder, hashes in [(INPUT, inputs), (SOURCE, response), (BANK, bank)]:
        assert all(sha(folder / n) == h for n, h in hashes.items())
    sources = ['dataset/asian_faces', 'dataset/thumbnails128x128']
    selected = select_references(read(analysis / 'boundary_review.json')['case_records'], old['cases'], sources)
    ids = {r['reference'] for r in selected}
    blocks = [b for b in source['reference_blocks'] if old['cases'][b['begin']]['source_person_or_reference'] in ids]
    assert len(blocks) == 2
    packet = OUT / NAME; assert not packet.exists(); packet.mkdir()
    copies = [(ROOT / 'scripts/cctv_dgp_recognizer_precision_v1_vm.py', 'scripts/cctv_dgp_recognizer_precision_v1_vm.py'),
              (ROOT / 'scripts/cctv_dgp_recognizer_precision_v1_contract.py', 'precision_contract.py'),
              (BANK / 'cctv_dgp_pilot.py', 'cctv_dgp_pilot.py'),
              (BANK / 'models/identity_loss.py', 'models/identity_loss.py')]
    origins = {}
    for path, name in copies:
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 10))
        target = packet / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
        origins[name] = dict(source_sha256=sha(path), exact_copy=True)
    text_file(packet / 'models/__init__.py', '')
    evidence = {'boundary_review.json': analysis / 'boundary_review.json', 'input_audit.json': analysis / 'independent_audit.json',
        'publication_audit.json': analysis / 'publication/publication_audit.json', 'saved_fixture.json': fixture,
        'contract_green.json': OUT / 'cctv_dgp_recognizer_precision_v1_signmap_green.json',
        'archive_green.json': OUT / 'cctv_dgp_recognizer_precision_v1_archive_green.json'}
    for name, path in evidence.items():
        target = packet / 'evidence' / name; target.parent.mkdir(exist_ok=True); shutil.copyfile(path, target)
    write(packet / 'evidence/manual_guard.json', dict(complete=True, worker_sha256=sha(copies[0][0]),
        checker_sha256=sha(ROOT / 'scripts/check_cctv_dgp_recognizer_precision_v1_guard.py'),
        exit_code=guard.returncode, output=guard.stdout, local_measurement_rejected=True, neural_or_training_calls=0))
    shell = f'''#!/usr/bin/env bash
set -u -o pipefail
cd "$HOME/forensic-dgp/{NAME}" || exit 1
test -n "${{TMUX:-}}" || {{ echo 'Human manual tmux required'; exit 1; }}
test "$#" -eq 1 || exit 1
test ! -e review.log || {{ echo 'Used diagnostic; retain it'; exit 1; }}
PIN="$1"
PY="$HOME/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python"
timeout --signal=TERM --kill-after=30s 480s "$PY" -B -u scripts/cctv_dgp_recognizer_precision_v1_vm.py --root . --input-root ../cctv_dgp_appearance_input_response_v1_vm --response-root ../cctv_dgp_conditioning_response_v1_vm --bank-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha "$PIN" --measure >review.log 2>&1
code=$?
printf '%s\\n' "$code" >review_exit_code.txt
cat review.log
printf 'Diagnostic exit code: %s\\n' "$code"
timeout --signal=TERM --kill-after=30s 210s "$PY" -B -u scripts/cctv_dgp_recognizer_precision_v1_vm.py --root . --input-root ../cctv_dgp_appearance_input_response_v1_vm --response-root ../cctv_dgp_conditioning_response_v1_vm --bank-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha "$PIN" --export >export.log 2>&1
export_code=$?
printf '%s\\n' "$export_code" >export_exit_code.txt
cat export.log
printf 'Export exit code: %s\\n' "$export_code"
exit "$export_code"
'''
    text_file(packet / 'scripts/run_precision.sh', shell)
    p = dict(format='recognizer-precision-v1', name=NAME, UTC=datetime.now(timezone.utc).isoformat(),
        input_required_sha256=inputs, response_required_sha256=response, bank_required_sha256=bank,
        excluded_volatile_VM_paths=dict(input=['export.log'], response=['export.log'], reason='Retain live logs; archived empty export logs grow after export'),
        original_restorer_checkpoint_sha256=source['original_restorer_checkpoint_sha256'], recognizer_state=source['recognizer_state'],
        expected_torch='2.9.1+cu129', sources=sources, selected_references=selected,
        reference_blocks=blocks, case_ids=[c for b in blocks for c in b['case_ids']],
        labels=['initial', 'plus02', 'minus02', 'repeat'], precisions=['FP32', 'FP64'],
        retained_smallest_magnitude=next(r['RGB_input_response']['magnitude'] for r in read(analysis / 'independent_audit.json')['RGB_input_and_parameter_boundary_comparisons'] if r['pair'] == 2),
        scientific_thresholds=old['scientific_thresholds'], scientific_gates_unchanged=True,
        FP64_diagnostic_only_not_official_scientific_metric=True,
        original_repeat_elementwise_rtol=1e-4, original_repeat_elementwise_atol=1e-8,
        original_repeat_norm_rtol=1e-4, original_repeat_norm_atol=1e-10,
        diagnostic_central_fraction=.05, scientific_and_existing_failure_bounds_relaxed=False,
        derivative_target='RGB leaf only; all neural parameters frozen', forward_cap=16, gradient_cap=40,
        optimizer_updates=0, accepted_updates=0, learned_weight_updates=0, restorer_forward_cap=0, full_epochs=0,
        precision_copy='Disposable double copy must round-trip to the exact frozen FP32 state',
        input_precision_policy='Promote canonical saved RGB, fixed float32 grid and float32 neutral fill; retain full FP64 encoder normalization',
        observed_reference_and_target_roles_unchanged=True, final_identity_pixels=0, native_pixels_used=0,
        worker_seconds=390, measurement_seconds=360, FP64_projection_cap_seconds=300, export_seconds=180,
        external_measurement_seconds=480, external_export_seconds=210, kill_grace_seconds=30,
        minimum_free_bytes=3 * 1024**3, disk_reserve_bytes=1024**3, output_byte_cap=256 * 1024**2,
        peak_VRAM_bytes=12 * 1024**3, peak_RSS_bytes=8 * 1024**3,
        activation_policy='Packed positive/zero flags for first five candidate items at each PReLU/ReLU; independently recountable',
        source_origins=origins, source_run_protocol_sha256=sha(INPUT / 'protocol.json'),
        no_new_restoration_or_target=True, no_new_direction_or_rate_search=True,
        app_promotion=False, automatic_follow_on=False, failed_run_resume_allowed=False,
        estimate_minutes=[2,6], estimate_with_export_minutes=[3,8], estimate_not_fresh_CUDA_evidence=True,
        no_model_or_more_epochs_qualified=True, goal_complete=False)
    p['assets_sha256'] = {f.relative_to(packet).as_posix(): sha(f) for f in packet.rglob('*') if f.is_file()}
    write(packet / 'protocol.json', p); pin = sha(packet / 'protocol.json')
    importer = ROOT / 'scripts/import_cctv_dgp_recognizer_precision_v1_return.py'; template = importer.read_text(encoding='utf-8')
    assert template.count("PIN = '__PIN__'") == 1
    text_file(OUT / 'cctv_dgp_recognizer_precision_v1_importer_template.py', template)
    importer.write_text(template.replace("PIN = '__PIN__'", "PIN = '" + pin + "'"), encoding='utf-8', newline='\n')
    archive = OUT / (STEM + '-execution.tar.gz')
    with tarfile.open(archive, 'x:gz', compresslevel=6) as tar:
        for f in sorted(packet.rglob('*')):
            if f.is_file():
                tar.add(f, arcname=NAME + '/' + f.relative_to(packet).as_posix(), recursive=False)
    digest = sha(archive); text_file(OUT / (archive.name + '.sha256'), digest + '  ' + archive.name + '\n', 'ascii')
    write(OUT / 'cctv_dgp_recognizer_precision_v1_preparation.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        preparer_sha256=sha(__file__), name=NAME, protocol_sha256=pin, archive_sha256=digest, archive_bytes=archive.stat().st_size,
        packet_bindings=len(p['assets_sha256']) + 1, dependency_bindings={k: len(p[k]) for k in ['input_required_sha256', 'response_required_sha256', 'bank_required_sha256']},
        selected_references=selected, preparation_evidence_sha256={f.relative_to(ROOT).as_posix(): sha(f) for f in proofs},
        immutable_importer_sha256=sha(importer), local_neural_or_training_calls=0, VM_launch=False, manual_UNRUN=True,
        model_qualified=False, more_epochs_qualified=False, goal_complete=False))
    print(dict(complete=True, protocol_sha256=pin, archive_sha256=digest, cases=10, manual_UNRUN=True))


if __name__ == '__main__':
    main()
