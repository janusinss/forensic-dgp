"""Prepare one batch-formation pilot after the independently audited V30 diagnostic."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tarfile
import time
from cctv_dgp_profile_batches_v31_schedule import make_profile_batches

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'outputs/cctv_dgp_broader_mean_vm_v30'
CLOSED = ROOT / 'outputs/cctv_dgp_broader_mean_v30_return'
DIAGNOSTIC = ROOT / 'outputs/cctv_dgp_v30_sampling_gradient_v1_return'
NAME = 'cctv_dgp_profile_batches_vm_v31'
STEM = 'cctv-dgp-profile-batches-v31'
BUNDLE = ROOT / 'outputs' / NAME
PREP = ROOT / 'outputs/cctv_dgp_profile_batches_v31_preparation'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream: stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def replace_once(value, old, new):
    assert value.count(old) == 1, old
    return value.replace(old, new)


def renamed(text):
    for old, new in [('own-DGP-broader-mean-centered-original-decoder-v30', 'own-DGP-profile-matched-mean-centered-original-decoder-v31'),
                     ('cctv_dgp_broader_mean_vm_v30', NAME), ('cctv_dgp_broader_mean_v30_return', 'cctv_dgp_profile_batches_v31_return'),
                     ('cctv-dgp-broader-mean-v30', STEM), ('cctv_dgp_broader_mean_v30', 'cctv_dgp_profile_batches_v31'),
                     ('dgp_candidate_v30', 'dgp_candidate_v31'), ('V30', 'V31')]:
        text = text.replace(old, new)
    return text


ADDED_CHECK = '''
def paired_batch_and_closed_check(root, p):
    from cctv_dgp_profile_batches_v31_schedule import validate_profile_batches
    schedule = read(root / 'schedule.json')
    validate_profile_batches(p['case_rows'], schedule['batches'])
    assert len(schedule['batches']) == 800 and schedule['clear_controls_per_batch'] == 1
    assert sorted(i for b in schedule['batches'][:781] for i in b) == list(range(3905))
    assert len(set(i for b in schedule['batches'][781:] for i in b)) == 95
    for folder_name, key in [('cctv_dgp_broader_mean_vm_v30', 'closed_V30_evidence_sha256'),
                             ('cctv_dgp_v30_sampling_gradient_v1_vm', 'closed_sampling_evidence_sha256')]:
        folder = root.parent / folder_name
        for name, digest in p[key].items():
            path = (folder / name).resolve()
            assert path.is_relative_to(folder) and path.is_file() and sha(path) == digest, name
    old = root.parent / 'cctv_dgp_broader_mean_vm_v30'
    failure = read(old / 'outputs/failure.json'); early = read(old / 'outputs/early_structure_stop.json')
    assert failure['optimizer_updates'] == 50 and not failure['resume_permitted']
    assert early['minimum'] == .01 and not early['pass'] and not (old / 'outputs/results.json').exists()
    result = read(root.parent / 'cctv_dgp_v30_sampling_gradient_v1_vm/outputs/results.json')
    assert result['complete'] and result['component_gradient_calls'] == 280
    assert result['optimizer_updates'] == result['backwards'] == result['epochs'] == 0
    assert result['V30_failure_retained'] and result['DGP_and_recognizer_unmodified']

'''


def worker():
    original = (OLD / 'scripts/cctv_dgp_broader_mean_v30_vm.py').read_text()
    text = renamed(original)
    text = replace_once(text, '\ndef run(root, parent, p, pin):', ADDED_CHECK + '\ndef run(root, parent, p, pin):')
    text = replace_once(text, '        broader_check(root, p)\n        base = parent_check(parent, p)',
                        '        broader_check(root, p)\n        paired_batch_and_closed_check(root, p)\n        base = parent_check(parent, p)')
    text = replace_once(text, '        broader_check(root, p)\n        base = parent_check(parent,p)',
                        '        broader_check(root, p)\n        paired_batch_and_closed_check(root, p)\n        base = parent_check(parent,p)')
    text = replace_once(text, 'lambda:(parent_check(parent,p),closed_v28_and_diagnostic_check(parent,p),broader_check(root,p))',
                        'lambda:(parent_check(parent,p),closed_v28_and_diagnostic_check(parent,p),broader_check(root,p),paired_batch_and_closed_check(root,p))')
    return text


def training():
    text = renamed((OLD / 'cctv_dgp_broader_mean_v30_training.py').read_text())
    text = replace_once(text, "    schedule=json.loads((root/'schedule.json').read_text(encoding='utf-8'))['batches']",
                        "    schedule=json.loads((root/'schedule.json').read_text(encoding='utf-8'))['batches']\n    from cctv_dgp_profile_batches_v31_schedule import validate_profile_batches\n    validate_profile_batches(p['case_rows'], schedule)")
    text = replace_once(text, "'coverage_changed_only':True", "'batch_formation_changed_only':True,'clear_and_degraded_views_paired_in_every_batch':True")
    return text


def return_auditor_template():
    text = renamed((ROOT / 'scripts/audit_cctv_dgp_broader_mean_v30_return.py').read_text())
    text = replace_once(text, "PIN = 'b7b6e26bef102b5ca873d4ff01cd4c4df6bcabfb898e56bd899757ac99b65aa1'", "PIN = 'PROFILE_BATCHES_PROTOCOL_PIN'")
    gate = '''
def verify_early_receipt(receipt, gain):
    assert receipt['update'] == 50 and receipt['minimum'] == .01
    assert np.isfinite(gain) and np.isfinite(receipt['relative_feature_error_gain'])
    assert abs(receipt['relative_feature_error_gain'] - gain) <= 1e-12
    assert receipt['pass'] == (gain >= .01) == (receipt['relative_feature_error_gain'] >= .01)

'''
    text = replace_once(text, '\ndef audit(expected_sha, expected_bytes):', gate + '\ndef audit(expected_sha, expected_bytes):')
    text = replace_once(text, "        assert e['update']==50 and e['minimum']==.01 and e['relative_feature_error_gain']==gain and e['pass']==(gain>=.01)",
                        '        verify_early_receipt(e,gain)')
    text = replace_once(text, "        assert terminal['coverage_changed_only'] and terminal['mean_centered_path_used'] and terminal['original_seven_loss_weights_unchanged']",
                        "        assert terminal['batch_formation_changed_only'] and terminal['clear_and_degraded_views_paired_in_every_batch'] and terminal['mean_centered_path_used'] and terminal['original_seven_loss_weights_unchanged']")
    addition = '''    spec = importlib.util.spec_from_file_location('pinned_V31_metadata_schedule', BUNDLE/'cctv_dgp_profile_batches_v31_schedule.py')
    schedule_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(schedule_module)
    schedule_module.validate_profile_batches(p['case_rows'], read(BUNDLE/'schedule.json')['batches'])
    assert sorted(i for b in read(BUNDLE/'schedule.json')['batches'][:781] for i in b)==list(range(3905))
    old = ROOT/'outputs/cctv_dgp_broader_mean_vm_v30'
    returned_old = ROOT/'outputs/cctv_dgp_broader_mean_v30_return'
    oldp=read(old/'protocol.json')
    for name,digest in p['closed_V30_evidence_sha256'].items():
        path=old/name if name=='protocol.json' or name in oldp['assets_sha256'] else returned_old/name
        assert sha(path)==digest
    sampling=ROOT/'outputs/cctv_dgp_v30_sampling_gradient_v1_return'
    for name,digest in p['closed_sampling_evidence_sha256'].items(): assert sha(sampling/name)==digest
'''
    text = replace_once(text, '    exported, imported = import_return(expected_sha,expected_bytes,p)', addition + '    exported, imported = import_return(expected_sha,expected_bytes,p)')
    return text


def guide(pin, digest):
    return f'''# V31: paired clear/degraded batches - five manual steps

The independently audited 280-query diagnostic is complete, with zero training
updates. Four unexposed clear controls show raw pixel drift; preservation terms
therefore retain a valid role. V31 tests one change: every minibatch contains a
reference's clear control and its four degradations. All 781 approved TRAIN
references, original initialization, 12 selected decoder tensors, seven losses,
normalizers, AdamW settings and numeric gates remain fixed.

This is a new finite experiment, not a resume of V30. Maximum 800 updates
(1 complete 781-reference epoch plus 19 batches). The unchanged early gate
requires 1% structure gain at update50. Final TRAIN gates require 10% gain,
all 17 preservation groups, both source gains >=0 and mean-only fraction <=20%.
Training success remains insufficient for native CCTV or app qualification.

Existing NVIDIA L4/g2-standard-4 VM only, below ~/forensic-dgp. Require 6 GiB
free. Estimate 15-35 minutes plus export. Enforced preflight 300s, cache 900s,
fit 3600s, worker 4500s, external 4800s + 30s grace, export 900s
(external 930s + 30s grace), allocated VRAM <=20 GiB, uncompressed export <=3 GiB.
No other GPU workload may be running. Original research and all failed gates remain.

Protocol SHA256: {pin}
Execution archive SHA256: {digest}

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{STEM}-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{STEM}-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c {STEM}-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -f ~/forensic-dgp/cctv_dgp_v30_sampling_gradient_v1_vm/outputs/results.json &&
test ! -e ~/forensic-dgp/{NAME} &&
tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_profile_batches_v31
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_profile_batches_v31_vm.py --root . --protocol-sha {pin} --verify-transfer &&
bash scripts/run_v31.sh {pin}
```

Detach with Ctrl+B, release both keys, then D. Retain any stop or failed gate.
Export `complete:true` confirms packaging. The local independent return audit
and visible-structure review determine the next step. App qualification remains
pending useful development outputs and independent review.

5. Download from **Windows Google Cloud SDK Shell**, using separate PuTTY calls:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-export.json" "."
```
'''


def main():
    start = time.monotonic(); assert not BUNDLE.exists() and not PREP.exists()
    audit_path = ROOT / 'outputs/cctv_dgp_v30_sampling_gradient_v1_independent_audit.json'
    audit = read(audit_path)
    assert audit['complete'] and audit['diagnostic_complete'] and audit['optimizer_updates'] == 0
    assert audit['CPU_replay']['cases_at_both_states'] == 200
    analysis_path = ROOT / 'outputs/cctv_dgp_v30_sampling_gradient_v1_analysis/analysis.json'
    analysis = read(analysis_path); assert analysis['complete'] and analysis['all_selected12_improvement_gradients_nonzero']
    assert len(analysis['pixel_regressions']) == 5
    old = read(OLD / 'protocol.json'); assert sha(OLD / 'protocol.json') == 'b7b6e26bef102b5ca873d4ff01cd4c4df6bcabfb898e56bd899757ac99b65aa1'
    failure = read(CLOSED / 'outputs/failure.json'); assert failure['optimizer_updates'] == 50 and not failure['resume_permitted']
    schedule, references = make_profile_batches(old['case_rows'], read(OLD / 'schedule.json')['batches'], 781, 800)
    assert len(set(references[:50])) == 50
    for name, digest in old['assets_sha256'].items(): assert sha(OLD / name) == digest
    mixed = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'
    for name, digest in old['mixed_TRAIN_assets_sha256'].items(): assert sha(mixed / name) == digest
    BUNDLE.mkdir(); (BUNDLE / 'scripts').mkdir(); PREP.mkdir()
    for name in ['cctv_dgp_mean_centered_decoder_v29.py', 'cctv_dgp_app_input_v28.py']:
        (BUNDLE / name).write_bytes((OLD / name).read_bytes())
    (BUNDLE / 'cctv_dgp_profile_batches_v31_schedule.py').write_bytes((ROOT / 'scripts/cctv_dgp_profile_batches_v31_schedule.py').read_bytes())
    (BUNDLE / 'cctv_dgp_profile_batches_v31_cache.py').write_text(renamed((OLD / 'cctv_dgp_broader_mean_v30_cache.py').read_text()), encoding='utf-8', newline='\n')
    (BUNDLE / 'cctv_dgp_profile_batches_v31_training.py').write_text(training(), encoding='utf-8', newline='\n')
    (BUNDLE / 'scripts/cctv_dgp_profile_batches_v31_vm.py').write_text(worker(), encoding='utf-8', newline='\n')
    (BUNDLE / 'scripts/run_v31.sh').write_text(renamed((OLD / 'scripts/run_v30.sh').read_text()), encoding='ascii', newline='\n')
    write(BUNDLE / 'schedule.json', {'seed': 293001, 'selection': 'Reference first-appearance order from the frozen V30 case permutation in each epoch; pair all five profiles',
                                    'batches': schedule, 'reference_order': references, 'updates': 800,
                                    'first_epoch_complete': 781, 'second_epoch_batches': 19, 'clear_controls_per_batch': 1})
    p = copy.deepcopy(old)
    p.update({'format': 'own-DGP-profile-matched-mean-centered-original-decoder-v31',
              'purpose': 'Test paired clear/degraded reference anchoring after audited V30 early failure and measured clear-control drift.',
              'design': 'Only batch formation changes. Same full TRAIN corpus, original initialization, decoder, losses, normalizers, optimizer and numerical gates.',
              'difference_from_failed_recipes': 'V30 mixed arbitrary cases across references; V31 pairs every reference clear control with all four degraded views in each batch. It does not continue V30 or repeat V29 on only ten references.',
              'coverage_policy': 'All781 references/all3905 cases once in the first781 updates, then19 distinct references/all95 cases. First50 updates use50 references,250 cases and50 clear controls.',
              'hypothesis': 'Paired anchoring may improve simultaneous degraded reconstruction and clear-control preservation. The diagnostic did not establish a unique cause or predict success.',
              'schedule_profile_order': ['clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24'],
              'losses_optimizer_architecture_initialization_and_numeric_gates_unchanged': True,
              'batch_formation_changed_only': True, 'human_manual_VM_execution_required': True,
              'capacity_policy': old['capacity_policy'], 'next': 'Audit return and all50 TRAIN previews, then repeat fixed520 paired DEV and24 unpaired native development if capacity passes. Never auto-promote or open reserved final pixels.'})
    closed_names = ['protocol.json', *old['assets_sha256'], 'outputs/failure.json', 'outputs/early_structure_stop.json',
                    'outputs/execution_receipt.json', 'outputs/update50/metrics.json', 'outputs/update50/dgp_candidate_v30.pth']
    p['closed_V30_evidence_sha256'] = {name: sha(OLD / name if name == 'protocol.json' or name in old['assets_sha256'] else CLOSED / name) for name in closed_names}
    names = ['protocol.json', 'scripts/cctv_dgp_v30_sampling_gradient_v1_vm.py', 'scripts/run_sampling.sh',
             'outputs/results.json', 'supervisor_receipt.json', 'diagnostic_exit_code.txt']
    for state in [0, 50]:
        for cohort in ['exposed', 'unexposed']: names.extend(f'outputs/state{state}_{cohort}/{n}' for n in ['receipt.json', 'gradient_components.npy'])
    p['closed_sampling_evidence_sha256'] = {name: sha(DIAGNOSTIC / name) for name in names}
    basis = [audit_path, analysis_path, ROOT / 'CCTV_DGP_V30_SAMPLING_GRADIENT_V1_RESULTS.md', Path(__file__),
             ROOT / 'scripts/cctv_dgp_profile_batches_v31_schedule.py', ROOT / 'scripts/verify_cctv_dgp_profile_batches_v31_packet.py',
             ROOT / 'tests/test_cctv_dgp_profile_batches_v31.py', ROOT / 'scripts/audit_cctv_dgp_broader_mean_v30_return.py']
    p['local_basis_sha256'] = {path.relative_to(ROOT).as_posix(): sha(path) for path in basis}
    p['assets_sha256'] = {path.relative_to(BUNDLE).as_posix(): sha(path) for path in sorted(BUNDLE.rglob('*')) if path.is_file()}
    for path in BUNDLE.rglob('*.py'): ast.parse(path.read_text(), feature_version=(3, 10))
    write(BUNDLE / 'protocol.json', p); pin = sha(BUNDLE / 'protocol.json')
    template = return_auditor_template(); assert template.count('PROFILE_BATCHES_PROTOCOL_PIN') == 1
    template_path = ROOT / 'scripts/cctv_dgp_profile_batches_v31_return_audit_template.py'
    with template_path.open('x', encoding='utf-8', newline='\n') as stream: stream.write(template)
    checker = ROOT / 'scripts/audit_cctv_dgp_profile_batches_v31_return.py'
    with checker.open('x', encoding='utf-8', newline='\n') as stream: stream.write(template.replace('PROFILE_BATCHES_PROTOCOL_PIN', pin))
    archive = ROOT / 'outputs' / (STEM + '-execution.tar.gz')
    with tarfile.open(archive, 'x:gz', compresslevel=6) as tar:
        for path in sorted(BUNDLE.rglob('*')):
            if path.is_file(): tar.add(path, arcname=NAME + '/' + path.relative_to(BUNDLE).as_posix(), recursive=False)
    digest = sha(archive)
    with Path(str(archive) + '.sha256').open('x', encoding='ascii', newline='\n') as stream: stream.write(digest + '  ' + archive.name + '\n')
    with (ROOT / 'CCTV_DGP_PROFILE_BATCHES_V31_VM.md').open('x', encoding='ascii', newline='\n') as stream: stream.write(guide(pin, digest))
    receipt = {'complete': True, 'protocol_sha256': pin, 'archive_sha256': digest, 'archive_bytes': archive.stat().st_size,
               'packet_files': len(p['assets_sha256']) + 1, 'TRAIN_assets_verified': 5467, 'training_cases': 3905,
               'training_references': 781, 'finite_updates': 800, 'first50_clear_controls': 50, 'first50_distinct_references': 50,
               'prospective_return_auditor_sha256': sha(checker), 'prospective_template_sha256': sha(template_path),
               'original_data_or_weights_uploaded': False, 'local_neural_or_gradient_calls': 0, 'actual_VM_training_started': False,
               'human_manual_VM_execution_required': True, 'app_promotion': False, 'goal_complete': False,
               'seconds': time.monotonic() - start}
    write(PREP / 'preparation.json', receipt); print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
