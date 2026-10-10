"""Prepare a distinct finite V38 packet from audited arrays; never launch a VM."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
NAME = 'cctv_dgp_delivered_margin_probe_v38_vm'
STEM = 'cctv-dgp-delivered-margin-probe-v38'
BUNDLE = ROOT / 'outputs' / NAME
PREP = ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_preparation'
OLD = ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_vm'
MATH = ROOT / 'outputs/cctv_dgp_v37_delivered_margins_v1'
PNG = ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_return'
V36 = ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_return'


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def text_file(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream: stream.write(value)


def replacement(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def guide(pin, digest):
    return f'''# V38: four disposable PNG-margin trials, five manual steps

V37's diagnostic is independently audited. It made zero training updates and
predicted none of V36's11 PNG failures. V38 tests a different direction with210
checks, retaining all108 earlier checks and adding102 PNG checks with empirical
margins from all four V36 scales. These are TRAIN design measurements, not a
preservation guarantee or independent evaluation. No new gradients, optimizer,
training trajectory or checkpoint. Original models and all failed gates remain.

Require **6 GiB free** on the existing L4 VM. Estimate **3–6 minutes** for the
probe plus **1–3 minutes** for export. Enforced worker900s, external930s with30s
kill grace; export300s/external330s with30s grace; VRAM20GiB. Use the existing
V32-loss, V34, V36 and V37 saved dependencies; do not delete or rerun them.
Stop on any missing/hash-mismatched dependency. The agent has not launched V38.

All17 finite PNG preservation groups, source gains and brightness fraction stay
fixed. The1%-at50 and10%-at800 training requirements remain for a separately
justified future training pilot; this four-trial probe cannot pass them.
Returned outputs need independent audit and all100 comparisons reviewed before
any training decision. V29 development failures, native unpaired criteria and
all seven covering families remain binding. No app promotion or goal completion.

Protocol SHA256: `{pin}`
Execution archive SHA256: `{digest}`

1. Upload in **Windows Google Cloud SDK Shell**:

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
test -f ~/forensic-dgp/cctv_dgp_delivered_guard_grad_v37_vm/outputs/results.json &&
test ! -e ~/forensic-dgp/{NAME} &&
tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_delivered_margin_v38
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/{NAME}.py --root . --protocol-sha {pin} --verify-transfer &&
bash scripts/run_v38_probe.sh {pin}
```

Detach with Ctrl+B, then D. Reattach with `tmux attach -t dgp_delivered_margin_v38`.
Inspect `tail -n 25 ~/forensic-dgp/{NAME}/probe.log` if needed. Download after
`Export exit code: 0`; `complete:true` in the export means packaging only.
If the probe fails, retain the failure and download it; do not rerun or resume.

5. Download in **Windows Google Cloud SDK Shell**, one remote source per command:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-export.json" "."
```

The PuTTY-backed gcloud client accepts one remote source per download command.
All three files are saved in `C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs`.
''' 


def main():
    assert not BUNDLE.exists() and not PREP.exists()
    a = read(MATH / 'analysis.json'); checked = read(MATH / 'independent_geometry_audit.json')
    assert a['complete'] and checked['complete'] and a['geometry_qualified_for_disposable_probe']
    assert a['full_rows'] == 210 and a['nonzero_new_PNG_clearances'] == 97
    assert checked['all65_V36_clearances_unchanged'] and checked['all210_rows_and408_calibration_observations_verified']
    p = read(OLD / 'protocol.json')
    for name, digest in p['assets_sha256'].items(): assert sha(OLD / name) == digest, name
    for name, digest in a['basis_sha256'].items(): assert sha(ROOT / name) == digest, name
    for name, digest in a['artifact_sha256'].items(): assert sha(MATH / name) == digest, name
    assert read(ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_independent_audit.json')['diagnostic_complete']
    worker = (OLD / 'scripts/cctv_dgp_finite_clearance_probe_v36_vm.py').read_text()
    worker = worker.replace('cctv_dgp_finite_clearance_probe_v36_vm', NAME).replace('cctv-dgp-finite-clearance-probe-v36', STEM)
    worker = worker.replace('cctv_dgp_finite_clearance_probe_v36_return/', 'cctv_dgp_delivered_margin_probe_v38_return/')
    worker = worker.replace('own-DGP-empirical-clearance-finite-image-probe-v36', 'own-DGP-delivered-margin-finite-image-probe-v38')
    worker = worker.replace('cctv_dgp_finite_clearance_geometry_v36', 'cctv_dgp_delivered_margin_readback_v38')
    worker = worker.replace('clearance_', 'margin_').replace('V36', 'V38').replace('all108_linear_function_changes', 'all210_linear_function_changes')
    extra_dependencies = '''    png_root=root.parent/'cctv_dgp_delivered_guard_grad_v37_vm'
    check_files(png_root,p['PNG_gradient_return_sha256'])
    png_p=read(png_root/'protocol.json');png_result=read(png_root/'outputs/results.json')
    assert sha(png_root/'protocol.json')==p['PNG_gradient_protocol_sha256']
    assert png_p['cohorts']==p['cohorts'] and png_p['parameter_layout']==p['parameter_layout']
    assert png_p['original_DGP_state']==basis['original_DGP_state'] and png_p['recognizer_state']==basis['recognizer_state']
    assert png_p['retained_capacity_gates']==p['retained_capacity_gates']
    assert png_result['complete'] and png_result['gradient_queries']==300 and png_result['optimizer_updates']==png_result['parameter_updates']==0
    assert not (png_root/'outputs/failure.json').exists()
    check_files(root.parent/'cctv_dgp_finite_clearance_probe_v36_vm',p['V36_margin_basis_sha256'])
    check_files(root.parent/OLD,p['restoration_gradient_sha256'])
'''
    worker = replacement(worker, '    return root.parent/OLD,basis,parent,active,closed,mixed,helper', extra_dependencies + '    return root.parent/OLD,basis,parent,active,closed,mixed,helper')
    auditor = (ROOT / 'scripts/audit_cctv_dgp_finite_clearance_probe_v36_return.py').read_text()
    auditor = auditor.replace('cctv_dgp_finite_clearance_probe_v36', 'cctv_dgp_delivered_margin_probe_v38').replace('cctv-dgp-finite-clearance-probe-v36', STEM)
    auditor = auditor.replace('all108_linear_function_changes', 'all210_linear_function_changes')
    extra_basis = '''    png=ROOT/'outputs/cctv_dgp_delivered_guard_grad_v37_return'
    png_audit=read(ROOT/'outputs/cctv_dgp_delivered_guard_grad_v37_independent_audit.json')
    assert png_audit['complete'] and png_audit['diagnostic_complete'] and not png_audit['failure_retained']
    for name,digest in p['PNG_gradient_return_sha256'].items():assert sha(png/name)==digest,name
    assert sha(png/'protocol.json')==p['PNG_gradient_protocol_sha256']
    png_p=read(png/'protocol.json')
    assert png_p['cohorts']==p['cohorts'] and png_p['parameter_layout']==p['parameter_layout']
    assert png_p['original_DGP_state']==basis['original_DGP_state'] and png_p['recognizer_state']==basis['recognizer_state']
    for name,digest in p['V36_margin_basis_sha256'].items():assert sha(ROOT/'outputs/cctv_dgp_finite_clearance_probe_v36_return'/name)==digest,name
    for name,digest in p['restoration_gradient_sha256'].items():assert sha(ROOT/'outputs/cctv_dgp_v32_loss_gradient_v1_return'/name)==digest,name
'''
    auditor = replacement(auditor, '    return basis\n', extra_basis + '    return basis\n')
    start = auditor.index('        matrix=np.load(')
    end = auditor.index("        for cohort in p['cohorts']:", start)
    # Replace the entire old108-row reconstruction, not the subsequent comparisons.
    end = auditor.index("        for cohort in p['cohorts']:", end + 1)
    auditor = auditor[:start] + '        theta,direction,guards,local_geometry=geometry_readback(p)\n        assert geometry["full_rows"]==local_geometry["full_rows"]==210 and geometry["coarse_PNG_guard_rows"]==102\n        assert geometry["new_PNG_margin_rows"]==97 and geometry["calibration_observations"]==408\n' + auditor[end:]
    geometry_function = '''def geometry_readback(p):
    import numpy as np
    from unittest.mock import patch
    helper=module('pinned_V38_array_readback',BUNDLE/'cctv_dgp_delivered_margin_readback_v38.py')
    mappings=[('cctv_dgp_group_guard_grad_v34_vm','cctv_dgp_group_guard_grad_v34_return'),
              ('cctv_dgp_v32_loss_gradient_v1_vm','cctv_dgp_v32_loss_gradient_v1_return'),
              ('cctv_dgp_delivered_guard_grad_v37_vm','cctv_dgp_delivered_guard_grad_v37_return'),
              ('cctv_dgp_finite_clearance_probe_v36_vm','cctv_dgp_finite_clearance_probe_v36_return')]
    def local(path):
        path=Path(path)
        for source,target in mappings:
            base=BUNDLE.parent/source
            if path.is_relative_to(base):return BUNDLE.parent/target/path.relative_to(base)
        return path
    load=np.load
    with patch.object(np,'load',side_effect=lambda path,**kw:load(local(path),**kw)):
        return helper.verify_geometry(BUNDLE,p,lambda path:sha(local(path)),lambda path:read(local(path)))


'''
    auditor = replacement(auditor, 'def audit(digest,size):', geometry_function + 'def audit(digest,size):')
    auditor = auditor.replace('Prospective V35 audit:', 'Prospective V38 audit:')
    for source in [worker, auditor]:
        ast.parse(source, feature_version=(3, 10))
        assert not any(v in source for v in ['torch.optim', 'autograd.grad', '.backward(', 'torch.save'])
    BUNDLE.mkdir(); (BUNDLE / 'scripts').mkdir(); PREP.mkdir()
    checker = ROOT / 'scripts/audit_cctv_dgp_delivered_margin_probe_v38_return.py'
    text_file(ROOT / 'scripts' / (NAME + '.py'), worker)
    text_file(BUNDLE / 'scripts' / (NAME + '.py'), worker); text_file(checker, auditor)
    shell = (OLD / 'scripts/run_v36_probe.sh').read_text().replace('cctv_dgp_finite_clearance_probe_v36_vm', NAME).replace('V36', 'V38')
    text_file(BUNDLE / 'scripts/run_v38_probe.sh', shell)
    for source, name in [(ROOT / 'scripts/cctv_dgp_delivered_margin_readback_v38.py', 'cctv_dgp_delivered_margin_readback_v38.py'),
                         (OLD / 'cctv_dgp_loss_cone_probe_v33_metrics.py', 'cctv_dgp_loss_cone_probe_v33_metrics.py'),
                         (OLD / 'theta_before.npy', 'theta_before.npy'), (OLD / 'mean_original_displacement.npy', 'mean_original_displacement.npy'),
                         (MATH / 'candidate_displacement.npy', 'projected_displacement.npy')]:
        with (BUNDLE / name).open('xb') as stream: stream.write(source.read_bytes())
    proof = {key:a['projection'][key] for key in ['multipliers', 'KKT_tolerance', 'raw_dots_after']}
    import numpy as np
    targets = np.load(MATH / 'clearance_targets.npy', allow_pickle=False)
    proof.update({'policy': a['policy'], 'clearance_targets': targets.tolist(), 'magnitude_ratio': a['projection']['candidate_magnitude_ratio'],
                  'original108_constraints_retained': True, 'all210_constraints_retained': True, 'empirical_not_a_validated_bound': True,
                  'finite_preservation_implied': False, 'source_analysis_sha256': sha(MATH / 'analysis.json')})
    text_file(BUNDLE / 'projection.json', json.dumps(proof, indent=2, allow_nan=False) + '\n')
    png_p = read(PNG / 'protocol.json')
    png_names = ['protocol.json', *png_p['assets_sha256'], 'outputs/results.json', 'supervisor_receipt.json']
    margin_names = ['protocol.json', 'projection.json', 'projected_displacement.npy', 'theta_before.npy', 'mean_original_displacement.npy', 'outputs/results.json']
    restoration = ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_return'
    for cohort in p['cohorts']:
        png_names += [f'outputs/{cohort["name"]}/receipt.json'] + [f'outputs/{cohort["name"]}/batch{batch}_guard_gradients.npy' for batch in range(10)]
        margin_names += [f'outputs/state0_{cohort["name"]}/{variant}/receipt.json' for variant in ['before', 'clearance_1', 'clearance_half', 'clearance_quarter', 'clearance_eighth']]
    p.update({'format': 'own-DGP-delivered-margin-finite-image-probe-v38', 'frozen_UTC': datetime.now(timezone.utc).isoformat(),
              'hypothesis': 'Retain108 original guards and65 margins; add102 PNG coarse guards with97 margins from all4 V36 scales; actual finite images decide',
              'constraint_labels': a['constraint_labels'], 'clearance_targets': targets.tolist(),
              'PNG_guard_metrics': png_p['guard_metrics'], 'PNG_gradient_protocol_sha256': sha(PNG / 'protocol.json'),
              'PNG_gradient_return_sha256': {name:sha(PNG / name) for name in png_names},
              'V36_margin_basis_sha256': {name:sha(V36 / name) for name in margin_names},
              'restoration_gradient_sha256': {f'outputs/state0_{c["name"]}/gradient_components.npy':sha(restoration / f'outputs/state0_{c["name"]}/gradient_components.npy') for c in p['cohorts']},
              'variants': [{'name':name, 'proposal':'projected_restoration', 'scale':scale} for name,scale in [('margin_1',1.),('margin_half',.5),('margin_quarter',.25),('margin_eighth',.125)]],
              'all210_constraints_retained': True, 'empirical_clearance_not_validated_bound': True, 'true_PNG_derivative_claimed': False,
              'V36_all4_scales_failed_preservation_retained': True, 'V37_coarse_predictions_missed_all11_failures': True, 'VM_execution_started': False})
    basis = [MATH / 'analysis.json', MATH / 'independent_geometry_audit.json', MATH / 'clearance_targets.npy', MATH / 'candidate_displacement.npy',
             ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_independent_audit.json', ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_return_import.json',
             ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_analysis_v1/analysis.json',
             ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_analysis_v1/independent_analysis_audit.json',
             ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_analysis_v1/PNG_group_guard_matrix.npy',
             ROOT / 'CCTV_DGP_DELIVERED_GUARD_GRAD_V37_RESULTS.md', ROOT / 'scripts/cctv_dgp_delivered_margin_readback_v38.py', checker]
    p['local_basis_sha256'].update({path.relative_to(ROOT).as_posix():sha(path) for path in basis})
    p['assets_sha256'] = {path.relative_to(BUNDLE).as_posix():sha(path) for path in sorted(BUNDLE.rglob('*')) if path.is_file()}
    text_file(BUNDLE / 'protocol.json', json.dumps(p, indent=2, allow_nan=False) + '\n'); pin = sha(BUNDLE / 'protocol.json')
    archive = ROOT / 'outputs' / (STEM + '-execution.tar.gz')
    with tarfile.open(archive, 'x:gz', compresslevel=3) as tar:
        for path in sorted(BUNDLE.rglob('*')):
            if path.is_file(): tar.add(path, arcname=NAME + '/' + path.relative_to(BUNDLE).as_posix(), recursive=False)
    digest = sha(archive); text_file(Path(str(archive) + '.sha256'), digest + '  ' + archive.name + '\n')
    text_file(ROOT / 'CCTV_DGP_DELIVERED_MARGIN_PROBE_V38_VM.md', guide(pin, digest))
    receipt = {'complete':True, 'protocol_sha256':pin, 'archive_sha256':digest, 'archive_bytes':archive.stat().st_size, 'assets':len(p['assets_sha256']),
               'VM_calls':0, 'neural_or_gradient_calls':0, 'optimizer_updates':0, 'independent_packet_audit_pending':True,
               'VM_execution_started':False, 'app_promotion':False, 'goal_complete':False}
    text_file(PREP / 'preparation_receipt.json', json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__': main()
