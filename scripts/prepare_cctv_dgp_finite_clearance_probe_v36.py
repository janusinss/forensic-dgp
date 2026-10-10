"""Prepare a changed, finite manual probe; never launch VM work or neural training."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
NAME = 'cctv_dgp_finite_clearance_probe_v36_vm'
STEM = 'cctv-dgp-finite-clearance-probe-v36'
BUNDLE = ROOT / 'outputs' / NAME
PREP = ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_preparation'
MATH = ROOT / 'outputs/cctv_dgp_v35_finite_clearance_v1'
OLD = ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_vm'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def text_file(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(value)


def guide(pin, digest):
    return f'''# V36: four disposable finite-clearance trials, five manual steps

V35's independently audited500 outputs improve detail but all four scales fail
preservation in both TRAIN subsets. Raw outputs also regress; PNG conversion
alone does not explain the failure. All100 case comparisons were actually viewed.

V36 changes the direction. All108 original rows remain;65 raw MSE/ArcFace rows
require twice V35 scale1's measured positive raw/PNG departure from the linear
prediction. SSIM and six restoration rows retain zero minimum clearance because
the saved skimage SSIM differs from the differentiable VM definition. This is
an empirical hypothesis, not a validated bound or a weakened quality gate.
The independent full-row KKT check must pass before using these commands.

Four reset trials at scales1,1/2,1/4,1/8;100 original plus400 trial outputs.
Zero optimizer updates, gradients, backwards, epochs, committed trajectory or
new checkpoint. No V32/V35 continuation or historical pilot. The original DGP
and recognizer stay frozen. Actual17 PNG groups, source gains, brightness and
full-corpus1%-at50/10%-at800 gates remain. TRAIN subsets cannot qualify the app,
independent final faces or native CCTV. Source labels are not ethnicity.

Require idle running NVIDIA L4/g2-standard-4 and6GiB free. V35 took94seconds
plus26seconds export. Estimate2-4minutes plus1-2minutes export for V36; unrun.
Worker900s/external930s+30s grace; export300s/external330s+30s grace;
20GiB allocated VRAM;768MiB uncompressed return;2300 files. Failures are exported.

Protocol SHA256:{pin}
Execution archive SHA256:{digest}

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
test ! -e ~/forensic-dgp/{NAME} &&
tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_finite_clearance_v36
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/{NAME}.py --root . --protocol-sha {pin} --verify-transfer &&
bash scripts/run_v36_probe.sh {pin}
```

Detach with **Ctrl+B**, release, **D**. Reattach with
`tmux attach -t dgp_finite_clearance_v36`. Inspect
`tail -n 25 ~/forensic-dgp/{NAME}/probe.log`.
An export `complete:true` records a transfer; it does not imply preservation passed.

5. Download from **Windows Google Cloud SDK Shell**, one remote source per command:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-export.json" "."
```

No VM training or connection is started by packet preparation. Returned files
require independent bounded CPU replay and complete numerical/visual review.
Do not rerun an existing or failed directory. Original checkpoints, splits,
research caches, failures, local backups and the current app remain preserved.
'''


def main():
    assert not BUNDLE.exists() and not PREP.exists()
    a = read(MATH / 'analysis.json')
    assert a['complete'] and a['geometry_qualified_for_disposable_probe'] and a['nonzero_clearance_rows'] == 65
    assert a['candidate_magnitude_ratio'] <= 2 and a['neural_or_gradient_calls'] == 0
    import numpy as np
    p = read(OLD / 'protocol.json')
    for name, digest in p['assets_sha256'].items():
        assert sha(OLD / name) == digest, name
    for name, digest in a['basis_sha256'].items():
        assert sha(ROOT / name) == digest, name
    worker = (OLD / 'scripts/cctv_dgp_group_guard_probe_v35_r1_vm.py').read_text()
    worker = worker.replace('cctv_dgp_group_guard_probe_v35_r1_vm', NAME)
    worker = worker.replace('cctv-dgp-group-guard-probe-v35-r1', STEM)
    worker = worker.replace('cctv_dgp_group_guard_probe_v35_return/', 'cctv_dgp_finite_clearance_probe_v36_return/')
    worker = worker.replace('own-DGP-all-group-guard-finite-image-probe-v35', 'own-DGP-empirical-clearance-finite-image-probe-v36')
    worker = worker.replace('cctv_dgp_group_guard_geometry_v35', 'cctv_dgp_finite_clearance_geometry_v36')
    worker = worker.replace('joint_', 'clearance_').replace('V35', 'V36')
    auditor = (ROOT / 'scripts/audit_cctv_dgp_group_guard_probe_v35_r1_return_r2.py').read_text()
    auditor = auditor.replace('cctv_dgp_group_guard_probe_v35_r1_vm', NAME)
    auditor = auditor.replace('cctv-dgp-group-guard-probe-v35-r1', STEM)
    auditor = auditor.replace('cctv_dgp_group_guard_probe_v35_r1_return', 'cctv_dgp_finite_clearance_probe_v36_return')
    auditor = auditor.replace('cctv_dgp_group_guard_probe_v35_return/', 'cctv_dgp_finite_clearance_probe_v36_return/')
    auditor = auditor.replace('cctv_dgp_group_guard_probe_v35_r1_independent_audit_r2', 'cctv_dgp_finite_clearance_probe_v36_independent_audit')
    for text in [worker, auditor]:
        ast.parse(text, feature_version=(3, 10))
        assert 'torch.optim' not in text and 'autograd.grad' not in text and '.backward(' not in text
    BUNDLE.mkdir()
    (BUNDLE / 'scripts').mkdir()
    PREP.mkdir()
    worker_name = NAME + '.py'
    checker = ROOT / 'scripts/audit_cctv_dgp_finite_clearance_probe_v36_return.py'
    text_file(ROOT / 'scripts' / worker_name, worker)
    text_file(BUNDLE / 'scripts' / worker_name, worker)
    text_file(checker, auditor)
    shell = (OLD / 'scripts/run_v35_probe.sh').read_text().replace('cctv_dgp_group_guard_probe_v35_r1_vm', NAME).replace('V35', 'V36')
    text_file(BUNDLE / 'scripts/run_v36_probe.sh', shell)
    for source, name in [(ROOT / 'scripts/cctv_dgp_loss_cone_probe_v33_metrics.py', 'cctv_dgp_loss_cone_probe_v33_metrics.py'),
                         (ROOT / 'scripts/cctv_dgp_finite_clearance_geometry_v36.py', 'cctv_dgp_finite_clearance_geometry_v36.py'),
                         (OLD / 'theta_before.npy', 'theta_before.npy'),
                         (OLD / 'mean_original_displacement.npy', 'mean_original_displacement.npy'),
                         (MATH / 'candidate_displacement.npy', 'projected_displacement.npy')]:
        with (BUNDLE / name).open('xb') as stream:
            stream.write(source.read_bytes())
    length = float(np.linalg.norm(np.load(OLD / 'mean_original_displacement.npy', allow_pickle=False)))
    targets = np.load(MATH / 'clearance_targets.npy', allow_pickle=False)
    matrix = np.load(ROOT / 'outputs/cctv_dgp_group_guard_grad_v34_analysis_v1_r1/group_guard_matrix.npy', allow_pickle=False)
    extra = [np.load(ROOT / f'outputs/cctv_dgp_v32_loss_gradient_v1_return/outputs/state0_{c["name"]}/gradient_components.npy', allow_pickle=False)[:3]
             for c in p['cohorts']]
    full = np.concatenate([matrix, *extra], axis=0)
    direction = np.load(BUNDLE / 'projected_displacement.npy', allow_pickle=False)
    proof = {'policy': a['policy'], 'clearance_targets': targets.tolist(),
        'multipliers': (length * np.asarray(a['multipliers'])).tolist(), 'KKT_tolerance': 2e-10 * length,
        'raw_dots_after': (full @ direction).tolist(), 'magnitude_ratio': a['candidate_magnitude_ratio'],
        'all108_constraints_retained': True, 'empirical_not_a_validated_bound': True,
        'finite_preservation_implied': False, 'source_analysis_sha256': sha(MATH / 'analysis.json')}
    text_file(BUNDLE / 'projection.json', json.dumps(proof, indent=2, allow_nan=False) + '\n')
    p.update({'format': 'own-DGP-empirical-clearance-finite-image-probe-v36', 'frozen_UTC': datetime.now(timezone.utc).isoformat(),
        'hypothesis': 'Reserve twice the measured V35 scale1 finite raw/PNG departure for MSE/ArcFace; finite outputs must still pass unchanged gates',
        'variants': [{'name': name, 'proposal': 'projected_restoration', 'scale': scale}
                     for name, scale in [('clearance_1', 1.), ('clearance_half', .5), ('clearance_quarter', .25), ('clearance_eighth', .125)]],
        'clearance_targets': targets.tolist(), 'clearance_policy_empirical_only': True,
        'V35_all4_scales_failed_preservation_retained': True, 'VM_execution_started': False})
    new_basis = [ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_independent_audit_r2.json',
        ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_analysis_v1_r1/analysis.json',
        ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_analysis_v1_r1/independent_analysis_page_audit.json',
        MATH / 'analysis.json', MATH / 'candidate_displacement.npy', MATH / 'clearance_targets.npy', checker]
    p['local_basis_sha256'].update({q.relative_to(ROOT).as_posix(): sha(q) for q in new_basis})
    p['assets_sha256'] = {q.relative_to(BUNDLE).as_posix(): sha(q) for q in sorted(BUNDLE.rglob('*')) if q.is_file()}
    text_file(BUNDLE / 'protocol.json', json.dumps(p, indent=2, allow_nan=False) + '\n')
    pin = sha(BUNDLE / 'protocol.json')
    archive = ROOT / 'outputs' / (STEM + '-execution.tar.gz')
    with tarfile.open(archive, 'x:gz', compresslevel=3) as stream:
        for q in sorted(BUNDLE.rglob('*')):
            if q.is_file():
                stream.add(q, arcname=NAME + '/' + q.relative_to(BUNDLE).as_posix(), recursive=False)
    digest = sha(archive)
    text_file(Path(str(archive) + '.sha256'), digest + '  ' + archive.name + '\n')
    text_file(ROOT / 'CCTV_DGP_FINITE_CLEARANCE_PROBE_V36_VM.md', guide(pin, digest))
    receipt = {'complete': True, 'protocol_sha256': pin, 'archive_sha256': digest, 'archive_bytes': archive.stat().st_size,
        'assets': len(p['assets_sha256']), 'VM_calls': 0, 'neural_or_gradient_calls': 0, 'optimizer_updates': 0,
        'independent_packet_audit_pending': True, 'VM_execution_started': False, 'app_promotion': False, 'goal_complete': False}
    text_file(PREP / 'preparation_receipt.json', json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'complete': True, 'protocol_sha256': pin, 'archive_sha256': digest, 'bytes': receipt['archive_bytes']}))


if __name__ == '__main__':
    main()
