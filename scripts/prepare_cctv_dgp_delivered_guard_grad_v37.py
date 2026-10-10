"""Freeze a distinct, zero-update PNG guard diagnostic; no VM operations."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
NAME = 'cctv_dgp_delivered_guard_grad_v37_vm'
STEM = 'cctv-dgp-delivered-guard-grad-v37'
BUNDLE = ROOT / 'outputs' / NAME
PREP = ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_preparation'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def text_file(path, text):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream: stream.write(text)


def write(path, value): text_file(path, json.dumps(value, indent=2, allow_nan=False) + '\n')


SHELL = '''#!/usr/bin/env bash
set -uo pipefail
PIN="${1:?Pass the exact V37 protocol SHA256}"
ROOT="$HOME/forensic-dgp/cctv_dgp_delivered_guard_grad_v37_vm"
PYTHON="$HOME/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python"
WORKER="$ROOT/scripts/cctv_dgp_delivered_guard_grad_v37_vm.py"
cd "$ROOT" || exit 1
test ! -e outputs && test ! -e diagnostic.log && test ! -e supervisor_receipt.json && test ! -e export_manifest.json || exit 1
"$PYTHON" -B "$WORKER" --root "$ROOT" --protocol-sha "$PIN" --verify-transfer || exit 1
START=$("$PYTHON" -c 'import time; print(time.monotonic())')
timeout --signal=TERM --kill-after=30s 930s "$PYTHON" -B -u "$WORKER" --root "$ROOT" --protocol-sha "$PIN" --run 2>&1 | tee diagnostic.log
CODE=${PIPESTATUS[0]}
printf '%s\\n' "$CODE" > diagnostic_exit_code.txt
ELAPSED=$("$PYTHON" -c 'import sys,time; print(time.monotonic()-float(sys.argv[1]))' "$START")
"$PYTHON" -B "$WORKER" --root "$ROOT" --protocol-sha "$PIN" --record-supervision --elapsed "$ELAPSED" --exit-code "$CODE" || exit 1
printf 'Diagnostic exit code: %s\\n' "$CODE"
timeout --signal=TERM --kill-after=30s 330s "$PYTHON" -B -u "$WORKER" --root "$ROOT" --protocol-sha "$PIN" --export
EXPORT_CODE=$?
printf 'Export exit code: %s\\n' "$EXPORT_CODE"
if [ "$CODE" -ne 0 ]; then exit "$CODE"; fi
exit "$EXPORT_CODE"
'''


def guide(pin, digest):
    return f'''# V37 delivered PNG guard diagnostic: five manual steps

V36 is audited but fails preservation. V37 measures a changed PNG-valued
preservation path before another finite displacement or training design.
The same100 photographic TRAIN cases, original DGP state, five-case context
and23 tensors remain. No native, development or reserved-final cases are used.

**300 coarse gradient queries; zero optimizer/parameter updates, epochs or new
checkpoints.** The PNG forward is exact; its identity backward surrogate is
approximate. Custom backward machinery is used inside `torch.autograd.grad`;
zero `.backward()` API calls does not mean zero VM gradient computations.
All17 finite PNG preservation groups, source/brightness requirements and
1% at50/10% at800 training gates remain. A gradient receipt is not a quality pass.

Require the existing idle **NVIDIA L4/g2-standard-4** and **6GiB free**.
Estimated diagnostic **3–10 minutes**, export **1–3 minutes**; V37 is not yet
timed on the VM. Worker900s; external930s plus30s kill grace; export300s,
external330s plus30s grace; allocated VRAM20GiB; uncompressed return1.5GiB.
Metric mismatch, changed original PNG/identity context, nonfinite gradients,
dependencies, GPU competition or count/time/disk/VRAM limits stop and retain
evidence. No cleanup, optimizer or historical worker is launched.

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
test -f ~/forensic-dgp/cctv_dgp_finite_clearance_probe_v36_vm/outputs/results.json &&
test ! -e ~/forensic-dgp/{NAME} &&
tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_png_guard_v37
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/{NAME}.py --root . --protocol-sha {pin} --verify-transfer &&
bash scripts/run_v37_guard.sh {pin}
```

Detach with **Ctrl+B**, release, then **D**. Step3 reconnects. Successful
diagnostic ends with gradient_queries:300 and optimizer_updates:0.
`complete:true` in the export reports packaging. Download stopped evidence too;
do not edit the frozen packet, resume it or launch follow-on training.

5. Download from **Windows Google Cloud SDK Shell** after export completes:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-export.json" "."
```

Each SCP command has one remote source for Windows PuTTY. The prospective local
audit verifies hashes, all300 saved arrays, 102 group/metric coarse-gradient
assemblies, all100 actual PNG values, original-state parity and20 frozen CPU
replay outputs. It uses no local gradients or optimizer. Missing historical
dependencies must be restored from verified files; do not rerun old recipes.
The DGP-led app, checkpoints, splits and failed qualification records remain.
Native useful structure and separate automatic/assisted covering-family quality
still require review. No final evaluation or app promotion follows from V37.

[V36 findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FINITE_CLEARANCE_PROBE_V36_RESULTS.md>)
[PNG-path evidence and primary research](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V36_DELIVERED_METRIC_REVIEW.md>)
'''


def main():
    assert not BUNDLE.exists() and not PREP.exists()
    audit = ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_independent_audit.json'
    analysis = ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_analysis_v1/analysis.json'
    review = ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_analysis_v1/visual_review.json'
    forward = ROOT / 'outputs/cctv_dgp_v36_delivered_metric_path_v1/analysis.json'
    forward_audit = ROOT / 'outputs/cctv_dgp_v36_delivered_metric_path_v1/independent_forward_audit.json'
    assert read(audit)['complete'] and read(audit)['finite_probe_complete']
    assert not read(analysis)['jointly_eligible_subset_variants']
    assert read(review)['actually_viewed_cases'] == 100 and not read(review)['app_adoption']
    assert read(forward)['all500_canonical_PNG_tensors_exact'] and read(forward_audit)['complete']
    assert read(forward_audit)['PNG_only_failures'] == 9
    old = ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_vm'; basis = read(old / 'protocol.json')
    prior = ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_vm'; prior_p = read(prior / 'protocol.json')
    returned = ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_return'
    BUNDLE.mkdir(); (BUNDLE / 'scripts').mkdir(); PREP.mkdir()
    for name in [NAME + '.py', 'cctv_dgp_delivered_png_guard_v37.py']:
        text = (ROOT / 'scripts' / name).read_text(encoding='utf-8')
        ast.parse(text, feature_version=(3, 10)); text_file(BUNDLE / 'scripts' / name, text)
    text_file(BUNDLE / 'scripts/run_v37_guard.sh', SHELL)
    binding_names = {'protocol.json', 'outputs/results.json'} | set(prior_p['assets_sha256'])
    for cohort in basis['cohorts']:
        folder = f'outputs/state0_{cohort["name"]}/before/'
        binding_names.add(folder + 'receipt.json')
        for case in cohort['cases']:
            binding_names.update(folder + case['id'] + suffix for suffix in ['.npy', '.png', '_target_embedding.npy', '_embedding.npy'])
    dependencies = {name: sha(returned / name) for name in sorted(binding_names)}
    local_paths = [audit, analysis, review, forward, forward_audit,
        ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_analysis_v1/independent_analysis_page_audit.json',
        ROOT / 'CCTV_DGP_FINITE_CLEARANCE_PROBE_V36_RESULTS.md', ROOT / 'CCTV_DGP_V36_DELIVERED_METRIC_REVIEW.md',
        ROOT / 'scripts/audit_cctv_dgp_finite_clearance_probe_v36_return.py',
        ROOT / 'scripts/audit_cctv_dgp_delivered_guard_grad_v37_return.py',
        ROOT / 'scripts/analyze_cctv_dgp_v36_delivered_metric_path_v1.py',
        ROOT / 'scripts/verify_cctv_dgp_v36_delivered_metric_path_v1.py']
    local = {path.relative_to(ROOT).as_posix(): sha(path) for path in local_paths}
    local.update(read(ROOT / 'outputs/completion_input_footprints_comparison_v1_r1/protocol.json')['app_preservation_sha256'])
    p = {'format': 'own-DGP-delivered-PNG-coarse-preservation-gradient-v37', 'frozen_UTC': datetime.now(timezone.utc).isoformat(),
         'hypothesis': 'Measure declared coarse gradients at actual PNG-valued MSE/SSIM/fixed-grid identity forwards before another finite proposal.',
         'purpose': 'Diagnose delivered preservation path; no adopted loss or optimizer change and no finite preservation guarantee.',
         'states': [0], 'cohorts': basis['cohorts'], 'cases': 100,
         'guard_metrics': ['PNG_MSE', 'one_minus_PNG_SSIM', 'one_minus_PNG_ArcFace'],
         'gradient_queries': 300, 'optimizer_updates': 0, 'parameter_updates': 0, 'backwards': 0, 'epochs': 0,
         'backwards_field_means_backward_API_calls': True, 'custom_coarse_backward_inside_autograd_grad': True,
         'true_PNG_derivative_claimed': False, 'surrogate': 'identity derivative on observed support; zero outside',
         'selected_tensors': 23, 'selected_parameters': 978243, 'parameter_layout': basis['parameter_layout'],
         'original_batch_size': 5, 'delivered_identity_batch_size': 5, 'all17_group_derivatives_from_saved_cases': True,
         'original_checkpoint_sha256': basis['original_checkpoint_sha256'], 'original_DGP_state': basis['original_DGP_state'],
         'recognizer_state': basis['recognizer_state'], 'basis_protocol_sha256': sha(old / 'protocol.json'),
         'basis_VM_sha256': {n: sha(old / n) for n in ['protocol.json', *basis['assets_sha256']]},
         'V36_original_readback_sha256': dependencies, 'local_basis_sha256': local,
         'same_VM_raw_tolerance': 1e-5, 'same_VM_PNG_byte_tolerance': 0, 'same_VM_embedding_tolerance': 1e-6,
         'forward_metric_tolerances': [1e-12, 1e-7, 1e-6], 'CPU_replay_case_ids': prior_p['CPU_replay_case_ids'],
         'forward_call_limits': {'reference_DGP_forwards': 20, 'candidate_DGP_forwards': 20, 'recognizer_forwards': 40},
         'retained_capacity_gates': prior_p['retained_capacity_gates'],
         'budgets': {'worker_seconds': 900, 'external_seconds': 930, 'kill_grace_seconds': 30, 'export_seconds': 300,
                    'external_export_seconds': 330, 'minimum_free_disk_bytes': 6 * 1024 ** 3, 'peak_vram_bytes': 20 * 1024 ** 3,
                    'export_uncompressed_bytes': 3 * 1024 ** 3 // 2, 'return_files_maximum': 500, 'local_audit_seconds': 1200},
         'assets_sha256': {path.relative_to(BUNDLE).as_posix(): sha(path) for path in sorted((BUNDLE / 'scripts').iterdir())},
         'TRAIN_only': True, 'source_labels_not_ethnicity': True, 'unexposed_cohort_is_now_design_data': True,
         'VM_execution_started': False, 'new_checkpoint_created': False, 'native_or_reserved_used': False,
         'app_promotion': False, 'goal_complete': False}
    write(BUNDLE / 'protocol.json', p); pin = sha(BUNDLE / 'protocol.json')
    archive = ROOT / 'outputs' / (STEM + '-execution.tar.gz')
    with tarfile.open(archive, 'x:gz', compresslevel=6) as tar:
        for path in sorted(q for q in BUNDLE.rglob('*') if q.is_file()):
            tar.add(path, arcname=NAME + '/' + path.relative_to(BUNDLE).as_posix(), recursive=False)
    digest = sha(archive); text_file(Path(str(archive) + '.sha256'), digest + '  ' + archive.name + '\n')
    text_file(ROOT / 'CCTV_DGP_DELIVERED_GUARD_GRAD_V37_VM.md', guide(pin, digest))
    write(PREP / 'preparation_receipt.json', {'complete': True, 'protocol_sha256': pin, 'archive_sha256': digest,
        'archive_bytes': archive.stat().st_size, 'packet_files': 4, 'worker_Python310_syntax': True,
        'independent_packet_audit_pending': True, 'coarse_gradient_queries_prepared': 300,
        'actual_gradient_queries': 0, 'actual_optimizer_updates': 0, 'VM_launches': 0, 'goal_complete': False})
    print(json.dumps({'complete': True, 'protocol_sha256': pin, 'archive_sha256': digest, 'bytes': archive.stat().st_size, 'VM_launches': 0}))


if __name__ == '__main__': main()
