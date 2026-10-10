"""Freeze a finite zero-update V34 diagnostic after the audited V33 failure."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
NAME = 'cctv_dgp_group_guard_grad_v34_vm'
STEM = 'cctv-dgp-group-guard-grad-v34'
BUNDLE = ROOT / 'outputs' / NAME
PREP = ROOT / 'outputs/cctv_dgp_group_guard_grad_v34_preparation'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def text_file(path, text):
    with path.open('x', encoding='utf-8', newline='\n') as stream: stream.write(text)


def write(path, value): text_file(path, json.dumps(value, indent=2, allow_nan=False) + '\n')


SHELL = '''#!/usr/bin/env bash
set -uo pipefail
PIN="${1:?Pass the exact V34 protocol SHA256}"
ROOT="$HOME/forensic-dgp/cctv_dgp_group_guard_grad_v34_vm"
PYTHON="$HOME/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python"
WORKER="$ROOT/scripts/cctv_dgp_group_guard_grad_v34_vm.py"
cd "$ROOT" || exit 1
test ! -e outputs && test ! -e diagnostic.log && test ! -e supervisor_receipt.json && test ! -e export_manifest.json || exit 1
"$PYTHON" -B "$WORKER" --root "$ROOT" --protocol-sha "$PIN" --verify-transfer || exit 1
START=$("$PYTHON" -c 'import time; print(time.monotonic())')
timeout --signal=TERM --kill-after=30s 630s "$PYTHON" -B -u "$WORKER" --root "$ROOT" --protocol-sha "$PIN" --run 2>&1 | tee diagnostic.log
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
    return f'''# V34 preservation-gradient diagnostic: five manual steps

V33 passed its independent return audit but failed preservation qualification.
This diagnostic measures the protected functions before their zero-at-baseline
hinges. It covers the same two 50-case photographic TRAIN subsets, original
DGP state only, same batch context and 23 selected tensors. Source labels do
not establish ethnicity, native CCTV performance or Zamboanga performance.

**300 gradient queries; zero optimizer updates, parameter updates or new
checkpoints.** No V32 continuation, V33 repeat or historical worker is launched.
All 17 group checks, original loss weights, 1% at50 and10% at800 requirements,
source/brightness limits and the current app remain unchanged. Measured raw
derivatives cannot substitute for finite PNG preservation or useful structure.

Use the existing, running **NVIDIA L4/g2-standard-4** with an idle GPU and
**6GiB free**. Estimated diagnostic **1–6 minutes**, export **1–3 minutes**,
based on prior runs; V34 has not been timed. Worker600s; external630s plus30s
kill grace; export300s/external330s plus30s grace; allocated VRAM20GiB;
uncompressed return1.5GiB. Nonfinite gradients, altered dependencies, different
context, competing GPU work, count, time, VRAM or disk violations stop and
retain evidence. No files are deleted. This packet must run manually.

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
test -f ~/forensic-dgp/cctv_dgp_loss_cone_probe_v33_vm/outputs/results.json &&
test ! -e ~/forensic-dgp/{NAME} &&
tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_guard_v34
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/{NAME}.py --root . --protocol-sha {pin} --verify-transfer &&
bash scripts/run_v34_guard.sh {pin}
```

Detach with **Ctrl+B**, release, then **D**. Step3 reconnects. Expected completed
diagnostic: gradient_queries:300, optimizer_updates:0, parameter_updates:0.
Export complete:true means packaging. Retain and download failures too; do
not change the frozen packet or run follow-on training.

5. Download from **Windows Google Cloud SDK Shell** after export completes:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-export.json" "."
```

Each SCP command has one remote source for Windows PuTTY. An independent local
audit will verify hashes, all300 saved case gradients, all17 raw group derivative
assemblies, original output parity and frozen CPU inference. No local autograd
or optimizer is allowed. An audited derivative alone cannot approve a training
recipe, final evaluation, app promotion or completion of the thesis goal.
'''


def main():
    assert not BUNDLE.exists() and not PREP.exists()
    audit_path = ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_independent_audit_r2.json'
    analysis_path = ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_analysis_v1/analysis.json'
    review_path = ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_analysis_v1/visual_review.json'
    audit, analysis, review = map(read, [audit_path, analysis_path, review_path])
    assert audit['complete'] and audit['finite_probe_complete'] and audit['CPU_replay']['outputs'] == 280
    assert review['all4_pages_actually_viewed_at_original_detail'] and not review['app_adoption']
    assert all(r['preservation_against_original']['failures'] for r in analysis['variants'] if r['state'] == 0)
    assert all(not r['preservation_against_original']['all17_preservation_groups_pass'] for r in analysis['variants'] if r['state'] == 50 and r['cohort'] == 'unexposed')
    old = ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_vm'; basis = read(old / 'protocol.json')
    prior = ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_vm'; prior_p = read(prior / 'protocol.json')
    returned = ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_return'
    BUNDLE.mkdir(); (BUNDLE / 'scripts').mkdir(); PREP.mkdir()
    target = 'scripts/' + NAME + '.py'; source = ROOT / 'scripts' / (NAME + '.py')
    text = source.read_text(encoding='utf-8'); ast.parse(text, feature_version=(3,10)); text_file(BUNDLE / target, text)
    text_file(BUNDLE / 'scripts/run_v34_guard.sh', SHELL)
    binding_names = {'protocol.json', 'outputs/results.json'} | set(prior_p['assets_sha256'])
    for cohort in basis['cohorts']:
        folder = f'outputs/state0_{cohort["name"]}/before/'
        binding_names.add(folder + 'receipt.json')
        for case in cohort['cases']:
            binding_names.update(folder + case['id'] + suffix for suffix in ['.npy','.png','_target_embedding.npy','_raw_embedding.npy'])
    dependencies = {n:sha(returned/n) for n in sorted(binding_names)}
    local_names = [audit_path, analysis_path, review_path, ROOT/'CCTV_DGP_LOSS_CONE_PROBE_V33_RESULTS.md',
        ROOT/'scripts/audit_cctv_dgp_group_guard_grad_v34_return.py',
        ROOT/'scripts/audit_cctv_dgp_loss_cone_probe_v33_return_r2.py']
    completion = read(ROOT/'outputs/completion_input_footprints_comparison_v1_r1/protocol.json')
    local = {q.relative_to(ROOT).as_posix():sha(q) for q in local_names}
    local.update(completion['app_preservation_sha256'])
    p = {'format':'own-DGP-original-nonhinged-group-preservation-gradient-v34','frozen_UTC':datetime.now(timezone.utc).isoformat(),
        'hypothesis':'Non-hinged source/profile raw derivatives expose preservation conflicts absent from zero original-state hinges and aggregate averages.',
        'purpose':'Measure per-case preservation functions before choosing another finite update probe; no training loss change.',
        'states':[0],'cohorts':basis['cohorts'],'cases':100,'guard_metrics':['raw_MSE','one_minus_raw_SSIM','one_minus_raw_ArcFace'],
        'gradient_queries':300,'optimizer_updates':0,'parameter_updates':0,'backwards':0,'epochs':0,
        'selected_tensors':23,'selected_parameters':978243,'parameter_layout':basis['parameter_layout'],
        'original_batch_size':5,'baseline_and_prediction_identity_same_call':True,'all17_group_derivatives_from_saved_cases':True,
        'original_checkpoint_sha256':basis['original_checkpoint_sha256'],'original_DGP_state':basis['original_DGP_state'],
        'recognizer_state':basis['recognizer_state'],'basis_protocol_sha256':sha(old/'protocol.json'),
        'basis_VM_sha256':{n:sha(old/n) for n in ['protocol.json',*basis['assets_sha256']]},
        'V33_original_readback_sha256':dependencies,'local_basis_sha256':local,
        'same_VM_raw_tolerance':1e-5,'CPU_guard_value_tolerance':1e-4,
        'forward_call_limits':{'reference_DGP_forwards':20,'candidate_DGP_forwards':20,'recognizer_forwards':40},
        'retained_capacity_gates':prior_p['retained_capacity_gates'],
        'budgets':{'worker_seconds':600,'external_seconds':630,'kill_grace_seconds':30,'export_seconds':300,
            'external_export_seconds':330,'minimum_free_disk_bytes':6*1024**3,'peak_vram_bytes':20*1024**3,
            'export_uncompressed_bytes':3*1024**3//2,'return_files_maximum':500,'local_audit_seconds':1200},
        'assets_sha256':{target:sha(BUNDLE/target),'scripts/run_v34_guard.sh':sha(BUNDLE/'scripts/run_v34_guard.sh')},
        'TRAIN_only':True,'source_labels_not_ethnicity':True,'VM_execution_started':False,'new_checkpoint_created':False,
        'native_or_reserved_used':False,'app_promotion':False,'goal_complete':False}
    write(BUNDLE/'protocol.json',p);pin=sha(BUNDLE/'protocol.json')
    archive=ROOT/'outputs'/(STEM+'-execution.tar.gz')
    with tarfile.open(archive,'x:gz',compresslevel=6) as tar:
        for path in sorted(q for q in BUNDLE.rglob('*') if q.is_file()): tar.add(path,arcname=NAME+'/'+path.relative_to(BUNDLE).as_posix(),recursive=False)
    digest=sha(archive);text_file(Path(str(archive)+'.sha256'),digest+'  '+archive.name+'\n')
    text_file(ROOT/'CCTV_DGP_GROUP_GUARD_GRAD_V34_VM.md',guide(pin,digest))
    write(PREP/'preparation_receipt.json',{'complete':True,'protocol_sha256':pin,'archive_sha256':digest,'archive_bytes':archive.stat().st_size,
        'packet_files':3,'worker_Python310_syntax':True,'independent_packet_audit_pending':True,'gradient_queries_prepared':300,
        'actual_gradient_queries':0,'actual_optimizer_updates':0,'VM_launches':0,'goal_complete':False})
    print(json.dumps({'complete':True,'protocol_sha256':pin,'archive_sha256':digest,'bytes':archive.stat().st_size,'VM_launches':0}))


if __name__ == '__main__': main()
