"""Prepare a distinct manual finite spatial-path learning pilot after audited V39."""
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tarfile
import time
from cctv_dgp_spatial_fit_v40_contract import sha, read, write, validate_schedule

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/cctv_dgp_spatial_decoder_vm_v39'
BROAD = ROOT/'outputs/cctv_dgp_profile_batches_vm_v31'
MIXED = ROOT/'outputs/cctv_dgp_mixed_vm_v9_r2'
NAME = 'cctv_dgp_spatial_fit_vm_v40'
STEM = 'cctv-dgp-spatial-fit-v40'
OUT = ROOT/'outputs'/NAME
PREP = ROOT/'outputs/cctv_dgp_spatial_fit_v40_preparation'

SHELL = '''#!/usr/bin/env bash
set -uo pipefail
PIN="${1:?Pass the exact V40 protocol SHA256}"
ROOT="$HOME/forensic-dgp/cctv_dgp_spatial_fit_vm_v40"
PYTHON="$HOME/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python"
WORKER="$ROOT/scripts/cctv_dgp_spatial_fit_v40_vm.py"
cd "$ROOT" || exit 1
test ! -e outputs && test ! -e trainer.log && test ! -e supervisor_receipt.json && test ! -e export_manifest.json || exit 1
"$PYTHON" -B "$WORKER" --root "$ROOT" --protocol-sha "$PIN" --verify-transfer || exit 1
START=$("$PYTHON" -c 'import time; print(time.monotonic())')
timeout --signal=TERM --kill-after=30s 4800s "$PYTHON" -B -u "$WORKER" --root "$ROOT" --protocol-sha "$PIN" --run 2>&1 | tee trainer.log
CODE=${PIPESTATUS[0]}
printf '%s\\n' "$CODE" > trainer_exit_code.txt
ELAPSED=$("$PYTHON" -c 'import sys,time; print(time.monotonic()-float(sys.argv[1]))' "$START")
"$PYTHON" -B "$WORKER" --root "$ROOT" --protocol-sha "$PIN" --record-supervision --elapsed "$ELAPSED" --exit-code "$CODE" || exit 1
printf 'Trainer exit code: %s\\n' "$CODE"
timeout --signal=TERM --kill-after=30s 930s "$PYTHON" -B -u "$WORKER" --root "$ROOT" --protocol-sha "$PIN" --export
EXPORT_CODE=$?
printf 'Export exit code: %s\\n' "$EXPORT_CODE"
if [ "$CODE" -ne 0 ]; then exit "$CODE"; fi
exit "$EXPORT_CODE"
'''


def text(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream: stream.write(value)


def guide(pin, digest, size):
    return f'''# V40 spatial DGP learning pilot: five manual steps

V39's independently audited return passes all57 improvement-gradient partitions
and exact initial preservation, with zero optimizer updates. V40 learns that
same17,952-parameter spatial decoder using the existing781 TRAIN references and
3,905 cases. Original DGP weights, the fixed initial decoder and recognizer stay
frozen. All visible facial features remain in scope. Pretrained restoration is
not a training target or substituted primary model.

**Maximum800 updates: one781-reference epoch plus19 paired batches.** The fixed
V31 metadata schedule pairs one clear control and four degradations. Initial50
normalizers and all seven original losses stay fixed. A fixed AdamW3e-4 rate is
for the NEW initialized decoder, rather than original-weight fine-tuning. There
is no rate sweep, checkpoint selection from DEV or continuation of V38.

The unchanged1%-at50 and10%-at800 structure thresholds, all17 PNG preservation
groups, both-source nonregression and20% brightness-only limit remain. At50,
failure of structure OR preservation stops and exports evidence. At800 a failed
capacity result is exported without adoption. The broader/final/native quality
requirements remain separate. All50 raw previews and all3,905 PNG/mean-only
images/vectors are saved per complete snapshot at0/50/800; raw stages for other
cases are hashed, not retained. This is paired photographic TRAIN evidence.

Require the existing idle **NVIDIA L4/g2-standard-4**, existing venv, and **6GiB
free after installation**. This self-contained{size:,}-byte packet copies verified
inputs, original weights, code and the untrained seed; no old pilot runs.
Estimated training **15–40 minutes**, export **3–15 minutes**; these are estimates,
not measured V40 timing. Cache900s, fit3600s, worker4500s/external4800s+30s grace,
export900s/external930s+30s grace, VRAM20GiB, return3GiB are enforced. Cache and
update20 timing projections use1.25 safety factors. No automatic retry or cleanup.

Protocol SHA256: `{pin}`
Execution archive SHA256: `{digest}`

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
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test ! -e ~/forensic-dgp/{NAME} &&
tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp
```

3. Open tmux:

```bash
tmux new-session -A -s dgp_spatial_fit_v40
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_spatial_fit_v40_vm.py --root . --protocol-sha {pin} --verify-transfer &&
bash scripts/run_v40.sh {pin}
```

Detach with Ctrl+B, release, D. Reattach with
`tmux attach-session -t dgp_spatial_fit_v40`. Keep the trainer log and every stop.
No other GPU task is terminated. A successful export is not a training-quality
pass; return the files for an independent audit and all50 preview review.

5. Download from **Windows Google Cloud SDK Shell**, one remote file per command:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-export.json" "."
```

The app, automatic and assisted completion, independent final review and useful
native output are not qualified by this pilot. No real Zamboanga CCTV exists yet;
source labels do not infer ethnicity. Reserved final crops stay unopened.
'''


def main():
    started = time.monotonic(); assert not OUT.exists() and not PREP.exists()
    q = read(BASE/'protocol.json'); old = read(BROAD/'protocol.json')
    audit = read(ROOT/'outputs/cctv_dgp_spatial_decoder_v39_independent_audit.json')
    assert audit['complete'] and audit['diagnostic_complete'] and audit['protocol_sha256'] == sha(BASE/'protocol.json')
    assert not audit['saved_gradient_readback']['zero_improvement_tensor_names'] and audit['CPU_replay']['all_states_unchanged']
    for name, digest in q['assets_sha256'].items(): assert sha(BASE/name) == digest, name
    for name, digest in q['local_basis_sha256'].items(): assert sha(ROOT/name) == digest, name
    validate_schedule(old['case_rows'], read(BROAD/'schedule.json')['batches'])
    assert old['preview_case_ids'] == [c['id'] for c in q['cases']]
    old_cases = {c['id']: c for c in old['case_rows']}
    for c in q['cases']:
        for key in ['input', 'target', 'observed']:
            assert sha(BASE/c[key]) == sha(MIXED/old_cases[c['id']][key]), (c['id'], key)
    OUT.mkdir(); PREP.mkdir(); mapping = {}
    def copy(source, name):
        destination = OUT/name; assert destination.resolve().is_relative_to(OUT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination); assert sha(source) == sha(destination)
        mapping[name] = source.relative_to(ROOT).as_posix()
    for name in q['assets_sha256']:
        if name.startswith('data/') or name in ['cctv_dgp_spatial_decoder_v39.py', 'scripts/cctv_dgp_spatial_decoder_v39_vm.py', 'scripts/run_v39_gradient.sh']: continue
        copy(BASE/name, name)
    for name in ['scripts/cctv_dgp_spatial_fit_v40_vm.py', 'scripts/cctv_dgp_spatial_fit_v40_contract.py']:
        copy(ROOT/name, name if name.endswith('_vm.py') else Path(name).name)
    source = (ROOT/'scripts/cctv_dgp_spatial_decoder_v39.py').read_text(encoding='utf-8')
    routed = source.replace('cctv_dgp_spatial_decoder_vm_v39', NAME).replace('SpatialDGPCandidateV39', 'SpatialDGPCandidateV40').replace('Distinct V39', 'Distinct V40')
    restored = routed.replace(NAME, 'cctv_dgp_spatial_decoder_vm_v39').replace('SpatialDGPCandidateV40', 'SpatialDGPCandidateV39').replace('Distinct V40', 'Distinct V39')
    assert restored == source
    text(OUT/'cctv_dgp_spatial_decoder_v40.py', routed)
    names = {c[k] for c in old['case_rows'] for k in ['input', 'target', 'observed']}
    for name in sorted(names):
        assert sha(MIXED/name) == old['mixed_TRAIN_assets_sha256'][name], name
        copy(MIXED/name, name)
    copy(BROAD/'schedule.json', 'schedule.json'); text(OUT/'scripts/run_v40.sh', SHELL)
    assets = {path.relative_to(OUT).as_posix(): sha(path) for path in OUT.rglob('*') if path.is_file()}
    for path in OUT.rglob('*.py'): ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 10))
    local_names = ['scripts/prepare_cctv_dgp_spatial_fit_v40.py', 'scripts/cctv_dgp_spatial_fit_v40_vm.py',
                   'scripts/cctv_dgp_spatial_fit_v40_contract.py', 'scripts/verify_cctv_dgp_spatial_fit_v40_packet.py',
                   'scripts/audit_cctv_dgp_spatial_fit_v40_return.py', 'scripts/cctv_dgp_spatial_decoder_v39.py',
                   'outputs/cctv_dgp_spatial_decoder_v39_independent_audit.json', 'outputs/cctv_dgp_spatial_decoder_v39_return_import.json',
                   'outputs/cctv_dgp_spatial_decoder_v39_return/outputs/results.json', 'outputs/cctv_dgp_spatial_decoder_v39_return/outputs/gradient_summary.json',
                   'outputs/cctv_dgp_profile_batches_vm_v31/protocol.json', 'outputs/cctv_dgp_profile_batches_vm_v31/schedule.json',
                   'outputs/cctv_dgp_profile_batches_v31_return/outputs/failure.json', 'CCTV_DGP_V38_QUARTER_DEVELOPMENT_RESULTS.md']
    local = {name: sha(ROOT/name) for name in set(local_names) | set(mapping.values())}
    p = {'format': 'own-DGP-proven-spatial-path-full-TRAIN-v40', 'UTC': datetime.now(timezone.utc).isoformat(),
         'hypothesis': 'Learn the independently connected two-scale spatial path across the full existing TRAIN corpus while retaining visible structure. Connectivity does not imply finite gain or generalization.',
         'V39_audit_sha256': sha(ROOT/'outputs/cctv_dgp_spatial_decoder_v39_independent_audit.json'),
         'architecture_equivalent_to_verified_V39_except_manual_root_and_class_name': True,
         'cases': old['case_rows'], 'references': old['training_references'], 'preview_case_ids': old['preview_case_ids'],
         'reference_order_from_frozen_V31': read(BROAD/'schedule.json')['reference_order'],
         'updates': 800, 'completed_epochs_bound': 1, 'second_epoch_batches_bound': 19, 'snapshots': [0, 50, 800],
         'decoder_parameters': 17952, 'decoder_tensors': 57, 'parameter_layout': q['parameter_layout'],
         'initial_states': q['expected_states'], 'original_checkpoint_sha256': q['original_checkpoint_sha256'],
         'recognizer_weights_sha256': q['recognizer_weights_sha256'], 'terms': q['terms'],
         'normalizers': 'Same exposed50 cases and original filter convention; no full-corpus rescaling',
         'optimizer': {'type': 'AdamW new spatial decoder only', 'learning_rate': .0003, 'weight_decay': .01, 'gradient_clip_norm': 1, 'AMP': False, 'EMA': False},
         'retained_capacity_gates': q['retained_capacity_gates'],
         'early_stop': 'Apply unchanged preservation, source and brightness gates at50 in addition to1% structure. Final800 requires10%. Stop on first failed necessary gate.',
         'budgets': {'cache_seconds': 900, 'fit_seconds': 3600, 'worker_seconds': 4500, 'external_seconds': 4800,
                     'kill_grace_seconds': 30, 'export_seconds': 900, 'export_external_seconds': 930,
                     'peak_vram_bytes': 20*1024**3, 'minimum_free_disk_bytes': 6*1024**3, 'export_uncompressed_bytes': 3*1024**3},
         'timing_safety_factor': 1.25, 'timing_update': 20, 'remaining_snapshot_allowance': '2*measured full snapshot0 seconds*1.25',
         'raw_storage': 'All50 prospective raw previews per snapshot; other3855 raw stages hashed, not retained. All3905 delivered and mean-only PNGs and vectors saved.',
         'prospective_return_audit': {'all3905_PNG_metrics_per_complete_snapshot': True, 'all17_groups': True, 'raw_compositions_per_snapshot': 50,
                                     'frozen_CPU_replay_cases_per_snapshot': 50, 'CPU_raw_tolerance': 1e-5, 'PNG_byte_tolerance': 1, 'embedding_tolerance': 1e-4, 'local_seconds_cap': 1800},
         'sources_sha256': local, 'copied_source_mapping': mapping, 'assets_sha256': assets,
         'original_and_fixed_initial_and_recognizer_frozen': True, 'native_or_reserved_used': False,
         'pretrained_restorer_targets_or_weights_used': False, 'local_optimizer_updates': 0, 'VM_calls_here': 0,
         'manual_transfer_and_launch_required': True, 'automatic_follow_on': False, 'training_run_here': False,
         'app_promotion': False, 'independent_final_review': False, 'goal_complete': False}
    write(OUT/'protocol.json', p); pin = sha(OUT/'protocol.json')
    archive = ROOT/'outputs'/(STEM+'-execution.tar.gz'); assert not archive.exists()
    with tarfile.open(archive, 'w:gz', compresslevel=1) as tar:
        for path in sorted(OUT.rglob('*')):
            if path.is_file(): tar.add(path, arcname=NAME+'/'+path.relative_to(OUT).as_posix(), recursive=False)
    digest = sha(archive); text(Path(str(archive)+'.sha256'), digest+'  '+archive.name+'\n')
    text(ROOT/'CCTV_DGP_SPATIAL_FIT_V40_VM.md', guide(pin, digest, archive.stat().st_size))
    write(PREP/'prepared.json', {'complete': True, 'protocol_sha256': pin, 'archive_sha256': digest, 'archive_bytes': archive.stat().st_size,
          'packet_files': len(assets)+1, 'assets_sha256': assets, 'sources_verified': len(local), 'training_cases': 3905,
          'training_references': 781, 'updates_bound': 800, 'seconds': time.monotonic()-started,
          'local_neural_calls': 0, 'local_gradient_calls': 0, 'local_optimizer_updates': 0, 'VM_calls': 0, 'prepared_only': True})
    print({'prepared': True, 'files': len(assets)+1, 'bytes': archive.stat().st_size, 'protocol_sha256': pin, 'archive_sha256': digest}, flush=True)


if __name__ == '__main__': main()
