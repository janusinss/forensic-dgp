"""Freeze a self-contained manual diagnostic. No neural calls or VM connection."""
import ast
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import tarfile
import time
from cctv_dgp_original_feature_probe_v1_contract import (
    NAME, STEM, CHECKPOINT, STATE, TERMS, SCOPES, FRACTIONS, BUDGETS, sha, read, write, validate)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/cctv_dgp_residual_epochs_vm_v42'
OUT = ROOT/'outputs'/NAME
PREP = ROOT/'outputs/cctv_dgp_original_feature_probe_v1_preparation'

SHELL = '''#!/usr/bin/env bash
set -uo pipefail
PIN="${1:?Pass the exact original-feature diagnostic protocol SHA256}"
ROOT="$HOME/forensic-dgp/cctv_dgp_original_feature_probe_v1_vm"
PYTHON="$HOME/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python"
WORKER="$ROOT/scripts/cctv_dgp_original_feature_probe_v1_vm.py"
test -n "${TMUX:-}" || { printf 'Launch inside tmux.\\n'; exit 1; }
cd "$ROOT" || exit 1
test ! -e outputs && test ! -e diagnostic.log && test ! -e supervisor_receipt.json && test ! -e export_manifest.json || exit 1
"$PYTHON" -B "$WORKER" --root "$ROOT" --protocol-sha "$PIN" --verify-transfer || exit 1
START=$("$PYTHON" -c 'import time; print(time.monotonic())')
timeout --signal=TERM --kill-after=30s 1230s "$PYTHON" -B -u "$WORKER" --root "$ROOT" --protocol-sha "$PIN" --run 2>&1 | tee diagnostic.log
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


def text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f: f.write(value)


def guide(pin, digest, size):
    return f'''# Original DGP feature-path diagnostic: manual L4 commands

Prepared diagnostic, not launched. The current app DGP remains selected.
V42's failed stop is retained; it is not resumed. This packet measures the
original feature path and reconstruction convolutions on a separate unchanged
current-checkpoint copy. It compares decoder-only, feature-only and joint
structure-gradient displacements. It trains no added decoder and substitutes
no pretrained restorer.

**Zero optimizer updates and zero epochs.** There are30 gradient queries on
50 photographic TRAIN cases, followed by nine finite trial directions on those
50 and50 different photographic TRAIN cases. All1000 raw images, delivered
PNGs, mean-only PNGs and embedding vectors are retained. Each trial starts from
the original parameters and restores them immediately afterward. These trial
parameters are diagnostic observations, not a selected trained checkpoint or
an exact-resume training state. Neither TRAIN cohort is a final evaluation.

The nine trials use three scopes and relative L2 displacements0.00001,0.0001
and0.001. This scale grid is a predeclared sensitivity test, not an AdamW
learning-rate recommendation. The same17 preservation groups,1% structure,
both-source nonregression and20% brightness limit are reported separately for
raw/PNG and each cohort. Passing this small diagnostic cannot qualify a model;
failing trials remain in the archive. No native, DEV or reserved final case,
completion generation or app change enters this run.

Expected diagnostic runtime **5–15 minutes**, export **1–5 minutes**; these are
preparation estimates. Require **4 GiB free after installation**. Actual bounds:
cache120s, gradients180s, trials600s, worker1200s/external1230s with30s kill grace,
export300s/external330s with30s grace,20GiB allocated VRAM,1.75GiB uncompressed
return and512MiB disk reserve. A measured baseline projects trial time and
output-plus-export storage with factor1.25 before any gradient query.
The packet is self-contained for100 inputs,20 targets/masks, model code,
retained DGP and fixed recognizer; it reuses the existing VM Python venv.
No file is deleted. Keep the archived failures and migration cache backup.

Execution archive: **{size:,} bytes**.
Protocol SHA256: `{pin}`
Archive SHA256: `{digest}`

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
tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp &&
df -h ~/forensic-dgp
```

3. Open tmux:

```bash
tmux new-session -A -s dgp_original_feature_probe_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_original_feature_probe_v1_vm.py --root . --protocol-sha {pin} --verify-transfer &&
bash scripts/run_feature_probe.sh {pin}
```

The transfer check makes zero neural or gradient calls. The manually launched
diagnostic verifies the idle existing L4/g2-standard-4, unchanged normalization,
initial parity and declared tensor ownership. It exports partial evidence after
a stop. Detach with Ctrl+B, release, D. Reattach with
`tmux attach-session -t dgp_original_feature_probe_v1`.

5. Download from **Windows Google Cloud SDK Shell**, one remote source per command:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-export.json" "."
```

Export `complete: true` confirms packaging, not model quality. Return all three
files and the terminal output for independent hashing, gradient/displacement
arithmetic, saved raw/PNG metric checks, CPU inference/embedding replay and
whole-face visual review before a new training design. Source labels are not
ethnicity labels; no Zamboanga or hidden-identity performance is inferred.
All five restoration milestones and seven automatic/assisted covering families
remain required. The full goal stays active/incomplete.
'''


def main():
    start = time.monotonic(); assert not OUT.exists() and not PREP.exists()
    parent = read(BASE/'protocol.json')
    assert sha(BASE/'protocol.json') == 'c19ca1790ae9ebe7f99000a81678c8fc756d70ce2114debb6f81691c741a6f2d'
    inventory_path = ROOT/'outputs/cctv_dgp_post_v42_original_path_inventory_v1/inventory.json'
    inventory = read(inventory_path); layout = []; offset = 0
    for r in inventory['parameter_inventory']:
        if r['name'].startswith(('head4.', 'fpn.features.16.', 'fpn.features.17.', 'fpn.features.18.')): continue
        layout.append({'name': r['name'], 'aliases': r['aliases'], 'shape': r['shape'], 'elements': r['elements'],
            'partition': 'feature_only' if r['role'] == 'encoder_FPN' else 'decoder_control',
            'start': offset, 'end': offset+r['elements']}); offset += r['elements']
    first = parent['preview_case_ids']; first_refs = {c['source_person_or_reference'] for c in parent['cases'] if c['id'] in first}
    second_refs = []
    for source in sorted({r['source'] for r in parent['references']}):
        available = [r['id'] for r in parent['references'] if r['source'] == source and r['id'] not in first_refs]
        available.sort(key=lambda rid: hashlib.sha256(('original-feature-probe-v1:'+rid).encode()).hexdigest())
        second_refs.extend(available[:5])
    second = [c['id'] for rid in second_refs for c in parent['cases'] if c['source_person_or_reference'] == rid]
    assert len(first) == len(second) == 50
    by = {c['id']: c for c in parent['cases']}; cases = [by[cid] for cid in first+second]
    refids = first_refs | set(second_refs); references = [r for r in parent['references'] if r['id'] in refids]
    mapping = {}
    common = ['cctv_dgp_frozen_norm.py', 'cctv_dgp_pilot.py', 'dgp_face_restoration.py',
              'dgp_frozen_inference_v2.py', 'frozen_capacity_contract.py', 'frozen_raw_metrics.py', 'frozen_definitions.py',
              'weights/dgp_v2.pth', 'weights/w600k_r50.onnx']
    common.extend(n for n in parent['assets_sha256'] if n.startswith('models/'))
    common.extend(c['input'] for c in cases)
    common.extend(r[k] for r in references for k in ['target', 'observed'])
    # Native/reduced names can be provenance pointers rather than packaged assets.
    common.extend(r[k] for r in references for k in ['native', 'reduced_target']
                  if k in r and r[k] in parent['assets_sha256'])
    for name in sorted(set(common)):
        assert name in parent['assets_sha256'] and (BASE/name).is_file(), name
        assert sha(BASE/name) == parent['assets_sha256'][name], name
    OUT.mkdir(); PREP.mkdir()
    for name in sorted(set(common)):
        source = BASE/name; assert sha(source) == parent['assets_sha256'][name], name
        destination = OUT/name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination); mapping[name] = source.relative_to(ROOT).as_posix()
    for stem in ['contract', 'candidate', 'vm']:
        name = f'cctv_dgp_original_feature_probe_v1_{stem}.py'
        target = OUT/('scripts' if stem == 'vm' else '')/name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/'scripts'/name, target)
        mapping[target.relative_to(OUT).as_posix()] = 'scripts/'+name
    text(OUT/'scripts/run_feature_probe.sh', SHELL)
    local_names = ['CCTV_DGP_CURRENT_TRAINING_IMPROVEMENT_PLAN.md', 'CCTV_DGP_V42_RESULTS.md',
        'CCTV_DGP_POST_V42_ARCHITECTURE_REVIEW.md', 'CCTV_DGP_POST_V42_ORIGINAL_PATH_INVENTORY.md',
        'outputs/cctv_dgp_post_v42_original_path_inventory_v1/inventory.json',
        'outputs/cctv_dgp_post_v42_original_path_inventory_v1/forward_ownership.json',
        'outputs/cctv_dgp_original_decoder_gradient_v1_r2_return/outputs/gradient_summary.json',
        'outputs/cctv_dgp_v42_return_review_v1/fixed_cohort_correction_loss_review.json',
        'outputs/cctv_dgp_residual_epochs_vm_v42/protocol.json',
        'scripts/prepare_cctv_dgp_original_feature_probe_v1.py',
        'scripts/audit_cctv_dgp_original_feature_probe_v1_return.py',
        'CCTV_DGP_ORIGINAL_FEATURE_PROBE_V1_DESIGN.md']
    assets = {q.relative_to(OUT).as_posix(): sha(q) for q in sorted(OUT.rglob('*')) if q.is_file()}
    p = {'format': 'own-DGP-original-feature-connectivity-and-finite-displacement-v1',
        'UTC': datetime.now(timezone.utc).isoformat(),
        'hypothesis': 'Joint original-feature adaptation may expose a structure direction unavailable to a frozen-base added decoder; mechanism and preservation remain unproven.',
        'architecture_direction': 'Original-feature diagnostic default under existing user best-approach instruction; pending preference has not been answered or treated as approval.',
        'cases': cases, 'references': references,
        'cohorts': [{'name': 'TRAIN_gradient', 'case_ids': first}, {'name': 'TRAIN_cross_cohort', 'case_ids': second}],
        'cross_cohort_selection': 'Five remaining TRAIN references per source by ascending SHA256(original-feature-probe-v1:reference_id), before output inspection; photographic provenance/overlap limitations unchanged',
        'parameter_layout': layout, 'original_checkpoint_sha256': CHECKPOINT, 'original_state': STATE,
        'recognizer_weights_sha256': parent['recognizer_weights_sha256'], 'recognizer_state': parent['initial_states']['recognizer'],
        'terms': TERMS, 'scopes': SCOPES, 'relative_displacement_fractions': FRACTIONS,
        'proposal_rule': 'Subtract normalized mean degraded-landmark gradient in declared scope; L2 displacement = fraction * initial scope weight L2; round once to float32; every trial independently starts at original',
        'forward_policy': 'Unchanged original DGP with observed(candidate-original) RGB mean removed before output clamp; target/profile/source never model inputs',
        'gradient_queries': 30, 'trial_variants': 9, 'optimizer_updates': 0, 'epochs': 0, 'committed_trajectory_updates': 0,
        'budgets': BUDGETS, 'raw_storage': 'All1000 float32 raw arrays, PNGs, mean-only PNGs and raw/PNG/target embeddings, all gradients and trial vectors retained',
        'scientific_thresholds': {'early_structure_gain': .01, 'final_structure_gain': .1,
            'MSE_regression_tolerance': 1e-12, 'SSIM_ArcFace_regression_tolerance': 1e-6, 'brightness_fraction_maximum': .2},
        'CPU_replay_tolerances': {'raw_max_abs': 3e-6, 'PNG_byte_max': 1, 'embedding_max_abs': 5e-5},
        'full_TRAIN_capacity_not_tested': True, 'native_or_DEV_or_final_used': False,
        'model_qualification': False, 'automatic_follow_on': False, 'app_promotion': False, 'goal_complete': False,
        'terms_and_overlap': parent['source_overlap_limits'], 'pretrained_restoration_targets_used': False,
        'manual_tmux_required': True, 'no_failed_recipe_resume': True,
        'copied_source_mapping': mapping, 'assets_sha256': assets,
        'local_sources_sha256': {n: sha(ROOT/n) for n in local_names}}
    validate(p); write(OUT/'protocol.json', p); pin = sha(OUT/'protocol.json')
    archive_path = ROOT/'outputs'/(STEM+'-execution.tar.gz'); assert not archive_path.exists()
    for q in OUT.rglob('*.py'): ast.parse(q.read_text(encoding='utf-8'), feature_version=(3, 10))
    with tarfile.open(archive_path, 'w:gz', compresslevel=1) as tar:
        for q in sorted(OUT.rglob('*')):
            if not q.is_file(): continue
            info = tar.gettarinfo(str(q), arcname=NAME+'/'+q.relative_to(OUT).as_posix())
            info.uid = info.gid = 0; info.uname = info.gname = ''; info.mtime = 0; info.mode = 0o644
            with q.open('rb') as f: tar.addfile(info, f)
    digest = sha(archive_path); text(Path(str(archive_path)+'.sha256'), digest+'  '+archive_path.name+'\n')
    text(ROOT/'CCTV_DGP_ORIGINAL_FEATURE_PROBE_V1_VM.md', guide(pin, digest, archive_path.stat().st_size))
    write(PREP/'prepared.json', {'complete': True, 'protocol_sha256': pin, 'archive_sha256': digest,
        'archive_bytes': archive_path.stat().st_size, 'assets': len(assets), 'cases': 100, 'references': 20,
        'seconds': time.monotonic()-start, 'neural_calls': 0, 'gradient_queries': 0, 'optimizer_updates': 0,
        'VM_connections': 0, 'training_launched': False, 'independent_packet_audit_pending': True})
    print({'complete': True, 'protocol_sha256': pin, 'archive_sha256': digest, 'bytes': archive_path.stat().st_size})


if __name__ == '__main__': main()
