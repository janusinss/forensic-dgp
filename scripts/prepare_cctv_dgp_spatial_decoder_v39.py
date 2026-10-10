"""Freeze a portable V39 zero-update gradient proof after exact initial parity."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'
PROOF = ROOT / 'outputs/cctv_dgp_spatial_decoder_v39_initial_review_v1'
NAME = 'cctv_dgp_spatial_decoder_vm_v39'
STEM = 'cctv-dgp-spatial-decoder-v39'
BUNDLE = ROOT / 'outputs' / NAME
PREP = ROOT / 'outputs/cctv_dgp_spatial_decoder_v39_preparation'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def text_file(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(value)


def write(path, value):
    text_file(path, json.dumps(value, indent=2, allow_nan=False) + '\n')


SHELL = '''#!/usr/bin/env bash
set -uo pipefail
PIN="${1:?Pass the exact V39 protocol SHA256}"
ROOT="$HOME/forensic-dgp/cctv_dgp_spatial_decoder_vm_v39"
PYTHON="$HOME/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python"
WORKER="$ROOT/scripts/cctv_dgp_spatial_decoder_v39_vm.py"
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
    return f'''# V39 spatial decoder gradient proof: five manual steps

V38 passes two small TRAIN displacement checks, but its fixed quarter step fails
paired development preservation and adds no convincing native clarity. V39 tests
a different spatial path within our DGP: observed256 RGB plus frozen original
multiscale features, a two-scale16/32-channel gated decoder, and subtraction of
its separately frozen initial response. It uses no pretrained-restorer targets
or pretrained NAFNet weights. Initial raw/PNG parity is independently verified
on all50 already-exposed TRAIN cases. All17,952 new parameters start untrained.

**This is70 gradient queries, zero optimizer/parameter updates and no epochs.**
All57 new tensors must receive finite nonzero improvement gradients; all initial
preservation values/gradients must remain exactly zero. Original DGP, fixed initial
decoder and recognizer must remain unchanged. Connectivity is not a quality pass.
The1%-at50/10%-at800 and all preservation gates remain for any later training.

Require the existing idle **NVIDIA L4/g2-standard-4** and **2GiB free**. The packet
contains its own verified model/data sources; only the existing Python venv is
required. No deleted historical pilot or research cache is recreated or rerun.
Estimated diagnostic **2–6 minutes**, export **1–2 minutes**. Worker600s;
external630s plus30s kill grace; export300s/external330s plus30s grace; allocated
VRAM20GiB; uncompressed return192MiB. A failed proof is exported and retained.
No cleanup, automatic follow-on or new actual training is launched by the agent.

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
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test ! -e ~/forensic-dgp/{NAME} &&
tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_spatial_decoder_v39
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_spatial_decoder_v39_vm.py --root . --protocol-sha {pin} --verify-transfer &&
bash scripts/run_v39_gradient.sh {pin}
```

Detach with **Ctrl+B**, release, then **D**. Step3 reconnects. The log prints
`V39_batch`1 through10 and70 gradient queries if completed. An export
`complete:true` means packaging, with optimizer_updates:0. Download failure
evidence too. Do not edit, resume or repeat the frozen packet unchanged.

5. Download from **Windows Google Cloud SDK Shell** after export completes:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-export.json" "."
```

Each SCP has one remote source for Windows PuTTY. The prospective independent
checker verifies every returned file, all70 saved component queries,57 tensor
partitions, baseline PNGs, unchanged states and ten frozen CPU replay cases.
It runs no local gradients or optimizer and never executes returned Python.
A returned proof is required before choosing a new finite actual-training recipe.
Native useful structure, independent final review and separate automatic/assisted
quality for all seven covering families remain required. Goal active/incomplete.

[V38 development rejection](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V38_QUARTER_DEVELOPMENT_RESULTS.md>)
[V39 spatial design review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_V38_SPATIAL_DECODER_REVIEW.md>)
'''


def main():
    assert not BUNDLE.exists() and not PREP.exists(), 'Preserve every prior/partial preparation'
    proof = read(PROOF / 'results.json'); checked = read(PROOF / 'independent_initial_audit.json'); initial = read(PROOF / 'plan.json')
    assert proof['complete'] and checked['complete'] and proof['exact_initial_raw_and_PNG_parity']
    assert proof['decoder_parameters'] == 17952 and proof['decoder_tensors'] == 57
    assert proof['gradient_calls'] == proof['optimizer_updates'] == 0
    assert not proof['gradient_connectivity_established'] and checked['all50_raw_pairs_exact']
    for name, digest in initial['sources_sha256'].items(): assert sha(ROOT / name) == digest, name
    parent = read(PARENT / 'protocol.json'); assert parent['cases'] == initial['cases']
    v38 = read(ROOT / 'outputs/cctv_dgp_v38_return_development_milestone_v1/independent_closure_audit.json')
    assert v38['complete'] and v38['paired_ArcFace_failure_retained']
    BUNDLE.mkdir(); (BUNDLE / 'scripts').mkdir(); PREP.mkdir()
    mapping = {}
    def copy(source, destination):
        destination = BUNDLE / destination; destination.parent.mkdir(parents=True, exist_ok=True)
        assert not destination.exists()
        shutil.copyfile(source, destination)
        assert sha(source) == sha(destination)
        mapping[destination.relative_to(BUNDLE).as_posix()] = source.relative_to(ROOT).as_posix()
    for name in ['dgp_face_restoration.py', 'dgp_frozen_inference_v2.py', 'cctv_dgp_frozen_norm.py',
                 'dgp_mean_centered_inference_v29.py', 'cctv_dgp_pilot.py']:
        copy(ROOT / name, name)
    for path in sorted((ROOT / 'models').glob('*.py')):
        copy(path, path.relative_to(ROOT).as_posix())
    for name in ['cctv_dgp_degraded_objective_v24.py', 'cctv_dgp_batchmatched_identity_v26.py']:
        copy(PARENT / name, name)
    copy(ROOT / 'scripts/cctv_dgp_spatial_decoder_v39.py', 'cctv_dgp_spatial_decoder_v39.py')
    copy(ROOT / 'scripts/cctv_dgp_spatial_decoder_v39_vm.py', 'scripts/cctv_dgp_spatial_decoder_v39_vm.py')
    copy(PROOF / 'untrained_initial_decoder.pth', 'untrained_initial_decoder.pth')
    for name in ['weights/dgp_v2.pth', 'weights/w600k_r50.onnx']:
        assert sha(PARENT / name) == parent['assets_sha256'][name]; copy(PARENT / name, name)
    paths = {c[k] for c in parent['cases'] for k in ['input', 'target', 'observed', 'raw_dgp']}
    for name in sorted(paths):
        assert sha(PARENT / name) == parent['assets_sha256'][name]; copy(PARENT / name, name)
    # Preserve function ASTs exactly while omitting all historical trainer bodies.
    old = ast.parse((PARENT / 'scripts/cctv_dgp_feature_skips_v27_vm.py').read_text(encoding='utf-8'))
    guard = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == 'require_vm')
    run = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    functions = [next(n for n in run.body if isinstance(n, ast.FunctionDef) and n.name == name) for name in ['mean', 'feature_errors', 'ssim']]
    head = next(n for n in ast.parse((PARENT / 'cctv_dgp_feature_skips_v27.py').read_text(encoding='utf-8')).body if isinstance(n, ast.ClassDef) and n.name == 'SpatialFeatureHead')
    filters = [next(n for n in head.body if isinstance(n, ast.FunctionDef) and n.name == name) for name in ['blur', 'high']]
    nodes = [guard, *filters, *functions]
    source = ast.unparse(ast.Module(body=nodes, type_ignores=[])) + '\n'
    assert ast.dump(ast.parse(source), include_attributes=False) == ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)
    text_file(BUNDLE / 'frozen_definitions.py', source); text_file(BUNDLE / 'scripts/run_v39_gradient.sh', SHELL)
    for path in BUNDLE.rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 10))
    names = ['CCTV_DGP_POST_V38_SPATIAL_DECODER_REVIEW.md', 'CCTV_DGP_V38_QUARTER_DEVELOPMENT_RESULTS.md',
             'outputs/cctv_dgp_v38_return_development_milestone_v1/milestone.json',
             'outputs/cctv_dgp_v38_return_development_milestone_v1/independent_closure_audit.json',
             'outputs/cctv_dgp_spatial_decoder_v39_initial_review_v1/plan.json',
             'outputs/cctv_dgp_spatial_decoder_v39_initial_review_v1/results.json',
             'outputs/cctv_dgp_spatial_decoder_v39_initial_review_v1/independent_initial_audit.json',
             'scripts/prepare_cctv_dgp_spatial_decoder_v39.py', 'scripts/verify_cctv_dgp_spatial_decoder_v39_packet.py',
             'scripts/audit_cctv_dgp_spatial_decoder_v39_return.py']
    local = {name: sha(ROOT / name) for name in names}
    for name in ['scripts/cctv_dgp_feature_skips_v27_vm.py', 'cctv_dgp_feature_skips_v27.py', 'protocol.json']:
        local[(PARENT / name).relative_to(ROOT).as_posix()] = sha(PARENT / name)
    for source in mapping.values(): local[source] = sha(ROOT / source)
    assets = {path.relative_to(BUNDLE).as_posix(): sha(path) for path in BUNDLE.rglob('*') if path.is_file()}
    expected = dict(proof['states_before_after'])
    expected['recognizer'] = read(ROOT / 'outputs/cctv_dgp_feature_fusion_vm_v32_r2/protocol.json')['frozen_recognizer_state']
    p = {'format': 'own-DGP-two-scale-spatial-decoder-gradient-proof-v39', 'UTC': datetime.now(timezone.utc).isoformat(),
         'hypothesis': initial['hypothesis'], 'architecture': initial['architecture'], 'seed': 390039, 'width': 16,
         'cases': parent['cases'], 'references': parent['references'], 'parameter_layout': initial['parameter_layout'],
         'decoder_parameters': 17952, 'decoder_tensors': 57, 'expected_states': expected,
         'original_checkpoint_sha256': assets['weights/dgp_v2.pth'], 'recognizer_weights_sha256': assets['weights/w600k_r50.onnx'],
         'assets_sha256': assets, 'local_basis_sha256': local, 'copied_source_mapping': mapping,
         'terms': ['degraded_landmark_detail', 'degraded_observed_detail', 'degraded_pixel', 'clear_baseline_anchor',
                   'pixel_regression', 'SSIM_regression', 'ArcFace_regression'],
         'initial_preservation_terms_exact_zero': ['clear_baseline_anchor', 'pixel_regression', 'SSIM_regression', 'ArcFace_regression'],
         'retained_capacity_gates': parent['prospective_gates'], 'new_training_loss_adopted': False,
         'component_gradient_calls': 70, 'optimizer_updates': 0, 'parameter_updates': 0, 'backwards': 0, 'epochs': 0,
         'forward_call_limits': {'original_DGP_forwards': 20, 'decoder_forwards': 10, 'reference_decoder_forwards': 10, 'recognizer_forwards': 20},
         'budgets': {'worker_seconds': 600, 'external_seconds': 630, 'kill_grace_seconds': 30,
                     'export_seconds': 300, 'export_external_seconds': 330,
                     'peak_vram_bytes': 20 * 1024**3, 'minimum_free_disk_bytes': 2 * 1024**3,
                     'export_uncompressed_bytes': 192 * 1024**2},
         'historical_raw_tolerance': 1e-5, 'CPU_raw_replay_tolerance': 1e-5, 'CPU_PNG_byte_tolerance': 1,
         'CPU_scalar_replay_tolerance': 5e-5, 'CPU_embedding_absolute_tolerance': 1e-4,
         'prospective_CPU_replay_selection': 'First reference in each source, allfive profiles; ten cases, both sources.',
         'original_buffers_encoder_and_reconstruction_weights_frozen': True,
         'fixed_initial_decoder_is_separate_frozen_copy': True, 'initial_seed_asset_is_untrained': True,
         'pretrained_restorer_targets_or_weights_used': False, 'not_NAFNet_architecture_or_benchmark_claim': True,
         'photographic_exposed_TRAIN_only': True, 'native_or_reserved_used': False,
         'actual_training_recipe_not_yet_selected': True, 'automatic_follow_on': False,
         'app_promotion': False, 'training_capacity_pass': False, 'goal_complete': False}
    write(BUNDLE / 'protocol.json', p); pin = sha(BUNDLE / 'protocol.json')
    archive = ROOT / 'outputs' / (STEM + '-execution.tar.gz'); assert not archive.exists()
    with tarfile.open(archive, 'w:gz') as tar:
        for path in sorted(BUNDLE.rglob('*')):
            if path.is_file(): tar.add(path, arcname=NAME + '/' + path.relative_to(BUNDLE).as_posix(), recursive=False)
    digest = sha(archive); text_file(Path(str(archive) + '.sha256'), digest + '  ' + archive.name + '\n')
    text_file(ROOT / 'CCTV_DGP_SPATIAL_DECODER_V39_VM.md', guide(pin, digest))
    write(PREP / 'prepared.json', {'complete': True, 'protocol_sha256': pin, 'archive_sha256': digest,
          'archive_bytes': archive.stat().st_size, 'files': len(assets) + 1, 'assets_sha256': assets,
          'VM_calls': 0, 'neural_calls': 0, 'gradient_calls': 0, 'optimizer_updates': 0,
          'manual_gradient_proof_prepared_only': True, 'independent_packet_audit_pending': True,
          'app_promotion': False, 'goal_complete': False})
    print(json.dumps({'prepared': True, 'files': len(assets) + 1, 'bytes': archive.stat().st_size,
                      'protocol_sha256': pin, 'archive_sha256': digest}), flush=True)


if __name__ == '__main__':
    main()
