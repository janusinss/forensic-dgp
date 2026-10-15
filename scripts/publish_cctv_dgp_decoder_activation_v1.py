"""Publish verified preparation and functional fixes without claiming training."""
from datetime import datetime, timezone
from pathlib import Path

from cctv_dgp_decoder_activation_v1_contract import NAME, STEM, CHECKPOINT, read, write, sha
from publish_cctv_dgp_png_group_gradients_v1_r2 import finish_prepend, retained_new

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs'
PREP = OUT/'cctv_dgp_decoder_activation_v1_preparation'


def link(label, path):
    return f'[{label}](<{(ROOT/path).as_posix()}>)'


def main():
    receipt = PREP/'publication/publication.json'
    assert not receipt.exists()
    prepared = read(PREP/'preparation.json')
    audit = read(PREP/'independent_preparation_audit.json')
    imported = read(PREP/'local_importer_preparation_r1.json')
    bash = read(PREP/'bash_syntax.json')
    ui_path = OUT/'dgp_native_mask_roundtrip_v1_analysis/browser_binding_audit.json'
    native_path = OUT/'cctv_dgp_hinge_native_exclusion_v1_analysis/r1/independent_readback.json'
    ui,native = read(ui_path),read(native_path)
    assert all(v['complete'] for v in [prepared,audit,imported,bash,ui,native])
    for name, digest in imported['source_sha256'].items():
        assert sha(ROOT/name) == digest
    assert sha(PREP/'local_importer_preparation_r1.txt') == imported['log_sha256']
    pin, archive_sha = prepared['protocol_sha256'],prepared['archive_sha256']
    assert sha(OUT/NAME/'protocol.json') == pin
    assert sha(OUT/(STEM+'-execution.tar.gz')) == archive_sha
    assert sha(ROOT/'static/face_workflow.js') == ui['current_JS_sha256']
    for name, digest in ui['unchanged_four_sources'].items(): assert sha(ROOT/name) == digest
    assert sha(OUT/'cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth') == CHECKPOINT
    previous = read(OUT/'cctv_dgp_decoder_hinge_repair_v1_analysis/publication/publication.json')
    names = ['PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md',
             'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md','CCTV_DGP_NEXT_PILOT.md','CCTV_DGP_POST_HINGE_REPAIR_REVIEW_V1.md']
    expected = {name: previous['documents'][name]['after_sha256'] if name in previous['documents']
                else previous['new_document_sha256'][name] for name in names}
    assert all(sha(ROOT/name) == digest for name,digest in expected.items()), 'Preserve concurrent status edits'
    guide_name = 'CCTV_DGP_DECODER_ACTIVATION_V1_VM.md'
    preparation_name = 'CCTV_DGP_DECODER_ACTIVATION_V1_PREPARATION.md'
    mask_name = 'CCTV_DGP_NATIVE_MASK_EDITOR_V1_RESULTS.md'
    native_name = 'CCTV_DGP_HINGE_NATIVE_EXCLUSION_V1_RESULTS.md'
    guide = f'''# Decoder activation V1: five manual steps

**VERIFIED / PREPARED / UNRUN. This measures the failed direction; it performs
zero optimizer updates, zero new amplitudes and no direction search.** It uses
the existing L4 VM and retained research directories. Do not resume the failed
hinge-repair optimizer. The permanent app checkpoint stays unchanged.

Require at least **3 GiB free**, **8 GiB available RAM**, the existing venv and
an idle L4. Estimated total runtime is **5–15 minutes**, subject to measured
projection stops. Worker cap:1200s; saved audit/export:300s each. External
caps:1230/330/330s, with30s kill grace. VRAM20GiB, RSS12GiB, output768MiB and
disk reserve1GiB are enforced; model loading is included in peak accounting.

1. Upload from **Windows Google Cloud SDK Shell (cmd.exe)**:

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
tmux new-session -A -s dgp_decoder_activation_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_decoder_activation_v1_vm.py --root . --protocol-sha {pin} --verify-transfer &&
bash scripts/run_activation.sh {pin}
```

5. Download from **Windows Google Cloud SDK Shell** after export:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-export.json" "."
```

Each SCP command has one remote source, as required by Windows PuTTY. `Ctrl+B`,
then `D` detaches tmux safely. Export completion means evidence was packaged;
it cannot qualify the model. Keep failed/partial files if any stop occurs.

Expected successful measurement:400 exact retained RAW/PNG controls,80 fresh
batch-five objective records,70 gradient queries, zero optimizer/backward calls
and exact initial state/RNG restoration. Quality remains unqualified even if
all these measurements succeed. A killed partial run may have an unknown exact
query count; the export explicitly preserves that uncertainty.

Execution archive:{prepared['archive_bytes']:,}bytes. SHA256:`{archive_sha}`.
Protocol SHA256:`{pin}`.

{link('Independent preparation and scope', preparation_name)}.
After download, the agent uses the source-pinned local importer:
{link('Local import and saved-data audit', 'scripts/import_cctv_dgp_decoder_activation_v1_return.py')}.
No VM execution or VM connection was performed by the agent.
'''
    prep_doc = f'''# Decoder activation V1: independently verified preparation

Status on15October2026: **VERIFIED / PREPARED / UNRUN**, not a trained candidate.
The preceding hinge-repair candidate remains rejected at0.113913%RAW and
0.128470%PNG structure-error reduction with30 preservation failures in each
path. No checkpoint has been selected or promoted.

This diagnostic checks a specific remaining assumption: whether the chosen
gradient of the corrected current A decoder explains finite objective changes
on the same five-image recognizer path. The earlier saved diagnostic used
15-image batches. Fresh five-image measurements keep that evidence separate.
The full objective retains its original weights, anchors and regression slack;
the per-image brightness guard retains coefficient10/fraction0.2. The two
derivative references are the existing first two scheduled TRAIN blocks,
`tr_asian_00176` and `tr_ffhq_53945`, chosen before new measurements.

The worker replays only initial,1/16,1/4 andA1 vectors already retained from
the rejected run. All400 fresh RAW/floor-PNG images must equal those controls.
It records80 exact batch-five objective decompositions. On the two references
only,70 full parameter gradients measure underlying RGB,SSIM,ArcFace,landmark
error,channel-mean and complete objective slopes along the retained A1 change.
Actual gradient-pass embeddings are retained to avoid assuming grad/no-grad
bit parity. No optimizer, backward call, new amplitude, new direction search,
parameter proposal, new epoch or automatic follow-up exists in this worker.

The one-sided calculation is an affine residual model using selected neural
branches and idealized real-valued clipping. Actual0/1 mean-shift clipping and
finite regression slack are explicit. It is not a certified derivative of the
literal float32 or PNG program, nor proof that every decoder direction fails.
Only two references have measured derivatives; the other18 have finite replay
records. Eight finite-versus-linear comparisons retain these scope limits.

**44 CPU tests pass**, including120 retained scalar rows, strict roles/fields,
actual Windows pre-neural guard, malformed/SIGKILL exports, count and gradient
corruption checks. A separate local-importer suite tests hashes and safe archive
names before extraction. Git Bash syntax and LF-only shell/sidecar checks pass.
Independent packet review verifies17 bound assets,18 archive files and1,577
ancestor bindings. Numerical helper bounds are frozen prospectively:objective
arithmetic1e-10, norm metadata2e-6relative/1e-12absolute, clipping-factor metadata
twofloat32ULPs. Actual clipped bytes remain exact, first-gradient replay remains
maxabs2e-7/cosine≥0.999999, and all scientific quality gates remain unchanged.

The finite limits are in the protocol and manual guide. No GPU time was consumed
by this preparation. Syntax/CPU success does not verify CUDA execution. Windows
fixture ACL failures, source revisions and all preceding stops are retained.

RouteA reuses the corrected DeblurGAN-v2-compatible conditional reconstruction
path. Frozen external GAN features are cached for source parity and are not
injected into this A decoder. This diagnostic is not a newly trained GAN or
qualified generative face prior. Native CCTV, reserved final identities and
covering pixels are not used. The permanent baseline stays
`{CHECKPOINT}`.

{link('Independent preparation receipt', 'outputs/cctv_dgp_decoder_activation_v1_preparation/independent_preparation_audit.json')}.
{link('CPU test evidence', 'outputs/cctv_dgp_decoder_activation_v1_preparation/cpu_tests.json')}.
{link('Five exact manual steps', guide_name)}.
{link('Selected causal review', 'CCTV_DGP_POST_HINGE_REPAIR_REVIEW_V1.md')}.
All five goal milestones, useful native DEV/final review and seven separately
reported automatic/assisted covering families remain required and unqualified.
'''
    mask_doc = f'''# Native removal-mask fidelity repair: functional verification

Location:`static/face_workflow.js`. Cause:an untouched native mask was reduced
to a≤512px editor and enlarged for export. Chromium changed768pixels on the
1024×768 fixture,720 on1280×720 and2,616 on the1024² thin-feature fixture.
The256²/512² controls were exact. This could change removal areas and correctly
make the server classify an untouched automatic proposal as assisted.

The repair keeps native mask pixels through import,review,editing and export.
The displayed editor dimensions/design stay unchanged. Pointer/keyboard strokes
map display coordinates into native support; only the edited support is
normalized. Hover reads the preview raster. Undo retains at most12frames and
96MiB, evicting before allocation; reset retains the automatic proposal.

All five Chromium exports now have zero changed mask pixels.21 named interaction
checks plus zero page/console errors cover paint/erase,exactundo/reset/clear,
delayed/invalid/superseded imports, review gating and submitted native mask bytes.
The4096×3906 fixture(15,998,976pixels) independently exports all40,001 selected
pixels exactly. The maximum-size undo ceiling and reset passed. A harness
self-comparison was discarded and replaced with a distinct-source targeted
browser check; two coordinate/focus harness failures are retained separately.

The11 backend tests include1024×768/1280×720 native automatic→one-pixel-assisted
→exact-reset transitions, even when the delivered256mask/output stay unchanged.
They use mocked components. Browser endpoints were intercepted synthetic
fixtures; no trained detector,completion or restorer forward was executed.
The local server was started for these checks; loading its existing components
does not constitute training or output-quality evidence.

Root inspected375/768/1280 responsive screenshots:three result cards, editor
and controls render with no horizontal overflow/page/console errors. CSS,
template and both backend source hashes remain unchanged. SourceJS is
`{ui['current_JS_sha256']}`; the exact original source and tests are retained.

{link('Independent 79-file/pixel binding audit', 'outputs/dgp_native_mask_roundtrip_v1_analysis/browser_binding_audit.json')}.
{link('Root inline browser artifacts', 'scratch/dgp-native-mask-v1/after-layout.json')}.
{link('Backend regression evidence', 'outputs/dgp_native_mask_roundtrip_v1_analysis/backend_tests_after.json')}.
This verifies processing fidelity only. Automatic detector quality, useful
completion, all seven covering families and independent final review remain
unqualified. It does not show restoration or hidden-identity recovery.
'''
    native_doc = '''# Native exclusion audit for the current exposed TRAIN diagnostic

The previous encoded-file exclusion omitted the original QMUL development
inputs. The new metadata/hash-only audit closes that specific gap, including
both retained QMUL24 prepared-input versions, extension inputs and ChokePoint
DEV originals/prepared inputs. The frozen roles and review criteria stay fixed.

Independent readback rehashed 965 file bindings. Separately, the audit recounts
767 transitive source/parent/replay/packet bindings, 208 native face/frame-hash
entries and 300 TRAIN input/target/support records for 100 cases/20 references.
These overlapping categories are not an additive breakdown of 965. Exact
encoded-face and ancestral hash matches are zero. Entries include retained
variants and frame metadata; they are not unique identities or unique images.

Opaque native files total158:QMULDEV72including two prepared versions,reserved
QMUL32,extension6 andChokePointDEV48. ChokePoint50frame-hash metadata entries
include26reserved entries. No reserved frame archive member or final image was
decoded. No image was used to select thresholds or training directions.

Encoded inequality does not prove person-level or pretrained-data disjointness.
Cross-source/pretraining overlap remains unknown. Source names do not establish
ethnicity or Zamboanga performance. No local Zamboanga CCTV evidence exists.
Native CCTV remains unpaired; this audit supplies no PSNR/SSIM or model quality.
The initial missing-versus-null metadata-reader failure is retained; R1 permits
only the frozen absent-or-null fields, without changing data or roles.
'''+f'''
{link('Audit and per-source entries', 'outputs/cctv_dgp_hinge_native_exclusion_v1_analysis/r1/audit.json')}.
{link('Independent readback', 'outputs/cctv_dgp_hinge_native_exclusion_v1_analysis/r1/independent_readback.json')}.
{link('Limits and preserved failure', 'outputs/cctv_dgp_hinge_native_exclusion_v1_analysis/README.md')}.
'''
    texts = {guide_name:guide,preparation_name:prep_doc,mask_name:mask_doc,native_name:native_doc}
    for name,text in texts.items(): retained_new(ROOT/name,text)
    prefix = f'''# Next manual diagnostic verified; native mask fidelity fixed — 15 October 2026

**The model remains unqualified and the original checkpoint is unchanged.**
The failed hinge-repair run is not resumed. A new finite activation diagnostic
is **VERIFIED / PREPARED / UNRUN**:70 gradient measurements,400 exact retained
RAW/PNG controls,zero optimizer updates/new directions/new amplitudes. It checks
the batch-five objective and finite penalty activation before a new direction
search.44 CPU tests, independent transfer review and strict local-import tests
pass. Actual gradients remain human-manual on the existing L4 through tmux.

{link('Five upload/install/tmux/run/download steps', guide_name)}.
{link('Scope, finite stops and independent evidence', preparation_name)}.

The local native-mask editor now preserves untouched removal pixels above512px;
five Chromium fixtures, editing/race/undo/reset checks, the maximum-size export,
11 mocked backend tests and375/768/1280 layout checks pass. This is functional
mask fidelity, not covering/model quality. Native encoded exclusion now covers
the omitted QMUL variants; zero exact-byte matches do not prove identity overlap
absence. No final pixels or native tuning were used.

{link('Mask-editor processing result', mask_name)}.
{link('Native exclusion result and limits', native_name)}.
All five milestones, useful DEV/final output review and seven automatic/assisted
covering families remain active. Checkpoints,caches,splits,failedstates andprior
instruction bytes are retained. Exact earlier status follows.

'''
    stage = PREP/'publication'
    (stage/'before').mkdir(parents=True,exist_ok=True)
    documents = {name: finish_prepend(ROOT/name,prefix,stage/'before'/name,digest) for name,digest in expected.items()}
    evidence = [PREP/'preparation.json',PREP/'independent_preparation_audit.json',PREP/'local_importer_preparation_r1.json',
                PREP/'cpu_tests.json',PREP/'bash_syntax.json',ui_path,native_path]
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        previous_goal_turn_classification='progress: independently completed repair return audit and causal review',
        current_goal_turn_classification='progress: fixed verified native-mask flow, closed encoded exclusion gap, and prepared a distinct finite manual diagnostic',
        documents=documents,new_document_sha256={name:sha(ROOT/name) for name in texts},
        evidence_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in evidence},
        protocol_sha256=pin,archive_sha256=archive_sha,archive_bytes=prepared['archive_bytes'],
        baseline_sha256=CHECKPOINT,status='VERIFIED_PREPARED_UNRUN; MASK_FIDELITY_FIXED; MODEL_UNQUALIFIED',
        new_local_optimizer_updates=0,new_local_gradient_queries=0,agent_VM_calls=0,
        model_qualified=False,goal_complete=False))
    print(dict(complete=True,new_documents=4,preserved_documents=6,status='PREPARED_UNRUN',goal_complete=False))


if __name__ == '__main__':
    main()
