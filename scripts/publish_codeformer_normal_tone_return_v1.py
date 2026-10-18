"""Publish the audited stopped diagnostic, preserving exact historical bytes."""
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_normal_tone_v1 import contract as c


def main():
    review = ROOT / 'outputs/codeformer_normal_tone_v1_return_review'
    imported = c.read_json(review / 'return_import_v1.json')
    pixels = c.read_json(review / 'pixels_v1/audit.json')
    transport = c.read_json(review / 'independent_transport_v1.json')
    cause = c.read_json(review / 'source_root_cause_v1.json')
    c.require(imported['complete'] is True and pixels['complete'] is True and
              transport['complete'] is True and cause['complete'] is True,
              'Complete independent return evidence required')
    publication = review / 'publication_v1'
    c.require(not publication.exists(), 'Preserve previous return publication')
    publication.mkdir()
    visual = {'complete': True, 'reviewer': 'root', 'personal_visual_review': True,
        'UTC': datetime.now(timezone.utc).isoformat(), 'sheets_seen_at_original_resolution': [
            {'path': str((review / 'pixels_v1/visual_sheets' / name).relative_to(ROOT)).replace('\\', '/'),
             'sha256': c.sha256(review / 'pixels_v1/visual_sheets' / name)}
            for name in ('normal_sources.png', 'dark_sources.png')],
        'rows': 16, 'exact256_cells': 48,
        'observation': 'Prior original0 and fresh source columns are visually identical across both sheets, consistent with exact saved-array parity. These are untrained source aliases, not new candidate outputs.',
        'retained_artifacts': 'Original CodeFormer can invent or alter details on severe observations; u0006_s2 has glasses-like structure already in the original source output. Clear controls are sharper than inputs but these views do not establish identity recovery.',
        'candidate_exists': False, 'treatment_visual_result': False,
        'DEV_FINAL_or_native_qualification': False, 'model_qualified': False}
    c.write_new_json(review / 'root_visual_review_v1.json', visual)
    report = '''# CodeFormer normal-tone V1 returned: exact-gradient control stop

17 October 2026. **ARCHIVE VERIFIED / DIAGNOSTIC FAILED / ZERO TRAINING UPDATES.**

All four returned companions and199 regular archive members/91,344,209 payload
bytes passed hash-bound import and independent gzip/TAR/member readback. Archive
SHA-256 `f5f8700fd5b02be9221c908689af5507d83c8b6d00204295799adb1ac5d01f92`;
protocol `6a3e162f6383cefac9d14fa11e08032d0149a7583d6ff60ba2ff893056335d22`.
All20 static source files match the sealed packet. The external final companion
binds exact export-log bytes and the parsed finished receipt; its recorded
serialization reproduces the finished hash. No returned code/checkpoint executed.

## Authoritative stop

Location: `original0/fresh_all_cached_coarse_parity.json`, worker run_snapshot.
Fresh all-coarse failed the frozen bit-identity requirement before every branch.
All72 physical fusion-vector hashes differ;278 None routes match. Source-recorded
maximum absolute difference1.4901161193847656e-08 and L2 difference
1.3180001063936446e-07 are not independent vector replay: fresh aggregate vectors
were not exported before this stop and old per-source vectors are unavailable.
The accepted error was zero and the result remains failed. Small magnitude alone
does not prove harmlessness, a rounding-only cause or a successful intervention.

Actual work:16 source forwards,16 ordinary-coarse and8 centered queries, zero
probes/repeats, zero Adam attempts/completions, zero backward calls/committed
updates and no model checkpoint. Neither stopped50 nor any treatment was reached.
Worker54.344s; supervisor preflight15.887s, run72.016s and export38.496s; internal
export22.168s. No timeout/log cap fired. Measured worker RSS3,875,602,432B,
CUDA reserved6,931,087,360B/allocated5,674,107,904B and minimum sampled root free
45,179,736,064B stayed inside declared limits. They are partial-run measurements.

## What the outputs establish

All16 fresh source native/RAW/logit/feature/token arrays and PNG pixels exactly
match their earlier original0 source outputs. All16 native-to-RAW and floor-PNG
checks match. All16 camera/input/clean/target receipts and eight independently
read direct metric targets pass. Root personally viewed both planned exact256
source-only sheets:16 rows/48 cells. There is no trained-candidate improvement
to judge in these files; no treatment output was produced.

The ordinary coarse domain/pool32/L1/weight4/divisor16/canonical FP32 order is
source-equivalent. Sixteen CPU pooled-loss/analytic-derivative checks agree,
without running a neural network. Old and new query/allocation contexts differ;
deterministic algorithms and cuDNN deterministic were false. Execution-context
nondeterminism is plausible, but the exact kernel/cause remains unidentified.
[PyTorch2.9 reproducibility documentation](https://docs.pytorch.org/docs/2.9/notes/randomness.html)
explains why disabled benchmarking does not ensure deterministic convolution.
Preserve this stop; no tolerance is retrospectively changed.

## Direct next step toward a trained model

The latest user request prioritizes creating a real improved model. The previous
eight-step-return-first sequence is explicitly superseded prospectively by the
[new finite training design](CCTV_CODEFORMER_NORMAL_TONE_TRAINING_V1_DESIGN.md).
It is a50-update hypothesis pilot from untouched CodeFormer, changing normal-only
coarse luminance sensitivity and retaining dark coarse, all other losses/data/
rates/anchors/route and full61-clear/53-blur guards. All gradients are computed
fresh per training exposure. Cached-gradient mixing and historical bit-identity
are not part of training; this does not relabel the failed diagnostic as passed.

The new pilot must save genuine initial/update50/stopped checkpoints before
quality assessment, preserving all states even on a quality stop. It offers
no automatic resume or unchanged historical replay. Its proposed loss remains
unproven; the evidence does not establish a solved brightness cause. Only the
user launches actual training via a separately sealed manual L4/tmux packet.
Preparation is incomplete until source/data/CPU/independent review and guide
exist. No training packet was launched by this returned audit.

Original CodeFormer remains the credited initializer/benchmark, original DGP
checkpoints and every failure stay intact. Native CCTV remains unpaired and
source-separated; paired synthetic metrics/untouched FINAL stay separate.
No ethnicity, recovered hidden identity or Zamboanga performance claim follows.
The app, input-only insufficiency requirement, Auto/override/PNG and independent
final/user acceptance remain outstanding. Separate covering detection/completion
development stays deferred until restoration acceptance. No cleanup/offload was
repeated. The overall goal remains incomplete.

Evidence: [import](outputs/codeformer_normal_tone_v1_return_review/return_import_v1.json),
[transport](outputs/codeformer_normal_tone_v1_return_review/independent_transport_v1.md),
[pixels](outputs/codeformer_normal_tone_v1_return_review/pixels_v1/audit.json),
[cause boundaries](outputs/codeformer_normal_tone_v1_return_review/source_root_cause_v1.md),
[root visual review](outputs/codeformer_normal_tone_v1_return_review/root_visual_review_v1.json).
'''
    path = ROOT / 'CCTV_CODEFORMER_NORMAL_TONE_V1_RETURN_REVIEW.md'
    with path.open('xb') as stream:
        stream.write(report.encode('utf-8'))
    prefix = '''# Normal-tone diagnostic stopped; direct training pilot design — 17 October 2026

**CURRENT: RETURN AUDITED / EXACT-GRADIENT CONTROL FAILED / ZERO MODEL UPDATES.**
[Return review](CCTV_CODEFORMER_NORMAL_TONE_V1_RETURN_REVIEW.md).
[New checkpoint-producing training design](CCTV_CODEFORMER_NORMAL_TONE_TRAINING_V1_DESIGN.md).

All199 returned members/91,344,209 payload bytes and four companions verified;
all20 static files exact. Fresh original0 coarse gradients failed strict cached
bit-identity after16 source forwards and24 queries; zero Adam steps/probes/
committed updates or new checkpoint. All16 source image/token/logit/feature
outputs equal the original controls; root reviewed both source-only sheets.
Small recorded gradient differences do not establish harmlessness or exact
physical cause. The failed control and original zero tolerance stay preserved.

The latest user request prioritizes an actual model. The optional eight-step
sequence is prospectively superseded by a NEW50-update training hypothesis:
normal-only coarse luminance centering, fresh full-objective gradients, original
initializer, unchanged dark/other losses/data/optimizer/route and full61-clear/
53-blur gates. Save genuine candidate/stopped states before gate assessment;
never promote from checkpoint/export completion. No cached-gradient training,
failed recipe resume or historical replay. The loss remains an unproven choice.

Only the user's finite manual L4/tmux workflow may launch training after a new
sealed and reviewed packet exists. No new training has run here. Model/app/user/
independent-final qualification, input-only insufficiency threshold and later
separate seven-family completion scope remain required. Native unpaired and
paired synthetic evidence/untouched FINAL stay separate; no identity/ethnicity/
Zamboanga performance claim. Existing app design and original assets/failures
stay intact. Storage offload is complete and was not repeated. Goal incomplete.

Earlier command guides below are historical; do not relaunch the failed test.

---

'''
    names = ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
             'PROJECT_HANDOFF.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md',
             'CCTV_CODEFORMER_NORMAL_TONE_V1_VM.md')
    changed = {}
    encoded = prefix.encode('utf-8')
    for name in names:
        path = ROOT / name
        before = path.read_bytes()
        backup = publication / (name + '.before')
        with backup.open('xb') as stream:
            stream.write(before)
        path.write_bytes(encoded + before)
        c.require(path.read_bytes()[len(encoded):] == before, 'Historical suffix differs')
        changed[name] = {'before_sha256': c.sha256(backup), 'after_sha256': c.sha256(path),
                         'preserved_suffix_offset_bytes': len(encoded)}
    evidence = ['return_import_v1.json', 'independent_transport_v1.json', 'pixels_v1/audit.json',
                'source_root_cause_v1.json', 'root_visual_review_v1.json']
    c.write_new_json(publication / 'receipt.json', {'complete': True,
        'UTC': datetime.now(timezone.utc).isoformat(), 'status': 'DIAGNOSTIC_CONTROL_FAILED',
        'documents': changed, 'report_sha256': c.sha256(ROOT / 'CCTV_CODEFORMER_NORMAL_TONE_V1_RETURN_REVIEW.md'),
        'evidence_sha256': {name: c.sha256(review / name) for name in evidence},
        'publisher_sha256': c.sha256(__file__), 'local_neural_gradient_optimizer_VM_calls': 0,
        'model_created': False, 'model_qualified': False})
    print({'complete': True, 'return_status': 'CONTROL_FAILED_BEFORE_TRAINING',
           'source_sheets_reviewed': 2, 'documents_updated': len(changed), 'model_created': False})


if __name__ == '__main__':
    main()
