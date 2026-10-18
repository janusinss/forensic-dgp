"""Publish the root's actual 24-sheet review; no framework or returned code execution."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'outputs/codeformer_fixed_state_response_v1_return_review'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload)


def main():
    visual = REVIEW / 'pixels_v1/visual_manifest.json'
    document = json.loads(visual.read_text(encoding='utf-8'))
    expected = ([f'fitted_normal_{i:02d}.png' for i in range(1, 10)] +
                [f'fitted_dark_{i:02d}.png' for i in range(1, 10)] +
                [f'heldout_normal_{i:02d}.png' for i in range(1, 4)] +
                [f'heldout_dark_{i:02d}.png' for i in range(1, 4)])
    rows = document['sheets']
    assert len(rows) == 24 and [Path(r['path']).name for r in rows] == expected
    assert sum(row['case_count'] for row in rows) == 264
    for row in rows:
        path = REVIEW / 'pixels_v1' / row['path']
        assert path.stat().st_size == row['bytes'] and digest(path) == row['sha256']
    pins = {name: digest(REVIEW / name) for name in [
        'return_import_v1.json', 'independent_transport_v1.json',
        'root_metadata_readback_v1.json', 'pixels_v1/audit.json',
        'pixels_v1/visual_manifest.json', 'pixels_v1/sheet_cells.json',
        'state_v1/audit.json', 'state_v1/mechanism_review_v1.json']}
    receipt = {'complete': True, 'date': '2026-10-17',
        'reviewer': 'root; personally viewed all 24 saved comparison sheets with view_image',
        'planned_sheets': 24, 'reviewed_sheets': 24, 'unique_inputs': 264,
        'panels_per_input': 5, 'evidence_panels': 1320,
        'sheets': rows, 'evidence_sha256': pins,
        'display_note': 'The tool resized tall 1280px-wide sheets to fit its display. Saved sheet panels remain exact256; independent array/PNG verification covers their original bytes.',
        'findings': [
            'Final head visibly darkens normal and dark photographs; background and face shift together.',
            'Initial-head and full-coarse columns match. Final changes are tonal; existing softened eye, mouth, hair and skin detail is not repaired.',
            'Full-coarse outputs often suppress original-CodeFormer artifacts but also soften detail; neither appearance nor hidden identity is established from these teacher-behavior comparisons.',
            'Darkening is present on fitted as well as unused TRAIN references. No clean target, independent FINAL image or native CCTV case is shown on these sheets.'],
        'new_training_updates': 0, 'model_qualified': False,
        'app_promotion': False, 'full_goal_complete': False}
    write_new(REVIEW / 'root_visual_review_v1.json',
              (json.dumps(receipt, indent=2, allow_nan=False) + '\n').encode())
    report = '''# Fixed-state response return review — 17 October 2026

**Measurement completed and independently audited; no training updates or new checkpoint. The tone-controller model remains unqualified.**

The archive is 516,081,377 bytes, SHA-256
`e60258356e148ae426dcc8863794ad62377559f858a2b5dbf75c3e0eee0d50ad`.
Its 2,536 regular files, four companions, physical archive boundaries, 153 frozen
files and declared dependencies passed readback. Root independently hashed the
whole archive. The worker recorded 656 neural forwards, no gradients, backwards,
optimizers, updates or new checkpoints. Frozen initial/final state witnesses
match. Worker time was 82.068 seconds; resource and supervised phase limits held.

## Actual fixed-final results

These are recorded FP32 behavioral mean-luminance errors, not PSNR/SSIM against
clean targets. The independent CPU64 array reductions agree within the disclosed
6.667e-8 mean-readout difference; the original readings remain retained.

| TRAIN cohort | Inputs | Initial mean MSE | Final mean MSE | Inputs with worse MSE |
|---|---:|---:|---:|---:|
| Fitted normal | 100 | 0.001052076389 | 0.000338663753 | 26 |
| Fitted dark | 100 | approximately 0 | 0.000509992052 | 100 |
| Unused-reference normal | 32 | 0.001427812125 | 0.000547091189 | 8 |
| Unused-reference dark | 32 | 0 | 0.000495880117 | 32 |

**All 264 final offsets are negative.** The mean dark shift is -0.0221823 on
fitted inputs and -0.0218936 on unused references. All 100 fitted dark cases
already worsen: this is not solely a failure to transfer to new photographs.
Even the role-balanced fitted MSE improves by 19.34% while every dark case worsens.
A lower combined loss does not establish selective restoration improvement.

## Images and diagnosis

The independent audit checked all 528 RAW/PNG responses, 128 new parent RAW
arrays, 400 reused fitted parent arrays, source/input/features/target joins,
FP32 correction arithmetic, floor-to-PNG delivery and all 264 zero-head controls.
Root personally viewed all 24 planned sheets: 264 inputs and 1,320 panels.
[All comparison sheets](outputs/codeformer_fixed_state_response_v1_return_review/pixels_v1/visual_sheets/INDEX.md).
The displayed final changes are brightness shifts across faces and backgrounds;
softened facial details remain. The sheets contain no aligned clean target and
support no clean-reference quality, independent FINAL or native CCTV claim.
All support masks are full; covering/protected-pixel checks are vacuous here.

The scalar head cannot directly change facial structure. Source review also
shows the earlier restorer anchored untouched-model behavior only on B-normal,
while other clean-label losses continued on normal inputs. Centering the normal
brightness loss and then adding a scalar head both failed in retained returns.
These observations motivate a new objective design; they do not isolate a unique
cause or prove that the next recipe will succeed.

## Next actual-model work

Prepare a new finite direct-CodeFormer training pilot from the retained
full-coarse research parent. Preserve both A-normal and B-normal with full
spatial RGB and token behavior targets from untouched CodeFormer. Keep full
dark clean-reference supervision and explicit frozen-parent retention. Train
restoration weights with fresh Adam; omit the failed tone head. This is a new
prospective objective, not a continuation of its optimizer or a gate relaxation.

The proposed first pilot is capped at 50 committed restoration updates and
200 microbatches, with retained checkpoints and complete 406-case DEV RAW/PNG
comparisons. It is **being prepared, not sealed or launched** at this publication.
Its exact commands will be published only after source, transfer and finite
resource review. Actual training remains the user's manual L4/tmux action.

Preserve original CodeFormer, original DGP, caches, splits, provenance/terms,
failure archives and genuine failed trained models. App promotion, user-visible
acceptance, independent FINAL review and full app verification remain pending.
Native CCTV is unpaired and source-separated; no ethnicity, Zamboanga performance
or exact hidden-identity claim follows. Separate covering/completion development
stays deferred until restoration acceptance. Storage offload is already complete.

## Evidence

- [Transport](outputs/codeformer_fixed_state_response_v1_return_review/independent_transport_v1.json)
- [Independent saved-pixel audit](outputs/codeformer_fixed_state_response_v1_return_review/pixels_v1/audit.json)
- [Independent state/chronology audit](outputs/codeformer_fixed_state_response_v1_return_review/state_v1/audit.json)
- [Root visual review](outputs/codeformer_fixed_state_response_v1_return_review/root_visual_review_v1.json)
- [Source-based mechanism review](outputs/codeformer_fixed_state_response_v1_return_review/state_v1/mechanism_review_v1.md)
'''
    report_path = ROOT / 'CCTV_CODEFORMER_FIXED_STATE_RESPONSE_V1_RETURN_REVIEW.md'
    write_new(report_path, report.encode('utf-8'))
    prefix = '''# Fixed-state return audited; direct restoration training being prepared — 17 October 2026

**CURRENT: 656-FORWARD MEASUREMENT COMPLETE / ZERO NEW UPDATES / NO QUALIFIED MODEL.**
[Returned evidence and all 24 reviewed sheets](CCTV_CODEFORMER_FIXED_STATE_RESPONSE_V1_RETURN_REVIEW.md).

All 2,536 return files and 528 saved RAW/PNG responses passed integrity and
independent pixel readback. Root reviewed all 24 sheets. The frozen final head
darkens all 264 TRAIN inputs; all 100 fitted dark and 32 unused-reference dark
cases worsen. This disconfirms a generalization-only explanation. The existing
trained composite remains research-only; no new checkpoint was made by this run.

The next preparation targets CodeFormer's actual restoration weights, with a
coherent full-spatial/token preservation objective on BOTH normal exposures,
full dark supervision and frozen-parent retention. Proposed finite first pilot:
50 restoration updates / 200 microbatches with fresh Adam, checkpoints and full
406-case DEV RAW/PNG review. **Not sealed or launched yet.** No more scalar-head
training, unchanged historical recipe, gate relaxation or app promotion is planned.

The full goal and manual L4/tmux boundary remain active. Untouched CodeFormer,
original DGP, trained failures, caches/splits/provenance/terms remain preserved;
native CCTV stays unpaired/source-separated, reserved FINAL separate. User
acceptance, independent FINAL and app Auto/override/PNG/Playwright verification
remain necessary. Separate covering/completion stays deferred. Storage offload
is already complete. Earlier status and commands below are historical only.

---

'''.encode('utf-8')
    targets = ['SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md', 'PROJECT_HANDOFF.md',
        'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md', 'CCTV_CODEFORMER_FIXED_STATE_RESPONSE_V1_VM.md',
        'CCTV_CODEFORMER_FIXED_STATE_RESPONSE_V1_PREPARATION.md']
    backups = REVIEW / 'publication_v1'
    publication = []
    for name in targets:
        path = ROOT / name
        before = path.read_bytes()
        write_new(backups / name, before)
        path.write_bytes(prefix + before)
        assert path.read_bytes()[len(prefix):] == before
        publication.append({'path': name, 'before_sha256': hashlib.sha256(before).hexdigest(),
            'prefix_bytes': len(prefix), 'published_sha256': digest(path),
            'previous_bytes_preserved_as_exact_suffix': True})
    result = {'complete': True, 'root_visual_sha256': digest(REVIEW / 'root_visual_review_v1.json'),
        'report_sha256': digest(report_path), 'evidence_sha256': pins, 'publications': publication,
        'script_sha256': digest(Path(__file__)), 'qualified': False, 'goal_complete': False}
    write_new(backups / 'publication.json', (json.dumps(result, indent=2) + '\n').encode())
    print(json.dumps({'complete': True, 'reviewed_sheets': 24, 'new_updates': 0,
                      'published_documents': len(targets), 'model_qualified': False}))


if __name__ == '__main__':
    main()
