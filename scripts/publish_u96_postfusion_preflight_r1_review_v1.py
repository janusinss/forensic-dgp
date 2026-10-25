"""Publish saved-return and root visual review; no numerical/model execution."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/codeformer_u96_postfusion_preflight_v1_r1_return_audit'
REPORT = ROOT / 'CCTV_CODEFORMER_U96_POSTFUSION_PREFLIGHT_V1_R1_REVIEW.md'


def pin(path):
    data = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def write_new(path, data):
    assert path.resolve().is_relative_to(ROOT)
    with path.open('xb') as stream:
        stream.write(data)


def encode(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')


def main():
    audit_path = OUT / 'saved_return_audit_v1.json'
    peer_path = ROOT / 'outputs/codeformer_u96_postfusion_preflight_v1_r1_return_peer/independent_return_review_v1.json'
    manifest_path = OUT / 'visuals/manifest.json'
    audit = json.loads(audit_path.read_bytes())
    peer = json.loads(peer_path.read_bytes())
    manifest = json.loads(manifest_path.read_bytes())
    assert audit['saved_return_artifact_audit_pass'] and peer['independent_peer_saved_return_pass']
    assert audit['archive']['sha256'] == peer['archive']['sha256'] == 'a3ae70606487083068fc8ee8e0358636df8f1adc0a43caa236e12ccd13a4d57e'
    assert audit['protocol']['sha256'] == peer['protocol_sha256'] == 'ffa457e9b7573537fb006c02910cd109af0e17d1b4b7a576502c9f7aa6d14cdf'
    assert peer['training_optimizer_updates'] == 0 and not audit['model_qualified']
    assert len(manifest['sheets']) == 2 and manifest['generated_PNG_cells'] == 20
    sheets = []
    for sheet in manifest['sheets']:
        path = Path(sheet['file']['absolute_path'])
        actual = pin(path)
        assert actual['sha256'] == sheet['file']['sha256'] and actual['bytes'] == sheet['file']['bytes']
        sheets.append({'file': actual, 'witnesses': sheet['witnesses'],
                       'all_generated_PNG_cells_personally_viewed': sheet['generated_PNG_cells']})
    review = {
        'complete': True, 'review_type': 'root personal saved-sheet visual review',
        'recorded_UTC': datetime.now(timezone.utc).isoformat(),
        'auditor': pin(audit_path), 'independent_peer': pin(peer_path),
        'visual_manifest': pin(manifest_path), 'sheets': sheets,
        'planned_sheets_viewed': 2, 'generated_PNG_cells_viewed': 20,
        'all_planned_visuals_reviewed': True,
        'immutable_auditor_pending_flag_closed_by_this_companion': True,
        'observation': [
            'All zero-head candidate and gradient-forward cells visually match their incumbent U96 reference, consistent with the exact saved arrays.',
            'The incumbent already changes clear controls relative to their paired clean targets: smoothing, tonal/brightness changes and changes around eyes, glasses, smiles and teeth remain visible.',
            'Localized-blur witnesses remain soft. Eye/glasses and mouth detail differs from the actual clean targets; the preflight introduces no learned improvement.',
            'No blank or missing generated cell was found. Clear rows intentionally have no gradient-forward cell.'
        ],
        'per_photograph': {
            'tr_ffhq_00084': 'Glasses and mouth are softer than the clean target; clear-input deviations remain visible.',
            'tr_ffhq_00178': 'Smile/teeth and skin rendering differ from the target; localized facial softness remains.',
            'tr_ffhq_00323': 'Eye/skin and tonal rendering differ from the target; localized facial detail remains soft.',
            'tr_ffhq_00616': 'Face brightening/smoothing and eye/teeth changes remain visible relative to the target.'
        },
        'optimizer_updates': 0, 'new_trained_model': False,
        'useful_restoration_or_preservation_after_learning_established': False,
        'four_TRAIN_photographs_not_verified_independent_subjects': True,
        'native_unpaired_FINAL_or_automatic_preparation_qualification': False,
        'independent_human_FINAL_review': False,
        'new_model_forward_gradient_training_VM_browser_calls_in_this_publication': 0,
        'model_qualified': False
    }
    visual_path = OUT / 'root_visual_review_v1.json'
    write_new(visual_path, encode(review))
    report = """# U96 post-fusion R1: feasibility passed; actual learning remains to be tested

21 October 2026. The real returned archive passed the saved-output auditor and a separate peer's readback. Root personally reviewed both planned sheets and all20 generated PNG cells. This closes the preflight return, with zero optimizer updates. U96 remains the selected incumbent.

## Evidence

| Check | Returned result | Meaning |
|---|---|---|
| Transfer/source | 95,588,247-byte archive;316 files;109,726,056 payload bytes;43 nonweight static files exact | The saved return matches the sealed R1 protocol and source. |
| Initial output parity | Eight clear/local inputs;40 tensor pairs; native512, logits, LQ, RAW256, unclipped256 and delivered PNG parity | The complete R2-plus-U96 initializer is preserved by the zero-output heads on these same-device witnesses. |
| Image gradients | Four queries;144 finite arrays; all four final-projection site groups nonzero on each local case | The proposed trainable path receives an image-loss signal. Actual improvement and optimizer stability have not yet been measured. |
| State | 72 initial/final portable head arrays exact; final projections remain zero; zero optimizer constructions/updates | No learned successor was produced. |
| Resources | 58.357-second supervised worker; recorded peak RSS2.117GiB and CUDA reserved5.137GiB;85 samples | The preflight stayed within its finite limits. A training pilot still needs its own budget. |

The archive SHA256 is `a3ae70606487083068fc8ee8e0358636df8f1adc0a43caa236e12ccd13a4d57e`. Protocol SHA256 is `ffa457e9b7573537fb006c02910cd109af0e17d1b4b7a576502c9f7aa6d14cdf`. Export, checksum, data, predecessor failure and original packet remain retained.

[Saved-output audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_preflight_v1_r1_return_audit/saved_return_audit_v1.json>) and [independent peer readback](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_preflight_v1_r1_return_peer/independent_return_review_v1.json>) independently recomputed PNG conversion, objective reductions and gradient norms from saved arrays. Maximum native-to-delivery FP32 reconstruction difference was1.1921e-7; maximum gradient-norm reduction difference1.3878e-17; maximum objective FP32-versus-float64 reduction difference1.7967e-8. These are readback reduction differences, not loosened image parity or preservation gates.

## Visual review

[Root review](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_preflight_v1_r1_return_audit/root_visual_review_v1.json>) covers both exact saved256 sheets and every generated cell. Candidate and gradient-forward outputs visually match the retained U96 references, consistent with exact array parity. The localized-blur faces remain soft. Existing clear-control smoothing, brightness/tonal shifts and eye/glasses/mouth/teeth differences from the clean targets remain visible. No learned quality gain exists in this run.

The auditor and peer were written before root visual review, so their pending-visual flags remain immutable history. The companion root review closes that pending work. This is four TRAIN photographs, not four verified independent identities, native CCTV evidence, or an independent human FINAL review.

## Limits and next decision

The return omits the large frozen539-tensor payload. Its unchanged-state/grad-None claims are source-bound runtime receipts, not a fresh frozen-weight rehash from returned arrays. Portable head arrays were independently checked; associated PT files were hash-bound without unpickling. Same-device parity does not prove CPU/L4 parity or independent architecture correctness. Saved resource telemetry is not a fresh external VM observation. No backward warning was recorded; bitwise gradient reproducibility is still unproven. External installer-cache recovery evidence was not part of this return; installed source/model hash checks did pass.

The connected post-fusion path is now eligible for design of one distinct finite actual-learning experiment. The four current witness inputs were admitted for a nonoptimizing preflight only; they must not silently become optimizer data. Audit a separate optimization/monitor partition within existing genuine clean TRAIN sources, review the degraded inputs alone, retain clear controls and original terms/overlap limitations, and seal a fresh protocol before manual L4 execution. Preserve all current structure/preservation gates and save new optimizer/RNG/sampler state plus every failed state. Nonzero gradients are a feasibility result, not permission for automatic training or evidence that more epochs will help.

The bounded residual head cannot change frozen code indices and has zero correction wherever its stock residual is zero. The next actual-step experiment must test whether its connected signal yields useful image correction while preserving clear faces. Avoid further zero-optimizer checks as a substitute for that question. No optimizer, model, browser or VM action ran during this saved-artifact audit/publication.

U96, its exact R2 base, original CodeFormer, all historical checkpoints/splits/gate failures and research caches remain retained. Optional protected-surroundings preparation remains separate with confirmation/correction/bypass; its paired results do not justify always-on blur. Restoration remains first. Native/unpaired evidence, paired synthetic measurements, independent FINAL, user acceptance, full app integration and later covering-family qualification remain outstanding. The full thesis goal remains incomplete.
"""
    write_new(REPORT, report.encode('utf-8'))
    note = """# 21 October 2026: real U96 post-fusion R1 preflight passed; learning trial next

[Returned-evidence review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_U96_POSTFUSION_PREFLIGHT_V1_R1_REVIEW.md>): saved-output and independent peer audits passed the real316-file R1 return. Eight same-device parity cases/40 tensor pairs, all20 generated PNGs,144 finite gradient arrays and72 unchanged head arrays passed. All four final-projection site groups receive a nonzero image-loss gradient on each of four localized TRAIN witnesses. Root personally reviewed both planned sheets/all20 generated cells. Candidate outputs match U96 exactly; existing softness and clear-target deviations remain. This is technical feasibility, with0 optimizer constructions/updates and no trained successor or quality promotion.

The supervised worker finished in58.357s, with recorded peak RSS2.117GiB and CUDA reserved5.137GiB, within the sealed limits. Frozen539-tensor unchanged claims remain source/runtime-receipt-bound because their payload is omitted. PT files were hash-bound without unpickling; portable head arrays were independently recomputed. Same-device parity does not establish CPU/L4 equivalence, gradient replay or useful restoration. The R1 missing-dependency and installation issues are superseded by the successful returned runtime; original failures/recovery sources remain exact.

Next: prepare one distinct finite actual-learning pilot from complete U96, with freshly admitted optimization/monitor TRAIN roles, clear-control anchoring, actual-step preservation checks and saved optimizer/RNG/sampler state. Current four witness inputs are preflight-only and cannot silently become training data. New actual training stays under the user's manual verified-transfer/tmux workflow. No new VM/model/training call ran during this audit. No unchanged failed recipe, automatic historical launch or extra epochs are authorized by preflight success.

U96 remains incumbent. Optional protected-surroundings preparation, automatic/assisted separation and bypass remain in scope; paired studies do not support always-on blur. Preserve all models, splits, terms and gate failures. Restoration-first development, native/unpaired versus paired synthetic evidence, user acceptance, independent FINAL, full app flow and later covering-family qualification remain required. The full goal is incomplete. Older dated records below remain history; this entry supersedes their earlier next-action statements only.

---

""".encode('utf-8')
    docs = []
    for name in ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
                 'PROJECT_HANDOFF.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md'):
        path = ROOT / name
        previous = path.read_bytes()
        before = pin(path)
        pending = ROOT / (name + '.u96-r1-return-review.pending')
        write_new(pending, note + previous)
        assert path.read_bytes() == previous
        os.replace(pending, path)
        after = path.read_bytes()
        assert after[len(note):] == previous
        docs.append({'before': before, 'after': pin(path),
                     'preserved_suffix_offset_bytes': len(note),
                     'previous_bytes_preserved_exactly': True})
    receipt = {'complete': True, 'report': pin(REPORT), 'root_review': pin(visual_path),
               'saved_audit': pin(audit_path), 'peer_review': pin(peer_path),
               'documents': docs, 'model_or_app_or_VM_changed': False,
               'optimizer_updates': 0, 'model_qualified': False}
    write_new(OUT / 'milestone_publication_v1.json', encode(receipt))
    print(json.dumps({'complete': True, 'report': str(REPORT), 'root_visual_cells': 20,
                      'peer_audit_pass': True, 'optimizer_updates': 0, 'model_qualified': False}))


if __name__ == '__main__':
    main()
