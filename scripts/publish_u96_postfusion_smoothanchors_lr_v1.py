"""Publish a reviewed fresh LR-factor comparison; never launches training."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_packet'
REV = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_design_review'


def doc(path):
    return json.loads(path.read_bytes())


def pin(path):
    raw = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


def write(path, raw):
    with path.open('xb') as stream:
        stream.write(raw)


def main():
    build_path = OUT / 'build_receipt_v1.json'
    audit_path = OUT / 'independent_packet_audit_v1.json'
    cli_path = OUT / 'local_cli_verification_v1.json'
    build = doc(build_path)
    assert build['complete'] and build['neural_gradient_optimizer_training_VM_calls'] == 0
    assert doc(audit_path)['independent_packet_audit_pass'] is True
    assert doc(cli_path)['complete'] and doc(cli_path)['transfer_verification_pass']
    for name in ['packet_helper_review_v1_r1.json', 'worker_source_review_v1.json']:
        review = doc(REV / name)
        assert review['complete'] and review['source_review_pass']
        for row in review['sources']:
            assert pin(ROOT / row['path']) == {key: row[key] for key in ('path', 'bytes', 'sha256')}
    data_path = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_data_review/conditional_data_reuse_admission_v1.json'
    data = doc(data_path)
    assert data['complete'] and data['independent_data_reuse_audit_pass'] and data['optimization_permitted']
    visual_path = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_v1_return_audit/root_visual_review_v1.json'
    visual = doc(visual_path)
    assert visual['complete'] and visual['all_planned_sheets_reviewed'] and visual['generated_PNG_cells_reviewed'] == 712
    old_out = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_v1_packet'
    old = doc(old_out / 'build_receipt_v1.json')
    prior_guide = ROOT / 'CCTV_CODEFORMER_U96_POSTFUSION_SMOOTHANCHORS_V1_VM.md'
    assert doc(old_out / 'milestone_publication_v1.json')['guide'] == pin(prior_guide)
    prefix = 'cctv-codeformer-u96-postfusion-smoothanchors-lr-v1'
    stage = 'cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm'
    protocol = build['protocol']['sha256']
    archive_sha = build['archive']['sha256']
    archive_bytes = build['archive']['bytes']
    guide = prior_guide.read_text(encoding='utf-8')
    replacements = [
        ('# U96 smoother-anchor comparison V1: five manual VM steps', '# U96 learning-rate comparison V1: five manual VM steps'),
        ('Prepared and **UNRUN**. The previous B64 trial completed64 real updates but failed the structure endpoint. Preserve it. This separate trial starts fresh from complete U96, changing only the clear/dark anchor loss to SmoothL1(beta0.01). It does not resume B64 or add epochs. U96 remains current.',
         'Prepared and **UNRUN**. The smoother-anchor trial completed64 updates with global PNG structure gain0.919% and localized gain0.665%, below1%. Preserve its quality stop. This distinct comparison starts fresh from complete U96 and changes ONLY maximum Adam learning rate0.00005 to0.0001. SmoothL1 beta0.01, all data,64-update cap and preservation gates stay unchanged. It does not resume a failed checkpoint or add epochs. U96 remains current.'),
        ('cctv-codeformer-u96-postfusion-smoothanchors-v1', prefix),
        ('codeformer_u96_postfusion_smoothanchors_v1', 'codeformer_u96_postfusion_smoothanchors_lr_v1'),
        ('run_u96_postfusion_smoothanchors_v1', 'run_u96_postfusion_smoothanchors_lr_v1'),
        ('u96_smoothanchors_v1', 'u96_smoothanchors_lr_v1'),
        (old['protocol']['sha256'], protocol),
        (old['archive']['sha256'], archive_sha),
        (f"{old['archive']['bytes']:,}", f'{archive_bytes:,}'),
        ('this new unbenchmarked criterion', 'this new unbenchmarked learning rate'),
        ('The same8 monitor photographs are reused adaptive TRAIN evidence after B64 feedback, not unseen validation.',
         'The same8 monitor photographs are reused adaptive TRAIN evidence after both preceding trials, not unseen validation. A higher learning rate can break preservation; the unchanged checks stop that run.'),
    ]
    for before, after in replacements:
        assert before in guide, 'Expected prior guide phrase absent: ' + before[:80]
        guide = guide.replace(before, after)
    assert guide.count('gcloud compute scp') == 5 and guide.count('## ') == 5
    assert prefix + '-results.tar.gz' in guide and protocol in guide
    guide_path = ROOT / 'CCTV_CODEFORMER_U96_POSTFUSION_SMOOTHANCHORS_LR_V1_VM.md'
    write(guide_path, guide.encode())
    report_path = ROOT / 'CCTV_CODEFORMER_U96_POSTFUSION_SMOOTHANCHORS_LR_V1_PREPARATION.md'
    report = f'''# Complete-U96 LR-only comparison V1: prepared, unrun

## Evidence and intended comparison

[Corrected prior return review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_U96_POSTFUSION_SMOOTHANCHORS_V1_RETURN_REVIEW_R1.md>) establishes64 committed updates, all preservation/dark TRAIN checks passed, and failed terminal structure gains: global RAW/PNG0.92155%/0.91927%, local-center0.67425%/0.66466% against1%. Independent pixel/state/metadata audits passed; root personally reviewed all30 planned sheets/712 generated images. Changes remain subtle with existing softness; there is no clear consistent jump in facial readability or a qualified replacement. U96 remains current. Original stopped states, model files, labels, splits, gates and research evidence remain immutable.

The actual learning route is connected and can improve the paired errors. Four fixed data-cycle paired objectives decreased; saved same-slot gradient directions became more persistent than under L1 anchoring, and clipping never activated. The late measured structure curve still increased. These observations support investigating step size before changing architecture again. They do not prove LR is too small, forecast continued gains, or establish that a higher LR preserves appearance. [Evidence-grounded method review](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_smoothanchors_v1_return_comparison/NEXT_METHOD_REVIEW_V1.md>).

This separate fresh comparison changes **only maximum Adam LR5e-5→1e-4**. Same complete U96 plus exact seeded new heads, fresh Adam, same16 optimization/8 monitor photographs/96 cases and fixed role order, same SmoothL1 beta0.01, losses/coefficients,539 frozen/36 trainable tensors,64-update cap,6snapshot schedule, scientific gates and resource limits. Namespace/format strings change for isolation. The first four LR values are2.5e-5,5e-5,7.5e-5,1e-4, then1e-4; actual FP64 expression represents the third as7.500000000000001e-5. No rounding or numerical-tolerance adjustment changes the recipe. No B2 resume,128-update extension, criterion change, new head or automatic surroundings blur is combined with this factor.

Higher LR is an exploratory hypothesis with a real risk of preservation failure. Prior legacy-clear mean headroom was only0.00239893 and minimum case headroom0.00389103 across snapshots. Do not treat those values as permission to change appearance. All historical original teachers/thresholds and new monitor checks remain exact. Stop on any failed preservation/resource gate; preserve the result. All four terminal blur endpoints still require1%. No automatic retry, appended epochs, retrospective winning checkpoint, threshold relaxation or model promotion follows.

## Dataset and qualification limits

The new explicit conditional reuse admission passed1,671 checks; original222 evidence pins,24 source1024/24 exact target256 images and96 input bytes/decoded RGBs were verified.16/8 photograph ordering,72 earlier readable degraded labels,24 clean controls, exclusions and all terms remain intact. This is data-only permission for the newly named finite LR comparison. Both previous experiments have informed design: the8 monitor photographs are adaptive reused TRAIN evidence, not unseen validation, verified separate identities or FINAL. No new image selection or native/wild/FINAL optimization occurs; preflight witnesses stay outside optimization. Source/pretraining/person overlap limits and NC/attribution restrictions remain recorded.

Even a passed small TRAIN screen must progress to broader fixed DEV, separately reported native unpaired evidence, user acceptance and independent FINAL. No ethnicity, exact identity or Zamboanga-performance claim is supported. Paired synthetic measurements remain separate from native/wild images without aligned targets.

## Finite execution and retained state

Same ceiling64 updates/256 role exposures, four passes over64 optimization cases, not64epochs. Adam betas(.9,.99), eps1e-8, no weight decay/AMP/foreach/fused optimization; clip norm1. Snapshot0/1/4/16/32/64 evaluates32 TRAIN-monitor and24 historical guard cases. RAW and floored PNG stay separate. All712 full forwards,256 backwards, one separately declared non-neural-base hook exception probe, one Adam and64commits are the maxima. Every committed head/gradient/Adam/RNG/sampler state, durable status and failure stays retained. Frozen539 payloads remain intentionally omitted from results, limiting unchanged claims to source/runtime associations and retained originals. CUDA backward bitwise replay is not claimed.

Require8GiB free after installing the two exact initializer copies; continuing4GiB reserve,10GiB hostRSS,18GiB CUDAreserved and1536MiB outputs. Worker2700seconds internal/2760external plus20seconds termination grace; export330seconds internal/345external,1536MiB payload/archive cap. Previous worker took416seconds; this LR's runtime is unbenchmarked, with10–20minutes planned excluding transfer/verification and the unchanged hard limits. Installation uses the retained successful R1 preflight and exactR2/U96 weights, not stoppedB2heads. Every Python command disables bytecode. Each Windows download has one remote source.

## Verification and next action

Independent worker/contract and helper/builder reviews, explicit data reuse, archive byte/namespace checks and actual inert stagedCLI verification passed. No local model/optimizer construction or step, forward/backward, VM connection or new training launch occurred. Tests and source checks establish preparation only; Linux/L4 actual execution and quality remain UNRUN.

[Five-step upload/install/tmux/download guide](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_U96_POSTFUSION_SMOOTHANCHORS_LR_V1_VM.md>). Thin execution archive{archive_bytes:,}bytes, SHA`{archive_sha}`; protocolSHA`{protocol}`. The user manually executes it. Returned archives, all saved RAW/PNG/state/gates and every planned sheet must again be independently audited before any app/model decision. The existing app/default model is unchanged.

Restoration stays first. Optional protected-surroundings preparation remains separately evaluated with exact protected input pixels, immutable-original previews, editable protection/application, correction/confirmation/bypass. Prior paired results do not justify always-on blur or arbitrary noise/JPEG/mosaic copying. Independent FINAL/fullapp/user acceptance and later completion/covering-family scope remain required; full goal incomplete.
'''
    write(report_path, report.encode())
    note = f'''# 22 October2026: fresh LR-only U96 comparison prepared; manual launch only

[Prior returned review, corrected R1](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_U96_POSTFUSION_SMOOTHANCHORS_V1_RETURN_REVIEW_R1.md>) retains verified64 commits and passing TRAIN preservation/dark checks, with failed1% endpoints: PNG global0.91927%,localcenter0.66466%. All712 RAW/PNG pairs,65 checkpoint states and all30 planned sheets/712 generated cells were audited/root personally reviewed. Learning improved measurably; changes remain subtle and no consistent facial-detail jump or qualified successor exists. U96 remains current; preserve the quality stop and all original assets. R1 corrects only static-file wording:242 nonweightfiles+protocol=243returned staticmembers; originalV1publication remains intact.

[New preparation](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_U96_POSTFUSION_SMOOTHANCHORS_LR_V1_PREPARATION.md>) and [five-step guide](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_U96_POSTFUSION_SMOOTHANCHORS_LR_V1_VM.md>) define one fresh64-update LR comparison5e-5→1e-4. Same originalcompleteU96/headseed/freshAdam, SmoothL1criterion, dataset/order, all1%/preservation/resource limits. It tests step size based on saved paired learning/gradient course; higherLRsafety and usefulness are unproven, with narrow legacyclear headroom. No stoppedstate resume/extraepochs, newarchitecture, thresholdchange or automatic launch.

New data-only reuse passed1,671 checks/all96 exact inputs/24targets; monitors are adaptiveTRAIN after both preceding experiments, not unseen DEV/FINAL or verified separate people. Labels/exclusions/terms remain intact. Independent source/packet/CLI verification establishes preparation only. Upload{archive_bytes:,}bytes; protocolSHA`{protocol}`. User manually uploads, installs and launches in tmux;8GiB free afterinstallation, worker46minutes plus20secondsgrace/export5minutes45seconds hardcaps. ActualLRpilot remains UNRUN. Returned evidence again requires all saved-pixel/state/gate audits and visual reviews; no model promotion follows archive/training completion.

Preserve original app design, checkpoints/splits/provenance/failures. Native unpaired evidence remains separate from paired synthetic metrics. Optional protected-surroundings preparation retains exact protected input pixels, original/derived previews/correction/confirmation/bypass; always-on blur remains unjustified. Restoration first; later completion/removal-mask/covering-family scope remains unchanged. BroaderDEV/native/useracceptance/independentFINAL/fullapp qualification and the full goal remain ACTIVE and incomplete. Older dated records remain exact history below; earlier next actions are superseded only by this entry.

---

'''.encode()
    documents = []
    for name in ['SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md', 'PROJECT_HANDOFF.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md']:
        path = ROOT / name
        old_bytes = path.read_bytes()
        before = pin(path)
        pending = ROOT / (name + '.smooth-lr-v1.pending')
        write(pending, note + old_bytes)
        assert path.read_bytes() == old_bytes
        os.replace(pending, path)
        assert path.read_bytes()[len(note):] == old_bytes
        documents.append({'before': before, 'after': pin(path), 'preserved_suffix_offset_bytes': len(note),
                          'old_bytes_preserved_exactly': True})
    receipt = {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
               'guide': pin(guide_path), 'preparation_report': pin(report_path), 'documents': documents,
               'build': pin(build_path), 'independent_packet_audit': pin(audit_path), 'inert_CLI': pin(cli_path),
               'new_model_optimizer_training_VM_calls': 0, 'actual_pilot_run': False, 'model_qualified': False}
    write(OUT / 'milestone_publication_v1.json', (json.dumps(receipt, indent=2) + '\n').encode())
    print(json.dumps({'complete': True, 'guide': str(guide_path), 'protocol_sha256': protocol,
                      'upload_bytes': archive_bytes, 'manual_training_UNRUN': True}))


if __name__ == '__main__':
    main()
