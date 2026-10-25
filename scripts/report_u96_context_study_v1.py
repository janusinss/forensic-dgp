"""Render a source-bound study report after audit and all root visual review."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/codeformer_u96_context_study_v1'
OUT=BASE/'recovery_r1'
NAMES={'original:original':'Untouched CodeFormer','original:r2':'R2','original:spatial_u':'U96 / original','prepared:spatial_u':'U96 / prepared'}
def pin(p):
    b=p.read_bytes();return {'path':p.relative_to(ROOT).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def link(label,path):return '['+label+'](<'+(ROOT/path).as_posix()+'>)'
def main():
    audit=json.loads((OUT/'saved_output_audit_v1.json').read_bytes());assert audit['audit_pass'] and audit['complete']
    review=json.loads((OUT/'root_visual_review_v1.json').read_bytes());assert review['complete'] and review['sheets_personally_viewed']==6
    admission_path=BASE/'ancillary_input_only_review_v1.json'
    admission=json.loads(admission_path.read_bytes());assert admission['complete'] and len(admission['cases'])==16
    decisions_by_id={r['id']:r for r in admission['cases']}
    expected={r['id']:r for r in audit['cases'] if r['role']!='clear'}
    assert len(decisions_by_id)==16 and set(decisions_by_id)==set(expected)
    assert dict(Counter(r['input_only_decision'] for r in admission['cases']))==admission['counts']
    for id_,row in decisions_by_id.items():
        assert row['role']==expected[id_]['role'] and row['reference_id']==expected[id_]['reference_id']
        assert row['original_input']==expected[id_]['source']['input']
    roles=[g for g in audit['summary'] if 'arms' in g]
    assert len(roles)==3
    lines=['# U96: broader paired development and protected surroundings study','',
        '21 October 2026. Eight additional historical DEV photographs, three synthetic roles and four output conditions give24 inputs/96 verified outputs. The references were not in the earlier four-photograph filter comparison. This is a descriptive development study, not independent subject/identity validation or a qualified model promotion. U96 remains the user-selected development model.','',
        '## Fixed intervention and limits','',
        'The same original input is restored by untouched CodeFormer, R2 and U96. U96 also receives a reviewed protected-surroundings version. All outputs use the same frozen weights, w1/AdaIN, native square256 geometry, internal512 and delivered256, with no display enhancement. Eight manually selected protection rectangles were frozen from inputs before outputs. The generated synthetic central128 corruption footprint is known; this is assisted/known-footprint evidence, not automatic detection.','',
        'Clear, global Gaussian2.4 and local Gaussian3.0 inputs use retained clean targets and SciPy float64 reflect/truncate3 corruption. The existing page applies Pillow nominal radius6 outside protection, with8pixel guard and16pixel outside transition. This is the fixed current-page intervention, not an exact matched kernel/strength. Clear/global prepared cases are forced harm controls; they do not verify automatic bypass.','',
        'The primary comparison below uses the central128 patch, which equals the known local corruption footprint only in the local role. Whole-image, protected-mask and surroundings metrics are retained separately, as are RAW and delivered PNG. Adjacent residual-gradient and centered errors help distinguish structure/offset changes, but do not replace facial preservation requirements. No historical quality gate is changed or claimed passed from this table.','',
        '## Input admission review correction','',
        'The original candidate sheets placed clear controls beside degraded counterparts. Their original usable labels are retained as provisional assisted judgments; they do not meet the strict no-clean-reference rule. A fresh isolated reviewer, with no parent history or model/target access, separately judged sixteen anonymous degraded inputs using the frozen input-only criteria. This was an ancillary post-freeze review while inference was running. The frozen plan, masks and all outputs remain intact; no case was removed after results. This is agent review, not independent human FINAL evaluation.','',
        'Ancillary decisions: '+', '.join(f'{k}: {v}' for k,v in sorted(admission['counts'].items()))+'.','',
        link('Admission correction and per-input reasons',admission_path.relative_to(ROOT).as_posix())+'.','',
        'The all-eight role means below include borderline and insufficient stress cases. They are diagnostic results, not an admitted-usefulness cohort. No generated image can reverse an insufficient-input decision. The readable subset is shown separately; its small post-freeze agent review cannot establish a production threshold.','',
        'No local-blur case was admitted readable; this run cannot establish useful localized preparation on an input-admitted cohort.','',
        '## Paired results','']
    for group in roles:
        lines+=['### '+group['role'],'','| Condition | Central PNG MAE ↓ | Central gradient error ↓ | Central centered MAE ↓ | Whole-image MAE ↓ |','|---|---:|---:|---:|---:|']
        for key,name in NAMES.items():
            row=group['arms'][key];center=row['central128_patch']
            lines.append(f"| {name} | {center['mae']:.6f} | {center['adjacent_RGB_residual_gradient_mae']:.6f} | {center['mean_centered_residual_mae']:.6f} | {row['whole_image']['mae']:.6f} |")
        delta=group['preparation_vs_original_U96']['central128_patch']
        lines+=['',f"Prepared versus original U96: central PNG MAE better/equal/worse = {delta['better']}/{delta['equal']}/{delta['worse']}; mean prepared-minus-original difference {delta['mean_MAE_difference']:.8f}.",'']
    admission_by_id={r['id']:r['input_only_decision'] for r in admission['cases']}
    lines+=['## Ancillary input-only strata','',
        '| Role | Input-only decision | Photographs | U96 original central PNG MAE | U96 prepared central PNG MAE |',
        '|---|---|---:|---:|---:|']
    for role in ('global_gaussian_2p4','local_gaussian_3p0'):
        for label in ('readable','borderline','insufficient','out_of_scope'):
            subset=[r for r in audit['cases'] if r['role']==role and admission_by_id[r['id']]==label]
            if not subset:continue
            mean=lambda key:sum(r['metrics'][key]['PNG']['central128_patch']['mae'] for r in subset)/len(subset)
            lines.append(f"| {role} | {label} | {len(subset)} | {mean('original:spatial_u'):.6f} | {mean('prepared:spatial_u'):.6f} |")
    lines+=['## Decision and next bounded work','',
        'Retain U96 as the user-selected development baseline. This run changes no weights and accepts no replacement. U96 has lower mean central PNG image error than R2 in all three roles, while untouched CodeFormer has lower mean central image error than both adaptations. The adjacent-gradient ranking differs, and U96 brightness bias is separately retained. None of those rankings alone establishes visible facial fidelity or a qualified restorer.','',
        'The fixed current-page surroundings recipe is not a demonstrated general improvement: local central error is lower on four cases and higher on four, with worse mean error; globally blurred controls worsen on all eight central patches. All twenty-four whole-image errors worsen versus original-input U96. Permitted surroundings are intentionally changed, so whole-image error alone is not the facial outcome. The six-sheet visual review does not show a consistent localized-face benefit. The earlier unpaired wild5 observation remains a distinct single-image result.','',
        'Keep preparation optional and bypassable; do not automatically apply nominal radius6, qualify the estimator or repeat this unchanged recipe. First generate milder localized inputs from the retained DEV sources, review degraded inputs alone before model outputs, and freeze a readable cohort. Then compare reviewed protection and a prospective strength/operator contract. This is a proposed next diagnostic, not a sealed or executed training plan. The broad current protection rectangles also preserve some sharp context inside the guard; a future footprint comparison must preserve readable appearance while distinguishing footprint and strength effects.','',
        'Further learning must be justified separately from input preparation, retaining complete U96 initialization, clear controls, brightness versus structure diagnostics, old failures and manual finite L4 execution. No epoch extension or new pilot is justified by this study alone.','',
        '## Appearance flag and source dependence','',
        'The19791 makeup/jewelry source is retained, not silently dropped after outputs. Its score and the other seven photographs are separately recorded below. Repeated roles of one photograph are correlated. Photo/file separation from project TRAIN does not prove subject separation or absence from CodeFormer pretraining. These FFHQ photographs are paired synthetic development evidence, not native CCTV or evidence of Asian ethnicity or Zamboanga performance.','',
        '| Role | Appearance flagged | Photos | U96 original central MAE | U96 prepared central MAE |','|---|---|---:|---:|---:|']
    for g in audit['summary']:
        if 'appearance_flag' in g:
            d=g['central128_patch_PNG_MAE'];lines.append(f"| {g['role']} | {g['appearance_flag']} | {g['photographs']} | {d['original:spatial_u']:.6f} | {d['prepared:spatial_u']:.6f} |")
    lines+=['','## Visual review','',*review['findings'],'',
        'Root personally inspected all six coded sheets and96 generated PNG cells. The key was opened only after recording coded findings. This is author review with known project context, not an independent blinded human study. Native CCTV and reserved FINAL were not included in this run.','',
        '## Execution and integrity','',
        'The original client stopped at its first response, whose body was2705bytes shorter than Content-Length. Its worker, partial body and failure remain unchanged and are not counted as a verified output. Root observed one completed original-CodeFormer forward in an earlier live server status query, but did not save a standalone post-failure counter JSON. Therefore the reported combined97 forwards (original25, R224, U9648) depends on that earlier root observation, not independently retained prior server-after telemetry. The completed recovery independently verifies96 outputs/forwards (original24, R224, U9648), with fresh zero-before/exact-after counters. A separately pinned transport-only recovery used keep-alive and the same frozen cases/masks/model settings after two large zero-neural static responses passed. This does not prove the original socket failure cause.','',
        link('Retrospective prior-forward observation and evidence limit',str((OUT/'prior_forward_observation_v1.json').relative_to(ROOT)).replace('\\','/'))+'. This R2 report supersedes the first draft report regarding counter provenance; the earlier report/manifest remain intact as history.','',
        'The independent saved-pixel implementation reconstructs all Gaussian inputs, reviewed protection/alpha/derived images, entire preprocessing bundles, original-versus-derived payloads, model lineage and RAW/PNG quantization. All image/geometry/source bindings and recovery call counters agree; the metadata annotation97 is not independent verification of the earlier prior1. Every protected input pixel stays exact; generated appearance can still change. New training, optimizer updates, VM actions, app-source changes and automatic-detector qualification: zero.','',
        '## Evidence','',
        link('Frozen study plan','outputs/codeformer_u96_context_study_v1/comparison_plan_v1.json')+', '+link('Saved output audit',str((OUT/'saved_output_audit_v1.json').relative_to(ROOT)).replace('\\','/'))+', '+link('Region metrics CSV',str((OUT/'paired_region_metrics_v1.csv').relative_to(ROOT)).replace('\\','/'))+'.','',
        link('Primary-source research and plugin availability','CCTV_CODEFORMER_U96_CONTEXT_RESEARCH_V1.md')+'. No new plugin was installed: catalog calls were unavailable; the readable local Academic Research Suite design workflow and built-in primary-source browsing were used.','',
        '| Visual sheet | Inputs | Outputs |','|---|---:|---:|']
    visual=json.loads((OUT/'visuals/manifest.json').read_bytes())
    for row in visual['sheets']:lines.append('| '+link(Path(row['file']['path']).name,row['file']['path'])+f" | {len(row['cases'])} | {row['model_cells']} |")
    lines+=['','The next decision must use the source-specific numeric and visual findings above. No average improvement alone establishes useful structure, correct identity, automatic preprocessing readiness or full thesis completion. A future matching-strength method remains a separate prospective comparison.']
    path=ROOT/'CCTV_CODEFORMER_U96_CONTEXT_STUDY_V1_R2_REVIEW.md'
    with path.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    result={'complete':True,'report':pin(path),'audit':pin(OUT/'saved_output_audit_v1.json'),'root_review':pin(OUT/'root_visual_review_v1.json'),'ancillary_input_review':pin(admission_path),'model_qualified':False,'goal_complete':False}
    result['counter_provenance_note']=pin(OUT/'prior_forward_observation_v1.json')
    result['supersedes_report_counter_wording']=pin(ROOT/'CCTV_CODEFORMER_U96_CONTEXT_STUDY_V1_REVIEW.md')
    with (OUT/'report_manifest_v1_r2.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))

if __name__=='__main__':main()
