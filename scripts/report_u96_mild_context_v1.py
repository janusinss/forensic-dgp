"""Publish the completed prospective DEV study, without further model calls."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/codeformer_u96_mild_context_v1'
ROLES=('clear','global_gaussian_1p2','local_gaussian_1p2')


def pin(p):
    b=p.read_bytes();return {'path':p.relative_to(ROOT).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def link(label,p):return '['+label+'](<'+(ROOT/p).as_posix()+'>)'
def write(p,d):
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')


def main():
    audit=json.loads((OUT/'saved_output_audit_v1.json').read_bytes())
    review=json.loads((OUT/'root_visual_review_v1.json').read_bytes())
    admission=json.loads((OUT/'input_admission_v1.json').read_bytes())
    plan=json.loads((OUT/'comparison_plan_v1.json').read_bytes())
    independent=json.loads((OUT/'independent_saved_readback_v1.json').read_bytes())
    assert audit['complete'] and audit['audit_pass'] and len(audit['cases'])==21
    assert review['complete'] and review['sheets_personally_viewed']==6 and review['generated_PNG_cells_personally_viewed']==42
    assert review['condition_key_opened_before_this_record'] is False
    assert review['numeric_scores_read_before_this_record'] is False
    assert admission['feasibility_pass'] and len(admission['selected_reference_ids'])==7
    assert independent['complete']
    summaries=[]
    for role in ROLES:
        rows=[r for r in audit['cases'] if r['role']==role];assert len(rows)==7
        summary={'role':role,'photographs':7,'metrics':{},'direction':{}}
        for fmt in ('RAW','PNG'):
            summary['metrics'][fmt]={}
            for state in ('original','prepared'):
                summary['metrics'][fmt][state]={region:{m:float(np.mean([r['metrics'][state][fmt][region][m] for r in rows])) for m in ('mae','adjacent_RGB_residual_gradient_mae','mean_centered_residual_mae','signed_luminance_bias','absolute_luminance_bias')} for region in ('whole_image','protected_region','central128_patch')}
                summary['metrics'][fmt][state]['central128_patch']['ssim']=float(np.mean([r['metrics'][state][fmt]['central128_patch']['ssim'] for r in rows]))
            summary['direction'][fmt]={}
            for region in ('whole_image','protected_region','central128_patch'):
                delta=[r['metrics']['prepared'][fmt][region]['mae']-r['metrics']['original'][fmt][region]['mae'] for r in rows]
                summary['direction'][fmt][region]={'better':sum(v<0 for v in delta),'equal':sum(v==0 for v in delta),'worse':sum(v>0 for v in delta),'mean_difference':float(np.mean(delta))}
        summaries.append(summary)
    lines=['# U96: readable localized-blur study','',
        '21 October 2026. Completed local inference: seven historical DEV photographs, three roles and two U96 conditions, giving 21 inputs and 42 verified outputs. There were 35 new forwards and seven exact reused clear-control outputs. This is a preprocessing comparison with unchanged weights, not a newly trained model.','',
        '## Decision','',
        'Retain U96 as the selected development model. The prospective matched-operator preparation did not demonstrate a consistent visible facial improvement. Local central PNG error improves on three photographs and worsens on four; mean central and protected-face errors worsen. Small center SSIM/gradient/centered-error improvements coexist with increased luminance bias. These mixed measures do not pass preservation or establish useful automatic preparation.','',
        'Keep surroundings preparation optional and bypassable. Do not deploy an always-on blur rule or continue sweeping masks/strengths on this same DEV cohort. The single unpaired wild5 response remains valid as a separate context-sensitivity observation, but neither it nor this study establishes an automatic rule for arbitrary crops. The existing app and its default model were not changed by this study.','',
        '## Prospective admission and operator','',
        'Forty inputs were prepared from eight term-audited historical DEV photographs: clear plus local/global Gaussian1.2/1.8. A fresh isolated reviewer saw only32 anonymous degraded inputs, without clean references or restored outputs. It labelled17 readable and15 borderline; none was insufficient or out of scope. All8 local1.2 inputs were readable; seven global1.2 counterparts were readable. The selection contract, fixed before reading decisions, selected those seven paired photographs. Sigma1.8 was reviewed for coverage only and never used for neural selection. Exclusions and all original decisions remain retained. The clear control is the exact retained clean input; historical source knowledge is disclosed, not called new blind admission.','',
        'Each selected photograph contributes clear, global1.2 and local1.2 roles. Local corruption replaces only the half-open central128 patch[64:192,64:192]. Gaussian generation and preparation both use SciPy float64 reflect/truncate3 at nominal sigma1.2, finite spatial radius4, followed by rint/clipRGB8. Preparation filters the observed input, never the clean target. Whole-face rectangles protect readable eyes, mouth, jaw and glasses, expanded by6pixels with an8pixel outside smoothstep transition. Every protected input RGB value is unchanged. The auditor also verifies that no positive-alpha filter footprint samples the already blurred patch.','',
        'These are known synthetic operator/mask interventions. Broad protection preserves some sharp context and an outside transition, so the result is not globally uniform blur or exact unknown-blur estimation. Clear/global prepared arms intentionally test harm; globally soft inputs would normally bypass. The sigma, kernel, guard and feather differ from the earlier stress study, so cross-study differences do not isolate one causal factor. Input-only agent review is not a calibrated threshold or independent human FINAL assessment.','',
        '## Paired delivered-PNG outcomes','',
        '| Role | Central MAE original → prepared ↓ | Central SSIM original → prepared ↑ | Protected MAE original → prepared ↓ | Central MAE better / worse |',
        '|---|---:|---:|---:|---:|']
    for s in summaries:
        o=s['metrics']['PNG']['original'];p=s['metrics']['PNG']['prepared'];d=s['direction']['PNG']['central128_patch']
        lines.append(f"| {s['role']} | {o['central128_patch']['mae']:.6f} → {p['central128_patch']['mae']:.6f} | {o['central128_patch']['ssim']:.6f} → {p['central128_patch']['ssim']:.6f} | {o['protected_region']['mae']:.6f} → {p['protected_region']['mae']:.6f} | {d['better']} / {d['worse']} |")
    lines+=['','On localized inputs, protected-face MAE worsens on6/7 photographs. Whole-image MAE worsens on all21 inputs, although intentionally altered surroundings mean that whole-image error alone is not the facial outcome. Center128 equals the corruption footprint only in the local role; protected rectangles are approximate face regions, not anatomical/identity ground truth. Paired metrics compare against the actual clean target; they are not CCTV identity-accuracy estimates.','',
        '| Role | Central gradient original → prepared ↓ | Central centered MAE original → prepared ↓ | Central absolute luminance bias original → prepared ↓ |',
        '|---|---:|---:|---:|']
    for s in summaries:
        o=s['metrics']['PNG']['original']['central128_patch'];p=s['metrics']['PNG']['prepared']['central128_patch']
        lines.append(f"| {s['role']} | {o['adjacent_RGB_residual_gradient_mae']:.6f} → {p['adjacent_RGB_residual_gradient_mae']:.6f} | {o['mean_centered_residual_mae']:.6f} → {p['mean_centered_residual_mae']:.6f} | {o['absolute_luminance_bias']:.6f} → {p['absolute_luminance_bias']:.6f} |")
    lines+=['','## RAW output kept separate','',
        'RAW here is clipped/resized delivery float32 before floorRGB8 PNG conversion, not unclipped native network activations. Neither RAW nor PNG uses display enhancement. The source-bound CSV retains both formats and all four regions.','',
        '| Role | Central RAW MAE original → prepared ↓ | Protected RAW MAE original → prepared ↓ |',
        '|---|---:|---:|']
    for s in summaries:
        o=s['metrics']['RAW']['original'];p=s['metrics']['RAW']['prepared']
        lines.append(f"| {s['role']} | {o['central128_patch']['mae']:.6f} → {p['central128_patch']['mae']:.6f} | {o['protected_region']['mae']:.6f} → {p['protected_region']['mae']:.6f} |")
    lines+=['','## Per-photograph local outcome','',
        '| Photograph | Appearance flag | Central PNG MAE original | Prepared | Protected PNG MAE original | Prepared |',
        '|---|---|---:|---:|---:|---:|']
    for r in audit['cases']:
        if r['role']!='local_gaussian_1p2':continue
        o=r['metrics']['original']['PNG'];p=r['metrics']['prepared']['PNG']
        lines.append(f"| {r['reference_id']} | {r['appearance_flag']} | {o['central128_patch']['mae']:.6f} | {p['central128_patch']['mae']:.6f} | {o['protected_region']['mae']:.6f} | {p['protected_region']['mae']:.6f} |")
    lines+=['','The makeup/jewelry19791 source remains in the cohort and is flagged. Source-specific errors remain available rather than silently removing it. Seven repeated photographs, with correlated roles, do not establish population performance. Source/photo separation from project TRAIN does not prove human-identity separation or absence from CodeFormer pretraining. Terms are retained source snapshots, not newly verified web terms. No native CCTV, reserved FINAL, ethnicity or Zamboanga-performance claim follows from this FFHQ synthetic study.','',
        '## Visual review and integrity','',*review['findings'],'',
        'Root personally inspected all6 coded sheets/42 generated cells before opening the condition key or numerical results. The key mapped A=prepared/B=original for all three roles. This is author/agent review with prior project context, not an independent blinded human study. The independent saved-only auditor checked42 response bodies, lengths, ZIP/RAW/PNG/provenance records, all40 candidate inputs,32 anonymous admissions and21 preparations. A separate NumPy/skimage readback checked336 metric rows and2856 scalar values, agreeing within1.365e-12. Counter receipts prove35 new U96 forwards; seven reused outputs are physically joined to their original provenance and never counted as new.','',
        'A dedicated batch server completed in268.3seconds and was stopped after its terminal client result. The shared interactive server was left alone. The finite plan capped the worker at1200seconds and each request at150seconds, with4GiB disk/384MiB RAM reserves and1GiB outputs. No weights, optimizer, gradients, training, VM state, app source or quality gates changed.','',
        '## Next model work','',
        'Stop treating context softening as the general quality fix. The separate learning-path audit found that U96 continuation optimizes cached encoder-feature losses; its old delivery helper detaches the image graph. A future image-learning route needs a connected delivery expression and complete U96 parity. The current original-base prototype must not silently replace the user-selected U96 initializer. A new bounded post-fusion route is a hypothesis; source preparation or gradients alone do not qualify a trained model.','',
        link('U96 learning-path review','CCTV_CODEFORMER_U96_LEARNING_PATH_REVIEW_V1.md')+'. A finite manual-L4 zero-optimizer preflight must first prove initial complete U96 parity, unchanged frozen539 tensors, finite delivered-image gradients and residual leverage. No training pilot or new model acceptance is authorized by this study result alone. Keep all predecessors/gates; any later learning comparison must retain clear controls and use the same source-bound inputs/preparation for incumbent and challenger.','',
        '## Evidence','',
        link('Frozen plan','outputs/codeformer_u96_mild_context_v1/comparison_plan_v1.json')+', '+link('Input admission','outputs/codeformer_u96_mild_context_v1/input_admission_v1.json')+', '+link('Saved-output audit','outputs/codeformer_u96_mild_context_v1/saved_output_audit_v1.json')+', '+link('Independent readback','outputs/codeformer_u96_mild_context_v1/independent_saved_readback_v1.json')+', '+link('RAW/PNG metrics','outputs/codeformer_u96_mild_context_v1/region_metrics_v1.csv')+'.','',
        '| Sheet | Generated outputs |','|---|---:|']
    visual=json.loads((OUT/'visuals/manifest.json').read_bytes())
    for row in visual['sheets']:
        lines.append('| '+link(Path(row['file']['path']).name,row['file']['path'])+f" | {row.get('model_cells',len(row.get('cases',[]))*2)} |")
    lines+=['','The full restoration, independent FINAL, automatic preparation, main-app and later covering-family goal remains ACTIVE and incomplete.']
    path=ROOT/'CCTV_CODEFORMER_U96_MILD_CONTEXT_V1_REVIEW.md'
    with path.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    result={'complete':True,'report':pin(path),'auditor':pin(OUT/'saved_output_audit_v1.json'),'independent_readback':pin(OUT/'independent_saved_readback_v1.json'),'visual_review':pin(OUT/'root_visual_review_v1.json'),'summary':summaries,'model_qualified':False,'goal_complete':False,'new_training_or_VM_calls':0}
    write(OUT/'report_manifest_v1.json',result)
    print(json.dumps({'complete':True,'report':result['report'],'model_qualified':False,'goal_complete':False}))


if __name__=='__main__':main()
