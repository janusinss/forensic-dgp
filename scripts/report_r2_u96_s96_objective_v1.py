"""Render the completed saved-only comparison report; no inference or training."""
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/codeformer_r2_u96_s96_objective_v1'
NAMES={'r2':'R2','spatial_u':'U96','spatial_s':'S96'}
ARMS=list(NAMES)


def pin(p):
    data=p.read_bytes();return {'path':p.relative_to(ROOT).as_posix(),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def main():
    path=OUT/'saved_output_audit_v1.json';audit=json.loads(path.read_bytes())
    assert audit['audit_pass'] and audit['complete'] and audit['counts']['cases']==68
    summaries={row['label']:row for row in audit['summary']}
    root_review=json.loads((OUT/'root_visual_review_v1.json').read_bytes())
    assert root_review['complete'] and root_review['personally_reviewed_sheets']==17
    key=json.loads((OUT/'visuals/panel_key.json').read_bytes())
    assert key['A']=='r2' and key['B']=='spatial_u' and key['C']=='spatial_s'
    closure=json.loads((OUT/'server_closure_v1.json').read_bytes())
    assert closure['complete'] and closure['interactive_status_after']['http_status']==200
    overall=summaries['all36paired_DEV_four_reference_photographs']
    ranking=sorted(ARMS,key=lambda arm:overall['arms'][arm]['PNG']['mae'])
    structural=sorted(ARMS,key=lambda arm:overall['arms'][arm]['PNG']['ssim'],reverse=True)
    def table(group):
        rows=['| Model | PNG MAE ↓ | PSNR dB ↑ | SSIM ↑ | Adjacent residual error ↓ | Centered residual MAE ↓ |',
              '|---|---:|---:|---:|---:|---:|']
        for arm in ARMS:
            row=group['arms'][arm]['PNG']
            rows.append(f"| {NAMES[arm]} | {row['mae']:.6f} | {row['psnr']:.3f} | {row['ssim']:.6f} | {row['adjacent_RGB_residual_gradient_mae']:.6f} | {row['mean_centered_residual_mae']:.6f} |")
        return '\n'.join(rows)
    lines=['# R2, U96 and S96: fixed development comparison','',
        '21 October 2026. This comparison answers the user request to check the three models objectively. It does not change the preferred model, promote an app default or qualify forensic identity recovery. All original states and the interrupted historical run remain intact.','',
        f"**Measured PNG reconstruction order on all 36 paired DEV observations: {' > '.join(NAMES[a] for a in ranking)}**, where better means lower image MAE to the saved clean photograph. The SSIM order is {' > '.join(NAMES[a] for a in structural)}. These are explicitly metric-specific orders, not a universal face-quality or identity ranking.",'',
        '## Fair input and execution contract','',
        'The exact previously frozen 68 inputs are retained: 36 paired filter-stress observations from four DEV reference photographs, 24 native ChokePoint crops and eight native QMUL-SurvFace crops. R2, U96 and S96 receive identical original-upload geometry, observed support, prepared RGB256, w=1/AdaIN and 256 delivery. No context blur, brightness enhancement, sharpening, alignment or model-specific input processing was added. The 40 input-only refusal controls remain excluded. Borderline filter cases and coarse QMUL cases remain explicit diagnostic probes, not usable-input admissions.','',
        'Thirty exact outputs from the ten previously audited cases were reused. The remaining 174 frozen CPU forwards ran on a separate loopback server, without occupying the interactive comparison server. All 204 output bundles/RAW/PNG/provenance records were checked from saved files with exact source bindings. Counts, Content-Length bodies, ZIP CRC/member sets, model file hashes, RAW-to-PNG floor quantization and exact outside-support preservation were verified. No training, gradient, optimizer, VM or checkpoint mutation occurred. The new audit is an author saved-only readback using the retained independently implemented metric helper, not an independent blinded clinical/forensic study.','',
        '## Paired results','',
        'Values below are arithmetic means of per-observation scores. The four photographs each have nine profiles; these are 36 observations, not 36 independent subjects. RAW and PNG scores are separately retained in JSON/CSV. SSIM uses the unchanged uniform7x7/sample-covariance RGB definition; it is not a facial-landmark or recognition score. Adjacent and centered residual errors are diagnostics, not the historical structure gate.','',
        '### All36 paired DEV observations','',table(overall),'',
        '### Readable degraded inputs only','',
        'Nine input-admitted degraded observations from three reference photographs; clear controls, borderline probes and the appearance-flagged reference are not mixed into this table.','',
        table(summaries['input_readable_degraded9']),'',
        '### Clear controls','',table(summaries['all4clear_controls']),'',
        'This table includes the separately flagged08025 reference. Passing a relative R2 comparison does not resolve historical clear/normal appearance failures or qualify automatic restoration.','',
        '### Source/appearance stratification','',
        '| Stratum | Observations | PNG MAE order, lowest first |', '|---|---:|---|']
    for label in ['all32degraded','input_readable12','input_borderline24','appearance_unflagged27','appearance_flagged9']:
        group=summaries[label];order=sorted(ARMS,key=lambda a:group['arms'][a]['PNG']['mae'])
        lines.append(f"| {label} | {group['cases']} | {' > '.join(NAMES[a] for a in order)} |")
    for group in audit['summary']:
        if group['label'].startswith('reference_'):
            order=sorted(ARMS,key=lambda a:group['arms'][a]['PNG']['mae'])
            lines.append(f"| {group['label']} | {group['cases']} | {' > '.join(NAMES[a] for a in order)} |")
    lines += ['', '### Direct paired wins','',
        '| Pair | PNG MAE better/equal/worse | PNG SSIM better/equal/worse |', '|---|---:|---:|']
    for pair,row in overall['PNG_pairwise'].items():
        def count(metric):return '/'.join(str(row[metric][k]) for k in ('better','equal','worse'))
        lines.append(f"| {pair} | {count('mae')} | {count('ssim')} |")
    lines += ['', 'The exact-zero equality band is numerical bookkeeping only; a tiny metric win is not a meaningful visible improvement. One photograph contributes nine correlated observations. Repeated DEV exposure and possible unknown pretrained/subject overlap remain limitations.','',
        '## Native CCTV: unpaired evidence','',
        '| Capture source | Crops | Clean target | U96 difference from R2, mean RGB8 levels | S96 difference from R2, mean RGB8 levels |',
        '|---|---:|---|---:|---:|']
    for source,row in audit['native_source_separate'].items():
        delta=row['mean_RGB8_difference_to_R2_observed_support_not_quality']
        lines.append(f"| {source} | {row['cases']} | None | {delta['spatial_u']:.4f} | {delta['spatial_s']:.4f} |")
    lines += ['', 'Output distance measures change, not correctness. Native PSNR/SSIM and a clean-target winner are intentionally null. Do not rank generated native details by apparent sharpness or invent target accuracy. Source labels do not establish ethnicity or Zamboanga performance. Root visual preservation findings are recorded separately after every planned sheet is personally inspected.','',
        '## Prior training evidence remains separate','',
        '[The retained independent TRAIN review](outputs/codeformer_spatial_continuation_return_review_v1/INDEPENDENT_NUMERIC_REVIEW.md) reports U96 moderate/strong PNG MAE reductions of8.8875%/4.8837% versus R2, compared with S96 reductions1.0739%/0.6705%. S96 lowers centered feature L1 more (0.6890% versus U96 0.1296%). These fitted/repeated TRAIN measurements do not replace the new DEV comparison. Small TRAIN guards passed, but inherited full-development preservation failures remain unresolved.','',
        '## Visual evidence and limits','',
        'All sheets paste exact saved 256 PNGs beside inputs and paired targets/native notes. Coded A/B/C panels reduce model-label emphasis, but root review is not an independent blinded study. The panel key is retained. No screenshot enhancement or resampling is used in sheet construction.','']
    lines += ['Coded findings were recorded before opening the key: A=R2, B=U96, C=S96. R2 and S96 look extremely similar. U96 mainly changes tone/local contrast and is modestly darker on several cases. All three share smoothing, altered eye/mouth/glasses detail and persistent local-blur/mosaic defects. The review did not establish a material new facial-structure winner. Coarse QMUL probes can produce weakly supported or malformed features with all three models; input uncertainty remains in force.','',
        'Root personally reviewed all17 sheets/204 model PNG cells. The nine final paired sheets are byte-identical to the reviewed coded previews. Paired visual review preceded the full numerical score readout; the remaining native review followed the paired scores, so coding does not imply fully blinded review. [Root visual observations](outputs/codeformer_r2_u96_s96_objective_v1/root_visual_review_v1.json).','',
        '| Sheet | Cases / outputs |','|---|---|']
    visual=json.loads((OUT/'visuals/manifest.json').read_bytes())
    for row in visual['sheets']:
        binding=row['file'];lines.append(f"| [{Path(binding['path']).name}]({binding['path']}) | {len(row['case_ids'])} / {row['model_output_cells']} |")
    lines += ['', 'Full478 historical/filter coverage, reserved independent FINAL, formal preservation gates and user restoration acceptance remain outstanding. This fixed68 subset can support a metric-specific measured preference; it does not prove an overall qualified replacement. R2 remains a protected permanent reference regardless of the descriptive numeric order. No app default or current training priority is changed automatically.','',
        'Evidence: [frozen plan](outputs/codeformer_r2_u96_s96_objective_v1/comparison_plan_v1.json), [saved output audit + all metrics](outputs/codeformer_r2_u96_s96_objective_v1/saved_output_audit_v1.json), [paired RAW/PNG CSV](outputs/codeformer_r2_u96_s96_objective_v1/paired_metrics_v1.csv), [metric sanity fixtures](outputs/codeformer_r2_u96_s96_objective_v1/metric_fixture_check_v1.json), [coded panel key](outputs/codeformer_r2_u96_s96_objective_v1/visuals/panel_key.json).']
    report=ROOT/'CCTV_CODEFORMER_R2_U96_S96_OBJECTIVE_V1_REVIEW.md'
    rendered='\n'.join(lines)+'\n'
    rendered=rendered.replace('Thirty exact outputs from the ten previously audited cases were reused.', 'Thirty exact outputs from the ten previously audited cases were reused. The174 new CPU forwards completed in988.8seconds.')
    rendered=rendered.replace('Root visual preservation findings are recorded separately after every planned sheet is personally inspected.', 'All17 planned sheets were personally inspected; findings are bound in the root review receipt.')
    rendered=rendered.replace('### Readable degraded inputs only', 'U96 reduces mean PNG MAE by5.6209% versus R2; S96 reduces it by0.6358%. U96 wins MAE on36/36 observations and SSIM on34/36. This is not a percentage improvement in facial identity accuracy or the historical1% structure gate.\n\n### Readable degraded inputs only')
    rendered=rendered.replace('### Clear controls', 'U96 reduces mean PNG MAE by7.5597% on the nine readable degraded observations. This small subset does not cover every CCTV degradation.\n\n### Clear controls')
    rendered += '\nThe batch server was stopped after completion. The previous interactive process was later found absent; its cause was not established. The unmodified interactive launcher was restarted on61815 and its zero-forward status verified. [Server lifecycle](outputs/codeformer_r2_u96_s96_objective_v1/server_closure_v1.json).\n'
    rendered=re.sub(r'\]\((?!http)([^)]+)\)',lambda m:'](<'+(ROOT/m.group(1)).as_posix()+'>)',rendered)
    assert not report.exists();report.write_text(rendered,encoding='utf-8')
    record={'complete':True,'report':pin(report),'audit':pin(path),'numeric_PNG_MAE_order':ranking,
        'numeric_PNG_SSIM_order':structural,'root_visual_review_complete':True,
        'root_visual_review':pin(OUT/'root_visual_review_v1.json'),'server_closure':pin(OUT/'server_closure_v1.json'),
        'overall_qualified_winner':None,'main_app_default_changed':False,'training_optimizer_VM_calls':0}
    write=OUT/'report_manifest_v1.json';write.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record))


if __name__=='__main__':main()
