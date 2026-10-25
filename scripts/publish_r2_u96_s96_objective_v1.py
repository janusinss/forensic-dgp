"""Publish the measured comparison and retain complete preceding document bytes."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/codeformer_r2_u96_s96_objective_v1'

def pin(p):
    b=p.read_bytes()
    return {'path':p.relative_to(ROOT).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

def main():
    authority=ROOT/'outputs/codeformer_r2_focus_v1/publication_v1/publication_v1.json'
    assert pin(authority)['sha256']=='2bd69bcf56ec80caa734722e485fec44947343a9b3d27db8b27979d0e143dfc7'
    previous=json.loads(authority.read_bytes())
    report=json.loads((OUT/'report_manifest_v1.json').read_bytes())
    assert report['complete'] and report['root_visual_review_complete']
    assert report['numeric_PNG_MAE_order']==['spatial_u','spatial_s','r2']
    assert pin(ROOT/report['report']['path'])==report['report']
    audit=json.loads((OUT/'saved_output_audit_v1.json').read_bytes())
    assert audit['audit_pass'] and audit['complete']
    originals={name:(ROOT/name).read_bytes() for name in previous['documents']}
    intervening={}
    for name in originals:
        expected=previous['documents'][name]['current']
        if pin(ROOT/name)!=expected:
            # A disjoint document branch appended a publication note during this comparison.
            assert name=='PROJECT_HANDOFF.md'
            assert pin(ROOT/name)['sha256']=='787154823aced67b26211bcd92befb37e9006583bee71ceb7ec8e7fce4c40cd6'
            assert hashlib.sha256(originals[name][:expected['bytes']]).hexdigest()==expected['sha256']
            extension=originals[name][expected['bytes']:]
            assert b'Thesis document branch publication - 21 October 2026' in extension
            intervening[name]={'previous_published':expected,'current_before_this_publication':pin(ROOT/name),
                'previous_published_bytes_preserved_as_prefix':True,
                'extension_bytes':len(extension),'extension_sha256':hashlib.sha256(extension).hexdigest(),
                'reason':'Disjoint thesis document branch publication note; retained byte-exact, not reverted.'}
    evidence_paths=[
        'CCTV_CODEFORMER_R2_U96_S96_OBJECTIVE_V1_REVIEW.md',
        'scripts/compare_r2_u96_s96_objective_v1.py','scripts/audit_r2_u96_s96_objective_v1.py',
        'scripts/report_r2_u96_s96_objective_v1.py','scripts/record_r2_u96_s96_root_visual_v1.py',
        'scripts/audit_codeformer_spatial_dev_local_v1_r2.py','scripts/publish_r2_u96_s96_objective_v1.py',
        'outputs/codeformer_spatial_dev_review_v1/cohort_manifest_v1.json',
        'outputs/codeformer_spatial_dev_review_v1/run_v1_r2/comparison.json',
        'scratch/codeformer_r2_u96_s96_objective_server_v1.json',
        'scratch/codeformer_spatial_research_server_objective_return_v1.json',
    ]
    for name in ['comparison_plan_v1.json','comparison.json','saved_output_audit_v1.json',
                 'paired_metrics_v1.csv','visuals/manifest.json','visuals/panel_key.json',
                 'root_visual_review_v1.json','report_manifest_v1.json','model_file_readback_v1.json',
                 'metric_fixture_check_v1.json','server_closure_v1.json']:
        evidence_paths.append((OUT/name).relative_to(ROOT).as_posix())
    evidence={name:pin(ROOT/name) for name in evidence_paths}
    entry='''# 21 October 2026: objective R2 / U96 / S96 comparison completed

The requested fixed three-model comparison is complete: 68 exact inputs, 204 saved outputs, 30 exact reused outputs and 174 new CPU forwards in 988.8 seconds. All saved response bodies, ZIP members, RAW/PNG/provenance, observed support and model-file bindings passed the author saved-only audit. Root personally inspected all 17 coded sheets/204 model PNG cells. This is not a new independent blinded study. [Full measured comparison](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_R2_U96_S96_OBJECTIVE_V1_REVIEW.md>).

On 36 paired DEV observations from only four reference photographs, the delivered PNG reconstruction order is U96 > S96 > R2. Mean MAE: 0.044855 / 0.047224 / 0.047526 respectively; SSIM: 0.795592 / 0.791194 / 0.789688. U96 reduces mean MAE by 5.6209% versus R2 and wins MAE on 36/36 observations. S96 reduces mean MAE by 0.6358%. Nine readable degraded observations from three photographs separately favor U96 by 7.5597% MAE. These are pixel/SSIM diagnostics, not identity accuracy, the historical one-percent structure requirement or an overall qualified replacement.

Native ChokePoint (24 crops) and QMUL-SurvFace (8 coarse probes) remain separate unpaired evidence, with no aligned targets or native PSNR/SSIM winner. R2 and S96 look extremely close; U96 chiefly changes tone/local contrast. All three retain smoothing, altered fine facial detail and local-blur/mosaic limitations. Root visual review did not establish a material new structure winner. Coarse inputs remain diagnostic probes, not usable-input acceptance. Repeated DEV exposure, only four paired source photographs, unknown overlap and unresolved full-development/FINAL preservation limit generalization. No ethnicity or Zamboanga-performance claim is supported.

This completes only the requested three-model comparison. The earlier interrupted four-arm run is preserved as 10 completed cases/40 outputs with 58 pending; it has not been relabeled complete. No model weights, main-app defaults, input threshold, recipe or VM state changed. R2 stays a protected permanent reference and the user-selected improvement initializer; U96 is the metric-specific leading challenger. A preferred successor still requires useful visible improvement, preservation and user review. The isolated batch server was stopped; the prior interactive process was found absent for an undetermined reason. The unchanged launcher was restarted on 61815 and idle status verified without neural calls.

The R2 spatial prototype remains untrained/unrun and no training packet is sealed. Automatic localized-blur estimation, two editable Protect/Apply areas and adaptive matching remain planned; the existing optional manually reviewed surroundings page retains fixed strength and immutable-original preparation. All actual training stays under the user's manual finite L4/tmux workflow. Full native/DEV usefulness, independent FINAL, app automatic selection/override and restoration acceptance remain outstanding. Completion development stays deferred. The overall thesis goal remains active and incomplete; earlier dated entries are retained history.

---

'''.encode('utf-8')
    publication=OUT/'publication_v1'
    publication.mkdir(exist_ok=False)
    documents={}
    for name,original in originals.items():
        backup=publication/(name+'.before')
        with backup.open('xb') as f:f.write(original)
        (ROOT/name).write_bytes(entry+original)
        assert (ROOT/name).read_bytes()[len(entry):]==original
        documents[name]={'before':pin(backup),'current':pin(ROOT/name),
            'preserved_suffix_offset':len(entry),'full_previous_bytes_preserved':True}
    record={'complete':True,'UTC':datetime.now(timezone.utc).isoformat(),
        'previous_publication':pin(authority),'intervening_document_readback':intervening,
        'documents':documents,'evidence':evidence,
        'requested_three_model_comparison_complete':True,'historical_four_arm_run_complete':False,
        'new_CPU_forwards':174,'total_model_outputs':204,'root_visual_sheets_reviewed':17,
        'numeric_paired_PNG_MAE_order':['spatial_u','spatial_s','r2'],
        'independent_review_performed_this_turn':False,'overall_qualified_winner':None,
        'main_app_default_changed':False,'training_optimizer_VM_calls':0,'overall_goal_complete':False}
    target=publication/'publication_v1.json'
    with target.open('x',encoding='utf-8') as f:json.dump(record,f,indent=2);f.write('\n')
    # Final author readback verifies document continuity and every declared file binding.
    for row in record['documents'].values():
        assert pin(ROOT/row['current']['path'])==row['current']
        assert pin(ROOT/row['before']['path'])==row['before']
    for row in record['evidence'].values():assert pin(ROOT/row['path'])==row
    closure={'complete':True,'publication':pin(target),'author_readback_pass':True,
        'independent_review':False,'all_previous_document_bytes_retained':True,
        'main_app_default_changed':False,'overall_goal_complete':False}
    with (publication/'author_closure_v1.json').open('x',encoding='utf-8') as f:
        json.dump(closure,f,indent=2);f.write('\n')
    print(json.dumps(closure))

if __name__=='__main__':main()
