"""Freeze a single full-cohort mask ablation after actual input/overlay review."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/completion_input_footprints_v1'
DRAFT = PARENT / 'mask_draft_v1_r1'
OUT = ROOT / 'outputs/completion_input_footprints_comparison_v1_r1'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(data,indent=2,allow_nan=False)+'\n')


def main():
    assert not OUT.exists(), 'Preserve existing comparison and its failure evidence'
    assert sha(PARENT/'protocol.json') == '84032b200aa6108f5e920b1a3124aefe39fd9ff307599ee2431b9b6f16585cd9'
    p,d,r = read(PARENT/'protocol.json'),read(DRAFT/'annotations.json'),read(DRAFT/'draft_receipt.json')
    assert r['complete'] and r['annotations_sha256']==sha(DRAFT/'annotations.json')
    for name,digest in r['artifacts_sha256'].items(): assert sha(DRAFT/name)==digest,name
    assets = dict(p['sources_sha256'])
    for name,digest in {**assets,**p['app_preservation_sha256']}.items(): assert sha(ROOT/name)==digest,name
    for folder in [PARENT/'input_atlases',DRAFT,PARENT/'draft_preparation_failure_v1']:
        for q in sorted(folder.rglob('*')):
            if q.is_file(): assets[q.relative_to(ROOT).as_posix()] = sha(q)
    files = [PARENT/'protocol.json',PARENT/'atlas_receipt.json',Path(__file__),
        ROOT/'scripts/draft_completion_input_footprints_v1.py',ROOT/'scripts/draft_completion_input_footprints_v1_r1.py',
        ROOT/'scripts/run_completion_input_footprints_comparison_v1_r1.py',ROOT/'scripts/supervise_completion_input_footprints_comparison_v1_r1.py',ROOT/'scripts/audit_completion_input_footprints_v1_r1.py',
        ROOT/'scripts/audit_completion_input_footprints_comparison_v1_r1.py']
    previous = read(ROOT/'outputs/dgp_app_covering_review_v3/results.json')
    cases = []
    for c in p['cases']:
        suffix = '_native' if c['condition']=='original_photo' else '_degraded'
        base = c['id'][:-len(suffix)]; v = {**c,'base_id':base,'rejected':c['input_review']!='usable'}
        if not v['rejected']:
            prepared = d['prepared'][base]
            v['masks'] = {k:(DRAFT/name).relative_to(ROOT).as_posix() for k,name in prepared['paths'].items()}
            v['prior_output'] = 'outputs/dgp_app_covering_review_v3/'+next(a for a in previous['rows'] if a['id']==c['id'])['assisted']['off']['output']
            assets[v['prior_output']] = sha(ROOT/v['prior_output'])
            assert assets[v['prior_output']] == previous['artifacts_sha256'][v['prior_output'].removeprefix('outputs/dgp_app_covering_review_v3/')]
        cases.append(v)
    for name in p['app_preservation_sha256']: files.append(ROOT/name)
    files.append(ROOT/'outputs/completion_input_footprints_runtime_v1/runtime_preflight.json')
    failed=ROOT/'outputs/completion_input_footprints_comparison_v1'
    assert read(failed/'external_receipt.json')['worker_exit_code']==1
    assert not (failed/'execution.json').exists() and not (failed/'results.json').exists()
    assert 'ModuleNotFoundError: No module named' in (failed/'inference.log').read_text()
    files.extend(q for q in failed.iterdir() if q.is_file())
    files.extend(ROOT/'scripts'/(name+'.py') for name in ['freeze_completion_input_footprints_v1','audit_completion_input_footprints_v1','run_completion_input_footprints_comparison_v1','supervise_completion_input_footprints_comparison_v1','audit_completion_input_footprints_comparison_v1'])
    for q in files: assets[q.relative_to(ROOT).as_posix()]=sha(q)
    OUT.mkdir()
    visual = {'complete':True,'datetime_UTC':datetime.now(timezone.utc).isoformat(),'draft_annotations_sha256':sha(DRAFT/'annotations.json'),
        'all6_draft_input_pages_actually_viewed_at_original_detail':True,'all36_input_cells_reviewed':True,
        'reviewer':'Implementing assistant input-only development review; not independent final evaluation',
        'pages':r['pages'],'supported_pairs':16,'excluded_pairs_retained':2,'source_assistance_on_degraded_pairs':True,
        'visual_findings':['Cloth/pink front, opaque lenses and mixed mask/hand remain complete within approximate facial extent; visible ears/hair/eyes outside remain.',
            'Clear frames are explicitly protected; a five-pixel glare/core conflict stopped preparation before any images and is retained. R1 traces below that rim.',
            'Hand footprints cover only facial overlap; exposed central nose/open mouth, head-top glasses, earrings and external hands remain.',
            'Hair trace follows right eye/cheek strands, retaining observed left gaze/nose/lips and ordinary hairstyle; hidden contour remains approximate.',
            'Scarf/glove, rose and leaf/finger traces are clipped to approximate facial overlap; uncovered/clear controls are exactly empty.'],
        'uncertainties_retained':['Soft glare/reflection and individual thread/strand/petal edges cannot be labeled as exact segmentation at256px.',
            'Hidden chin/cheek contour is approximate: object material outside the intended facial region remains and can affect a perceived join.',
            'A visible-mask review authorizes this one ablation, not a useful estimate or hidden identity claim.'],
        'new_generator_outputs_seen':False,'core_trace_not_output_fit':True,'quality_qualification':False,'model_forwards':0}
    write(OUT/'input_visual_review.json',visual)
    assets[(OUT/'input_visual_review.json').relative_to(ROOT).as_posix()]=sha(OUT/'input_visual_review.json')
    protocol = {'format':'completion-input-footprints-fixed-ablation-v1','complete':True,'frozen_UTC':datetime.now(timezone.utc).isoformat(),
        'parent_protocol_sha256':sha(PARENT/'protocol.json'),'cases':cases,'sources_sha256':assets,
        'app_preservation_sha256':p['app_preservation_sha256'],'input_visual_review_sha256':sha(OUT/'input_visual_review.json'),
        'source_scope':p['source_scope'],'exposed_development_photos':True,'hidden_ground_truth':None,'unknown_pretraining_overlap':True,
        'restoration':'Off in this causal mask-only ablation; current own-DGP main app restorer unchanged',
        'comparison':'All32 eligible cached current CodeFormer Off outputs against the same frozen current CodeFormer Off pipeline with corrected input-only masks',
        'geometry':'Same256x256 source pixels; original-photo assistance reused for synthetic degraded pair',
        'automatic':'Cached automatic proposals compared separately against approximate assisted footprint; no new automatic generation or automatic qualification',
        'mask_policy':p['mask_policy'],'seed':20261007,'search_trials_per_case':1,'device':'cpu',
        'runtime_preflight_sha256':sha(ROOT/'outputs/completion_input_footprints_runtime_v1/runtime_preflight.json'),
        'first_failed_system_python_launch_retained':True,'same_masks_model_settings_limits':True,
        'completion':'Existing official CodeFormer inpainting checkpoint, internal512, w1, adainFalse; visible-normalized-resize-v1',
        'display':'Same floor(float32*255), existing grayscale palette policy, exact original outside new removal support',
        'max_requests':36,'expected_rejections':4,'expected_eligible':32,'expected_empty_bypasses':4,
        'max_forwards':{'completion':28,'internal512':28,'DGP':0,'detector':0},
        'cap_seconds':600,'external_timeout_seconds':630,'artifact_cap_bytes':268435456,'new_checkpoint':False,'optimizer_updates':0,'backward_calls':0,
        'saved_stages':'All28 raw internal512 tensors and composited256 float32 tensors before PNG/display conversion; four empty controls save empty stages',
        'prospective_audit':'All saved PNG/raw compositions, exact support and protected bytes, all source/app hashes, count/time limits; all32 outputs reviewed on eight256px pages',
        'quality_criteria':['Covering removed within intended face, with outside-face objects retained by policy',
            'One plausible facial estimate; no severe anatomical duplication or invented visible-feature changes',
            'Visible features, ordinary clear glasses/hair and body/background preserved',
            'No conspicuous join, source covering remnant within mask target, or patterned replacement',
            'Empty controls bypass and unsupported/insufficient inputs reject before any neural generation'],
        'quality_qualification':False,'independent_final_review':False,'native_CCTV_or_reserved_final_used':False,
        'ethnicity_or_Zamboanga_inference':False,'app_changes':False,'goal_complete':False}
    write(OUT/'protocol.json',protocol)
    print(json.dumps({'complete':True,'protocol_sha256':sha(OUT/'protocol.json'),'cases':36,'source_bindings':len(assets),'generator_calls':0}))


if __name__ == '__main__': main()
