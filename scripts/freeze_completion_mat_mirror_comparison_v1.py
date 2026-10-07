"""Freeze all32 eligible exposed development cases before any MAT model forward."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/completion_mat_mirror_comparison_v1'


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    assert not OUT.exists();OUT.mkdir()
    prior=ROOT/'outputs/completion_pixel_support_v1_r1/plan.json'
    old=read(prior);assert old['complete'] and len(old['cases'])==36
    assets=read(ROOT/'outputs/completion_mat_mirror_assets_v1/independent_assets_audit.json')
    assert assets['complete'] and assets['adapter_regressions_passed']==11 and assets['pretrained_model_forwards']==0
    for name,digest in old['sources_sha256'].items():assert sha(ROOT/name)==digest,name
    cases=[{k:r[k] for k in ['id','family','condition','input','reviewed','output','stages','metadata','operator_input_review']} for r in old['cases'] if not r['rejected']]
    excluded=[{k:r[k] for k in ['id','family','condition','operator_input_review']} for r in old['cases'] if r['rejected']]
    assert len(cases)==32 and len(excluded)==4 and all(r['operator_input_review']=='usable' for r in cases)
    families={r['family'] for r in cases}
    assert {'face_mask','sunglasses','strong_lens_glare','hand','obstructing_hair','scarf','other_object','uncovered_control','clear_glasses_control'}<=families
    sources=[Path(__file__),ROOT/'scripts/run_completion_mat_mirror_comparison_v1.py',
             ROOT/'scripts/audit_completion_mat_mirror_comparison_v1.py',ROOT/'mat_mirror_completion_v1.py',
             ROOT/'tests/test_mat_mirror_completion_v1.py',prior,
             ROOT/'outputs/completion_pixel_support_v1_r1/independent_audit.json',
             ROOT/'outputs/completion_mat_mirror_review_v1_r1/acquisition.json',
             ROOT/'outputs/completion_mat_mirror_review_v1_r1/independent_source_audit.json',
             ROOT/'outputs/completion_mat_mirror_assets_v1/acquisition.json',
             ROOT/'outputs/completion_mat_mirror_assets_v1/preparation.json',
             ROOT/'outputs/completion_mat_mirror_assets_v1/independent_assets_audit.json',
             ROOT/'outputs/completion_mat_mirror_assets_v1/MAT_FFHQ_512_fp16.safetensors']
    sources.extend(ROOT/r[k] for r in cases for k in ['input','reviewed','output','stages','metadata'])
    prepared=read(ROOT/'outputs/completion_mat_mirror_assets_v1/preparation.json')
    sources.extend(ROOT/'outputs/completion_mat_mirror_assets_v1'/n for n in prepared['vendor_sha256'])
    app=read(ROOT/'outputs/dgp_app_v3_integration_record.json')
    app_sources={n:d for n,d in app['sources_sha256'].items() if n!='static/face_workflow.js'}
    for name,digest in app_sources.items():assert sha(ROOT/name)==digest
    app_sources['static/face_workflow.js']=sha(ROOT/'static/face_workflow.js')
    checkpoint='outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth'
    app_sources[checkpoint]=app['checkpoint_sha256'];assert sha(ROOT/checkpoint)==app['checkpoint_sha256']
    p={'format':'converted-MAT-FFHQ512-completion-development-comparison-v1','status':'frozen_before_generation',
       'frozen_UTC':datetime.now(timezone.utc).isoformat(),'cases':cases,'excluded_before_generation':excluded,
       'assisted_only':True,'automatic_outputs_not_generated_or_qualified':True,
       'restoration_selection':'Off to isolate completion; retained original own-DGP app unchanged.',
       'baseline':'Reuse all32 audited current-app Off outputs on identical source/mask.',
       'source_scope':'Previously exposed original photographs and their synthetic degradations. Legacy native suffix is not native CCTV.',
       'hidden_ground_truth':None,'hidden_PSNR_SSIM':None,'pretraining_overlap':'Unknown; no final independent qualification.',
       'seed':240,'noise_mode':'const','truncation':1,'precision':'FP32 CPU computation from published FP16 weights.',
       'latent':'Same first np.random.RandomState240 vector for each case; reset torch RNG240 per forward inside fork_rng.',
       'mask':'Reviewed binary removal mask unchanged; zero additional margin; no posthoc rectangles or color selection.',
       'input_geometry':'RGB256 -> visible-support-normalized bilinear512; nearest removal mask. Covered values zeroed.',
       'output_geometry':'Raw512 [-1,1] -> clamp to RGB[0,1] -> bilinear256 -> exact input outside mask -> round255 PNG.',
       'internal_resolution':512,'delivered_resolution':256,'threads':4,
       'budgets':{'requests':32,'nonempty_pretrained_forwards':28,'empty_control_bypasses':4,
                  'case_seconds':90,'total_seconds':1200,'external_seconds':1230,'maximum_output_bytes':200*1024**2},
       'review_criteria':['Plausible coherent eye/nose/mouth/cheek/jaw estimate; no exact hidden identity claim.',
                          'Visible facial appearance, clear glasses and non-obstructing hair preserved outside the frozen mask.',
                          'No covering/edge remains inside requested removal support; record outside-mask leftovers separately.',
                          'Gaze/skin/anatomy and boundary transitions judged at actual256 size across every eligible case.',
                          'No app selection or automatic-family quality claim until independent saved-output audit and full review.'],
       'sources_sha256':{f.relative_to(ROOT).as_posix():sha(f) for f in sources},
       'app_preservation_sha256':app_sources,'new_training':False,'optimizer_updates':0,
       'native_CCTV_or_reserved_final_used':False,'goal_complete':False}
    with (OUT/'protocol.json').open('x',encoding='utf-8',newline='\n') as stream:json.dump(p,stream,indent=2)
    print(json.dumps({'complete':True,'protocol_sha256':sha(OUT/'protocol.json'),'eligible_cases':32,
                      'excluded_cases':4,'all_seven_covering_families':True,'new_model_forwards':0},indent=2))


if __name__=='__main__':main()
