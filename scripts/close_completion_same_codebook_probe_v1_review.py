"""Root saved-PNG endpoint replay and completed all-sheet visual-review record.

No returned Python, model runtime, inference, training or VM operation is used.
Visual observations were made separately with original-detail view_image calls.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/completion_codeformer_same_codebook_probe_v1_return'
RUN = BASE / 'cctv_completion_codeformer_same_codebook_probe_v1_results'
STAGE = ROOT / 'outputs/completion_codeformer_same_codebook_probe_v1_packet/draft_001/cctv_completion_codeformer_same_codebook_probe_v1_vm'
PINS = {
    'transport_audit.json': 'ee4980dcd22a04f4315d7369a16c6fb2ce8ad8704532f6e1b284d35a45d24fa4',
    'independent_state_code_audit.json': '794fed73804eabfa081805cdd0d3db3dda89d5a1fbd6138df725b6b404ad922d',
    'independent_saved_output_audit.json': '4f5042edc19496bd3dee515dc052928c03e4f6d9e12b246efdcca4262a54d02f',
    'visual_sheet_index.json': '13bb379941d6e333d2e4dbb754fc21aad08ca8ac066f26e23c83210743b68263',
    'visual_index_coordinate_recheck_full.json': '70ba4ae3fd1c2a7a7f5bc39e9de4e2c0d76231b1b4272b3797f0a8e3aa7f2dfd',
}
NOTES = [
    '05179: Clean-code C improves the lower smile/teeth shape relative to A; D has more target-like texture but changes mouth/microphone detail. B gives a different open mouth. Eye-band, single-lens and cheek estimates remain imperfect; C is generally closer, without an exact-reconstruction claim.',
    '05302: C improves eye-band shape over soft A. D gives clearer eyes and fuller target-like lower lips, but still differs. B sharpens some detail while changing eyelids/mouth. Small lens and cheek masks show modest differences; sharper alone is not sufficient.',
    '06781: A and B turn a puffed closed mouth into a smile with teeth. C restores a closed mouth but misses the full puff; D more closely retains the known expression. C/D eye shape is nearer the narrow target eyes. All four masks were inspected.',
    '06809: C lower-face output opens the mouth more than the closed-lip target, so oracle substitution is not universally better. D is closer to the target mouth/eye appearance but shows a hard eye-band transition; C stays soft. Lens and cheek differences are small.',
    '06995: C lower-face and eye-band estimates remain smooth and alter expression. D restores more wrinkles and mouth detail, with transition differences. B adds texture but is not a reliable improvement. Single-lens and cheek changes are subtle.',
    '07087: C restores clearer glasses in the hidden eye band than A, but remains soft and pale; D is nearer the target frame/eyes. Lower-face C is closer to the closed target mouth than B. Small lens and cheek masks remain relatively close across arms.',
    '07202: A/C lower-face estimates add a toothier smile than the reference; D is closer to the subtle open mouth and skin texture. C eye band remains soft, while D has a visible tonal rectangle. All small-mask outputs were inspected.',
    '07386: C improves the hidden smile/teeth relative to A; D is more target-like, with altered fine detail. Eye-band C remains softer than the reference. B changes lip/eye appearance. Lens and cheek differences are modest.',
    'Challenge01: Gray 05179 A/B alter the smile and introduce colored lips; C restores teeth but keeps some color, D widens the smile and has a stronger transition. RGB irregular lens B/D have hard textured patches. Gray 05302 eyes stay colored in A/C, stronger in B; cheek differences are small.',
    'Challenge02: Gray 06781 lens remains slightly colored in every arm. RGB lower-face A/B invent a smile; C closes the mouth, D is closer to the puffed target but is patchier. Gray 06809 cheek and RGB eye-band D/B show hard transitions, while C remains soft.',
    'Challenge03: Gray 06995 lower-face outputs retain false color; C is less smooth but not an exact match. RGB lens differences are small. Gray 07087 C/D restore glasses from hidden clean-reference codes; A/B largely miss them, and color remains. This cannot prove glasses can be recovered from a real covered input. All cheek outputs were inspected.',
    'Challenge04: Gray 07202 lens shows colored/etched eyebrow texture, especially B/D; C still has residual artifacts. RGB lower-face D is more target-like but has a sharp textured patch; C still changes the smile. Gray 07386 cheek changes are modest; RGB eye-band D is detailed but visibly discontinuous, C is softer.',
]

def need(ok, message):
    if not ok: raise ValueError(message)

def physical(path):
    p = Path(path).resolve()
    return Path('\\\\?\\' + str(p)) if os.name == 'nt' and not str(p).startswith('\\\\?\\') else p

def sha(path):
    h = hashlib.sha256()
    with physical(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(physical(path).read_text(encoding='utf-8-sig'))

def pin(path):
    p = Path(path).resolve()
    return {'path': str(p), 'bytes': physical(p).stat().st_size, 'sha256': sha(p)}

def pixels(path, mode):
    with Image.open(physical(path)) as im:
        need(im.format == 'PNG' and im.size == (256, 256) and im.mode == mode, 'PNG schema')
        return np.array(im, copy=True)

def write_new(path, obj):
    with physical(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, indent=2, allow_nan=False); f.write('\n')

def main():
    for name, digest in PINS.items(): need(sha(BASE/name) == digest, 'Admission drift: '+name)
    p = read(RUN/'protocol.json')
    need(sha(RUN/'protocol.json') == '51782b57fca9ad9cf7161b0f4451a65e545e3eb2f8f58788fa40b64aec48d1ec', 'Protocol drift')
    transport = read(BASE/'transport_audit.json')
    # A fresh full returned-payload hash pass, without executing returned sources.
    for name, row in transport['files'].items():
        path = RUN/name
        need(physical(path).stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Payload drift '+name)
    records = []
    for kind, key in [('monitor','monitor_cases'), ('challenge','challenge_cases')]:
        for case in p[key]:
            target = pixels(STAGE/case['clean256'], 'RGB')
            source = pixels(STAGE/case['input256'], 'RGB')
            mask = pixels(STAGE/case['mask'], 'L') == 255
            # F.max_pool2d inverse-mask geometry: padding cannot add a neighbor.
            inv = np.pad((~mask).astype(np.uint8), 3, constant_values=0)
            near_visible = np.lib.stride_tricks.sliding_window_view(inv, (7,7)).max(axis=(-1,-2)).astype(bool)
            boundary = mask & near_visible
            for arm in p['arms']:
                suffix = '__ORACLE_unavailable_clean_information' if arm['oracle'] else ''
                relative = 'outputs/'+arm['filename_tag']+'/'+kind+'/'+case['id']+suffix+'.png'
                result = pixels(RUN/relative, 'RGB')
                need(np.array_equal(result[~mask], source[~mask]), 'Outside source pixels changed')
                delta = np.abs(result.astype(np.float64)-target)
                row = {'id':case['id'], 'kind':kind, 'arm':arm['id'], 'mask_kind':case['mask_kind'], 'condition':case['condition'], 'PNG_hole_L1':float(delta[mask].mean()/255), 'PNG_boundary_L1':float(delta[boundary].mean()/255)}
                saved = read(RUN/'outputs'/arm['filename_tag']/kind/'results.json')
                original = next(r for r in saved['records'] if r['id']==case['id'])
                need(all(abs(row[k]-original[k])<=1e-12 for k in ('PNG_hole_L1','PNG_boundary_L1')), 'Endpoint mismatch')
                records.append(row)
    comparisons = {}
    for newer, older in [('B','A'),('C','A'),('D','A'),('D','B')]:
        key = newer+'_versus_'+older; comparisons[key] = {}
        for kind in ('monitor','challenge'):
            aa = [r for r in records if r['arm']==newer and r['kind']==kind]
            bb = [r for r in records if r['arm']==older and r['kind']==kind]
            a = sum(r['PNG_hole_L1'] for r in aa)/len(aa); b = sum(r['PNG_hole_L1'] for r in bb)/len(bb)
            comparisons[key][kind] = {'PNG_hole_L1_new':a, 'PNG_hole_L1_reference':b, 'hole_error_reduction_percent':100*(b-a)/b, 'individual_holes_worse':sum(x['PNG_hole_L1']>y['PNG_hole_L1']+1e-12 for x,y in zip(aa,bb)), 'individual_boundaries_worse':sum(x['PNG_boundary_L1']>y['PNG_boundary_L1']+1e-12 for x,y in zip(aa,bb))}
    admission = read(ROOT/'outputs/completion_codeformer_finetune_r1_app/source_admission.json')
    pins = []
    for item in [*admission['source_files'], admission['candidate_manifest'], admission['checkpoint']]:
        q=Path(item['path']); need(physical(q).stat().st_size==item['bytes'] and sha(q)==item['sha256'], 'R1 asset drift'); pins.append(pin(q))
    # Twenty sources plus the exact candidate manifest and inference checkpoint.
    need(len(pins)==22, 'Expected all22 R1 source/model assets')
    index = read(BASE/'visual_sheet_index.json')
    pages=[]
    for sheet, note in zip(index['sheets'], NOTES):
        need(sha(sheet['file']['path'])==sheet['file']['sha256'], 'Reviewed sheet drift')
        pages.append({'sheet':sheet['file'], 'case_ids':[r['id'] for r in sheet['cases']], 'reviewed_by_root':True, 'inspection':'view_image original detail; one whole1792x1260 sheet per call; all seven256px cells per row examined', 'observations':note})
    need(len(pages)==12 and sum(len(r['case_ids']) for r in pages)==48, 'All-case review coverage')
    common={'created_UTC':datetime.now(timezone.utc).isoformat(), 'complete':True, 'model_qualified':False, 'new_model_created':False, 'app_changed':False, 'scope':'Adaptive paired TRAIN diagnosis. Oracle C/D use unavailable clean context. No hidden identity, native CCTV, independent FINAL or full covering-family qualification.'}
    closure={**common, 'format':'root-same-codebook-saved-PNG-replay-and-preservation-v1', 'author':pin(Path(__file__)), 'input_audits':{n:pin(BASE/n) for n in PINS}, 'returned_payload_files_rehashed':len(transport['files']), '192_PNG_hole_and_boundary_endpoints_recomputed':True, 'endpoint_rows':records, 'comparisons':comparisons, 'R1_app_model_assets_exact':pins, 'neural_calls':0, 'optimizer_steps':0, 'VM_calls':0}
    write_new(BASE/'root_endpoint_and_preservation_closure.json', closure)
    visual={**common, 'format':'root-same-codebook-probe-all12-sheet-visual-review-v1', 'reviewer':'root agent development assessment, not blinded independent human FINAL', 'index_sha256':PINS['visual_sheet_index.json'], 'transport_sha256':PINS['transport_audit.json'], 'sheets_reviewed':12, 'cases_reviewed':48, 'all_actual_saved_outputs_covered':192, 'all336_cells_examined':True, 'sheet_pages':pages, 'visual_outcome':'Clean-context SAME-codebook C with fixed R1 fusion gives useful diagnostic expression/eye/glasses improvements, but remains soft and can alter hidden appearance/color. Actual no-fusion B is rejected; D is not a deployment route and has hard seam/color artifacts. Retain R1. Review an isolated predictor-learning design, preserving context encoder/fusion/prior and all gates; oracle gain is not an achievable guarantee.', 'major_remaining_defects':['Soft generated eyes and skin even with clean-context codes','Hidden-expression uncertainty; C can introduce teeth into a closed-mouth target','False color on strictly gray inputs','Hard seams/texture and altered eyewear details in no-fusion controls','These masks are synthetic adaptive TRAIN; real covering families and final review remain unproven'], 'input_display_clarification':'The observed-input column displays prepared source RGB; the separate reviewed-support overlay identifies pixels removed from neural conditioning. Hidden clean pixels were not pasted into the network input.', 'oracle_is_not_deployable':True, 'user_acceptance_pending':True, 'independent_human_FINAL_pending':True, 'local_model_inference_calls':0}
    write_new(BASE/'root_visual_review.json', visual)
    print(json.dumps({'visual':pin(BASE/'root_visual_review.json'), 'closure':pin(BASE/'root_endpoint_and_preservation_closure.json'), 'comparisons':comparisons}))

if __name__=='__main__': main()
