"""Record the completed root original-detail review, with exact sheet pins."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/completion_codeformer_predictor_ce_v1_return'
NOTES=[
('original_monitor_01','05179 allfour masks examined at allfour snapshots. The generated lower smile remains narrower/lessopen than the known target, with a smooth cheek/chin. Eye-band shape is still soft; u32 does not give a clear visible improvement. Small lens and cheek results remain very similar.'),
('original_monitor_02','05302 allfour masks examined at allfour snapshots. Lowerface keeps thin softened lips rather than the known target mouth. The eye band stays soft; small lens and cheek changes are subtle. No consistent material final32 improvement over R1.'),
('original_monitor_03','06781 allfour masks examined at allfour snapshots. The important puffed closed-mouth expression is replaced by a smile with teeth in R1 and all new snapshots. The eye band changes narrow target eyes into rounder eyes. Updating the predictor did not repair these generated-anatomy/expression defects.'),
('original_monitor_04','06809 allfour masks examined at allfour snapshots. The generated mouth remains paler and narrower than the target smile. The eye band and lens remain softer than the visible known target; u32 changes are minor and do not establish improvement.'),
('original_monitor_05','06995 allfour masks examined at allfour snapshots. Lowerface remains overly smooth, with missing target wrinkles/stubble and a changed lip shape. The eye band remains soft. Small lens/cheek differences are slight; no clear final32 advantage.'),
('original_monitor_06','07087 allfour masks examined at allfour snapshots. When glasses are hidden by the synthetic eye-band support, neither R1 nor final32 reconstructs the known frame/eyes well; the covered region stays pale/soft. Lower mouth and small-mask cases show only minor changes. Visible glasses outside support remain exact.'),
('original_monitor_07','07202 allfour masks examined at allfour snapshots. Generated lowerface opens a toothier smile than the target expression. Eye band and small lens remain soft; the lens has etched/irregular local texture. Final32 does not resolve these differences.'),
('original_monitor_08','07386 allfour masks examined at allfour snapshots. Lowerface smile is smoother and narrower than the known target; eye band lacks the sharp eye detail. Small lens and cheek differences remain modest. No broad final32 clarity improvement.'),
('challenge_01','Allfour challengecases/allfour snapshots examined. Gray05179 generated lowerface has a closed altered mouth and false lip color despite a toothy target; gray05302 eyes have bluecolor. RGB lens and cheek estimates change subtly. These defects persist at final32.'),
('challenge_02','Allfour challengecases/allfour snapshots examined. Gray06781 hidden lens is blue; RGB06781 lowerface still invents a smile instead of puffed closed lips. Gray06809 cheek is smooth and RGB eye-band differs from target. No consistent final32 improvement.'),
('challenge_03','Allfour challengecases/allfour snapshots examined. Gray06995 lowerface keeps falsecolor; gray07087 eye band invents blueeyes and loses known glasses. RGB small-lens and cheek changes are subtle. The final32 output does not correct the central representation defects.'),
('challenge_04','Allfour challengecases/allfour snapshots examined. Gray07202 lens has an etched eyebrow/eye patch and falsecolor; RGB lowerface changes mouth shape. Gray07386 cheek is smooth and RGB eyeband is soft. Intermediate readouts are evidence only; none is selected as a rescue.'),
('public_unpaired_01','Allfour real-mask observations at nativephoto/degradedphoto conditions, R1 and final32 examined. Both produce plausible but smooth lowerface estimates; the paired columns are very similar. A residual edge near the pink-mask chin must be separated from mask-footprint/outside-copy issues. No unavailable hidden target is claimed.'),
('public_unpaired_02','Allfour sunglasses observations examined. Baby estimates are similar in R1/final32. Adult nativephoto gives soft estimated eyes; the degraded case invents glasses/frame-like features in both versions after dark glasses are removed. Final32 is not a clear repair. These public photos have no known hidden eyes.'),
('public_unpaired_03','Allfour glare/mirrored-glasses observations examined. Strong-glare estimates remain soft and can retain/frame-like residual appearance. Mirrored glasses get plausible estimated eyes in both versions; final32 differences are small. Visible frames/context outside support are preserved rather than silently erased.'),
('public_unpaired_04','Hand-over-mask native/degraded and uncovered native/degraded controls all examined. Generated lowerface is smooth and expression varies between the two input conditions, with R1/final32 similar. Both uncovered controls are exact source bypasses, not enhanced images.'),
('public_unpaired_05','Clear-glasses native/degraded and hand-over-eyes native/degraded all examined. Clear-glasses controls are exact bypasses. Hand outputs give plausible eyes but leave an odd nose/transition and retained visible hand portions outside support. Final32 does not materially improve them; do not expand the delivered mask to hide defects.'),
('public_unpaired_06','Gray hand-over-mouth and obstructing-hair native/degraded observations all examined. Generated gray mouth has falsecolor. Hair completion invents a conspicuously blue/large eye with asymmetric appearance relative to the visible eye. Both R1 and final32 retain these problems.'),
('public_unpaired_07','Both scarf examples/native/degraded all examined. Knit-scarf case gives plausible but condition-dependent hidden mouth/beard estimates; no hidden identity truth exists. Scarf-with-gloves case has a conspicuous orange/plastic-looking central lowerface/chin transition in both versions. This inside-support generation defect remains distinct from any residual covering outside fixed support.'),
('public_unpaired_08','Flower-mouth and leaf-eye native/degraded observations all examined. Both versions give plausible smooth mouth estimates for the flower; leaf-eye estimates remain asymmetric and smooth, with a changed estimated face contour. Final32 differences are subtle. Hidden appearance remains unverified.'),
]

def need(ok,label):
    if not ok:raise ValueError(label)
def physical(path):
    p=Path(path).resolve()
    return Path('\\\\?\\'+str(p)) if os.name=='nt' and not str(p).startswith('\\\\?\\') else p
def sha(path):
    h=hashlib.sha256()
    with physical(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def pin(path):return {'path':str(Path(path).resolve()),'bytes':physical(path).stat().st_size,'sha256':sha(path)}
def read(path):return json.loads(physical(path).read_text(encoding='utf-8-sig'))

def main():
    target=BASE/'root_visual_review.json';need(not target.exists(),'Preserve previous review')
    need(sha(BASE/'visual_sheet_index.json')=='c09f368601e54281e52dbe74bb26057ba5cd9c94c11339604e55bd07ddfca2e9','Exact image index')
    need(sha(BASE/'independent_saved_output_audit.json')=='d1c93643925154fbcdb6310614ebb507b51ddc66d54d649b3b3c03774d1496e0','Independent saved-image audit')
    index=read(BASE/'visual_sheet_index.json');pages=[]
    need(len(index['sheets'])==len(NOTES)==20,'All20 sheets')
    for sheet,(stem,note) in zip(index['sheets'],NOTES):
        file=sheet['file'];path=Path(file['path']);need(path.stem==stem and sha(path)==file['sha256'] and physical(path).stat().st_size==file['bytes'],'Reviewed sheet binding')
        need(sheet['all_displayed_cells_exact'] is True,'Exact256 cells')
        pages.append({'sheet':file,'case_ids':[r['id'] for r in sheet['cases']],'all_columns_and_rows_reviewed':True,'inspection':'Root view_image original detail; each1792x1260paired or1280x1260public sheet individually returned, without resampling, enhancements or ranking','observations':note})
    need(sum(len(p['case_ids']) for p in pages)==80 and index['actual_evaluated_PNGs']==256 and index['exact256_cells']==496,'All-case/output/cell coverage')
    report={'format':'root-predictor-ce-v1-complete-all20-sheet-visual-review-v1','complete':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'author':pin(Path(__file__)),'reviewer':'Root development assessment, not independent blinded human FINAL','index_sha256':sha(BASE/'visual_sheet_index.json'),'transport_sha256':sha(BASE/'transport_audit.json'),'sheets_reviewed':20,'cases_reviewed':80,'paired_adaptive_TRAIN_cases':48,'public_previously_exposed_unpaired_cases':32,'all_actual_saved_outputs_covered':256,'all496_exact256_cells_examined':True,'sheet_pages':pages,
            'visual_outcome':'No consistent material final32 improvement over working R1. Lowerface expression, soft eyes, invented eyewear, gray falsecolor and scarf/glove central defects remain. Final32 fails the retained numeric preservation/style requirements and is not adopted. Keep R1 and use the saved code/gradient/loss allocation evidence to distinguish the next design before further training.',
            'input_display_clarification':'Paired observed-input column displays prepared sourceRGB plus a separate reviewed-support overlay. The network removes hiddenRGB under that support; the visible clean photograph display does not mean it received the hidden clean region. No cleanRGB is pasted into generated holes.',
            'mask_boundary_vs_generation':'The exact fixed masks and copied outside pixels are preserved. A covering fragment outside support is a reviewed-mask issue; soft/wrong facial content, falsecolor and central plastic texture inside support are generation defects. No mask is enlarged to conceal them.',
            'only_final32_quality_endpoint':True,'intermediate_8_16_rescue_forbidden':True,'working_R1_unchanged':True,'candidate_adopted':False,'user_acceptance_of_new_model':False,'independent_human_FINAL_pending':True,'model_qualified':False,'goal_complete':False,'local_neural_calls':0,'VM_calls':0}
    with physical(target).open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(pin(target)))

if __name__=='__main__':main()
