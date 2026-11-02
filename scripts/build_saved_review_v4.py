"""Build a read-only, saved-evidence-only V4 report after all required audits and visual review."""
from pathlib import Path
from datetime import datetime,timezone
from html import escape
from html.parser import HTMLParser
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'outputs/completion_codeformer_delivered_objective_v4_return'
EXPECTED='8bbb3dd611e95cfbce76e7e6426ffdadb31ce23dadb1dfdcd0246e501213069d'
INDEX='a14fffd642141e51743c5a61476e6e0e59860f80c9566169da22c6af65a5dc31'
ROOT_VISUAL='4b127549313b7bc800505d8f2e647ee06802df9a3f98682c3fd41d2c1a748cc3'
TRANSPORT='187be2e99a5439528288ccbdf3d196752b14dd3e8262429046ec9b1958088570'

def need(ok,label):
    if not ok:raise ValueError(label)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def pin(path):return {'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':sha(path)}
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def pct(value):return ('+' if value>=0 else '')+format(value*100,'.4f')+'%'

def main():
    required=['transport_audit.json','visual_sheet_index.json','visual_index_coordinate_recheck_full.json','independent_saved_output_audit.json','independent_state_audit.json','root_visual_review.json']
    docs={name:read(PARENT/name) for name in required};need(all(v.get('complete') is True for v in docs.values()),'Required audit/review incomplete')
    index=docs['visual_sheet_index.json'];pixels=docs['independent_saved_output_audit.json'];state=docs['independent_state_audit.json'];visual=docs['root_visual_review.json'];transport=docs['transport_audit.json']
    need(sha(PARENT/'visual_sheet_index.json')==INDEX and index['protocol']['sha256']==EXPECTED and transport['protocol_sha256']==EXPECTED and pixels['protocol']['sha256']==EXPECTED and state['protocol_sha256']==EXPECTED,'Protocol/index audit scope differs')
    need(pixels['independent'] and state['independent'] and pixels['evaluated_raw_PNG_outputs']==224,'Independent all-output audits required')
    need(sha(PARENT/'root_visual_review.json')==ROOT_VISUAL and sha(PARENT/'transport_audit.json')==TRANSPORT,'Root review or transport receipt drift')
    need(visual['index_sha256']==INDEX and visual['sheets_reviewed']==20 and visual['cases_reviewed']==80 and visual['all_actual_saved_outputs_covered']==224,'Root all20sheet review required')
    need(index['sheets_count']==20 and index['unique_cases']==80 and index['available_evaluated_PNGs']==224 and index['displayed_cells']==656,'Complete visual coverage required')
    clear=pixels['clear_incremental_vs_R1'];challenge=pixels['challenge_incremental_vs_R1'];cumulative=pixels['clear_cumulative_vs_original']
    failed=not pixels['all_numeric_screen_endpoints_pass_if_available'];need(failed and not state['model_qualified'] and not pixels['model_qualified'],'This V4 report is defined for the retained failed-quality return')
    supervisor=pixels['supervisor'];need(supervisor['worker_exit']==2 and supervisor['timeout'] is False,'Expected quality-stop status differs')
    valid=state['state_status']['valid_new_recovery_checkpoint'];need(valid,'Recovery-state status differs; revise report explicitly')
    outside=pixels['outside_source_FP32_and_PNG_all_exact'];need(outside and pixels['saved_empty_output_bypasses_exact'],'Outside/bypass evidence differs; revise report explicitly')
    sections=[]
    for prefix,title in [('original_monitor_','Original clear TRAIN monitor  -  paired'),('challenge_','Synthetic TRAIN challenges  -  paired'),('public_DEV_','Public development photographs  -  unpaired')]:
        articles=[]
        for sheet in index['sheets']:
            path=Path(sheet['file']['path']);need(pin(path)==sheet['file'],'Sheet drift before review')
            if not path.stem.startswith(prefix):continue
            relative=path.relative_to(PARENT).as_posix();ids=', '.join(row['id'] for row in sheet['cases'])
            articles.append('<article><h3>'+escape(path.stem)+'</h3><p>'+escape(ids)+'</p><a href="'+relative+'" target="_blank"><img src="'+relative+'" loading="lazy" alt="'+escape(sheet['title'],quote=True)+'"></a><p class="hint">Click to inspect original-resolution256pixel cells. The embedded image is a navigation thumbnail.</p></article>')
        sections.append('<section><h2>'+title+'</h2>'+''.join(articles)+'</section>')
    rows=[('32 original clear adaptive TRAIN monitor cases',pct(clear['PNG_macro_hole_gain']),'Below the required 1% gain; declared raw/PNG subgroup requirements also fail.'),('16 synthetic gray/irregular/degraded adaptive TRAIN challenges',pct(challenge['PNG_macro_hole_gain']),'Worse on average than R1; declared raw/PNG subgroup requirements fail.'),('Same 32 clear cases versus original CodeFormer',pct(cumulative['PNG_macro_hole_gain']),'Different cumulative endpoint; it cannot replace incremental improvement versus R1.'),('32 public development photographs','Unpaired visual evidence only','No aligned hidden clean reference; no hidden-identity, PSNR/SSIM or native-CCTV claim.')]
    tablerows=''.join('<tr>'+''.join('<td>'+escape(c)+'</td>' for c in row)+'</tr>' for row in rows)
    nav=''.join('<a href="'+name+'">'+label+'</a>' for name,label in [('root_visual_review.json','Root all-case visual review'),('independent_saved_output_audit.json','Independent saved-output audit'),('independent_state_audit.json','Independent state audit'),('transport_audit.json','Return integrity'),('visual_sheet_index.json','All sheet and cell pins'),('visual_index_coordinate_recheck_full.json','Exact-cell recheck')])
    html="""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self' file:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"><title>Completion V4: all saved comparisons</title><style>body{margin:0;background:#f3f1eb;color:#172027;font:16px/1.5 system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:28px}h1,h2,h3{line-height:1.2}h1{font-size:30px}h2{margin-top:38px}.status{background:#fff3d9;border:1px solid #b67925;padding:18px;border-radius:8px}.note{background:#e8eef1;padding:16px;border-radius:8px}table{border-collapse:collapse;width:100%;background:#fff}td,th{border:1px solid #bac2c7;text-align:left;padding:12px}a{color:#124e7a}article{margin:22px 0 34px;padding:18px;background:#fff;border:1px solid #c6cbc8;border-radius:8px}img{display:block;max-width:100%;height:auto;background:#dedbd3}.hint{font-size:13px;color:#4d5960}code{overflow-wrap:anywhere}nav{display:flex;flex-wrap:wrap;gap:14px}</style></head><body><main><h1>Completion delivered-objective V4: all retained results</h1><p>26 October 2026  |  FACE_MASK_COMPLETION research  |  Read-only local report</p><div class="status"><strong>V4 did not qualify. Trained R1 remains the working completion baseline.</strong><p>The finite run completed all 64 fusion cycles and committed a valid terminal recovery state. It returned quality-stop code 2, without timeout. More completed updates or a usable saved state do not establish a better model. Root visual review: __ROOT_VISUAL_OUTCOME__ The original app and all historical checkpoints remain unchanged.</p></div><h2>Separate outcomes</h2><table><thead><tr><th>Evidence cohort</th><th>Delivered PNG hole-L1 gain</th><th>Implication</th></tr></thead><tbody>__ROWS__</tbody></table><p class="note">V4 starts from full trained R1. It changes the learning objective to the app-delivered float256 hole-only RGB L1 against exact existing clean256 targets, removing the separate additive boundary loss while retaining every boundary acceptance gate. Historical joint V2 u64 and V3 fusion u64 are saved comparison controls; neither supplies a V4 parent or qualifies app adoption. Clamping diagnostics and the delivered256 loss are checked separately from visible image usefulness.</p><nav>__NAV__</nav><p>All 224 actual V4 evaluation PNGs appear across 20 complete sheets and 80 fixed cases. The 656 image cells have exact256 x 256pixels, file hashes, per-cell pixel hashes and coordinates in the linked index. Clear monitor sheets show target, observed context, reviewed support, original CodeFormer, R1, historical joint V2 u64, historical V3 u64 and V4 u16/u32/u64. Challenge and public sheets show R1 and terminal controls/results. No ranking, sharpening, resampling or omitted failures is used in the sheets.</p>__SECTIONS__<h2>Review limits</h2><p>Root inspected all 20 original-resolution sheets and every planned case/output. That is development visual review, not independent human FINAL review or user acceptance. The paired cohorts remain adaptive original TRAIN; public photographs remain unpaired. The filename suffix native means original photograph condition, not native CCTV. No ethnicity, Zamboanga performance or exact hidden-identity claim follows from this evidence.</p><p>Visible/source pixels outside the reviewed support and empty bypass controls passed independent exactness checks. Copied covering fragments outside a reviewed mask must be distinguished from generated defects inside it. Automatic mask proposals, deployment to other devices and usefulness across every covering family remain separate unfinished requirements. This page displays saved evidence only; it runs no model and changes no application.</p></main></body></html>
"""
    html=html.replace('__ROOT_VISUAL_OUTCOME__',escape(visual['visual_outcome'])).replace('__ROWS__',tablerows).replace('__NAV__',nav).replace('__SECTIONS__',''.join(sections))
    class StaticCheck(HTMLParser):
        def __init__(self):super().__init__();self.links=[];self.images=0;self.articles=0
        def handle_starttag(self,tag,attrs):
            need(tag not in ('script','iframe','form','input','button','object','embed'),'Interactive or active content refused');d=dict(attrs);need(not any(key.startswith('on') for key in d),'Eventhandler refused')
            if tag=='img':self.images+=1
            if tag=='article':self.articles+=1
            for attr in ('href','src'):
                if attr in d:
                    value=d[attr];need(value and not any(s in value for s in (':','\\','..','?','#')) and not value.startswith('/'),'Unconfined report link')
                    path=PARENT/value;need(path.is_file() and path.resolve().is_relative_to(PARENT.resolve()),'Missing or escaped local report link');self.links.append(value)
    parser=StaticCheck();parser.feed(html);need(parser.images==20 and parser.articles==20 and len(parser.links)==46,'Report coverage or navigation differs')
    need(not any(n=='torch' or n=='pretrained_completion' for n in sys.modules),'Modelruntime imported')
    output=PARENT/'review.html';need(not output.exists(),'Preserve prior review HTML')
    with output.open('x',encoding='utf-8',newline='\n') as f:f.write(html)
    report={'format':'completion-delivered-objective-v4-readonly-HTML-admission-v1','complete':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'HTML':pin(output),'builder':pin(Path(__file__)),'input_reports':{name:pin(PARENT/name) for name in required},'sheets_linked':20,'fixed_cases':80,'actual_V4_outputs':224,'exact256_cells':656,'links_confined_existing':len(parser.links),'script_form_external_network_links':0,'neural_calls':0,'VM_calls':0,'app_changes':False,'model_qualified':False,'goal_complete':False}
    with (PARENT/'review_html_admission.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'HTML':pin(output),'admission':pin(PARENT/'review_html_admission.json')}))
if __name__=='__main__':main()
