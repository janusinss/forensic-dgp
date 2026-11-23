"""Saved-only SAME-codebook representation diagnostic sheets and static report; no model calls."""
from pathlib import Path
from datetime import datetime,timezone
from html import escape
from html.parser import HTMLParser
import argparse,hashlib,json,sys,time,stat,os
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/completion_codeformer_same_codebook_probe_v1_return'
RETURN=BASE/'cctv_completion_codeformer_same_codebook_probe_v1_results'
STAGE=ROOT/'outputs/completion_codeformer_same_codebook_probe_v1_packet/draft_001/cctv_completion_codeformer_same_codebook_probe_v1_vm'
EXPECTED='51782b57fca9ad9cf7161b0f4451a65e545e3eb2f8f58788fa40b64aec48d1ec'
ARCHIVE='e68faa3360b0421c15598a12d472db191395a1790b3b81833f7d611da71577d9'
HEADERS=['Observed input','Known clean TRAIN target','Reviewed removal support','A: ACTUAL predicted / R1 w1','B: predicted / no fusion w0','C: ORACLE clean codes / w1','D: ORACLE clean codes / w0']

def need(ok,label):
    if not ok:raise ValueError(label)
def io_path(path):
    path=Path(path)
    return Path('\\\\?\\'+str(path.resolve())) if os.name=='nt' and not str(path).startswith('\\\\?\\') else path
def sha(path):
    h=hashlib.sha256()
    with io_path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def pin(path):return {'path':str(path.resolve()),'bytes':io_path(path).stat().st_size,'sha256':sha(path)}
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def checkpin(path,row):need(io_path(path).stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Saved evidence drift '+str(path))
def pix(path,mode='RGB'):
    physical=io_path(path);need(physical.is_file() and not physical.is_symlink() and stat.S_ISREG(physical.lstat().st_mode),'Only regular PNG sources')
    with Image.open(physical) as im:
        need(im.format=='PNG' and im.size==(256,256) and im.mode==mode,'Exact native256 source mode required')
        return np.array(im,copy=True)
def pixelpin(array):return {'dtype':str(array.dtype),'shape':list(array.shape),'C_order_SHA256':hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}

def admitted():
    need((BASE/'transport_audit.json').is_file(),'Wait for independent transport admission before reading actual return')
    t=read(BASE/'transport_audit.json')
    need(t.get('complete') is True and t.get('protocol_sha256')==EXPECTED and t.get('archive',{}).get('sha256')==ARCHIVE and t.get('all_payload_manifest_hashes_exact') is True and t.get('closed_namespace_and_safe_paths') is True,'Return transport not admitted')
    need(sha(RETURN/'protocol.json')==EXPECTED and (RETURN/'protocol.json').read_bytes()==(STAGE/'protocol.json').read_bytes(),'Sealed/actual protocol differs')
    p=read(RETURN/'protocol.json');return t,p

def render():
    start=time.monotonic();transport,p=admitted();files={r['path']:r for r in p['files']}
    out=BASE/'visual_sheets';need(not out.exists() and not (BASE/'visual_sheet_index.json').exists(),'Preserve any previous sheets/index');out.mkdir()
    try:font=ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf',15);small=ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf',12)
    except OSError:font=small=ImageFont.load_default()
    inputs={};outputs={};sheets=[];cases=[];cells_total=0;seen=set()
    def source(relative,mode='RGB'):
        need(relative in files,'Unpinned source asset');path=STAGE/relative;checkpin(path,files[relative]);inputs[relative]=pin(path)
        return pix(path,mode),{'source_type':'sealed_source','file':inputs[relative]}
    def result(case,kind,arm):
        suffix='__ORACLE_unavailable_clean_information' if arm['oracle'] else ''
        relative='outputs/'+arm['filename_tag']+'/'+kind+'/'+case['id']+suffix+'.png'
        need(relative in transport['files'],'Planned actual arm output absent');path=RETURN/relative;checkpin(path,transport['files'][relative]);outputs[relative]=pin(path);seen.add(relative)
        return pix(path),{'source_type':'saved_ORACLE_unavailable_clean_information' if arm['oracle'] else 'saved_observed_prediction_control','file':outputs[relative],'arm':arm['id'],'fusion_w':arm['w'],'oracle_unavailable_clean_information':arm['oracle'],'deployable_new_model':False}
    for kind,key,prefix in [('monitor','monitor_cases','original_monitor'),('challenge','challenge_cases','challenge')]:
        rows=p[key];need(len(rows)==(32 if kind=='monitor' else 16),'Fixed cohort count differs')
        for offset in range(0,len(rows),4):
            canvas=Image.new('RGB',(1792,116+286*4),(241,239,232));d=ImageDraw.Draw(canvas)
            title=prefix+' | adaptive TRAIN paired diagnosis | fixed protocol order'
            d.text((8,8),title,fill=(10,20,25),font=font)
            d.text((8,31),'All 497 model tensors frozen. No training. ORACLE C/D uses UNAVAILABLE CLEAN INFORMATION; never an app result.',fill=(120,20,20),font=small)
            d.text((8,50),'Exact saved 256-pixel cells; no enhancement, resampling or selection. Known targets are adaptive TRAIN, not independent FINAL/CCTV.',fill=(40,40,40),font=small)
            for col,text in enumerate(HEADERS):
                d.text((col*256+5,76),text,fill=(10,20,25),font=small)
                if col>=5:d.text((col*256+5,94),'UNAVAILABLE CLEAN INFORMATION',fill=(145,20,20),font=small)
            indexed=[]
            for row,case in enumerate(rows[offset:offset+4]):
                ident=case['id'];observed,op=source(case['input256']);target,tp=source(case['clean256']);mask,mp=source(case['mask'],'L');need(set(np.unique(mask)).issubset({0,255}),'Nonbinary reviewed source mask')
                overlay=observed.copy();inside=mask==255;overlay[inside]=np.rint(overlay[inside].astype(np.float32)*.4+np.array([230,45,45],np.float32)*.6).astype(np.uint8)
                cells=[(observed,op),(target,tp),(overlay,{'source_type':'derived_reviewed_support_overlay','input_file':op['file'],'mask_file':mp['file'],'formula':'Mask255 only round(input*.4+[230,45,45]*.6); outside unchanged'})]
                for arm in p['arms']:cells.append(result(case,kind,arm))
                need(len(cells)==7,'Four arms plus input/target/mask required')
                y=116+286*row;d.text((7,y),ident+' | '+case['condition']+' | '+case['mask_kind'],fill=(10,20,25),font=small)
                detail={'id':ident,'kind':kind,'condition':case['condition'],'mask_kind':case['mask_kind'],'style':case['style'],'teacher_key':case['teacher_key'],'display_cells':[]}
                for col,(array,provenance) in enumerate(cells):
                    need(array.shape==(256,256,3) and array.dtype==np.uint8,'Exact RGB256 cell required');box=[col*256,y+22,(col+1)*256,y+278];canvas.paste(Image.fromarray(array),(box[0],box[1]));cells_total+=1
                    detail['display_cells'].append({'column':col,'label':HEADERS[col],'box_xyxy':box,'pixels':pixelpin(array),**provenance})
                indexed.append(detail);cases.append(detail)
            name=prefix+'_%02d'%(offset//4+1);path=out/(name+'.png');canvas.save(path)
            with Image.open(path) as im:saved=np.array(im)
            for row in indexed:
                for cell in row['display_cells']:
                    x0,y0,x1,y1=cell['box_xyxy'];a=saved[y0:y1,x0:x1];need(pixelpin(a)==cell['pixels'],'Saved sheet cell differs')
            sheets.append({'file':pin(path),'title':title,'headers':HEADERS,'size':list(canvas.size),'cases':indexed,'all_displayed_cells_exact':True,'root_visual_review_pending':True,'ORACLE_unavailable_clean_information_columns':[5,6]})
    actual={path.relative_to(RETURN).as_posix() for arm in p['arms'] for kind in ('monitor','challenge') for path in (RETURN/'outputs'/arm['filename_tag']/kind).glob('*.png')}
    need(actual==seen and len(seen)==192 and len(cases)==48 and len(sheets)==12 and cells_total==336,'Fixed all-case/all-arm coverage differs')
    for row in [*inputs.values(),*outputs.values()]:checkpin(Path(row['path']),row)
    need(not any(name=='torch' or name=='pretrained_completion' or name.startswith('third_party.codeformer') for name in sys.modules),'Modelruntime imported')
    index={'format':'same-codebook-probe-v1-all-saved-visual-index-v1','complete':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'renderer':pin(Path(__file__)),'protocol':pin(RETURN/'protocol.json'),'source_transport_audit':pin(BASE/'transport_audit.json'),'fixed_case_order':True,'output_ranking_or_selection':False,'sheets':sheets,'sheets_count':12,'unique_cases':48,'actual_evaluated_PNGs':192,'ordinary_prediction_outputs':96,'ORACLE_unavailable_clean_information_outputs':96,'all_actual_evaluated_PNGs_displayed':True,'exact256_cells':336,'per_cell_pixel_SHA256_and_coordinates':True,'cases':cases,'source_pins':inputs,'output_pins':outputs,'scope':'Adaptive TRAIN paired zero-optimizer diagnosis only. ORACLE C/D use clean hidden information unavailable at deployment. Not an optimized upper bound, native CCTV, identity recovery, a new model or independent FINAL. Exact copied outside pixels are separate from plausible generated anatomy.','root_visual_review_pending':True,'model_qualified':False,'local_execution':{'model_loads':0,'neural_calls':0,'gradients':0,'optimizer_updates':0,'VM_calls':0},'seconds':time.monotonic()-start}
    write(BASE/'visual_sheet_index.json',index)
    print(json.dumps({'index':pin(BASE/'visual_sheet_index.json'),'sheets':12,'cases':48,'outputs':192,'exact256_cells':336}))

class ConfinedStatic(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.images=0;self.articles=0
    def handle_starttag(self,tag,attrs):
        need(tag not in ('script','iframe','form','input','button','embed','object'),'Active/interactive content refused');d=dict(attrs);need(not any(k.startswith('on') for k in d),'Event handler refused')
        if tag=='img':self.images+=1
        if tag=='article':self.articles+=1
        for key in ('href','src'):
            if key in d:
                value=d[key];need(value and not any(x in value for x in (':','\\','..','?','#')) and not value.startswith('/'),'Unconfined local report link')
                path=BASE/value;need(path.is_file() and path.resolve().is_relative_to(BASE.resolve()),'Missing/escaped report link');self.links.append(value)

def html_review():
    admitted();required=['visual_sheet_index.json','transport_audit.json','independent_saved_output_audit.json','independent_state_code_audit.json','root_visual_review.json','root_endpoint_and_preservation_closure.json']
    reports={name:read(BASE/name) for name in required};need(all(p.get('complete') is True for p in reports.values()),'Wait for complete independent pixel/state-invariance and root visual reviews')
    index=reports['visual_sheet_index.json'];audit=reports['independent_saved_output_audit.json'];visual=reports['root_visual_review.json']
    need(index['unique_cases']==48 and index['actual_evaluated_PNGs']==192 and index['sheets_count']==12,'All-case index scope differs')
    need(sha(BASE/'root_visual_review.json')=='3d2568cfd19614f7d8b98da71b027d471fc32c27753cee017ed512b394931a53' and sha(BASE/'root_endpoint_and_preservation_closure.json')=='7d2a94a61a50d9fbc5d4fae0ec07c95fafe807b12382c288f274cfa4d6802160','Root visual/closure drift')
    need(visual['index_sha256']==sha(BASE/'visual_sheet_index.json') and visual['sheets_reviewed']==12 and visual['cases_reviewed']==48 and visual['all_actual_saved_outputs_covered']==192,'All12sheet root review required')
    need(audit.get('independent') is True and reports['independent_state_code_audit.json'].get('independent') is True,'Independent output and state/code-invariance audits required')
    need(sha(BASE/'independent_saved_output_audit.json')=='4f5042edc19496bd3dee515dc052928c03e4f6d9e12b246efdcca4262a54d02f' and audit['evaluated_outputs']==192 and audit['oracle_outputs_explicitly_labeled']==96,'Independent all-output audit scope differs')
    need(sha(BASE/'independent_state_code_audit.json')=='794fed73804eabfa081805cdd0d3db3dda89d5a1fbd6138df725b6b404ad922d','Independent state/code audit drift')
    nav=''.join('<a href="'+name+'">'+label+'</a>' for name,label in [('root_visual_review.json','Root all-case visual review'),('root_endpoint_and_preservation_closure.json','Root endpoint/preservation closure'),('independent_saved_output_audit.json','Independent saved-output audit'),('independent_state_code_audit.json','Independent frozen-state/code audit'),('transport_audit.json','Return integrity'),('visual_sheet_index.json','All image and cell pins'),('visual_index_coordinate_recheck_full.json','Exact-cell recheck')])
    sections=[]
    for prefix,title in [('original_monitor_','32 original clear adaptive TRAIN cases'),('challenge_','16 synthetic adaptive TRAIN challenges')]:
        articles=[]
        for sheet in index['sheets']:
            p=Path(sheet['file']['path']);checkpin(p,sheet['file'])
            if not p.stem.startswith(prefix):continue
            relative=p.relative_to(BASE).as_posix();ids=', '.join(r['id'] for r in sheet['cases'])
            articles.append('<article><h3>'+escape(p.stem)+'</h3><p>'+escape(ids)+'</p><p class="oracle">C/D: ORACLE — UNAVAILABLE CLEAN INFORMATION. These images cannot be produced from the covered input alone.</p><a href="'+relative+'" target="_blank"><img src="'+relative+'" loading="lazy" alt="'+escape(sheet['title'],quote=True)+'"></a><p class="hint">Click to inspect exact original 256 pixel cells. The embedded thumbnail is navigation only.</p></article>')
        sections.append('<section><h2>'+title+'</h2>'+''.join(articles)+'</section>')
    numeric_rows=[]
    labels={'B_vs_A':'B observed/no-fusion vs A observed/R1','C_vs_A':'ORACLE C clean-codes/w1 vs A observed/R1','D_vs_B':'ORACLE D clean-codes/w0 vs B observed/w0','D_vs_A':'ORACLE D clean-codes/w0 vs A observed/R1','D_vs_C':'ORACLE D no-fusion vs ORACLE C R1-fusion','C_vs_B':'ORACLE C clean-codes/w1 vs B observed/w0'}
    for key,cohorts in audit['independent_comparisons'].items():
        values=[labels[key]]
        for kind in ('monitor','challenge'):
            row=cohorts[kind];value=row['PNG_macro_hole_gain_percent'];values.append(('+' if value>=0 else '')+format(value,'.4f')+'%')
        values.append(' / '.join('nonworse' if cohorts[k]['all_raw_PNG_hole_boundary_groups_nonworse'] else str(len(cohorts[k]['nonworse_failures']))+' group failures' for k in ('monitor','challenge')))
        numeric_rows.append('<tr>'+''.join('<td>'+escape(v)+'</td>' for v in values)+'</tr>')
    numeric='<h2>Separate saved-output diagnostic comparisons</h2><p>Positive percentages mean lower delivered PNG hole-L1 relative to the named baseline. These are adaptive TRAIN measurements. Oracle gains require unavailable clean information; they are not gains from a new trained model. Raw/PNG hole and boundary requirements remain binding, so a favorable hole mean does not erase subgroup regressions.</p><table><thead><tr><th>Comparison</th><th>32 clear cases: PNG hole gain</th><th>16 challenges: PNG hole gain</th><th>Raw/PNG group check: clear / challenge</th></tr></thead><tbody>'+''.join(numeric_rows)+'</tbody></table>'
    outcome=escape(str(visual.get('visual_outcome',visual.get('decision','See the full root visual review.'))))
    text="""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self' file:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"><title>SAME-codebook probe: saved diagnostic comparisons</title><style>body{margin:0;background:#f3f1eb;color:#172027;font:16px/1.5 system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:28px}h1,h2,h3{line-height:1.2}h2{margin-top:36px}.status{background:#fff3d9;border:1px solid #b67925;padding:18px}.oracle{color:#812323;font-weight:600}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #bac2c7;padding:12px;text-align:left}article{margin:22px 0 34px;padding:18px;background:#fff;border:1px solid #c6cbc8;border-radius:8px}img{display:block;max-width:100%;height:auto}.hint{font-size:13px;color:#4d5960}nav{display:flex;flex-wrap:wrap;gap:14px}a{color:#124e7a}</style></head><body><main><h1>SAME-codebook probe: all saved comparisons</h1><p>Read-only local evidence · FACE_MASK_COMPLETION research</p><div class="status"><strong>This is a zero-optimizer representation diagnostic. No new model was trained or adopted.</strong><p>All 497 model tensors remain frozen. Worker exit 0 means the diagnostic finished, regardless of comparison scores. Oracle results are not deployable completion or recovered identity.</p><p>Root visual review: __OUTCOME__</p></div><h2>Keep ordinary predictions and oracle controls separate</h2><table><thead><tr><th>Arm</th><th>Codes and fusion</th><th>Interpretation</th></tr></thead><tbody><tr><td>A</td><td>Actual covered-input predicted codes; R1 fusion w1</td><td>Retained trained R1 ordinary baseline.</td></tr><tr><td>B</td><td>Same actual predicted codes; fusion w0</td><td>No-fusion diagnostic control; not an app replacement.</td></tr><tr><td>C — ORACLE</td><td>Retained SAME-codebook clean-context codes; fusion w1</td><td>UNAVAILABLE CLEAN INFORMATION. Uses codes derived from clean hidden context.</td></tr><tr><td>D — ORACLE</td><td>Same clean-context oracle codes; fusion w0</td><td>UNAVAILABLE CLEAN INFORMATION. Neither a deployable output nor mathematical upper bound.</td></tr></tbody></table><p>The exact same observed RGB and reviewed mask are used across arms. Oracle arms replace selected code indices only; clean RGB is never pasted into observed context. The already retained labels come from the same original inpainting codebook, not a substituted HQ512 prior. Independent review checks predicted/used indices, copied outside pixels, PNG/raw values and frozen-state invariance separately from plausible hidden anatomy.</p><nav>__NAV__</nav>__NUMERIC__<p>All 192 actual outputs across 48 fixed cases appear in 12 complete sheets. Each of 336 cells has exact 256×256 saved pixels, a pixel hash and coordinates in the linked index. Input, clean target and mask overlay accompany all four arms. The sheets use no image enhancement, resampling, ranking or omitted failures.</p>__SECTIONS__<h2>Evidence limits</h2><p>The paired cohorts are adaptive original TRAIN, not independent FINAL or native CCTV. No public unpaired hidden references are used. Root all-case visual review is development assessment, not independent human FINAL review or user acceptance. A result from unavailable clean information cannot demonstrate real covered-input recovery. Exact visible-pixel copying does not establish plausible generated anatomy, identity, all-covering-family usefulness or Zamboanga performance. The working trained R1/app remains unchanged. This page runs no model or network requests.</p></main></body></html>
"""
    text=text.replace('__NUMERIC__',numeric).replace('__OUTCOME__',outcome).replace('__NAV__',nav).replace('__SECTIONS__',''.join(sections))
    parser=ConfinedStatic();parser.feed(text);need(parser.images==12 and parser.articles==12 and len(parser.links)==31,'All12sheet report coverage differs')
    path=BASE/'review.html';need(not path.exists(),'Preserve prior report');
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    report={'format':'same-codebook-probe-v1-readonly-HTML-admission-v1','complete':True,'HTML':pin(path),'builder':pin(Path(__file__)),'input_reports':{name:pin(BASE/name) for name in required},'sheets_linked':12,'fixed_cases':48,'actual_outputs':192,'exact256_cells':336,'confined_existing_links':len(parser.links),'JS_forms_external_network':False,'new_model_created':False,'model_qualified':False,'neural_calls':0,'VM_calls':0,'app_changes':False}
    write(BASE/'review_html_admission.json',report);print(json.dumps({'HTML':pin(path),'admission':pin(BASE/'review_html_admission.json')}))

def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--render',action='store_true');g.add_argument('--html',action='store_true');a=p.parse_args();html_review() if a.html else render()
if __name__=='__main__':main()
