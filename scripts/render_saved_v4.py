"""Render every fixed saved V4 evaluation PNG and historical controls; no model inference."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,time,stat
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'outputs/completion_codeformer_delivered_objective_v4_return'
RETURN=PARENT/'cctv_completion_codeformer_delivered_objective_v4_results'
STAGE=ROOT/'outputs/completion_codeformer_delivered_objective_v4_packet/draft_001/cctv_completion_codeformer_delivered_objective_v4_vm'
V3_PARENT=ROOT/'outputs/completion_codeformer_fusion_only_v3_return'
V3=V3_PARENT/'cctv_completion_codeformer_fusion_only_v3_results'
EXPECTED='8bbb3dd611e95cfbce76e7e6426ffdadb31ce23dadb1dfdcd0246e501213069d'
ARCHIVE_SHA='c5a40b0aa8473a3d4e43f5750e8c301ff42deed17494b3c4bf1bdfcaff828f38'
V3_ROOT_VISUAL='44d478963a56a16f09f6f6050257957132d6bba98494e3c352f5d72ccba90f24'
V3_TRANSPORT='fffe0ea070487585ee61e19e545595a2a2858295778a7603551a99f9eb0cec56'

def need(ok,label):
    if not ok:raise ValueError(label)

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def pin(path):return {'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':sha(path)}
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def pix(path,mode='RGB'):
    need(path.is_file() and not path.is_symlink() and stat.S_ISREG(path.lstat().st_mode),'Expected regular PNG')
    with Image.open(path) as im:
        need(im.format=='PNG' and im.size==(256,256),'Expected exact saved256 PNG')
        return np.array(im.convert(mode),copy=True)
def pixel_pin(array):return {'dtype':str(array.dtype),'shape':list(array.shape),'C_order_SHA256':hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}
def same(a,b,label):need(a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b),label)

def main():
    started=time.monotonic();transport_path=PARENT/'transport_audit.json'
    need(transport_path.is_file(),'Wait for root transport receipt before reading actual return')
    transport=read(transport_path)
    need(transport.get('complete') is True and transport.get('protocol_sha256')==EXPECTED and transport.get('archive',{}).get('sha256')==ARCHIVE_SHA and transport.get('all_payload_manifest_hashes_exact') is True and transport.get('closed_namespace_and_safe_paths') is True,'Actual return transport not admitted')
    need(sha(RETURN/'protocol.json')==EXPECTED and (RETURN/'protocol.json').read_bytes()==(STAGE/'protocol.json').read_bytes(),'Actual/staged protocol differs')
    p=read(RETURN/'protocol.json');files={r['path']:r for r in p['files']};need(len(files)==846,'V4 payload count differs')
    need(sha(V3_PARENT/'root_visual_review.json')==V3_ROOT_VISUAL and sha(V3_PARENT/'transport_audit.json')==V3_TRANSPORT,'Historical V3 evidence drift')
    v3visual=read(V3_PARENT/'root_visual_review.json');need(sha(V3_PARENT/'visual_sheet_index.json')==v3visual['index_sha256'],'Historical V3 index drift')
    v3index=read(V3_PARENT/'visual_sheet_index.json');v3transport=read(V3_PARENT/'transport_audit.json')
    out=PARENT/'visual_sheets';need(not out.exists() and not (PARENT/'visual_sheet_index.json').exists(),'Preserve prior V4 render');out.mkdir()
    try:font=ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf',15);small=ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf',12)
    except OSError:font=small=ImageFont.load_default()
    sourcepins={};outputpins={};historicalpins={};available=set();sheetindex=[];allrows=[];cellcount=0
    def verify_pin(path,row):need(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Pinned image drift '+str(path))
    def source(relative,mode='RGB'):
        need(relative in files,'Unpinned stage source '+relative);path=STAGE/relative;verify_pin(path,files[relative]);sourcepins[relative]=pin(path)
        return pix(path,mode),{'source_type':'sealed_stage','file':sourcepins[relative]}
    def actual(kind,cycle,ident):
        relative='outputs/u%04d/%s/%s.png'%(cycle,kind,ident);path=RETURN/relative
        need(relative in transport['files'],'Planned output missing from transport '+relative);verify_pin(path,transport['files'][relative]);outputpins[relative]=pin(path);available.add(relative)
        return pix(path),{'source_type':'actual_V4_saved_PNG','cycle':cycle,'file':outputpins[relative]}
    def historical_v3(kind,ident):
        relative='outputs/u0064/'+kind+'/'+ident+'.png';path=V3/relative
        need(relative in v3transport['files'],'V3 terminal PNG missing');verify_pin(path,v3transport['files'][relative]);historicalpins[str(path)]=pin(path)
        return pix(path),{'source_type':'historical_V3_terminal_saved_control','file':historicalpins[str(path)]}
    def original(ident):
        path=ROOT/'outputs/completion_codeformer_finetune_v1_r1_return/cctv_completion_codeformer_finetune_v1_r1_results/outputs/u0000/monitor'/(ident+'.png')
        row=v3index['output_pins'].get(str(path));need(row is not None,'Original CodeFormer output unpinned');verify_pin(path,row);historicalpins[str(path)]=pin(path)
        return pix(path),{'source_type':'historical_original_CodeFormer','file':historicalpins[str(path)]}
    def placeholder():
        im=Image.new('RGB',(256,256),(220,216,208));d=ImageDraw.Draw(im)
        for j,line in enumerate(('UNPAIRED photograph','No aligned clean reference','No hidden-identity metric')):d.text((12,65+22*j),line,fill=(60,55,50),font=small)
        arr=np.array(im);return arr,{'source_type':'unpaired_target_placeholder','no_clean_target':True}
    def render(name,title,headers,rows):
        nonlocal cellcount
        canvas=Image.new('RGB',(len(headers)*256,82+286*len(rows)),(241,239,232));d=ImageDraw.Draw(canvas)
        d.text((8,8),title,fill=(10,20,25),font=font)
        d.text((8,31),'Exact256 saved pixels. Historical controls are separate. Quality is pending independent and all-case review.',fill=(40,40,40),font=small)
        for col,header in enumerate(headers):d.text((col*256+5,56),header,fill=(10,20,25),font=small)
        indexed=[]
        for row,(label,cells,detail) in enumerate(rows):
            need(len(cells)==len(headers),'Header/cell column mismatch');y=82+286*row;d.text((7,y),label,fill=(10,20,25),font=small);detail['display_cells']=[]
            for col,(array,provenance) in enumerate(cells):
                need(array.dtype==np.uint8 and array.shape==(256,256,3),'Invalid cell array');box=[col*256,y+22,(col+1)*256,y+278];canvas.paste(Image.fromarray(array),(box[0],box[1]));cellcount+=1
                detail['display_cells'].append({'column':col,'label':headers[col],'box_xyxy':box,'pixels':pixel_pin(array),**provenance})
            indexed.append(detail)
        path=out/(name+'.png');canvas.save(path)
        with Image.open(path) as im:saved=np.array(im.convert('RGB'))
        for row,(_,cells,detail) in enumerate(rows):
            for col,(array,_) in enumerate(cells):
                x0,y0,x1,y1=detail['display_cells'][col]['box_xyxy'];same(saved[y0:y1,x0:x1],array,'Rendered cell changed saved pixels')
        sheetindex.append({'file':pin(path),'title':title,'headers':headers,'size':list(canvas.size),'cases':indexed,'all_displayed_cells_exact':True,'root_visual_review_pending':True})
    cohort_specs=[('monitor','monitor_cases',[0,16,32,64],'original_monitor'),('challenge','challenge_cases',[0,64],'challenge'),('public_development','development_cases',[0,64],'public_DEV')]
    expected_outputs=set()
    for kind,key,cycles,label in cohort_specs:
        cases=p[key];need(len(cases)=={'monitor':32,'challenge':16,'public_development':32}[kind],'Fixed cohort size differs')
        headers=['Known clean target' if kind!='public_development' else 'Unpaired; no clean target','Observed context','Reviewed support (red)']
        if kind=='monitor':headers+=['Historical original CodeFormer']
        headers+=['R1 frozen u0','Historical joint V2 u64','Historical V3 fusion u64']+['V4 delivered256 u%d'%c for c in cycles if c!=0]
        for offset in range(0,len(cases),4):
            rows=[]
            for case in cases[offset:offset+4]:
                ident=case['id'];observed,op=source(case['input256']);mask8,mp=source(case['mask'],'L');need(set(np.unique(mask8)).issubset({0,255}),'Nonbinary reviewed support');mask=mask8==255
                overlay=observed.copy();overlay[mask]=np.rint(overlay[mask].astype(np.float32)*.4+np.array([230,45,45],np.float32)*.6).astype(np.uint8)
                target,tp=source(case['clean256']) if kind!='public_development' else placeholder()
                cells=[(target,tp),(observed,op),(overlay,{'source_type':'derived_mask_overlay','input_file':op['file'],'mask_file':mp['file'],'overlay':'Inside255 support: round(input*.4+[230,45,45]*.6); outside exactsource'})]
                if kind=='monitor':cells.append(original(ident))
                row={'id':ident,'kind':kind,'condition':case.get('condition'),'mask_kind':case.get('mask_kind'),'family':case.get('family'),'available_snapshots':cycles,'missing_snapshots':[],'all_snapshot_output_pins':[]}
                for cycle in cycles:expected_outputs.add('outputs/u%04d/%s/%s.png'%(cycle,kind,ident))
                arr,meta=actual(kind,0,ident);cells.append((arr,meta));row['all_snapshot_output_pins'].append(meta['file'])
                joint=next((r for r in p['historical_control']['joint_rows'][kind] if r['id']==ident),None);need(joint is not None,'Missing historical V2 case');arr,meta=source(joint['PNG']);meta['source_type']='historical_joint_V2_terminal_saved_control';cells.append((arr,meta))
                cells.append(historical_v3(kind,ident))
                for cycle in cycles:
                    if cycle==0:continue
                    arr,meta=actual(kind,cycle,ident);cells.append((arr,meta));row['all_snapshot_output_pins'].append(meta['file'])
                rows.append((ident+' | '+str(case.get('condition') or case.get('mask_kind') or case.get('family')),cells,row));allrows.append(row)
            context='adaptive TRAIN known-target cohort' if kind!='public_development' else 'unpaired public photographs; native suffix is not CCTV'
            render(label+'_%02d'%(offset//4+1),label+' | '+context+' | fixed protocol order',headers,rows)
    actualpng={q.relative_to(RETURN).as_posix() for folder in (RETURN/'outputs').glob('u[0-9][0-9][0-9][0-9]') for kind in ('monitor','challenge','public_development') for q in (folder/kind).glob('*.png')}
    need(actualpng==available==expected_outputs and len(available)==224,'Actual evaluation output omitted or unexpected');need(len(sheetindex)==20 and len(allrows)==80 and cellcount==656,'Complete sheet/case/cell counts differ')
    for row in [*sourcepins.values(),*outputpins.values(),*historicalpins.values()]:verify_pin(Path(row['path']),row)
    need(not any(n=='torch' or n=='pretrained_completion' or n.startswith('third_party.codeformer') for n in sys.modules),'Modelruntime imported')
    report={'format':'completion-delivered-objective-v4-all-saved-visual-index-v1','complete':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'renderer':pin(Path(__file__)),'protocol':pin(RETURN/'protocol.json'),'source_transport_audit':pin(transport_path),'historical_V3_root_visual_review':pin(V3_PARENT/'root_visual_review.json'),'historical_V3_transport_audit':pin(V3_PARENT/'transport_audit.json'),'historical_V3_visual_index':pin(V3_PARENT/'visual_sheet_index.json'),'fixed_case_order':True,'output_ranking_or_selection':False,'sheets':sheetindex,'sheets_count':len(sheetindex),'unique_cases':len(allrows),'available_evaluated_PNGs':len(available),'all_available_evaluated_PNGs_displayed':True,'displayed_cells':cellcount,'per_cell_pixel_SHA256_and_exact_coordinates':True,'cases':allrows,'source_pins':sourcepins,'output_pins':outputpins,'historical_pins':historicalpins,'historical_V2_and_V3_are_saved_controls_only':True,'no_historical_checkpoint_retraining_or_adoption_implied':True,'no_saved_snapshot_implied_complete_training':True,'limits':'Exact displayed pixels and complete coverage only; independent raw/state audit establishes outside preservation and training status. All paired cohorts are adaptive TRAIN, not independent FINAL. Public photographs are unpaired, not native CCTV; native suffix is original photograph condition. No identity or Zamboanga performance claim.','root_visual_review_pending':True,'quality_qualified':False,'local_execution':{'model_loads':0,'neural_calls':0,'gradients':0,'optimizer_updates':0,'VM_calls':0},'seconds':time.monotonic()-started}
    path=PARENT/'visual_sheet_index.json'
    with path.open('x',encoding='utf-8',newline='\n') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'index':pin(path),'sheets':len(sheetindex),'cases':len(allrows),'evaluatedPNGs':len(available),'exact256_cells':cellcount,'seconds':report['seconds']}))
if __name__=='__main__':main()
