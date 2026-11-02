"""Render every available saved V3 evaluated case with historical V2 control. No model/runtime/training."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
RETURN=ROOT/'outputs/completion_codeformer_fusion_only_v3_return/cctv_completion_codeformer_fusion_only_v3_results'
PARENT=RETURN.parent
STAGE=ROOT/'outputs/completion_codeformer_fusion_only_v3_packet/draft_002/cctv_completion_codeformer_fusion_only_v3_vm'
OLD=ROOT/'outputs/completion_codeformer_finetune_v1_r1_return/cctv_completion_codeformer_finetune_v1_r1_results'
EXPECTED='3dc71dac3002e82587cbc2cb758f36df604ed4c7131a3e592db4d8dcb5c18033'
def need(ok,label):
    if not ok:raise ValueError(label)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def pin(p):return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p)}
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def pix(path,mode='RGB'):
    with Image.open(path) as im:need(im.format=='PNG' and im.size==(256,256),'Expected exact256 PNG');return np.array(im.convert(mode),copy=True)
def same(a,b,label):need(a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b),label)
def main():
    started=time.monotonic();need(RETURN.is_dir(),'Transport extraction not ready');need(sha(RETURN/'protocol.json')==EXPECTED,'Wrong V3 protocol');p=read(RETURN/'protocol.json')
    need((RETURN/'protocol.json').read_bytes()==(STAGE/'protocol.json').read_bytes(),'Stagedprotocol differs')
    transport=read(PARENT/'transport_audit.json');need(transport.get('complete') is True,'Transportaudit notcomplete')
    out=PARENT/'visual_sheets';need(not out.exists(),'Preserve prior sheets');out.mkdir()
    try:font=ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf',15);small=ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf',12)
    except OSError:font=small=ImageFont.load_default()
    placeholder_cache={}
    def placeholder(label):
        if label not in placeholder_cache:
            im=Image.new('RGB',(256,256),(220,216,208));d=ImageDraw.Draw(im)
            for i,line in enumerate(label.split('|')):d.text((12,65+22*i),line,fill=(60,55,50),font=small)
            placeholder_cache[label]=np.array(im)
        return placeholder_cache[label]
    files={row['path']:row for row in p['files']};sourcepins={};outputpins={};sheetindex=[];allrows=[];available=set();totalcells=0
    def source(relative,mode='RGB'):
        path=STAGE/relative;row=files[relative]
        if relative not in sourcepins:need(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Pinnedinput drift');sourcepins[relative]=pin(path)
        return pix(path,mode)
    def saved(path):
        if not path.exists():return None
        if str(path) not in outputpins:outputpins[str(path)]=pin(path)
        return pix(path)
    def output(kind,cycle,ident):
        path=RETURN/'outputs'/('u%04d'%cycle)/kind/(ident+'.png');arr=saved(path)
        if arr is not None:available.add(path.relative_to(RETURN).as_posix())
        return arr
    def render(name,title,headers,rows):
        nonlocal totalcells
        canvas=Image.new('RGB',(len(headers)*256,82+286*len(rows)),(241,239,232));draw=ImageDraw.Draw(canvas)
        draw.text((8,8),title,fill=(10,20,25),font=font)
        draw.text((8,31),'Exact saved256 pixels; no sharpening/resampling. Missing snapshots are explicitly marked. All candidates unqualified.',fill=(40,40,40),font=small)
        for col,header in enumerate(headers):draw.text((col*256+5,56),header,fill=(10,20,25),font=small)
        rowindex=[]
        for row,(label,cells,detail) in enumerate(rows):
            y=82+286*row;draw.text((7,y),label,fill=(10,20,25),font=small)
            for col,cell in enumerate(cells):canvas.paste(Image.fromarray(cell),(col*256,y+22));totalcells+=1
            rowindex.append(detail)
        path=out/(name+'.png');canvas.save(path);actual=np.array(Image.open(path))
        for row,(_,cells,_) in enumerate(rows):
            for col,cell in enumerate(cells):same(actual[104+286*row:360+286*row,col*256:(col+1)*256],cell,'Sheet actualcell differs')
        sheetindex.append({'file':pin(path),'title':title,'headers':headers,'cases':rowindex,'all_displayed_cells_exact':True,'root_visual_review_pending':True})
    for kind,key,cycles,label in [('monitor','monitor_cases',[0,16,32,64],'original_monitor'),('challenge','challenge_cases',[0,64],'challenge'),('public_development','development_cases',[0,64],'public_DEV')]:
        cases=p[key]
        extra_cycles=sorted({int(folder.name[1:]) for folder in (RETURN/'outputs').glob('u[0-9][0-9][0-9][0-9]') if (folder/kind).exists()}-set(cycles))
        actualcycles=sorted(set(cycles+extra_cycles))
        if kind=='monitor':headers=['Known clean target','Observed context','Reviewed support (red)','Historical original CodeFormer','R1 frozen u0','Historical joint V2 u64']+['V3 fusion u%d'%c for c in actualcycles if c!=0]
        else:headers=['Known TRAIN target' if kind=='challenge' else 'Unpaired photograph; no target','Observed context','Reviewed support (red)','R1 frozen u0','Historical joint V2 u64']+['V3 fusion u%d'%c for c in actualcycles if c!=0]
        for offset in range(0,len(cases),4):
            rows=[]
            for case in cases[offset:offset+4]:
                ident=case['id'];observed=source(case['input256']);mask8=source(case['mask'],'L');mask=mask8==255;overlay=observed.copy();overlay[mask]=np.rint(overlay[mask].astype(np.float32)*.4+np.array([230,45,45],np.float32)*.6).astype(np.uint8)
                target=source(case['clean256']) if kind!='public_development' else placeholder('UNPAIRED photograph|No aligned clean reference|No hidden-identity metrics')
                cells=[target,observed,overlay];metadata={'id':ident,'kind':kind,'condition':case.get('condition'),'mask_kind':case.get('mask_kind'),'family':case.get('family'),'available_snapshots':[],'missing_snapshots':[],'all_snapshot_output_pins':[]}
                if kind=='monitor':
                    historical=OLD/'outputs/u0000/monitor'/(ident+'.png');hp=saved(historical);need(hp is not None,'Missing historicaloriginal');cells.append(hp);metadata['historical_original']=pin(historical)
                joint_rows=p['historical_control']['joint_rows'][kind]
                joint=next((row for row in joint_rows if row['id']==ident),None);need(joint is not None,'Historical joint case missing')
                joint_image=source(joint['PNG']);metadata['historical_joint_V2']=pin(STAGE/joint['PNG'])
                for cycle in actualcycles:
                    image=output(kind,cycle,ident)
                    if image is None:
                        cells.append(placeholder('NOT RETURNED|u%d snapshot'%cycle+'|Do not infer model output'));metadata['missing_snapshots'].append(cycle)
                    else:
                        cells.append(image);metadata['available_snapshots'].append(cycle);metadata['all_snapshot_output_pins'].append(pin(RETURN/'outputs'/('u%04d'%cycle)/kind/(ident+'.png')))
                    if cycle==0:cells.append(joint_image)
                if not metadata['available_snapshots']:continue
                title=ident+' | '+str(case.get('condition') or case.get('mask_kind') or case.get('family'))
                rows.append((title,cells,metadata));allrows.append(metadata)
            if rows:
                context='adaptive TRAIN known-target cohort' if kind!='public_development' else 'unpaired public photographs; native suffix is not CCTV'
                render(label+'_%02d'%(offset//4+1),label+' | '+context+' | all cases in fixed protocol order',headers,rows)
    actualpng={path.relative_to(RETURN).as_posix() for folder in (RETURN/'outputs').glob('u[0-9][0-9][0-9][0-9]') for kind in ('monitor','challenge','public_development') for path in (folder/kind).glob('*.png')}
    need(actualpng==available,'Evaluated output PNG omitted orunexpected in fixedprotocol')
    for row in sourcepins.values():need(sha(Path(row['path']))==row['sha256'],'Input changed during render')
    for row in outputpins.values():need(sha(Path(row['path']))==row['sha256'],'Saved output changed during render')
    need(not any(n=='torch' or n=='pretrained_completion' or n.startswith('third_party.codeformer') for n in sys.modules),'Modelruntime imported')
    report={'format':'completion-fusion-only-v3-all-saved-visual-index-v1','complete':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'renderer':pin(Path(__file__)),'protocol':pin(RETURN/'protocol.json'),'source_transport_audit':pin(PARENT/'transport_audit.json'),'fixed_case_order':True,'output_ranking_or_selection':False,'sheets':sheetindex,'sheets_count':len(sheetindex),'unique_cases':len(allrows),'available_evaluated_PNGs':len(available),'all_available_evaluated_PNGs_displayed':True,'displayed_cells':totalcells,'cases':allrows,'source_pins':sourcepins,'output_pins':outputpins,'historical_joint_V2_is_saved_control_only':True,'no_historical_V2_checkpoint_or_retraining_implied':True,'no_saved_snapshot_implied_complete_training':True,'raw_and_visible_preservation_qualification':'This renderer proves displayedpixels/coverage only. Independent savedarray/state audit must establish copiedoutsidepixels and actualtraining status. Maskoverlay identifies reviewed support; fragments outsideit remain copied source, defects insideit are generation.','root_visual_review_pending':True,'quality_qualified':False,'local_execution':{'model_loads':0,'neural_calls':0,'gradients':0,'optimizer_updates':0,'VM_calls':0},'seconds':time.monotonic()-started}
    with (PARENT/'visual_sheet_index.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'index':pin(PARENT/'visual_sheet_index.json'),'sheets':len(sheetindex),'cases':len(allrows),'evaluatedPNGs':len(available),'seconds':report['seconds']}))
if __name__=='__main__':main()