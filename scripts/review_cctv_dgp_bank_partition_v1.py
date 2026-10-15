"""Render fixed-weight cross-compositions; require explicit sheet observations."""
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image, ImageDraw

from import_cctv_dgp_bank_path_ablation_v1_return import ROOT, RETURN, PARENT, read, sha, write

ANALYSIS=ROOT/'outputs/cctv_dgp_bank_partition_v1_analysis'
DEST=ANALYSIS/'visual_review'


def render():
    assert not DEST.exists() and read(ANALYSIS/'independent_audit.json')['complete']
    p,old=read(RETURN/'protocol.json'),read(PARENT/'protocol.json')
    roots=[PARENT,RETURN,ANALYSIS]
    manifests={r:read(r/('file_manifest.json' if r==ANALYSIS else 'export_manifest.json'))['files'] for r in roots}
    folders={'initial':(PARENT,PARENT/'outputs/bank_comparison_v1/baseline'),
        'joint':(RETURN,RETURN/'outputs/path_ablation_v1/enabled'),
        'decoder_only':(RETURN,RETURN/'outputs/path_ablation_v1/disabled'),
        'bank_only':(ANALYSIS,ANALYSIS/'bank_only')}
    labels={'initial':'original DGP / initial','joint':'jointly trained','decoder_only':'trained decoder only','bank_only':'trained bank / initial decoder'}
    DEST.mkdir(); sheets=[]; bindings={}
    def tile(sheet,draw,root,path,label,row,col,cells):
        expected=manifests[root][path.relative_to(root).as_posix()]; assert sha(path)==expected
        x,y=col*256,row*278
        draw.text((x+2,y+3),label,fill='black')
        with Image.open(path) as im:
            assert im.size==(256,256); sheet.paste(im.convert('RGB'),(x,y+22))
        bindings[path.relative_to(ROOT).as_posix()]=expected
        cells.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=expected,x=x,y=y+22,width=256,height=256,row=row,column=col,label=label))
    selected=[c for c in old['cases'] if c['id'] in p['case_ids']]
    refs={r['id']:r for r in old['references']}
    ref_ids=list(dict.fromkeys(c['source_person_or_reference'] for c in selected)); assert len(ref_ids)==20
    for ref_id in ref_ids:
        cases=[c for c in selected if c['source_person_or_reference']==ref_id]; assert len(cases)==5
        sheet=Image.new('RGB',(1536,1390),'white'); draw=ImageDraw.Draw(sheet); cells=[]
        for row,c in enumerate(cases):
            paths=[(PARENT,PARENT/c['input'],'input'),(PARENT,PARENT/refs[ref_id]['target'],'clean TRAIN target')]
            paths += [(root,folder/'previews'/(c['id']+'.png'),labels[label]) for label,(root,folder) in folders.items()]
            for col,(root,path,label) in enumerate(paths): tile(sheet,draw,root,path,c['profile']+': '+label,row,col,cells)
        name='paired_'+ref_id+'.png'; sheet.save(DEST/name)
        sheets.append(dict(name=name,sha256=sha(DEST/name),role='paired photographic TRAIN',case_ids=[c['id'] for c in cases],cells=cells))
    for page in range(6):
        cases=old['native_development'][page*4:(page+1)*4]; assert len(cases)==4
        sheet=Image.new('RGB',(1280,1112),'white'); draw=ImageDraw.Draw(sheet); cells=[]
        for row,c in enumerate(cases):
            paths=[(PARENT,PARENT/c['input'],'input')]+[(root,folder/'native'/(c['id']+'.png'),labels[label]) for label,(root,folder) in folders.items()]
            for col,(root,path,label) in enumerate(paths): tile(sheet,draw,root,path,c['id']+': '+label,row,col,cells)
        name=f'native_{page:02d}.png'; sheet.save(DEST/name)
        sheets.append(dict(name=name,sha256=sha(DEST/name),role='native CCTV DEV unpaired',case_ids=[c['id'] for c in cases],cells=cells))
    write(DEST/'sheet_manifest.json',dict(complete=True,renderer_sha256=sha(__file__),
        independent_audit_sha256=sha(ANALYSIS/'independent_audit.json'),source_tile_bindings=bindings,sheets=sheets,
        paired_sheets=20,native_sheets=6,paired_cases=100,native_cases=24,variants=list(folders),
        cells=sum(len(s['cells']) for s in sheets),viewed_sheets=0,rendering_does_not_imply_review=True,
        final_identities_used=False,independent_final_review_pending=True,model_qualified=False,goal_complete=False))
    print(dict(sheets=len(sheets),cells=sum(len(s['cells']) for s in sheets),viewed_sheets=0),flush=True)


def record(sheet,notes):
    manifest=read(DEST/'sheet_manifest.json'); plan={s['name']:s['sha256'] for s in manifest['sheets']}
    assert sheet in plan and Path(sheet).name==sheet and sha(DEST/sheet)==plan[sheet]
    assert len(notes.strip())>=40
    folder=ANALYSIS/'visual_observations'; folder.mkdir(exist_ok=True)
    write(folder/(sheet+'.json'),dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),sheet=sheet,
        sheet_sha256=plan[sheet],sheet_manifest_sha256=sha(DEST/'sheet_manifest.json'),
        displayed_through='view_image at original resolution',reviewer='implementing assistant',
        independent_final_review=False,notes=notes,model_qualified=False,goal_complete=False))


if __name__=='__main__': render()
