"""Generate the five prospectively specified diagnostic sheets after audit."""
import hashlib
import json
from pathlib import Path

from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'outputs/cctv_dgp_bank_comparison_v1_r4_B_return'
PROBE=ROOT/'outputs/cctv_dgp_bank_feature_DC_probe_v1'
ANALYSIS=ROOT/'outputs/cctv_dgp_bank_feature_DC_probe_v1_analysis'
DEST=ANALYSIS/'visual_review'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not DEST.exists()
    audited=read(ANALYSIS/'independent_audit.json');plan=read(PROBE/'plan.json')
    assert audited['complete'] and audited['paired_compositions_checked']==200 and audited['native_unpaired_compositions_checked']==48
    assert audited['plan_sha256']==sha(PROBE/'plan.json')
    assert len(plan['visual_reference_ids'])==4 and plan['planned_visual_sheets']==5
    p=read(PARENT/'protocol.json');refs={row['id']:row for row in p['references']}
    parent_hashes=read(PARENT/'export_manifest.json')['files'];probe_hashes=read(PROBE/'manifest.json')['files_sha256']
    hashes={};sheets=[];DEST.mkdir()
    labels=['input','paired target','retained DGP','B50 CUDA PNG','B50 CPU PNG','bank feature DC removed CPU']

    def tile(canvas,draw,path,label,row,column):
        if path.is_relative_to(PARENT):expected=parent_hashes[path.relative_to(PARENT).as_posix()]
        else:expected=probe_hashes[path.relative_to(PROBE).as_posix()]
        assert path.is_file() and not path.is_symlink() and sha(path)==expected
        hashes[path.relative_to(ROOT).as_posix()]=expected
        draw.text((column*256+3,row*278+3),label,fill='black')
        with Image.open(path) as image:
            assert image.size==(256,256)
            canvas.paste(image.convert('RGB'),(column*256,row*278+22))

    for reference in plan['visual_reference_ids']:
        selected=[case for case in p['cases'] if case['source_person_or_reference']==reference]
        assert len(selected)==5
        canvas=Image.new('RGB',(1536,1390),'white');draw=ImageDraw.Draw(canvas)
        for row,case in enumerate(selected):
            id_=case['id'];paths=[PARENT/case['input'],PARENT/refs[reference]['target'],
                PARENT/'outputs/bank_comparison_v1/baseline/previews'/(id_+'.png'),
                PARENT/'outputs/bank_comparison_v1/B50/previews'/(id_+'.png')]
            paths.extend(PROBE/label/'paired'/(id_+'.png') for label in plan['variants'])
            for column,(path,label) in enumerate(zip(paths,labels)):
                tile(canvas,draw,path,case['profile']+': '+label,row,column)
        name='paired_'+reference+'.png';canvas.save(DEST/name);sheets.append(name)
    canvas=Image.new('RGB',(1280,1112),'white');draw=ImageDraw.Draw(canvas)
    selected=[case for case in p['native_development'] if case['id'] in plan['native_visual_case_ids']]
    assert len(selected)==4
    for row,case in enumerate(selected):
        id_=case['id'];paths=[PARENT/case['input'],PARENT/'outputs/bank_comparison_v1/baseline/native'/(id_+'.png'),
            PARENT/'outputs/bank_comparison_v1/B50/native'/(id_+'.png')]
        paths.extend(PROBE/label/'native'/(id_+'.png') for label in plan['variants'])
        for column,(path,label) in enumerate(zip(paths,[labels[index] for index in [0,2,3,4,5]])):
            tile(canvas,draw,path,id_+': '+label,row,column)
    name='native_00.png';canvas.save(DEST/name);sheets.append(name)
    record=dict(complete=True,generator_sha256=sha(__file__),audit_sha256=sha(ANALYSIS/'independent_audit.json'),
        plan_sha256=sha(PROBE/'plan.json'),source_tile_bindings=hashes,
        sheets=[dict(name=name,sha256=sha(DEST/name)) for name in sheets],
        paired_sheets=4,native_sheets=1,viewed_sheets=0,all_planned_sheets_must_be_viewed=True,
        visual_scope='20 paired TRAIN counterfactuals and4 unpaired native DEV cases; no broader visual qualification',
        selected_for_diagnosis_not_final_evaluation=True,independent_final_review=False,
        trained_model_qualified=False,app_promotion=False,goal_complete=False)
    with (DEST/'sheet_manifest.json').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(dict(sheets=len(sheets),viewed_sheets=0,trained_candidate_qualified=False),flush=True)


if __name__=='__main__':main()
