"""Exact saved-pixel gallery after independent return verification; no inference."""
import json
from pathlib import Path
import time
from PIL import Image, ImageDraw
from cctv_dgp_finite_guard_v1_r1_contract import sha,read,write

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_vm'
RETURN = ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_vm_return'
OUT = ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_return_review_v1/gallery'
AUDIT = ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_independent_audit.json'
PIN = 'efdd62759213136114a56f0aaa256278753cac8bac4bc6c6a52f1f8b7550598f'


def main():
    started = time.monotonic(); assert not OUT.exists()
    assert sha(BUNDLE/'protocol.json') == PIN
    p,audit,result = read(BUNDLE/'protocol.json'),read(AUDIT),read(RETURN/'outputs/results.json')
    assert audit['complete'] and audit['protocol_sha256'] == result['protocol_sha256'] == PIN
    assert audit['all_raw_and_PNG_cases_recomputed'] == 400 and audit['proposals_verified'] == 3
    assert audit['accepted_training_changes_verified'] == 0 and audit['CPU_replay_cases'] == 80
    assert result['gradient_queries'] == 320 and result['optimizer_updates'] == result['epochs'] == 0
    labels = ['baseline']+[t['variant'] for t in result['trial_summaries']]
    assert labels == ['baseline','proposal_s01_2em05','proposal_s01_1em05','proposal_s01_5em06']
    by_id = {c['id']:c for c in p['cases']}; OUT.mkdir(parents=True)
    pages,bindings,model_files = [],{},set(); cells = 0
    for co in p['cohorts']:
        for begin in range(0,50,5):
            number = len(pages)+1
            cases = [by_id[cid] for cid in co['case_ids'][begin:begin+5]]
            assert len({c['source_person_or_reference'] for c in cases}) == 1
            sheet = Image.new('RGB',(1608,1484),'white'); draw = ImageDraw.Draw(sheet)
            headings = ['input','paired target']+labels
            for col,heading in enumerate(headings): draw.text((5+268*col,4),heading,fill='black')
            source_cells = []
            for row,case in enumerate(cases):
                yy = 24+292*row
                draw.text((5,yy),co['name']+' / '+case['id'],fill='black')
                paths = [BUNDLE/case['input'],BUNDLE/case['target']]+[
                    RETURN/'outputs'/label/(case['id']+'.png') for label in labels]
                for col,path in enumerate(paths):
                    name = path.relative_to(ROOT).as_posix(); bindings[name] = sha(path)
                    with Image.open(path) as image:
                        assert image.size == (256,256)
                        sheet.paste(image.convert('RGB'),(5+268*col,yy+24))
                    source_cells.append({'file':name,'box':[5+268*col,yy+24,5+268*col+256,yy+24+256]})
                    cells += 1
                    if col >= 2: model_files.add(name)
            filename = f'{number:02d}-finite.png'; sheet.save(OUT/filename)
            pages.append({'file':filename,'cohort':co['name'],'reference':cases[0]['source_person_or_reference'],
                'case_ids':[c['id'] for c in cases],'headings':headings,'size':list(sheet.size),
                'sha256':sha(OUT/filename),'cells':source_cells})
            assert time.monotonic()-started < 120
    assert len(pages) == 20 and cells == 600 and len(model_files) == 400
    write(OUT/'gallery_manifest.json',{'complete':True,'protocol_sha256':PIN,
        'results_sha256':sha(RETURN/'outputs/results.json'),'independent_audit_sha256':sha(AUDIT),
        'generator_sha256':sha(Path(__file__)),'pages':pages,'source_bindings':bindings,
        'unique_model_outputs':400,'exact256_cells':600,'pages_count':20,
        'neural_calls':0,'gradient_queries':0,'optimizer_updates':0,'visual_review_pending':True,
        'model_qualification':False,'goal_complete':False,'seconds':time.monotonic()-started})
    print({'complete':True,'pages':20,'exact256_cells':600,'unique_model_outputs':400})


if __name__ == '__main__': main()
