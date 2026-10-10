"""Describe all finite proposals after full independent audit; no optimization."""
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from cctv_dgp_actual_step_review_v1_contract import NAME,PROPOSALS,ROLES,read,write,sha,output_prefix

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'outputs'/NAME
RETURN=ROOT/'outputs'/(NAME+'_return')
OUT=ROOT/'outputs/cctv_dgp_actual_step_review_v1_analysis'


def pixels(path):
    with Image.open(path) as image:
        assert image.mode=='RGB' and image.size==(256,256);return np.asarray(image).copy()


def main():
    start=time.monotonic();plan=read(OUT/'prospective_review.json');p=read(BUNDLE/'protocol.json')
    audit_path=ROOT/'outputs/cctv_dgp_actual_step_review_v1_independent_audit.json';audit=read(audit_path)
    assert audit['complete'] and audit['full_3150_review_complete'] and audit['all3150_raw_and_PNG_and_mean_only_outputs_checked']
    assert audit['CPU_replay']['cases']==150 and audit['optimizer_updates']==audit['gradient_queries']==0
    assert plan['protocol_sha256']==audit['protocol_sha256']==sha(BUNDLE/'protocol.json')
    assert not (OUT/'analysis.json').exists() and not (OUT/'sheets').exists()
    imported=read(ROOT/'outputs/cctv_dgp_actual_step_review_v1_return_import.json');assert imported['complete']
    bindings={audit_path.relative_to(ROOT).as_posix():sha(audit_path),(OUT/'prospective_review.json').relative_to(ROOT).as_posix():sha(OUT/'prospective_review.json')}
    conditions={};term_changes=[];rows=[]
    for probe in p['probes']:
        receipt={}
        for proposal in PROPOSALS:
            path=RETURN/output_prefix(probe['update'],proposal)/'metrics.json';name=path.relative_to(RETURN).as_posix()
            assert sha(path)==imported['files_sha256'][name];bindings[path.relative_to(ROOT).as_posix()]=sha(path)
            receipt[proposal]=read(path);conditions[(probe['update'],proposal)]=receipt[proposal]
        for proposal in PROPOSALS[1:]:
            terms=np.array(receipt[proposal]['objective_term_means']['current_batch'])-np.array(receipt['zero']['objective_term_means']['current_batch'])
            term_changes.append({'update':probe['update'],'proposal':proposal,'current_batch_raw_objective_term_changes':terms.tolist(),
                'sum_of_seven_term_changes':float(terms.sum()),'landmark_term_nonincrease':bool(terms[0]<=0)})
            for role in ROLES:
                for stage in ['raw','PNG']:
                    comparison=receipt[proposal]['comparison_to_same_before_state'][role][stage]
                    rows.append({'update':probe['update'],'proposal':proposal,'role':role,'stage':stage,
                        'cases':receipt[proposal]['groups'][role][stage]['all']['cases'],**comparison})
    assert len(rows)==120 and len(term_changes)==20
    summaries=[]
    for proposal in PROPOSALS[1:]:
        for role in ROLES:
            for stage in ['raw','PNG']:
                subset=[r for r in rows if (r['proposal'],r['role'],r['stage'])==(proposal,role,stage)];assert len(subset)==10
                gains=np.array([r['relative_feature_gain'] for r in subset]);preserved=[r['update'] for r in subset if r['finite_preservation_pass']]
                summaries.append({'proposal':proposal,'role':role,'stage':stage,'states':10,'finite_preservation_pass_updates':preserved,
                    'finite_preservation_failure_updates':[r['update'] for r in subset if not r['finite_preservation_pass']],
                    'positive_feature_gain_updates':[r['update'] for r in subset if r['relative_feature_gain']>0],
                    'feature_gain_mean':float(gains.mean()),'feature_gain_minimum':float(gains.min()),'feature_gain_maximum':float(gains.max()),
                    'mean_is_repeated_case_diagnostic_summary_not_independent_population_estimate':True})
    (OUT/'sheets').mkdir();sheet_records=[];cases={c['id']:c for c in p['cases']}
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    width,left,top,row_height=1816,280,64,288
    for begin in range(0,250,5):
        selected=plan['visual_rows'][begin:begin+5];canvas=Image.new('RGB',(width,top+len(selected)*row_height),'white');draw=ImageDraw.Draw(canvas)
        draw.text((8,4),'Paired photographic TRAIN diagnostic; original256-pixel cells; not native CCTV or final evidence',font=font,fill='black')
        for index,label in enumerate(plan['columns']):draw.text((left+256*index+3,30),label,font=font,fill='black')
        cells=[]
        for slot,row in enumerate(selected):
            assert time.monotonic()-start<300;c=cases[row['id']];camera=pixels(BUNDLE/c['input']);target=pixels(BUNDLE/c['target'])
            zero_folder=RETURN/output_prefix(row['update'],'zero')/row['role']
            raw_path=zero_folder/(row['id']+'.npz');assert sha(raw_path)==imported['files_sha256'][raw_path.relative_to(RETURN).as_posix()]
            with np.load(raw_path,allow_pickle=False) as data:raw=data['original_rgb'].copy()
            with Image.open(BUNDLE/c['observed']) as image:mask=np.asarray(image.convert('L'))>0
            original=np.floor(raw*np.float32(255)).astype(np.uint8);original[~mask]=camera[~mask]
            images=[camera,target,original]
            paths=[BUNDLE/c['input'],BUNDLE/c['target'],raw_path,BUNDLE/c['observed']]
            for proposal in PROPOSALS:
                path=RETURN/output_prefix(row['update'],proposal)/row['role']/(row['id']+'.png')
                assert sha(path)==imported['files_sha256'][path.relative_to(RETURN).as_posix()];images.append(pixels(path));paths.append(path)
            for path in paths:bindings[path.relative_to(ROOT).as_posix()]=sha(path)
            yy=top+slot*row_height
            draw.text((8,yy+5),f"update{row['update']} / {row['role']}\n{row['id']}\n{row['source']}\n{row['profile']}",font=font,fill='black')
            for column,image in enumerate(images):
                box=(left+column*256,yy,left+(column+1)*256,yy+256)
                canvas.paste(Image.fromarray(image),(box[0],box[1]));assert np.array_equal(np.asarray(canvas.crop(box)),image)
                cells.append({'row':row,'column':plan['columns'][column],'box':list(box),'pixel_sha256':__import__('hashlib').sha256(image.tobytes()).hexdigest()})
        path=OUT/f'sheets/page{begin//5+1:03d}.png';canvas.save(path)
        sheet_records.append({'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'cells':cells,'actually_viewed':False})
    assert len(sheet_records)==50 and sum(len(s['cells']) for s in sheet_records)==1500
    write(OUT/'analysis.json',{'complete':True,'protocol_sha256':audit['protocol_sha256'],'independent_audit_sha256':sha(audit_path),
          'prospective_selection_sha256':sha(OUT/'prospective_review.json'),'all120_role_stage_state_proposal_rows':rows,
          'all20_current_batch_raw_term_changes':term_changes,'summaries':summaries,'sheets':sheet_records,'source_evidence_sha256':bindings,
          'paired_TRAIN_diagnostic_only':True,'finite_proposals_not_optimizer_updates':True,'full_TRAIN_capacity_pass':False,
          'visual_review_complete':False,'independent_final_review':False,'native_or_DEV_or_reserved_final_used':False,
          'new_trained_checkpoint':False,'app_promotion':False,'model_forwards':0,'gradient_queries':0,'optimizer_updates':0,
          'checker_sha256':sha(Path(__file__)),'seconds':time.monotonic()-start,'cap_seconds':300})
    print({'complete':True,'conditions':30,'summary_rows':120,'sheets':50,'cells':1500,'actually_viewed':False,'seconds':time.monotonic()-start},flush=True)


if __name__=='__main__':main()
