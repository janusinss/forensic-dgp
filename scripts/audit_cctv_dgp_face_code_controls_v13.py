"""Verify/extract and numerically audit the inference-only V13 control return."""
import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.render_cctv_dgp_face_code_controls_v13 import ARMS, PARENT_SHA, RESULTS_SHA, PLAN_NAME


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition,message):
    if not condition:raise ValueError(message)


def import_return(archive,completion,destination):
    require(not destination.exists(),'Preserve prior/partial extraction')
    terminal=read(completion)
    require(terminal['complete'] and terminal['backward_calls']==terminal['optimizer_updates']==0,
            'Not a completed inference-only return')
    plan_root=ROOT/'outputs/cctv_dgp_face_code_render_controls_vm_v13'
    require(terminal['protocol_sha256']==sha(plan_root/PLAN_NAME),'Protocol differs')
    require(archive.stat().st_size==terminal['bytes'] and sha(archive)==terminal['archive_sha256'],'Archive differs')
    require(Path(str(archive)+'.sha256').read_text().split()==[terminal['archive_sha256'],archive.name], 'Checksum differs')
    with tarfile.open(archive,'r:gz') as stream:
        members=stream.getmembers()
        require(len({m.name for m in members})==len(members) and sum(m.size for m in members)<1024**3,'Unexpected archive size/duplicates')
        for member in members:
            p=PurePosixPath(member.name)
            require(bool(p.parts) and not p.is_absolute() and '..' not in p.parts and ':' not in member.name and
                    '\\' not in member.name and (member.isfile() or member.isdir()),'Unsafe archive member')
            require((destination/member.name).resolve().is_relative_to(destination.resolve()),'Archive escapes workspace')
        require(destination.resolve().is_relative_to((ROOT/'outputs').resolve()),'Extraction must remain below outputs')
        destination.mkdir();stream.extractall(destination,filter='data')
    require(sha(destination/PLAN_NAME)==terminal['protocol_sha256'],'Returned protocol differs')
    return terminal


def audit(returned,terminal):
    start=time.monotonic();p=read(returned/PLAN_NAME)
    parent=ROOT/'outputs/cctv_dgp_face_code_fit_vm_v12_r2'
    source=ROOT/'outputs/cctv_dgp_face_code_fit_return_v12_r2_verified/outputs/cctv_dgp_face_code_fit_v12'
    require(p['parent_protocol_sha256']==sha(parent/'face_code_fit_protocol_v12.json')==PARENT_SHA and
            p['parent_results_sha256']==sha(source/'results.json')==RESULTS_SHA,'Parent binding differs')
    require(p['arms']==ARMS and len(p['references'])==10 and all(ref['role']=='train' for ref in p['references']), 'Design/roles differ')
    for name,pin in p['sources_sha256'].items():require(sha(returned/name)==pin,'Returned control source differs')
    out=returned/'outputs/cctv_dgp_face_code_render_controls_v13'
    r=read(out/'results.json');neural=read(out/'neural_receipt.json')
    require(r['complete'] and neural['complete'] and sha(out/'results.json')==terminal['results_sha256'],'Incomplete/different terminal result')
    require(r['protocol_sha256']==neural['protocol_sha256']==terminal['protocol_sha256'],'Control fingerprint differs')
    counts={'generator':162,'encoder':0,'transformer_head':0,'conditioner':0}
    require(r['counts']==neural['counts']==terminal['counts']==counts and neural['frozen_before']==neural['frozen_after'], 'Recorded neural state/counts differ')
    require(neural['seconds']<=240 and terminal['seconds']<=360,'Runtime budget exceeded')
    for item in [r,neural,terminal]:require(item['backward_calls']==item['optimizer_updates']==0,'Unexpected fitting')
    require(r['oracle_clean_target_access'] is True and all(r[k] is False for k in
            ['training','validation_used','native_used','native_reserved_used','production_promoted']), 'Wrong target access/scope')
    require(len(neural['parity'])==2 and all(x['maximum_float_difference']<=2e-6 for x in neural['parity']), 'Parity evidence differs')
    for name,pin in r['artifacts_sha256'].items():require(sha(out/name)==pin,'Changed control artifact: '+name)
    require(not list(out.rglob('*.pth')),'Inference control created a checkpoint')
    tree=ast.parse((parent/'cctv_dgp_pilot.py').read_text())
    fn=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='exported_pixel_metrics']
    require(len(fn)==1,'Pinned numeric helper differs')
    ns={'np':np,'cv2':cv2,'structural_similarity':structural_similarity}
    exec(compile(ast.Module(body=fn,type_ignores=[]),'pinned_pixel_metrics','exec'),ns)

    def rgb(path):
        with Image.open(path) as im:
            require(im.mode=='RGB' and im.size==(256,256),'Invalid PNG geometry')
            return np.asarray(im).copy()

    refs={ref['id']:ref for ref in p['references']};cases={c['id']:c for c in p['cases']}
    require([(row['id'],row['arm']) for row in r['rows']]==[(c['id'],arm) for c in p['cases'] for arm in ARMS], 'Missing/order-different control rows')
    lookup={};groups={};raw_names=set()
    for row in r['rows']:
        c=cases[row['id']];ref=refs[c['reference_id']]
        require(all(row[k]==c[k] for k in ['id','source','profile','reference_id']) and row['oracle']==row['arm'].startswith('teacher_'),'Control metadata differs')
        for key in ['target','observed']:
            require(sha(parent/ref[key])==p['parent_assets_sha256'][ref[key]],'Target/support differs')
        support=np.asarray(Image.open(parent/ref['observed']))>0
        camera=rgb(parent/c['input']);target=rgb(parent/ref['target']);prediction=rgb(out/row['prediction'])
        raw=np.load(out/row['raw'],allow_pickle=False);raw_names.add(row['raw'])
        require(raw.dtype==np.float32 and raw.shape==(256,256,3) and np.isfinite(raw).all() and 0<=raw.min()<=raw.max()<=1,'Invalid raw render')
        expected=(raw*255).astype(np.uint8);expected[~support]=camera[~support]
        np.testing.assert_array_equal(expected,prediction)
        for key,value in ns['exported_pixel_metrics'](prediction,target,support).items():
            if value is None or isinstance(value,bool):require(row[key]==value,'Pixel flag differs')
            else:require(np.isclose(row[key],value,rtol=1e-7,atol=1e-7),'Pixel metric differs')
        lookup[(row['id'],row['arm'])]=row
        for group in ['all',row['source']+'/all',row['source']+'/'+row['profile'],'profile/'+row['profile']]:
            groups.setdefault(row['arm']+'/'+group,[]).append(row)
    require(len(raw_names)==160 and len(r['rows'])==200,'Render/cache count differs')
    cached={(row['id'],row['fidelity']):row for row in read(source/'update300/metrics.json')['rows']}
    cells=0
    for profile in p['profiles']:
        name=f'grids/{profile}_10_rows.png'
        with Image.open(out/name) as im:sheet=np.asarray(im)
        require(sheet.shape==(2904,1820,3),'Control grid shape differs')
        for i,ref in enumerate(p['references']):
            c=next(c for c in p['cases'] if c['reference_id']==ref['id'] and c['profile']==profile)
            images=[rgb(parent/c['input']),rgb(source/cached[(c['id'],0.0)]['prediction'])]
            images.extend(rgb(out/lookup[(c['id'],arm)]['prediction']) for arm in ARMS)
            images.append(rgb(parent/ref['target']))
            for j,image in enumerate(images):
                y=52+i*288;np.testing.assert_array_equal(sheet[y:y+256,2+j*260:258+j*260],image);cells+=1
    summary={}
    for group,items in groups.items():
        mse=float(np.mean([i['MSE'] for i in items]))
        summary[group]={'cases':len(items),'MSE':mse,'PSNR_from_mean_MSE':float(-10*np.log10(mse)) if mse else None,
                        'SSIM':float(np.mean([i['SSIM'] for i in items])),'MAE':float(np.mean([i['MAE'] for i in items]))}
    receipt={'complete':True,'protocol_sha256':terminal['protocol_sha256'],'results_sha256':sha(out/'results.json'),
             'pngs_checked':200,'unique_raw_renders_checked':160,'grid_cells_checked':cells,'summaries':summary,
             'seconds':time.monotonic()-start,'neural_forwards_in_audit':0,'backward_calls':0,'optimizer_updates':0,
             'oracle_clean_target_access':True,'native_used':False,'validation_used':False,'production_promoted':False,
             'limitation':'Independent file/pixel/grid arithmetic and recorded counts/state only; no decoder replay or identity measurement. Clean-label controls are target-informed training-cohort oracles, not model improvements.'}
    require(cells==350,'Grid cell count differs')
    (returned/'local_independent_audit.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='summaries'},indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k.endswith('/all') and k.count('/')==1},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--completion',type=Path,required=True)
    parser.add_argument('--extract-to',type=Path,required=True)
    args=parser.parse_args();destination=args.extract_to.resolve()
    terminal=import_return(args.archive.resolve(),args.completion.resolve(),destination)
    audit(destination,terminal)
