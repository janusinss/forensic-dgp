"""Propose eyes for a fixed training-only cohort; proposals never grant eligibility."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageOps

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from completion_data_v2 import square_preserving_geometry, require

PROTOCOL_SHA='46c438f6e72e9bd4709102dd8147dc926fff84de0e7e8a89f3bee1c22a9590fd'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def propose_eyes(rgb,detector):
    base=square_preserving_geometry(rgb,256)
    boxes,landmarks=detector.detect(base['rgb'][:,:,::-1].copy(),max_num=0)
    result={'decision':'needs_review','eyes_native':None,'eyes_square':None,
            'affine':base['affine'].tolist(),'eyes_reviewed':False,'training_enabled':False,
            'reason':'missing/multiple/unreliable detections'}
    if len(boxes)!=1 or landmarks is None:return result
    boxes=np.asarray(boxes);landmarks=np.asarray(landmarks)
    if boxes.shape!=(1,5) or landmarks.shape!=(1,5,2) or not np.isfinite(boxes).all() or not np.isfinite(landmarks).all():return result
    result['face_confidence']=float(boxes[0,4])
    if boxes[0,4]<.6:return result
    eyes=landmarks[0,:2].astype(np.float64);inverse=cv2.invertAffineTransform(base['affine'])
    native=eyes@inverse[:,:2].T+inverse[:,2]
    height,width=rgb.shape[:2]
    if np.any(native<0) or np.any(native[:,0]>width-1) or np.any(native[:,1]>height-1):
        result['reason']='eye falls in padded/outside native image';return result
    order=np.argsort(eyes[:,0]);eyes=eyes[order];native=native[order]
    result.update(decision='proposed',reason='input-only landmark proposal; native visibility still requires review',
                  eyes_native=native.tolist(),eyes_square=eyes.tolist(),landmarks_square=landmarks[0].tolist())
    return result


def prepare(root,output,detector_path):
    root=Path(root).resolve();out=(root/output).resolve()
    require(out.is_relative_to(root) and not out.exists(), 'Preserve existing source cohort')
    detector_path=Path(detector_path).resolve();require(detector_path.is_file(), 'Existing local eye detector required; no download')
    protocol_path=root/'outputs/coverage_protocol_v1/protocol.json'
    require(sha(protocol_path)==PROTOCOL_SHA, 'Training source protocol changed')
    protocol=json.loads(protocol_path.read_text())
    scope_path=root/'outputs/clear_replay_scope_v1/cohort.json';scope=json.loads(scope_path.read_text())
    require(scope['protocol_sha256']==PROTOCOL_SHA, 'Source-screen protocol differs')
    sources=[r for r in scope['records'] if 6<=r['source_id']<31 or 108<=r['source_id']<133]
    require(len(sources)==44 and len({r['source_id'] for r in sources})==44, 'Fixed next source block differs')
    from insightface.model_zoo import get_model
    detector=get_model(str(detector_path),providers=['CPUExecutionProvider'])
    detector.prepare(ctx_id=-1,input_size=(256,256),det_thresh=.6)
    records=[]
    for index,source in enumerate(sources):
        relative=source['path'];path=(root/relative).resolve()
        require(path.is_relative_to(root) and sha(path)==source['sha256']
                and protocol['sources'][source['source_id']]['path']==relative
                and protocol['sources'][source['source_id']]['sha256']==source['sha256'],
                'Source is not an exact unchanged training member')
        with Image.open(path) as im:rgb=np.array(ImageOps.exif_transpose(im).convert('RGB'))
        proposal=propose_eyes(rgb,detector)
        records.append({**source,'split':'train','native_size':[rgb.shape[1],rgb.shape[0]],**proposal})
        if (index+1)%8==0:print(f'Native eye proposals: {index+1}/44',flush=True)
    out.mkdir(parents=True);sheets=[]
    for start in range(0,len(records),16):
        sheet=Image.new('RGB',(1024,1168),'white');draw=ImageDraw.Draw(sheet)
        for n,row in enumerate(records[start:start+16]):
            xx=n%4*256;yy=n//4*292
            with Image.open(root/row['path']) as im:
                original=ImageOps.exif_transpose(im).convert('RGB')
                tile=ImageOps.contain(original,(256,256),Image.Resampling.BILINEAR)
            tx=xx+(256-tile.width)//2;ty=yy+36+(256-tile.height)//2
            sheet.paste(tile,(tx,ty))
            draw.text((xx+2,yy+2),f'{row["source_id"]} {Path(row["path"]).name}',fill='black')
            draw.text((xx+2,yy+18),f'{row["native_size"]} {row["decision"]}',fill='black')
            if row['eyes_native']:
                width,height=row['native_size']
                for ex,ey in row['eyes_native']:
                    px=tx+ex*(tile.width-1)/(width-1);py=ty+ey*(tile.height-1)/(height-1)
                    draw.ellipse((px-3,py-3,px+3,py+3),outline=(32,213,150),width=1)
        name=f'native_sources_{start//16:02}.png';sheet.save(out/name)
        sheets.append({'path':name,'source_ids':[r['source_id'] for r in records[start:start+16]],'sha256':sha(out/name)})
    result={'format':'dgp-native-reflection-source-proposals-v2','records':records,'sheets':sheets,
            'protocol_sha256':PROTOCOL_SHA,'scope_cohort_sha256':sha(scope_path),'script_sha256':sha(__file__),
            'geometry_module_sha256':sha(root/'completion_data_v2.py'),'detector_sha256':sha(detector_path),
            'dependencies':{name:version(name) for name in ('insightface','onnxruntime','opencv-python','numpy')},
            'selection':'Next fixed source-ID blocks6..30 and108..132 in the already screened training-clear cohort; no error/held-out ranking',
            'native_review_complete':False,'training_enabled':False,'optimizer_updates_locally':0,
            'model_inference':'Existing CPU face/keypoint detector only; no candidate occlusion model or held-out photo',
            'limits':'Detected second eyes can be hallucinated on profiles/coverings. Native review is required; landmarks never automatically approve sources.'}
    (out/'proposals.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'photos':len(records),'proposals':sum(r['decision']=='proposed' for r in records),
                      'training_enabled':False,'output':str(out)},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(ROOT));parser.add_argument('--output',default='outputs/reflection_source_cohort_v2')
    parser.add_argument('--detector',required=True)
    args=parser.parse_args();prepare(args.root,args.output,args.detector)
