"""Separate experimental completion UI; requires an explicitly selected checkpoint."""
import base64
import io
import logging
import os
import threading
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool
from completion_inference import load_completion, load_restorer, predict

app = FastAPI(title='Face restoration and completion experiment')
ROOT = Path(__file__).resolve().parent
LIMIT = 10*1024*1024
lock = threading.Lock()
engine = None


def decode_upload(contents):
    if not contents or len(contents)>LIMIT:
        raise ValueError('Use an image smaller than 10 MB')
    try:
        with Image.open(io.BytesIO(contents)) as image:
            if image.width*image.height>16_000_000 or min(image.size)<16:
                raise ValueError('Image dimensions are unsupported')
            return np.array(ImageOps.exif_transpose(image).convert('RGB'))
    except (OSError,Image.DecompressionBombError) as exc:
        raise ValueError('Cannot read this image') from exc


def decode_mask(contents,shape):
    array = decode_upload(contents)
    if array.shape[:2]!=shape:
        raise ValueError('Mask dimensions must match the uploaded face crop')
    return (array.mean(-1)>=127.5).astype(np.float32)


def encoded(tensor):
    array = tensor.detach().cpu().clamp(0,1)
    if array.ndim == 4:
        array = array[0]
    if array.shape[0] == 1:
        array = array.expand(3,-1,-1)
    array = (array.permute(1,2,0).numpy()*255).round().astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')


def get_engine():
    global engine
    if engine is None:
        path = os.environ.get('COMPLETION_CHECKPOINT','')
        if not path:
            raise HTTPException(503,'Select a trained completion checkpoint before using this experiment.')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            model,state = load_completion(path,device)
            engine = {'model':model,'state':state,'device':device,'restorer':None}
        except Exception:
            logging.exception('Completion checkpoint failed to load')
            raise HTTPException(503,'The completion checkpoint is missing, incompatible, or a smoke test.')
    return engine


def execute(contents,mask_contents,restore,only_mask=False):
    # Serialize access to the single GPU model; heavy work runs off the event loop.
    with lock:
        e = get_engine()
        rgb = decode_upload(contents)
        h,w = rgb.shape[:2]
        if h != w:
            raise ValueError('Select a square face crop first')
        size = e['state']['size']
        rgb = np.array(Image.fromarray(rgb).resize((size,size),Image.Resampling.BILINEAR))
        x = torch.from_numpy(rgb.copy()).permute(2,0,1).float()[None].to(e['device'])/255
        if only_mask:
            with torch.no_grad():
                mask = (e['model'].detect(x).sigmoid()>=.5).float()
            return {'mask':encoded(mask),'message':'Review and correct this estimated region.'}
        mask = None
        if mask_contents is not None:
            array = decode_mask(mask_contents,(h,w))
            array = np.array(Image.fromarray((array*255).astype(np.uint8)).resize((size,size),Image.Resampling.NEAREST))
            mask = torch.from_numpy(array.copy()).float()[None,None].to(e['device'])/255
        if restore and e['restorer'] is None:
            path = os.environ.get('RESTORATION_CHECKPOINT',str(ROOT/'checkpoints/dgp_zamboanga_final.pth'))
            try:
                e['restorer'] = load_restorer(path,e['device'])
            except Exception:
                logging.exception('Restoration checkpoint failed to load')
                raise HTTPException(503,'The visible-region restoration model is unavailable.')
        result = predict(e['model'],x,mask,restore,e['restorer'])
        return {'output':encoded(result['output']),'generated_region':encoded(result['alpha']),
                'mask_source':result['mask_source'],'epoch':e['state']['epoch'],
                'message':'Hidden facial features are generated estimates. Review the marked region.'}


@app.get('/',response_class=HTMLResponse)
def page():
    return HTMLResponse((ROOT/'templates/completion.html').read_text(encoding='utf-8'))


@app.post('/mask')
async def mask_endpoint(file: UploadFile=File(...)):
    try:
        return await run_in_threadpool(execute,await file.read(LIMIT+1),None,False,True)
    except ValueError as exc:
        raise HTTPException(422,str(exc))
    except HTTPException:
        raise
    except Exception:
        logging.exception('Mask prediction failed')
        raise HTTPException(500,'Region estimation failed. Try another face crop.')


@app.post('/complete')
async def complete_endpoint(file: UploadFile=File(...),mask: UploadFile|None=File(None),restore_visible: bool=Form(False)):
    try:
        return await run_in_threadpool(execute,await file.read(LIMIT+1),
                                       await mask.read(LIMIT+1) if mask else None,restore_visible)
    except ValueError as exc:
        raise HTTPException(422,str(exc))
    except HTTPException:
        raise
    except Exception:
        logging.exception('Completion failed')
        raise HTTPException(500,'Completion failed. Review the crop and covered region.')
