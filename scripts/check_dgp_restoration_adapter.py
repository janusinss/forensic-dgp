"""Verify new DGP adapter against eight already-audited native raw outputs."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cctv_dgp_pilot import sha,read,write,state_hash
from dgp_face_restoration import load_dgp_restorer,restore_crop,png_rgb,PHASE3_SHA256

NATIVE=ROOT/'outputs/cctv_native_development_v2'
BASE=ROOT/'outputs/cctv_native_comparison_v1'
OUT=ROOT/'outputs/dgp_restoration_adapter_v1'
CASE_IDS=[f'dev_{b}_{n:02}' for b in ('le15','16to23','24to39','ge40') for n in (1,2)]


def prepare():
    if OUT.exists():
        raise ValueError('Preserve partial/completed adapter evidence')
    if sha(NATIVE/'frozen_subset.json')!='c063985b258b808efaa897c88593cbdbcee2848d686d3d056219f8a8e555a98e':
        raise ValueError('Frozen native subset differs')
    if sha(BASE/'results.json')!='3be73af7b0cdae85f54adc8173fd8f48b42511a604fcd60244e3b1d2565a4c34':
        raise ValueError('Audited native baseline differs')
    subset=read(NATIVE/'frozen_subset.json');case_map={c['id']:c for c in subset['cases'] if c['role']=='development'}
    cases=[case_map[c] for c in CASE_IDS]
    assets=['dgp_face_restoration.py','scripts/check_dgp_restoration_adapter.py','tests/test_dgp_face_restoration.py',
            'cctv_dgp_pilot.py','checkpoints/dgp_zamboanga_final.pth','outputs/cctv_native_development_v2/frozen_subset.json',
            'outputs/cctv_native_comparison_v1/results.json']
    assets.extend(p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'models').glob('*.py')))
    for c in cases:
        assets.extend([(NATIVE/c['source_file']).relative_to(ROOT).as_posix(),
                       f"outputs/cctv_native_comparison_v1/stages/{c['id']}_dgp.npy",
                       f"outputs/cctv_native_comparison_v1/images/{c['id']}_input.png",
                       f"outputs/cctv_native_comparison_v1/images/{c['id']}_dgp_raw.png"])
    protocol={'format':'dgp256-inference-adapter-check-v1','date':'2026-10-03','cases':cases,
              'assets_sha256':{n:sha(ROOT/n) for n in assets},'frozen_before_forward_checks':True,
              'purpose':'Check adapter/input/quantization fidelity against cached Phase3 raw outputs; not a new quality evaluation',
              'budget':{'dgp_forwards':8,'seconds_after_loading':60,'optimizer_updates':0,'backward_calls':0,'device':'cpu'},
              'reserved_evaluation_used':False,'application_default_change':False,'new_checkpoint_selected':False}
    OUT.mkdir();write(OUT/'frozen_protocol.json',protocol)
    print({'prepared':True,'protocol_sha256':sha(OUT/'frozen_protocol.json')})


def run():
    if (OUT/'execution.json').exists():
        raise ValueError('Preserve prior execution; do not repeat')
    protocol=read(OUT/'frozen_protocol.json')
    for name,pin in protocol['assets_sha256'].items():
        if sha(ROOT/name)!=pin:
            raise ValueError('Changed adapter/check asset: '+name)
    torch.set_num_threads(4)
    model,provenance=load_dgp_restorer(ROOT/'checkpoints/dgp_zamboanga_final.pth',expected_sha256=PHASE3_SHA256)
    before=state_hash(model.net);count=0
    def hook(*_):
        nonlocal count
        count+=1
    handle=model.net.register_forward_hook(hook)
    write(OUT/'execution.json',{'protocol_sha256':sha(OUT/'frozen_protocol.json'),'model_state_before':before,
                               'provenance':provenance,'training':False,'optimizer_constructed':False,'device':'cpu'})
    rows=[];start=time.monotonic()
    try:
        for c in protocol['cases']:
            with Image.open(NATIVE/c['source_file']) as image:
                source=np.asarray(image.convert('RGB'))
            result=restore_crop(model,source)
            old=np.load(BASE/f"stages/{c['id']}_dgp.npy",allow_pickle=False)
            with Image.open(BASE/f"images/{c['id']}_input.png") as image:
                old_input=np.asarray(image.convert('RGB'))
            with Image.open(BASE/f"images/{c['id']}_dgp_raw.png") as image:
                old_png=np.asarray(image.convert('RGB'))
            np.testing.assert_array_equal(result['input'],old_input)
            np.testing.assert_allclose(result['raw_rgb'],old,atol=1e-7,rtol=0)
            np.testing.assert_array_equal(png_rgb(result['raw_rgb']),old_png)
            expected=np.where(result['observed'][...,None],old,old_input.astype(np.float32)/255)
            np.testing.assert_array_equal(result['observed_rgb'],expected)
            if time.monotonic()-start>60 or count>8:
                raise TimeoutError('Finite adapter check exceeded')
            rows.append({'id':c['id'],'source_sha256':sha(NATIVE/c['source_file']),'input_pixels_exact':True,
                         'raw_float_max_error':float(np.abs(result['raw_rgb']-old).max()),'raw_png_exact':True,
                         'uncaptured_context_preserved':True,'geometry':result['geometry']})
    finally:
        handle.remove()
    if count!=8 or state_hash(model.net)!=before or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError('Forward-only state/count invariant failed')
    write(OUT/'results.json',{'complete':True,'protocol_sha256':sha(OUT/'frozen_protocol.json'),'rows':rows,
          'dgp_forwards':count,'model_state_unchanged':True,'seconds_after_loading':time.monotonic()-start,
          'optimizer_updates':0,'backward_calls':0,'reserved_evaluation_used':False,'application_change':False,
          'new_training_quality_claim':False,'PSNR':None,'SSIM':None})
    print(read(OUT/'results.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','run'])
    args=parser.parse_args();prepare() if args.stage=='prepare' else run()
