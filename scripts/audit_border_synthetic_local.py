"""Inference-only localization on all exact local synthetic cache matches."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np,torch
from PIL import Image
from expanded_feature_data import ExpandedCoveringDataset
from scripts.compare_pixel_heads_vm import PixelHead
from scripts.compare_presence_heads_vm import PresenceHead

def main():
 torch.set_num_threads(4);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 cache=Path('outputs/feature_mixed_training');metadata=json.loads((cache/'cache_progress.json').read_text())
 expanded=json.loads(Path('outputs/downloaded_expanded_feature/outputs/expanded_feature_training/fixed_cache/state.json').read_text())
 lookup={r['record']['input_rgb_sha256']:i for i,r in enumerate(expanded['rows']) if r['record']['domain']=='synthetic'}
 train=Path('outputs/downloaded_expanded_border');history=json.loads((train/'results.json').read_text())
 manifest=Path('outputs/expanded_feature_data_v1/manifest.json');assert sha(manifest)==history['protocol']['prior_protocol']['manifest_sha256']
 data=ExpandedCoveringDataset(json.loads(manifest.read_text()),Path('.').resolve(),placement='fixed')
 models={}
 for arm in ('control','border2'):
  path=train/(arm+'_epoch_10.pth');assert sha(path)==history['arms'][arm]['checkpoint_sha256']
  state=torch.load(path,weights_only=True,map_location='cpu');assert state['protocol']==history['protocol']
  pixel=PixelHead(3).eval().requires_grad_(False);gate=PresenceHead(4).eval().requires_grad_(False)
  pixel.load_state_dict(state['pixel']);gate.load_state_dict(state['presence']);models[arm]=(pixel,gate)
 out=Path('outputs/border_synthetic_local');out.mkdir(exist_ok=False)
 report={'scope':'Incomplete local cache; all exact input matches; not full training fit','optimizer_updates':0,'cases':[]}
 for i,r in enumerate(metadata['records']):
  name=f'{i:04d}';input_path=cache/'input'/(name+'.png');mask_path=cache/'mask'/(name+'.png')
  assert sha(input_path)==r['input_sha256'] and sha(mask_path)==r['mask_sha256'] and r['partition']=='train'
  rgb=np.array(Image.open(input_path).convert('RGB'));digest=hashlib.sha256(rgb.tobytes()).hexdigest()
  if digest not in lookup:continue
  index=lookup[digest];row=expanded['rows'][index]['record'];item=data[index-68]
  assert row['path']==r['source']['path']==item['path'] and row['source_sha256']==r['source']['sha256']
  assert sha(r['source']['path'])==r['source']['sha256']
  assert row['kind']==r['kind'] and row['degraded']==r['degraded']
  assert np.array_equal(rgb,(item['input'].permute(1,2,0).numpy()*255).round().astype('uint8'))
  target=np.array(Image.open(mask_path).convert('L'))>0;assert np.array_equal(target,item['mask'][0].numpy().astype(bool))
  saved=torch.load(cache/'features'/(name+'.pt'),weights_only=True,map_location='cpu');assert saved['record']==r
  assert r['encoder_sha256']==history['protocol']['prior_protocol']['encoder_sha256']
  features=saved['features'];assert features.shape==(1,256,64,64) and torch.isfinite(features).all()
  near=cv2.dilate(target.astype('uint8'),np.ones((17,17),np.uint8)).astype(bool)&~target
  case={'local_index':i,'expanded_index':index,'record':row,'arms':{}}
  for arm,(pixel,gate) in models.items():
   with torch.inference_mode():raw=(pixel(features)[0,0]>=0).numpy();prob=float(gate(features).sigmoid()[0])
   case['arms'][arm]={}
   for mode,pred in (('raw',raw),('gated',raw if prob>=.5 else np.zeros_like(raw))):
    fp=pred&~target
    case['arms'][arm][mode]={'tp':int((pred&target).sum()),'fn':int((~pred&target).sum()),'near_fp':int((fp&near).sum()),'far_fp':int((fp&~near).sum()),'probability':prob}
    folder=out/arm/mode;folder.mkdir(parents=True,exist_ok=True);Image.fromarray(pred.astype('uint8')*255).save(folder/(name+'.png'))
  report['cases'].append(case)
 assert len(report['cases'])==46
 report['groups']={}
 for arm in models:
  report['groups'][arm]={}
  for kind in ('none','object','irregular'):
   for degraded in (False,True):
    cases=[c for c in report['cases'] if c['record']['kind']==kind and c['record']['degraded']==degraded]
    report['groups'][arm][kind+'/'+str(degraded)]={'cases':len(cases),**{mode:{k:sum(c['arms'][arm][mode][k] for c in cases) for k in ('tp','fn','near_fp','far_fp')} for mode in ('raw','gated')}}
 report['complete']=True;(out/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['groups'],indent=2))
if __name__=='__main__':main()
