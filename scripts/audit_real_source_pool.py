"""Source-only duplicate screening; no split assignment or automatic labels."""
import json,hashlib
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageOps

def signature(path):
 raw=path.read_bytes()
 with Image.open(path) as im:
  rgb=ImageOps.exif_transpose(im).convert('RGB');size=rgb.size
  decoded=hashlib.sha256(str(size).encode()+rgb.tobytes()).hexdigest()
  small=np.array(rgb.convert('L').resize((32,32)),dtype=np.float32)
 d=cv2.dct(small)[:8,:8].flatten()[1:];bits=d>np.median(d)
 return {'sha256':hashlib.sha256(raw).hexdigest(),'decoded_sha256':decoded,'phash':sum(int(v)<<i for i,v in enumerate(bits)),'width':size[0],'height':size[1]}

def main():
 root=Path('.').resolve();out=root/'outputs/real_source_pool_audit';out.mkdir(exist_ok=False)
 manifest_path=root/'dataset/detector_glare_review_v3/manifest.json';review=json.loads(manifest_path.read_text())['records']
 split_path=root/'outputs/downloaded_phase4/outputs/phase4_with_progress/split.json';split=json.loads(split_path.read_text())
 benchmark_path=root/'outputs/downloaded_completion_benchmark/outputs/completion_pretrained_vm/manifest.json';bench=json.loads(benchmark_path.read_text())['cases']
 refs={}
 for r in review:
  refs[r['source'].replace('\\','/')]='reviewed_source_'+r['split']
  refs['dataset/detector_glare_review_v3/'+r['image']]='reviewed_crop_'+r['split']
 for name in split['validation']:refs[name.replace('\\','/')]='original_validation'
 for c in bench:refs[c['source'].replace('\\','/')]='benchmark_source'
 reference=[]
 for i,(name,role) in enumerate(sorted(refs.items())):
  path=root/name
  reference.append({'path':name,'role':role,**signature(path)})
  if i%1000==0:print('references',i,len(refs),flush=True)
 paths=sorted((root/'dataset/real_occlusion_review').glob('*.jpg'));records=[]
 for i,path in enumerate(paths):
  row={'path':path.relative_to(root).as_posix(),**signature(path),'training_enabled':False,'partition':None,'flags':[]}
  for ref in reference:
   same=row['sha256']==ref['sha256'] or row['decoded_sha256']==ref['decoded_sha256']
   dist=(row['phash']^ref['phash']).bit_count()
   if same or dist<=6:row['flags'].append({'reference':ref['path'],'role':ref['role'],'exact':same,'phash_distance':dist})
  records.append(row)
  if i%300==0:print('sources',i,len(paths),flush=True)
 pairs=[]
 for i,row in enumerate(records):
  for other in records[:i]:
   dist=(row['phash']^other['phash']).bit_count()
   if row['sha256']==other['sha256'] or row['decoded_sha256']==other['decoded_sha256'] or dist<=6:
    pairs.append({'a':row['path'],'b':other['path'],'distance':dist})
 duplicates={p[k] for p in pairs for k in ('a','b')}
 for row in records:row['candidate_for_visual_review']=not row['flags'] and row['path'] not in duplicates
 prior=json.loads((root/'outputs/training_diversity_audit/real_occlusion_annotation_queue.json').read_text())
 assert all(hashlib.sha256((root/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in prior['records'])
 result={'complete':True,'training_enabled':False,'records':records,'within_pool_pairs':pairs,'reference_count':len(reference),'existing_queue_verified':len(prior['records']),'candidate_count':sum(r['candidate_for_visual_review'] for r in records),'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),'limitations':'Whole-image exact/DCT screen is not identity or crop-disjoint proof. All candidates need visual/group review and annotations.'}
 (out/'results.json').write_text(json.dumps(result,indent=2));print({k:result[k] for k in ('reference_count','existing_queue_verified','candidate_count')})
if __name__=='__main__':main()
