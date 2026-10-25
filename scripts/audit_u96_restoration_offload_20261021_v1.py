"""Author readback of exact plan and final offload; no removal/network/model calls."""
import argparse,gzip,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/u96_inactive_restoration_offload_20261021_v1'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8-sig'))
def save(name,value):
 with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')
def main():
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['plan','closure']);parser.add_argument('--plan-sha',required=True);args=parser.parse_args()
 assert sha(OUT/'plan_v1.json')==args.plan_sha
 plan=read('plan_v1.json');files=plan['archives']+plan['members'];names=[x['path'] for x in files]
 assert plan['complete'] and plan['version']=='u96-inactive-restoration-local-only-v1'
 assert sha(ROOT/plan['source']['path'])==plan['source']['sha256']
 assert len(names)==len(set(names)) and not set(names)&{x['path'] for x in plan['protected']}
 for row in plan['archives']:
  path=Path(row['path']);assert path.parent==Path('outputs') and path.name.startswith(('cctv-codeformer-','cctv-dgp-')) and path.name.endswith('.tar.gz')
  assert 'spatial-continuation' not in path.name and row['nlink']==1
 old={x['remote']['Path']:x for x in plan['old_runs']}
 for row in plan['members']:
  path=Path(row['path']);assert row['archive'] in old and path.suffix in ('.npy','.pt')
  run=old[row['archive']];assert row['path']==run['retained_root']+'/'+row['member'] and row['path'] in run['candidate_paths']
  assert row['member'].startswith('outputs/'+run['run']+'/')
  assert not set(row['member'].lower().split('/'))&{'cache','caches','data','weights','targets','features','inputs','training_cache'}
  assert row['path']!='outputs/codeformer_role_preserving_training_v1_r2_return/outputs/role_preserving_training_v1_r2/snapshot_0050/checkpoint.pt'
  assert row['nlink']==1
 assert sum(x['bytes'] for x in plan['archives'])==plan['archive_bytes']
 assert sum(x['bytes'] for x in plan['members'])==plan['member_bytes']
 if args.mode=='plan':
  for x in files:
   p=ROOT/x['path'];s=p.stat();assert s.st_size==x['bytes'] and s.st_ino==x['inode'] and s.st_mtime_ns==x['mtime_ns']
  result={'complete':True,'plan_readback_pass':True,'author_review_not_independent_agent':True,
   'plan_sha256':args.plan_sha,'source_sha256':sha(ROOT/plan['source']['path']),'archives':len(plan['archives']),
   'bulk_RAW_checkpoint_files':len(plan['members']),'total_bytes':plan['archive_bytes']+plan['member_bytes'],
   'only_regular_nlink1_local_outputs_candidates':True,'code_data_PNGs_metrics_completion_runtime_kept':True,
   'VM_training_neural_calls':0}
  save('plan_author_readback_v1.json',result)
 else:
  for op in ('copy','stream_verify','remote_verified','remote_preapply'):
   receipt=read(op+'_receipt.json');assert receipt['complete'] and receipt['exit_code']==0 and not receipt['timeout']
   for key in ('stdout','stderr'):assert sha(ROOT/receipt[key]['path'])==receipt[key]['sha256']
  assert sorted((OUT/'combined_v1.txt').read_text().splitlines())==sorted('= '+x['path'] for x in plan['archives'])
  remote=read('remote_preapply_stdout.txt');assert len(remote)==len(plan['archives'])
  remote_by={x['Path']:x for x in remote};assert len(remote_by)==len(remote)
  for x in plan['archives']:
   y=remote_by[x['path']];h={k.lower():v for k,v in y['Hashes'].items()}
   assert y['Size']==x['bytes'] and h['sha256']==x['sha256'] and y['ID']
  proven=[]
  for i,run in enumerate(plan['old_runs']):
   proof=read('old_run_'+str(i+1).zfill(2)+'_verified.json')
   assert proof['complete'] and proof['plan_sha256']==args.plan_sha and proof['remote']==run['remote']
   assert proof['downloaded_archive_bytes']==run['source_expected']['bytes'] and proof['downloaded_archive_sha256']==run['source_expected']['sha256']
   proven+=proof['members']
  assert {x['path']:(x['bytes'],x['sha256']) for x in proven}=={x['path']:(x['bytes'],x['sha256']) for x in plan['members']}
  def ns(label):
   with gzip.open(OUT/(label+'_namespace.jsonl.gz'),'rt',encoding='utf-8') as f:return {x['path']:x for x in map(json.loads,f)}
  before,after=ns('before'),ns('after')
  assert set(before)-set(after)==set(names)
  assert all(before[n]==after[n] for n in set(before)&set(after))
  for x in files:assert not (ROOT/x['path']).exists()
  journal=[json.loads(x) for x in (OUT/'deletion_journal_v1.jsonl').read_text().splitlines()]
  assert len(journal)==2*len(files)
  for i,row in enumerate(files):assert journal[2*i]=={'intent':row} and journal[2*i+1]=={'removed':row['path'],'bytes':row['bytes']}
  for x in plan['protected']:assert sha(ROOT/x['path'])==x['sha256']
  b={x['path']:x for x in read('before_snapshot_v1.json')['models'] if x['path'] not in set(names)}
  a={x['path']:x for x in read('after_snapshot_v1.json')['models']};assert a==b
  for x in a.values():assert sha(ROOT/x['path'])==x['sha256']
  result={'complete':True,'author_reconstructed_closure_pass':True,'independent_agent_review':False,
   'plan_sha256':args.plan_sha,'removed_files':len(files),'logical_bytes_removed':plan['archive_bytes']+plan['member_bytes'],
   'fully_downloaded_archive_and_member_SHA256_matches':True,'unexpected_missing':[],'retained_metadata_changes':[],
   'retained_model_hashes':len(a),'retained_namespace_files':len(after),'VM_training_inference_calls':0}
  save('closure_author_reconstruction_v1.json',result)
 print(json.dumps(result))

if __name__=='__main__':main()
