"""Preserve a concurrent document publication; unchanged cold offload membership.

Original plan/helper/failed guard remain immutable. A fresh metadata baseline and
hash-bound document drift proof are required; no storage candidate may change.
"""
import argparse,copy,gzip,hashlib,json,time,os,stat
from pathlib import Path
import offload_u96_inactive_restoration_20261021_v1 as off

ROOT,OUT=off.ROOT,off.OUT
MARKER='scripts/recover_u96_restoration_offload_20261021_v1.py'
ORIGINAL_SHA='2cb1605e8d0b3b952083814190d1121e0551805abf9ec89591753744ecd50414'

def ns(name):
 with gzip.open(OUT/(name+'_namespace.jsonl.gz'),'rt',encoding='utf-8') as f:
  return {x['path']:x for x in map(json.loads,f) if not x['path'].startswith('.git/')}
def source_pin(p):
 b=p.read_bytes();return {'path':p.relative_to(ROOT).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

def prepare():
 plan=off.plan(ORIGINAL_SHA);changes=off.read('preapply_concurrent_change_inventory_v1.json')
 assert changes['complete'] and all(n.startswith('.git/') for n in changes['unexpected_missing'])
 assert not (OUT/'deletion_journal_v1.jsonl').exists()
 # The handoff must retain its whole old content as an exact prefix.
 protected=next(x for x in plan['protected'] if x['path']=='PROJECT_HANDOFF.md')
 hand=(ROOT/'PROJECT_HANDOFF.md').read_bytes()
 assert hashlib.sha256(hand[:protected['bytes']]).hexdigest()==protected['sha256']
 assert hand[protected['bytes']:].decode('utf-8').strip().startswith('## Academic manuscript rewrite publication - 21 October 2026')
 assert source_pin(ROOT/'docs/THESIS MANUSCRIPT.docx')['sha256']=='3693d39598388941e5f76e8681266efe5c79db74830ba08ad6548fd007f4af63'
 docpub=ROOT/'scratch/thesis_academic_rewrite_20261021/publication.json'
 assert docpub.is_file()
 allowed_prefixes=('docs/','scratch/thesis_academic_rewrite_20261021/')
 changed=[]
 for name in changes['changed_before_apply']:
  if name.startswith('.git/'):continue
  assert name=='PROJECT_HANDOFF.md' or name.startswith(allowed_prefixes),name
  changed.append(off.protected_hash(ROOT/name))
 # No weight or research asset may acquire a different pre-removal identity.
 for row in plan['archives']+plan['members']:
  actual=off.identity(ROOT/row['path'])
  assert all(actual[k]==row[k] for k in actual),row['path']
 corrected=[]
 for row in plan['protected']:
  now=off.protected_hash(ROOT/row['path'])
  if row['path']=='PROJECT_HANDOFF.md':corrected.append(now)
  else:assert now==row;corrected.append(row)
 result={'complete':True,'original_plan_sha256':ORIGINAL_SHA,'source':source_pin(Path(__file__)),
  'original_helper':plan['source'],'candidate_membership_and_all_identity_rows_unchanged':True,
  'original_failed_apply_had_zero_deletions':True,'original_handoff_prefix_retained':True,
  'document_publication':source_pin(docpub),'concurrent_inventory':source_pin(OUT/'preapply_concurrent_change_inventory_v1.json'),
  'fresh_metadata_baseline':source_pin(OUT/'reconciled_before_namespace.jsonl.gz'),
  'retained_concurrent_changed_files':changed,'corrected_protected':corrected,
  'volatile_git_metadata_excluded_from_research_namespace':True,
  'preapply_git_missing':changes['unexpected_missing'],
  'preapply_git_metadata_changes':[n for n in changes['changed_before_apply'] if n.startswith('.git/')],
  'candidate_files':len(plan['archives'])+len(plan['members']),'candidate_bytes':plan['archive_bytes']+plan['member_bytes'],
  'new_review_scope':'Explicit concurrent document changes only; no quality gate, archive proof or deletion scope weakened'}
 off.save('concurrent_publication_recovery_plan_v1.json',result)
 off.save('apply_zero_deletion_stop_v1.json',{'complete':True,'original_apply_exit':1,'reason':'Protected PROJECT_HANDOFF.md appended by disjoint manuscript branch','deletion_journal_present':False,'local_research_files_removed':0})
 print(json.dumps({'complete':True,'recovery_plan_sha256':off.base.sha(OUT/'concurrent_publication_recovery_plan_v1.json'),'retained_changed_files':changes['changed_before_apply'],'added_files':len(changes['added_before_apply']),'candidate_membership_unchanged':True}),flush=True)

def checked(expected):
 assert off.base.sha(OUT/'concurrent_publication_recovery_plan_v1.json')==expected
 recovery=off.read('concurrent_publication_recovery_plan_v1.json')
 assert recovery['source']==source_pin(Path(__file__)) and recovery['complete']
 assert recovery['fresh_metadata_baseline']==source_pin(OUT/'reconciled_before_namespace.jsonl.gz')
 plan=off.plan(ORIGINAL_SHA);assert plan['source']==recovery['original_helper']
 for row in recovery['retained_concurrent_changed_files']:assert off.protected_hash(ROOT/row['path'])==row
 plan=copy.deepcopy(plan);plan['protected']=recovery['corrected_protected']
 off.verify_protected(plan);return plan,recovery

def apply(expected):
 plan,recovery=checked(expected);off.MARKER=MARKER
 # Recheck the old archive objects as well as all original streamed receipts.
 off.base.connection()
 off.command(['lsjson',off.OLD_DEST,'--recursive','--files-only','--hash','--max-depth','4','--timeout','1m','--retries','1'],'old_remote_reconciled',600)
 rows=json.loads((OUT/'old_remote_reconciled_stdout.txt').read_text(encoding='utf-8-sig'))
 for run in plan['old_runs']:
  matches=[x for x in rows if x['Path']==run['remote']['Path']];assert len(matches)==1
  actual=matches[0];hashes={k.lower():v for k,v in actual['Hashes'].items()}
  assert actual['ID']==run['remote']['ID'] and actual['Size']==run['remote']['Size'] and hashes['sha256']==run['source_expected']['sha256']
 off.save('old_remote_reconciled_verified_v1.json',{'complete':True,'objects':10,'recovery_plan_sha256':expected,'all_original_old_object_IDs_sizes_SHA256_current':True})
 original_remote=off.remote_check
 def fresh_remote(p,name):
  assert name=='remote_preapply'
  return original_remote(p,'remote_preapply_reconciled')
 off.remote_check=fresh_remote
 off.apply(plan)

def stable_snapshot():
 start=time.monotonic();stack=[ROOT];count=0;models=[];links=[];target=OUT/'after_namespace.jsonl.gz'
 with target.open('xb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as z:
  while stack:
   assert time.monotonic()-start<900
   with os.scandir(stack.pop()) as es:
    for e in es:
     path=Path(e.path)
     if path in (OUT,ROOT/'.git'):continue
     s=e.stat(follow_symlinks=False);rel=path.relative_to(ROOT).as_posix()
     if stat.S_ISLNK(s.st_mode) or getattr(s,'st_file_attributes',0)&1024:links.append(rel);continue
     if stat.S_ISDIR(s.st_mode):stack.append(path)
     elif stat.S_ISREG(s.st_mode):
      row={'path':rel,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns};z.write((json.dumps(row,separators=(',',':'))+'\n').encode());count+=1
      if path.suffix.lower() in ('.pth','.pt','.ckpt','.safetensors'):models.append(off.protected_hash(path))
 result={'complete':True,'files':count,'models':models,'links_not_followed':links,'namespace':source_pin(target),
  'seconds':time.monotonic()-start,'volatile_git_metadata_excluded':True}
 off.save('after_snapshot_v1.json',result)
 return result

def audit(expected):
 plan,recovery=checked(expected)
 # checked() reads protected files only, all of which must remain available.
 after=stable_snapshot();before=off.read('before_snapshot_v1.json')
 a,b=ns('after'),ns('reconciled_before');original=ns('before')
 intended={x['path'] for x in plan['archives']+plan['members']}
 assert set(b)-set(a)==intended
 changed=sorted(n for n in set(a)&set(b) if a[n]!=b[n]);assert not changed,changed[:10]
 assert set(original)-set(a)==intended
 oldmodels={x['path']:x for x in before['models'] if x['path'] not in intended}
 assert oldmodels=={x['path']:x for x in after['models']}
 for row in recovery['retained_concurrent_changed_files']:assert off.protected_hash(ROOT/row['path'])==row
 off.save('closure_v1.json',{'complete':True,'missing_exact_planned':len(intended),'unexpected_missing':[],
  'retained_metadata_changes':[],'retained_model_hashes':len(oldmodels),'added':sorted(set(a)-set(b)),
  'retained_namespace_files':len(a),'apply':off.base.pin(OUT/'apply_v1.json'),'all_protected_files_verified':True,
  'baseline':'reconciled_before_namespace.jsonl.gz','model_baseline':'original before_snapshot_v1.json',
  'volatile_git_metadata_excluded_from_research_namespace':True,
  'recovery_plan':source_pin(OUT/'concurrent_publication_recovery_plan_v1.json'),
  'explicit_concurrent_document_drift_retained':recovery['retained_concurrent_changed_files'],
  'VM_training_inference_calls':0,'overall_goal_complete':False})
 print(json.dumps({'complete':True,'removed':len(intended),'retained_models':len(oldmodels),'retained_metadata_changes':[],'concurrent_document_changes_retained':len(recovery['retained_concurrent_changed_files'])}),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','apply','audit']);p.add_argument('--recovery-sha',default='');a=p.parse_args()
 if a.mode=='prepare':prepare()
 elif a.mode=='apply':apply(a.recovery_sha)
 elif a.mode=='audit':audit(a.recovery_sha)

if __name__=='__main__':main()
