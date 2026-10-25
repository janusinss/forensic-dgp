"""Independent read-only reconstruction of the actual archive-copy cleanup."""
import gzip, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/storage_offload_20261019_v1'
def read(name): return json.loads((OUT/name).read_text(encoding='utf-8-sig'))
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def namespace(label):
    with gzip.open(OUT/(label+'_namespace.jsonl.gz'),'rt',encoding='utf-8') as stream:
        rows=list(map(json.loads,stream))
    result={r['path']:r for r in rows}
    assert len(rows)==len(result),'Duplicate retained namespace'
    return result
def main():
    plan=read('plan.json'); planned={r['path']:r for r in plan['files']}
    assert len(planned)==19 and sum(r['bytes'] for r in planned.values())==38608921227
    assert digest(ROOT/plan['source']['path'])==plan['source']['sha256'],'Offload source differs from sealed plan'
    for kind in ('copy','stream_verify','remote_verified','remote_preapply'):
        r=read(kind+'_receipt.json'); assert r['complete'] and r['exit_code']==0 and not r['timeout']
        for field in ('stdout','stderr'):
            assert digest(ROOT/r[field]['path'])==r[field]['sha256'],'Operation log changed'
    assert sorted((OUT/'combined.txt').read_text().splitlines())==sorted('= '+p for p in planned)
    remote=read('remote_preapply_stdout.txt')
    assert len(remote)==19 and {r['Path'] for r in remote}==set(planned)
    for r in remote:
        expected=planned[r['Path']]; hashes={k.lower():v for k,v in r['Hashes'].items()}
        assert r['Size']==expected['bytes'] and hashes['sha256']==expected['sha256'] and r['ID']
    journal=list(map(json.loads,(OUT/'deletion_journal.jsonl').read_text().splitlines()))
    assert len(journal)==38
    for i,r in enumerate(plan['files']):
        assert journal[2*i]=={'intent':r}
        assert journal[2*i+1]=={'removed':r['path'],'bytes':r['bytes']}
        assert not (ROOT/r['path']).exists()
        assert (ROOT/plan['retained_roots'][r['path']]['retained_root']).is_dir()
    b,a=namespace('before'),namespace('after')
    assert set(b)-set(a)==set(planned),'Unexpected missing retained file'
    changed=sorted(k for k in a.keys()&b.keys() if a[k]!=b[k])
    assert not changed,'Retained metadata changed'
    before,after=read('before_snapshot.json'),read('after_snapshot.json')
    assert {r['path']:r for r in before['models']}=={r['path']:r for r in after['models']}
    for label,s in [('before',before),('after',after)]:
        assert digest(ROOT/s['namespace']['path'])==s['namespace']['sha256']
        assert len(namespace(label))==s['files']
    for r in plan['critical']:
        assert digest(ROOT/r['path'])==r['sha256'],'Current packet, guide or research instruction changed'
    apply=read('apply.json');assert apply['complete'] and apply['logical_bytes_removed']==38608921227 and apply['removed_files']==19
    result={'complete':True,'independently_reconstructed_actual_cleanup':True,'plan_sha256':digest(OUT/'plan.json'),
        'removed_files':19,'logical_bytes_removed':38608921227,'observed_free_bytes_increase':apply['observed_free_bytes_increase'],
        'free_bytes_after_apply':apply['free_bytes_after'],'retained_namespace_files':len(a),'retained_metadata_changes':changed,
        'protected_model_hashes':len(after['models']),'added_paths':sorted(a.keys()-b.keys()),
        'streamed_full_remote_SHA256_verified':True,'source':str(__file__),'source_sha256':digest(Path(__file__)),
        'VM_training_inference_or_optimizer_calls':0,'model_quality_or_goal_completion_implied':False}
    with (OUT/'independent_closure.json').open('x',encoding='utf-8',newline='\n') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result))
if __name__=='__main__': main()
