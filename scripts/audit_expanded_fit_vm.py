"""Inference-only grouped fit audit of existing VM caches; no optimizer or updates."""
import argparse,hashlib,json,sys,tarfile
from pathlib import Path
import torch


def summarize(cases):
    if not cases:raise ValueError('Empty audit group')
    result={}
    for mode in ('raw','gated'):
        s={k:sum(c[mode][k] for c in cases) for k in cases[0][mode]}
        result[mode]={'counts':s,'iou':s['tp']/max(1,s['tp']+s['fp']+s['fn']),
                      'missed_fraction':s['fn']/max(1,s['tp']+s['fn']),
                      'visible_false_positive':s['fp']/max(1,s['visible'])}
    result['cases']=len(cases)
    return result


def main(root):
    if not torch.cuda.is_available():raise RuntimeError('Run this cache audit on the VM GPU')
    root=Path(root).resolve();sys.path.insert(0,str(root))
    from feature_disk_cache import FeatureDiskCache
    from expanded_feature_data import local_path
    from scripts.compare_pixel_heads_vm import PixelHead
    from scripts.compare_presence_heads_vm import PresenceHead
    from scripts.train_expanded_feature_vm import pixel_counts,head_digest
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    train=root/'outputs/expanded_feature_training';history=json.loads((train/'results.json').read_text())
    protocol=json.loads((train/'protocol.json').read_text())
    assert history['complete'] and history['protocol']==protocol
    inventory_path=root/'expanded_inventory.json';assert sha(inventory_path)==protocol['inventory_sha256']
    inventory=json.loads(inventory_path.read_text())
    for name in ('feature_disk_cache.py','scripts/compare_pixel_heads_vm.py','scripts/compare_presence_heads_vm.py','scripts/train_expanded_feature_vm.py'):
        assert sha(local_path(root,name))==inventory[name]
    out=root/'outputs/expanded_fit_audit';out.mkdir(exist_ok=False)
    report={'complete':False,'optimizer_updates':0,'threshold':.5,'script_sha256':sha(__file__),'protocol':protocol,'arms':{}}
    for arm in ('fixed','anatomical'):
        checkpoint=train/(arm+'_epoch_20.pth');assert sha(checkpoint)==history['arms'][arm]['checkpoint_sha256']
        state=torch.load(checkpoint,weights_only=True,map_location='cpu');assert state['protocol']==protocol
        pixel=PixelHead(3).cuda().eval().requires_grad_(False);pixel.load_state_dict(state['pixel'])
        gate=PresenceHead(4).cuda().eval().requires_grad_(False);gate.load_state_dict(state['presence'])
        before=head_digest(pixel,gate)
        context={'arm':arm,'manifest_sha256':protocol['manifest_sha256'],'inventory_sha256':protocol['inventory_sha256'],'encoder_sha256':protocol['encoder_sha256']}
        cache=FeatureDiskCache.open(train/(arm+'_cache'),context);cases=[]
        try:
            with torch.inference_mode():
                for start in range(0,cache.state['count'],12):
                    f,t,records=cache.batch(range(start,min(start+12,cache.state['count'])))
                    f=torch.from_numpy(f).cuda();t=torch.from_numpy(t).cuda().bool()
                    raw=pixel(f)>=0;prob=gate(f).sigmoid();gated=raw&(prob>=.5)[:,None,None,None]
                    for j,r in enumerate(records):
                        cases.append({'cache_index':start+j,'record':r,'probability':float(prob[j]),
                                      'raw':pixel_counts(raw[j:j+1],t[j:j+1]),'gated':pixel_counts(gated[j:j+1],t[j:j+1])})
                    if (start//12)%25==0:print(arm,'audit',min(start+12,cache.state['count']),cache.state['count'],flush=True)
        finally:cache.close()
        assert before==head_digest(pixel,gate)
        groups={}
        for domain in ('real','synthetic'):
            subset=[c for c in cases if c['record']['domain']==domain];groups[domain]=summarize(subset)
            for mode in ('raw','gated'):
                assert groups[domain][mode]['counts']==history['arms'][arm]['training_fit'][domain+'/'+mode]['counts'],'Fit count mismatch; preserve audit and investigate runtime'
        syn=[c for c in cases if c['record']['domain']=='synthetic']
        for kind in sorted({c['record']['kind'] for c in syn}):
            for degraded in (False,True):
                subset=[c for c in syn if c['record']['kind']==kind and c['record']['degraded']==degraded]
                groups[kind+'/'+str(degraded)]=summarize(subset)
        for source in sorted({c['record']['source'] for c in syn}):
            groups[source]=summarize([c for c in syn if c['record']['source']==source])
        report['arms'][arm]={'checkpoint_sha256':sha(checkpoint),'weights_unchanged':True,'fit_counts_match_original':True,'groups':groups,'cases':cases}
        (out/'results.partial.json').write_text(json.dumps(report,indent=2))
        del pixel,gate;torch.cuda.empty_cache()
    report['complete']=True;(out/'results.json').write_text(json.dumps(report,indent=2))
    archive=root/'expanded-fit-audit-results.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        tar.add(out/'results.json',arcname='results.json');tar.add(Path(__file__),arcname='audit_expanded_fit_vm.py')
    print('DONE:',archive,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default='.')
    main(parser.parse_args().root)
