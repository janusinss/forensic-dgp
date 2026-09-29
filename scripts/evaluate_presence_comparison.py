"""Fixed validation of final presence heads using verified raw pixel masks."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from scripts.compare_presence_heads_vm import PresenceHead
from detector_training import load_manifest
from detector_replay import retention_passes


def gated_counts(raw,probability):
    if probability>=.5:return dict(raw)
    target=raw['tp']+raw['fn']
    return dict(tp=0,fp=0,fn=target,visible=raw['visible'],empty=target>0,negative_fp=False)


def aggregate(cases):
    s={k:sum(c[k] for c in cases) for k in ('tp','fp','fn','visible','empty','negative_fp')}
    return dict(iou=s['tp']/max(1,s['tp']+s['fp']+s['fn']),missed_fraction=s['fn']/max(1,s['tp']+s['fn']),
                visible_false_positive=s['fp']/max(1,s['visible']),empty_mask_cases=s['empty'],negative_false_positive_cases=s['negative_fp'])


def main():
    torch.set_num_threads(4)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    train=Path('outputs/downloaded_presence_comparison');report=json.loads((train/'results.json').read_text())
    assert report['complete'] and all(len(h)==20 and h[-1]['epoch']==20 for h in report['history'].values())
    protocol=report['protocol'];assert protocol==json.loads((train/'protocol.json').read_text())
    assert protocol['script_sha256']==sha('scripts/compare_presence_heads_vm.py')
    parent=Path('outputs/downloaded_feature_mixed_vm/outputs/feature_mixed_training')
    assert protocol['parent_checkpoint_sha256']==sha(parent/'final_epoch_20.pth')
    assert protocol['source_manifest_sha256']==sha(parent/'sources.json')
    prior_path=Path('outputs/feature_mixed_validation/results.json');prior=json.loads(prior_path.read_text())
    assert prior['complete'] and prior['checkpoint_sha256']==protocol['parent_checkpoint_sha256']
    assert prior['protocol']['encoder_sha256']==protocol['encoder_sha256']
    manifest=Path('dataset/detector_glare_review_v3/manifest.json')
    assert sha(manifest)==protocol['v3_manifest_sha256']
    real=[r for r in load_manifest(manifest) if r['split']=='validation']
    real_cache_path=Path('outputs/feature_presence_validation/validation_features.pt')
    assert sha(real_cache_path)==prior['real_feature_cache_sha256']
    cache=torch.load(real_cache_path,weights_only=True)
    assert cache['images']==[r['image'] for r in real] and cache['image_sha256']==[r['image_sha256'] for r in real]
    bench=Path('outputs/downloaded_completion_benchmark/outputs/completion_pretrained_vm')
    sources=json.loads((parent/'sources.json').read_text())
    assert sha(bench/'manifest.json')==sources['benchmark_manifest_sha256']
    benchmark=json.loads((bench/'manifest.json').read_text())['cases']
    out=Path('outputs/presence_comparison_validation');out.mkdir(exist_ok=False)
    heads={};result={'protocol':protocol,'raw_results_sha256':sha(prior_path),'arms':{},'complete':False}
    for arm,grid in [('global',1),('spatial',4)]:
        path=train/(arm+'_epoch_20.pth');state=torch.load(path,weights_only=True)
        assert state['format']=='presence-architecture-diagnostic-v1' and state['grid']==grid and state['protocol']==protocol
        head=PresenceHead(grid).eval().requires_grad_(False);head.load_state_dict(state['model']);heads[arm]=head
        result['arms'][arm]={'checkpoint_sha256':sha(path),'real_cases':[],'synthetic_cases':[]}
        for domain in ('real','synthetic'):(out/arm/domain).mkdir(parents=True)
    for domain,cases in [('real',prior['real_cases']),('synthetic',prior['synthetic_cases'])]:
        expected=25 if domain=='real' else 400
        assert len(cases)==len({c['image'] for c in cases})==expected
        assert [c['image'] for c in cases]==[r['image'] for r in real] if domain=='real' else [c['image'] for c in cases]==[b['file'] for b in benchmark]
        for i,c in enumerate(cases):
            name=Path(c['image']).name
            if domain=='real':
                f=cache['features'][i:i+1];truth=Path(real[i]['mask_path'])
            else:
                cached=torch.load(Path('outputs/feature_presence_synthetic/features')/(Path(name).stem+'.pt'),weights_only=True)
                assert cached['encoder_sha256']==protocol['encoder_sha256'] and cached['input_sha256']==sha(bench/'input'/name)
                f=cached['features'];truth=bench/'mask'/name
            assert f.shape==(1,256,64,64) and torch.isfinite(f).all()
            t=np.array(Image.open(truth).convert('L'))>0
            raw=np.array(Image.open(Path('outputs/feature_mixed_validation')/domain/'raw'/name))>0
            counts=dict(tp=int((raw&t).sum()),fp=int((raw&~t).sum()),fn=int((~raw&t).sum()),visible=int((~t).sum()),empty=bool(t.any() and not raw.any()),negative_fp=bool(not t.any() and raw.any()))
            assert counts==c['raw']
            for arm,head in heads.items():
                with torch.inference_mode():p=float(head(f).sigmoid()[0])
                assert np.isfinite(p)
                predicted=raw if p>=.5 else np.zeros_like(raw)
                score=gated_counts(counts,p)
                # Independent pixel recount checks the aggregate shortcut.
                assert score['tp']==int((predicted&t).sum()) and score['fn']==int((~predicted&t).sum()) and score['fp']==int((predicted&~t).sum())
                Image.fromarray(predicted.astype('uint8')*255).save(out/arm/domain/name)
                metadata={k:v for k,v in c.items() if k not in ('raw','gated','presence_probability')}
                result['arms'][arm][domain+'_cases'].append(dict(metadata,probability=p,counts=score,target_present=bool(t.any())))
            if (i+1)%50==0:print(domain,i+1,expected,flush=True)
    real_base=json.loads(Path('outputs/glare_policy_review/v3_baseline.json').read_text())['metrics']['validation_all']
    synthetic_base=json.loads(Path('outputs/detector_penalty_comparison/results.json').read_text())['baseline']['synthetic_validation']
    for arm,a in result['arms'].items():
        for domain in ('real','synthetic'):
            cases=a[domain+'_cases'];groups={'all':cases}
            if domain=='real':groups.update({'glare':[c for c in cases if c['glare']],'excluding_mannequin':[c for c in cases if not c['mannequin']]})
            else:
                for key in ('source','stratum'):groups.update({v:[c for c in cases if c[key]==v] for v in sorted({c[key] for c in cases})})
            a[domain+'_metrics']={k:aggregate([c['counts'] for c in cs]) for k,cs in groups.items()}
            a[domain+'_classifier']={'missed_covered':sum(c['target_present'] and c['probability']<.5 for c in cases),'false_positive_clear':sum(not c['target_present'] and c['probability']>=.5 for c in cases)}
        r=a['real_metrics']['all'];s=a['synthetic_metrics']['all']
        a['real_gate']=r['iou']>real_base['iou'] and all(r[k]<=real_base[k] for k in ('visible_false_positive','empty_mask_cases','negative_false_positive_cases'))
        a['synthetic_gate']=retention_passes(s,synthetic_base)
        a['selection_pass']=a['real_gate'] and a['synthetic_gate']
    result['complete']=True
    (out/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({arm:{k:a[k] for k in ('real_classifier','synthetic_classifier','real_gate','synthetic_gate')} for arm,a in result['arms'].items()},indent=2))


if __name__=='__main__':main()
