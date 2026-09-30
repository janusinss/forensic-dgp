"""Evaluate matched border-loss final heads against unchanged development validation."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from PIL import Image
from scripts.compare_pixel_heads_vm import PixelHead
from scripts.compare_presence_heads_vm import PresenceHead
from scripts.evaluate_presence_comparison import aggregate
from detector_training import load_manifest
from detector_replay import retention_passes

def main():
    torch.set_num_threads(4)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    train=Path('outputs/downloaded_expanded_border')
    history=json.loads((train/'results.json').read_text());protocol=history['protocol']
    assert history['complete'] and protocol==json.loads((train/'protocol.json').read_text())
    assert sha(train/'train_expanded_border_vm.py')==protocol['script_sha256']==sha('scripts/train_expanded_border_vm.py')
    expected={'format':'expanded-border-v1','arms':{'control':1,'border2':2},'epochs':10,'steps_per_epoch':80,'batch_size':12,'seed':42,'lr':1e-4,'weight_decay':1e-4,'gradient_clip':1.,'gate_frozen':True,'threshold':.5}
    assert all(protocol[k]==v for k,v in expected.items())
    preparation=json.loads(Path('outputs/expanded_border_preparation.json').read_text())
    assert protocol['schedule_sha256']==preparation['schedule_sha256']
    prior_train=Path('outputs/downloaded_expanded_feature/outputs/expanded_feature_training')
    parent_state=torch.load(prior_train/'fixed_epoch_20.pth',weights_only=True,map_location='cpu')
    assert sha(prior_train/'fixed_epoch_20.pth')==protocol['parent_sha256']
    assert protocol['prior_protocol']==parent_state['protocol']
    from scripts.train_expanded_feature_vm import head_digest
    pm=PixelHead(3);pg=PresenceHead(4);pm.load_state_dict(parent_state['pixel']);pg.load_state_dict(parent_state['presence'])
    assert head_digest(pm,pg)==protocol['initial_heads_sha256']
    source_path=Path('outputs/expanded_feature_data_v1/manifest.json')
    assert sha(source_path)==protocol['prior_protocol']['manifest_sha256']
    source=json.loads(source_path.read_text())
    parent=Path('outputs/downloaded_feature_mixed_vm/outputs/feature_mixed_training')
    manifest=Path('dataset/detector_glare_review_v3/manifest.json');assert sha(manifest)==source['v3_manifest_sha256']
    rows=[r for r in load_manifest(manifest) if r['split']=='validation']
    cache_path=Path('outputs/feature_presence_validation/validation_features.pt')
    prior=json.loads(Path('outputs/feature_mixed_validation/results.json').read_text())
    assert sha(cache_path)==prior['real_feature_cache_sha256']
    cache=torch.load(cache_path,weights_only=True)
    assert cache['images']==[r['image'] for r in rows] and cache['image_sha256']==[r['image_sha256'] for r in rows]
    bench=Path('outputs/downloaded_completion_benchmark/outputs/completion_pretrained_vm')
    assert sha(bench/'manifest.json')==json.loads((parent/'sources.json').read_text())['benchmark_manifest_sha256']
    synthetic=json.loads((bench/'manifest.json').read_text())['cases']
    assert len(rows)==25 and len(synthetic)==400
    out=Path('outputs/expanded_border_validation');out.mkdir(exist_ok=False)
    models={};gates={};report={'protocol':protocol,'arms':{},'complete':False}
    for arm,kernel in [('control',3),('border2',3)]:
        path=train/(arm+'_epoch_10.pth');state=torch.load(path,weights_only=True)
        assert state['format']=='expanded-border-v1' and state['kernel']==kernel and state['grid']==4
        assert state['protocol']==protocol
        assert sha(path)==history['arms'][arm]['checkpoint_sha256']
        assert history['arms'][arm]['complete'] and len(history['arms'][arm]['history'])==10
        assert history['arms'][arm]['updates']==800 and history['arms'][arm]['seen']==3540
        assert all(torch.equal(v,parent_state['presence'][k]) for k,v in state['presence'].items())
        gates[arm]=PresenceHead(4).eval().requires_grad_(False);gates[arm].load_state_dict(state['presence'])
        model=PixelHead(kernel).eval().requires_grad_(False);model.load_state_dict(state['pixel']);models[arm]=model
        report['arms'][arm]={'checkpoint_sha256':sha(path),'real_cases':[],'synthetic_cases':[]}
        for domain in ('real','synthetic'):
            for mode in ('raw','gated'):(out/arm/domain/mode).mkdir(parents=True)
    for domain,cases in [('real',rows),('synthetic',synthetic)]:
        for i,c in enumerate(cases):
            if domain=='real':
                name=Path(c['image']).name;f=cache['features'][i:i+1];truth=Path(c['mask_path'])
                meta={'mannequin':c['audit_stratum']=='known_mannequin','glare':c['glare_stratum']=='strong_lens_reflection'}
            else:
                name=c['file'];truth=bench/'mask'/name
                cached=torch.load(Path('outputs/feature_presence_synthetic/features')/(Path(name).stem+'.pt'),weights_only=True)
                assert cached['encoder_sha256']==protocol['prior_protocol']['encoder_sha256'] and cached['input_sha256']==sha(bench/'input'/name)
                f=cached['features'];meta={'source':c['dataset_source'],'stratum':c['kind']+'/'+str(c['degraded'])}
            assert f.shape==(1,256,64,64) and torch.isfinite(f).all()
            t=np.array(Image.open(truth).convert('L'))>0
            for arm,model in models.items():
                with torch.inference_mode():probability=float(gates[arm](f).sigmoid()[0])
                with torch.inference_mode():p=(model(f)[0,0]>=0).numpy()
                record=dict(image=name,probability=probability,**meta)
                for mode,pred in [('raw',p),('gated',p if probability>=.5 else np.zeros_like(p))]:
                    record[mode]=dict(tp=int((pred&t).sum()),fp=int((pred&~t).sum()),fn=int((~pred&t).sum()),visible=int((~t).sum()),empty=bool(t.any() and not pred.any()),negative_fp=bool(not t.any() and pred.any()))
                    Image.fromarray(pred.astype('uint8')*255).save(out/arm/domain/mode/name)
                report['arms'][arm][domain+'_cases'].append(record)
            if (i+1)%50==0:print(domain,i+1,len(cases),flush=True)
    real_base=json.loads(Path('outputs/glare_policy_review/v3_baseline.json').read_text())['metrics']['validation_all']
    syn_base=json.loads(Path('outputs/detector_penalty_comparison/results.json').read_text())['baseline']['synthetic_validation']
    for a in report['arms'].values():
        for domain in ('real','synthetic'):
            cases=a[domain+'_cases'];groups={'all':cases}
            if domain=='real':
                groups.update({'glare':[c for c in cases if c['glare']],'mannequin':[c for c in cases if c['mannequin']],'excluding_mannequin':[c for c in cases if not c['mannequin']]})
            else:
                for key in ('source','stratum'):groups.update({v:[c for c in cases if c[key]==v] for v in sorted({c[key] for c in cases})})
            a[domain+'_metrics']={g:{mode:aggregate([c[mode] for c in cs]) for mode in ('raw','gated')} for g,cs in groups.items()}
        r=a['real_metrics']['all']['gated'];s=a['synthetic_metrics']['all']['gated']
        a['real_gate']=r['iou']>real_base['iou'] and all(r[k]<=real_base[k] for k in ('visible_false_positive','empty_mask_cases','negative_false_positive_cases'))
        a['synthetic_gate']=retention_passes(s,syn_base);a['selection_pass']=a['real_gate'] and a['synthetic_gate']
    report['complete']=True;(out/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({arm:{'real':a['real_metrics']['all'],'synthetic':a['synthetic_metrics']['all'],'passes':a['selection_pass']} for arm,a in report['arms'].items()},indent=2))

if __name__=='__main__':main()
