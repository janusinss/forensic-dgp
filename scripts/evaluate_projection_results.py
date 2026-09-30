"""Recount saved projection masks and verify reported protocol/state invariants."""
import json
import math
from pathlib import Path
import sys
import torch
from PIL import Image,ImageDraw
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_coverage_results import binary,recount
from scripts.train_coverage_vm import sha,real_gate,PROTOCOL_SHA,PARENT_SHA
from detector_replay import retention_passes


def main():
    root=Path('outputs/downloaded_projection');runroot=root/'outputs/projection_training_vm'
    run=json.loads((runroot/'run.json').read_text());complete=json.loads((runroot/'complete.json').read_text())
    assert complete['complete'] and complete['updates_per_arm']==210 and not run['preflight']
    assert run['protocol_sha256']==PROTOCOL_SHA and run['parent_sha256']==PARENT_SHA
    assert sha(root/'scripts/train_projection_vm.py')==sha('scripts/train_projection_vm.py')==run['script_sha256']
    assert sha(root/'coverage_projection.py')==sha('coverage_projection.py')==run['helper_sha256']
    assert sha(runroot/'initial.pth')==run['initial_sha256']
    inventory=Path('outputs/coverage_protocol_v1/vm_inventory.json');assert sha(inventory)==run['inventory_sha256']
    for path,digest in json.loads(inventory.read_text()).items():assert sha(path)==digest
    cache=Path('outputs/downloaded_coverage/outputs/coverage_training_vm/replay_pixels.pth')
    assert sha(cache)==run['replay_sha256']
    initial=torch.load(runroot/'initial.pth',weights_only=True,map_location='cpu')
    parent=Path('outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth');assert sha(parent)==PARENT_SHA
    parent_state=torch.load(parent,weights_only=True,map_location='cpu')['model']
    assert initial.keys()==parent_state.keys() and all(torch.equal(v,parent_state[k]) for k,v in initial.items())
    protocol=json.loads(Path('outputs/coverage_protocol_v1/protocol.json').read_text())
    schedule=protocol['schedules']['extended']['batches']
    realroot=Path('dataset/detector_glare_review_v3');real=[r for r in json.loads((realroot/'manifest.json').read_text())['records'] if r['split']=='validation']
    bench=Path('outputs/downloaded_completion_benchmark/outputs/completion_pretrained_vm');cases=json.loads((bench/'manifest.json').read_text())['cases']
    targets={'real':[binary(realroot/r['mask']) for r in real],'synthetic':[binary(bench/'mask'/r['file']) for r in cases]}
    baseline=json.loads((runroot/'baseline.json').read_text())
    prior=Path('outputs/downloaded_coverage/outputs/coverage_training_vm')
    assert baseline==json.loads((prior/'baseline.json').read_text())
    report={'arms':{},'mask_count':0,'initial_matches_parent':True,'promotion':False}
    final={}
    for arm in ('ordinary','projected'):
        history=[json.loads(s) for s in (runroot/arm/'metrics.jsonl').read_text().splitlines()]
        steps=[json.loads(s) for s in (runroot/arm/'steps.jsonl').read_text().splitlines()]
        assert len(steps)==210 and [s['indices'] for s in steps]==[b for e in schedule for b in e]
        assert [(s['epoch'],s['step']) for s in steps]==[(e+1,s) for e in range(10) for s in range(21)]
        assert max(s['gradient_relative_error'] for s in steps)<=1e-4
        assert [r['epoch'] for r in history]==list(range(1,11))
        best=baseline['real']['iou'];checked=[]
        for row in history:
            epoch=row['epoch'];assert row['updates']==epoch*21
            scores={}
            for domain,truth in targets.items():
                folder=runroot/arm/f'epoch_{epoch}_masks'/domain
                assert {p.name for p in folder.glob('*.png')}=={f'{i:04}.png' for i in range(len(truth))}
                preds=[binary(folder/f'{i:04}.png') for i in range(len(truth))]
                scores[domain]=recount(preds,truth)
                assert all(math.isclose(v,row[domain][k],abs_tol=1e-12,rel_tol=0) for k,v in scores[domain].items())
                report['mask_count']+=len(truth)
                if domain=='real':
                    for label,indices in {'human_real':[i for i,r in enumerate(real) if Path(r['image']).name!='new_covered_40.png'],
                        'mannequin':[i for i,r in enumerate(real) if Path(r['image']).name=='new_covered_40.png'],
                        'glare':[i for i,r in enumerate(real) if r.get('glare_stratum')=='strong_lens_reflection']}.items():
                        sub=recount([preds[i] for i in indices],[truth[i] for i in indices])
                        assert all(math.isclose(v,row[label][k],abs_tol=1e-12,rel_tol=0) for k,v in sub.items())
                    if epoch==10:final[arm]=preds
            ro=real_gate(scores['real'],baseline['real'],best);so=retention_passes(scores['synthetic'],baseline['synthetic'])
            assert (ro,so,ro and so)==(row['real_gate'],row['synthetic_gate'],row['selected'])
            if ro and so:best=scores['real']['iou']
            cp=torch.load(runroot/arm/f'epoch_{epoch}.pth',map_location='cpu',weights_only=True)
            assert cp['projection_arm']==arm and cp['detector_training']==run and cp['detector_finetune_epoch']==epoch
            assert all(torch.equal(v,cp['model'][k]) for k,v in initial.items() if k.startswith('generator.'))
            checked.append(row)
        assert (runroot/arm/'best_detector.pth').exists()==any(r['selected'] for r in history)
        applied=[s for s in steps if s['projection_applied']]
        report['arms'][arm]={'history':checked,'final_checkpoint_sha256':sha(runroot/arm/'epoch_10.pth'),
            'projection_count':len(applied),'raw_conflict_count':sum(s['projection']['applied'] for s in steps),
            'replay_ascent_steps':sum(s['replay_first_order_loss_change']>0 for s in steps),
            'ascent_after_projection':sum(s['replay_first_order_loss_change']>0 for s in applied),
            'all_generators_unchanged':True,'max_decomposition_error':max(s['gradient_relative_error'] for s in steps)}
    out=Path('outputs/projection_validation');out.mkdir(exist_ok=False)
    canvas=Image.new('RGB',(512,1480),'white')
    for i,r in enumerate(real[:10]):
        tiles=[Image.open(realroot/r['image']).convert('RGB'),Image.fromarray(targets['real'][i].astype('uint8')*255).convert('RGB')]
        tiles.extend(Image.fromarray(final[a][i].astype('uint8')*255).convert('RGB') for a in ('ordinary','projected'))
        ImageDraw.Draw(canvas).text((2,i*148+2),f'{i} input | target | ordinary | projected',fill='black')
        for j,t in enumerate(tiles):canvas.paste(t.resize((128,128)),(j*128,i*148+20))
    canvas.save(out/'preview.png');(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print({a:{k:v for k,v in r.items() if k!='history'} for a,r in report['arms'].items()});print('masks',report['mask_count'])


if __name__=='__main__':main()
