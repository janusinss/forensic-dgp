"""Verify returned state ancestry/replay tensors and reproduce final real masks."""
import json
from pathlib import Path
import sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from completion_inference import load_completion
from detector_training import load_manifest, ReviewedMasks
from detector_replay import SyntheticMasks
from scripts.train_coverage_vm import freeze_pixels, sha, PARENT_SHA


def main():
    torch.set_num_threads(4)
    root=Path('outputs/downloaded_coverage/outputs/coverage_training_vm')
    parent=Path('outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth')
    assert sha(parent)==PARENT_SHA
    model,state=load_completion(parent,'cpu')
    initial=torch.load(root/'initial.pth',map_location='cpu',weights_only=True)
    assert initial['parent_sha256']==PARENT_SHA
    assert initial['state_dict'].keys()==state['model'].keys()
    assert all(torch.equal(v,state['model'][k]) for k,v in initial['state_dict'].items())
    protocol=json.loads(Path('outputs/coverage_protocol_v1/protocol.json').read_text())
    cached=torch.load(root/'replay_pixels.pth',map_location='cpu',weights_only=True)
    expected={i-68 for epoch in protocol['schedules']['control']['batches'] for batch in epoch for i in batch if i>=68}
    assert set(cached)==expected
    synthetic=SyntheticMasks([r['path'] for r in protocol['sources']],state['size'],42)
    replay_mismatches=[]
    for index,(x,m) in cached.items():
        a,b=freeze_pixels(*synthetic[index])
        if not torch.equal(a,x) or not torch.equal(b,m):
            replay_mismatches.append(index)
    print('Replay cases checked',len(cached),'mismatches',len(replay_mismatches),flush=True)
    rows=[r for r in load_manifest('dataset/detector_glare_review_v3/manifest.json') if r['split']=='validation']
    real=ReviewedMasks(rows,state['size'])
    report={'initial_state_equals_parent':True,'replay_cases':len(cached),
            'replay_regeneration_mismatches':replay_mismatches,'arms':{}}
    run=json.loads((root/'run.json').read_text())
    for arm in ('control','extended'):
        for epoch in range(1,11):
            cp=torch.load(root/arm/f'epoch_{epoch}.pth',map_location='cpu',weights_only=True)
            assert cp['detector_training']==run and cp['coverage_arm']==arm
            assert cp['detector_finetune_epoch']==epoch and cp['detector_only']
            assert all(torch.equal(v,cp['model'][k]) for k,v in state['model'].items() if k.startswith('generator.'))
        model.load_state_dict(cp['model']);model.eval()
        differences=[]
        with torch.inference_mode():
            for i,(x,_) in enumerate(real):
                predicted=(model.detect(x[None]).sigmoid()[0,0]>=.5).numpy()
                saved=np.array(Image.open(root/arm/'epoch_10_masks/real'/f'{i:04}.png'))>0
                count=int(np.count_nonzero(predicted!=saved))
                if count: differences.append({'index':i,'pixels':count})
        report['arms'][arm]={'all_ten_generators_unchanged':True,'final_real_mask_differences_cpu_vs_vm':differences,
                              'final_checkpoint_sha256':sha(root/arm/'epoch_10.pth')}
        print(arm,report['arms'][arm],flush=True)
    Path('outputs/coverage_validation/checkpoint_audit.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
