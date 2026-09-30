"""Recreate reported initialization on original VM; no fitting or optimizer."""
import argparse,hashlib,json,sys,tarfile
from pathlib import Path
import torch


def main(root):
    if not torch.cuda.is_available():raise RuntimeError('Use original VM GPU/runtime')
    root=Path(root).resolve();sys.path.insert(0,str(root));sys.path.insert(0,str(root/'refinement_experiment'))
    from refinement_head import RefinementHead
    import refinement_head
    from scripts.compare_presence_heads_vm import PresenceHead
    from scripts.train_expanded_feature_vm import head_digest
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    protocol_path=root/'outputs/refinement_training/protocol.json';protocol=json.loads(protocol_path.read_text())
    assert str(torch.__version__)==protocol['torch'],'Original runtime required'
    assert sha(refinement_head.__file__)==protocol['module_sha256']
    assert sha(root/'refinement_experiment/train_refinement_vm.py')==protocol['script_sha256']
    parent=root/'outputs/expanded_border_training/control_epoch_10.pth';assert sha(parent)==protocol['parent_sha256']
    state=torch.load(parent,weights_only=True,map_location='cpu')
    torch.manual_seed(42)
    model=RefinementHead(state['pixel'],use_rgb=True).cuda()
    gate=PresenceHead(4).cuda().eval().requires_grad_(False);gate.load_state_dict(state['presence'])
    digest=head_digest(model,gate)
    out=root/'outputs/refinement_initial_audit';out.mkdir(exist_ok=False)
    result={'matches':digest==protocol['initial_heads_sha256'],'recreated_digest':digest,
            'reported_digest':protocol['initial_heads_sha256'],'protocol_sha256':sha(protocol_path),
            'torch':str(torch.__version__),'optimizer_updates':0,'historical_states_exported':False,
            'note':'Recreated initial tensors, not independently captured historical training states.'}
    torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'presence':{k:v.detach().cpu() for k,v in gate.state_dict().items()}},out/'initial.pth')
    result['initial_sha256']=sha(out/'initial.pth');(out/'results.json').write_text(json.dumps(result,indent=2))
    archive=root/'refinement-initial-audit.tar.gz'
    if archive.exists():raise RuntimeError('Preserve existing archive')
    with tarfile.open(archive,'w:gz') as tar:
        for path in out.iterdir():tar.add(path,arcname=path.name)
        tar.add(__file__,arcname='export_refinement_initial_vm.py')
    print(json.dumps(result,indent=2));print('DONE:',archive)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='.')
    main(p.parse_args().root)
