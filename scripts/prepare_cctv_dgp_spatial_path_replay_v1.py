"""Prepare a small manual saved-path packet; no VM or neural construction."""
import ast
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import tarfile

from cctv_dgp_spatial_path_replay_v1_contract import (
    NAME,STEM,DESIGN_SHA,read,write,sha,validate,verify_transfer,expected_state_hash,label,selected_vector,
)

ROOT=Path(__file__).resolve().parent.parent
DESIGN=ROOT/'outputs/cctv_dgp_spatial_path_replay_v1_design'
SOURCE=ROOT/'outputs/cctv_dgp_spatial_readout_v1_return'
PARENT=ROOT/'outputs/cctv_dgp_bank_photometric_guard_v1_return'


def make_protocol(d,assets):
    fields=['source_protocol_sha256','parent_protocol_sha256','source_required_sha256','parent_required_sha256',
        'states','reference_blocks','case_ids','scientific_thresholds','original_distance_maximum_fraction',
        'common_radius_L2','paired_TRAIN_cases_per_snapshot','snapshot_compositions','additional_fresh_original_controls',
        'visualization','budgets','checkpoint_sha256']
    p={k:d[k] for k in fields}
    p.update(name=NAME,design_sha256=DESIGN_SHA,assets_sha256=assets,
        no_training_or_gradients=True,no_historical_train_call=True,
        source_cwd_for_guarded_inference_only=True,model_qualified=False,automatic_launch=False,
        new_incumbent_selection=False,full_corpus_epochs=0,maximum_new_optimizer_updates=0,
        local_neural_construction_allowed=False,export_external_seconds=630,saved_audit_external_seconds=630,
        status='PREPARED_UNRUN',method_origin='Current fine-tuned DeblurGAN-v2-compatible decoder; external frozen StyleGAN2; saved research states only')
    return validate(p,d)


def main():
    import numpy as np
    import torch  # read-only existing source states and hash byte streaming
    assert sha(DESIGN/'design.json')==DESIGN_SHA
    d=read(DESIGN/'design.json')
    selected=read(DESIGN/'saved_state_selection_audit.json')
    assert selected['complete'] and selected['design_sha256']==DESIGN_SHA
    test=read(DESIGN/'cpu_tests_final.json')
    assert test['complete'] and test['exit_code']==0
    target=ROOT/'outputs'/NAME
    archive=ROOT/'outputs'/(STEM+'-execution.tar.gz')
    assert not target.exists() and not archive.exists()
    controls=[]
    old=read(SOURCE/'protocol.json')
    for route in ['A','B']:
        frozen=torch.load(SOURCE/('outputs/spatial_readout_v1/state_'+route+'0/full_state.pth'),map_location='cpu',weights_only=True)['model']
        layout=read(SOURCE/('outputs/spatial_readout_v1/'+route+'_parameter_layout.json'))
        initial=np.load(SOURCE/('outputs/spatial_readout_v1/'+route+'_initial_vector.npy'),allow_pickle=False)
        for state in [s for s in d['states'] if s['route']==route]:
            v=selected_vector(SOURCE,state,initial)
            h=expected_state_hash(frozen,old['trainable_parameter_names'][route],layout,v)
            if state['update'] in [0,50]:
                assert h==read(SOURCE/('outputs/spatial_readout_v1/state_'+route+str(state['update'])+'/receipt.json'))['model_state']
            controls.append(dict(snapshot=label(state),vector_sha256=state['vector_sha256'],expected_complete_state_hash=h,
                source_tensor_byte_stream_only=True,no_model_mutation=True))
        del frozen
    target.mkdir();(target/'scripts').mkdir();(target/'evidence').mkdir()
    copies={
        'cctv_dgp_spatial_path_replay_v1_contract.py':ROOT/'scripts/cctv_dgp_spatial_path_replay_v1_contract.py',
        'scripts/cctv_dgp_spatial_path_replay_v1_vm.py':ROOT/'scripts/cctv_dgp_spatial_path_replay_v1_vm.py',
        'audit_cctv_dgp_spatial_path_replay_v1_return.py':ROOT/'scripts/audit_cctv_dgp_spatial_path_replay_v1_return.py',
        'design.json':DESIGN/'design.json',
        'evidence/cpu_tests.json':DESIGN/'cpu_tests_final.json',
        'evidence/cpu_tests.txt':DESIGN/'cpu_tests_final.txt',
        'evidence/source_results.md':ROOT/'CCTV_DGP_SPATIAL_READOUT_V1_RESULTS.md',
        'evidence/source_review.md':ROOT/'CCTV_DGP_POST_SPATIAL_READOUT_REVIEW_V1.md',
    }
    for name,path in copies.items():
        assert path.is_file();shutil.copyfile(path,target/name)
    write(target/'evidence/expected_state_hashes.json',dict(complete=True,states=controls,neural_calls=0,weight_assignments=0))
    shell='''#!/usr/bin/env bash
set -euo pipefail
test -n "${TMUX:-}" || { echo "Run manually inside tmux" >&2; exit 1; }
test "$#" -eq 1 || { echo "Pass the exact protocol SHA256" >&2; exit 1; }
python -B -u scripts/cctv_dgp_spatial_path_replay_v1_vm.py --root . --protocol-sha "$1" --supervise
'''
    with (target/'scripts/run_spatial_path.sh').open('x',encoding='utf-8',newline='\n') as s:s.write(shell)
    for path in target.rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'),filename=str(path),feature_version=(3,10))
    assert b'\r' not in (target/'scripts/run_spatial_path.sh').read_bytes()
    files=sorted(p for p in target.rglob('*') if p.is_file())
    assets={p.relative_to(target).as_posix():sha(p) for p in files}
    p=make_protocol(d,assets)
    write(target/'protocol.json',p)
    pin=sha(target/'protocol.json')
    verify_transfer(target,SOURCE,PARENT,pin)
    with archive.open('xb') as s:
        with tarfile.open(fileobj=s,mode='w:gz',compresslevel=9) as pack:
            for path in sorted(v for v in target.rglob('*') if v.is_file()):
                pack.add(path,arcname=NAME+'/'+path.relative_to(target).as_posix(),recursive=False)
    digest=sha(archive)
    with Path(str(archive)+'.sha256').open('x',encoding='utf-8',newline='\n') as s:s.write(digest+'  '+archive.name+'\n')
    record=dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),packet=str(target),
        protocol_sha256=pin,archive_sha256=digest,archive_bytes=archive.stat().st_size,
        packet_assets=len(assets),source_assets=len(p['source_required_sha256']),parent_assets=len(p['parent_required_sha256']),
        saved_states=14,expected_complete_state_hashes=controls,
        status='PREPARED_UNRUN_PENDING_INDEPENDENT_PACKET_AUDIT',neural_calls=0,
        optimizer_updates=0,gradient_queries=0,weight_assignments=0,agent_VM_calls=0,model_qualified=False)
    write(DESIGN/'preparation.json',record)
    print(json.dumps({k:v for k,v in record.items() if k!='expected_complete_state_hashes'}))


if __name__=='__main__':main()
