"""Independent packet, state/schema and guard checks; no neural/gradient calls."""
import ast
from datetime import datetime, timezone
import hashlib
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile

import numpy as np
import torch

from import_cctv_dgp_joint_raw_png_forward_v1_return import ROOT, OUT, ANALYSIS, PARENT, read, sha, write


def state_hash(state):
    digest=hashlib.sha256()
    for key in sorted(state):
        value=state[key].detach().cpu().contiguous()
        for part in [key,str(value.dtype),str(tuple(value.shape))]:digest.update(part.encode()+b'\0')
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main():
    receipt=OUT/'cctv_dgp_live_conditioning_v1_r1_transfer_audit.json'
    assert not receipt.exists()
    packet=OUT/'cctv_dgp_live_conditioning_v1_r1_vm'
    archive=OUT/'cctv-dgp-live-conditioning-v1-r1-execution.tar.gz'
    prepared=read(OUT/'cctv_dgp_live_conditioning_v1_r1_preparation.json')
    p=read(packet/'protocol.json');pin=sha(packet/'protocol.json')
    assert prepared['complete'] and prepared['unrun'] and not prepared['training_launch_ready']
    assert pin==prepared['protocol_sha256'] and sha(archive)==prepared['archive_sha256']
    assert archive.stat().st_size==prepared['archive_bytes']
    checksum=Path(str(archive)+'.sha256').read_bytes()
    assert b'\r' not in checksum and checksum==f"{prepared['archive_sha256']}  {archive.name}\n".encode('ascii')
    expected={**p['assets_sha256'],'protocol.json':pin}
    assert len(expected)==8 and all(sha(packet/name)==digest for name,digest in expected.items())
    assert not (packet/'outputs').exists() and not (packet/'review.log').exists()
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();assert len(members)==8
        names=set()
        for member in members:
            path=PurePosixPath(member.name)
            assert member.isfile() and not member.islnk() and not member.issym()
            assert path.parts[0]==packet.name and not path.is_absolute() and '..' not in path.parts
            name=PurePosixPath(*path.parts[1:]).as_posix()
            assert name in expected and name not in names and '\\' not in name and ':' not in name
            with tar.extractfile(member) as stream:assert hashlib.sha256(stream.read()).hexdigest()==expected[name]
            assert member.size==(packet/name).stat().st_size;names.add(name)
        assert names==set(expected)
    parent_bytes=0
    for name,digest in p['parent_required_sha256'].items():
        path=PARENT/name
        assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(PARENT.resolve())
        assert sha(path)==digest;parent_bytes+=path.stat().st_size
    assert len(p['parent_required_sha256'])==5852
    torch.set_grad_enabled(False)
    initial=torch.load(PARENT/'initializers/initial_B.pth',map_location='cpu',weights_only=True)
    with np.load(packet/'readouts.npz',allow_pickle=False) as arrays: readouts={k:arrays[k] for k in arrays.files}
    virtual={'live.'+n:v for n,v in initial.items()}
    for size in [16,32,64,128,256]:
        weight=readouts[f'f{size}_weight'];bias=readouts[f'f{size}_bias']
        assert weight.dtype==np.float32 and tuple(weight.shape)==tuple(initial[f'fusions.{size}.weight'].shape)
        assert np.isfinite(weight).all() and np.count_nonzero(bias)==0
        matrix=weight[:,:,0,0].astype(np.float64)
        assert np.allclose(matrix@matrix.T,np.eye(len(matrix))*.015625,rtol=0,atol=2e-9)
        assert np.linalg.norm(matrix)>0
        for term,array in [('weight',weight),('bias',bias)]:
            value=torch.from_numpy(array)
            virtual[f'live.fusions.{size}.{term}']=value
            virtual[f'anchor_fusions.{size}.{term}']=value
    for name,value in initial.items():
        if name.startswith('style_conditioner.'): virtual['anchor_style_conditioner.'+name[18:]]=value
        if name.startswith('spatial_conditioners.'): virtual['anchor_spatial_conditioners.'+name[21:]]=value
    # Verify the prefix slices explicitly; a dropped first character invalidates
    # the full state hash even though tensor values would remain plausible.
    assert 'anchor_style_conditioner.0.weight' in virtual
    assert 'anchor_spatial_conditioners.8.weight' in virtual
    assert state_hash(virtual)==p['initial_model_state']==prepared['virtual_initial_state_sha256']
    assert state_hash(initial)==p['parent_initial_B_state']
    assert len(p['learning_names'])==37 and sum(int(np.prod(s)) for s in p['selected_parameter_shapes'].values())==3364643
    old=read(PARENT/'protocol.json');assert p['preserved_scientific_thresholds']==old['scientific_thresholds']
    assert p['optimizer_updates']==p['backwards']==p['full_epochs']==p['parameter_displacements']==0
    assert p['maximum_gradient_queries']==320 and p['model_forward_cap']==40 and p['feature_branch_passes_cap']==80
    assert p['conditioning_tensors_required']==12 and p['nonzero_readout_row_L2']==.125
    assert p['minimum_free_bytes']==4*1024**3 and p['disk_reserve_bytes']==1024**3
    assert p['output_byte_cap']==1280*1024**2 and p['peak_VRAM_bytes']==20*1024**3 and p['peak_RSS_bytes']==12*1024**3
    assert (p['worker_seconds'],p['measurement_seconds'],p['external_seconds'])==(900,600,930)
    assert (p['export_seconds'],p['export_external_seconds'],p['kill_grace_seconds'])==(300,330,30)
    worker=packet/'scripts/cctv_dgp_live_conditioning_v1_r1_vm.py'
    code=worker.read_text(encoding='utf-8');ast.parse(code,feature_version=(3,10))
    model=(packet/'cctv_dgp_live_conditioning_v1_r1_model.py').read_text(encoding='utf-8')
    ast.parse(model,feature_version=(3,10))
    assert 'candidate = AnchoredBankDGP(candidate,' in code and 'candidate.branch_calls==dict(live=40,anchor=40)' in code
    assert 'value>0' not in code or "v>0" in code
    assert "np.isfinite(v) and v>0 for v in live_norms[source_name+term].values()" in code
    assert "self.live.fusions[key](features[key])-reference[key]" in model
    assert 'copy.deepcopy(self.live.style_conditioner)' in model and 'copy.deepcopy(self.live.spatial_conditioners)' in model
    assert 'with torch.no_grad():' in model and 'reference={key:self.anchor_fusions[key](value)' in model
    assert 'require_manual(root)' in model and 'require_manual(self.root)' in model
    assert '.detach()' not in model, 'No always-zero straight-through output construction'
    assert 'optimizer' not in model and 'target' not in model[model.index('    def forward'):]
    for filename,key in [('import_cctv_dgp_live_conditioning_v1_r1_return.py','future_importer_sha256'),
                         ('audit_cctv_dgp_live_conditioning_v1_r1_return.py','future_auditor_sha256')]:
        path=ROOT/'scripts'/filename;assert sha(path)==prepared[key];ast.parse(path.read_text(encoding='utf-8'),feature_version=(3,10))
    import sys as system
    system.path.insert(0,str(packet))
    from cctv_dgp_live_conditioning_v1_r1_contract import scopes,fixed_reference_matrices,projection_seconds
    blocks,groups=scopes(old['cases'],p['case_ids'])
    assert blocks==p['reference_blocks'] and groups==p['gradient_groups']
    refs={r['id']:r for r in old['references']}
    assert len(fixed_reference_matrices([c for c in old['cases'] if c['id'] in p['case_ids']],refs))==100
    assert projection_seconds([2.,3.])==90 and projection_seconds([21.,22.])==660>p['measurement_seconds']
    verification=subprocess.run([sys.executable,'-B',str(worker),'--root',str(packet),'--parent-root',str(PARENT),
        '--protocol-sha',pin,'--verify-transfer'],capture_output=True,text=True,check=True,timeout=180)
    verified=ast.literal_eval(verification.stdout.strip())
    assert verified['complete'] and verified['packet_assets']==7 and verified['parent_assets']==5852
    assert verified['actual_neural_or_gradient_calls']==verified['optimizer_updates']==0
    guard_code="import runpy,sys,json;sys.argv="+repr([str(worker),'--root',str(packet),'--parent-root',str(PARENT),'--protocol-sha',pin,'--run'])+"\ntry: runpy.run_path("+repr(str(worker))+",run_name='__main__')\nexcept AssertionError: print(json.dumps({'Windows_run_rejected_before_Torch_import':'torch' not in sys.modules}))\nelse: raise RuntimeError('Windows run was not rejected')"
    guard=subprocess.run([sys.executable,'-B','-c',guard_code],capture_output=True,text=True,check=True,timeout=30)
    assert read_json_line(guard.stdout)['Windows_run_rejected_before_Torch_import']
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        protocol_sha256=pin,archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,archive_files=8,
        parent_assets_verified=5852,parent_bytes_verified=parent_bytes,LF_ASCII_checksum_exact=True,
        virtual_full_initial_state_hash_independently_reconstructed=True,new_frozen_reference_tensors=22,
        new_readouts_nonzero_and_same_input_anchor_distinct_from_live_path=True,
        all100_fixed_reference_affines_and32_scopes_checked=True,static_counts_timing_storage_limits_checked=True,
        Windows_run_rejected_before_Torch_import=True,hash_only_transfer_preflight_passed=verified,
        future_importer_and_auditor_hashes_verified=True,
        CUDA_parity_and_conditioning_gradient_activation_unrun_and_unproven=True,
        no_local_model_construction_forward_gradient_optimizer_or_parameter_displacement=True,
        no_VM_connection_upload_or_launch=True,original_app_and_research_bindings_unchanged=True,
        diagnostic_manual_launch_ready_after_shell_syntax_check=True,training_launch_ready=False,
        model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,archive_files=8,parent_assets=5852,Windows_rejection_before_Torch=True,
        initial_state_bytes_checked=True,neural_calls=0,VM_run_unproven=True))


def read_json_line(text):
    import json
    return json.loads(text.strip())


if __name__=='__main__':main()
