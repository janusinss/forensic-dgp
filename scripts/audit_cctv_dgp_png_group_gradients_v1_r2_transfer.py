"""Independent saved-source and packet checks; no neural model or gradients."""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import time
import numpy as np

from import_cctv_dgp_png_group_gradients_v1_r2_return import ROOT, OUT, NAME, STEM, PIN, PARENT, read, sha, write
from prepare_cctv_dgp_png_group_gradients_v1_r2 import build_worker, PIXEL_PREFLIGHT


def load(path, label):
    spec=importlib.util.spec_from_file_location(label,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def rejected(fn,*args):
    try:fn(*args)
    except (AssertionError,ValueError):return True
    raise AssertionError('Invalid fixture accepted')


def main():
    start=time.monotonic()
    receipt=OUT/'cctv_dgp_png_group_gradients_v1_r2_transfer_audit.json';assert not receipt.exists()
    root=OUT/NAME;archive=OUT/(STEM+'-execution.tar.gz')
    prep=read(OUT/'cctv_dgp_png_group_gradients_v1_r2_preparation.json');p=read(root/'protocol.json')
    assert prep['complete'] and PIN==sha(root/'protocol.json')==prep['protocol_sha256']
    assert p['name']==NAME and p['stem']==STEM
    assert sha(archive)==prep['archive_sha256'] and archive.stat().st_size==prep['archive_bytes']
    assert Path(str(archive)+'.sha256').read_bytes()==(sha(archive)+'  '+archive.name+'\n').encode('ascii')
    hashes={**p['assets_sha256'],'protocol.json':PIN}
    assert len(hashes)==7 and set(hashes)=={f.relative_to(root).as_posix() for f in root.rglob('*') if f.is_file()}
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();assert len(members)==7
        seen=set()
        for member in members:
            path=PurePosixPath(member.name)
            assert member.isfile() and not member.issym() and not member.islnk()
            assert not path.is_absolute() and '..' not in path.parts and path.parts[0]==NAME
            assert path.as_posix()==member.name and '\\' not in member.name and ':' not in member.name
            name=PurePosixPath(*path.parts[1:]).as_posix();assert name in hashes and name.casefold() not in seen
            seen.add(name.casefold())
            with tar.extractfile(member) as f:data=f.read()
            assert len(data)==member.size==(root/name).stat().st_size
            assert hashlib.sha256(data).hexdigest()==hashes[name]==sha(root/name)
        assert seen=={name.casefold() for name in hashes}
    worker_path=root/'scripts'/(NAME+'.py');worker=worker_path.read_text()
    expected,source_sha=build_worker();assert worker==expected
    tree=ast.parse(worker,feature_version=(3,10))
    names={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and
        isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['NAME','STEM']}
    assert names==dict(NAME=NAME,STEM=STEM)
    assert worker.count(PIXEL_PREFLIGHT)==1 and worker.index(PIXEL_PREFLIGHT)<worker.index('original, _ = load_frozen_dgp_restorer')
    assert worker.index('Initial delivered loss definition differs')<worker.index('torch.autograd.grad')
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in
        ['step','backward','copy_','add_','sub_','Adam','AdamW','SGD'] for n in ast.walk(tree))
    assert sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='grad' for n in ast.walk(tree))==1
    helper_path=root/'cctv_dgp_png_group_gradients_v1_r2_losses.py'
    helper=helper_path.read_text();old_helper=(OUT/'cctv_dgp_png_group_gradients_v1_r1_return/cctv_dgp_png_group_gradients_v1_losses.py').read_text()
    htree=ast.parse(helper,feature_version=(3,10));oldtree=ast.parse(old_helper,feature_version=(3,10))
    newloss=next(n for n in htree.body if isinstance(n,ast.FunctionDef) and n.name=='pixel_and_structure_losses')
    oldloss=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef) and n.name=='pixel_and_structure_losses')
    assert ast.unparse(newloss).replace('ssim_loss = exact + (surrogate - surrogate.detach())',
        'ssim_loss = surrogate + (exact - surrogate).detach()')==ast.unparse(oldloss)
    assert 'byte_array.astype(np.float32) / np.float32(255)' in helper and 'byte_array.astype(np.float64) / 255.' in helper
    assert 'rgb32 = exact32 + (raw - raw.detach())' in helper and 'feature64 = exact64 + (raw64 - raw64.detach())' in helper
    forward_path=OUT/'cctv_dgp_png_group_gradients_v1_r2_forward_audit.json';forward=read(forward_path)
    assert forward['complete'] and forward['saved_forward_cases']==100 and forward['loss_helper_sha256']==sha(helper_path)
    assert forward['prepared_worker_draft_sha256']==sha(worker_path)
    assert forward['all256_byte_levels_checked'] and forward['zero_local_model_inference_autograd_or_optimizer_calls']
    module=load(worker_path,'R2_worker_hash_verify_only')
    verified,parent=module.verify(root.resolve(),PARENT.resolve(),PIN);assert verified==p
    result=subprocess.run([sys.executable,'-B',str(worker_path),'--root',str(root),'--parent-root',str(PARENT),'--protocol-sha',PIN,'--run'],capture_output=True,text=True,timeout=30)
    assert result.returncode!=0 and 'require_vm' in result.stderr and 'AssertionError' in result.stderr
    assert 'torch' not in sys.modules and not (root/'outputs').exists() and not (root/'review.log').exists()
    previous=read(OUT/'cctv_dgp_png_group_gradients_v1_r1_return/protocol.json')
    for key in ['case_ids','reference_blocks','gradient_groups','learning_names','parameters','parent_required_sha256',
        'initial_model_state','preserved_scientific_thresholds','optimizer_updates','full_epochs','parameter_displacements',
        'maximum_gradient_queries','model_forward_cap','recognizer_forward_cap','worker_seconds','measurement_seconds','export_seconds',
        'external_seconds','export_external_seconds','kill_grace_seconds','minimum_free_bytes','disk_reserve_bytes',
        'output_byte_cap','peak_VRAM_bytes','peak_RSS_bytes','forward_loss_numeric_tolerances']:
        assert p[key]==previous[key]
    checker_path=ROOT/'scripts/audit_cctv_dgp_png_group_gradients_v1_r2_return.py'
    checker=load(checker_path,'R2_saved_return_checker_fixtures')
    baseline=read(PARENT/'outputs/bank_comparison_v1/baseline/metrics.json')['rows']
    selected=[r for r in baseline if r['id'] in p['case_ids']];assert len(selected)==100
    for group in p['gradient_groups']:
        assert np.isfinite(checker.baseline_value(group,selected)) and rejected(checker.baseline_value,group,baseline)
    v=np.array([1.,2.,3.],np.float64);d={'balanced':np.array([-1.,2.,-3.],np.float64)}
    row=dict(gradient_norm=float(np.linalg.norm(v)),saved_direction_dot={'balanced':float(v@d['balanced'])})
    assert checker.check_vector(v,row,d)==row['gradient_norm']
    assert rejected(checker.check_vector,v.astype(np.float32),row,d)
    assert rejected(checker.check_vector,v,dict(row,gradient_norm=0),d)
    assert rejected(checker.check_vector,v,dict(row,saved_direction_dot={'balanced':0}),d)
    assert 'canonical_PNG_pixel_table.npz' in checker_path.read_text() and 'view(np.uint64)' in checker_path.read_text()
    importer=ROOT/'scripts/import_cctv_dgp_png_group_gradients_v1_r2_return.py'
    assert sha(importer)==prep['importer_sha256'] and sha(checker_path)==prep['prospective_return_auditor_sha256']
    protected=read(OUT/'cctv_dgp_png_group_gradients_v1_r1_analysis/local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    manifest=read(OUT/'cctv_dgp_png_group_gradients_v1_r1_return/export_manifest.json')
    assert all(sha(OUT/'cctv_dgp_png_group_gradients_v1_r1_return'/name)==digest for name,digest in manifest['files'].items())
    shell=(root/'scripts/run_png_groups.sh').read_text();assert '\r' not in shell and shell.count(NAME+'.py')==3
    assert shell.count('statuses=("${PIPESTATUS[@]}")')==2
    assert 'timeout --signal=TERM --kill-after=30 930' in shell and 'timeout --signal=TERM --kill-after=30 330' in shell
    assert time.monotonic()-start<180
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        protocol_sha256=PIN,archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
        exact_regular_archive_files_checked=7,packet_assets=6,parent_assets=len(p['parent_required_sha256']),
        saved_forward_cases=100,all256_CPU_byte_levels_passed=True,CUDA_byte_preflight_not_yet_executed=True,
        old_failed_R1_return_and_sources_preserved=True,mathematical_loss_definition_and_all_gates_unchanged=True,
        same320_queries_zero_updates_finite_limits_and_parent_states=True,exact_CPU_canonical_values_copied_to_device=True,
        scientific_SSIM_forward_and_approximate_gradient_expression_preserved=True,
        worker_hash_only_verify_passed=True,Windows_pre_neural_run_guard_passed=True,
        selected_group_fixtures_checked=32,full_parent_group_scope_rejected=32,gradient_invalid_fixtures_rejected=3,
        prospective_return_checker_sha256=sha(checker_path),importer_sha256=sha(importer),
        local_neural_model_gradient_optimizer_calls=0,no_VM_connection_or_launch=True,
        model_qualified=False,more_epochs_qualified=False,goal_complete=False,seconds=time.monotonic()-start))
    print(dict(complete=True,packet_assets=6,parent_assets=5852,saved_forward_cases=100,local_neural_or_gradient_calls=0))


if __name__=='__main__':main()
