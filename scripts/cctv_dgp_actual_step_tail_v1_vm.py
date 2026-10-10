"""Manual, finite completion of the storage-stopped diagnostic's final state.

Re-evaluate the three fixed update45 proposals in a separate output directory.
The two completed controls and all196 partial cone files must replay exactly.
No optimizer, autograd, learned checkpoint, overwrite or cleanup is permitted.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback

NAME = 'cctv_dgp_actual_step_tail_v1_vm'
PARENT = 'cctv_dgp_actual_step_review_v1_vm'
STEM = 'cctv-dgp-actual-step-tail-v1'
PARENT_PIN = '339c20425b6fa6b4141001aae4ff3010bf9d438acc32eb71ec08f04f001476f5'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False);stream.write('\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_transfer(root, parent, pin):
    assert root.is_dir() and root.name == NAME and parent.is_dir() and parent.name == PARENT
    assert sha(root / 'protocol.json') == pin
    p = read(root / 'protocol.json')
    assert p['format'] == 'finite-inference-final-state-storage-recovery-v1'
    assert p['parent_protocol_sha256'] == PARENT_PIN == sha(parent / 'protocol.json')
    assert p['optimizer_updates'] == p['gradient_queries'] == 0 and not p['new_trained_checkpoint']
    assert not p['app_promotion'] and not p['automatic_follow_on'] and p['manual_VM_execution_required']
    assert p['updates'] == [45] and p['proposals'] == ['zero', 'recorded', 'cone'] and p['forward_slots'] == 315
    assert p['completed_controls_repeated_only_for_exact_overlap_validation']
    assert p['original_storage_failure_retained'] and p['original_capacity_and_preservation_gates_unchanged']
    assert p['budgets'] == {'cache_seconds':120,'review_seconds':300,'worker_seconds':600,'export_seconds':300,
        'minimum_free_bytes':2*1024**3,'maximum_allocated_VRAM_bytes':20*1024**3,
        'maximum_output_bytes':512*1024**2,'maximum_member_bytes':16*1024**2,
        'maximum_return_members':1100,'maximum_forward_slots':315,'timing_safety_factor':1.25}
    for name, digest in p['assets_sha256'].items():
        path = root / name
        assert not path.is_symlink() and path.resolve().is_relative_to(root) and sha(path) == digest, name
    sys.path.insert(0, str(parent))
    from cctv_dgp_actual_step_review_v1_contract import verify
    parent_p = verify(parent, PARENT_PIN)
    worker_path = parent / 'scripts/review_cctv_dgp_actual_steps_v1_vm.py'
    assert sha(worker_path) == p['original_worker_sha256']
    source = worker_path.read_text(encoding='utf-8')
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'review')
    original = ast.get_source_segment(source, node)
    adapted = original
    for row in p['review_source_changes']:
        assert adapted.count(row['before']) == 1
        adapted = adapted.replace(row['before'], row['after'], 1)
    inverse = adapted
    for row in reversed(p['review_source_changes']):
        assert inverse.count(row['after']) == 1
        inverse = inverse.replace(row['after'], row['before'], 1)
    assert inverse == original
    assert hashlib.sha256(adapted.encode()).hexdigest() == p['adapted_review_function_sha256']
    return p, parent_p, adapted


def vm_scope(root, parent):
    assert sys.platform == 'linux' and platform.node().split('.')[0] == 'forensic-dgp-thesis', 'Existing Linux VM only; no local model calls'
    assert root == (Path.home() / 'forensic-dgp' / NAME).resolve()
    assert parent == (Path.home() / 'forensic-dgp' / PARENT).resolve()


def check_parent_return(root, parent, p):
    assert not root.joinpath('outputs').exists(), 'Retain every tail attempt; no automatic repeat'
    for name, digest in p['retained_parent_return_sha256'].items():
        path = parent / name
        assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(parent) and sha(path) == digest, name
    failure = read(parent / 'outputs/failure.json')
    assert failure['error'] == 'Return-storage limit; retain partial review'
    assert failure['restoration'] == {'attempted':True,'decoder_restored':True}
    assert failure['progress']['completed_conditions'] == 29 and failure['progress']['forward_slots'] == 3110
    assert failure['progress']['optimizer_updates'] == failure['progress']['gradient_queries'] == failure['progress']['backward_calls'] == 0
    assert not (parent / 'outputs/results.json').exists()


def overlap(root, parent, p):
    import numpy as np
    from PIL import Image
    checked = []
    for name, digest in p['overlap_files_sha256'].items():
        source = parent / name;destination = root / name
        assert sha(source) == digest and destination.is_file(), name
        if source.suffix == '.npz':
            with np.load(source, allow_pickle=False) as a, np.load(destination, allow_pickle=False) as b:
                assert a.files == b.files
                for key in a.files:
                    x, y = a[key], b[key]
                    assert x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y), (name, key)
        elif source.suffix == '.png':
            with Image.open(source) as a, Image.open(destination) as b:
                assert a.mode == b.mode == 'RGB' and a.size == b.size == (256,256)
                assert np.array_equal(np.asarray(a), np.asarray(b)), name
        elif source.suffix == '.json':
            assert read(source) == read(destination), name
        else:
            raise AssertionError('Unexpected overlap file')
        checked.append(name)
    assert len(checked) == 828
    write(root / 'outputs/overlap_receipt.json', {'complete':True,'files_checked':len(checked),'all_arrays_pixels_and_saved_control_metrics_exact':True,
        'overlap_files':checked,'original_failure_sha256':p['retained_parent_return_sha256']['outputs/failure.json'],
        'optimizer_updates':0,'gradient_queries':0,'new_trained_checkpoint':False,'app_promotion':False})


def review(root, parent, p, parent_p, adapted):
    check_parent_return(root, parent, p)
    worker_path = parent / 'scripts/review_cctv_dgp_actual_steps_v1_vm.py'
    spec = importlib.util.spec_from_file_location('trusted_original_actual_step_worker', worker_path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    namespace = dict(module.__dict__)
    namespace.update({'TAIL_ROOT':root,'BUDGETS':dict(p['budgets'])})
    exec(compile(adapted, '<source-bound-final-state-only-review>', 'exec'), namespace)
    namespace['review'](parent, parent_p, PARENT_PIN)
    overlap(root, parent, p)
    results = read(root / 'outputs/results.json')
    assert results['progress']['forward_slots'] == 315 and results['progress']['completed_conditions'] == 3
    assert results['progress']['parameter_proposals_loaded'] == 3
    assert results['source_forward_counts'] == {'original_DGP':121,'decoder':92,'reference_decoder':92,'recognizer':155}
    assert results['final_restored_states'] == parent_p['initial_states']
    write(root / 'outputs/tail_scope_receipt.json', {'complete':True,'protocol_sha256':sha(root/'protocol.json'),'parent_protocol_sha256':PARENT_PIN,
        'conditions':3,'forward_slots':315,'overlap_files_checked':828,'original_storage_failure_retained':True,
        'optimizer_updates':0,'gradient_queries':0,'new_trained_checkpoint':False,'app_promotion':False,'full_TRAIN_capacity_pass':False,
        'standalone_tail_completion_does_not_relabel_original_failure':True})


def allowed_return(parent_p):
    from cctv_dgp_actual_step_review_v1_contract import allowed_return_names
    names = {'protocol.json','export_manifest.json','review.log','review_exit_code.txt','supervisor_receipt.json',
        'outputs/preflight.json','outputs/cache_receipt.json','outputs/timing_projection.json','outputs/results.json','outputs/failure.json',
        'outputs/overlap_receipt.json','outputs/tail_scope_receipt.json','outputs/tail_failure.json'}
    names |= {name for name in allowed_return_names(parent_p) if name.startswith('outputs/probes/update0045/')}
    return names


def export(root, p, parent_p):
    started = time.monotonic()
    archive = Path.home() / (STEM + '-results.tar.gz')
    checksum = Path(str(archive) + '.sha256')
    receipt = Path.home() / (STEM + '-export.json')
    assert all(not path.exists() for path in [archive,checksum,receipt,root/'export_manifest.json'])
    assert root.joinpath('outputs').is_dir()
    files, total = {}, 0
    for name in sorted(allowed_return(parent_p) - {'export_manifest.json'}):
        path = root / name
        if path.is_file():
            assert not path.is_symlink() and path.resolve().is_relative_to(root)
            assert path.stat().st_size <= p['budgets']['maximum_member_bytes']
            total += path.stat().st_size
            assert total <= p['budgets']['maximum_output_bytes'] - 8*1024**2
            assert time.monotonic()-started < p['budgets']['export_seconds']
            files[name] = sha(path)
    complete = (root/'outputs/tail_scope_receipt.json').exists()
    write(root/'export_manifest.json', {'complete':True,'protocol_sha256':sha(root/'protocol.json'),'files_sha256':files,
        'diagnostic_completed':complete,'original_storage_failure_retained':True,'optimizer_updates':0,'gradient_queries':0,'new_trained_checkpoint':False})
    with tarfile.open(archive,'w:gz',compresslevel=3) as tar:
        for name in sorted(set(files)|{'export_manifest.json'}):
            assert time.monotonic()-started < p['budgets']['export_seconds']
            tar.add(root/name,arcname=NAME+'_return/'+name,recursive=False)
    digest = sha(archive)
    with checksum.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(digest+'  '+archive.name+'\n')
    value = {'complete':True,'archive_sha256':digest,'bytes':archive.stat().st_size,'seconds':time.monotonic()-started,
        'tail_completed':complete,'optimizer_updates':0,'gradient_queries':0,'new_trained_checkpoint':False,'training_success_not_implied':True}
    write(receipt,value);print(value,flush=True)


def supervise(root, parent, p):
    assert not (root/'supervisor_receipt.json').exists() and not (root/'review.log').exists() and not root.joinpath('outputs').exists()
    calls = []
    for flag, logfile, cap in [('--review','review.log',630),('--export','export.log',330)]:
        command = [sys.executable,'-B','-u',str(Path(__file__)),'--root',str(root),'--parent-root',str(parent),'--protocol-sha',sha(root/'protocol.json'),flag]
        timed_out = False
        with (root/logfile).open('xb') as log:
            child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            print({'child_pid':child.pid,'log':logfile,'external_cap_seconds':cap},flush=True)
            try:
                code = child.wait(timeout=cap)
            except subprocess.TimeoutExpired:
                timed_out = True;os.killpg(child.pid,signal.SIGTERM)
                try:code=child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=10)
        calls.append({'flag':flag,'exit_code':code,'timeout':timed_out,'external_cap_seconds':cap})
        print({'flag':flag,'exit_code':code,'timeout':timed_out},flush=True)
        if flag=='--review':
            with (root/'review_exit_code.txt').open('x',encoding='utf-8') as stream:stream.write(str(code)+'\n')
            if code and not (root/'outputs/tail_failure.json').exists():
                write(root/'outputs/tail_failure.json',{'complete':False,'review_exit_code':code,'timeout':timed_out,
                    'original_failure_retained':True,'optimizer_updates':0,'gradient_queries':0,'new_trained_checkpoint':False,'app_promotion':False})
            write(root/'supervisor_receipt.json',{'complete':code==0,'calls':list(calls),'review_exit_code':code,
                'optimizer_updates':0,'gradient_queries':0,'new_trained_checkpoint':False,'automatic_training_follow_on':False,'app_promotion':False})
    return 0 if all(row['exit_code']==0 for row in calls) else 1


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--parent-root',type=Path,required=True)
    parser.add_argument('--protocol-sha',required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    for name in ['verify-transfer','review','export','supervise']:group.add_argument('--'+name,action='store_true')
    args=parser.parse_args();root=args.root.resolve();parent=args.parent_root.resolve()
    if not args.verify_transfer:vm_scope(root,parent)
    p,parent_p,adapted=verify_transfer(root,parent,args.protocol_sha)
    if args.verify_transfer:
        print({'complete':True,'packet_assets':len(p['assets_sha256']),'parent_assets_verified':len(parent_p['assets_sha256']),
            'forward_slots_planned':315,'optimizer_updates':0,'gradient_queries':0,'neural_calls':0},flush=True);return
    if args.supervise:sys.exit(supervise(root,parent,p))
    if args.export:export(root,p,parent_p);return
    try:
        review(root,parent,p,parent_p,adapted)
    except BaseException as error:
        if not (root/'outputs/tail_failure.json').exists():
            write(root/'outputs/tail_failure.json',{'complete':False,'error':str(error),'type':type(error).__name__,'traceback':traceback.format_exc(),
                'original_storage_failure_retained':True,'optimizer_updates':0,'gradient_queries':0,'new_trained_checkpoint':False,'app_promotion':False})
        raise


if __name__=='__main__':main()
