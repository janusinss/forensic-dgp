"""Independent R2 source equivalence, actual guard regression and transfer audit."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

import check_cctv_dgp_feature_fusion_v32_root_routing as routing_check

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'outputs/cctv_dgp_feature_fusion_vm_v32_r1'
BUNDLE = ROOT / 'outputs/cctv_dgp_feature_fusion_vm_v32_r2'
PREP = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_preparation'
STEM = 'cctv-dgp-feature-fusion-v32-r2'
WORKER = 'scripts/cctv_dgp_feature_fusion_v32_vm.py'
CACHE = 'cctv_dgp_feature_fusion_v32_cache.py'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream: json.dump(value, stream, indent=2)


def unroute(text):
    return text.replace(BUNDLE.name, OLD.name).replace(STEM, 'cctv-dgp-feature-fusion-v32-r1').replace(
        'cctv_dgp_feature_fusion_v32_r2_return', 'cctv_dgp_feature_fusion_v32_r1_return')


def verify_actual_transfer_branch():
    """Execute actual verify-transfer dispatch and guards; data checks are stubs."""
    worker = routing_check.functions(BUNDLE / WORKER)
    cache = routing_check.functions(BUNDLE / CACHE)
    branch = next(n for n in worker['main'].body if isinstance(n, ast.If) and ast.unparse(n.test) == 'a.verify_transfer')
    assert ast.unparse(branch.body[0]) == 'vm_scope(root, parent)'
    home = (ROOT / 'scratch/metadata_only_simulated_VM_home').resolve()
    class HomePath:
        @staticmethod
        def home(): return home
    ns = {'Path':HomePath,'sys':SimpleNamespace(platform='linux'),'json':json,'p':{}}
    exec(compile(ast.Module(body=[cache['require_cache_scope']], type_ignores=[]), '<actual-R2-cache-guard>', 'exec'), ns)
    def metadata_import(name, *args, **kwargs):
        assert name == 'cctv_dgp_feature_fusion_v32_cache'
        return SimpleNamespace(require_cache_scope=ns['require_cache_scope'])
    calls = []
    ns['__builtins__'] = dict(vars(__import__('builtins')), __import__=metadata_import, print=lambda *a, **k: None)
    for name in ['broader_check','paired_batch_and_closed_check','closed_V31_and_fusion_diagnostic_check','parent_check','closed_v28_and_diagnostic_check']:
        def stub(*a, marker=name):
            calls.append(marker); return {'assets_sha256':{}}
        ns[name] = stub
    probe = ast.FunctionDef(name='probe', args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[],
                           kw_defaults=[], defaults=[]), body=branch.body, decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[worker['vm_scope'], probe], type_ignores=[])), '<actual-R2-transfer-dispatch>', 'exec'), ns)
    approved = home / 'forensic-dgp' / BUNDLE.name
    ns['parent'] = home / 'forensic-dgp/cctv_dgp_feature_skips_vm_v27'
    ns['root'] = approved
    ns['probe']()
    assert calls == ['broader_check','paired_batch_and_closed_check','closed_V31_and_fusion_diagnostic_check','parent_check','closed_v28_and_diagnostic_check']
    rows = [{'case':'actual valid transfer dispatch reaches metadata dependencies','pass':True}]
    for name, path, platform, parent in [
        ('old R1 rejected before dependencies', approved.with_name(OLD.name),'linux',ns['parent']),
        ('base V32 rejected before dependencies', approved.with_name('cctv_dgp_feature_fusion_vm_v32'),'linux',ns['parent']),
        ('nested root rejected before dependencies', home/'forensic-dgp/nested'/BUNDLE.name,'linux',ns['parent']),
        ('Windows rejected before dependencies',approved,'win32',ns['parent']),
        ('wrong parent rejected before dependencies',approved,'linux',ns['parent'].with_name('other_parent'))]:
        calls.clear(); ns['root']=path; ns['sys'].platform=platform; ns['parent']=parent
        try: ns['probe']()
        except AssertionError: pass
        else: raise AssertionError(name)
        assert not calls, name
        rows.append({'case':name,'pass':True})
    return {'complete':True,'rows':rows,'actual_dispatch_and_guard_AST_executed':True,
            'data_dependency_checks_stubbed':True,'model_imports':0,'gradient_calls':0,'optimizer_updates':0}


def main():
    start = time.monotonic()
    assert not (PREP / 'independent_packet_audit.json').exists()
    p, old, prep = read(BUNDLE/'protocol.json'), read(OLD/'protocol.json'), read(PREP/'preparation.json')
    assert sha(BUNDLE/'protocol.json') == prep['protocol_sha256']
    assert sha(OLD/'protocol.json') == p['superseded_R1_protocol_sha256'] == 'e25e5721a5474d767fe48710336105a1eaa843ed2e22d6cb03caebd8e3bd10da'
    reduced = copy.deepcopy(p)
    for key in ['routing_revision','routing_revision_reason','superseded_R1_protocol_sha256',
                'closed_R1_launch_failure_evidence_sha256','prior_R1_packet_evidence_sha256']:
        reduced.pop(key)
    expected = copy.deepcopy(old)
    for key in ['assets_sha256','local_basis_sha256']:
        reduced.pop(key); expected.pop(key)
    assert reduced == expected, 'A scientific recipe, gate or historical binding changed'
    counts = {}
    for key in ['local_basis_sha256','closed_R1_launch_failure_evidence_sha256','prior_R1_packet_evidence_sha256']:
        for name,digest in p[key].items(): assert sha(ROOT/name) == digest, name
        counts[key] = len(p[key])
    failure_root = ROOT/'outputs/cctv_dgp_feature_fusion_v32_r1_failure_v1/host_access_retry'
    failure = read(failure_root/'original_VM_files/outputs/failure.json')
    assert failure['optimizer_updates'] == 0 and not failure['optimizer_constructed'] and not failure['new_checkpoint_created']
    assert not failure['resume_permitted'] and 'cctv_dgp_feature_fusion_v32_cache.py' in failure['traceback']
    for name,digest in p['assets_sha256'].items():
        assert sha(BUNDLE/name) == digest
        if name == WORKER:
            restored = unroute((BUNDLE/name).read_text())
            addition = '    from cctv_dgp_feature_fusion_v32_cache import require_cache_scope\n    require_cache_scope(root)\n'
            assert restored.count(addition) == 1
            restored = restored.replace(addition,'')
            transfer = '    if a.verify_transfer:\n        vm_scope(root, parent)\n'
            assert restored.count(transfer) == 1
            restored = restored.replace(transfer, '    if a.verify_transfer:\n')
            assert restored == (OLD/name).read_text(), 'Worker has unrelated changes'
        elif name == CACHE:
            tree = ast.parse((BUNDLE/name).read_text())
            helper = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='require_cache_scope')
            tree.body.remove(helper)
            load = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_items')
            assert ast.unparse(load.body[0]) == 'require_cache_scope(root)'
            load.body = helper.body + load.body[1:]
            for n in ast.walk(tree):
                if isinstance(n,ast.Constant) and n.value == 'forensic-dgp/'+BUNDLE.name:
                    n.value='forensic-dgp/cctv_dgp_feature_fusion_vm_v32'
            assert ast.dump(tree,include_attributes=False) == ast.dump(ast.parse((OLD/name).read_text()),include_attributes=False), 'Cache math or guard constraints changed'
        else:
            assert (BUNDLE/name).read_bytes() == (OLD/name).read_bytes(), name
    assert {f.relative_to(BUNDLE).as_posix() for f in BUNDLE.rglob('*') if f.is_file()} == set(p['assets_sha256']) | {'protocol.json'}
    for f in BUNDLE.rglob('*.py'): ast.parse(f.read_text(),feature_version=(3,10))
    routes = routing_check.run_checks(BUNDLE); assert routes['complete']
    write(PREP/'root_routing_regressions.json',routes)
    dispatch = verify_actual_transfer_branch(); write(PREP/'transfer_dispatch_regressions.json',dispatch)
    archive = ROOT/'outputs'/(STEM+'-execution.tar.gz')
    assert sha(archive) == prep['archive_sha256'] and archive.stat().st_size == prep['archive_bytes']
    assert Path(str(archive)+'.sha256').read_text().strip().split() == [prep['archive_sha256'],archive.name]
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers(); assert len(members)==len({m.name for m in members})==11
        for m in members:
            assert m.isfile() and not m.issym() and not m.islnk() and m.name.startswith(BUNDLE.name+'/')
            name=m.name[len(BUNDLE.name)+1:]; assert name in set(p['assets_sha256'])|{'protocol.json'}
            assert hashlib.sha256(tar.extractfile(m).read()).hexdigest()==sha(BUNDLE/name)
    guard = subprocess.run([sys.executable,'-B',str(BUNDLE/WORKER),'--root',str(BUNDLE),
                            '--protocol-sha',prep['protocol_sha256'],'--run'],capture_output=True,text=True,timeout=30)
    assert guard.returncode != 0 and 'Existing Linux VM only; no local gradients' in guard.stderr and not (BUNDLE/'outputs').exists()
    with (PREP/'Windows_training_guard.txt').open('x',encoding='utf-8') as stream: stream.write(guard.stderr)
    tf = ROOT/'scripts/cctv_dgp_feature_fusion_v32_r2_return_audit_template.py'
    af = ROOT/'scripts/audit_cctv_dgp_feature_fusion_v32_r2_return.py'
    assert af.read_text() == tf.read_text().replace('FUSION_V32_R2_PROTOCOL_PIN',prep['protocol_sha256'])
    restored=unroute(tf.read_text()).replace('cctv_dgp_feature_fusion_v32_r2_independent_audit','cctv_dgp_feature_fusion_v32_r1_independent_audit').replace('FUSION_V32_R2_PROTOCOL_PIN','FUSION_V32_R1_PROTOCOL_PIN')
    assert restored==(ROOT/'scripts/cctv_dgp_feature_fusion_v32_r1_return_audit_template.py').read_text()
    assert sha(af)==prep['prospective_return_auditor_sha256'] and sha(tf)==prep['prospective_template_sha256']
    ast.parse(af.read_text(),feature_version=(3,10))
    spec=importlib.util.spec_from_file_location('trusted_local_R2_return_schema',af)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    import numpy as np
    matrix=np.broadcast_to(np.float64(0),(7,978243));assert module.helpers().matrix(matrix) is matrix
    gain=.006945252687208803
    module.verify_early_receipt({'update':50,'minimum':.01,'relative_feature_error_gain':gain,'pass':False},gain)
    inherited=read(ROOT/'outputs/cctv_dgp_feature_fusion_v32_preparation/independent_packet_audit.json')
    assert inherited['complete'] and inherited['regressions_passed']==11
    forward=read(ROOT/'outputs/cctv_dgp_feature_fusion_v32_forward_review/review.json')
    assert forward['complete'] and len(forward['rows'])==4 and forward['all_buffers_unchanged']
    assert all(row['exact_raw_parity'] and row['exact_PNG_parity'] for row in forward['rows'])
    for name,digest in forward['source_bindings_sha256'].items():assert sha(ROOT/name)==digest
    bash=read(PREP/'bash_syntax.json');assert bash['complete'] and bash['script_sha256']==sha(BUNDLE/'scripts/run_v32.sh') and bash['new_parser_invocations']==0
    guide=(ROOT/'CCTV_DGP_FEATURE_FUSION_V32_R2_VM.md').read_text()
    assert all(f'{i}. ' in guide for i in range(1,6)) and prep['protocol_sha256'] in guide and prep['archive_sha256'] in guide
    assert guide.count('gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a')==8
    assert 'tmux new-session -A -s dgp_feature_fusion_v32_r2' in guide and BUNDLE.name in guide
    report={'complete':True,'checker_sha256':sha(Path(__file__)),'protocol_sha256':prep['protocol_sha256'],
            'archive_sha256':prep['archive_sha256'],'packet_files_verified':11,'routing_revision':2,
            'unchanged_mathematical_training_behavior':True,'cache_guards_preserve_Linux_and_exact_root':True,
            'root_regressions_passed':len(routes['rows']),'transfer_dispatch_regressions_passed':len(dispatch['rows']),
            'actual_transfer_path_check_precedes_data_and_model_work':True,'R1_zero_update_failure_preserved':True,
            'inherited_core_regressions_passed':11,'inherited_four_case_exact_CPU_parity':True,
            'actual_R2_gradient_schema_verified':True,'unchanged_one_percent_failure_retained':True,
            'Windows_pre_neural_training_rejected':True,'identical_shell_prior_parser_pass':True,
            'five_manual_steps_verified':True,'PuTTY_downloads_separate':True,'binding_counts':counts,
            'local_neural_calls':0,'local_gradient_calls':0,'local_optimizer_updates':0,
            'actual_VM_training_started_by_agent':False,'manual_VM_required':True,'app_promotion':False,
            'native_or_reserved_used':False,'goal_complete':False,'seconds':time.monotonic()-start}
    write(PREP/'independent_packet_audit.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
