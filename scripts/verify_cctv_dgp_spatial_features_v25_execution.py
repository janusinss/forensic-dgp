"""Independent finite-packet/source/transfer checks; no model imports or fitting."""
import ast
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'outputs/cctv_dgp_degraded_detail_vm_v24'
NEW = ROOT / 'outputs/cctv_dgp_spatial_features_vm_v25'
OUT = ROOT / 'outputs/cctv_dgp_spatial_features_v25_preparation'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda: stream.read(1024**2), b''): h.update(b)
    return h.hexdigest()


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def dump(node): return ast.dump(node, include_attributes=False)
def fn(tree, name): return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def normalize(tree):
    class Names(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                for new, old in [('dgp-spatial-feature-capacity-v25','dgp-degraded-detail-cohort-capacity-v24'),
                    ('scripts/cctv_dgp_spatial_features_v25_vm.py','scripts/cctv_dgp_degraded_detail_v24.py'),
                    ('cctv_dgp_spatial_features_v25_return','cctv_dgp_degraded_detail_v24_return'),
                    ('cctv-dgp-spatial-features-v25','cctv-dgp-degraded-detail-v24'),
                    ('scripts/run_v25.sh','scripts/run_v24.sh'), ('V25 update','V24 update')]:
                    node.value = node.value.replace(new, old)
            return node
    return Names().visit(copy.deepcopy(tree))


def source_contracts():
    original = ast.parse((OLD/'scripts/cctv_dgp_degraded_detail_v24.py').read_text())
    changed = normalize(ast.parse((NEW/'scripts/cctv_dgp_spatial_features_v25_vm.py').read_text()))
    for name in ['sha','write','verify','require_vm','main']:
        assert dump(fn(original,name)) == dump(fn(changed,name)), name
    wrapper = ast.parse('''def prepare(root, p):
    require_vm(root, idle=True)
    from cctv_dgp_spatial_features_v25_prepare import prepare_features
    return prepare_features(root, p, require_vm)
''').body[0]
    assert dump(fn(changed,'prepare')) == dump(wrapper)
    run = fn(changed,'run'); oldrun = fn(original,'run')
    # Only these prospective feature-cache statements may be inserted here.
    cache_start = next(i for i,n in enumerate(run.body) if isinstance(n,ast.Assign) and
        any(isinstance(t,ast.Name) and t.id=='cache' for t in n.targets))
    cache_end = next(i for i,n in enumerate(run.body) if isinstance(n,ast.Assign) and
        any(isinstance(t,ast.Name) and t.id=='progress' for t in n.targets))
    cache_nodes = run.body[cache_start:cache_end]
    assert len(cache_nodes)==5
    cache_text = dump(ast.Module(body=cache_nodes, type_ignores=[]))
    assert all(token in cache_text for token in ["'frozen_DGP_features'", "'frozen_DGP_features.json'", "'fpn'", "'DGP_state_unchanged'", "'files_sha256'"])
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ['head','identity'] for node in cache_nodes for n in ast.walk(node))
    del run.body[cache_start:cache_end]
    batch = fn(run,'batch')
    expected_batch = ast.parse('''def batch(ids):
    values = {key: torch.cat([items[i][key] for i in ids]) for key in ['x','base','mask','target','feature','interior','valid7','grid','truth','base_cosine','degraded_weight','clear_weight']}
    values['fpn'] = tuple(torch.cat([items[i]['fpn'][index] for i in ids]) for index in range(5))
    return values
''').body[0]
    assert dump(batch) == dump(expected_batch)
    run.body[run.body.index(batch)] = copy.deepcopy(fn(oldrun,'batch'))
    proof = ast.parse('''if update == 2:
    projection_gradients = {str(index): float(layer.weight.grad.double().square().sum()) for index,layer in enumerate(head.projections)}
    assert all(value > 0 for value in projection_gradients.values()), 'Nonzero gradients through all five own-DGP feature projections required'
    write(out/'feature_path_gradient_update2.json', {'complete': True, 'update_before_optimizer': 2,
        'per_projection_gradient_sum_squares': projection_gradients,
        'frozen_feature_tensors_have_no_gradients': all(not value.requires_grad and value.grad is None for item in items for value in item['fpn']),
        'DGP_state_unchanged': preflight['DGP_state_unchanged']})
''').body[0]
    class Allowed(ast.NodeTransformer):
        def __init__(self): self.head_calls=0; self.gradient_proofs=0
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name) and node.func.id=='head' and len(node.args)==4:
                assert isinstance(node.args[-1],ast.Subscript) and isinstance(node.args[-1].slice,ast.Constant) and node.args[-1].slice.value=='fpn'
                node.args.pop(); self.head_calls+=1
            return self.generic_visit(node)
        def visit_If(self,node):
            if dump(node)==dump(proof): self.gradient_proofs+=1; return None
            return self.generic_visit(node)
    allowed=Allowed(); run=allowed.visit(run)
    assert allowed.head_calls==2 and allowed.gradient_proofs==1
    assert dump(run)==dump(oldrun), 'Non-feature training/snapshot/quality/timing code differs'
    export = fn(changed,'export'); additions=[]
    for node in export.body:
        if isinstance(node,ast.AugAssign) and isinstance(node.target,ast.Name) and node.target.id=='files':
            text=dump(node)
            if "'feature_preflight_*.json'" in text or "'cctv_dgp_spatial_features_v25.py'" in text: additions.append(node)
    assert len(additions)==2
    assert "'cctv_dgp_spatial_features_v25_prepare.py'" in dump(additions[1]) and "'cctv_dgp_degraded_objective_v24.py'" in dump(additions[1])
    for node in additions: export.body.remove(node)
    assert dump(export)==dump(fn(original,'export'))
    helper=ast.parse((NEW/'cctv_dgp_spatial_features_v25_prepare.py').read_text())
    prepare=fn(helper,'prepare_features')
    assert isinstance(prepare.body[0],ast.Expr) and prepare.body[0].value.func.id=='require_vm'
    for node in ast.walk(helper):
        if isinstance(node,ast.Attribute): assert node.attr not in ['backward','step','AdamW','Adam','SGD']
    head_tree=ast.parse((NEW/'cctv_dgp_spatial_features_v25.py').read_text())
    cls=next(n for n in head_tree.body if isinstance(n,ast.ClassDef) and n.name=='SpatialFeatureHead')
    assert [a.arg for a in fn(cls,'forward').args.args]==['self','x','base','mask','fpn']
    assert not any(isinstance(n,ast.Constant) and n.value in ['target','profile','source','identity'] for n in ast.walk(fn(cls,'forward')))
    old_head=next(n for n in ast.walk(fn(original,'prepare')) if isinstance(n,ast.ClassDef))
    for method in ['blur','high']:
        # Local variable renaming only; coefficients/filter operations remain fixed.
        left=dump(fn(old_head,method)).replace("id='c'", "id='channels'")
        assert left==dump(fn(cls,method)), method
    return {'unchanged_functions':5,'exact_run_AST_after_declared_changes':True,
        'exact_export_AST_after_declared_changes':True,'head_calls_extended':2,'update2_gradient_proofs':1,
        'fixed_blur_HF_methods_equivalent':True,'no_training_in_feature_prepare':True}


def main():
    start=time.monotonic(); receipt=read(OUT/'preparation.json'); p=read(NEW/'protocol.json'); old=read(OLD/'protocol.json')
    assert sha(NEW/'protocol.json')==receipt['protocol_sha256']
    assert (NEW/'protocol.sha256').read_text()==receipt['protocol_sha256']+'  protocol.json\n'
    for key in ['references','cases','budgets','prospective_gates','fresh_DGP_parity_cases']:
        assert p[key]==old[key], key
    for key,value in old['design'].items():
        if key not in ['architecture','trainable_parameters','correction','frozen_components']:
            assert p['design'][key]==value, key
    assert p['design']['trainable_parameters']==53781 and p['all_facial_features_required_together']==['eyes','nose','mouth','face outline','overall visible appearance']
    assert sha(NEW/'schedule.json')==sha(OLD/'schedule.json')
    schedule=read(NEW/'schedule.json')['batches']; assert len(schedule)==800
    for offset in range(0,800,10):
        assert sorted(i for batch in schedule[offset:offset+10] for i in batch)==list(range(50))
        assert all(len(batch)==5 for batch in schedule[offset:offset+10])
    for name,digest in old['assets_sha256'].items():
        copied='lineage/v24_original_'+Path(name).name if name in ['scripts/cctv_dgp_degraded_detail_v24.py','scripts/run_v24.sh'] else name
        assert sha(OLD/name)==sha(NEW/copied)==digest
    for name,digest in p['assets_sha256'].items():
        file=(NEW/name).resolve(); assert file.is_relative_to(NEW) and sha(file)==digest, name
    for name,source in [('cctv_dgp_spatial_features_v25.py','scripts/cctv_dgp_spatial_features_v25.py'),
        ('cctv_dgp_spatial_features_v25_prepare.py','scripts/cctv_dgp_spatial_features_v25_prepare.py'),
        ('scripts/cctv_dgp_spatial_features_v25_vm.py','scripts/cctv_dgp_spatial_features_v25_vm.py')]:
        assert sha(NEW/name)==sha(ROOT/source), name
    source=source_contracts()
    python_files=list(NEW.rglob('*.py'))
    for file in python_files: ast.parse(file.read_text(encoding='utf-8'), feature_version=(3,10))
    bash=Path('C:/Program Files/Git/bin/bash.exe')
    check=subprocess.run([str(bash),'-n',str(NEW/'scripts/run_v25.sh')],capture_output=True,text=True,timeout=30)
    assert check.returncode==0, check.stderr
    base=[sys.executable,'-B',str(NEW/'scripts/cctv_dgp_spatial_features_v25_vm.py'),'--root',str(NEW),'--protocol-sha',receipt['protocol_sha256']]
    good=subprocess.run(base+['--verify-transfer'],capture_output=True,text=True,timeout=120)
    assert good.returncode==0 and json.loads(good.stdout)=={'complete':True,'assets':235,'cases':50,'neural_or_training_calls':0}
    guard=subprocess.run(base+['--run'],capture_output=True,text=True,timeout=120)
    assert guard.returncode!=0 and 'Existing Linux VM only' in guard.stderr
    assert not (NEW/'outputs').exists() and not list(NEW.glob('*preflight*.json'))
    archive=ROOT/'outputs/cctv-dgp-spatial-features-v25-execution.tar.gz'
    assert sha(archive)==receipt['archive_sha256'] and archive.stat().st_size==receipt['archive_bytes']
    assert Path(str(archive)+'.sha256').read_text()==receipt['archive_sha256']+'  '+archive.name+'\n'
    expected={file.relative_to(NEW).as_posix():sha(file) for file in NEW.rglob('*') if file.is_file()}
    actual={}; total=0
    with tarfile.open(archive,'r:gz') as stream:
        for member in stream:
            part=PurePosixPath(member.name)
            assert member.isfile() and not member.issparse() and not part.is_absolute() and part.parts[0]==NEW.name
            name='/'.join(part.parts[1:]); assert str(part)==member.name and not re.search(r'[\\:\x00-\x1f]',name) and '..' not in part.parts
            assert name.casefold() not in {n.casefold() for n in actual}
            assert member.mode==(0o755 if name=='scripts/run_v25.sh' else 0o644)
            payload=stream.extractfile(member); h=hashlib.sha256(); size=0
            for block in iter(lambda:payload.read(1024**2),b''): h.update(block); size+=len(block)
            assert size==member.size; actual[name]=h.hexdigest(); total+=size
    assert actual==expected and set(actual)==set(p['assets_sha256'])|{'protocol.json','protocol.sha256'}
    assert len(actual)==237 and len(p['assets_sha256'])==235
    interface=read(ROOT/'outputs/cctv_dgp_spatial_features_v25_interface/results.json')
    contract=read(ROOT/'outputs/cctv_dgp_spatial_features_v25_contract_regressions/results.json')
    for record in [interface,contract]:
        assert record['complete'] and record['backward_calls']==record['optimizer_updates']==0
        for name,digest in record['source_bindings_sha256'].items(): assert sha(ROOT/name)==digest, name
    assert contract['count']==12 and len(set(contract['regressions_passed']))==12
    result={'complete':True,'date':'2026-10-06','checker_sha256':sha(Path(__file__)),
        'preparation_sha256':sha(OUT/'preparation.json'),'protocol_sha256':receipt['protocol_sha256'],
        'archive_sha256':receipt['archive_sha256'],'archive_bytes':receipt['archive_bytes'],
        'archive_regular_members':len(actual),'archive_uncompressed_bytes':total,
        'frozen_assets_verified':len(p['assets_sha256']),'original_V24_assets_unchanged':len(old['assets_sha256']),
        'source_contracts':source,'Python310_sources_parsed':len(python_files),'bash_syntax_pass':True,
        'actual_Windows_verify_transfer_pass':True,'actual_Windows_run_guard_pass':True,
        'schedule_exact_80epochs_800updates':True,'quality_objective_budgets_unchanged':True,
        'regressions_bound':12,'VM_training_gradient_timing_quality_pending':True,
        'source_bindings_sha256':{file.relative_to(ROOT).as_posix():sha(file) for file in
            [Path(__file__),ROOT/'scripts/prepare_cctv_dgp_spatial_features_v25.py',ROOT/'outputs/cctv_dgp_spatial_features_v25_contract_regressions/results.json']},
        'neural_calls':0,'backward_calls':0,'optimizer_updates':0,'VM_actions':False,'goal_complete':False,
        'seconds':time.monotonic()-start}
    with (OUT/'independent_execution_audit.json').open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
