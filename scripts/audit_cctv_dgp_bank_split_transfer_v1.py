"""Independent resource arithmetic, exact science/AST and real archive/CLI audit."""
import ast
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import time
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs'


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def dump(node): return ast.dump(node, include_attributes=False)


def main():
    start = time.monotonic()
    receipt = OUT/'cctv_dgp_bank_split_transfer_v1_audit.json'
    assert not receipt.exists()
    parent = OUT/'cctv_dgp_bank_comparison_v1_vm'
    old = read(parent/'protocol.json')
    prep = read(OUT/'cctv_dgp_bank_comparison_v1_r2_preparation.json'); assert prep['complete']
    # Default raw deflate bound from upstream zlib deflate.c, with ZIP/NPY allowances.
    # No external code, model, gradient or optimizer is run here.
    sizes = [5*256*256*3*4+128, 5*3*512*4+128,
        5*max(len(c['id']) for c in old['cases'])*4+128, 5*8+128, 1+128]
    member_bound = lambda n: n+(n>>12)+(n>>14)+(n>>25)+13
    pack_bound = sum(member_bound(n) for n in sizes) + 8192
    png_bound = member_bound(256*(256*3+1))+4096
    # The writer's fixed metrics schema is unchanged (checked by AST below).
    # Expand every numeric/bool/null leaf to a quoted 64-character placeholder:
    # this exceeds every finite float64 repr, bool, null and bounded case count.
    # Input strings and keys are frozen by the protocol; digest strings are 64.
    baseline_path = OUT/'cctv_dgp_bank_comparison_v1_return/outputs/bank_comparison_v1/baseline/metrics.json'
    template = read(baseline_path)
    def expanded(value):
        if isinstance(value, dict): return {k: expanded(v) for k,v in value.items()}
        if isinstance(value, list): return [expanded(v) for v in value]
        return value if isinstance(value,str) else '9'*64
    widest_row = expanded(template['rows'][0])
    for key in ['id','source','profile']:
        widest_row[key] = max((c[key] for c in old['cases']), key=lambda v:len(json.dumps(v)))
    native = expanded(template['native_unpaired_rows'][0])
    native['id'] = max((c['id'] for c in old['native_development']),key=len)
    def metadata_bound(count):
        record=expanded(template)
        record['label']='B50_bank_disabled'
        record['rows']=[widest_row]*count
        record['native_unpaired_rows']=[native]*24
        return len((json.dumps(record,indent=2,allow_nan=False)+'\n').encode('utf-8'))
    metadata_bounds={'full':metadata_bound(3905),'ablation':metadata_bound(100)}
    assert metadata_bounds['full'] < 16*1024**2
    assert metadata_bounds['ablation'] < 8*1024**2
    # Full snapshot has 3905 rows; the ablation has only 100 of the same schema.
    full_bound = 781*pack_bound + 24*(256*256*3*4+128) + 124*png_bound + 16*1024**2
    ablation_bound = 20*pack_bound + 24*(256*256*3*4+128) + 124*png_bound + 8*1024**2
    assert full_bound < 3*1024**3+64*1024**2 and ablation_bound < 128*1024**2
    results=[]
    allowed_changes = {'format','UTC','budgets','assets_sha256','selected_routes','parent_protocol_sha256',
        'resource_layout','baseline_storage_receipt','separate_return_audit_required_before_next_route',
        'comparison_complete_in_this_packet'}
    original_worker = ast.parse((parent/'scripts/cctv_dgp_bank_comparison_v1_vm.py').read_text())
    old_funcs = {n.name: dump(n) for n in ast.walk(original_worker) if isinstance(n,ast.FunctionDef)}
    for r in prep['routes']:
        route=r['route']; root=OUT/r['name']; p=read(root/'protocol.json')
        assert sha(root/'protocol.json')==r['protocol_sha256']
        changed = {k for k in set(old)|set(p) if old.get(k)!=p.get(k)}
        assert changed <= allowed_changes, changed-allowed_changes
        spec=importlib.util.spec_from_file_location('split_contract_'+route,root/'cctv_dgp_bank_comparison_v1_contract.py')
        m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); m.verify(root,r['protocol_sha256'])
        assert p['selected_routes']==[route] and p['budgets']['return_uncompressed_bytes']==6*1024**3
        assert p['budgets']['maximum_optimizer_updates']==50 and p['budgets']['maximum_backwards']==250
        assert all(p['budgets'][k] <= v for k,v in old['budgets'].items())
        permitted_modified={'cctv_dgp_bank_comparison_v1_contract.py','cctv_dgp_bank_comparison_v1_model.py',
            'scripts/cctv_dgp_bank_comparison_v1_vm.py'}
        assert {n for n,h in old['assets_sha256'].items() if sha(root/n)!=h}==permitted_modified
        assert all(sha(parent/n)==h for n,h in old['assets_sha256'].items())
        old_model=ast.parse((parent/'cctv_dgp_bank_comparison_v1_model.py').read_text())
        new_model=ast.parse((root/'cctv_dgp_bank_comparison_v1_model.py').read_text())
        for tree in [old_model,new_model]:
            for n in tree.body:
                if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NAME' for t in n.targets): n.value=ast.Constant('ROOT_ONLY')
        assert dump(old_model)==dump(new_model), 'Model computation must be identical'
        tree=ast.parse((root/'scripts/cctv_dgp_bank_comparison_v1_vm.py').read_text())
        new_funcs={n.name:dump(n) for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
        unchanged=['clock','objective','batch','create','frozen','save_state','metrics','base_raw','export','vm','pixels','tensor']
        assert all(old_funcs[n]==new_funcs[n] for n in unchanged), 'Learning/output computation changed'
        loops=[dump(n.iter) for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='route']
        assert len(loops)==2 and all("selected_routes" in s for s in loops)
        layout=p['resource_layout']
        states=2*((root/('initializers/initial_'+route+'.pth')).stat().st_size+3*p['routes'][route]['elements']*4+1024**2)+48*1024**2
        assert states < layout['state_and_fixture_reserve_bytes']
        projected=2123116593+layout['full_candidate_snapshot_bound_bytes']+layout['state_and_fixture_reserve_bytes']+(layout['B_ablation_bound_bytes'] if route=='B' else 0)
        assert projected==r['output_bound_bytes']<p['budgets']['return_uncompressed_bytes']
        rejected=[]
        for label in ['lower_gate','final_leak','duplicate_schedule','remove_anchor','wrong_route','extra_updates']:
            q=copy.deepcopy(p)
            if label=='lower_gate':q['scientific_thresholds']['early_structure_gain']=.001
            elif label=='final_leak':q['cases'][0]['role']='final'
            elif label=='duplicate_schedule':q['schedule'][1]=q['schedule'][0]
            elif label=='remove_anchor':q['active_clear_blur_anchor_coefficients']=[0.,1.]
            elif label=='wrong_route':q['selected_routes']=['B' if route=='A' else 'A']
            else:q['budgets']['maximum_optimizer_updates']=100
            try:m.validate(q)
            except AssertionError:rejected.append(label)
            else:raise AssertionError('Unsafe mutation accepted: '+label)
        cli=subprocess.run([sys.executable,'-B',str(root/'scripts/cctv_dgp_bank_comparison_v1_vm.py'),
            '--root',str(root),'--protocol-sha',r['protocol_sha256'],'--verify-transfer'],capture_output=True,text=True,check=True,timeout=120)
        status=json.loads(cli.stdout); assert status['complete'] and status['neural_or_gradient_calls']==status['optimizer_updates']==0
        archive=OUT/(r['stem']+'-execution.tar.gz'); assert archive.stat().st_size==r['archive_bytes'] and sha(archive)==r['archive_sha256']
        assert Path(str(archive)+'.sha256').read_bytes()==(r['archive_sha256']+'  '+archive.name+'\n').encode('ascii')
        expected={**p['assets_sha256'],'protocol.json':r['protocol_sha256']}; seen=set()
        with tarfile.open(archive,'r|gz') as tar:
            for member in tar:
                path=PurePosixPath(member.name); n=PurePosixPath(*path.parts[1:]).as_posix()
                assert member.isfile() and path.parts[0]==r['name'] and '..' not in path.parts and n not in seen
                assert n in expected
                with tar.extractfile(member) as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
                assert digest==expected[n]; seen.add(n)
        assert seen==set(expected)
        results.append(dict(route=route,protocol_sha256=r['protocol_sha256'],archive_sha256=r['archive_sha256'],
            archive_members_checked=len(seen),scientific_fields_unchanged=True,identical_model_AST=True,
            unchanged_computational_functions=unchanged,unsafe_mutations_rejected=rejected,transfer_CLI_verified=True,
            output_bound_GiB=projected/1024**3,strict_state_fixture_bound_bytes=states))
        print(dict(route=route,archive_members_checked=len(seen),unsafe_mutations_rejected=len(rejected)),flush=True)
    record=dict(complete=True,checker_sha256=sha(__file__),preparation_sha256=sha(OUT/'cctv_dgp_bank_comparison_v1_r2_preparation.json'),
        routes=results,calculated_worst_case_snapshot_bytes=full_bound,calculated_worst_case_ablation_bytes=ablation_bound,
        metrics_json_upper_bounds=metadata_bounds,metrics_template_sha256=sha(baseline_path),
        original_checker_stop_retained='cctv_dgp_bank_comparison_v1_analysis/split_transfer_check_attempt1',
        deflate_bound_source='https://raw.githubusercontent.com/madler/zlib/v1.3.1/deflate.c',
        archive_source_bytes_verified=True,local_neural_gradient_or_optimizer_calls=0,manual_VM_training_pending=True,
        quality_qualified=False,goal_complete=False,seconds=time.monotonic()-start)
    with receipt.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(record,indent=2)+'\n')
    print(dict(complete=True,routes_checked=2,seconds=record['seconds']),flush=True)


if __name__=='__main__': main()
