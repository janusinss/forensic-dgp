"""Independent archive/geometry/CLI/guard audit. No VM or learned-model calls."""
import ast
import copy
from datetime import datetime,timezone
import json
from pathlib import Path,PurePosixPath
import subprocess
import sys
import tarfile

import numpy as np
from PIL import Image

from cctv_dgp_spatial_path_replay_v1_contract import NAME,STEM,DESIGN_SHA,read,write,sha,verify_transfer,validate

ROOT=Path(__file__).resolve().parent.parent
DESIGN=ROOT/'outputs/cctv_dgp_spatial_path_replay_v1_design'
SOURCE=ROOT/'outputs/cctv_dgp_spatial_readout_v1_return'
PARENT=ROOT/'outputs/cctv_dgp_bank_photometric_guard_v1_return'


def main():
    prepared=read(DESIGN/'preparation.json');packet=ROOT/'outputs'/NAME
    assert prepared['complete'] and not (packet/'outputs').exists()
    archive=ROOT/'outputs'/(STEM+'-execution.tar.gz')
    assert sha(archive)==prepared['archive_sha256'] and archive.stat().st_size==prepared['archive_bytes']
    sidecar=Path(str(archive)+'.sha256').read_bytes()
    assert sidecar==(prepared['archive_sha256']+'  '+archive.name+'\n').encode('utf-8') and b'\r' not in sidecar and not sidecar.startswith(b'\xef\xbb\xbf')
    pin=sha(packet/'protocol.json')
    assert pin==prepared['protocol_sha256'] and sha(packet/'design.json')==DESIGN_SHA
    p,old,data=verify_transfer(packet,SOURCE,PARENT,pin)
    expected={NAME+'/'+name for name in p['assets_sha256']} | {NAME+'/protocol.json'}
    with tarfile.open(archive,'r:gz') as pack:
        members=pack.getmembers()
        assert len(members)==11 and {m.name for m in members}==expected and len({m.name for m in members})==len(members)
        for member in members:
            posix=PurePosixPath(member.name)
            assert member.isreg() and not posix.is_absolute() and '..' not in posix.parts and str(posix)==member.name
            assert '\\' not in member.name and ':' not in member.name
            assert pack.extractfile(member).read()==(packet/member.name.removeprefix(NAME+'/')).read_bytes()
    refs,cases={r['id']:r for r in data['references']},{c['id']:c for c in data['cases']}
    source_targets={r['reference']:r for r in read(SOURCE/'evidence/target_bindings.json')['targets']}
    target_rows=[]
    for block in p['reference_blocks']:
        ref=refs[block['reference']]
        with Image.open(PARENT/ref['target']) as image:
            assert image.size==(256,256);pixels=np.asarray(image.convert('RGB')).copy()
        assert __import__('hashlib').sha256(pixels.tobytes()).hexdigest()==source_targets[block['reference']]['canonical_target_RGB_sha256']
        assert np.asarray(ref['matrix112']).shape==(2,3)
        for cid in block['case_ids']:
            c=cases[cid];assert c['role']=='train' and c['reference_id']==ref['id']
            with Image.open(PARENT/c['input']) as image:assert image.size==(256,256)
            assert np.asarray(c['landmarks5_canvas_xy']).shape==(5,2)
        target_rows.append(dict(reference=block['reference'],source=block['source'],canonical_target_RGB_exact=True))
    assert len(target_rows)==20
    tree=ast.parse((packet/'scripts/cctv_dgp_spatial_path_replay_v1_vm.py').read_text(encoding='utf-8'),feature_version=(3,10))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in ['backward','step','enable_learning','train','replay','supervise'] for n in calls)
    assert not any(isinstance(n,ast.Attribute) and n.attr=='optim' for n in ast.walk(tree))
    review=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='review')
    assert isinstance(review.body[0],ast.Expr) and isinstance(review.body[0].value,ast.Call) and review.body[0].value.func.id=='machine_guard'
    context=next(n for n in ast.walk(review) if isinstance(n,ast.With) and any(isinstance(x.context_expr,ast.Call) and isinstance(x.context_expr.func,ast.Name) and x.context_expr.func.id=='source_cwd' for x in n.items))
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='loader' and n.func.attr=='load_dependencies' for n in ast.walk(context))
    assert all(n.func.attr=='load_dependencies' for n in calls if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='loader')
    for path in packet.rglob('*.py'):ast.parse(path.read_text(encoding='utf-8'),feature_version=(3,10))
    d=read(packet/'design.json');mutations=0
    for key in ['maximum_new_optimizer_updates','full_corpus_epochs','model_qualified','automatic_launch','new_incumbent_selection','local_neural_construction_allowed']:
        altered=copy.deepcopy(p);altered[key]=1 if type(altered[key]) is int else True
        try:validate(altered,d)
        except AssertionError:mutations+=1
        else:raise AssertionError('Mutation accepted: '+key)
    command=[str(ROOT/'venv/Scripts/python.exe'),'-B',str(packet/'scripts/cctv_dgp_spatial_path_replay_v1_vm.py'),
        '--root',str(packet),'--source-root',str(SOURCE),'--parent-root',str(PARENT),'--protocol-sha',pin]
    checked=subprocess.run(command+['--verify-transfer'],capture_output=True,text=True,timeout=90)
    with (DESIGN/'actual_transfer_CLI.txt').open('x',encoding='utf-8',newline='\n') as stream:stream.write(checked.stdout+checked.stderr)
    assert checked.returncode==0
    cli=json.loads(checked.stdout)
    assert cli['source_assets']==410 and cli['parent_assets']==203 and cli['saved_states']==14 and cli['neural_calls']==0
    help_result=subprocess.run(command+['--help'],capture_output=True,text=True,timeout=20)
    assert help_result.returncode==0 and all(flag in help_result.stdout for flag in ['--review','--audit-saved','--export','--supervise'])
    assert '--train' not in help_result.stdout
    tests=read(DESIGN/'cpu_tests_final.json');bash=read(DESIGN/'bash_syntax.json')
    assert tests['complete'] and tests['tests_run']==15 and tests['actual_saved_array_calibrations']==30
    assert bash['exit_code']==0 and bash['shell_sha256']==sha(packet/'scripts/run_spatial_path.sh')
    shell=(packet/'scripts/run_spatial_path.sh').read_bytes()
    assert b'\r' not in shell and b'--supervise' in shell and b'${TMUX:-}' in shell
    protected=read(DESIGN/'protected_before_preparation.json')['files']
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    proof=read(packet/'evidence/expected_state_hashes.json')
    assert proof['complete'] and len(proof['states'])==14 and proof['neural_calls']==proof['weight_assignments']==0
    assert proof['states']==prepared['expected_complete_state_hashes']
    record=dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),status='VERIFIED_PREPARED_UNRUN',
        protocol_sha256=pin,archive_sha256=prepared['archive_sha256'],archive_members=11,packet_assets=10,
        source_assets=410,parent_assets=203,canonical_targets=20,all100_input_geometry_checked=True,
        states=14,CPU_tests=15,actual_saved_array_scorer_calibrations=30,invalid_protocol_mutations_rejected=mutations,
        actual_hash_only_CLI_checked=True,Windows_pre_neural_guard_checked=True,Python310_AST_pass=True,
        Bash_syntax_pass=True,existing_source_CWD_guard_preserved=True,historical_training_not_called=True,
        original_checkpoint_and_documents_unchanged=True,neural_calls=0,gradient_queries=0,optimizer_updates=0,
        agent_VM_calls=0,model_qualified=False,new_CUDA_parity_or_timing_verified=False)
    write(DESIGN/'independent_preparation_audit.json',record);print(json.dumps(record))


if __name__=='__main__':main()
