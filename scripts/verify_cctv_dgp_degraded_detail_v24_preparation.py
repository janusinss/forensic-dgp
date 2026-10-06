"""Independent V24 source/transfer/loss contracts. No backward or optimization."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_cctv_dgp_detail_skip_v23 as a
from import_cctv_dgp_detail_skip_v23 import relative_name

BUNDLE = ROOT / 'outputs/cctv_dgp_degraded_detail_vm_v24'
EVIDENCE = ROOT / 'outputs/cctv_dgp_degraded_detail_v24_preparation'
PIN = '76ab24695a40b411832e2d678c79b0e2a80f6320d4525970b2dd32db2379e114'
ARCHIVE_PIN = 'b448b1b5d918be71b5ebba3e46f8865caffe8e0c8992976c1fde7ebd52ac7906'


def node(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def equivalent_runtime(original, new):
    """Whitelisted AST changes only; all other execution/gate nodes identical."""
    revised = new
    for before, after in [('dgp-degraded-detail-cohort-capacity-v24', 'dgp-learned-detail-skip-capacity-v23'),
                          ('cctv_dgp_degraded_detail_v24', 'cctv_dgp_detail_skip_v23'),
                          ('cctv-dgp-degraded-detail-v24', 'cctv-dgp-detail-skip-v23'),
                          ('run_v24.sh', 'run_v23.sh'), ('V24 update', 'V23 update')]:
        revised = revised.replace(before, after)
    old, new_tree = ast.parse(original), ast.parse(revised)
    old_run, new_run = node(old, 'run'), node(new_tree, 'run')
    old_loss = node(old_run, 'loss'); new_loss = node(new_run, 'loss')
    a.require(ast.dump(old_loss) != ast.dump(new_loss), 'Old failed loss retained')
    expected_loss = ast.parse("""def loss(ids):
    b=batch(ids); pred=head(b['x'],b['base'],b['mask'])
    terms=objective_terms(b,pred,identity,mean,feature_errors,ssim,normalizers)
    result=sum(terms.values()).mean()
    assert torch.isfinite(result)
    return result
""").body[0]
    a.require(ast.dump(new_loss) == ast.dump(expected_loss), 'New loss path differs')
    old_run.body.remove(old_loss); new_run.body.remove(new_loss)
    additions = ast.parse("""from cctv_dgp_degraded_objective_v24 import cohort_normalizers, objective_terms
normalizers, loss_setup = cohort_normalizers(items, head)
write(out/'cohort_loss_setup.json', loss_setup)
""").body
    for addition in additions:
        matches = [n for n in new_run.body if ast.dump(n) == ast.dump(addition)]
        a.require(len(matches) == 1, 'Missing/duplicate objective setup')
        new_run.body.remove(matches[0])
    batch = node(new_run, 'batch')
    lists = [n for n in ast.walk(batch) if isinstance(n, ast.List)]
    a.require(len(lists) == 1 and [n.value for n in lists[0].elts][-2:] == ['degraded_weight', 'clear_weight'], 'Training flags differ')
    lists[0].elts = lists[0].elts[:-2]
    a.require(ast.dump(old) == ast.dump(new_tree), 'Change beyond objective/setup/batch flags')
    return True


def source_head(source):
    import torch
    from torch.nn import functional as F
    tree = ast.parse(source)
    prepare = node(tree, 'prepare')
    cls = next(n for n in prepare.body if isinstance(n, ast.ClassDef) and n.name == 'DetailHead')
    # Only the unchanged CPU class is compiled; no VM entry point is called.
    scope = {'torch': torch, 'nn': torch.nn, 'F': F}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<pinned-V24-filter-class-only>', 'exec'), scope)
    torch.manual_seed(20261005)
    return scope['DetailHead']().cpu().eval().requires_grad_(False)


def verify():
    import numpy as np
    import torch
    started = time.monotonic(); torch.set_num_threads(4)
    a.require(not (EVIDENCE/'independent_preparation_audit.json').exists(), 'Preserve previous verifier results')
    a.require(a.sha(BUNDLE/'protocol.json') == PIN, 'V24 protocol differs')
    p = a.read(BUNDLE/'protocol.json'); old = a.verify_bundle(a.BUNDLE)
    source = BUNDLE/'scripts/cctv_dgp_degraded_detail_v24.py'
    helper = BUNDLE/'cctv_dgp_degraded_objective_v24.py'
    plan = {'date':'2026-10-06', 'scope':'Independent transfer/source equivalence, fixed CPU cohort filters and loss contracts only',
            'checker_sha256':a.sha(Path(__file__)), 'protocol_sha256':PIN, 'archive_sha256':ARCHIVE_PIN,
            'maximum_fixed_high_pass_calls':100, 'head_DGP_recognizer_forwards':0,
            'cohort_scalar_absolute_tolerance':1e-9, 'pure_loss_absolute_tolerance':1e-6,
            'optimizer_updates':0, 'backward_calls':0, 'VM_actions':False, 'quality_verified':False}
    a.write(EVIDENCE/'verification_plan_r1.json', plan)
    a.require(p['format'] == 'dgp-degraded-detail-cohort-capacity-v24', 'Protocol format differs')
    for key in ['cases','references','budgets','prospective_gates','normalization_scope','fresh_DGP_parity_cases','execution','execution_evidence']:
        a.require(p[key] == old[key], 'Frozen data/gate/budget policy changed: '+key)
    expected_design = copy.deepcopy(old['design']); actual_design = copy.deepcopy(p['design'])
    actual_design.pop('clear_policy'); expected_design.pop('clear_policy')
    expected_design.pop('objective'); actual_design.pop('objective')
    a.require(actual_design == expected_design, 'Head/hyperparameter design changed')
    a.require(p['native_or_reserved_used'] is False and p['completion_training'] is False and p['goal_complete'] is False,
              'Scope alteration')
    a.require(a.sha(BUNDLE/'schedule.json') == a.sha(a.BUNDLE/'schedule.json'), '800-update schedule differs')
    schedules = a.read(BUNDLE/'schedule.json')['batches']
    for epoch in range(80):
        a.require(sorted(i for batch in schedules[epoch*10:(epoch+1)*10] for i in batch) == list(range(50)), 'Epoch exposure differs')
    python_files = 0
    for name, digest in p['assets_sha256'].items():
        file = a.safe(BUNDLE, name); a.require(a.sha(file) == digest, 'Asset differs: '+name)
        if name.endswith('.py'):
            ast.parse(file.read_text(encoding='utf-8'), feature_version=(3,10)); python_files += 1
    a.require(len(p['assets_sha256']) == 221, 'Asset count differs')
    a.require(a.sha(source) == a.sha(ROOT/'scripts/cctv_dgp_degraded_detail_v24.py') and
              a.sha(helper) == a.sha(ROOT/'scripts/cctv_dgp_degraded_objective_v24.py'), 'Workspace source differs')
    equivalent_runtime((a.BUNDLE/'scripts/cctv_dgp_detail_skip_v23.py').read_text(), source.read_text())
    old_shell = (a.BUNDLE/'scripts/run_v23.sh').read_text(encoding='utf-8')
    new_shell = (BUNDLE/'scripts/run_v24.sh').read_bytes()
    a.require(new_shell == old_shell.replace('cctv_dgp_detail_skip_v23','cctv_dgp_degraded_detail_v24').encode(), 'Supervisor changed')
    a.require(b'\r' not in new_shell, 'Shell is not LF')
    archive = ROOT/'outputs/cctv-dgp-degraded-detail-v24-execution.tar.gz'
    a.require(a.sha(archive) == ARCHIVE_PIN and archive.stat().st_size == 218041986, 'Transfer bytes/hash differ')
    expected = {f.relative_to(BUNDLE).as_posix():a.sha(f) for f in BUNDLE.rglob('*') if f.is_file()}
    a.require(len(expected) == 223, 'Local packet member count differs')
    with tarfile.open(archive,'r:gz') as stream:
        members = stream.getmembers()
        a.require(len(members) == len(expected) and len({m.name.casefold() for m in members}) == len(members), 'Duplicate or missing member')
        for member in members:
            path = relative_name(member.name)
            a.require(member.isfile() and not member.issparse() and path.parts[0] == BUNDLE.name, 'Unsafe transfer member')
            name = '/'.join(path.parts[1:]); payload = stream.extractfile(member).read()
            a.require(name in expected and hashlib.sha256(payload).hexdigest() == expected[name], 'Archive payload differs')
            a.require(member.mode == (0o755 if name == 'scripts/run_v24.sh' else 0o644), 'Archive mode differs')
    a.require(Path(str(archive)+'.sha256').read_bytes() == (ARCHIVE_PIN+'  '+archive.name+'\n').encode('ascii'), 'Archive checksum differs')
    a.require((BUNDLE/'protocol.sha256').read_bytes() == (PIN+'  protocol.json\n').encode('ascii'), 'Protocol checksum differs')
    modes = {}
    for mode in ['--verify-transfer','--run']:
        result = subprocess.run([sys.executable,'-B',str(source),'--root',str(BUNDLE),'--protocol-sha',PIN,mode], capture_output=True,text=True,timeout=30)
        (EVIDENCE/(mode[2:]+'_local.log')).write_text(result.stdout+result.stderr,encoding='utf-8')
        a.require(result.returncode == 0 if mode=='--verify-transfer' else
                  result.returncode != 0 and 'Existing Linux VM only' in result.stderr, 'Platform guard failed')
        modes[mode] = {'exit_code':result.returncode,'expected':True}
    a.require(not (BUNDLE/'outputs').exists() and not list(BUNDLE.glob('preflight_*.json')), 'Local execution produced training evidence')
    sys.path.insert(0, str(BUNDLE))
    import cctv_dgp_degraded_objective_v24 as objective
    a.require(Path(objective.__file__).resolve() == helper.resolve(), 'Wrong helper module')
    head = source_head(source.read_text()); before = a.state_hash(head.state_dict())
    calls = {'fixed_high_pass':0,'head_forward':0}
    head.register_forward_hook(lambda *_:calls.__setitem__('head_forward', calls['head_forward']+1))
    original_high = head.high
    def counted_high(value):
        calls['fixed_high_pass'] += 1; return original_high(value)
    head.high = counted_high
    items = []
    for c in p['cases']:
        base = a.raw_rgb(BUNDLE/c['raw_dgp']); target = a.rgb(BUNDLE/c['target']); mask = a.observed(BUNDLE/c['observed'])
        items.append({'case':c, 'base':torch.from_numpy(base.copy()).permute(2,0,1)[None],
            'target':torch.from_numpy(target.astype(np.float32)/np.float32(255)).permute(2,0,1)[None],
            'feature':torch.from_numpy(a.feature_mask(c,mask).astype(np.float32))[None,None],
            'interior':torch.from_numpy(a.interior(mask,6).astype(np.float32))[None,None]})
    with torch.inference_mode():
        normalizers, receipt = objective.cohort_normalizers(items, head)
    loss_rows = a.read(ROOT/'outputs/cctv_dgp_detail_skip_v23_loss_audit_v1/results.json')['rows']
    by_id = {r['id']:r for r in loss_rows}
    maximum_row_error = 0.
    for row, item in zip(receipt['rows'],items):
        old_row = by_id[row['id']]['states']['0']
        for field, old_field in [('feature_MSE','raw_feature_MSE'),('interior_MSE','raw_interior_MSE')]:
            err = abs(row[field] - old_row[old_field]); maximum_row_error = max(maximum_row_error,err)
            a.require(err <= 1e-9, 'CPU fixed baseline differs from separately audited loss')
        clear = item['case']['profile'] == 'clear'
        a.require(float(item['degraded_weight']) == (0. if clear else 1.25) and float(item['clear_weight']) == (1. if clear else 0.), 'Wrong loss flags')
    for index, field in enumerate(['raw_feature_MSE','raw_interior_MSE']):
        reference = max(sum(r['states']['0'][field] for r in loss_rows if r['profile']!='clear')/40,1e-6)
        a.require(abs(float(normalizers[index])-reference) <= 1e-9 and not normalizers[index].requires_grad, 'Frozen cohort normalizer differs')
    a.require(calls == {'fixed_high_pass':100,'head_forward':0} and a.state_hash(head.state_dict()) == before and
              not any(v.grad is not None or v.requires_grad for v in head.parameters()), 'Fixed-filter state/calls changed')
    a.write(EVIDENCE/'CPU_cohort_contract.json',receipt)
    # Actual cohort baseline term assembly; no recognizer or new head prediction.
    rows = [by_id[c['id']]['states']['0'] for c in p['cases']]
    t = lambda field:torch.tensor([r[field] for r in rows],dtype=torch.float32)
    weights = torch.cat([i['degraded_weight'] for i in items]); clear = torch.cat([i['clear_weight'] for i in items])
    with torch.inference_mode():
        terms = objective.assemble_terms(t('raw_feature_MSE'),t('raw_interior_MSE'),t('raw_pixel_MSE'),t('raw_pixel_MSE'),
            t('raw_SSIM'),t('raw_SSIM'),t('raw_ArcFace'),t('raw_ArcFace'),torch.zeros(50),weights,clear,normalizers)
        initial_loss = float(sum(terms.values()).mean())
    a.require(abs(initial_loss-1.3) <= 1e-6 and bool((sum(terms.values())[clear.bool()]==0).all()), 'Cohort baseline/control assembly differs')
    result = {'complete':True, 'date':'2026-10-06', 'scope':'Transfer and objective contracts only; VM learning/quality pending',
        'plan_sha256':a.sha(EVIDENCE/'verification_plan_r1.json'), 'checker_sha256':a.sha(Path(__file__)),
        'protocol_sha256':PIN, 'archive_sha256':ARCHIVE_PIN, 'archive_bytes':archive.stat().st_size,
        'assets':221, 'regular_archive_members':223, 'Python310_files':python_files,
        'entire_runtime_AST_equal_except_allowed_loss_setup_flags_names':True, 'head_and_quality_gates_unchanged':True,
        'same50_cases_and800_updates80epochs':True, 'platform_checks':modes, 'fixed_filter_calls':calls,
        'cohort_contract_sha256':a.sha(EVIDENCE/'CPU_cohort_contract.json'), 'maximum_baseline_row_difference':maximum_row_error,
        'all50_initial_cohort_loss':initial_loss, 'all10_clear_initial_loss':0, 'parameters':4613,
        'CPU_cohort_is_not_CUDA_gradient_or_training_evidence':True, 'seconds':time.monotonic()-started,
        'recognizer_DGP_forwards':0, 'optimizer_updates':0, 'backward_calls':0, 'VM_actions':False,
        'app_promotion':False, 'goal_complete':False}
    a.write(EVIDENCE/'independent_preparation_audit.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':verify()
