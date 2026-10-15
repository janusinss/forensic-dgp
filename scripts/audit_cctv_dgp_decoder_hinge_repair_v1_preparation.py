"""Independent archive, source-equivalence and pre-neural guard audit.

Never contacts the VM or constructs a learned model.  The guarded Windows
subprocess is intentionally rejected before transfer verification or imports.
"""
import ast
import copy
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile

from cctv_dgp_decoder_hinge_repair_v1_contract import (
    NAME, STEM, LEGACY_HELPER_SHA, CORRECTED_HELPER_SHA,
    read, write, sha, validate, verify_transfer,
)
from audit_cctv_dgp_decoder_hinge_repair_v1_return import verify_corrected_guard_binding

ROOT = Path(__file__).resolve().parent.parent
DESIGN = ROOT/'outputs/cctv_dgp_decoder_hinge_repair_v1_design'
SOURCE = ROOT/'outputs/cctv_dgp_spatial_readout_v1_return'
PARENT = ROOT/'outputs/cctv_dgp_bank_photometric_guard_v1_return'
REPLAY = ROOT/'outputs/cctv_dgp_spatial_path_replay_v1_return'


def function(source, name):
    functions = [node for node in ast.walk(ast.parse(source, feature_version=(3, 10)))
                 if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(functions) == 1, name
    return copy.deepcopy(functions[0])


class NormalizeObjective(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id == 'p': node.id = 'old'
        return node

    def visit_AugAssign(self, node):
        # The new worker inlines a helper which counted recognizer calls.
        if (isinstance(node.target, ast.Subscript) and isinstance(node.target.value, ast.Name)
                and node.target.value.id == 'counts'):
            assert ast.literal_eval(node.target.slice) == 'recognizer_batch_calls'
            assert isinstance(node.op, ast.Add) and ast.literal_eval(node.value) == 1
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == 'emb':
            assert len(node.args) == 2 and ast.dump(node.args[1]) == ast.dump(ast.Name(id='g', ctx=ast.Load()))
            node = ast.Call(func=ast.Attribute(value=ast.Name(id='identity', ctx=ast.Load()), attr='embedding', ctx=ast.Load()),
                args=[node.args[0], ast.Subscript(value=ast.Name(id='g', ctx=ast.Load()), slice=ast.Constant('mask'), ctx=ast.Load()),
                      ast.Subscript(value=ast.Name(id='g', ctx=ast.Load()), slice=ast.Constant('grid'), ctx=ast.Load())], keywords=[])
        if isinstance(node.func, ast.Name) and node.func.id == 'brightness_guard':
            node.func = ast.Attribute(value=ast.Name(id='corrected', ctx=ast.Load()), attr='brightness_guard', ctx=ast.Load())
        return node


class NormalizeHinges(ast.NodeTransformer):
    def __init__(self): self.changed = 0
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == 'relu'
                and isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch'):
            assert len(node.args) == 1 and not node.keywords
            self.changed += 1
            return ast.Call(func=ast.Attribute(value=node.args[0], attr='clamp_min', ctx=ast.Load()),
                            args=[ast.Constant(0)], keywords=[])
        return node


def equivalence_checks(packet):
    old_source = (SOURCE/'scripts/cctv_dgp_spatial_readout_v1_vm.py').read_text(encoding='utf-8')
    new_source = (packet/'scripts/cctv_dgp_decoder_hinge_repair_v1_vm.py').read_text(encoding='utf-8')
    old_objective = NormalizeObjective().visit(function(old_source, 'objective'))
    new_objective = NormalizeObjective().visit(function(new_source, 'objective'))
    assert ast.dump(old_objective, include_attributes=False) == ast.dump(new_objective, include_attributes=False), 'Unexpected objective arithmetic change'
    old_helper = function((PARENT/'cctv_dgp_bank_photometric_guard_v1_loss.py').read_text(encoding='utf-8'), 'brightness_guard')
    new_helper = function((packet/'cctv_dgp_bank_hinge_boundary_v1_loss.py').read_text(encoding='utf-8'), 'brightness_guard')
    mapper = NormalizeHinges(); new_helper = mapper.visit(new_helper)
    assert mapper.changed == 3 and ast.dump(old_helper, include_attributes=False) == ast.dump(new_helper, include_attributes=False)
    verify_corrected_guard_binding(new_source)
    try: verify_corrected_guard_binding(old_source)
    except AssertionError: pass
    else: raise AssertionError('Known legacy training import regression was accepted')
    tree = ast.parse(new_source, feature_version=(3, 10))
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
    first = run.body[0]
    assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call) and first.value.func.id == 'machine_guard'
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) and any(alias.name.split('.')[0] == 'torch' for alias in node.names)
                   for node in tree.body)
    historical = [node for node in ast.walk(run) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and isinstance(node.func.value, ast.Name) and node.func.value.id == 'loader']
    assert len(historical) == 1 and historical[0].func.attr == 'load_dependencies'
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'train' for node in ast.walk(tree))
    return dict(source_objective_equivalent_except_declared_guard=True, helper_only_three_boundary_operators_changed=True,
                known_legacy_objective_rejected=True, worker_guard_precedes_model_imports=True, historical_training_not_called=True)


def replay_publication():
    base = ROOT/'outputs/cctv_dgp_spatial_path_replay_v1_analysis'
    receipt = read(base/'publication/publication.json')
    assert receipt['complete'] and len(receipt['documents']) == 10
    for name, row in receipt['documents'].items():
        before = Path(row['backup']).read_bytes(); after = (ROOT/name).read_bytes()
        assert sha(Path(row['backup'])) == row['before_sha256']
        assert sha(ROOT/name) == row['after_sha256']
        assert after[row['prefix_bytes']:] == before and row['old_bytes_preserved_as_exact_suffix']
    for name, digest in receipt['new_document_sha256'].items(): assert sha(ROOT/name) == digest
    for name, digest in receipt['evidence_sha256'].items(): assert sha(base/name) == digest
    assert sha(base/'learning_path.png') == receipt['plot_sha256']
    assert sha(ROOT/'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth') == receipt['baseline_sha256']
    return dict(complete=True, receipt_sha256=sha(base/'publication/publication.json'),
        preserved_documents=10, evidence_receipts=len(receipt['evidence_sha256']), original_checkpoint_unchanged=True)


def main():
    prepared = read(DESIGN/'preparation.json'); packet = ROOT/'outputs'/NAME
    assert prepared['complete'] and not (packet/'outputs').exists()
    archive = ROOT/'outputs'/(STEM+'-execution.tar.gz')
    assert sha(archive) == prepared['archive_sha256'] and archive.stat().st_size == prepared['archive_bytes']
    expected_sidecar = (prepared['archive_sha256']+'  '+archive.name+'\n').encode('utf-8')
    assert Path(str(archive)+'.sha256').read_bytes() == expected_sidecar
    assert b'\r' not in expected_sidecar and not expected_sidecar.startswith(b'\xef\xbb\xbf')
    pin = sha(packet/'protocol.json'); assert pin == prepared['protocol_sha256']
    p, old, data = verify_transfer(packet, SOURCE, PARENT, pin, replay=REPLAY)
    assert sha(packet/'design.json') == sha(DESIGN/'design.json')
    assert len(p['source_required_sha256']) == 410 and len(p['parent_required_sha256']) == 203
    expected = {NAME+'/'+name for name in p['assets_sha256']} | {NAME+'/protocol.json'}
    with tarfile.open(archive, 'r:gz') as pack:
        members = pack.getmembers()
        assert len(members) == len(expected) and {member.name for member in members} == expected
        assert len({member.name for member in members}) == len(members)
        for member in members:
            name = PurePosixPath(member.name)
            assert member.isreg() and not name.is_absolute() and '..' not in name.parts and str(name) == member.name
            assert '\\' not in member.name and ':' not in member.name
            assert pack.extractfile(member).read() == (packet/member.name.removeprefix(NAME+'/')).read_bytes()
    assert sha(PARENT/'cctv_dgp_bank_photometric_guard_v1_loss.py') == LEGACY_HELPER_SHA
    assert sha(packet/'cctv_dgp_bank_hinge_boundary_v1_loss.py') == CORRECTED_HELPER_SHA
    for path in packet.rglob('*.py'): ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 10))
    checks = equivalence_checks(packet)
    design = read(packet/'design.json'); mutations = 0
    for key in ['automatic_launch', 'model_qualified', 'new_incumbent_selection', 'failed_recipe_resume_allowed',
                'local_neural_construction_allowed', 'learning_rate_changed']:
        changed = copy.deepcopy(p); changed[key] = True
        try: validate(changed, design)
        except AssertionError: mutations += 1
        else: raise AssertionError('Invalid design mutation accepted: '+key)
    command = [str(ROOT/'venv/Scripts/python.exe'), '-B', str(packet/'scripts/cctv_dgp_decoder_hinge_repair_v1_vm.py'),
        '--root', str(packet), '--source-root', str(SOURCE), '--parent-root', str(PARENT),
        '--replay-root', str(REPLAY), '--protocol-sha', pin]
    transfer = subprocess.run(command+['--verify-transfer'], capture_output=True, text=True, timeout=180)
    with (DESIGN/'actual_transfer_CLI.txt').open('x', encoding='utf-8', newline='\n') as stream: stream.write(transfer.stdout+transfer.stderr)
    assert transfer.returncode == 0
    cli = json.loads(transfer.stdout)
    assert cli['optimizer_updates'] == cli['gradient_queries'] == cli['neural_calls'] == 0
    assert cli['source_assets'] == 410 and cli['parent_assets'] == 203 and cli['packet_assets'] == len(p['assets_sha256'])
    guarded = subprocess.run(command+['--run'], capture_output=True, text=True, timeout=20)
    assert guarded.returncode != 0 and 'Human manual Linux L4 tmux only' in guarded.stderr
    assert not (packet/'outputs').exists()
    with (DESIGN/'Windows_guard.txt').open('x', encoding='utf-8', newline='\n') as stream: stream.write(guarded.stdout+guarded.stderr)
    help_result = subprocess.run(command+['--help'], capture_output=True, text=True, timeout=20)
    assert help_result.returncode == 0 and all(flag in help_result.stdout for flag in ['--run', '--audit-saved', '--export', '--supervise'])
    assert '--train' not in help_result.stdout
    tests = read(DESIGN/'cpu_tests.json')
    assert tests['complete'] and tests['exit_code'] == 0 and tests['tests_run'] >= 14
    assert tests['test_sha256'] == sha(ROOT/'scripts/test_cctv_dgp_decoder_hinge_repair_v1.py')
    assert tests['checker_sha256'] == sha(packet/'audit_cctv_dgp_decoder_hinge_repair_v1_return.py')
    assert tests['worker_sha256'] == sha(packet/'scripts/cctv_dgp_decoder_hinge_repair_v1_vm.py')
    assert tests['local_neural_calls'] == tests['local_gradient_queries'] == tests['local_optimizer_updates'] == 0
    calibration = read(DESIGN/'scorer_calibration.json')
    assert calibration['complete'] and calibration['compositions'] == calibration['raw_compositions'] == calibration['png_compositions'] == 200
    assert calibration['source_count'] == 2 and calibration['exact_control_pixels_checked'] and calibration['saved_metrics_match_prospective_checker']
    assert calibration['checker_sha256'] == tests['checker_sha256']
    assert calibration['neural_calls'] == calibration['gradient_queries'] == calibration['optimizer_updates'] == 0
    bash = read(DESIGN/'bash_syntax.json')
    shell_path = packet/'scripts/run_decoder_hinge.sh'
    assert bash['exit_code'] == 0 and bash['shell_sha256'] == sha(shell_path)
    shell = shell_path.read_bytes()
    assert b'\r' not in shell and b'${TMUX:-}' in shell and b'--supervise' in shell
    protected = read(DESIGN/'protected_before_preparation.json')
    assert protected['complete'] and all(sha(ROOT/name) == digest for name, digest in protected['files'].items())
    publication = replay_publication()
    record = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), status='VERIFIED_PREPARED_UNRUN',
        protocol_sha256=pin, archive_sha256=prepared['archive_sha256'], archive_bytes=archive.stat().st_size,
        archive_members=len(expected), packet_assets=len(p['assets_sha256']), source_assets=410, parent_assets=203,
        replay_assets=len(p['replay_required_sha256']), Python310_AST_pass=True, actual_hash_only_CLI_checked=True,
        Windows_pre_neural_guard_checked=True, invalid_protocol_mutations_rejected=mutations,
        CPU_tests=tests['tests_run'], CPU_test_receipt_sha256=sha(DESIGN/'cpu_tests.json'), Bash_syntax_pass=True,
        scorer_calibration_compositions=200, scorer_calibration_receipt_sha256=sha(DESIGN/'scorer_calibration.json'),
        replay_publication_verified=publication, source_control_bindings_unchanged=True,
        manual_L4_parity_derivatives_timing_not_yet_verified=True, neural_calls=0, gradient_queries=0,
        optimizer_updates=0, agent_VM_calls=0, model_qualified=False, **checks)
    record['preparation_auditor_sha256'] = sha(Path(__file__))
    write(DESIGN/'preparation_audit.json', record)
    print(json.dumps(record))


if __name__ == '__main__': main()
