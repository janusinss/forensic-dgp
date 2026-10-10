"""Run frozen packet checks with inheriting workspace fixture permissions only."""
import ast
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/cctv_dgp_spatial_decoder_v39_preparation'
CHECKER = ROOT / 'scripts/verify_cctv_dgp_spatial_decoder_v39_packet.py'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@contextmanager
def fixture_directory(prefix, dir):
    assert Path(dir).resolve() == PREP.resolve() and prefix == 'V39-array-fixtures-'
    path = (PREP / 'r1_synthetic_fixture_workspace').resolve()
    assert path.is_relative_to(PREP.resolve()) and path.is_relative_to(ROOT)
    assert not path.exists(), 'Retain previous fixture evidence'
    path.mkdir()  # Inherit workspace access; no chmod/security-descriptor rewrite.
    yield str(path)
    # Keep these synthetic fixtures explicitly separate from returned VM evidence.


def main():
    start = time.monotonic()
    protocol = json.loads((ROOT / 'outputs/cctv_dgp_spatial_decoder_vm_v39/protocol.json').read_text(encoding='utf-8'))
    assert sha(CHECKER) == protocol['local_basis_sha256'][CHECKER.relative_to(ROOT).as_posix()]
    assert (PREP / 'packet_audit_sandbox_failure.json').is_file()
    assert (PREP / 'Windows_pre_neural_rejection.txt').is_file()
    assert not (PREP / 'independent_packet_audit.json').exists()
    original = ast.parse(CHECKER.read_text(encoding='utf-8'))
    class FilesystemOnly(ast.NodeTransformer):
        changes = 0
        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == 'tempfile' and node.attr == 'TemporaryDirectory':
                self.changes += 1
                return ast.copy_location(ast.Name(id='fixture_directory', ctx=ast.Load()), node)
            return self.generic_visit(node)
        def visit_Constant(self, node):
            if node.value == 'Windows_pre_neural_rejection.txt':
                self.changes += 1
                return ast.copy_location(ast.Constant(value='Windows_pre_neural_rejection_r1.txt'), node)
            return node
    transform = FilesystemOnly(); changed = ast.fix_missing_locations(transform.visit(original))
    assert transform.changes == 2
    source = ast.unparse(changed)
    inverse = source.replace("'Windows_pre_neural_rejection_r1.txt'", "'Windows_pre_neural_rejection.txt'").replace('with fixture_directory(', 'with tempfile.TemporaryDirectory(')
    assert ast.dump(ast.parse(inverse), include_attributes=False) == ast.dump(ast.parse(CHECKER.read_text(encoding='utf-8')), include_attributes=False)
    namespace = {'__file__': str(CHECKER), '__name__': 'V39_frozen_packet_checker_filesystem_r1', 'fixture_directory': fixture_directory}
    exec(compile(changed, '<frozen-V39-packet-checker-filesystem-only-r1>', 'exec'), namespace)
    namespace['main']()
    assert sha(CHECKER) == protocol['local_basis_sha256'][CHECKER.relative_to(ROOT).as_posix()]
    result = {'complete': True, 'runner_sha256': sha(Path(__file__)), 'frozen_checker_sha256': sha(CHECKER),
              'independent_packet_audit_sha256': sha(PREP / 'independent_packet_audit.json'),
              'original_partial_log_preserved': True, 'original_sandbox_failure_preserved': True,
              'changes': ['Inherit access for a normal workspace fixture directory', 'Separate R1 pre-neural rejection log'],
              'all_scientific_and_archive_and_command_check_AST_unchanged': True,
              'packet_worker_protocol_and_gates_unchanged': True,
              'retained_fixture_arrays_are_synthetic_not_VM_evidence': True,
              'neural_calls': 0, 'gradient_calls': 0, 'optimizer_updates': 0, 'VM_calls': 0,
              'seconds': time.monotonic() - start}
    with (PREP / 'independent_packet_audit_r1_execution.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'R1_execution_complete': True, 'seconds': result['seconds']}), flush=True)


if __name__ == '__main__':
    main()
