"""Create a new initializer loader with one Python-3.10-compatible hash repair."""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    prior = ROOT / 'scripts/cctv_dgp_bank_pinned_initializer_v1.py'
    dest = ROOT / 'scripts/cctv_dgp_bank_pinned_initializer_v2.py'
    assert not dest.exists()
    assert hashlib.sha256(prior.read_bytes()).hexdigest() == '6fb9f84be1882c78bc8aaf87e2a6f4ca3ec5646d8b210db560ec059435236709'
    old = prior.read_text(encoding='utf-8')
    before = """def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()"""
    after = """def sha(path):
    # hashlib.file_digest is unavailable on the recorded Python 3.10 VM.
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()"""
    assert old.count(before) == 1
    new = old.replace(before, after)
    old_tree = ast.parse(old, feature_version=(3, 10))
    new_tree = ast.parse(new, feature_version=(3, 10))
    functions = lambda tree: {node.name: ast.dump(node, include_attributes=False)
                              for node in tree.body if isinstance(node, ast.FunctionDef)}
    old_functions, new_functions = functions(old_tree), functions(new_tree)
    assert old_functions.keys() == new_functions.keys()
    assert {name for name in old_functions if old_functions[name] != new_functions[name]} == {'sha'}
    with dest.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(new)
    assert prior.read_text(encoding='utf-8') == old
    print(dict(complete=True, only_function_changed='sha', old_loader_retained=True,
               neural_gradient_or_optimizer_calls=0), flush=True)


if __name__ == '__main__':
    main()
