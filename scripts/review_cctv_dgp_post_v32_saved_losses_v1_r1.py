"""Correct the saved-loss review's reference field; retain the failed preparation."""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'scripts/review_cctv_dgp_post_v32_saved_losses_v1.py'
PIN = '05f057f6cca63e88128fee7de2a45bbdc8c23dbc19ba8894299523ebe4987000'


def main():
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == PIN
    source = ORIGINAL.read_text(encoding='utf-8')
    replacements = {
        "OUT = ROOT / 'outputs/cctv_dgp_post_v32_saved_losses_v1'":
        "OUT = ROOT / 'outputs/cctv_dgp_post_v32_saved_losses_v1_r1'",
        "references[c['reference']]['matrix112']":
        "references[c['source_person_or_reference']]['matrix112']",
        "files = [Path(__file__), audit_path, import_path,":
        "files = [Path(__file__), ROOT / 'scripts/review_cctv_dgp_post_v32_saved_losses_v1.py', ROOT / 'outputs/cctv_dgp_post_v32_saved_losses_v1/plan.json', ROOT / 'outputs/cctv_dgp_post_v32_saved_losses_v1/analysis_failure.json', audit_path, import_path,",
    }
    for before, after in replacements.items():
        assert source.count(before) == 1
        source = source.replace(before, after)
    tree = ast.parse(source)
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ['grad', 'backward', 'step', 'AdamW', 'SGD'] for n in ast.walk(tree))
    namespace = {'__name__': 'pinned_saved_loss_review_r1', '__file__': str(Path(__file__).resolve())}
    exec(compile(tree, str(ORIGINAL), 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
