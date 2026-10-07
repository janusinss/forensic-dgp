"""Reuse the exact audited V25 stopped-layer diagnosis for the new saved V26 head."""
import ast
import json
from pathlib import Path
import time

from import_cctv_dgp_spatial_features_v25 import read, require, sha, write

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_diagnostic_preparation'


def main():
    started = time.monotonic()
    require(not OUT.exists(), 'Preserve previous diagnostic preparation')
    bases = {
        'scripts/diagnose_cctv_dgp_spatial_features_v25.py': '10229371d10746b74438e136363003966091e0daade1fb604922d49aeebc1e05',
        'scripts/verify_cctv_dgp_spatial_features_v25_diagnostic.py': 'd6d452852251cec7f14539b542f802faca16c709ac4c2eca8d4b95460465a29a',
    }
    audit_path = ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_independent_audit.json'
    audit = read(audit_path)
    require(audit['complete'] and audit['complete_snapshot_updates'] == [0, 50] and
            not audit['early_structure_stop']['pass'] and audit['identity_preflight_audit']['batchmatched_component_gradient_norm'] == 0,
            'Audited V26 stop and corrected identity proof required')
    mapping = {
        'cctv_dgp_spatial_features_v25_diagnostic': 'cctv_dgp_batchmatched_identity_v26_diagnostic',
        'cctv_dgp_spatial_features_v25_return': 'cctv_dgp_batchmatched_identity_v26_return',
        'cctv_dgp_spatial_features_v25_independent_audit': 'cctv_dgp_batchmatched_identity_v26_independent_audit',
        'audit_cctv_dgp_spatial_features_v25': 'audit_cctv_dgp_batchmatched_identity_v26',
        'saved V25 stop diagnosis': 'saved V26 stop diagnosis',
        'V25 stopped50': 'V26 stopped50',
    }
    generated = {}
    for name, digest in bases.items():
        require(sha(ROOT / name) == digest, 'Original diagnostic changed')
        destination = name.replace('spatial_features_v25', 'batchmatched_identity_v26')
        require(not (ROOT / destination).exists(), 'Preserve prepared diagnosis')
        text = (ROOT / name).read_text(encoding='utf-8')
        for old, new in mapping.items():
            text = text.replace(old, new)
        tree = ast.parse(text, feature_version=(3, 10))
        class RestoreNames(ast.NodeTransformer):
            def visit_Constant(self, node):
                if isinstance(node.value, str):
                    for old, new in mapping.items():
                        node.value = node.value.replace(new, old)
                return node
            def visit_ImportFrom(self, node):
                for old, new in mapping.items():
                    node.module = node.module.replace(new, old)
                return node
            def visit_Import(self, node):
                for alias in node.names:
                    for old, new in mapping.items():
                        alias.name = alias.name.replace(new, old)
                return node
        original = ast.parse((ROOT / name).read_text())
        require(ast.dump(RestoreNames().visit(tree)) == ast.dump(original), 'Unexpected diagnostic math/trace/budget change')
        generated[destination] = text
    OUT.mkdir()
    for name, text in generated.items():
        with (ROOT / name).open('x', encoding='utf-8', newline='\n') as handle:
            handle.write(text)
    result = {'complete': True, 'date': '2026-10-06', 'scope': 'Saved V26 failure analysis; no recipe change',
              'original_sources_sha256': bases, 'sources_sha256': {name: sha(ROOT / name) for name in generated},
              'independent_return_audit_sha256': sha(audit_path), 'runner_sha256': sha(Path(__file__)),
              'original_exact_layer_and_metric_AST_preserved': True, 'planned_head_layer_traces': 50,
              'seconds_cap': 300, 'planned_exact_review_cells': 200, 'local_model_calls': 0,
              'local_gradient_calls': 0, 'local_backward_calls': 0, 'local_optimizer_updates': 0,
              'VM_actions': False, 'app_promotion': False, 'goal_complete': False,
              'seconds': time.monotonic() - started}
    write(OUT / 'preparation.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
