"""Fresh CPU-only source/data/readiness audit; no model, VM or tensor loading."""
import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_normal_tone_v1 import contract as c
from research.codeformer_normal_tone_v1 import transport as t
from scripts import seal_codeformer_normal_tone_v1 as seal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path,
        default=ROOT / 'outputs/codeformer_normal_tone_v1_preparation/readiness_v1.json')
    args = parser.parse_args()
    c.require(not args.receipt.exists(), 'Preserve prior readiness audit')
    start = time.monotonic()
    stage = ROOT / 'outputs' / c.VM_NAME
    staged_receipt = ROOT / 'outputs/codeformer_normal_tone_v1_preparation/stage_v1.json'
    staged = c.read_json(staged_receipt)
    protocol_sha = c.sha256(stage / 'protocol.json')
    c.require(staged['complete'] is True and staged['protocol_sha256'] == protocol_sha,
              'Closed matching stage receipt required')
    p, files, wanted, sizes = seal.inventory(stage, protocol_sha)
    seal.verify_current_sources(stage, wanted)
    c.verify_cases(c.read_json(stage / p['cases_file']))
    old = c.read_json(ROOT / 'outputs/codeformer_loss_response_v1_return/protocol.json')
    old_manifest = c.read_json(ROOT / 'outputs/codeformer_loss_response_v1_return/outputs/loss_response_v1/export_manifest.json')
    c.require(p['dependencies']['adain'] == old['dependencies'] and
              set(p['dependencies']['loss_response']) == c.required_loss_response_paths(old, old_manifest),
              'Exact prior dependency/control namespace differs')
    byte_counts, file_counts = {}, {}
    for alias, mapping in p['dependencies'].items():
        c.require(set(mapping) == set(staged['local_dependency_resolution'][alias]), 'Physical resolution incomplete')
        byte_counts[alias] = file_counts[alias] = 0
        for name, row in sorted(mapping.items()):
            c.safe_relative(name)
            path = c.no_links(Path(staged['local_dependency_resolution'][alias][name]))
            c.require(path.resolve(strict=True).is_relative_to(ROOT), 'Local resolution escapes research workspace')
            before = path.stat()
            c.require(before.st_size == row['bytes'] and c.sha256(path) == row['sha256'],
                      'Fresh parent binding differs: ' + alias + '/' + name)
            after = path.stat()
            c.require((before.st_size, before.st_mtime_ns, before.st_ino) ==
                      (after.st_size, after.st_mtime_ns, after.st_ino), 'Dependency changed during readback')
            byte_counts[alias] += before.st_size
            file_counts[alias] += 1
    c.require(file_counts == {'adain': 2011, 'loss_response': 2619}, 'Exact closed two-root census differs')
    syntax = []
    for name, path in files.items():
        if name.endswith('.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=name, feature_version=(3, 10))
            syntax.append(name)
    worker = ast.parse((stage / 'research/codeformer_normal_tone_v1/worker.py').read_text(encoding='utf-8'))
    declared = {node.targets[0].id: ast.literal_eval(node.value) for node in worker.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and
        node.targets[0].id in {'COUNTS', 'BRANCHES', 'COMPONENTS', 'LUMINANCE'}}
    c.require(declared['COUNTS'] == c.COUNTS and list(declared['BRANCHES']) == c.BRANCHES and
              list(declared['COMPONENTS']) == c.COMPONENTS and list(declared['LUMINANCE']) == [.2126, .7152, .0722],
              'Source-derived count/order/formula constants differ')
    calls = [node for node in ast.walk(worker) if isinstance(node, ast.Call)]
    paths = [ast.unparse(node.func) for node in calls]
    c.require(paths.count('model.infer') == 3 and paths.count('torch.autograd.grad') == 1 and
              paths.count('optimizer.step') == 1 and not any(value.endswith('.backward') for value in paths) and
              not any(value in {'torch.save', 'torch.jit.save'} for value in paths),
              'Unexpected neural/optimizer/backward/checkpoint call site')
    review_path = ROOT / 'outputs/codeformer_normal_tone_v1_preparation/independent_source_review_v1.json'
    independent = c.read_json(review_path)
    c.require(independent.get('complete') is True and independent.get('unresolved_findings') == [],
              'Closed independent source review required')
    for name, digest in independent['current_source_sha256'].items():
        c.safe_relative(name)
        c.require(c.sha256(ROOT / name) == digest, 'Independent reviewed source changed: ' + name)
    modules = ['tests.test_codeformer_normal_tone_' + part + '_v1'
               for part in ('math', 'seal', 'transport', 'worker')]
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    c.require(result.wasSuccessful(), 'Preparation boundary fixture failed; preserve output')
    evidence_names = ['outputs/codeformer_normal_tone_v1_preparation/stage_v1.json',
        'outputs/codeformer_normal_tone_v1_preparation/independent_source_review_v1.json',
        'outputs/codeformer_normal_tone_v1_preparation/independent_source_review_v1.md',
        'outputs/codeformer_normal_tone_v1_design/source_design_review_v1.json',
        'outputs/codeformer_normal_tone_v1_design/source_design_review_v1.md',
        'scripts/audit_codeformer_normal_tone_preparation_v1.py',
        'scripts/seal_codeformer_normal_tone_v1.py']
    evidence_names += [name.replace('.', '/') + '.py' for name in modules]
    receipt = {'complete': True, 'status': 'PREPARED_UNRUN', 'protocol_sha256': protocol_sha,
        'UTC': datetime.now(timezone.utc).isoformat(), 'source_files': wanted, 'source_bytes': sizes,
        'dependencies_freshly_hashed': file_counts, 'dependency_bytes': byte_counts,
        'source_cases': 16, 'Python310_AST_sources': syntax,
        'source_call_sites': {'model_infer': 3, 'autograd_grad': 1, 'Adam_step': 1,
                              'backward': 0, 'checkpoint_save': 0},
        'finite_call_caps': c.COUNTS, 'neural_forwards_cap': 224, 'gradient_queries_cap': 48,
        'real_disposable_steps_cap': 8, 'committed_training_updates': 0,
        'independent_source_review_complete': True,
        'tests_run': result.testsRun, 'tests_failed': len(result.failures) + len(result.errors),
        'tests_skipped': len(result.skipped), 'seconds': time.monotonic() - start,
        'evidence_sha256': {name: c.sha256(ROOT / name) for name in evidence_names},
        'local_neural_gradient_optimizer_VM_calls': 0, 'new_trained_model_created': False,
        'model_qualified': False, 'Linux_CUDA_runtime_unmeasured': True}
    c.write_new_json(args.receipt, receipt)
    print(json.dumps({key: receipt[key] for key in ('complete', 'protocol_sha256', 'tests_run',
        'tests_failed', 'dependencies_freshly_hashed', 'seconds', 'local_neural_gradient_optimizer_VM_calls')}))


if __name__ == '__main__':
    main()
