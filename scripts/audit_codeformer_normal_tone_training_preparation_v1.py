"""Fresh CPU-only training packet readback, dependencies and finite census."""
import ast
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_normal_tone_training_v1 import contract as c
from research.codeformer_normal_tone_training_v1 import transport as t


def main():
    start = time.monotonic()
    review = ROOT / 'outputs/codeformer_normal_tone_training_v1_preparation'
    destination = review / 'readiness_v1.json'
    c.require(not destination.exists(), 'Preserve previous readiness result')
    staged = c.read_json(review / 'stage_v1.json')
    stage = ROOT / 'outputs' / c.VM_NAME
    p = c.read_json(stage / 'protocol.json')
    c.verify_protocol(p)
    protocol_sha = c.sha256(stage / 'protocol.json')
    c.require(staged['complete'] is True and staged['protocol_sha256'] == protocol_sha,
              'Exact closed stage receipt required')
    wanted = {**p['files_sha256'], 'protocol.json': protocol_sha}
    sizes = {**p['files_bytes'], 'protocol.json': (stage / 'protocol.json').stat().st_size}
    files = t._walk(stage)
    c.require(set(files) == set(wanted), 'Exact source-only namespace differs')
    current = {}
    syntax = []
    for name, path in files.items():
        c.require(path.stat().st_size == sizes[name] and c.sha256(path) == wanted[name], 'Source byte binding differs')
        if name == 'protocol.json':
            continue
        if name == 'research/__init__.py':
            c.require(path.read_bytes() == b'', 'Empty package marker required')
            continue
        origin = staged['current_sources'][name]
        c.safe_relative(origin)
        c.require(c.sha256(ROOT / origin) == wanted[name], 'Current source differs from staged source: ' + origin)
        current[origin] = wanted[name]
        if name.endswith('.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=name, feature_version=(3, 10))
            syntax.append(name)
    c.require(staged['dependencies'] == p['dependencies'] and
              set(staged['local_dependency_resolution']) == set(p['dependencies']), 'Dependency resolution differs')
    byte_count = 0
    for name, record in p['dependencies'].items():
        c.safe_relative(name)
        path = c.no_links(Path(staged['local_dependency_resolution'][name]))
        c.require(path.resolve(strict=True).is_relative_to(ROOT), 'Local dependency outside workspace')
        before = path.stat()
        c.require(path.is_file() and before.st_size == record['bytes'] and c.sha256(path) == record['sha256'],
                  'Fresh parent bytes differ: ' + name)
        after = path.stat()
        c.require((before.st_size, before.st_mtime_ns, before.st_ino) ==
                  (after.st_size, after.st_mtime_ns, after.st_ino), 'Dependency changed while hashing')
        byte_count += before.st_size
    c.require(len(p['dependencies']) == 2011, 'Exact dependency census differs')
    def old_json(name):
        return c.read_json(Path(staged['local_dependency_resolution'][name]))
    pairs = old_json('training_parity_v1.json')
    hq = old_json('hq_cache_parity_v1.json')
    validation = old_json('validation_saved_v1.json')
    comparisons = old_json('comparison_cases.json')
    c.require(len(pairs['cases']) == 200 and len(pairs['updates']) == 50 and
              len(pairs['behavior_anchors']) == 50 and len(hq['cases']) == 391 and
              validation['case_count'] == 318 and len(comparisons['cases']) == 88,
              'Complete TRAIN/HQ/validation/comparison support required')
    c.require(c.expected_call_caps() == c.TRAINING['call_caps'], 'Source-derived training caps differ')
    caps = c.TRAINING['call_caps']
    neural_keys = ('route_witness_forwards', 'route_witness_perceptual_forwards',
        'perceptual_prerequisite_forwards', 'candidate_training_forwards', 'perceptual_training_forwards',
        'behavior_teacher_forwards', 'teacher_forwards', 'snapshot_forwards', 'baseline_forwards',
        'initializer_parity_candidate_forwards')
    c.require(sum(caps[key] for key in neural_keys) == 1614 and sum(caps.values()) == 2075 and
              caps['gradient_queries'] == 11 and caps['backward_calls'] == 200 and
              caps['optimizer_step_attempts'] == 50, 'Finite 1614/11/200/50 ledger differs')
    independent_path = review / 'independent_source_review_v1.json'
    independent = c.read_json(independent_path)
    c.require(independent['complete'] is True and independent['unresolved_findings'] == [],
              'Closed independent source review required')
    for name, digest in independent['current_source_sha256'].items():
        c.safe_relative(name)
        c.require(c.sha256(ROOT / name) == digest, 'Reviewed source changed: ' + name)
    modules = ['tests.test_codeformer_normal_tone_math_v1',
               'tests.test_codeformer_normal_tone_training_math_v1',
               'tests.test_codeformer_normal_tone_training_worker_v1',
               'tests.test_codeformer_normal_tone_training_transport_v1',
               'tests.test_codeformer_normal_tone_seal_v1']
    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
    c.require(result.wasSuccessful(), 'Preparation fixture failure; preserve result')
    evidence = ['outputs/codeformer_normal_tone_training_v1_preparation/stage_v1.json',
        'outputs/codeformer_normal_tone_training_v1_preparation/independent_source_review_v1.json',
        'outputs/codeformer_normal_tone_training_v1_preparation/independent_source_review_v1.md',
        'scripts/audit_codeformer_normal_tone_training_preparation_v1.py',
        'scripts/seal_codeformer_normal_tone_training_v1.py',
        'scripts/seal_codeformer_normal_tone_v1.py']
    evidence += [name.replace('.', '/') + '.py' for name in modules]
    c.write_new_json(destination, {'complete': True, 'status': 'PREPARED_UNRUN',
        'UTC': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': protocol_sha,
        'source_files': wanted, 'source_bytes': sizes, 'current_sources_sha256': current,
        'dependencies_freshly_hashed': 2011, 'dependency_bytes': byte_count,
        'Python310_AST_sources': syntax, 'TRAIN_observations': 200, 'HQ_cache_entries': 391,
        'validation_saved': 318, 'comparisons': 88, 'clear_guards': 61, 'normal_blur_guards': 53,
        'call_caps': caps, 'neural_forward_cap': 1614, 'reservation_events': 2075,
        'checkpoint_paths': c.CHECKPOINT_PATHS,
        'independent_source_review_complete': True, 'tests_run': result.testsRun,
        'tests_failed': len(result.failures) + len(result.errors), 'tests_skipped': len(result.skipped),
        'evidence_sha256': {name: c.sha256(ROOT / name) for name in evidence},
        'seconds': time.monotonic() - start, 'local_neural_gradient_optimizer_VM_calls': 0,
        'candidate_checkpoint_created': False, 'model_qualified': False})
    print({'complete': True, 'protocol_sha256': protocol_sha, 'tests': result.testsRun,
           'dependencies': 2011, 'neural_forward_cap': 1614, 'actual_training_launched': False})


if __name__ == '__main__':
    main()
