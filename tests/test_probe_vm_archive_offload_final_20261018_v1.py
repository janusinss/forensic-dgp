"""Pure local/source fixtures; no VM/API, Torch, payload models or filesystem writes."""
import ast
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('final_probe', ROOT / 'scripts/probe_vm_archive_offload_final_20261018_v1.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def valid_apply():
    local_payload = dict(path=str(m.OUT / 'apply_r1/apply_payload.json.gz'), bytes=1234, sha256='a' * 64)
    guest_payload = dict(path='/home/janusdominic0/vm_archive_offload_20261018_v1/apply_payload.json.gz',
                         bytes=1234, sha256='a' * 64, metadata={'inode': 1})
    guest = dict(complete=True, mode='apply', files_removed=6, logical_bytes_removed=m.TOTAL_BYTES,
                 plan_sha256=m.PLAN_SHA, all_research_namespace_and_protected_content_unchanged=True,
                 training_calls=0, payload=guest_payload)
    return dict(complete=True, mode='apply', files_removed=6, logical_bytes_removed=m.TOTAL_BYTES,
                exact_plan_sha256=m.PLAN_SHA, training_calls=0, guest_receipt=guest, payload=local_payload)


class FinalProbeFixtures(unittest.TestCase):
    def test_valid_distinct_local_remote_payload_paths(self):
        value = valid_apply()
        self.assertEqual(m.check_apply(value), value['guest_receipt'])

    def test_reject_incomplete_or_partial_apply(self):
        for key, value in [('complete', False), ('files_removed', 5), ('logical_bytes_removed', 1),
                           ('exact_plan_sha256', '0' * 64), ('training_calls', 1)]:
            x = valid_apply(); x[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.check_apply(x)

    def test_reject_guest_scope_or_protection_failure(self):
        for key, value in [('complete', False), ('files_removed', 0), ('plan_sha256', '0' * 64),
                           ('all_research_namespace_and_protected_content_unchanged', False)]:
            x = valid_apply(); x['guest_receipt'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.check_apply(x)

    def test_reject_wrong_payload_hash_size_path(self):
        for key, value in [('sha256', 'b' * 64), ('bytes', 12), ('path', '/tmp/other')]:
            x = valid_apply(); x['payload'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.check_apply(x)

    def test_remote_source_uses_only_readonly_guest_entrypoints(self):
        tree = ast.parse(m.probe_source(valid_apply()['guest_receipt']))
        called = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
                func = node.func
                if isinstance(func.value, ast.Name) and func.value.id == 'g':
                    called.add(ast.literal_eval(func.slice))
        self.assertEqual(called, {'runtime_identity', 'bounded_read_json', 'validate_request', 'canonical_sha',
                                 'path_policy', 'workload_guard', 'open_reader_guard', 'namespace', 'disk'})
        self.assertNotIn('protected_hashes', called)
        self.assertNotIn('apply', called)
        self.assertIn("run_name='_readonly_final_probe_20261018'", m.PROBE)
        self.assertIn("not os.path.lexists(p)", m.PROBE)
        self.assertIn("signal.alarm(240)", m.PROBE)

    def test_source_and_lineage_prepare_readonly(self):
        client, prep, plan = m.prepare()
        self.assertEqual(prep['guest_source']['sha256'], m.GUEST_SHA)
        self.assertEqual(plan['verified']['namespace_sha256'], m.NAMESPACE_SHA)
        self.assertEqual(client.pin(m.PLAN)['sha256'], m.PLAN_SHA)
        # This constructs argv only. No invoke/API/SSH is called by prepare or ssh_args.
        argv = client.ssh_args(m.probe_source(valid_apply()['guest_receipt']))
        self.assertTrue(any(x.startswith('--command=python3 -B ') for x in argv))
        self.assertTrue(any('hostkey' in x for x in argv))

    def test_canonical_apply_receipt_hash_order_independent(self):
        guest = valid_apply()['guest_receipt']
        self.assertEqual(m.canonical_sha(guest), m.canonical_sha(dict(reversed(list(guest.items())))))


if __name__ == '__main__':
    unittest.main()
