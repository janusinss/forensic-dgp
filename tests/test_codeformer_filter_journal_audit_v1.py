"""Adversarial inert chronology fixtures; no models, outputs or frameworks."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('independent_journal_audit',
    ROOT / 'scripts/audit_codeformer_filter_comparison_journal_v1.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def fixture():
    resource = {'process_RSS_bytes': 1024**3, 'available_RAM_bytes': 2 * 1024**3,
                'disk_free_bytes': 2 * 1024**3}
    selected = [{'id': 'case_%02d' % n, 'input': {'sha256': '%064x' % n}} for n in range(72)]
    cases = {r['id']: dict(r, outputs={label: {'evidence': {'path': label + '/' + r['id']}}
             for _, label in a.ARMS}) for r in selected}
    events, call = [], 0
    for arm, label in a.ARMS:
        events.append({'event': 'model_loaded_frozen', 'arm': arm, 'state_sha256': a.STATE_SHA[arm],
                       'resources': dict(resource)})
        for case in selected:
            call += 1
            events.extend([
                {'event': 'forward_reserved', 'call': call, 'arm': arm, 'id': case['id'],
                 'input_sha256': case['input']['sha256'], 'seconds': float(call)},
                {'event': 'forward_returned', 'call': call, 'arm': arm, 'id': case['id'],
                 'seconds': .25, 'resources': dict(resource)},
                {'event': 'output_committed', 'call': call, 'arm': arm, 'id': case['id'],
                 'evidence': cases[case['id']]['outputs'][label]['evidence']}])
        events.append({'event': 'model_finished_frozen', 'arm': arm, 'state_sha256': a.STATE_SHA[arm]})
    events.append({'event': 'complete', 'progress': {'reserved_forwards': 216, 'returned_forwards': 216,
                   'committed_outputs': 216, 'in_flight': None, 'current_arm': 'r2'}, 'resources': dict(resource)})
    return events, selected, cases


class InertJournalFixtures(unittest.TestCase):
    def test_exact_full_chronology_without_framework(self):
        events, selected, cases = fixture()
        samples, times = a.journal_closure(events, selected, cases, {})
        self.assertEqual(len(events), 655)
        self.assertEqual(len(samples), 220)
        self.assertEqual(len(times), 216)
        self.assertEqual(sum(times), 54.)
        self.assertNotIn('torch', sys.modules)

    def test_missing_duplicate_or_swapped_events_fail(self):
        baseline, selected, cases = fixture()
        variants = [baseline[:-1], baseline + [baseline[-1]]]
        swapped = copy.deepcopy(baseline); swapped[1], swapped[2] = swapped[2], swapped[1]
        variants.append(swapped)
        duplicate = copy.deepcopy(baseline); duplicate[4]['call'] = 1
        variants.append(duplicate)
        for events in variants:
            with self.subTest(length=len(events)), self.assertRaises(ValueError):
                a.journal_closure(events, selected, cases, {})

    def test_wrong_model_or_committed_pin_fails(self):
        baseline, selected, cases = fixture()
        wrong_state = copy.deepcopy(baseline); wrong_state[0]['state_sha256'] = '0' * 64
        wrong_pin = copy.deepcopy(baseline); wrong_pin[3]['evidence'] = {'path': 'other'}
        for events in (wrong_state, wrong_pin):
            with self.assertRaises(ValueError):
                a.journal_closure(events, selected, cases, {})

    def test_resource_reserve_and_clock_violations_fail(self):
        baseline, selected, cases = fixture()
        for key, value in [('process_RSS_bytes', a.CAPS['process_RSS_bytes'] + 1),
                           ('available_RAM_bytes', a.CAPS['minimum_available_RAM_bytes'] - 1),
                           ('disk_free_bytes', a.CAPS['minimum_disk_free_bytes'] - 1)]:
            events = copy.deepcopy(baseline); events[2]['resources'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                a.journal_closure(events, selected, cases, {})
        events = copy.deepcopy(baseline); events[4]['seconds'] = .5
        with self.assertRaisesRegex(ValueError, 'clock'):
            a.journal_closure(events, selected, cases, {})

    def test_unfinished_final_progress_or_extra_model_interleaving_fails(self):
        baseline, selected, cases = fixture()
        events = copy.deepcopy(baseline); events[-1]['progress']['in_flight'] = {'call': 216}
        with self.assertRaisesRegex(ValueError, 'Final progress'):
            a.journal_closure(events, selected, cases, {})
        events = copy.deepcopy(baseline); events[4] = events[218]
        with self.assertRaises(ValueError):
            a.journal_closure(events, selected, cases, {})

    def test_strict_metadata_rejects_duplicate_nonfinite_and_paths(self):
        for payload in (b'{"a":1,"a":2}', b'{"a":NaN}'):
            with self.assertRaises(ValueError):
                a.strict_json(payload)
        for name in ('../x', 'a/../b', '/x', 'C:x', 'a\\b', 'a//b', ''):
            with self.subTest(name=name), self.assertRaises(ValueError):
                a.canonical(name)


if __name__ == '__main__':
    unittest.main()
