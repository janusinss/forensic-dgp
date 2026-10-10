"""Freeze a bounded audit of complete conditions in a stopped diagnostic.

The prospective full auditor remains immutable. Only condition membership and
the corresponding counts change; numerical and categorical checks are reused.
No model is constructed by this preparation script.
"""
from pathlib import Path
import ast
import inspect
import json
import time
from cctv_dgp_actual_step_review_v1_contract import NAME, PROPOSALS, read, write, sha

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs' / NAME
RETURN = ROOT / 'outputs' / (NAME + '_return')
OUT = ROOT / 'outputs/cctv_dgp_actual_step_review_v1_partial_audit'
GENERATED = ROOT / 'scripts/cctv_dgp_actual_step_review_v1_partial_checks.py'
ORIGINAL = ROOT / 'scripts/audit_cctv_dgp_actual_step_review_v1_return.py'


def replace_once(text, before, after, changes):
    assert text.count(before) == 1, before
    changes.append({'before': before, 'after': after})
    return text.replace(before, after, 1)


def main():
    start = time.monotonic()
    p = read(BUNDLE / 'protocol.json')
    assert sha(BUNDLE / 'protocol.json') == '339c20425b6fa6b4141001aae4ff3010bf9d438acc32eb71ec08f04f001476f5'
    assert sha(ORIGINAL) == p['source_evidence_sha256'][ORIGINAL.relative_to(ROOT).as_posix()]
    imported = read(ROOT / 'outputs/cctv_dgp_actual_step_review_v1_return_import.json')
    import_audit = read(ROOT / 'outputs/cctv_dgp_actual_step_review_v1_independent_audit.json')
    assert imported['complete'] and import_audit['complete'] and not import_audit['full_3150_review_complete']
    failure_path = RETURN / 'outputs/failure.json'
    assert sha(failure_path) == imported['files_sha256']['outputs/failure.json']
    failure = read(failure_path)
    assert failure['error'] == 'Return-storage limit; retain partial review'
    assert failure['restoration'] == {'attempted': True, 'decoder_restored': True}
    expected = [(probe['update'], proposal) for probe in p['probes'] for proposal in PROPOSALS]
    complete = [(r['update'], r['proposal']) for r in failure['completed_conditions']]
    assert complete == expected[:29] and expected[29:] == [(45, 'cone')]
    assert failure['progress']['completed_conditions'] == 29 and failure['progress']['forward_slots'] == 3110
    assert failure['progress']['optimizer_updates'] == failure['progress']['gradient_queries'] == failure['progress']['backward_calls'] == 0
    assert not OUT.exists() and not GENERATED.exists(), 'Preserve earlier preparations and failures'
    OUT.mkdir()

    source = ORIGINAL.read_text(encoding='utf-8')
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'CPU_replay')
    replay_original = ast.get_source_segment(source, node)
    begin = source.index('    sys.path.insert(0,str(BUNDLE))\n', source.index('def main():'))
    end = source.index('    replay=CPU_replay(p,start)', begin)
    body_original = source[begin:end]
    body_changes = []
    body = replace_once(body_original,
        '        for index,proposal in enumerate(PROPOSALS):\n',
        '        for index,proposal in enumerate(PROPOSALS):\n            if (probe["update"],proposal) not in completed:continue\n', body_changes)
    body = replace_once(body, '    assert checked==3150\n', '    assert checked==3045\n', body_changes)
    body = replace_once(body, "results['conditions']", "failure['completed_conditions']", body_changes)
    body = replace_once(body, "            if proposal=='zero':zero_condition=receipt\n",
        "            if proposal=='zero':zero_condition=receipt\n            print({'partial_condition_checked':len(condition_list),'of':29,'slots':checked},flush=True)\n", body_changes)
    replay_changes = []
    replay = replace_once(replay_original, 'def CPU_replay(p,start):', 'def CPU_partial_replay(p,start,completed):', replay_changes)
    replay = replace_once(replay, '            for index,proposal in enumerate(PROPOSALS):\n',
        '            for index,proposal in enumerate(PROPOSALS):\n                if (probe["update"],proposal) not in completed:continue\n', replay_changes)
    replay = replace_once(replay, "                    result['cases']+=1\n",
        "                    result['cases']+=1\n                print({'CPU_partial_replay_cases':result['cases'],'of':145,'update':probe['update'],'proposal':proposal},flush=True)\n", replay_changes)
    replay = replace_once(replay, "result['cases']==150", "result['cases']==145", replay_changes)
    for original, altered, changes in [(body_original, body, body_changes), (replay_original, replay, replay_changes)]:
        inverse = altered
        for change in reversed(changes):
            assert inverse.count(change['after']) == 1
            inverse = inverse.replace(change['after'], change['before'], 1)
        assert inverse == original
    generated = ('"""Separate partial-scope checks; original prospective checks remain unchanged."""\n'
        'from audit_cctv_dgp_actual_step_review_v1_return import *\n\n'
        'def scientific_partial(p,failure,start,completed):\n' + body +
        '    return term_errors,condition_list,checked\n\n' + replay + '\n')
    ast.parse(generated, feature_version=(3, 10))
    GENERATED.write_text(generated, encoding='utf-8', newline='\n')
    scope = {
        'complete': True, 'protocol_sha256': sha(BUNDLE / 'protocol.json'),
        'selection': 'All fully completed metric receipts in original scheduled order; no selection by quality or gain',
        'conditions': [{'update': u, 'proposal': proposal} for u, proposal in complete],
        'conditions_count': 29, 'raw_PNG_mean_only_slots': 3045, 'CPU_current_batch_replays': 145,
        'excluded_unfinished_condition': {'update': 45, 'proposal': 'cone'},
        'partial_unfinished_files_preserved': True, 'full_3150_review_complete': False,
        'original_prospective_tolerances': p['prospective_audit'],
        'original_categorical_decisions_and_capacity_gates_unchanged': True,
        'independent_audit_cap_seconds': 2400, 'original_storage_failure_retained': True,
        'source_changes': {'scientific_block': body_changes, 'CPU_replay': replay_changes},
        'inverse_source_changes_match_original': True,
        'source_bindings': {path.relative_to(ROOT).as_posix(): sha(path) for path in [
            ORIGINAL, GENERATED, Path(__file__), BUNDLE / 'protocol.json', failure_path,
            ROOT / 'outputs/cctv_dgp_actual_step_review_v1_return_import.json',
            ROOT / 'outputs/cctv_dgp_actual_step_review_v1_independent_audit.json']},
        'local_preparation_neural_calls': 0, 'optimizer_updates': 0, 'gradient_queries': 0,
        'VM_calls': 0, 'new_trained_checkpoint': False, 'app_promotion': False, 'goal_complete': False,
        'seconds': time.monotonic() - start,
    }
    write(OUT / 'prospective_partial_scope.json', scope)
    print({'complete': True, 'conditions': 29, 'slots': 3045, 'CPU_replays_planned': 145, 'model_forwards': 0}, flush=True)


if __name__ == '__main__':
    main()
