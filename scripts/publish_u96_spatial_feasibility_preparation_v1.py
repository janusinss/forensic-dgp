"""Publish reviewed inert progress; preserve each prior document byte exactly."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'outputs/codeformer_u96_next_spatial_design_v1'
PUBLICATION = EVIDENCE / 'publication_v1'
REPORT = ROOT / 'CCTV_CODEFORMER_U96_SPATIAL_FEASIBILITY_V1_PREPARATION.md'
DOCUMENTS = ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
             'PROJECT_HANDOFF.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pin(path):
    data = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(data), 'sha256': sha(data)}


def write_new(path, data):
    with path.open('xb') as stream:
        stream.write(data)


def main():
    assert 'torch' not in sys.modules and not PUBLICATION.exists()
    tests_path = EVIDENCE / 'root_inert_source_tests_v1.json'
    tests = json.loads(tests_path.read_bytes())
    assert tests['complete'] and tests['all_tests_pass'] and tests['tests'] == 53
    assert tests['neural_gradient_optimizer_training_VM_calls'] == 0
    sources = []
    for row in tests['sources']:
        assert pin(ROOT / row['path']) == row, 'Tested source changed: ' + row['path']
        sources.append(row)
    reviews = []
    for name in ('inert_source_review_v1.json', 'direction_independent_review_v1.json',
                 'contract_objective_source_readback_v1.json', 'independent_scope_readback_v1.json'):
        path = EVIDENCE / name
        value = json.loads(path.read_bytes())
        if name == 'contract_objective_source_readback_v1.json':
            assert value['format'] == 'independent-spatial-feasibility-contract-objective-source-readback-v1'
            assert value['status'] == 'REFRESHED_CURRENT_INERT_CONTRACT_CONFIRMED_NOT_RUNTIME_OR_PACKET'
            assert value['independent_readback_check_count'] == 20
        else:
            assert value['complete'] is True
        for row in value.get('source_pins', value.get('sources', value.get('pins', []))):
            assert pin(ROOT / row['path']) == row, 'Reviewed source changed: ' + row['path']
        reviews.append(pin(path))
    scope_path = EVIDENCE / 'prospective_scope_readback_v1.json'
    scope = json.loads(scope_path.read_bytes())
    assert scope['complete'] and scope['checks'] == 534 and len(scope['data_bindings']) == 250
    assert len(scope['decoded_pixel_bindings']) == 136 and len(scope['operation_plan']) == 218
    assert scope['Torch_model_neural_gradient_optimizer_training_VM_calls'] == 0
    independent = json.loads((EVIDENCE / 'independent_scope_readback_v1.json').read_bytes())
    assert independent['independent_check_count'] == 801 and independent['root_receipt'] == pin(scope_path)
    # Relative Markdown links must resolve. Web citations are retained separately.
    links = []
    for target in re.findall(r'\]\(([^)]+)\)', REPORT.read_text(encoding='utf-8')):
        if target.startswith(('http://', 'https://')):
            continue
        path = ROOT / target.split('#', 1)[0]
        assert path.is_file(), 'Broken preparation link: ' + target
        links.append(pin(path))
    prefix = (
        '# 23 October 2026: spatial decoder and low-light feasibility source reviewed\n\n'
        '[Current preparation](CCTV_CODEFORMER_U96_SPATIAL_FEASIBILITY_V1_PREPARATION.md): '
        'the next distinct path starts from complete exact incumbent U96 and opens only its existing final '
        'ResBlock22 plus RGBConv24, ten tensors/75,843 scalars with529 frozen. The completed two-tensor '
        'readout run remains a quality failure: do not rerun its guide, append epochs or promote its checkpoint. '
        'The new path tests richer spatial response together with low-light selectivity; no useful improvement is yet measured.\n\n'
        'Four inert modules now implement the guarded factory/full539-state capture, separate photographic RGB/adjacent '
        'components, bounded common-descent arithmetic and exact finite prospective contract. Root53pure/NumPy tests and '
        'nine Python3.10 syntax checks pass. Independent model/objective/contract and direction reviews found no unresolved '
        'blocker in their scoped source; direction review adds64 independent analytic/packing/FP32 checks. Fresh scope '
        'readback verified534checks over250 bindings/136 decoded RGB256 inputs-targets; an independent audit verified801checks. '
        'These are source/data proofs, not actual CUDA/gradient/quality evidence. Exact source/test/review pins are retained.\n\n'
        'Proposed manual-L4 diagnostic is capped at218 full forwards/32 component queries/two independent FP32 array '
        'applications from the SAME initializer, maximum coordinate displacements0.00001/0.00005, zero optimizer objects/steps. '
        'Every role/component is mandatory; uncertified or rounded-away descent stops before application. ProposalA failure '
        'blocksB and BOTH must pass; no favorable-arm selection. Retain all32 adaptive TRAIN monitors and24 legacy guards '
        'with their inherited roles/null dark targets, genuine clean optimization targets and all four full1% endpoints. '
        'Return actual complete539 values at baseline and each applied trial so frozen529 can be independently rehashed.\n\n'
        'STATUS: INERT_SOURCE_REVIEWED_UNSEALED_UNRUN. The finite worker, actual gate implementation, durable partial/unknown '
        'ledger, full-state serializer/payload/resource supervision, complete runner review, sealed execution archive and '
        'exact upload/install/tmux/download guide are NOT READY. No model/Torch/neural/gradient/optimizer/training/VM call '
        'or app/default/checkpoint promotion occurred in this preparation. U96 remains incumbent. Next complete those '
        'execution components before the user manually launches anything. Do not substitute a historical command. '
        'Reviewed optional Protect/Apply/bypass remains separate; automatic matching deferred, native/wild unpaired, '
        'reservedFINAL unopened/execution-unfrozen with reviewers pending, later completion deferred. Full goal ACTIVE/incomplete. '
        'Exact prior document bytes are retained below.\n\n---\n\n'
    ).encode('utf-8')
    PUBLICATION.mkdir()
    rows = []
    for name in DOCUMENTS:
        path = ROOT / name
        before = path.read_bytes()
        saved = PUBLICATION / (name + '.before')
        write_new(saved, before)
        assert saved.read_bytes() == before
        temporary = path.with_name(path.name + '.spatial_feasibility_v1.new')
        write_new(temporary, prefix + before)
        assert path.read_bytes() == before, 'Concurrent document change; preserve it'
        temporary.replace(path)
        after = path.read_bytes()
        assert after == prefix + before and after[len(prefix):] == saved.read_bytes()
        rows.append({'after': pin(path), 'previous_snapshot': pin(saved),
                     'prefix_bytes': len(prefix), 'previous_suffix_exact': True})
    receipt = {'complete': True, 'recorded_UTC': datetime.now(timezone.utc).isoformat(),
               'status': 'INERT_SOURCE_REVIEWED_UNSEALED_UNRUN',
               'previous_goal_turn_classification': 'progress: implemented distinct inert decoder/components/contract and validated retained data before interruption',
               'this_goal_turn_classification': 'progress: completed and independently reviewed finite direction arithmetic,53tests,data/source binding and authoritative documentation',
               'report': pin(REPORT), 'documents': rows, 'tests': pin(tests_path),
               'scope': pin(scope_path), 'independent_reviews': reviews,
               'tested_sources': sources, 'resolved_local_links': len(links),
               'builder': pin(Path(__file__).resolve()),
               'model_training_VM_calls': 0, 'app_changes': False,
               'sealed_packet_or_guide_ready': False, 'goal_complete': False}
    target = PUBLICATION / 'root_publication_readback_v1.json'
    write_new(target, (json.dumps(receipt, indent=2, allow_nan=False) + '\n').encode('utf-8'))
    assert 'torch' not in sys.modules
    print(json.dumps({'complete': True, 'documents': len(rows), 'resolved_local_links': len(links),
                      'receipt': pin(target), 'goal_complete': False}))


if __name__ == '__main__':
    main()
