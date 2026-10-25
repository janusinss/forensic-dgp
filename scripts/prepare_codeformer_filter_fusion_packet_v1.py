"""Prepare a self-contained inert manual151-update VM stage; never seal/launch.

All operational pixels are content addressed; exact original metadata is copied
separately. No checkpoint is deserialized and no neural framework is imported.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_filter_fusion_training_v1 import transport as t
from research.codeformer_filter_fusion_training_v1 import worker as w

PREP = ROOT / 'outputs/codeformer_filter_fusion_training_v1_preparation'
STAGE = ROOT / 'outputs' / t.VM_NAME
RUNTIME = [
    'research/codeformer_filter_fusion_training_v1/' + n + '.py'
    for n in ('data', 'data_v2', 'model', 'metrics', 'transport', 'worker')]
RUNTIME += ['scripts/supervise_codeformer_filter_fusion_training_v1_vm.py',
            'scripts/prepare_codeformer_filter_fusion_packet_v1.py']
ARCH = ['third_party/__init__.py', 'third_party/codeformer/__init__.py',
        'third_party/codeformer/codeformer_arch.py', 'third_party/codeformer/vqgan_arch.py',
        'third_party/codeformer/registry.py', 'third_party/codeformer/LICENSE', 'third_party/codeformer/NOTICE.md']
WEIGHTS = {
    'weights/codeformer_original.pth': 'outputs/codeformer_restoration_pretrained_v1/codeformer.pth',
    'weights/codeformer_r2.pth': 'outputs/codeformer_role_preserving_training_v1_r2_model/codeformer_role_preserving_r2.pth',
    'weights/codeformer_parent.pth': 'outputs/codeformer_role_preserving_training_v1_r2_return/outputs/role_preserving_training_v1_r2/snapshot_0000/checkpoint.pt'}
ORIGINALS = {
    'data_bundle_v2.json': 'provenance/original_training_bundle_v2.json',
    'evaluation_bundle_v1.json': 'provenance/original_evaluation_bundle_v1.json',
    'data_schedule_v2.json': 'data_schedule_v2.json',
    'data_admission_v2.json': 'data_admission_v2.json',
    'root_dark_linear_input_review_v2.json': 'root_dark_linear_input_review_v2.json',
    'data_source_hold_v2.json': 'provenance/data_source_hold_v2.json',
    'data_finalization_v2.json': 'provenance/data_finalization_v2.json',
    'data_guard_probes_v2.json': 'provenance/data_guard_probes_v2.json',
    'data_sampler_ledger_v2.json': 'provenance/data_sampler_ledger_v2.json',
    'data_dark_linear_contract_v1.json': 'provenance/data_dark_linear_contract_v1.json',
    'data_independent_scope_pass_v2.json': 'provenance/data_independent_scope_pass_v2.json'}


def file_pin(path, *, original_weight=False):
    path = t.no_links(Path(path).absolute())
    before = path.lstat()
    exception = (original_weight and path == (ROOT / WEIGHTS['weights/codeformer_original.pth']).absolute()
                 and before.st_nlink == 3)
    t.require(stat.S_ISREG(before.st_mode) and (before.st_nlink == 1 or exception),
              'Exact regular source/single-link rule required')
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * t.MiB), b''):
            digest.update(block)
    after = path.stat()
    t.require((before.st_ino, before.st_size, before.st_mtime_ns) ==
              (after.st_ino, after.st_size, after.st_mtime_ns), 'Source changed during hashing')
    result = {'sha256': digest.hexdigest(), 'bytes': before.st_size}
    if exception:
        t.require(result == t.WEIGHT_ASSETS['weights/codeformer_original.pth'],
                  'Historical hardlink exception only exact original restoration asset')
    return result


def read(path):
    return t.parse_json(t.no_links(path).read_bytes())


def build_plan(freeze_name, freeze_sha):
    freeze_path = ROOT / t.canonical(freeze_name)
    t.require(file_pin(freeze_path)['sha256'] == freeze_sha, 'Exact reviewed sourcefreeze required')
    freeze = read(freeze_path)
    t.require(freeze.get('complete') is True and type(freeze.get('source_files')) is dict
              and freeze.get('actual_neural_calls') == 0, 'Completed inert sourcefreeze receipt required')
    pins, source_map, generated = {}, {}, {}
    def add(source, name, expected=None, weight=False):
        t.canonical(name)
        source = t.no_links(Path(source).absolute())
        t.require(source.is_relative_to(ROOT), 'Source escaped workspace')
        p = file_pin(source, original_weight=weight)
        t.require(expected is None or p == expected, 'Source pin differs: ' + name)
        if name in pins:
            t.require(pins[name] == p, 'Conflicting destination pin')
            return {'path': name, **p}
        pins[name] = p
        source_map[name] = {'source': source.relative_to(ROOT).as_posix(), 'pin': p,
                            'readonly_historical_original_weight_hardlink_exception': weight and source.stat().st_nlink == 3}
        return {'path': name, **p}
    def new(name, value):
        payload = value if isinstance(value, bytes) else t.json_bytes(value)
        t.require(name not in pins and (not name.endswith('.json') or len(payload) <= t.LIMITS['one_json_bytes']),
                  'Generated namespace/JSON cap differs')
        pins[name] = {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
        generated[name] = payload
        source_map[name] = {'generated_by': 'scripts/prepare_codeformer_filter_fusion_packet_v1.py', 'pin': pins[name]}
    for name in RUNTIME:
        t.require(name in freeze['source_files'], 'Unreviewed current runtime source: ' + name)
        add(ROOT / name, name, freeze['source_files'][name])
    for name in ARCH + ['research/__init__.py']:
        add(ROOT / name, name)
    new('research/codeformer_filter_fusion_training_v1/__init__.py',
        b'"""Finite manual CodeFormer fusion pilot; importing never launches work."""\n')
    for name, source in WEIGHTS.items():
        add(ROOT / source, name, t.WEIGHT_ASSETS[name], name == 'weights/codeformer_original.pth')
    for original, target in ORIGINALS.items():
        add(PREP / original, target)
    extras = {
        'outputs/codeformer_filter_train_v2_preparation/candidate_holds_v1.json': 'provenance/candidate_holds_v1.json',
        'outputs/codeformer_filter_train_v2_data/README.md': 'provenance/filter_data_README.md',
        'outputs/codeformer_filter_train_v2_preparation/cohort_selection_v1.json': 'provenance/cohort_selection_v1.json',
        'outputs/codeformer_filter_train_v2_preparation/input_only_review_criteria_v1.json': 'provenance/input_only_review_criteria_v1.json',
        'outputs/cctv_codeformer_adaptation_v2_vm/references.json': 'provenance/reference_manifest.json',
        'outputs/cctv_codeformer_adaptation_v2_vm/provenance/upstream_CodeFormer_LICENSE': 'provenance/upstream_CodeFormer_LICENSE',
        'outputs/cctv_codeformer_adaptation_v2_vm/provenance/upstream_CodeFormer_README.md': 'provenance/upstream_CodeFormer_README.md',
        'outputs/codeformer_restoration_pretrained_v1/provenance.json': 'provenance/original_restoration_weights_provenance.json',
        'CCTV_CODEFORMER_FILTER_FUSION_TRAINING_V1_PREPARATION.md': 'provenance/preparation_report_at_stage.md',
        'research/codeformer_filter_fusion_training_v1/DESIGN_PILOT_V1.md': 'provenance/DESIGN_PILOT_V1.md',
        'research/codeformer_filter_fusion_training_v1/DESIGN_DRAFT.md': 'provenance/DESIGN_DRAFT_at_stage.md'}
    for name in ('PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
                 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md', 'AGENTS.md'):
        extras[name] = 'provenance/instructions/' + name
    for original, target in extras.items():
        add(ROOT / original, target, freeze['source_files'].get(original))
    add(freeze_path, 'provenance/packet_source_freeze_v1.json')
    bundle = read(PREP / 'data_bundle_v2.json')
    evaluation = read(PREP / 'evaluation_bundle_v1.json')
    def operational(row):
        answer = copy.deepcopy(row)
        for key in ('input', 'target', 'target256', 'support', 'corruption_footprint'):
            if row.get(key) is not None:
                p = row[key]
                name = 'data/pixels/' + p['sha256'] + Path(p['path']).suffix.lower()
                answer[key] = add(ROOT / p['path'], name, {k: p[k] for k in ('sha256', 'bytes')})
        answer['packet_original_metadata'] = {'file': 'provenance/original_training_bundle_v2.json'
            if 'reference_id' in row and row.get('split') == 'TRAIN' and row['id'] in bundle['case_by_id']
            else 'provenance/original_evaluation_bundle_v1.json', 'id': row['id']}
        return answer
    cases = {identifier: operational(row) for identifier, row in bundle['case_by_id'].items()}
    evaluation_rows = [operational(row) for row in evaluation['cases']]
    refused = [operational(row) for row in evaluation['refused_controls']]
    new('data_bundle.json', {**copy.deepcopy(bundle), 'case_by_id': cases,
         'operational_paths_remapped_to_hash_identical_packet_copies': True,
         'original_full_rows_retained_in': 'provenance/original_training_bundle_v2.json'})
    new('evaluation_bundle.json', {**copy.deepcopy(evaluation), 'cases': evaluation_rows,
         'refused_controls': refused, 'operational_paths_remapped_to_hash_identical_packet_copies': True,
         'original_full_rows_retained_in': 'provenance/original_evaluation_bundle_v1.json'})
    guard = read(PREP / 'data_guard_probes_v2.json')['guards']
    schedule = read(PREP / 'data_schedule_v2.json')
    first_anchor = schedule['anchor_source_order'][0]
    witness = {'clear': first_anchor + '__clear',
               'dark': first_anchor + '__linear_srgb_exposure0p35'}
    for label, profile in [('rows', 'repeated_rows_4_blend0p75'),
                           ('local_mosaic', 'local_mosaic_8'), ('local_blur', 'local_gaussian_3p0')]:
        witness[label] = min(identifier for identifier, row in cases.items() if row['profile'] == profile)
    protocol = {'format': 'codeformer-filter-fusion-training-finite-manual-v1',
        'vm_name': t.VM_NAME, 'outputs': t.OUTPUTS, 'weight_assets': copy.deepcopy(t.WEIGHT_ASSETS),
        'dependencies': {}, 'evaluation_path': 'evaluation_bundle.json',
        'data': {'bundle_path': 'data_bundle.json', 'schedule_path': 'data_schedule_v2.json',
                 'admission_path': 'data_admission_v2.json', 'dark_review_path': 'root_dark_linear_input_review_v2.json',
                 'holds_sha256': bundle['candidate_holds']['sha256'],
                 'dark_review_sha256': bundle['dark_input_review']['sha256']},
        'guard_cases': guard, 'witness_cases': witness, 'call_caps': w.call_caps(),
        'gradient_queries': list(w.GRADIENT_QUERY_NAMES), 'snapshots': [0, 50, 151],
        'guard_snapshots': [0, 20], 'automatic_resume': False, 'automatic_training_launch': False,
        'full_scope_478': {'historical_DEV': 406, 'mixed_filter_TRAIN': 36, 'mixed_filter_DEV': 36,
                          'native_unpaired': 48, 'refused_no_forward': 24},
        'prospective_resource_upper_ledger': t.resource_ledger(),
        'new_model_trained_or_qualified': False, 'stage_only_not_sealed_or_launch_ready': True}
    worker_names = w.planned_output_names(protocol, evaluation_rows, cases)
    supervisor = ['outputs/supervision/' + n for n in
                  ('preflight.log', 'worker.log', 'run_receipt.json', 'export_invocation.json', 'export.log', 'finished.json')]
    protocol['allowed_output_names'] = sorted(set(worker_names + supervisor))
    excluded = {t.OUTPUTS + '/' + n for n in ('failure.json', 'stopped_checkpoint.pt', 'export_manifest.json')}
    protocol['required_success_output_names'] = sorted((set(worker_names) - excluded) |
        set(supervisor[:4]))
    new('provenance/PACKET_NOTICE.md',
        b'# Inert manual filter-fusion research packet\n\n'
        b'Three separately fingerprinted restoration states: untouched CodeFormer, retained parent, and our R2 adaptation. '
        b'The historical architecture NOTICE is preserved verbatim and contains its earlier inpainting-context note; '
        b'this packet selects only the exact restoration state hashes in protocol.json. '
        b'S-Lab noncommercial terms, original photo attributions/licenses and FFHQ overlap limitations remain retained. '
        b'This packet grants no new distribution rights, native CCTV identity claim or automatic launch.\n')
    protocol['files_sha256'] = {n: p['sha256'] for n, p in sorted(pins.items())}
    protocol['files_bytes'] = {n: p['bytes'] for n, p in sorted(pins.items())}
    w.validate_plan(protocol)
    payload = t.json_bytes(protocol)
    nonweights = {n: p for n, p in pins.items() if n not in t.WEIGHT_ASSETS}
    t.require(len(payload) <= 4 * t.MiB and len(nonweights) + 1 <= 2048 and
              sum(v['bytes'] for v in nonweights.values()) + len(payload) <= 512 * t.MiB,
              'Protocol-inclusive nonweight stage cap exceeded')
    return {'protocol': protocol, 'protocol_bytes': payload, 'pins': pins, 'source_map': source_map,
            'generated': generated, 'sourcefreeze': {'path': freeze_name, 'sha256': freeze_sha},
            'nonweight_files_including_protocol': len(nonweights) + 1,
            'nonweight_bytes_including_protocol': sum(p['bytes'] for p in nonweights.values()) + len(payload)}


def prepare(freeze_name, freeze_sha, stage=False):
    plan = build_plan(freeze_name, freeze_sha)
    summary = {'complete': True, 'inert_plan': True, 'protocol_sha256': hashlib.sha256(plan['protocol_bytes']).hexdigest(),
        'protocol_bytes': len(plan['protocol_bytes']), 'static_files_including_protocol': len(plan['pins']) + 1,
        'static_bytes_including_protocol': sum(p['bytes'] for p in plan['pins'].values()) + len(plan['protocol_bytes']),
        'nonweight_files_including_protocol': plan['nonweight_files_including_protocol'],
        'nonweight_bytes_including_protocol': plan['nonweight_bytes_including_protocol'],
        'allowed_output_names': len(plan['protocol']['allowed_output_names']),
        'required_success_output_names': len(plan['protocol']['required_success_output_names']),
        'sourcefreeze': plan['sourcefreeze'], 'neural_gradient_optimizer_VM_calls': 0,
        'new_model_or_sealed_archive_created': False}
    if not stage:
        print(json.dumps(summary)); return
    t.require(not STAGE.exists() and shutil.disk_usage(ROOT).free >=
              summary['static_bytes_including_protocol'] + 4 * t.GiB,
              'Fresh stage namespace and4GiB post-copy local reserve required')
    STAGE.mkdir()
    for name, expected in sorted(plan['pins'].items()):
        target = STAGE / t.canonical(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as out:
            if name in plan['generated']:
                out.write(plan['generated'][name])
            else:
                source = ROOT / plan['source_map'][name]['source']
                before = source.stat()
                digest = hashlib.sha256()
                with source.open('rb') as inp:
                    for block in iter(lambda: inp.read(4 * t.MiB), b''):
                        out.write(block); digest.update(block)
                after = source.stat()
                t.require((before.st_ino, before.st_size, before.st_mtime_ns) ==
                          (after.st_ino, after.st_size, after.st_mtime_ns) and
                          digest.hexdigest() == expected['sha256'], 'Source changed while staged')
        t.require(t.file_pin(target) == expected, 'Staged copied bytes differ')
    with (STAGE / 'protocol.json').open('xb') as handle:
        handle.write(plan['protocol_bytes'])
    checked = t.verify_transfer(STAGE, summary['protocol_sha256'])
    w.validate_plan(checked['protocol']); w._local_metadata(STAGE, checked['protocol'])
    summary.update(stage=str(STAGE), stage_created=True, static_transfer_verified=True,
                   all411_admission_and478evaluation_metadata_joins_verified=True,
                   actual_stage_pin_map={**plan['pins'], 'protocol.json': t.file_pin(STAGE / 'protocol.json')},
                   local_source_map=plan['source_map'])
    t.write_new_json(PREP / 'packet_stage_v1.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('actual_stage_pin_map', 'local_source_map')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-freeze', required=True)
    parser.add_argument('--source-freeze-sha256', required=True)
    parser.add_argument('--stage', action='store_true')
    args = parser.parse_args()
    prepare(args.source_freeze, args.source_freeze_sha256, args.stage)


if __name__ == '__main__':
    main()
