"""Independent sealed finite-step archive audit; no builder/worker/VM run.

Only source, JSON, filesystem and compressed framing are examined. All local
dependency substitutions are explicit hash provenance; VM paths remain exact.
The guide is deliberately not approved by this archive-only program.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/codeformer_u64_png_checked_step_v1_preparation'
STAGE_NAME = 'cctv_codeformer_u64_png_checked_step_v1_vm'
STAGE = ROOT / 'outputs' / STAGE_NAME
PARENT = ROOT / 'outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm'
PACKAGE = 'research/codeformer_u64_png_checked_step_v1'
PREFIX = 'cctv-codeformer-u64-png-checked-step-v1'
MEASUREMENT_NAME = 'cctv_codeformer_u64_png_surrogate_measurement_v1_vm'
MEASUREMENT = ROOT / 'outputs' / MEASUREMENT_NAME
MEASUREMENT_OUTPUT = 'outputs/u64_png_surrogate_measurement_v1'
MEASUREMENT_AUDIT = ROOT / 'outputs/codeformer_u64_png_surrogate_measurement_v1_return_audit'
MEASUREMENT_EXTRACTION = MEASUREMENT_AUDIT / 'extracted'
OWN_SOURCES = [PACKAGE + '/' + name for name in
    ('__init__.py', 'contract.py', 'selection.py', 'quality.py', 'runtime.py', 'worker.py', 'run_png_checked.sh')]
OWN_SOURCES += ['research/__init__.py',
    'research/codeformer_u64_role_gradients_v1/__init__.py',
    'research/codeformer_u64_role_gradients_v1/contract.py',
    'research/codeformer_u64_gate_aligned_v1/__init__.py',
    'research/codeformer_u64_gate_aligned_v1/solver.py',
    'research/codeformer_u64_gate_aligned_pilot_v1/__init__.py',
    'research/codeformer_u64_gate_aligned_pilot_v1/quality.py',
    'scripts/run_u64_png_checked_step_v1.py', 'scripts/build_u64_png_checked_step_v1.py']
REPORTS = ['CCTV_CODEFORMER_U64_PNG_SURROGATE_MEASUREMENT_V1_RETURN_REVIEW.md',
    'CCTV_CODEFORMER_U64_PNG_CHECKED_STEP_V1_DESIGN.md']
ADMISSIONS = ['outputs/codeformer_u64_png_surrogate_measurement_v1_return_transport/receipt_v1.json',
    'outputs/codeformer_u64_png_surrogate_measurement_v1_return_gradients/receipt_v1.json',
    'outputs/codeformer_u64_png_surrogate_measurement_v1_return_pixels/receipt_v1.json']
PARENT_PIN = {'bytes': 279879, 'sha256': '4db752a04222210a9fbccf176cf422c5fadeef443d39a15ae7b62086e9c41419'}
STANDARD_PIN = {'bytes': 197079, 'sha256': 'a9b7899cb23ada04779f350dfd37f11818533bf9bf462404e342b8d7e575294e'}
MEASUREMENT_PIN = {'path': 'protocol.json', 'bytes': 188285,
    'sha256': '0a61c2d6433399e9e748f3782e4d87bf9da3089848b2502a761fcd029690797c'}
FULL_SHA = '47e0707db446d3e39e9c101e8eedd631e49b45ba4f03a5f00aa2ad02e0281a8a'
HEAD_SHA = '6629d8414321b0d2ba3a9cc476e415d31911fd4bd1fa121a1a8db9097ce235be'
FROZEN_SHA = 'f096108dc16dff56687a61a57d7ee367f8dd50accc385d2d72be8b53d4510cdf'
CAPS = (2e-5, 1.5e-5, 1e-5, 5e-6)
HELPERS = {
 'research/codeformer_u64_role_gradients_v1/__init__.py': (82, 'b13cc93cf4d889fde3d255356e1c9e7e483aa845de75e2ea4a94a2c562554baf'),
 'research/codeformer_u64_role_gradients_v1/contract.py': (11189, '90a11da6be2e531648e4c5051471946469e287d2bc151177006554932395074e'),
 'research/codeformer_u64_gate_aligned_v1/__init__.py': (87, '9eb734f64e4e7871f2357f8af11cabaf3a4b4efea1bf4d148bfb6455eaafadca'),
 'research/codeformer_u64_gate_aligned_v1/solver.py': (7511, '0e68b9ab6d7fbd019dbdee6119530b718fa29f38e0ea80aefc1891ecccea91ba'),
 'research/codeformer_u64_gate_aligned_pilot_v1/__init__.py': (79, '1e063d3fd033063403eefefb7a27327e9716ed77c4731f9c921d1d95ca77bed2'),
 'research/codeformer_u64_gate_aligned_pilot_v1/quality.py': (5124, '2b4b15e99a28d0f45667a15a48983db61c032aa324ed77fbd28c0afd778c739e')}


class BlockScientific(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'PIL', 'scipy', 'torch', 'torchvision', 'tensorflow', 'jax', 'basicsr'}:
            raise ImportError('Scientific import blocked in packet audit: ' + fullname)
        return None


sys.meta_path.insert(0, BlockScientific())
SPEC = importlib.util.spec_from_file_location('packet_framing_parser',
    ROOT / 'scripts/audit_u64_role_gradients_return_transport_v1.py')
F = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(F)
need, strict_json = F.need, F.strict_json


def pin(path, base=ROOT):
    path, base = Path(path).absolute(), Path(base).absolute()
    before = path.lstat()
    need(path.resolve().is_relative_to(base.resolve()) and stat.S_ISREG(before.st_mode)
         and before.st_nlink == 1 and not path.is_symlink()
         and not getattr(before, 'st_file_attributes', 0) & 1024, 'Ordinary confined single-link file required')
    for parent in path.parents:
        value = parent.lstat()
        need(stat.S_ISDIR(value.st_mode) and not parent.is_symlink()
             and not getattr(value, 'st_file_attributes', 0) & 1024, 'Ordinary ancestry required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    after = path.stat()
    need((before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_ino, after.st_size, after.st_mtime_ns), 'Source changed during hash')
    return {'path': path.relative_to(base).as_posix(), 'bytes': after.st_size, 'sha256': digest.hexdigest()}


def payload(value):
    return {key: value[key] for key in ('bytes', 'sha256')}


def staged_help():
    code = """import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,n,path=None,target=None):
  if n.split('.')[0] in {'numpy','PIL','scipy','torch','torchvision','tensorflow','jax','basicsr'}:raise ImportError('blocked scientific import')
  return None
sys.meta_path.insert(0,Block())
script=sys.argv[1];sys.argv=[script,'--help'];runpy.run_path(script,run_name='__main__')
"""
    result = subprocess.run([sys.executable, '-B', '-c', code,
        str(STAGE / 'scripts/run_u64_png_checked_step_v1.py')], cwd=STAGE,
        capture_output=True, text=True, timeout=10,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    need(result.returncode == 0 and not result.stderr, 'Staged inert help failed')
    need(all('--' + name in result.stdout for name in
        ('root', 'protocol-sha', 'verify-transfer', 'status', 'supervise', 'run', 'export')), 'Staged CLI options differ')
    need(not any(STAGE.rglob('__pycache__')), 'Staged help wrote bytecode')
    return {'exit_code': result.returncode, 'stdout': result.stdout,
        'stderr': result.stderr, 'scientific_imports_blocked': True}


def independent_plan(selection):
    rows = []
    def forward(source, phase, category, trial_index=None):
        rows.append({'slot': len(rows) + 1, 'kind': 'forward', 'id': source['id'],
            'reference_id': source['reference_id'], 'role': source['role'],
            'metric_region': 'central128' if source['role'] == 'local_gaussian_1p2' else 'whole',
            'phase': phase, 'category': category, 'trial_index': trial_index, 'gradient_queries': []})
    def application(action, index):
        rows.append({'slot': len(rows) + 1, 'kind': 'parameter_application',
            'id': action + '_' + ('selected' if index is None else 'trial_' + str(index)),
            'action': action, 'trial_index': index, 'phase': action, 'gradient_queries': [],
            'source_state': 'candidate_trial_' + str(index) if action == 'trial_restore' else 'original_u64',
            'destination_state': 'original_u64' if action == 'trial_restore' else
                ('candidate_trial_' + str(index) if action == 'trial_apply' else 'selected_eligible_candidate'),
            'destination_head_canonical_sha256': HEAD_SHA if action == 'trial_restore' else None,
            'worker_must_record_exact_before_after_canonical': True})
    bank = selection['monitors'] + selection['guards']
    for row in bank: forward(row, 'initial_u64_quality_bank', 'initial_quality_forwards')
    for row in selection['monitors'][:4]: forward(row, 'deterministic_same_u64_repeat', 'initial_repeat_forwards')
    for row in selection['optimization']: forward(row, 'initial_optimization_bank', 'initial_optimization_forwards')
    for index in range(4):
        application('trial_apply', index)
        for row in selection['optimization']: forward(row, 'trial_optimization_bank', 'trial_optimization_forwards', index)
        application('trial_restore', index)
    application('selected_commit', None)
    for row in bank: forward(row, 'selected_quality_bank', 'selected_quality_forwards')
    return rows


def independent_measurement_names(measurement):
    names = set(measurement['files'])
    need(len(names) == 21 and len(measurement['plan']) == 72, 'Retained21static/72call namespace')
    need([row['slot'] for row in measurement['plan']] == list(range(1, 73)), 'Contiguous retained slots')
    for name in ('result.json', 'RNG_sampler_initial.json', 'RNG_sampler_closing.json',
                 'full_states/initial.json', 'full_states/closing.json', 'head_schema.json', 'operation_plan.json', 'status.json'):
        names.add(MEASUREMENT_OUTPUT + '/' + name)
    names.update(('supervisor.start.json', 'supervisor.worker.started.json', 'supervisor.worker.json'))
    for row in measurement['plan'][8:]:
        need(row['gradient_queries'] == ['raw_mae', 'raw_adjacent', 'png_surrogate_mae', 'png_surrogate_adjacent'],
             'Retained four-query bindings required')
        stem = MEASUREMENT_OUTPUT + '/calls/' + str(row['slot']).zfill(4) + '/'
        names.update(stem + name for name in ('record.json', 'gradients.json', 'gradients_raw_mae.npz', 'gradients_raw_adjacent.npz'))
    need(len(names) == 288, 'Exactly288 retained file bindings required')
    return names


def audit_archive(archive_sha, protocol_sha):
    consumed, checks = {}, []
    def checked(path):
        value = pin(path); consumed[value['path']] = value; return value
    def doc(path):
        checked(path); return strict_json(path.read_bytes())
    def check(name, ok, evidence=None):
        need(ok, name); checks.append({'name': name, 'pass': True, 'evidence': evidence})
    checked(Path(__file__))
    checked(ROOT / 'scripts/audit_u64_role_gradients_return_transport_v1.py')
    build = doc(PREP / 'build_receipt_v1.json')
    p = doc(STAGE / 'protocol.json')
    ap = ROOT / 'outputs' / (PREFIX + '-execution.tar.gz')
    cp = ap.with_name(ap.name + '.sha256')
    archive, checksum = checked(ap), checked(cp)
    protocol = pin(STAGE / 'protocol.json', STAGE)
    check('independently_supplied_sealed_archive_protocol_pins', archive == build['archive']
        and protocol == build['protocol'] and archive['sha256'] == archive_sha and protocol['sha256'] == protocol_sha,
        {'archive': archive, 'protocol': protocol})
    check('checksum_exact_digest_two_spaces_basename_LF', checksum == build['checksum']
        and cp.read_bytes() == (archive_sha + '  ' + ap.name + '\n').encode('ascii'), checksum)
    expected = {'protocol.json': protocol, **{name: {'path': name, **value} for name, value in p['files'].items()}}
    with ap.open('rb') as stream:
        parsed = F.stream_archive(stream, expected, STAGE_NAME)
    check('single_gzip_CRC_ISIZE_and_complete_USTAR_34members',
        parsed['framing']['compressed_sha256'] == archive_sha and parsed['framing']['compressed_bytes'] == archive['bytes']
        and parsed['framing']['regular_members'] == 34 and len(p['files']) == 33, parsed['framing'])
    names = set(OWN_SOURCES) | {'scripts/run_png_checked.sh'}
    names |= {'evidence/review_' + str(i) + '_' + name for i, name in enumerate(REPORTS)}
    names |= {'evidence/' + name for name in ('source_review.json', 'source_tests.json', 'parent_protocol.json',
        'standard_protocol.json', 'local_parent_readback.json', 'measurement_protocol.json', 'measurement_result.json',
        'measurement_initial_state.json', 'measurement_closing_state.json', 'retained_measurement_manifest.json',
        'local_measurement_readback.json', 'measurement_admission_0.json', 'measurement_admission_1.json', 'measurement_admission_2.json')}
    check('independently_enumerated_33static_names_only', set(p['files']) == names, sorted(names))
    actual, dirs = {}, set()
    expected_dirs = {part.as_posix() for name in expected for part in PurePosixPath(name).parents if part.as_posix() != '.'}
    for current, children, files in os.walk(STAGE, followlinks=False):
        for name in children:
            path = Path(current) / name; value = path.lstat()
            need(stat.S_ISDIR(value.st_mode) and not path.is_symlink()
                and not getattr(value, 'st_file_attributes', 0) & 1024, 'Linked stage directory refused')
            dirs.add(path.relative_to(STAGE).as_posix())
        for name in files:
            path = Path(current) / name; relative = F.confined(path.relative_to(STAGE).as_posix())
            actual[relative] = pin(path, STAGE); checked(path)
    check('actual_stage_files_and_directories_exact_archive', actual == expected and dirs == expected_dirs,
        {'files': len(actual), 'directories': sorted(dirs)})
    review, tests = doc(STAGE / 'evidence/source_review.json'), doc(STAGE / 'evidence/source_tests.json')
    check('complete_review_and_pure_tests_no_blockers', review['complete'] is True and review['source_review_pass'] is True
        and review['unresolved_blockers'] == [] and tests['complete'] is True and tests['all_tests_pass'] is True
        and tests['neural_gradient_optimizer_training_VM_calls'] == 0,
        {'review': p['source_review'], 'tests': p['source_tests'], 'test_count': tests.get('test_count', tests.get('tests_run'))})
    tables = [{row['path']: row for row in rows} for rows in (build['sources'], review['sources'], tests['sources'])]
    check('all16_source_admissions_unique_exact', all(len(table) == 16 for table in tables)
        and tables[0] == tables[1] == tables[2] and set(tables[0]) == set(OWN_SOURCES), tables[0])
    for name in OWN_SOURCES:
        check('current_staged_admitted_' + name, checked(ROOT / name) == tables[0][name] == pin(STAGE / name, STAGE), tables[0][name])
        if name.endswith('.py'): ast.parse((STAGE / name).read_text(), feature_version=(3, 10))
    for name, (size, digest) in HELPERS.items():
        check('exact_original_helper_' + name, p['files'][name] == {'bytes': size, 'sha256': digest})
    check('shell_alias_exact_reviewed_source', p['files']['scripts/run_png_checked.sh'] == p['files'][PACKAGE + '/run_png_checked.sh'])
    for index, name in enumerate(REPORTS):
        check('copied_current_report_' + name, payload(checked(ROOT / name)) == p['files']['evidence/review_' + str(index) + '_' + name])
    for key in ('source_review', 'source_tests', 'parent_protocol_copy', 'standard_protocol_copy',
                'measurement_protocol_copy', 'measurement_result_copy', 'measurement_initial_state_copy',
                'measurement_closing_state_copy', 'retained_measurement_manifest'):
        value = p[key]
        check('protocol_metadata_binding_' + key, value == {'path': value['path'], **p['files'][value['path']]})
    parentraw, standardraw = (STAGE / 'evidence/parent_protocol.json').read_bytes(), (STAGE / 'evidence/standard_protocol.json').read_bytes()
    parent, standard = strict_json(parentraw), strict_json(standardraw)
    check('exact_parent_standard_copies', p['files']['evidence/parent_protocol.json'] == PARENT_PIN
        and p['files']['evidence/standard_protocol.json'] == STANDARD_PIN
        and (PARENT / 'protocol.json').read_bytes() == parentraw
        and (PARENT / parent['parent_protocol']['path']).read_bytes() == standardraw)
    checked(PARENT / 'protocol.json'); checked(PARENT / parent['parent_protocol']['path'])
    sys.path.insert(0, str(STAGE))
    contract = importlib.import_module('research.codeformer_u64_png_checked_step_v1.contract')
    for name in ('research', 'research.codeformer_u64_png_checked_step_v1',
                 'research.codeformer_u64_png_checked_step_v1.contract', 'research.codeformer_u64_role_gradients_v1.contract'):
        need(Path(sys.modules[name].__file__).resolve().is_relative_to(STAGE), 'Wrong staged inert contract origin')
    selection, pilot = contract.select_cases(parentraw, standardraw), contract.prospective_contract()
    plan = contract.operation_plan(selection)
    check('actual_staged_stdlib_contract_recomputes_protocol', p['selection'] == selection and p['plan'] == plan
        and p['pilot'] == pilot and p['limits'] == contract.LIMITS)
    check('independently_reconstructed_445_rows_exact', plan == independent_plan(selection)
        and len(plan) == 445 and sum(row['kind'] == 'forward' for row in plan) == 436
        and sum(row['kind'] == 'parameter_application' for row in plan) == 9
        and not any(row['gradient_queries'] for row in plan), {'plan_rows': 445, 'forwards': 436, 'applications': 9, 'queries': 0})
    prefix = plan[:388]
    check('noeligible_prefix_restores_fourth_trial_before_commit', len(prefix) == 388
        and sum(row['kind'] == 'forward' for row in prefix) == 380
        and sum(row['kind'] == 'parameter_application' for row in prefix) == 8
        and prefix[-1]['action'] == 'trial_restore' and prefix[-1]['trial_index'] == 3
        and plan[388]['action'] == 'selected_commit' and contract.NO_ELIGIBLE_PREFIX['final_head_canonical_sha256'] == HEAD_SHA,
        {'pairs': 388, 'forwards': 380, 'applications': 8, 'trial_apply': 4, 'trial_restore': 4, 'selected_commit': 0})
    coordinates = 0
    shapes = contract.head_shapes()
    for shape in shapes.values():
        size = 1
        for dimension in shape: size *= dimension
        coordinates += size
    check('575_539_36_168960_original_state_bindings', len(shapes) == 36 and coordinates == 168960
        and pilot['total_tensors'] == 575 and pilot['frozen_R2_U96_tensors'] == 539
        and pilot['original_full575_canonical_sha256'] == FULL_SHA and pilot['original_head36_canonical_sha256'] == HEAD_SHA
        and pilot['frozen539_canonical_sha256'] == FROZEN_SHA)
    selector = (STAGE / PACKAGE / 'selection.py').read_text()
    tree = ast.parse(selector, feature_version=(3, 10))
    constants = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
        and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in
        ('CANDIDATE_MAXABS', 'NUMERICAL_HEADROOM', 'OLD_MAXABS_STEP', 'MAX_CUMULATIVE')}
    check('fixed_four_subcaps_point99_and_original_coordinate_limits', constants == {'CANDIDATE_MAXABS': CAPS,
        'NUMERICAL_HEADROOM': .99, 'OLD_MAXABS_STEP': 2.5e-5, 'MAX_CUMULATIVE': 1e-4}
        and tuple(pilot['candidate_maximum_absolute_coordinate_bounds']) == CAPS
        and pilot['fixed_numerical_headroom'] == .99 and 'alpha = NUMERICAL_HEADROOM * float(cap) / peak' in selector
        and 'candidate = (original.astype(np.float64) + alpha * direction).astype(np.float32)' in selector
        and 'actual_peak <= cap and actual_peak <= OLD_MAXABS_STEP and actual_peak <= MAX_CUMULATIVE' in selector
        and 'math.isfinite(value) and value < 0 for value in dots' in selector, constants)
    limits = p['limits']
    check('finite_1200_1260_20_180_300_330_and_resource_caps', [limits[key] for key in
        ('worker_seconds', 'external_seconds', 'termination_grace_seconds', 'per_phase_seconds', 'export_seconds', 'export_external_seconds')]
        == [1200, 1260, 20, 180, 300, 330] and [limits[key] for key in
        ('initial_free_disk_bytes', 'continuing_free_disk_bytes', 'maximum_RSS_bytes', 'maximum_CUDA_reserved_bytes', 'maximum_output_bytes', 'maximum_archive_bytes')]
        == [8 * 1024**3, 4 * 1024**3, 10 * 1024**3, 18 * 1024**3, 2 * 1024**3, 2 * 1024**3], limits)
    check('zero_new_gradients_and_optimizers_original_gates_binding', all(limits[key] == 0 for key in
        ('gradient_query_attempts', 'gradient_query_returns', 'backward_attempts', 'backward_returns', 'optimizer_constructions', 'optimizer_update_attempts', 'optimizer_updates'))
        and pilot['all_original_quality_gates_unchanged'] is True and pilot['no_new_gradient_queries'] is True
        and pilot['actual_trial_and_restore_applications_counted'] is True)
    proof = doc(STAGE / 'evidence/local_parent_readback.json')
    bindings = {row['parent_relative']: row['actual_local'] for row in proof['files']}
    need(len(bindings) == len(proof['files']) == len(parent['files']) == 469 and set(bindings) == set(parent['files']), 'Exact469 parent proof')
    parentrows, aliases = [], {}
    for name, value in sorted(parent['files'].items()):
        F.confined(name); source = PARENT / name
        if not source.exists():
            need(name in {'weights/codeformer_r2.pth', 'weights/codeformer_u96_adapter.pth'}, 'Unexpected parent alias')
            source = ROOT / 'outputs/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm' / name
            aliases[name] = source.relative_to(ROOT).as_posix()
        actual = checked(source)
        need(actual == bindings[name] and payload(actual) == value, 'Actual parent bytes differ')
        parentrows.append({'parent_relative': name, 'actual_local': actual})
    check('all469_parent_bindings_rehashed_only_explicit_model_aliases', proof['complete'] is True
        and len(aliases) == 2 and proof['local_only_model_aliases'] == aliases
        and proof['VM_requires_all469_at_actual_parent_stage'] is True and proof['VM_inventory_performed'] is False,
        {'rehashed': 469, 'direct': 467, 'builder_local_model_aliases': aliases, 'VM_aliases_permitted': False})
    mproof = doc(STAGE / 'evidence/local_measurement_readback.json')
    manifest = doc(STAGE / 'evidence/retained_measurement_manifest.json')
    measurement = doc(MEASUREMENT / 'protocol.json')
    inventory = doc(MEASUREMENT_AUDIT / 'inventory_v1.json')
    transport, gradients, pixels = [doc(ROOT / name) for name in ADMISSIONS]
    check('independent_retained_transport_gradient_pixel_admissions', inventory['complete'] is True
        and transport['complete'] is True and transport['transport_audit_pass'] is True and transport['unresolved_transport_blockers'] == []
        and gradients['complete'] is True and gradients['saved_state_gradient_geometry_audit_pass'] is True
        and gradients['all575_initial_closing_exact_original_u64'] is True
        and pixels['complete'] is True and pixels['saved_pixel_scalar_consistency_pass'] is True and pixels['failed_checks'] == []
        and inventory['protocol_sha256'] == transport['protocol']['sha256'] == gradients['protocol_sha256'] == pixels['protocol_sha256'] == MEASUREMENT_PIN['sha256']
        and inventory['files'] == transport['all_member_pins'] and inventory['archive']['sha256'] == transport['archive']['sha256'])
    check('exact_measurement_protocol_and_manifest', pin(MEASUREMENT / 'protocol.json', MEASUREMENT) == MEASUREMENT_PIN
        and (STAGE / 'evidence/measurement_protocol.json').read_bytes() == (MEASUREMENT / 'protocol.json').read_bytes()
        and set(manifest) == {'stage_name', 'protocol', 'files'} and manifest == p['retained_measurement']
        and manifest['stage_name'] == MEASUREMENT_NAME and manifest['protocol'] == MEASUREMENT_PIN
        and set(manifest['files']) == independent_measurement_names(measurement))
    mbindings = {row['measurement_relative']: row['actual_local'] for row in mproof['files']}
    need(len(mbindings) == len(mproof['files']) == 288 and set(mbindings) == set(manifest['files']), 'Exact288 actual dependency proof')
    mrows, maliases = [], []
    for name, value in sorted(manifest['files'].items()):
        F.confined(name); source = MEASUREMENT / name
        if not source.exists():
            need(name not in measurement['files'], 'Retained static dependency cannot alias')
            source = MEASUREMENT_EXTRACTION / name; maliases.append(name)
        actual = checked(source)
        need(actual == mbindings[name] and payload(actual) == value == payload(inventory['files'][name]), 'Actual retained dependency bytes differ')
        if name in measurement['files']: need(value == measurement['files'][name], 'Retained static bytes differ from old seal')
        mrows.append({'measurement_relative': name, 'actual_local': actual})
    check('all288_measurement_dependencies_rehashed_only_explicit_extraction_aliases', mproof['complete'] is True
        and mproof['local_runtime_extraction_aliases'] == maliases and mproof['local_aliases_permitted_on_VM'] is False
        and mproof['VM_requires_all288_at_actual_measurement_stage'] is True and mproof['VM_inventory_performed'] is False
        and mproof['admission_receipts'] == [checked(ROOT / name) for name in ADMISSIONS]
        and mproof['inventory'] == checked(MEASUREMENT_AUDIT / 'inventory_v1.json')
        and mproof['measurement_protocol'] == checked(MEASUREMENT / 'protocol.json'),
        {'rehashed': 288, 'static': 21, 'runtime': 267, 'explicit_runtime_extraction_aliases': len(maliases), 'VM_aliases_permitted': False})
    for index, name in enumerate(ADMISSIONS):
        check('copied_admission_' + str(index), p['files']['evidence/measurement_admission_' + str(index) + '.json'] == payload(checked(ROOT / name)))
    for key, name in (('measurement_result_copy', 'result.json'), ('measurement_initial_state_copy', 'full_states/initial.json'),
                      ('measurement_closing_state_copy', 'full_states/closing.json')):
        check('copied_measurement_metadata_' + name, payload(p[key]) == manifest['files'][MEASUREMENT_OUTPUT + '/' + name])
    check('direction_witnesses_bound_but_not_repacked', sum(name.endswith('.npz') for name in manifest['files']) == 128
        and not any(name.endswith(('.npz', '.npy', '.pth', '.pt')) for name in p['files'])
        and archive['bytes'] < 16 * 1024**2 and build['retained_measurement_payload_included'] is False,
        {'bound_RAW_gradient_arrays': 128, 'RAW_gradient_bytes_not_repacked': sum(value['bytes'] for name, value in manifest['files'].items() if name.endswith('.npz')),
         'retained_dependency_bytes': sum(value['bytes'] for value in manifest['files'].values())})
    workertext = (STAGE / PACKAGE / 'worker.py').read_text()
    calls = [node for node in ast.walk(ast.parse(workertext)) if isinstance(node, ast.Call)]
    forbidden = [ast.unparse(node.func) for node in calls if ast.unparse(node.func).endswith(('.backward', '.autograd.grad'))
        or ast.unparse(node.func).startswith(('torch.optim.', 'torch_module.optim.'))]
    check('source_reconstructs_saved_RAW8_no_new_autograd_or_optimizer', not forbidden
        and 'solver.solve_eight_pooled_gates(pooled)' in workertext
        and "direction = solved.pop('direction')" in workertext and "('gate_mae', 'gate_adjacent')" in workertext
        and "with bounded_phase('retained_RAW_gradients_and_original_direction_no_autograd')" in workertext,
        {'forbidden_calls': forbidden, 'new_16vector_solver': False})
    check('manual_unrun_no_promotion_and_zero_local_applications', p['format'] == 'u64-png-checked-step-manual-L4-v1'
        and p['stage_name'] == STAGE_NAME and p['manual_launch_only'] is True and p['automatic_retry_resume'] is False
        and p['model_qualified'] is False and p['original_u64_is_retained'] is True and p['app_incumbent_unchanged'] is True
        and build['actual_parameter_applications'] == 0 and build['neural_gradient_optimizer_training_VM_calls'] == 0)
    help_value = staged_help()
    check('actual_staged_help_exit0_with_all_scientific_imports_blocked', help_value['exit_code'] == 0, help_value)
    for path, value in list(consumed.items()): need(pin(ROOT / path) == value, 'Consumed bytes changed during audit: ' + path)
    need({name: pin(STAGE / name, STAGE) for name in expected} == expected, 'Stage changed after inert CLI help')
    result = {'complete': True, 'archive_review_pass': True, 'unresolved_blockers': [],
        'created_UTC': datetime.now(timezone.utc).isoformat(), 'archive': archive, 'checksum': checksum,
        'protocol': protocol, 'stage_protocol': pin(STAGE / 'protocol.json'),
        'build_receipt': pin(PREP / 'build_receipt_v1.json'), 'checks_pass': len(checks), 'checks': checks,
        'framing': parsed['framing'], 'actual_archive_file_pins': parsed['files'],
        'all16_sources_admitted_exact': list(tables[0].values()), 'parent_files_rehashed': parentrows,
        'local_parent_model_aliases': aliases, 'retained_measurement_files_rehashed': mrows,
        'explicit_builder_local_runtime_extraction_aliases': maliases,
        'staged_help': help_value, 'consumed_files': list(consumed.values()),
        'all_consumed_pins_unchanged_after': True, 'source': pin(Path(__file__)),
        'actual_worker_model_neural_parameter_optimizer_VM_browser_calls': 0,
        'scientific_imports_blocked': True, 'builder_imported_or_executed': False,
        'runtime_real_VM_verify_transfer_pass_claimed': False, 'manual_guide_approved_here': False,
        'scope': 'Actual archive and local757 dependency bindings only. All469 actual parent and288 actual retained-measurement VM paths remain mandatory; local hash-provenance aliases never authorize VM substitutions.',
        'model_qualified': False}
    target = PREP / 'archive_readback_v1.json'
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({'receipt': pin(target), 'checks_pass': len(checks), 'archive': archive,
        'protocol': protocol, 'framing': parsed['framing'], 'unresolved_blockers': []}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-sha', required=True)
    parser.add_argument('--protocol-sha', required=True)
    args = parser.parse_args()
    need(all(len(value) == 64 and set(value) <= set('0123456789abcdef') for value in
        (args.archive_sha, args.protocol_sha)), 'Explicit lowercase SHA256 pins required')
    audit_archive(args.archive_sha, args.protocol_sha)
