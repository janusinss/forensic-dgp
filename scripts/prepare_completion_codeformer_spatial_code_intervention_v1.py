"""Source-only spatial code intervention diagnostic packet; independent/root release required.
Standard library preparation only. No deserialization, neural/VM calls or learning.
"""
from __future__ import annotations
import argparse, ast, copy, gzip, hashlib, importlib.util, io, json, os, re, shutil, stat, tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
NAME = 'completion_codeformer_spatial_code_intervention_v1'
VM_NAME = 'cctv_' + NAME + '_vm'
STEM = 'cctv-completion-codeformer-spatial-code-intervention-v1'
FORMAT = 'completion-codeformer-spatial-code-intervention-v1'
RESEARCH = ROOT / ('research/' + NAME)
DEVELOPMENT = ROOT / ('research/' + NAME + '_development')
PACKETS = ROOT / ('outputs/' + NAME + '_packet')
TRANSFER = Path(r'C:\xampp\htdocs\YEAR 4\Testing\outputs')
MAX_EXECUTION_BYTES = 128 * 1024 ** 2
OLD = ROOT / 'outputs/completion_codeformer_same_codebook_probe_v1_packet/draft_001/cctv_completion_codeformer_same_codebook_probe_v1_vm'
OLD_RETURN = ROOT / 'outputs/completion_codeformer_same_codebook_probe_v1_return/cctv_completion_codeformer_same_codebook_probe_v1_results'
OLD_PROTOCOL_SHA = '51782b57fca9ad9cf7161b0f4451a65e545e3eb2f8f58788fa40b64aec48d1ec'
OLD_TRANSPORT = OLD_RETURN.parent / 'transport_audit.json'
OLD_TRANSPORT_SHA = 'ee4980dcd22a04f4315d7369a16c6fb2ce8ad8704532f6e1b284d35a45d24fa4'
DESIGN_PATH = ROOT / ('research/' + NAME + '_design/root_design.json')
DESIGN_SHA = '1983f9fed9bede2078ffc6a1af0dd6ca08fe6ec532e5c3de9c4ce41d4db2c657'
FORMAL_ADMISSION_PATH = ROOT / ('research/' + NAME + '_independent_review/formal_design_admission.json')
FORMAL_ADMISSION_SHA = '4f6ea7134e9c8628efb7deae4ca5dc137b35dc9f1d0471c796c8aa03692e63b5'
CLARIFICATION_PATH = ROOT / ('research/' + NAME + '_design/pre_release_clarifications_v1.json')
CLARIFICATION_SHA = '98c3521e5e7e0043479862923d03cc21c9b6b7c2e5371a0ffcd292d919ca1d78'
CURRENT_INSTRUCTIONS = ['SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md', 'PROJECT_HANDOFF.md', 'FACE_MASK_COMPLETION_SPEC_V1.md']
APP_SOURCES = ['completion.py', 'pretrained_completion.py', 'face_workflow.py', 'third_party/__init__.py', 'third_party/codeformer/__init__.py', 'third_party/codeformer/registry.py', 'third_party/codeformer/vqgan_arch.py', 'third_party/codeformer/codeformer_arch.py']
SOURCE_MAP = {name: OLD / name for name in APP_SOURCES}
for name in ('prepare', 'export'):
    source = 'scripts/' + name + '_' + NAME + '.py'
    SOURCE_MAP[source] = ROOT / source
SOURCE_MAP['scripts/run_' + NAME + '.sh'] = ROOT / ('scripts/run_' + NAME + '.sh')
SOURCE_MAP['scripts/cctv_' + NAME + '_vm.py'] = DEVELOPMENT / ('cctv_' + NAME + '_vm.py')
for name in ('contracts', 'optimized_budget', 'probe_forward', 'app_delivery'):
    SOURCE_MAP['research/' + NAME + '/' + name + '.py'] = DEVELOPMENT / (name + '.py')


def need(ok, label):
    if not ok: raise ValueError(label)


def physical(path):
    path = Path(path).absolute()
    return Path('\\\\?\\' + str(path)) if os.name == 'nt' and not str(path).startswith('\\\\?\\') else path


def canonical(path):
    value = str(path)
    return Path(value[4:] if value.startswith('\\\\?\\') else value)


def sha(path):
    h = hashlib.sha256()
    with physical(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''): h.update(block)
    return h.hexdigest()


def pin(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'bytes': physical(path).stat().st_size, 'sha256': sha(path)}


def read(path):
    def reject(value): raise ValueError('Nonfinite JSON ' + value)
    return json.loads(physical(path).read_text(encoding='utf-8-sig'), parse_constant=reject)


def write(path, value):
    physical(path.parent).mkdir(parents=True, exist_ok=True)
    with physical(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def regular(path):
    info = physical(path).lstat()
    need(stat.S_ISREG(info.st_mode) and not physical(path).is_symlink(), 'Regular non-symlink source required')
    return info


def safe_relative(name):
    need(isinstance(name, str) and name and '\\' not in name and ':' not in name and len(name.encode('utf-8')) <= 240, 'Unsafe member spelling/length')
    p = PurePosixPath(name)
    need(not p.is_absolute() and '..' not in p.parts and str(p) == name, 'Noncanonical/traversing member')
    return name


def dependencies():
    need(re.fullmatch('[a-f0-9]{64}', DESIGN_SHA) and re.fullmatch('[a-f0-9]{64}', FORMAL_ADMISSION_SHA), 'Source-only design/admission unbound; do not stage')
    need(sha(DESIGN_PATH) == DESIGN_SHA and sha(FORMAL_ADMISSION_PATH) == FORMAL_ADMISSION_SHA, 'Formal design/admission drift')
    need(sha(CLARIFICATION_PATH) == CLARIFICATION_SHA, 'Prospective interface/resource clarification drift')
    need(sha(OLD / 'protocol.json') == sha(OLD_RETURN / 'protocol.json') == OLD_PROTOCOL_SHA and sha(OLD_TRANSPORT) == OLD_TRANSPORT_SHA, 'Exact original same-codebook source/actual-return binding')
    design, old = read(DESIGN_PATH), read(OLD / 'protocol.json')
    for name, row in design['evidence'].items(): need(pin(Path(row['path'])) == row, 'Design evidence drift ' + name)
    admission = read(FORMAL_ADMISSION_PATH)
    need(admission.get('complete') is True and admission.get('independent') is True, 'Independent formal design approval incomplete')
    for name, path in SOURCE_MAP.items():
        safe_relative(name); regular(path)
        if path.suffix == '.py': ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 10))
    tree = ast.parse(SOURCE_MAP['scripts/cctv_' + NAME + '_vm.py'].read_text(encoding='utf-8'))
    declared = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'DESIGN_SHA' for t in node.targets)]
    need(declared == [DESIGN_SHA], 'Worker design remains unbound or differs')
    spec = importlib.util.spec_from_file_location('spatial_intervention_contract_only', DEVELOPMENT / 'contracts.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    c = module.CONTRACT
    need(c['forward_slots_planned'] == 108 and c['forward_slots_max'] == 120 and c['optimizer_steps_max'] == 0 and c['worker_seconds'] == 900 and c['output_bytes'] == 1024 ** 3 and c['minimum_free_bytes'] == 6 * 1024 ** 3, 'Prospective zero-update contract differs')
    need([r['id'] for r in module.ARMS] == ['H', 'V'], 'Exactlytwo prospective H/V arms')
    return design, old, c, module.ARMS


def allowed_output_paths(protocol):
    top = ['state_before.json', 'oracle_label_receipt.json', 'reused_control_binding.json', 'external_control_compatibility.json', 'arm_H_complete.json', 'arm_V_complete.json', 'state_after.json', 'results.json', 'failure.json', 'state_on_failure.json', 'state_receipt_failure.json', 'failed_actual_model_state.pth', 'failed_actual_model_state.pth.partial', 'failure_state_receipt.json']
    names = {'outputs/' + name for name in top}
    for folder, cases in [('fresh_app_parity', protocol['parity_cases']), ('external_bridge', protocol['bridge_cases'])]:
        names.add('outputs/' + folder + '/summary.json')
        for case in cases:
            for suffix in ('.npz', '.json'): names.add('outputs/' + folder + '/' + case['id'] + suffix)
    for arm in protocol['arms']:
        for kind, key in [('monitor', 'monitor_cases'), ('challenge', 'challenge_cases')]:
            prefix = 'outputs/' + arm['filename_tag'] + '/' + kind + '/'
            names.add(prefix + 'results.json')
            for case in protocol[key]:
                for suffix in ('.npz', '.png'): names.add(prefix + case['id'] + '__ORACLE_unavailable_clean_information' + suffix)
    return sorted(names)


def stage(stage_id):
    need(re.fullmatch('[A-Za-z0-9_-]{1,48}', stage_id), 'Invalid stage id')
    base, destination = PACKETS / stage_id, PACKETS / stage_id / VM_NAME
    need(not base.exists(), 'Preserve existing stage; use a new stage id')
    design, old, contract, arms = dependencies()
    frozen = {row['path']: row for row in old['files']}; planned = {}
    def add(relative, source, expected=None):
        safe_relative(relative); source = Path(source); info = regular(source)
        if expected is not None: need(info.st_size == expected['bytes'] and sha(source) == expected['sha256'], 'Exact source pin ' + relative)
        if relative in planned:
            need(planned[relative].resolve() == source.resolve(), 'Different duplicate source'); return
        planned[relative] = source
    def old_asset(relative):
        need(relative in frozen, 'Unpinned inherited source ' + relative)
        add(relative, OLD / relative, frozen[relative])
    monitor, challenge, parity = [copy.deepcopy(old[key]) for key in ('monitor_cases', 'challenge_cases', 'parity_cases')]
    need([r['id'] for r in monitor + challenge] == design['data']['case_ids_in_fixed_protocol_order'], 'Fixed48case order differs')
    cases_by_id = {row['id']: row for row in monitor + challenge}
    bridge = [copy.deepcopy(cases_by_id[ident]) for ident in design['data']['bridge_case_ids']]
    need(len(monitor) == 32 and len(challenge) == 16 and len(parity) == len(bridge) == 4, 'Fixed case counts')
    for row in monitor + challenge + parity:
        for key in ('input256', 'mask', 'clean256', 'clean512'): old_asset(row[key])
    for name, path in SOURCE_MAP.items():
        add(name, path, frozen[name] if name in APP_SOURCES else None)
    oracle = copy.deepcopy(old['oracle_labels'])
    for row in oracle: old_asset(row['path'])
    need(len(oracle) == 16, 'Reuse exactly16 same-codebook labels')
    evidence, evidence_sha = {}, {}
    def evidence_add(key, source, expected=None):
        relative = 'evidence/' + key + Path(source).suffix
        add(relative, source, expected); evidence[key] = relative; evidence_sha[key] = sha(source)
    evidence_add('root_design', DESIGN_PATH)
    evidence_add('formal_design_admission', FORMAL_ADMISSION_PATH)
    evidence_add('pre_release_clarifications', CLARIFICATION_PATH)
    installer_fixtures = ROOT / ('research/' + NAME + '_design/root_installer_fixtures.json')
    need(sha(installer_fixtures) == '610177441433cbb5f282c8264d35a0b63696e13b72cb4c3bcb627eb2c6b58e6e', 'Independent exact installer fixtures drift')
    need(read(installer_fixtures)['installer_literal_sha256'] == hashlib.sha256(INSTALL_CODE.encode()).hexdigest(), 'Independently tested installer literal differs')
    evidence_add('root_installer_fixtures', installer_fixtures)
    readable = ROOT / 'COMPLETION_CODEFORMER_SPATIAL_CODE_INTERVENTION_V1_DESIGN.md'
    need(sha(readable) == '6a03cc0ff96501b842606cae878bae00eabb8d5e44d0b90f74ba7bc0bac61c81', 'Readable formal design drift')
    evidence_add('readable_design', readable)
    fixtures = ROOT / ('research/' + NAME + '_packet_review/source_fixtures_release.json')
    fixture_data = read(fixtures)
    need(fixture_data.get('complete') is True and fixture_data.get('all_pass') is True and fixture_data.get('checks') == 17, 'Final pure-file installer/exporter fixture admission')
    expected_pins = [pin(ROOT / ('scripts/prepare_' + NAME + '.py')), pin(ROOT / ('scripts/export_' + NAME + '.py'))]
    need(fixture_data['source_pins'] == expected_pins, 'Final fixture source pins differ')
    evidence_add('author_transport_fixtures', fixtures)
    alias_fixtures = ROOT / ('research/' + NAME + '_packet_review/alias_revalidation_fixtures.json')
    alias_report = read(alias_fixtures)
    need(alias_report.get('complete') is True and alias_report.get('all_pass') is True and alias_report.get('checks') == 2 and alias_report['exporter'] == pin(ROOT / ('scripts/export_' + NAME + '.py')), 'Final actual post-tar alias race fixtures differ')
    evidence_add('author_alias_revalidation_fixtures', alias_fixtures)
    independent_alias = ROOT / 'research/completion_spatial_code_intervention_v1_transport_review/independent_alias_drift_repair_fixture.json'
    need(sha(independent_alias) == 'fb98877b30ed1cda7b60858cedbe16bea7f84d7353e74e33a51bcf6c7af78f51' and read(independent_alias).get('repair_refuses_before_publication') is True, 'Independent alias-race repair evidence differs')
    evidence_add('independent_alias_revalidation_fixture', independent_alias)
    evidence_add('original_probe_protocol', OLD / 'protocol.json')
    evidence_add('actual_original_probe_transport', OLD_TRANSPORT)
    for key, row in design['evidence'].items(): evidence_add('root_design_' + key, Path(row['path']), row)
    for name in ('oracle_label_index', 'parent_state_audit', 'parent_saved_output_audit'):
        old_asset(old['evidence'][name]); evidence[name] = old['evidence'][name]; evidence_sha[name] = old['evidence_sha256'][name]
    parent_protocol = old['original_parent_protocol']; old_asset(parent_protocol)
    for name in CURRENT_INSTRUCTIONS: add('evidence/current_instructions/' + name, ROOT / name)
    for name in ('source_tests.py', 'source_preparation_tests.json'): add('evidence/development/' + name, DEVELOPMENT / name)
    for name in ('LICENSE', 'NOTICE.md'): old_asset('legal/' + name)
    old_asset('evidence/original_source_admission.json')
    manifest = read(OLD_RETURN / 'export_manifest.json'); originals = {r['path']: r for r in manifest['files']}
    external_files, external_arms = [], {}
    for arm_id in ('A', 'C'):
        declared = next(a for a in old['arms'] if a['id'] == arm_id); external_arms[arm_id] = {}
        for kind, rows in [('monitor', monitor), ('challenge', challenge)]:
            prefix = declared['filename_tag'] + '/' + kind + '/'
            summary_path = prefix + 'results.json'; exact = originals['outputs/' + summary_path]
            source = OLD_RETURN / 'outputs' / summary_path
            need(pin(source)['bytes'] == exact['bytes'] and sha(source) == exact['sha256'], 'Exact reused control summary')
            summary = read(source); need([r['id'] for r in summary['records']] == [r['id'] for r in rows], 'External control fixed case order')
            external_arms[arm_id][kind] = summary
            for relative in [summary_path] + [prefix + case['id'] + ('__ORACLE_unavailable_clean_information' if arm_id == 'C' else '') + suffix for case in rows for suffix in ('.npz', '.png')]:
                exact = originals['outputs/' + relative]; path = OLD_RETURN / 'outputs' / relative
                need(regular(path).st_size == exact['bytes'] and sha(path) == exact['sha256'], 'Reused external A/C actual payload drift')
                external_files.append({'path': relative, 'bytes': exact['bytes'], 'sha256': exact['sha256']})
    need(len(external_files) == 196 and len({r['path'] for r in external_files}) == 196, 'Exactly196 A/C metadata-only external controls')
    records = [{'path': name, 'bytes': regular(path).st_size, 'sha256': sha(path)} for name, path in sorted(planned.items())]
    protocol = {'format': FORMAT, 'status': 'source_only_unrun_unreleased', 'frozen_UTC': datetime.now(timezone.utc).isoformat(), 'contract': contract, 'arms': arms,
                'checkpoint': copy.deepcopy(old['checkpoint']), 'parent': copy.deepcopy(old['parent']), 'references': copy.deepcopy(old['references']),
                'monitor_cases': monitor, 'challenge_cases': challenge, 'parity_cases': parity, 'bridge_cases': bridge, 'oracle_labels': oracle,
                'original_parent_protocol': parent_protocol, 'evidence': evidence, 'evidence_sha256': evidence_sha,
                'external_controls': {'root': '../cctv_completion_codeformer_same_codebook_probe_v1_vm/outputs', 'source_protocol_sha256': OLD_PROTOCOL_SHA, 'files': external_files, 'arms': external_arms, 'copied_to_packet': False},
                'terms': copy.deepcopy(old['terms']), 'data_limits': copy.deepcopy(old['data_limits']),
                'source_subset': {'source_original_probe_protocol_sha256': OLD_PROTOCOL_SHA, 'actual_cases': 48, 'monitor_references': 8, 'additional_parity_cases': 4, 'reused_oracle_labels': 16, 'teacher_network_calls': 0, 'public_or_FINAL_cases': 0},
                'instruction_snapshot': {'UTC': datetime.now(timezone.utc).isoformat(), 'status': 'Prospective source snapshots; current reviewed release/manual guide governs launch.'},
                'export': {'stem': STEM, 'whole_output_cap_bytes': 1024 ** 3, 'raw_tar_and_gzip_cap_bytes_each': 1280 * 1024 ** 2, 'cap_seconds': 900, 'external_hard_seconds': 900, 'external_hard_owner': 'GNU timeout --signal=KILL 900s in owned manual shell', 'minimum_free_bytes': 2 * 1024 ** 3, 'maximum_regular_members_including_manifest': 400, 'maximum_path_UTF8_bytes': 240, 'maximum_NPZ': 104, 'maximum_PNG': 96, 'unexpected_mutation_checkpoint_cap_bytes': 384 * 1024 ** 2, 'failure_partials_retained': True},
                'scope': 'Zero-update SAME-inpainting-codebook spatial intervention diagnosis. H/V deliberately use unavailable hidden clean-information codes at exact source-defined selectors; not deployable completion or identity recovery. Existing A/C comparisons are reused from admitted actual run; no trained model or app adoption.',
                'files': records, 'manual_execution_only': True, 'automatic_resume_or_promotion': False, 'diagnostic_only': True, 'new_model_created': False, 'model_qualified': False, 'goal_complete': False, 'release_required_before_archive': True, 'local_neural_calls': 0, 'optimizer_updates_locally': 0}
    protocol['export']['allowed_output_paths'] = allowed_output_paths(protocol)
    body = (json.dumps(protocol, indent=2, allow_nan=False) + '\n').encode()
    need(len(body) <= 2 * 1024 ** 2 and len(body) + sum(r['bytes'] for r in records) < MAX_EXECUTION_BYTES, 'Bounded compact protocol/payload')
    destination.mkdir(parents=True)
    for row in records:
        source, target = planned[row['path']], destination / row['path']; physical(target.parent).mkdir(parents=True, exist_ok=True)
        with physical(source).open('rb') as incoming, physical(target).open('xb') as outgoing: shutil.copyfileobj(incoming, outgoing, 8 * 1024 ** 2)
        need(physical(target).stat().st_size == row['bytes'] and sha(target) == sha(source) == row['sha256'], 'Staging source changed')
    with physical(destination / 'protocol.json').open('xb') as stream: stream.write(body)
    actual = {canonical(path).relative_to(destination).as_posix() for path in physical(destination).rglob('*') if path.is_file()}
    need(actual == {r['path'] for r in records} | {'protocol.json'}, 'Closed staging namespace')
    RESEARCH.mkdir(parents=True, exist_ok=True)
    need(not (RESEARCH / 'protocol.json').exists(), 'Preserve prior research protocol; bind new stage explicitly')
    with (RESEARCH / 'protocol.json').open('xb') as stream: stream.write(body)
    receipt = {'complete': True, 'source_only_stage': True, 'stage_id': stage_id, 'root': str(destination), 'protocol_sha256': sha(destination / 'protocol.json'), 'files': len(records), 'payload_bytes': len(body) + sum(r['bytes'] for r in records), 'external_controls_metadata_only': 196, 'archive_prepared': False, 'manual_root_release_required': True, 'model_loads': 0, 'neural_calls': 0, 'VM_calls': 0, 'optimizer_updates': 0}
    write(base / 'stage_receipt.json', receipt); print(json.dumps(receipt), flush=True); return receipt


INSTALL_CODE = r'''from pathlib import Path, PurePosixPath
import hashlib, json, shutil, tarfile, uuid
def need(ok, message):
    if not ok: raise ValueError(message)
base = Path.home() / 'forensic-dgp'
name = 'cctv_completion_codeformer_spatial_code_intervention_v1_vm'
destination = base / name
archive = Path.home() / 'cctv-completion-codeformer-spatial-code-intervention-v1-execution.tar.gz'
need(base.is_dir() and not destination.exists(), 'Preserve existing experiment; do not reinstall')
need(archive.stat().st_size <= 128 * 1024 ** 2, 'Execution archive size cap')
def safe(value):
    p = PurePosixPath(value)
    need(value and not p.is_absolute() and '..' not in p.parts and ':' not in value and '\\' not in value and str(p) == value and len(value.encode('utf-8')) <= 320, 'Unsafe member path')
    return p
with tarfile.open(archive, mode='r|gz') as stream:
    iterator = iter(stream)
    first = next(iterator)
    need(first.isreg() and first.name == name + '/protocol.json' and 0 < first.size <= 2 * 1024 ** 2, 'Expected first protocol member')
    body = stream.extractfile(first).read(2 * 1024 ** 2 + 1)
    need(len(body) == first.size and hashlib.sha256(body).hexdigest() == '__PROTOCOL_SHA__', 'Protocol SHA differs')
    p = json.loads(body)
    need(p['format'] == 'completion-codeformer-spatial-code-intervention-v1' and p['files'], 'Wrong/empty protocol')
    expected = {name + '/' + safe(row['path']).as_posix(): row for row in p['files']}
    need(len(expected) == len(p['files']), 'Duplicate protocol member')
    total = len(body) + sum(row['bytes'] for row in p['files'])
    need(total <= 128 * 1024 ** 2 and shutil.disk_usage(base).free >= 6 * 1024 ** 3 + total, 'Need6GiB plus exact installation payload free')
    stage = base / (name + '.installing-' + uuid.uuid4().hex)
    stage.mkdir()
    with (stage / 'protocol.json').open('xb') as output: output.write(body)
    seen = {first.name}
    for member in iterator:
        need(member.isreg() and member.name in expected and member.name not in seen, 'Unlisted/duplicate/nonregular member')
        parts = safe(member.name).parts
        need(parts[0] == name, 'Wrong member root')
        row = expected[member.name]
        need(member.size == row['bytes'], 'Member size differs')
        path = stage.joinpath(*parts[1:]); path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(); count = 0
        with stream.extractfile(member) as incoming, path.open('xb') as output:
            for block in iter(lambda: incoming.read(8 * 1024 ** 2), b''):
                count += len(block); need(count <= row['bytes'], 'Oversized member')
                digest.update(block); output.write(block)
        need(count == row['bytes'] and digest.hexdigest() == row['sha256'], 'Member SHA differs')
        seen.add(member.name)
    need(seen == set(expected) | {name + '/protocol.json'}, 'Missing payload members')
    need(not destination.exists(), 'Experiment appeared; preserve staging')
    stage.rename(destination)
print({'complete': True, 'installed': str(destination), 'protocol_sha256': '__PROTOCOL_SHA__', 'training_launched': False})
'''


def manual_guide(protocol_sha, archive_sha, archive_bytes, file_count):
    install = INSTALL_CODE.replace('__PROTOCOL_SHA__', protocol_sha)
    text = r'''# Spatial code intervention V1: manual zero-update diagnostic

PREPARED, UNRUN. R1 remains the working completion model. No automatic training.
Execution archive __ARCHIVE_BYTES__ bytes; __FILE_COUNT__ pinned payloads plus protocol.
ArchiveSHA256 `__ARCHIVE_SHA__`; protocolSHA256 `__PROTOCOL_SHA__`.

This distinct diagnostic separates whether clean-context code information helps
the mask-touched latent positions or the visible-context positions. Both H/V
use UNAVAILABLE CLEAN INFORMATION and cannot be deployed on covered inputs alone.
No model is trained. It does not replace R1 or repeat the rejected CE recipe.

## 1. Upload from Windows Google Cloud SDK Shell

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-completion-codeformer-spatial-code-intervention-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-completion-codeformer-spatial-code-intervention-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Verify and install in VM SSH

Require6GiB free after installation, plus the exact packet payload during extraction.
The existing original weights, full R1 state and admitted earlier same-codebook
probe A/C outputs must remain on this VM. No duplicate models or A/C arrays are
uploaded. If a source/control is absent, stop and retain the error; do not skip it.
The installer checks every closed regular member and refuses an existing target.

```bash
cd ~ &&
sha256sum -c cctv-completion-codeformer-spatial-code-intervention-v1-execution.tar.gz.sha256 &&
command -v timeout >/dev/null &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -f ~/forensic-dgp/cctv_completion_codeformer_gradient_v1_vm/weights/codeformer_inpainting.pth &&
test -f ~/forensic-dgp/cctv_completion_codeformer_finetune_v1_r1_vm/outputs/terminal_training_state.pth &&
test -d ~/forensic-dgp/cctv_completion_codeformer_same_codebook_probe_v1_vm/outputs &&
python3 - <<'PY'
__INSTALL_CODE__PY
```

Expected complete:true, protocol__PROTOCOL_SHA__, training_launched:false.
Keep failed install staging and all existing evidence; never bypass the checks.

## 3. Open tmux

Use your current session if already inside tmux. Otherwise:

```bash
tmux new-session -A -s completion_spatial_code_intervention_v1
```

## 4. Verify and launch inside tmux

```bash
cd ~/forensic-dgp/cctv_completion_codeformer_spatial_code_intervention_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_completion_codeformer_spatial_code_intervention_v1_vm.py --root . --protocol-sha __PROTOCOL_SHA__ --verify-transfer &&
bash scripts/run_completion_codeformer_spatial_code_intervention_v1.sh __PROTOCOL_SHA__
```

Verify-transfer checks files and196 reused A/C control payloads without importing
the neural model. The new run uses exact original and R1 weights; all497tensors
stay frozen. Four ordinary app parity pairs use8calls. Four fresh full-C bridge
cases use4calls and must match retained C raw/PNG outputs and A/C predicted
indices. H and V each produce
48outputs, making108plannedwholemodelcalls/120hard. Typed limits prevent any arm
from silently using another arm's allowance. No optimizer objects/steps, gradients,
backwards, teacher-network calls, new labels or ordinary-success checkpoint writes.

H substitutes the existing oracle indices only at mask-touched16x16positions;
V substitutes them at the complementary visible-context positions. Both keep
observed inputs, exact reviewed masks and R1 fusionw1 fixed. These labels come
from the SAME original inpainting codebook; the incompatible HQ512prior is not used.
The full48-case A/C cohorts are hash-verified and reused rather than rerun; only
the four declared C bridge calls are new. The48pairedcases are adaptiveTRAIN
(32clear/16syntheticchallenges), not
public/unpaired hidden truth, nativeCCTV performance or independentFINAL review.

All96newPNGs,104NPZs, actual predicted/used indices, selectors and before/after
497statehashes are retained. This is a bounded diagnosis, not a quality-selected
new model. No result is generated from unavailable clean information in the app.

Worker15minhard/14mincooperative with60ssave reserve;6GiBstart/2GiBfloor;
20GiBCUDAallocated cap. Wholeoutputs1GiB; each rawtar/gzip1280MiB. Export has
15-minute cooperative checks and an owned external900-second GNUtimeout KILL
deadline. If export is killed, retain its partial archive and all outputs; an
export timeout is not successful packaging and does not authorize relaunching
the model or repeating this diagnostic. The shell checks timeout exists before
any model work.
No ordinary checkpoint is saved. Only an unexpected model mutation may retain a
bounded actual497-tensor+RNG failure state or partial, at most384MiB cumulative;
that is evidence, not an adopted/resumable model. Failed arrays/partials remain.
Never auto-resume, repeat a historical pilot or promote from exportcomplete.

The worker intentionally writes receipts rather than progress lines; trainer.log
may stay empty until termination. Let the foreground run finish within its bounded
15-minute limit. An empty tail is not evidence that it stopped. Do not launch a
second process or repeat the command because the log is quiet. After the normal
prompt returns, inspect supervisor.json and the export receipt shown below.

Monitor from a second VM SSH terminal:

```bash
tail -F ~/forensic-dgp/cctv_completion_codeformer_spatial_code_intervention_v1_vm/trainer.log
```

Detach with Ctrl+B thenD. When finished:

```bash
cat ~/forensic-dgp/cctv_completion_codeformer_spatial_code_intervention_v1_vm/supervisor.json
cat ~/cctv-completion-codeformer-spatial-code-intervention-v1-export.json
```

Exit0 means the diagnostic finished, regardless of measured improvement; an
exportcomplete receipt means packaging succeeded. Neither qualifies a model.
The successful archive.partial is a hardlink to the same inode, not a second
payload copy. Retain all successful/failed evidence until independently audited.

## 5. Download from Windows Google Cloud SDK Shell

Use one remote source per command, on Windows rather than inside VM SSH:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-spatial-code-intervention-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-spatial-code-intervention-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-spatial-code-intervention-v1-export.json" "."
```

Return all three files for independent transport/state/indices/RAW/PNG and all-case
visual review. R1, RESTORATION and IMPLEMENTATION remain unchanged.
'''
    return text.replace('__ARCHIVE_BYTES__', str(archive_bytes)).replace('__FILE_COUNT__', str(file_count)).replace('__ARCHIVE_SHA__', archive_sha).replace('__PROTOCOL_SHA__', protocol_sha).replace('__INSTALL_CODE__', install)


def archive_stage(stage_id, release_path):
    need(re.fullmatch('[A-Za-z0-9_-]{1,48}', stage_id), 'Invalid stage id')
    base = PACKETS / stage_id; stage_root = base / VM_NAME; protocol_path = stage_root / 'protocol.json'
    p = read(protocol_path); expected = sha(protocol_path)
    need((RESEARCH / 'protocol.json').read_bytes() == protocol_path.read_bytes(), 'Research/staged protocol differs')
    release = read(release_path)
    need(release.get('format') == FORMAT + '-root-release-v1' and release.get('released_for_manual_vm_transfer') is True and release.get('automatic_vm_launch') is False and release.get('protocol_sha256') == expected, 'Exact root manual-transfer release required')
    admission_pin = release['independent_source_admission']; admission_path = Path(admission_pin['path'])
    need(pin(admission_path) == admission_pin, 'Independent source admission pin')
    admission = read(admission_path)
    need(admission.get('complete') is True and admission.get('independent') is True and admission.get('protocol_sha256') == expected, 'Independent final source/packet admission required')
    need(p['format'] == FORMAT and p['manual_execution_only'] and not p['automatic_resume_or_promotion'] and p['diagnostic_only'], 'Manual zero-update protocol')
    files = []
    for row in p['files']:
        safe_relative(row['path']); path = stage_root / row['path']; regular(path)
        need(physical(path).stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Staged source drift ' + row['path']); files.append((row, path))
    actual = {canonical(path).relative_to(stage_root).as_posix() for path in physical(stage_root).rglob('*') if path.is_file()}
    need(len({r['path'] for r in p['files']}) == len(p['files']) and actual == {r['path'] for r in p['files']} | {'protocol.json'}, 'Closed final staging namespace')
    destination = base / (STEM + '-execution.tar.gz'); partial = Path(str(destination) + '.partial'); checksum = Path(str(destination) + '.sha256')
    for path in (destination, partial, checksum): need(not path.exists(), 'Preserve prior archive/partial')
    total = protocol_path.stat().st_size + sum(r['bytes'] for r, _ in files)
    need(total + 2048 * (len(files) + 1) + 10240 < MAX_EXECUTION_BYTES, 'Execution payload/header cap')
    class Limited:
        def __init__(self, stream): self.stream = stream; self.count = 0
        def write(self, value):
            need(self.count + len(value) <= MAX_EXECUTION_BYTES, 'Execution raw/compressed cap')
            count = self.stream.write(value); need(count == len(value), 'Short execution write'); self.count += count; return count
        def flush(self): return self.stream.flush()
    with partial.open('xb') as raw:
        compressed = Limited(raw)
        with gzip.GzipFile(filename='', mode='wb', fileobj=compressed, compresslevel=1, mtime=0) as zipped:
            uncompressed = Limited(zipped)
            with tarfile.open(fileobj=uncompressed, mode='w|', bufsize=1024 * 1024, format=tarfile.PAX_FORMAT) as tar:
                for row, path in [({'path': 'protocol.json', 'bytes': protocol_path.stat().st_size, 'sha256': expected}, protocol_path)] + files:
                    need(sha(path) == row['sha256'], 'Archive source drift')
                    info = tarfile.TarInfo(VM_NAME + '/' + row['path']); info.size = row['bytes']; info.mode = 0o644; info.mtime = 0
                    with physical(path).open('rb') as stream: tar.addfile(info, stream)
                    need(sha(path) == row['sha256'], 'Source changed during archive')
        raw.flush(); os.fsync(raw.fileno())
    digest = sha(partial); os.link(partial, destination)
    with checksum.open('x', encoding='ascii', newline='\n') as stream: stream.write(digest + '  ' + destination.name + '\n')
    published = []
    for source in (destination, checksum):
        target = TRANSFER / source.name
        with source.open('rb') as incoming, target.open('xb') as outgoing: shutil.copyfileobj(incoming, outgoing, 8 * 1024 ** 2)
        need(sha(source) == sha(target), 'Transfer copy mismatch'); published.append(pin(target))
    guide = manual_guide(expected, digest, destination.stat().st_size, len(p['files']))
    guide_name = 'COMPLETION_CODEFORMER_SPATIAL_CODE_INTERVENTION_V1_VM.md'
    for target in (ROOT / guide_name, TRANSFER / guide_name):
        need(not target.exists(), 'Preserve prior guide')
        with target.open('x', encoding='utf-8', newline='\n') as stream: stream.write(guide)
    receipt = {'complete': True, 'archive': pin(destination), 'protocol_sha256': expected, 'archive_sha256': digest, 'payload_files': len(files) + 1, 'payload_bytes': total, 'raw_tar_bytes': uncompressed.count, 'root_release': pin(Path(release_path)), 'independent_source_admission': admission_pin, 'published': published, 'manual_VM_guide': pin(TRANSFER / guide_name), 'successful_archive_partial_hardlink_retained': str(partial), 'training_launched': False, 'VM_calls': 0, 'model_qualified': False}
    write(base / 'archive_publication_receipt.json', receipt); print(json.dumps(receipt), flush=True); return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--stage', action='store_true'); group.add_argument('--archive', action='store_true')
    parser.add_argument('--stage-id', required=True); parser.add_argument('--release-file', type=Path)
    args = parser.parse_args()
    if args.stage:
        need(args.release_file is None, 'No release-file during staging'); stage(args.stage_id)
    else:
        need(args.release_file is not None, 'Root release-file required'); archive_stage(args.stage_id, args.release_file)


if __name__ == '__main__': main()
