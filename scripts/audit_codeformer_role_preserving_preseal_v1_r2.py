"""Independent read-only actual R2 stage closure; no model or VM execution."""
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_role_preserving_training_v1_r2 import contract as c, transport


def need(value, message):
    if not value:
        raise ValueError(message)


def pin(path):
    path = Path(path).absolute()
    for ancestor in (path, *path.parents):
        if ancestor.exists() or ancestor.is_symlink():
            s = ancestor.lstat()
            need(not stat.S_ISLNK(s.st_mode) and not getattr(s, 'st_file_attributes', 0) & 1024,
                 'Linked/reparse source: ' + str(path))
    before = path.stat()
    need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and
         path.resolve().is_relative_to(ROOT), 'Exact unlinked regular workspace source required')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    after = path.stat()
    need((before.st_size, before.st_mtime_ns, before.st_ino) ==
         (after.st_size, after.st_mtime_ns, after.st_ino), 'Source mutated while hashing')
    return {'sha256': h.hexdigest(), 'bytes': before.st_size}


def main():
    started = time.monotonic()
    prep = ROOT / 'outputs/codeformer_role_preserving_training_v1_r2_preparation'
    stage_file = prep / 'stage_v1.json'
    output = prep / 'independent_preseal_v1.json'
    need(not output.exists(), 'Preserve previous independent preseal receipt')
    stage_pin = pin(stage_file)
    stage = c.read_json(stage_file)
    need(stage['complete'] is True and stage['source_stage_not_sealed'] is True and
         stage['local_neural_gradient_optimizer_VM_calls'] == 0, 'Completed inert stage required')
    folder = Path(stage['stage']).resolve(strict=True)
    need(folder == (ROOT / 'outputs' / c.VM_NAME).resolve(strict=True), 'Exact new R2 stage required')
    protocol_pin = pin(folder / 'protocol.json')
    need(protocol_pin['sha256'] == stage['protocol_sha256'], 'Stage/protocol identity differs')
    protocol = c.read_json(folder / 'protocol.json')
    c.verify_protocol(protocol)
    need(protocol['vm_name'] == 'cctv_codeformer_role_preserving_training_v1_r2_vm' and
         protocol['parent_dir'] == 'cctv_codeformer_adain_gradient_isolation_v1_vm' and
         protocol['outputs'] == 'outputs/role_preserving_training_v1_r2' and
         protocol['export_prefix'] == 'cctv-codeformer-role-preserving-training-v1-r2',
         'R2 namespace or immutable parent differs')
    expected = dict(protocol['files_sha256']) | {'protocol.json': protocol_pin['sha256']}
    sizes = dict(protocol['files_bytes']) | {'protocol.json': protocol_pin['bytes']}
    need(stage['source_files'] == protocol['files_sha256'] and
         stage['source_bytes'] == protocol['files_bytes'] and
         set(stage['current_sources']) == set(expected) - {'protocol.json'},
         'Actual source/protocol/current-origin namespace differs')
    actual = {}
    for current, directories, files in os.walk(folder, followlinks=False):
        for directory in directories:
            q = Path(current) / directory
            need(not q.is_symlink() and not getattr(q.lstat(), 'st_file_attributes', 0) & 1024,
                 'Linked stage directory')
        for filename in files:
            q = Path(current) / filename
            name = q.relative_to(folder).as_posix()
            actual[name] = pin(q)
    need(set(actual) == set(expected) and len(actual) <= c.LIMITS['source_packet_files'],
         'Unexpected/missing actual staged namespace')
    for name, digest in expected.items():
        need(actual[name] == {'sha256': digest, 'bytes': sizes[name]}, 'Staged exact source differs: ' + name)
        if name.endswith('.py'):
            ast.parse((folder / name).read_text(encoding='utf-8'), feature_version=(3, 10))
    current_origins = []
    for name, source in stage['current_sources'].items():
        if source is False:
            need(name in stage['generated_data_files'] or name == 'research/__init__.py',
                 'Undeclared generated stage member')
            if name == 'research/__init__.py':
                marker = stage['synthetic_sources'][name]
                need((folder / name).read_bytes() == b'' and
                     pin(ROOT / marker['local_reference_path']) == {
                         'sha256': marker['local_reference_sha256'], 'bytes': marker['local_reference_bytes']},
                     'Intentional empty research namespace reference differs')
            continue
        need(pin(ROOT / source) == actual[name], 'Current origin differs: ' + name)
        current_origins.append({'stage_path': name, 'current_path': source, **actual[name]})
    need(stage['generated_data_files'] == ['data_bundle.json'], 'New unreviewed generated inputs')
    bundle = c.read_json(folder / protocol['data_bundle_file'])
    need(len(bundle['training']) == 200 and len(bundle['hq']) == 100 and len(bundle['development']) == 406,
         'Actual TRAIN/HQ/DEV data scope differs')
    resolution = stage['local_dependency_resolution']
    all_dependencies = {}
    counts = {}
    for key in ('dependencies', 'retained_output_dependencies', 'extra_returned_dependencies'):
        need(stage[key] == protocol[key], 'Dependency map differs: ' + key)
        counts[key] = len(protocol[key])
        for name, expected_pin in protocol[key].items():
            need(name not in all_dependencies, 'Cross-group dependency overlap')
            need(pin(Path(resolution[name])) == expected_pin, 'Fresh parent asset/data differs: ' + name)
            all_dependencies[name] = expected_pin
    need(set(resolution) == set(all_dependencies) and sum(counts.values()) == 3362,
         'Exact physical minimal dependency resolution required')
    join = transport.verify_data_bundle(None, protocol, None,
        resolver=lambda name: Path(resolution[name]), bundle=bundle)
    numeric = stage['data_binding_receipt']
    need(numeric['complete'] is True and numeric['TRAIN_observations'] == 200 and
         numeric['HQ_selected_references'] == 100 and numeric['DEV_cases'] == 406 and
         numeric['neural_calls'] == 0, 'Actual CPU saved-pixel stage receipt differs')
    ledger = transport.resource_ledger(len(expected), sum(sizes.values()))
    need(ledger == stage['resource_ledger'], 'Full staged physical resource ledger differs')
    need(sum(v for k, v in protocol['call_caps'].items() if k.endswith('_forwards')) == 1399 and
         sum(protocol['call_caps'].values()) == 1854 and
         protocol['call_caps']['optimizer_step_attempts'] == 50 and
         protocol['call_caps']['backward_calls'] == 200 and protocol['call_caps']['gradient_queries'] == 5,
         'Finite R2 operation census differs')
    need(pin(stage_file) == stage_pin and pin(folder / 'protocol.json') == protocol_pin,
         'Stage/receipt changed during independent review')
    need(not any(name in sys.modules for name in ('torch', 'tensorflow', 'jax')),
         'Framework import forbidden')
    result = {
        'format': 'codeformer-role-preserving-training-r2-independent-actual-preseal-v1',
        'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
        'seconds': time.monotonic() - started, 'unresolved_findings': [],
        'stage': str(folder), 'stage_receipt_sha256': stage_pin['sha256'],
        'protocol_sha256': protocol_pin['sha256'], 'stage_member_sha256': expected,
        'stage_member_bytes': sizes, 'stage_members': len(expected),
        'stage_source_bytes': sum(sizes.values()),
        'all_actual_regular_unlinked_source_members_exact': True,
        'all_current_copied_origins_exact': current_origins,
        'exact_empty_research_namespace_and_only_declared_generated_data': True,
        'fresh_readonly_dependency_counts': counts,
        'fresh_readonly_binding_census': len(all_dependencies),
        'data_metadata_joins': join, 'actual_CPU_saved_pixel_receipt_bound': True,
        'resource_ledger': ledger,
        'call_caps': protocol['call_caps'], 'neural_forward_cap': 1399, 'reservation_cap': 1854,
        'source_sha256': pin(__file__)['sha256'],
        'local_neural_autograd_optimizer_VM_calls': 0,
        'prospective_only_no_model_quality_claim': True, 'manual_training_not_started': True,
        'model_qualified': False,
    }
    c.write_new_json(output, result)
    print(json.dumps({'complete': True, 'receipt': str(output), 'sha256': pin(output)['sha256'],
        'stage_members': len(expected), 'stage_payload_bytes': sum(sizes.values()),
        'protocol_sha256': protocol_pin['sha256'], 'parent_bindings': len(all_dependencies),
        'seconds': result['seconds']}))


if __name__ == '__main__':
    main()
