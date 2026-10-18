"""CPU-only source-stage builder; never seal, upload, instantiate a model, or launch."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_tone_control_training_v1 import contract as c

PRIOR_STAGE = ROOT / 'outputs/codeformer_normal_tone_training_v1_preparation/stage_v1.json'
PRIOR_STAGE_SHA256 = '869ad664a9b30eead9fe68c4f33e7afcc312751bbffb8d856a1b39f2c5c40bc7'
OLD_RETURN = ROOT / 'outputs/codeformer_adain_gradient_isolation_v1_return'
OLD_IMPORT = ROOT / 'outputs/codeformer_adain_gradient_isolation_v1_return_review/transport_v1.json'


def _fresh_binding(path, pin):
    path = c.no_links(path)
    c.require(path.resolve(strict=True).is_relative_to(ROOT), 'Local binding escapes workspace')
    before = path.stat()
    c.require(path.is_file() and before.st_nlink == 1 and before.st_size == pin['bytes'] and
              c.sha256(path) == pin['sha256'], 'Fresh exact local dependency differs: ' + str(path))
    after = path.stat()
    c.require((before.st_size, before.st_mtime_ns, before.st_ino) ==
              (after.st_size, after.st_mtime_ns, after.st_ino), 'Dependency changed while hashing')
    return str(path)


def dependency_bindings():
    """Reuse the exact installed parent namespace plus already returned output witnesses."""
    c.require(c.sha256(PRIOR_STAGE) == PRIOR_STAGE_SHA256, 'Pinned latest parent resolver differs')
    previous = c.read_json(PRIOR_STAGE)
    deps = deepcopy(previous['dependencies'])
    c.require(len(deps) == 2011 and c.canonical_sha256(deps) == c.DEPENDENCIES_SHA256,
              'Exact2011 installed parent dependency map required')
    local = {name: _fresh_binding(Path(previous['local_dependency_resolution'][name]), pin)
             for name, pin in sorted(deps.items())}
    c.require(c.sha256(OLD_IMPORT) == c.HISTORICAL_IMPORT_RECEIPT_SHA256,
              'Verified historical import receipt differs')
    imported = c.read_json(OLD_IMPORT)
    c.require(imported.get('complete') is True and imported.get('exact_namespace_verified') is True and
              imported.get('whole_archive_and_all_member_hashes_verified_before_import') is True and
              imported.get('protocol_sha256') == c.PARENT_PROTOCOL_SHA and
              Path(imported['import_directory']).resolve() == OLD_RETURN.resolve(),
              'Historical exact return closure required')
    manifest_path = c.safe_path(OLD_RETURN, imported['manifest_member'])
    c.require(c.sha256(manifest_path) == imported['manifest_sha256'] == c.HISTORICAL_EXPORT_MANIFEST_SHA256,
              'Pinned historical export manifest differs')
    manifest = c.read_json(manifest_path)
    c.require(manifest.get('complete') is True and manifest.get('protocol_sha256') == c.PARENT_PROTOCOL_SHA,
              'Historical output manifest provenance differs')
    retained = {}
    prefixes = ['outputs/adain_gradient_isolation_v1/snapshot_' + update + '/' for update in ('0000', '0050')]
    for name, digest in sorted(manifest['files_sha256'].items()):
        if any(name.startswith(prefix) for prefix in prefixes) and name.endswith(('/raw.npy', '/native.png')):
            path = c.safe_path(OLD_RETURN, name)
            pin = {'sha256': digest, 'bytes': path.stat().st_size}
            c.require(imported['files_sha256'].get(name) == digest, 'Import/output witness disagrees')
            local[name] = _fresh_binding(path, pin)
            retained[name] = pin
    c.require(len(retained) == 1624 and c.canonical_sha256(retained) == c.RETAINED_OUTPUTS_SHA256,
              'Exact1624 historical original/base output witnesses required')
    old = c.read_json(local['protocol.json'])
    c.require(c.sha256(local['protocol.json']) == c.PARENT_PROTOCOL_SHA and
              c.canonical_sha256(old) == c.PARENT_PROTOCOL_CANONICAL_SHA,
              'Exact historical parent protocol required')
    return deps, retained, local, old


def prepare(stage, receipt, provenance=()):
    stage, receipt = c.no_links(stage), c.no_links(receipt)
    c.require(stage.parent.resolve() == (ROOT / 'outputs').resolve() and stage.name == c.VM_NAME,
              'Exact canonical new local source stage required')
    c.require(receipt.resolve(strict=False).is_relative_to(ROOT / 'outputs') and
              not stage.exists() and not receipt.exists(), 'Preserve prior/partial stage and receipt')
    dependencies, retained, local, old = dependency_bindings()
    source_names = sorted(c.REQUIRED_SOURCE_FILES - {'research/__init__.py'})
    source_names += ['tests/test_codeformer_tone_control_' + suffix + '_v1.py'
                     for suffix in ('data', 'model', 'transport', 'worker')]
    instructions = ['AGENTS.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
                    'PROJECT_HANDOFF.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md',
                    'CCTV_CODEFORMER_TONE_CONTROL_TRAINING_V1_DESIGN.md']
    payloads = {name: c.safe_path(ROOT, name).read_bytes() for name in source_names}
    current = {name: name for name in source_names}
    for name in instructions:
        target = 'instructions/' + name
        payloads[target] = c.safe_path(ROOT, name).read_bytes()
        current[target] = name
    for name in provenance:
        path = c.safe_path(ROOT, name)
        c.require(path.suffix in ('.json', '.md'), 'Provenance must be bounded JSON/Markdown source')
        target = 'provenance/' + path.name
        c.require(target not in payloads, 'Provenance basename collision')
        payloads[target] = path.read_bytes()
        current[target] = name
    payloads['research/__init__.py'] = b''
    # The packet needs a neutral package namespace, not the local package's
    # initializer. Record that local source as provenance without importing or
    # copying it, and make the synthetic override explicit for the sealer.
    local_init = c.safe_path(ROOT, 'research/__init__.py').read_bytes()
    current['research/__init__.py'] = False
    synthetic = {'research/__init__.py': {
        'policy': 'intentional empty transport namespace; local initializer not copied',
        'packet_sha256': hashlib.sha256(b'').hexdigest(), 'packet_bytes': 0,
        'local_reference_path': 'research/__init__.py',
        'local_reference_sha256': hashlib.sha256(local_init).hexdigest(),
        'local_reference_bytes': len(local_init), 'copied_or_executed': False}}
    c.require(len({name.casefold() for name in payloads}) == len(payloads) and len(payloads) <= 48 and
              sum(map(len, payloads.values())) <= 16 * 1024 ** 2, 'Bounded source packet census exceeded')
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()}
    sizes = {name: len(value) for name, value in payloads.items()}
    design_document = 'instructions/CCTV_CODEFORMER_TONE_CONTROL_TRAINING_V1_DESIGN.md'
    protocol = {'format': c.FORMAT, 'module': c.MODULE, 'parent_dir': c.PARENT_NAME,
        'parent_protocol_sha256': c.PARENT_PROTOCOL_SHA, 'outputs': c.OUTPUTS, 'export_prefix': c.PREFIX,
        'design': deepcopy(c.DESIGN), 'design_sha256': c.DESIGN_SHA256,
        'design_document': {'path': design_document, 'sha256': hashes[design_document]},
        'call_caps': deepcopy(c.COUNTS), 'limits': deepcopy(c.LIMITS), 'assets': deepcopy(c.ASSETS),
        'data_manifest': 'references.json', 'comparison_cases': 'comparison_cases.json',
        **deepcopy(c.PARENT_CONTROLS), 'historical_parent_protocol': old,
        'dependencies': dependencies, 'retained_output_dependencies': retained,
        'parent_parity_outputs': c.parent_parity_paths(), 'files_sha256': hashes, 'files_bytes': sizes,
        'status': 'PREPARED_UNRUN', 'manual_tmux_required': True, 'new_training_recipe': True,
        'actual_candidate_checkpoints_required': True, 'automatic_training': False, 'automatic_resume': False,
        'model_qualified': False, 'automatic_app_promotion': False, 'quality_gates_changed': False,
        'FINAL_used': False, 'old_optimizer_loaded': False, 'cached_gradient_reuse': False}
    c.verify_protocol(protocol)
    # No stage is created until all source/data/parent bindings have passed.
    stage.mkdir()
    for name, payload in payloads.items():
        path = stage / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(payload)
    c.write_new_json(stage / 'protocol.json', protocol)
    c.write_new_json(receipt, {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
        'stage': str(stage), 'protocol_sha256': c.sha256(stage / 'protocol.json'),
        'source_files': hashes, 'source_bytes': sizes, 'current_sources': current,
        'synthetic_sources': synthetic,
        'dependencies': dependencies, 'retained_output_dependencies': retained,
        'local_dependency_resolution': local, 'dependency_count': len(dependencies),
        'retained_output_dependency_count': len(retained), 'total_readonly_bindings': len(local),
        'source_sha256': c.sha256(__file__), 'source_stage_not_sealed': True,
        'local_neural_gradient_optimizer_VM_calls': 0, 'model_qualified': False})
    return {'complete': True, 'protocol_sha256': c.sha256(stage / 'protocol.json'),
        'static_files': len(payloads) + 1, 'dependency_count': len(dependencies),
        'retained_output_dependency_count': len(retained), 'stage_only_not_sealed': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=ROOT / 'outputs' / c.VM_NAME)
    parser.add_argument('--receipt', type=Path,
                        default=ROOT / 'outputs/codeformer_tone_control_training_v1_preparation/stage_v1.json')
    parser.add_argument('--provenance', action='append', default=[])
    args = parser.parse_args(argv)
    print(prepare(args.stage, args.receipt, args.provenance))


if __name__ == '__main__':
    main()
