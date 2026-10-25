"""Independent execution-packet byte/framing audit; never extracts or runs sources."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time


ROOT = Path(__file__).resolve().parents[1]
STAGE = 'cctv_codeformer_u96_spatial_feasibility_v1_vm'
PACKAGE = 'research/codeformer_u96_spatial_feasibility_v1'
PREFIX = 'cctv-codeformer-u96-spatial-feasibility-v1'
MAX_EXPANDED = 128 * 1024**2
MAX_MEMBER = 32 * 1024**2
MAX_ARCHIVE = 128 * 1024**2
TRUSTED_REVIEW = ROOT / 'outputs/codeformer_u96_spatial_feasibility_v1_preparation/runner_review/new_runner_source_review_v1.json'

# Reuse only the independently tested R1 single-frame/complete-EOF primitives,
# not its old model/return counter logic or its archive namespace.
HELPER = ROOT / 'scripts/audit_u96_decoder_readout_transport_v1.py'
_spec = importlib.util.spec_from_file_location('retained_R1_gzip_tar_primitives', HELPER)
_helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helper)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def relative(name):
    need(type(name) is str and name.startswith(STAGE + '/'), 'Exact new execution prefix')
    value = name[len(STAGE) + 1:]
    path = PurePosixPath(value)
    need(bool(value) and path.as_posix() == value and not path.is_absolute() and
         '\\' not in value and ':' not in value and '..' not in path.parts,
         'Canonical confined member path')
    return value


def inspect_stream(raw, *, guard=lambda: None):
    """Complete gzip CRC/ISIZE and tar namespace, including buffered EOF tail."""
    stream = None
    def bounded_guard():
        guard()
        if stream is not None:
            need(stream.expanded <= MAX_EXPANDED, 'Execution expanded-byte cap before tar parsing')
    class ExecutionGzip(_helper.SingleGzip):
        def read(self, size):
            value = super().read(size)
            need(self.expanded <= MAX_EXPANDED, 'Execution expanded-byte cap before tar parsing')
            return value
    stream = ExecutionGzip(raw, bounded_guard)
    rows = {}
    contents = {}
    last = 0
    with tarfile.open(fileobj=stream, mode='r|') as archive:
        for member in archive:
            guard()
            need(member.isfile() and member.type in (tarfile.REGTYPE, tarfile.AREGTYPE) and
                 not member.linkname and 0 <= member.size <= MAX_MEMBER,
                 'Only bounded regular source/data members')
            name = relative(member.name)
            need(name not in rows and len(rows) < 1000, 'Unique bounded member namespace')
            value = archive.extractfile(member).read(MAX_MEMBER + 1)
            need(len(value) == member.size, 'Complete declared member content')
            rows[name] = {'bytes': len(value), 'sha256': sha(value)}
            if name.endswith(('.json', '.py', '.sh')):
                contents[name] = value
            last = ((member.offset_data + member.size + 511) // 512) * 512
            need(stream.expanded <= MAX_EXPANDED, 'Execution expanded-byte cap')
    while stream.read(65536):
        guard()
        need(stream.expanded <= MAX_EXPANDED, 'Execution expanded-byte cap')
    padding = _helper.verify_tar_end(stream, last)
    return rows, contents, {'gzip_single_frame_complete': stream.done,
                            'gzip_CRC_and_ISIZE_verified': True,
                            'expanded_bytes': stream.expanded,
                            'complete_zero_tar_padding_bytes': padding}


def bound_json_member(name, members, contents):
    """Read evidence from actual archived bytes, never a loose local alias."""
    need(name in members and name in contents and
         members[name] == {'bytes': len(contents[name]), 'sha256': sha(contents[name])},
         'Actual archived JSON byte binding required: ' + name)
    return _helper.strict_json(contents[name])


def bind_trusted_review(contents, trusted_bytes):
    """A packet's replaced green approval cannot authorize its own changes."""
    name = 'evidence/spatial_runner_source_review_v1.json'
    need(type(trusted_bytes) is bytes and contents.get(name) == trusted_bytes,
         'Archived approval must match the independently trusted local review bytes')


def parent_review_readback(p, members, contents, *, contract):
    parent_pin = p['parent_protocol']
    need(parent_pin['path'] == 'evidence/spatial_parent_protocol.json' and
         (parent_pin['bytes'], parent_pin['sha256']) ==
         (contract.PARENT_PROTOCOL_BYTES, contract.PARENT_PROTOCOL_SHA256) and
         members.get(parent_pin['path']) == {'bytes': parent_pin['bytes'], 'sha256': parent_pin['sha256']},
         'Exact archived immutable parent binding required')
    parent = bound_json_member(parent_pin['path'], members, contents)
    for field in ('model', 'data', 'cases', 'guards', 'quality_policy'):
        need(p[field] == parent[field], 'Exact archived parent field required: ' + field)
    selection = contract.select_parent_cases(contents[parent_pin['path']])
    review = bound_json_member('evidence/spatial_runner_source_review_v1.json', members, contents)
    need(review['complete'] is True and review['source_review_pass'] is True and
         review['unresolved_blockers'] == [], 'Bound archived complete-runner source review')
    rows = review['sources']
    need(len({row['path'] for row in rows}) == len(rows), 'Unique approved source paths required')
    return selection, review


def retained_namespace_readback(p, members, contents):
    """Check retained data/code and new executable namespaces independently."""
    parent = bound_json_member(p['parent_protocol']['path'], members, contents)
    retained = {name: row for name, row in parent['files'].items()
                if name.startswith(('assets/', 'evidence/', 'third_party/'))}
    for name, binding in retained.items():
        need(p['files'].get(name) == binding and members.get(name) == binding,
             'Immutable retained parent file changed or omitted: ' + name)
    need(contents.get('research/__init__.py') == b'"""Inert research namespace."""\n',
         'Exact inert research namespace initializer required')
    for name in members:
        if name in retained or name in ('protocol.json', 'research/__init__.py'):
            continue
        if name.startswith(PACKAGE + '/') or name in (
                'scripts/run_u96_spatial_feasibility_v1.py', 'scripts/run_feasibility.sh'):
            continue
        need(name.startswith('evidence/spatial_') and name.endswith(('.json', '.md')),
             'Unexpected new executable/data namespace: ' + name)
    return len(retained)


def reviewed_namespace_readback(p, members, approved):
    """Require every reviewed installed source and only two omitted weights."""
    weights = {row['path']: {'bytes': row['bytes'], 'sha256': row['sha256']}
               for row in p['model'].values()}
    need({name: row for name, row in p['files'].items() if name.startswith('weights/')} == weights,
         'Exactly the two fixed initializer paths may be omitted')
    installed = set()
    for name in approved:
        if not name.startswith(PACKAGE + '/'):
            continue
        installed.add('scripts/run_u96_spatial_feasibility_v1.py' if name == PACKAGE + '/script_entry.py' else
                      'scripts/run_feasibility.sh' if name == PACKAGE + '/run_feasibility.sh' else name)
    actual = {name for name in members if name.startswith((PACKAGE + '/', 'scripts/'))}
    need(actual == installed, 'Exact complete reviewed installed namespace required')


def audit(archive_path, protocol_path, checksum_path, destination):
    start = time.monotonic()
    need(not destination.exists(), 'Preserve prior independent audits')
    def guard():
        need(time.monotonic() - start < 180, 'Independent execution-audit180s cap')
    need(archive_path.is_file() and 0 < archive_path.stat().st_size <= MAX_ARCHIVE,
         'Bounded existing execution archive')
    initial_pins = {'archive': _helper.filepin(archive_path),
                    'protocol': _helper.filepin(protocol_path),
                    'checksum': _helper.filepin(checksum_path),
                    'trusted_runner_source_review': _helper.filepin(TRUSTED_REVIEW)}
    trusted_review_bytes = TRUSTED_REVIEW.read_bytes()
    need(sha(trusted_review_bytes) == initial_pins['trusted_runner_source_review']['sha256'],
         'Trusted review changed during initial read')
    expected_sha = initial_pins['archive']['sha256']
    checksum = checksum_path.read_bytes()
    need(checksum == (expected_sha + '  ' + archive_path.name + '\n').encode('ascii'),
         'Exact LF companion checksum')
    protocol_bytes = protocol_path.read_bytes()
    p = _helper.strict_json(protocol_bytes)
    need(p['format'] == 'complete-U96-spatial-feasibility-manual-L4-v1' and p['stage_name'] == STAGE,
         'New finite spatial-feasibility protocol')
    with archive_path.open('rb') as stream:
        members, contents, framing = inspect_stream(stream, guard=guard)
    expected = {name: binding for name, binding in p['files'].items() if not name.startswith('weights/')}
    expected['protocol.json'] = {'bytes': len(protocol_bytes), 'sha256': sha(protocol_bytes)}
    need(members == expected and contents['protocol.json'] == protocol_bytes,
         'Every sealed nonweight byte must match; no omitted or extra members')
    bind_trusted_review(contents, trusted_review_bytes)
    checks = 2 + len(expected)
    # Only current, trusted, inert local contract/objective are imported. Their
    # exact bytes must be the same reviewed source bytes carried by the packet.
    for name in ('contract.py', 'objective.py'):
        relative_source = PACKAGE + '/' + name
        need((ROOT / relative_source).read_bytes() == contents[relative_source],
             'Local inert contract differs from actual archived source')
    sys.path.insert(0, str(ROOT))
    from research.codeformer_u96_spatial_feasibility_v1 import contract
    selection, review = parent_review_readback(p, members, contents, contract=contract)
    retained_count = retained_namespace_readback(p, members, contents)
    need(p['feasibility'] == contract.prospective_contract() and p['limits'] == contract.LIMITS,
         'Exact218/32/2 resource contract')
    need(len(contract.operation_plan(selection)) == 218 and len(selection['optimization']) == 16 and
         len(selection['monitors']) == 32 and len(selection['guards']) == 24,
         'Exact source-admitted diagnostic cohorts')
    approved = {row['path']: row for row in review['sources']}
    reviewed_namespace_readback(p, members, approved)
    current_sources = []
    for name, payload in contents.items():
        if not name.startswith((PACKAGE + '/', 'scripts/')) or not name.endswith(('.py', '.sh')):
            continue
        source_name = (PACKAGE + '/script_entry.py' if name == 'scripts/run_u96_spatial_feasibility_v1.py' else
                       PACKAGE + '/run_feasibility.sh' if name == 'scripts/run_feasibility.sh' else name)
        if name == 'research/__init__.py':
            continue
        need(source_name in approved and (len(payload), sha(payload)) ==
             (approved[source_name]['bytes'], approved[source_name]['sha256']),
             'Executable archive bytes must join independently reviewed source: ' + name)
        need(b'\r' not in payload, 'LF new executable source')
        current_sources.append({'installed': name, 'reviewed_source': source_name, **members[name]})
        checks += 1
    need(PACKAGE + '/worker.py' in {row['installed'] for row in current_sources}, 'Worker approval required')
    need(p['manual_launch_only'] is True and p['automatic_retry_resume'] is False and p['model_qualified'] is False,
         'Manual-only, no qualification from archive')
    guard()
    final_pins = {'archive': _helper.filepin(archive_path),
                  'protocol': _helper.filepin(protocol_path),
                  'checksum': _helper.filepin(checksum_path),
                  'trusted_runner_source_review': _helper.filepin(TRUSTED_REVIEW)}
    need(final_pins == initial_pins and sha(protocol_bytes) == final_pins['protocol']['sha256'] and
         sha(checksum) == final_pins['checksum']['sha256'], 'Audited inputs changed during verification')
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = {'complete': True, 'independent_archive_audit_pass': True,
              'recorded_UTC': datetime.now(timezone.utc).isoformat(), 'checks': checks + 6,
              **final_pins, 'audited_input_pins_unchanged': True, 'members': len(members),
              'payload_bytes': sum(value['bytes'] for value in members.values()),
              'framing': framing, 'reviewed_executable_join': current_sources,
              'immutable_retained_parent_files': retained_count,
              'auditor': _helper.filepin(Path(__file__).resolve()),
              'retained_gzip_tar_primitives': _helper.filepin(HELPER),
              'seconds': time.monotonic() - start, 'Torch_imported': 'torch' in sys.modules,
              'neural_gradient_optimizer_training_VM_calls': 0,
              'actual_L4_parity_gradient_quality_verified': False, 'model_qualified': False}
    need('torch' not in sys.modules, 'No framework in execution-file audit')
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    need(args.audit, 'Explicit --audit required')
    result = audit(ROOT / 'outputs' / (PREFIX + '-execution.tar.gz'),
                   ROOT / 'outputs' / STAGE / 'protocol.json',
                   ROOT / 'outputs' / (PREFIX + '-execution.tar.gz.sha256'),
                   ROOT / 'outputs/codeformer_u96_spatial_feasibility_v1_preparation/independent_execution_archive_audit_v1.json')
    print(json.dumps({key: result[key] for key in ('complete', 'checks', 'members', 'framing', 'archive')}))


if __name__ == '__main__':
    main()
