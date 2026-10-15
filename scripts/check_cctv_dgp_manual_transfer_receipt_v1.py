"""Compare the actual hash-only transfer receipt to the sealed protocol."""
import argparse
import ast
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from import_cctv_dgp_conditioning_learning_v1_r4_return import ROOT, OUT, PARENT, read, sha, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    args = parser.parse_args()
    packet, destination = args.packet.resolve(), args.record.resolve()
    assert packet.is_relative_to(OUT) and destination.is_relative_to(OUT)
    p = read(packet / 'protocol.json')
    pin = sha(packet / 'protocol.json')
    worker_name = next(n for n in p['assets_sha256'] if n.startswith('scripts/') and n.endswith('_vm.py'))
    worker = packet / worker_name
    assert sha(worker) == p['assets_sha256'][worker_name]
    result = subprocess.run([sys.executable, '-B', str(worker), '--root', str(packet),
        '--parent-root', str(PARENT), '--protocol-sha', pin, '--verify-transfer'],
        capture_output=True, text=True, timeout=180, check=True)
    actual = ast.literal_eval(result.stdout.strip())
    expected = dict(packet_assets=len(p['assets_sha256']), parent_assets=len(p['parent_required_sha256']),
                    cases=p['paired_TRAIN_cases'], maximum_accepted_updates=p['maximum_accepted_updates'],
                    maximum_gradient_queries=p['maximum_gradient_queries'],
                    minimum_free_bytes=p['minimum_free_bytes'], actual_neural_calls=0)
    differences = {k: dict(actual=actual[k], expected=v) for k, v in expected.items() if actual[k] != v}
    assert actual['complete'] and actual['actual_neural_calls'] == 0
    write(destination, dict(complete=not differences, UTC=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__), protocol_sha256=pin, worker_sha256=sha(worker),
        actual_hash_only_worker_receipt=actual, expected_protocol_fields=expected, differences=differences,
        no_VM_connection_or_training_or_neural_execution=True, model_qualified=False, goal_complete=False))
    print(dict(complete=not differences, differences=differences, neural_calls=0), flush=True)
    assert not differences, 'Transfer receipt disagrees with the sealed protocol'


if __name__ == '__main__':
    main()
