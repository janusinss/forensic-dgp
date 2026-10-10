"""Independent archive/data/controller/guard inspection; no neural calls."""
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import time
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cctv_dgp_multiscale_calibration_contract_v1 import NAME, STEM, BUDGETS, read, write, sha, verify


def main():
    start = time.monotonic(); dest = ROOT / 'outputs/cctv_dgp_multiscale_calibration_v1_transfer_audit_r1'
    assert not dest.exists(); dest.mkdir()
    packet = ROOT / 'outputs' / NAME; archive = ROOT / 'outputs' / (STEM + '-execution.tar.gz')
    prep = read(ROOT / 'outputs/cctv_dgp_multiscale_calibration_v1_preparation.json')
    sources = {n: sha(ROOT / n) for n in ['scripts/verify_cctv_dgp_multiscale_transfer_v1_r1.py',
        'scripts/prepare_cctv_dgp_multiscale_calibration_v1.py', 'scripts/cctv_dgp_multiscale_calibration_contract_v1.py',
        'outputs/' + NAME + '/protocol.json', 'outputs/cctv_dgp_multiscale_calibration_v1_preparation.json']}
    write(dest / 'plan.json', {'source_sha256': sources, 'archive_sha256': prep['archive_sha256'],
        'guards_must_refuse_local_run': True, 'neural_calls': 0, 'optimizer_updates': 0, 'gradient_queries': 0})
    p = verify(packet, prep['protocol_sha256'])
    assert sha(archive) == prep['archive_sha256'] and archive.stat().st_size == prep['bytes']
    assert Path(str(archive) + '.sha256').read_text().strip() == prep['archive_sha256'] + '  ' + archive.name
    checked = 0
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers(); names = [m.name for m in members]
        wanted = {NAME + '/' + n for n in p['assets_sha256']} | {NAME + '/protocol.json'}
        assert len(set(names)) == len(names) and set(names) == wanted
        for m in members:
            n = PurePosixPath(m.name); assert m.isfile() and not n.is_absolute() and '..' not in n.parts and ':' not in m.name and '\\' not in m.name
            h = hashlib.sha256(); f = tar.extractfile(m)
            for block in iter(lambda: f.read(8 * 1024**2), b''):
                h.update(block)
            expected = prep['protocol_sha256'] if m.name == NAME + '/protocol.json' else p['assets_sha256'][m.name[len(NAME)+1:]]
            assert h.hexdigest() == expected, m.name; checked += 1
    for n, d in p['local_sources'].items():
        assert sha(ROOT / n) == d, n
    refs = {r['id']: r for r in p['references']}
    for r in refs.values():
        with Image.open(packet / r['target']) as im:
            assert im.size == (256, 256); a = np.asarray(im.convert('RGB')).copy()
        assert hashlib.sha256(a.tobytes()).hexdigest() == p['canonical_target_RGB_sha256'][r['id']]
    for c in p['cases']:
        with Image.open(packet / c['input']) as im:
            assert im.size == (256, 256)
            if c['profile'] == 'clear':
                rgb = np.asarray(im.convert('RGB')).copy()
                assert hashlib.sha256(rgb.tobytes()).hexdigest() == p['canonical_target_RGB_sha256'][c['source_person_or_reference']]
    assert {'provenance/native/LICENSE_SOURCE.html', 'provenance/native/DERIVATIVE_NOTICE.txt'} <= set(p['assets_sha256'])
    for c in p['native_development']:
        assert c['input_review']['input_sha256'] == p['assets_sha256'][c['input']]
        assert c['source_person_id'] == c['input_review']['source_person_id']
        for k in ['input', 'observed', 'native_crop']:
            assert p['assets_sha256'][c[k]] == c[k + '_sha256']
    for f in packet.rglob('*.py'):
        ast.parse(f.read_text(encoding='utf-8-sig'), filename=str(f))
    worker = packet / 'scripts/cctv_dgp_multiscale_calibration_vm_v1.py'
    verify_call = subprocess.run([sys.executable, '-B', str(worker), '--root', str(packet), '--protocol-sha', prep['protocol_sha256'], '--verify-transfer'],
        capture_output=True, text=True, timeout=60)
    assert verify_call.returncode == 0, verify_call.stderr
    guard = subprocess.run([sys.executable, '-B', str(worker), '--root', str(packet), '--protocol-sha', prep['protocol_sha256'], '--run'],
        capture_output=True, text=True, timeout=60)
    assert guard.returncode != 0 and 'Manual existing Linux VM only' in guard.stderr and not (packet / 'outputs').exists()
    supervisor = subprocess.run([sys.executable, '-B', str(packet / 'scripts/supervise_cctv_dgp_multiscale_calibration_v1.py'),
        '--root', str(packet), '--protocol-sha', prep['protocol_sha256']], capture_output=True, text=True, timeout=60)
    assert supervisor.returncode != 0 and 'AssertionError' in supervisor.stderr and not (packet / 'trainer.log').exists()
    bash = Path('C:/Program Files/Git/bin/bash.exe')
    assert bash.is_file(), 'Need an actual bash parser for the manual shell'
    syntax = subprocess.run([str(bash), '--noprofile', '--norc', '-n', str(packet / 'scripts/run_multiscale.sh')], capture_output=True, text=True, timeout=20)
    assert syntax.returncode == 0, syntax.stderr
    assert p['visual_review_plan']['total_pages'] == p['visual_review_plan']['paired_pages'] + p['visual_review_plan']['native_pages'] == 64
    planned_counts = {'original': 20+24, 'candidate': 20+24+20+12*(10+20+24), 'recognizer': 20+20+12*(10+20)}
    assert planned_counts == {k: BUDGETS[k + '_forward_calls'] for k in planned_counts}
    assert 20*14 == BUDGETS['maximum_gradient_queries'] and 12*10 == BUDGETS['maximum_backwards']
    assert BUDGETS['minimum_free_disk_bytes'] >= 2*BUDGETS['return_uncompressed_bytes'] + BUDGETS['disk_reserve_bytes']
    original_cp = ROOT / 'outputs/cctv_dgp_head4_capacity_vm_v1'
    source_protocol = read(original_cp / 'protocol.json')
    base_return = ROOT / 'outputs/cctv_dgp_head4_capacity_vm_v1_return/outputs/update0'
    by_id = {c['id']: i for i, c in enumerate(source_protocol['cases'])}
    measured_packs = sum((base_return / 'packs' / f'b{by_id[p["cases"][begin]["id"]]//5:04d}.npz').stat().st_size for begin in range(0, 100, 5))
    # Native floats plus worst-case uint8 PNG allowance; no new model forward.
    historical_sample_estimate = measured_packs + sum((base_return / 'previews' / (c['id'] + '.png')).stat().st_size for c in p['cases']) + 24*(256*256*3*4 + 200*1024) + 512*1024
    projection = 20*14*609219*4+256 + int(historical_sample_estimate*13*1.35) + 20*1024**2 + 12*32*1024**2 + (6*147456+6*609219)*4 + 1024**2
    assert projection < BUDGETS['return_uncompressed_bytes'], ('Conservative retained-output projection exceeds cap', projection)
    for n, d in sources.items():
        assert sha(ROOT / n) == d
    write(dest / 'local_guard_receipts.json', {'verify_stdout': verify_call.stdout, 'run_stderr': guard.stderr,
        'supervisor_stderr': supervisor.stderr, 'shell_parser': str(bash), 'shell_syntax_exit_code': syntax.returncode,
        'optimizer_updates': 0, 'gradient_queries': 0, 'neural_calls': 0})
    result = {'complete': True, 'protocol_sha256': prep['protocol_sha256'], 'archive_sha256': prep['archive_sha256'],
        'archive_members_and_GZIP_CRC_verified': checked, 'packet_assets_verified': len(p['assets_sha256']),
        'local_source_bindings_unchanged': len(p['local_sources']), 'canonical_TRAIN_targets': 20,
        'paired_TRAIN_inputs': 100, 'usable_unpaired_native_DEV_inputs': 24, 'reserved_final_pixels': 0,
        'historical_sample_storage_projection_bytes': projection, 'runtime_projection_still_required': True,
        'planned_forward_counts': planned_counts, 'planned_gradient_queries': 280, 'planned_backwards': 120,
        'planned_optimizer_updates': 12, 'shell_syntax_verified': True, 'local_training_guards_refused': 2,
        'local_optimizer_updates': 0, 'local_gradient_queries': 0, 'neural_calls': 0,
        'manual_L4_run_pending': True, 'model_qualification': False, 'goal_complete': False, 'seconds': time.monotonic()-start}
    write(dest / 'independent_audit.json', result); print(result, flush=True)


if __name__ == '__main__':
    main()
