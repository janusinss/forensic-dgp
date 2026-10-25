"""Independent return audit (transport, saved pixels, closed states, and exposure-decoupling diagnosis) for Conditioned SFT V1."""
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time
import zlib
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TESTING_ROOT = ROOT.parent / 'Testing'
STAGE_NAME = 'cctv_codeformer_conditioned_sft_learning_v1_vm'
RUN_REL = 'outputs/conditioned_sft_learning_v1'
PROTOCOL_SHA = 'f54c37e8b6b05473b1530395549b31f89189445f642ecfc49b1280ba8442d7f0'

SEALED_STAGE = ROOT / 'outputs' / STAGE_NAME
AUDIT_DIR = ROOT / 'outputs/codeformer_conditioned_sft_learning_v1_return_audit_v1'
RETAINED_STAGE = AUDIT_DIR / 'retained' / STAGE_NAME

import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.codeformer_conditioned_sft_learning_v1 import (
    conditioning,
    feasibility,
    quality,
    records,
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> dict:
    h = hashlib.sha256()
    size = 0
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
            size += len(chunk)
    return {'bytes': size, 'sha256': h.hexdigest()}


def find_return_archive() -> Path:
    candidates = [
        ROOT / 'outputs/cctv-codeformer-conditioned-sft-learning-v1-results.tar.gz',
        TESTING_ROOT / 'outputs/cctv-codeformer-conditioned-sft-learning-v1-results.tar.gz',
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError('cctv-codeformer-conditioned-sft-learning-v1-results.tar.gz not found')


def failed_leaves(value, path=''):
    results = []
    if isinstance(value, dict):
        children = []
        for k, child in value.items():
            if isinstance(child, (dict, list)):
                children.extend(failed_leaves(child, path + '/' + k))
        if value.get('pass') is False and not children:
            results.append({'path': path, **copy.deepcopy(value)})
        results.extend(children)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            results.extend(failed_leaves(child, path + '/' + str(idx)))
    return results


def run_return_audit():
    started = time.monotonic()
    archive_path = find_return_archive()
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    RETAINED_STAGE.mkdir(parents=True, exist_ok=True)

    protocol = json.loads((SEALED_STAGE / 'protocol.json').read_text(encoding='utf-8'))
    static_files = {
        k: v for k, v in protocol['files'].items()
        if not k.startswith('weights/')
    }

    # 1. Stream archive, verify hashes, and extract data-only retention namespace
    member_index_path = AUDIT_DIR / 'member_index.json'
    if member_index_path.exists() and (RETAINED_STAGE / RUN_REL / 'result.json').exists():
        member_index_doc = json.loads(member_index_path.read_text(encoding='utf-8'))
        members = member_index_doc['members']
        payload_bytes = member_index_doc['payload_bytes']
        retained_bytes = member_index_doc['retained_bytes']
        archive_pin = {'bytes': archive_path.stat().st_size, 'sha256': member_index_doc['archive_sha256']}
    else:
        archive_pin = sha256_file(archive_path)
        members = {}
        payload_bytes = 0
        retained_bytes = 0
        forbidden_suffix = ('.py', '.pyc', '.pyo', '.sh', '.bat', '.cmd', '.exe', '.dll', '.ps1')
        runtime_files = {
            'install.json', 'install_failure.json', 'run.log', 'export.log',
            'supervisor.start.json', 'supervisor.worker.started.json', 'supervisor.worker.json',
            'supervisor.export.started.json', 'supervisor.export.json', 'supervisor.json',
            'supervisor_failure.json', 'export.json', 'export_failure.json',
        }

        with tarfile.open(archive_path, mode='r|gz') as tf:
            for member in tf:
                if not member.isfile():
                    continue
                assert member.name.startswith(STAGE_NAME + '/'), f'Unexpected member prefix: {member.name}'
                rel = member.name[len(STAGE_NAME) + 1:]
                stream = tf.extractfile(member)
                assert stream is not None
                data = stream.read()
                assert len(data) == member.size
                digest = sha256_bytes(data)
                row = {'bytes': len(data), 'sha256': digest}
                if rel in static_files:
                    assert row == static_files[rel], f'Static file modified on VM: {rel}'
                if rel == 'protocol.json':
                    assert digest == PROTOCOL_SHA, 'Protocol SHA mismatch in return archive'
                members[rel] = row
                payload_bytes += len(data)

                keep = (rel == 'protocol.json' or rel in runtime_files or rel.startswith(RUN_REL + '/'))
                if keep and not rel.lower().endswith(forbidden_suffix):
                    dest = RETAINED_STAGE / rel
                    if not dest.exists():
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(data)
                    retained_bytes += len(data)

        # Create sidecar .sha256 in both outputs folders if missing
        sidecar_line = f"{archive_pin['sha256']}  {archive_path.name}\n"
        for out_root in (TESTING_ROOT / 'outputs', ROOT / 'outputs'):
            if out_root.exists():
                sc_path = out_root / (archive_path.name + '.sha256')
                if not sc_path.exists():
                    sc_path.write_text(sidecar_line, encoding='ascii', newline='\n')

        member_index_doc = {
            'complete': True,
            'archive_sha256': archive_pin['sha256'],
            'member_count': len(members),
            'payload_bytes': payload_bytes,
            'retained_bytes': retained_bytes,
            'members': members,
        }
        member_index_path.write_text(json.dumps(member_index_doc, indent=2) + '\n', encoding='utf-8', newline='\n')

    # 2. Verify saved calls, checkpoints, and update commits
    vm_out = RETAINED_STAGE / RUN_REL
    result_doc = json.loads((vm_out / 'result.json').read_text(encoding='utf-8'))
    status_doc = json.loads((vm_out / 'status.json').read_text(encoding='utf-8'))
    worker_doc = json.loads((RETAINED_STAGE / 'supervisor.worker.json').read_text(encoding='utf-8'))

    call_records = {}
    for rec_path in sorted((vm_out / 'calls').glob('*/record.json')):
        rec = json.loads(rec_path.read_text(encoding='utf-8'))
        c = rec['call']
        call_records[(c['arm'], c['bank'], c['id'], c['phase'], c['new_update'])] = rec

    checkpoints_verified = 0
    updates_verified = 0
    role_proj_summary = {'A': [], 'B': []}
    for arm in ('A', 'B'):
        for u in range(0, 9):
            cp_file = vm_out / f'arms/{arm}/checkpoints/u{u:04d}/checkpoint.json'
            if cp_file.exists():
                cp = json.loads(cp_file.read_text(encoding='utf-8'))
                npz_path = RETAINED_STAGE / cp['trainable']['path']
                assert sha256_file(npz_path)['sha256'] == cp['trainable']['sha256']
                checkpoints_verified += 1
        for u in range(1, 9):
            cm_file = vm_out / f'arms/{arm}/updates/u{u:04d}/commit.json'
            if cm_file.exists():
                cm = json.loads(cm_file.read_text(encoding='utf-8'))
                updates_verified += 1
                role_proj_summary[arm].append({
                    'update': u,
                    'preclip_norm': cm['preclip_norm'],
                    'displacement': cm['actual_displacement'],
                    'conflict_projection': cm['role_gradient_conflict_projection'],
                })

    transport_doc = {
        'complete': True,
        'created_UTC': datetime.now(timezone.utc).isoformat(),
        'protocol': {'bytes': protocol['files'].get('protocol.json', {}).get('bytes', 1248212), 'sha256': PROTOCOL_SHA},
        'archive': {'path': str(archive_path), **archive_pin},
        'member_count': len(members),
        'static_files_verified': len(static_files),
        'saved_call_records': len(call_records),
        'closed_checkpoints_verified': checkpoints_verified,
        'saved_updates_verified': updates_verified,
        'worker_exit_code': worker_doc['exit_code'],
        'worker_elapsed_seconds': worker_doc['seconds'],
        'member_index': sha256_file(member_index_path),
    }
    (AUDIT_DIR / 'transport.json').write_text(json.dumps(transport_doc, indent=2) + '\n', encoding='utf-8', newline='\n')

    # 3. Extract gate comparisons and test full-exposure drift decoupling on returned Arm B u0008
    helper = feasibility._load_pixel_helper()
    selection = protocol['selection']
    parent_protocol_bytes = (SEALED_STAGE / protocol['parent_protocol']['path']).read_bytes()
    historical_pixel_audit_bytes = (SEALED_STAGE / protocol['historical_pixel_audit']['path']).read_bytes()
    frozen_records = json.loads((vm_out / 'frozen_U96_control_metrics.json').read_text(encoding='utf-8'))['records']

    controls = {row['id']: row for row in protocol['controls']}
    frozen_pixels = {}
    teacher_pixels = {}
    for source in selection['old_quality']:
        cid = source['id']
        ctrl = controls[cid]
        f_raw = helper.decode_array_bytes((SEALED_STAGE / ctrl['raw']['path']).read_bytes(), np=np)
        with Image.open(SEALED_STAGE / ctrl['png']['path']) as im:
            f_png = np.asarray(im.convert('RGB'), dtype=np.uint8).copy()
        frozen_pixels[cid] = (f_raw, f_png)
        if source.get('behavior_teacher') is not None:
            t = source['behavior_teacher']
            t_raw = helper.decode_array_bytes((SEALED_STAGE / t['raw']['path']).read_bytes(), np=np)
            with Image.open(SEALED_STAGE / t['png']['path']) as im:
                t_png = np.asarray(im.convert('RGB'), dtype=np.uint8).copy()
            teacher_pixels[cid] = (t_raw, t_png)

    arm_gate_summaries = {}
    for arm in ('A', 'B'):
        b0 = json.loads((vm_out / f'arms/{arm}/banks/u0000/bank.json').read_text(encoding='utf-8'))
        b8 = json.loads((vm_out / f'arms/{arm}/banks/u0008/bank.json').read_text(encoding='utf-8'))
        gate8 = b8['gate']
        arm_gate_summaries[arm] = {
            'overall_pass': gate8['pass'],
            'inherited_pass': gate8['inherited']['pass'],
            'supplemental_pass': gate8['supplemental']['pass'],
            'failed_leaves': failed_leaves(gate8),
            'supplemental_checks': gate8['supplemental']['role_mean_nonincrease_checks'],
            'inherited_additional_retention': gate8['inherited'].get('additional_retention'),
        }

    # Test dark-exposure drift decoupling on the returned Arm B u0008 saved RAW outputs
    b0_records = json.loads((vm_out / 'arms/B/banks/u0000/bank.json').read_text(encoding='utf-8'))['records']
    old_recomputed = {}
    mon_recomputed = {}
    for bank_name, dest_map in (('old_quality', old_recomputed), ('new_monitors', mon_recomputed)):
        for case in selection[bank_name]:
            cid = case['id']
            r0_rec = call_records[('B', bank_name, cid, 'initial_quality', 0)]
            r8_rec = call_records[('B', bank_name, cid, 'quality_snapshot', 8)]
            raw0 = helper.decode_array_bytes((RETAINED_STAGE / r0_rec['outputs']['raw']['path']).read_bytes(), np=np)
            raw8 = helper.decode_array_bytes((RETAINED_STAGE / r8_rec['outputs']['raw']['path']).read_bytes(), np=np)
            with Image.open(SEALED_STAGE / case['input']['path']) as im:
                inp_u8 = np.asarray(im.convert('RGB'), dtype=np.uint8).copy()
            inp_f32 = inp_u8.astype(np.float32) / np.float32(255.0)
            gates = conditioning.compute_input_conditioning_numpy(inp_f32, np=np)
            # Preserve fine spatial detail from B8 while sign-aligning coarse 16x16 exposure drift on dark cases (g_dark > 0)
            if gates['g_dark'] > 0.0:
                coarse_drift = conditioning.block16_mean_numpy(raw8 - raw0, np=np)
                fixed_raw = np.clip(raw8 - 2.0 * gates['g_dark'] * coarse_drift, 0.0, 1.0).astype(np.float32)
            else:
                fixed_raw = raw8
            fixed_png = (np.clip(fixed_raw, 0.0, 1.0) * 255.0).astype(np.uint8)
            if case['target'] is None:
                targ_f32 = None
            else:
                with Image.open(SEALED_STAGE / case['target']['path']) as im:
                    targ_u8 = np.asarray(im.convert('RGB'), dtype=np.uint8).copy()
                targ_f32 = targ_u8.astype(np.float32) / np.float32(255.0)
            row_case = {**case, 'bank': bank_name, 'case_key': bank_name + ':' + cid}
            if bank_name == 'old_quality':
                f_raw, f_png = frozen_pixels[cid]
                kw = {}
                if cid in teacher_pixels:
                    kw = {'teacher_raw': teacher_pixels[cid][0], 'teacher_png': teacher_pixels[cid][1]}
                dest_map[cid] = records.record_inherited(
                    fixed_raw, fixed_png, targ_f32, row_case, np=np,
                    frozen_U96_raw=f_raw, frozen_U96_png=f_png, **kw
                )
            else:
                dest_map[cid] = records.record_new(fixed_raw, fixed_png, targ_f32, row_case, np=np)

    inh8_fixed = quality.evaluate_inherited(
        old_recomputed, b0_records['old_quality'], frozen_records,
        parent_protocol_bytes, historical_pixel_audit_bytes, update=8
    )
    inh48_fixed = quality.evaluate_inherited(
        old_recomputed, b0_records['old_quality'], frozen_records,
        parent_protocol_bytes, historical_pixel_audit_bytes, update=48
    )
    sup8_fixed = quality.evaluate_new_bank(
        mon_recomputed, b0_records['new_monitors'], selection, update=8
    )
    sup48_fixed = quality.evaluate_new_bank(
        mon_recomputed, b0_records['new_monitors'], selection, update=48
    )

    pixels_doc = {
        'complete': True,
        'created_UTC': datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds': time.monotonic() - started,
        'returned_arms_u0008': arm_gate_summaries,
        'role_gradient_projection_commits': role_proj_summary,
        'dark_drift_decoupled_B8_counterfactual': {
            'inh8_pass': inh8_fixed['pass'],
            'inh48_pass': inh48_fixed['pass'],
            'sup8_pass': sup8_fixed['pass'],
            'sup48_pass': sup48_fixed['pass'],
            'inh8_failed_leaves': failed_leaves(inh8_fixed),
            'inh48_failed_leaves': failed_leaves(inh48_fixed),
            'sup8_checks': sup8_fixed['role_mean_nonincrease_checks'],
            'sup48_checks': sup48_fixed['role_mean_nonincrease_checks'],
        },
    }
    (AUDIT_DIR / 'pixels.json').write_text(json.dumps(pixels_doc, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({
        'transport_complete': True,
        'members': len(members),
        'calls': len(call_records),
        'checkpoints': checkpoints_verified,
        'updates': updates_verified,
        'B8_returned_inh_pass': arm_gate_summaries['B']['inherited_pass'],
        'B8_returned_sup_pass': arm_gate_summaries['B']['supplemental_pass'],
        'B8_dark_decoupled_inh8_pass': inh8_fixed['pass'],
        'B8_dark_decoupled_inh48_pass': inh48_fixed['pass'],
        'B8_dark_decoupled_sup8_pass': sup8_fixed['pass'],
        'B8_dark_decoupled_sup48_pass': sup48_fixed['pass'],
    }, indent=2))


if __name__ == '__main__':
    run_return_audit()
