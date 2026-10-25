"""Deterministic builder for Conditioned SFT V2 manual NVIDIA L4 execution packet & 5-step VM guide."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.codeformer_conditioned_sft_learning_v2 import (
    contract_v2 as contract,
    packet_v2 as packet,
)

TESTING_ROOT = Path(r'C:\xampp\htdocs\YEAR 4\Testing')
V1_STAGE = ROOT / 'outputs/cctv_codeformer_conditioned_sft_learning_v1_vm'
V2_PREP = ROOT / 'outputs/codeformer_conditioned_sft_learning_v2_preparation'
FEAS_V2 = ROOT / 'outputs/codeformer_conditioned_sft_learning_v2_feasibility/v2_feasibility_receipt_v1.json'
STAGE_OUT = ROOT / 'outputs' / contract.STAGE_NAME
ARCHIVE_OUT = ROOT / 'outputs' / (contract.PREFIX + '-execution.tar.gz')
ARCHIVE_SHA_OUT = Path(str(ARCHIVE_OUT) + '.sha256')
SOURCE_PREP_OUT = ROOT / 'outputs/codeformer_conditioned_sft_learning_v2_source_preparation'
VM_GUIDE_OUT = ROOT / 'CCTV_CODEFORMER_CONDITIONED_SFT_LEARNING_V2_VM.md'

NATIVE_SOURCES = {
    'data/native_cctv/choke_dev_01_t033.png': TESTING_ROOT / 'outputs/cctv_chokepoint_native_development_v1/inputs/choke_dev_01_t033.png',
    'data/native_cctv/choke_dev_03_t033.png': TESTING_ROOT / 'outputs/cctv_chokepoint_native_development_v1/inputs/choke_dev_03_t033.png',
    'data/native_cctv/dev_24to39_01_input.png': TESTING_ROOT / 'outputs/cctv_native_comparison_v1/images/dev_24to39_01_input.png',
    'data/native_cctv/dev_ge40_01_input.png': TESTING_ROOT / 'outputs/cctv_native_comparison_v1/images/dev_ge40_01_input.png',
}


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def copy_lf_normalized(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    raw = src.read_bytes()
    if src.suffix in ('.py', '.sh', '.md', '.json') or src.name == 'LICENSE':
        raw = raw.replace(b'\r\n', b'\n')
        if src.suffix in ('.py', '.sh') and not raw.endswith(b'\n'):
            raw += b'\n'
    dst.write_bytes(raw)


def copy_exact(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def main():
    # Normalize LF in workspace V2 sources first so workspace and staged copies have identical SHA256
    v2_workspace_files = [
        ROOT / 'research/__init__.py',
        ROOT / 'research/codeformer_conditioned_sft_inference_v1.py',
        ROOT / 'research/codeformer_conditioned_sft_inference_v2.py',
        ROOT / 'scripts/run_conditioned_sft_learning_v2.py',
        *sorted((ROOT / 'research/codeformer_conditioned_sft_learning_v2').glob('*.py')),
        ROOT / 'research/codeformer_conditioned_sft_learning_v2/run_conditioned_sft_v2.sh',
    ]
    for wf in v2_workspace_files:
        raw = wf.read_bytes()
        norm = raw.replace(b'\r\n', b'\n')
        if not norm.endswith(b'\n'):
            norm += b'\n'
        if norm != raw:
            wf.write_bytes(norm)

    if STAGE_OUT.exists():
        shutil.rmtree(STAGE_OUT)
    STAGE_OUT.mkdir(parents=True, exist_ok=True)

    # 1. Stage code & third-party retained dependencies
    for wf in v2_workspace_files:
        rel = wf.relative_to(ROOT).as_posix()
        copy_exact(wf, STAGE_OUT / rel)

    retained_from_v1 = [
        'research/codeformer_conditioned_sft_learning_v1/__init__.py',
        'research/codeformer_conditioned_sft_learning_v1/conditioning.py',
        'research/codeformer_u64_cleananchors_continuation_v1/__init__.py',
        'research/codeformer_u64_cleananchors_continuation_v1/contract.py',
        'research/codeformer_u64_cleananchors_continuation_v1/metrics.py',
        'research/codeformer_u64_cleananchors_continuation_v1/model.py',
        'research/codeformer_u64_cleananchors_continuation_v1/objective.py',
        'research/codeformer_u96_postfusion_v1/__init__.py',
        'research/codeformer_u96_postfusion_v1/model.py',
        'research/codeformer_u96_spatial_feasibility_v1/__init__.py',
        'research/codeformer_u96_spatial_feasibility_v1/metrics.py',
        'research/codeformer_u96_spatial_feasibility_v1/worker.py',
        'third_party/__init__.py',
        'third_party/codeformer/LICENSE',
        'third_party/codeformer/NOTICE.md',
        'third_party/codeformer/__init__.py',
        'third_party/codeformer/codeformer_arch.py',
        'third_party/codeformer/registry.py',
        'third_party/codeformer/vqgan_arch.py',
        'weights/terminal_u0064_heads.npz',
    ]
    for rel in retained_from_v1:
        copy_exact(V1_STAGE / rel, STAGE_OUT / rel)

    # 2. Stage evidence files
    evidence_sources = {
        'v2_manifest': V2_PREP / 'v2_data_manifest.json',
        'v1_protocol': V1_STAGE / 'protocol.json',
        'retained_continuation': V1_STAGE / 'evidence/retained_continuation_protocol.json',
        'u64_parent_protocol': V1_STAGE / 'evidence/u64_parent_protocol.json',
        'u64_historical_pixel_audit': V1_STAGE / 'evidence/u64_historical_pixel_audit.json',
        'feasibility_v2_receipt': FEAS_V2,
    }
    evidence_bindings = {}
    for key, src_path in evidence_sources.items():
        rel_target, exp_bytes, exp_sha = contract.EVIDENCE_PINS[key]
        copy_exact(src_path, STAGE_OUT / rel_target)
        pinned = packet.pin(STAGE_OUT, STAGE_OUT / rel_target)
        assert (pinned['bytes'], pinned['sha256']) == (exp_bytes, exp_sha), f'Pin mismatch on {key}: {pinned}'
        evidence_bindings[key] = pinned

    # 3. Derive cohorts and stage all V2, V1 monitor, old_quality, controls, parity, and native CCTV assets
    m2_raw = (STAGE_OUT / 'evidence/v2_data_manifest.json').read_bytes()
    p1_raw = (STAGE_OUT / 'evidence/v1_protocol.json').read_bytes()
    p1_obj = packet.strict_json(p1_raw)
    cohorts = contract.select_v2_cohorts(m2_raw, p1_raw)

    # Stage V2 inputs & targets
    for bank in ('v2_optimization', 'v2_monitors'):
        for row in cohorts[bank]:
            for field in ('input', 'target'):
                rel_p = row[field]['path']
                assert rel_p.startswith('data/v2/')
                src = V2_PREP / rel_p[len('data/v2/'):]
                dst = STAGE_OUT / rel_p
                if not dst.exists():
                    copy_exact(src, dst)

    # Stage V1 monitors & old_quality inputs/targets from V1_STAGE
    def stage_from_v1(node):
        if type(node) is dict:
            if set(node) == {'path', 'bytes', 'sha256'}:
                dst = STAGE_OUT / node['path']
                if not dst.exists():
                    copy_exact(V1_STAGE / node['path'], dst)
            else:
                for child in node.values():
                    stage_from_v1(child)
        elif type(node) is list:
            for child in node:
                stage_from_v1(child)

    stage_from_v1(cohorts['v1_monitors'])
    stage_from_v1(cohorts['old_quality'])
    stage_from_v1(p1_obj['controls'])
    stage_from_v1(p1_obj['u64_parity'])

    # Stage 4 native CCTV DEV crops
    for rel_p, src_p in NATIVE_SOURCES.items():
        copy_exact(src_p, STAGE_OUT / rel_p)

    # 4. Build static files table (including pinned R2 and U96 weights to be copied by install_weights_v2.py on VM)
    files_table = {}
    for item in sorted(STAGE_OUT.rglob('*')):
        if item.is_dir():
            continue
        rel = item.relative_to(STAGE_OUT).as_posix()
        assert '__pycache__' not in rel and not rel.endswith(('.pyc', '.pyo'))
        p_info = packet.pin(STAGE_OUT, item)
        files_table[rel] = {'bytes': p_info['bytes'], 'sha256': p_info['sha256']}

    model_bindings = {
        'r2': {
            'path': 'weights/codeformer_r2.pth',
            'bytes': packet.MODEL_FIXED['r2'][0],
            'sha256': packet.MODEL_FIXED['r2'][1],
        },
        'u96': {
            'path': 'weights/codeformer_u96_adapter.pth',
            'bytes': packet.MODEL_FIXED['u96'][0],
            'sha256': packet.MODEL_FIXED['u96'][1],
        },
    }
    for m_val in model_bindings.values():
        files_table[m_val['path']] = {'bytes': m_val['bytes'], 'sha256': m_val['sha256']}

    protocol_obj = {
        'format': packet.FORMAT,
        'stage_name': packet.STAGE_NAME,
        'output': packet.OUTPUT_RELATIVE,
        'manual_launch_only': True,
        'automatic_retry_resume': False,
        'model_qualified': False,
        'training_success_not_implied': True,
        'model': model_bindings,
        'u64_head': packet.pin(STAGE_OUT, STAGE_OUT / 'weights/terminal_u0064_heads.npz'),
        'evidence': evidence_bindings,
        'comparison': contract.prospective_contract_v2(),
        'limits': contract.LIMITS,
        'selection': cohorts,
        'controls': p1_obj['controls'],
        'u64_parity': p1_obj['u64_parity'],
        'files': dict(sorted(files_table.items())),
    }

    protocol_bytes = (json.dumps(protocol_obj, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
    assert b'\r' not in protocol_bytes
    protocol_path = STAGE_OUT / 'protocol.json'
    protocol_path.write_bytes(protocol_bytes)
    protocol_sha = sha256_bytes(protocol_bytes)

    # Verify static stage in a clean subprocess where research resolves to STAGE_OUT
    import subprocess
    subprocess.run(
        [
            sys.executable, '-B', '-c',
            f"import sys; sys.path.insert(0, r'{STAGE_OUT}'); "
            f"from research.codeformer_conditioned_sft_learning_v2 import packet_v2; "
            f"packet_v2.verify_static(r'{STAGE_OUT}', '{protocol_sha}', missing_weights_for_install=True)",
        ],
        cwd=str(STAGE_OUT),
        check=True,
    )

    # 5. Build deterministic PAX execution archive (excluding R2 & U96 weights)
    if ARCHIVE_OUT.exists():
        ARCHIVE_OUT.unlink()
    if ARCHIVE_SHA_OUT.exists():
        ARCHIVE_SHA_OUT.unlink()

    staged_members = sorted(
        [p for p in STAGE_OUT.rglob('*') if p.is_file()],
        key=lambda p: p.relative_to(STAGE_OUT).as_posix(),
    )
    with ARCHIVE_OUT.open('wb') as raw_out:
        with gzip.GzipFile(filename='', fileobj=raw_out, mode='wb', compresslevel=6, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode='w|', format=tarfile.PAX_FORMAT) as tar:
                for src_file in staged_members:
                    rel = src_file.relative_to(STAGE_OUT).as_posix()
                    arcname = f'{packet.STAGE_NAME}/{rel}'
                    info = tarfile.TarInfo(name=arcname)
                    data = src_file.read_bytes()
                    info.size = len(data)
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = 'root'
                    info.gname = 'root'
                    info.mode = 0o755 if rel.endswith('.sh') else 0o644
                    import io
                    tar.addfile(info, io.BytesIO(data))
        raw_out.flush()
        os.fsync(raw_out.fileno())

    archive_bytes = ARCHIVE_OUT.stat().st_size
    archive_sha = packet.sha256(ARCHIVE_OUT)
    ARCHIVE_SHA_OUT.write_bytes(f'{archive_sha}  {ARCHIVE_OUT.name}\n'.encode('ascii'))

    # 6. Write build receipt & 5-step manual VM launch guide
    SOURCE_PREP_OUT.mkdir(parents=True, exist_ok=True)
    build_receipt = {
        'complete': True,
        'created_UTC': datetime.now(timezone.utc).isoformat(),
        'stage_name': packet.STAGE_NAME,
        'protocol_sha256': protocol_sha,
        'protocol_bytes': len(protocol_bytes),
        'archive_name': ARCHIVE_OUT.name,
        'archive_bytes': archive_bytes,
        'archive_sha256': archive_sha,
        'archive_pax_members': len(staged_members),
        'static_files_in_protocol': len(files_table),
        'v2_optimization_cases': len(cohorts['v2_optimization']),
        'v2_monitor_cases': len(cohorts['v2_monitors']),
        'v1_monitor_cases': len(cohorts['v1_monitors']),
        'old_quality_cases': len(cohorts['old_quality']),
        'native_cctv_dev_cases': len(cohorts['native_cctv_dev']),
        'neural_gradient_optimizer_training_calls': 0,
    }
    (SOURCE_PREP_OUT / 'build_receipt_v2.json').write_bytes(
        (json.dumps(build_receipt, indent=2, sort_keys=True) + '\n').encode('utf-8')
    )

    guide_md = f"""# Conditioned SFT V2 (Multi-Scale Dilated Pyramid + Frequency-Selective Head): Manual NVIDIA L4 VM Guide

**PREPARED / UNRUN.** Two-Phase high-capacity facial sharpening run (`Phase 1`: `40` epochs / `1,680` Exposure-Matched Clean-SFT distillation steps; `Phase 2`: `224` end-to-end Frozen-`R2` Perceptual + `SSIM7` + Multi-Scale Laplacian + Exposure-Split Gradient updates across `56` identities × `6` CCTV roles). All `575` trained-`u64` backbone tensors (`515` `R2` + `24` `U96` + `36` `u64` heads) remain frozen; step `0` reproduces `u64` bitwise.

- **Stage Directory**: `~/forensic-dgp/{packet.STAGE_NAME}`
- **Execution Upload**: `outputs/{ARCHIVE_OUT.name}` (**{archive_bytes:,} bytes**, `{len(staged_members)}` PAX members)
- **Protocol SHA256**: `{protocol_sha}`
- **Execution Archive SHA256**: `{archive_sha}`
- **Expected NVIDIA L4 Runtime**: `~11–14 minutes` (`7,200s` scientific cap)

---

## 1. Upload from Windows Google Cloud SDK Shell

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{ARCHIVE_OUT.name}" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{ARCHIVE_OUT.name}.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Verify and install weights in VM SSH

```bash
cd ~ &&
sha256sum -c {ARCHIVE_OUT.name}.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm/weights &&
rm -rf ~/forensic-dgp/{packet.STAGE_NAME} ~/{contract.PREFIX}-results.tar.gz ~/{contract.PREFIX}-results.tar.gz.sha256 ~/{contract.PREFIX}-export.json &&
tar -xzf {ARCHIVE_OUT.name} -C ~/forensic-dgp &&
cd ~/forensic-dgp/{packet.STAGE_NAME} &&
python3 -B research/codeformer_conditioned_sft_learning_v2/install_weights_v2.py --root . --source-root ~/forensic-dgp/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm --protocol-sha {protocol_sha}
```

## 3. Open tmux in VM SSH

```bash
tmux new-session -A -s codeformer_conditioned_sft_v2
```

## 4. Launch once inside tmux

```bash
cd ~/forensic-dgp/{packet.STAGE_NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/run_conditioned_sft_learning_v2.py --root . --protocol-sha {protocol_sha} --verify-transfer &&
bash research/codeformer_conditioned_sft_learning_v2/run_conditioned_sft_v2.sh {protocol_sha}
```

Detach with **Ctrl+B, then D**. Reattach with `tmux attach -t codeformer_conditioned_sft_v2`. To monitor live step-by-step progress (`journal.jsonl`) and worker/export logs from another VM SSH terminal:

```bash
tail -F ~/forensic-dgp/{packet.STAGE_NAME}/outputs/conditioned_sft_learning_v2/journal.jsonl ~/forensic-dgp/{packet.STAGE_NAME}/run.log ~/forensic-dgp/{packet.STAGE_NAME}/export.log
```

## 5. Download from Windows Google Cloud SDK Shell

Wait until `supervisor.json` / `export.log` prints `"complete": true` (exit code `0`). Use **one remote source per command**:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{contract.PREFIX}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{contract.PREFIX}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{contract.PREFIX}-export.json" "."
```
"""
    VM_GUIDE_OUT.write_bytes(guide_md.replace('\r\n', '\n').encode('utf-8'))
    print(json.dumps(build_receipt, indent=2))


if __name__ == '__main__':
    main()
