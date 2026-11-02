"""Root full archive, guide, 9-scenario installer, import-blocked transfer CLI, and 22-pin R1 audit for Unified R2 v1."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import ast
import gzip
import hashlib
import io
import json
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/completion_codeformer_unified_r2_v1_packet/draft_001'
NAME = 'cctv_completion_codeformer_unified_r2_v1_vm'
STEM = 'cctv-completion-codeformer-unified-r2-v1'
EXPECTED = '60549f671b5be8e260460e3787ec460dd5cd3aa404a09f3eefc5cf9e6c9b3347'
ARCHIVE = '83ee54a8a29b9a92bdb26ee1533957638a9c9a2dfa4065895354f690ca59f672'
GUIDE = '5f3d839ec17357ffc9bf0ccb8c1faeec601bafe489dcf2cd40c970c485891ab2'
TRANSFER = Path(r'C:\xampp\htdocs\YEAR 4\Testing\outputs')


def need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def pin(p: Path):
    return {'path': str(p.resolve()), 'bytes': p.stat().st_size, 'sha256': sha(p)}


def read(p: Path):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def run_installer_fixtures(literal: str):
    checks = []
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        for scenario in (
            'valid',
            'already_installed',
            'wrong_protocol_SHA',
            'missing_payload',
            'payload_hash_drift',
            'duplicate_payload',
            'symlink_payload',
            'unsafe_member',
            'insufficient_payload_plus8GiB',
        ):
            home = base / scenario
            home.mkdir()
            (home / 'forensic-dgp').mkdir()
            data = b'toy regular payload only'
            row = {'path': 'assets/toy.bin', 'bytes': len(data), 'sha256': digest(data)}
            p = {'format': 'completion-codeformer-unified-r2-v1', 'files': [row]}
            body = (json.dumps(p) + '\n').encode('utf-8')
            protocol_sha = digest(body)
            archive = home / (STEM + '-execution.tar.gz')
            with tarfile.open(archive, 'w:gz') as tar:
                m = tarfile.TarInfo(NAME + '/protocol.json')
                m.size = len(body)
                tar.addfile(m, io.BytesIO(body))
                if scenario != 'missing_payload':
                    m = tarfile.TarInfo(NAME + '/assets/toy.bin')
                    m.size = len(data)
                    if scenario == 'symlink_payload':
                        m.type = tarfile.SYMTYPE
                        m.linkname = '../escape'
                        m.size = 0
                        tar.addfile(m)
                    else:
                        tar.addfile(
                            m,
                            io.BytesIO(
                                b'x' * len(data) if scenario == 'payload_hash_drift' else data
                            ),
                        )
                if scenario == 'duplicate_payload':
                    m = tarfile.TarInfo(NAME + '/assets/toy.bin')
                    m.size = len(data)
                    tar.addfile(m, io.BytesIO(data))
                if scenario == 'unsafe_member':
                    m = tarfile.TarInfo(NAME + '/../escape')
                    m.size = 1
                    tar.addfile(m, io.BytesIO(b'x'))
            before = pin(archive)
            destination = home / 'forensic-dgp' / NAME
            if scenario == 'already_installed':
                destination.mkdir()
                (destination / 'owned.marker').write_bytes(b'preserve existing')
            expected = '0' * 64 if scenario == 'wrong_protocol_SHA' else protocol_sha
            code = literal.replace('__PROTOCOL_SHA__', expected)
            error = None
            free = 8 * 1024 ** 3 if scenario == 'insufficient_payload_plus8GiB' else 20 * 1024 ** 3
            try:
                with patch.object(Path, 'home', return_value=home), patch(
                    'shutil.disk_usage', return_value=type('Disk', (), {'free': free})()
                ):
                    exec(compile(code, '<exact-toy-installer>', 'exec'), {})
            except (ValueError, FileExistsError, tarfile.TarError) as exc:
                error = repr(exc)
            need((error is None) == (scenario == 'valid'), 'Wrong installer verdict ' + scenario)
            need(pin(archive) == before, 'Installer changed source archive')
            if scenario == 'valid':
                need(
                    (destination / 'assets/toy.bin').read_bytes() == data
                    and (destination / 'protocol.json').read_bytes() == body,
                    'Successful toy install not exact',
                )
            elif scenario == 'already_installed':
                need(
                    (destination / 'owned.marker').read_bytes() == b'preserve existing',
                    'Existing target changed',
                )
            else:
                need(not destination.exists(), 'Refused fixture published a target')
            need(not (home / 'forensic-dgp/escape').exists(), 'Archive path escaped')
            checks.append(
                {
                    'fixture': scenario,
                    'expected_accept': scenario == 'valid',
                    'accepted': error is None,
                    'error': error,
                }
            )
    return checks


def run_import_blocked_transfer_cli(stage: Path):
    code = f"""
import importlib.abc, json, runpy, sys
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {{'torch', 'numpy', 'PIL', 'cv2', 'scipy', 'torchvision'}}:
            raise ImportError('Blocked scientific import: ' + fullname)
        return None
sys.meta_path.insert(0, Blocker())
sys.argv = [
    r'{stage / "scripts/cctv_completion_codeformer_unified_r2_v1_vm.py"}',
    '--root', r'{stage}',
    '--protocol-sha', '{EXPECTED}',
    '--verify-transfer',
]
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
except SystemExit as exc:
    if exc.code != 0:
        raise RuntimeError(f'Non-zero exit: {{exc.code}}')
"""
    output = subprocess.check_output([sys.executable, '-B', '-c', code], text=True, timeout=60).strip()
    parsed = json.loads(output)
    need(
        parsed.get('complete') is True
        and parsed.get('files') == 652
        and parsed.get('neural_calls') == 0
        and parsed.get('optimizer_updates') == 0,
        'Import-blocked --verify-transfer output mismatch',
    )
    return parsed


def main():
    archive = BASE / (STEM + '-execution.tar.gz')
    stage = BASE / NAME
    need(archive.stat().st_size == 73376283 and sha(archive) == ARCHIVE, 'Archive identity')
    ppath = stage / 'protocol.json'
    need(sha(ppath) == EXPECTED, 'Protocol identity')
    p = read(ppath)
    expected = {NAME + '/protocol.json': {'bytes': ppath.stat().st_size, 'sha256': EXPECTED}}
    expected.update({NAME + '/' + r['path']: r for r in p['files']})
    seen = []
    total = 0
    with archive.open('rb') as raw, gzip.GzipFile(fileobj=raw) as gz:
        with tarfile.open(fileobj=gz, mode='r|') as tar:
            for m in tar:
                need(
                    m.isreg() and m.name in expected and m.name not in seen,
                    'Unsafe/extra/duplicate tar member',
                )
                need(seen or m.name == NAME + '/protocol.json', 'First member must be protocol')
                row = expected[m.name]
                need(m.size == row['bytes'], 'Tar member size')
                h = hashlib.sha256()
                n = 0
                with tar.extractfile(m) as f:
                    for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
                        h.update(b)
                        n += len(b)
                need(n == row['bytes'] and h.hexdigest() == row['sha256'], 'Tar payload SHA')
                seen.append(m.name)
                total += n
        while gz.read(8 * 1024 * 1024):
            pass
    need(
        set(seen) == set(expected) and len(seen) == 653 and total == 77190642,
        'Closed archive scope',
    )
    checksum = (ARCHIVE + '  ' + archive.name + '\n').encode('ascii')
    need((BASE / (archive.name + '.sha256')).read_bytes() == checksum, 'Exact LF checksum')
    need(
        sha(TRANSFER / archive.name) == ARCHIVE
        and (TRANSFER / (archive.name + '.sha256')).read_bytes() == checksum,
        'Published transfer drift',
    )
    guides = [
        ROOT / 'COMPLETION_CODEFORMER_UNIFIED_R2_V1_VM.md',
        TRANSFER / 'COMPLETION_CODEFORMER_UNIFIED_R2_V1_VM.md',
    ]
    need(all(q.stat().st_size == 6518 and sha(q) == GUIDE for q in guides), 'Published guide identity')
    body = guides[0].read_text(encoding='utf-8')
    need(ARCHIVE in body and EXPECTED in body, 'Guide artifact bindings')
    scp = [line for line in body.splitlines() if line.startswith('gcloud compute scp ')]
    need(len(scp) == 5, 'Expected two uploads / three single-source downloads')
    remotes = []
    for line in scp:
        tokens = shlex.split(line)
        need(tokens[:3] == ['gcloud', 'compute', 'scp'], 'SCP command prefix')
        need(
            '--project=forensic-dgp-thesis' in tokens and '--zone=us-central1-a' in tokens,
            'SCP VM project/zone',
        )
        remote = [v for v in tokens if v.startswith('janusdominic0@forensic-dgp-thesis:')]
        need(len(remote) == 1, 'PuTTY one-remote-source contract')
        remotes += remote
    need(
        remotes[-3:]
        == [
            'janusdominic0@forensic-dgp-thesis:/home/janusdominic0/' + STEM + x
            for x in ('-results.tar.gz', '-results.tar.gz.sha256', '-export.json')
        ],
        'Exact return filenames',
    )
    need('tmux new-session -A -s completion_unified_r2_v1' in body, 'Tmux session')
    need(
        'bash scripts/run_completion_codeformer_unified_r2_v1.sh ' + EXPECTED in body,
        'Exact manual launch',
    )
    install = body.split("python3 - <<'PY'\n", 1)[1].split('\nPY', 1)[0]
    ast.parse(install, feature_version=(3, 10))
    builder = stage / 'scripts/prepare_completion_codeformer_unified_r2_v1.py'
    tree = ast.parse(builder.read_text(encoding='utf-8-sig'))
    literal = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'INSTALL_CODE' for t in n.targets)
    )
    need(
        install == literal.replace('__PROTOCOL_SHA__', EXPECTED).rstrip('\n'),
        'Guide installer differs from tested literal',
    )
    installer_fixtures = run_installer_fixtures(literal)
    import_blocked_cli = run_import_blocked_transfer_cli(stage)
    for rel in (
        'scripts/cctv_completion_codeformer_unified_r2_v1_vm.py',
        'scripts/export_completion_codeformer_unified_r2_v1.py',
        'research/completion_codeformer_unified_r2_v1/contracts.py',
        'research/completion_codeformer_unified_r2_v1/unified_objectives.py',
        'research/completion_codeformer_unified_r2_v1/optimized_budget.py',
        'research/completion_codeformer_unified_r2_v1/terminal_admission.py',
    ):
        ast.parse((stage / rel).read_text(encoding='utf-8-sig'), feature_version=(3, 10))

    admission = read(ROOT / 'outputs/completion_codeformer_finetune_r1_app/source_admission.json')
    r1 = admission['source_files'] + [admission['candidate_manifest'], admission['checkpoint']]
    for r in r1:
        q = Path(r['path'])
        need(q.stat().st_size == r['bytes'] and sha(q) == r['sha256'], 'R1 source/model drift')
    need(len(r1) == 22, 'Exact R1 assets')

    receipt = {
        'format': 'root-unified-r2-v1-full-archive-and-published-guide-audit-v1',
        'complete': True,
        'created_UTC': datetime.now(timezone.utc).isoformat(),
        'author': pin(Path(__file__)),
        'archive': pin(archive),
        'protocol': pin(ppath),
        'archive_members': 653,
        'pinned_payload_files': 652,
        'uncompressed_payload_bytes': total,
        'all_archive_members_fullstream_hashes_pass': True,
        'whole_gzip_CRC_consumed': True,
        'exact_LF_checksums_and_published_copy': True,
        'guides': [pin(q) for q in guides],
        'SCP_commands_checked': scp,
        'each_download_single_remote_source': True,
        'installer_exact_tested_literal_and_Python310_parse': True,
        'installer_9_scenarios': installer_fixtures,
        'import_blocked_transfer_CLI_verified': import_blocked_cli,
        'manual_tmux_protocol_bound': True,
        'all22_R1_model_app_pins_exact': True,
        'neural_calls': 0,
        'training_steps': 0,
        'VM_calls': 0,
    }
    out = BASE / 'root_final_archive_audit.json'
    out.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(pin(out)))


if __name__ == '__main__':
    main()
