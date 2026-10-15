"""Create two-archive maintenance helpers; never remove scientific packet assets."""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def checked_replace(source, old, new):
    assert old in source, old
    return source.replace(old, new)


def main():
    source = ROOT / 'scripts/cctv_dgp_bank_B_return_offload_v1_vm.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == 'a3e15d97c8a6caa75526c70c44226e2dcd864c768dccee6cc87a486fa799c481'
    text = source.read_text(encoding='utf-8')
    text = checked_replace(text, '929 backed-up B baseline RGB outputs and three home archives', 'two backed-up B3 home archive copies')
    text = checked_replace(text, 'cctv_dgp_bank_comparison_v1_r2_B_vm', 'cctv_dgp_bank_comparison_v1_r3_B_vm')
    text = checked_replace(text, 'dgp_bank_B_return_offload_v1', 'dgp_bank_B3_archive_cleanup_v1')
    text = checked_replace(text, 'dgp-bank-B-return-offload-v1', 'dgp-bank-B3-archive-cleanup-v1')
    text = checked_replace(text, 'cctv-dgp-bank-comparison-v1-r2-B-', 'cctv-dgp-bank-comparison-v1-r3-B-')
    text = checked_replace(text, "A3_ARCHIVE=HOME/'cctv-dgp-bank-comparison-v1-r3-A-execution.tar.gz'\n", '')
    text = checked_replace(text, '{ARCHIVE,LEGACY_ARCHIVE,A3_ARCHIVE}', '{ARCHIVE,LEGACY_ARCHIVE}')
    text = checked_replace(text, '==932', '==2')
    text = checked_replace(text, "sum(Path(r['path']) in {ARCHIVE,LEGACY_ARCHIVE} for r in p['candidates'])==3", "sum(Path(r['path']) in {ARCHIVE,LEGACY_ARCHIVE} for r in p['candidates'])==2")
    text = checked_replace(text,
        'inactive B baseline RGB packs plus returned B and superseded B2/A3 execution home archives; full verified local backup',
        'two completed or superseded B3 home archives only; full verified local backups; all packet research assets retained')
    tree = ast.parse(text)
    allowed, = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'allowed']
    lines = text.splitlines(keepends=True)
    lines[allowed.lineno-1:allowed.end_lineno] = ["""def allowed(path):
    assert path.resolve()==path and path in {ARCHIVE,LEGACY_ARCHIVE}
    assert not any(parent.is_symlink() for parent in path.parents)
"""]
    text = ''.join(lines)
    ast.parse(text, feature_version=(3,10))
    worker = ROOT / 'scripts/cctv_dgp_bank_B3_archive_cleanup_v1_vm.py'
    assert not worker.exists()
    with worker.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
    checker_source = ROOT / 'scripts/audit_cctv_dgp_bank_B_return_offload_v1.py'
    assert hashlib.sha256(checker_source.read_bytes()).hexdigest() == 'de20a152dee875690111ee0d42bfe664b76205320d5ba38435805c7ed9e159df'
    text = checker_source.read_text(encoding='utf-8')
    for old, new in [('bank_B_return_offload', 'bank_B3_archive_cleanup'),
        ('bank-B-return-offload', 'bank-B3-archive-cleanup'),
        ('cctv_dgp_bank_comparison_v1_r2_B_return', 'cctv_dgp_bank_comparison_v1_r3_B_return'),
        ('cctv_dgp_bank_comparison_v1_r2_B_vm', 'cctv_dgp_bank_comparison_v1_r3_B_vm')]:
        text = checked_replace(text, old, new)
    text = checked_replace(text, '==932', '==2')
    text = checked_replace(text, 'exact_deletions_checked=932', 'exact_deletions_checked=2')
    text = checked_replace(text, 'complete_local_backup_hashes_rechecked=932', 'complete_local_backup_hashes_rechecked=2')
    checker = ROOT / 'scripts/audit_cctv_dgp_bank_B3_archive_cleanup_v1.py'
    assert not checker.exists()
    ast.parse(text)
    with checker.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
    print(dict(complete=True, archive_targets=2, research_file_deletions=0,
               no_VM_or_model_calls=True), flush=True)


if __name__ == '__main__':
    main()
