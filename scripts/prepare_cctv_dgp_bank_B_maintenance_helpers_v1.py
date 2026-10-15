"""Reuse the independently closed maintenance worker with an exact B-only ledger."""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT / 'scripts/cctv_dgp_bank_A_return_offload_v1_vm.py'
    assert sha(source) == '89dcbf2e7e3addf9cf23a4ec645cb2fb5d609b56faab5f7f939d28c0ee84ce9e'
    s = source.read_text(encoding='utf-8')
    s = s.replace('1858 backed-up A/baseline RGB outputs and two home archives',
                  '929 backed-up B baseline RGB outputs and three home archives')
    s = s.replace('cctv_dgp_bank_comparison_v1_r3_A_vm', 'cctv_dgp_bank_comparison_v1_r2_B_vm')
    s = s.replace('dgp_bank_A_return_offload_v1', 'dgp_bank_B_return_offload_v1')
    s = s.replace('dgp-bank-A-return-offload-v1', 'dgp-bank-B-return-offload-v1')
    s = s.replace('cctv-dgp-bank-comparison-v1-r3-A-results.tar.gz', 'cctv-dgp-bank-comparison-v1-r2-B-results.tar.gz')
    s = s.replace('cctv-dgp-bank-comparison-v1-execution.tar.gz', 'cctv-dgp-bank-comparison-v1-r2-B-execution.tar.gz')
    marker = "LEGACY_ARCHIVE=HOME/'cctv-dgp-bank-comparison-v1-r2-B-execution.tar.gz'"
    assert s.count(marker) == 1
    s = s.replace(marker, marker+"\nA3_ARCHIVE=HOME/'cctv-dgp-bank-comparison-v1-r3-A-execution.tar.gz'")
    s = s.replace('{ARCHIVE,LEGACY_ARCHIVE}', '{ARCHIVE,LEGACY_ARCHIVE,A3_ARCHIVE}')
    s = s.replace('1860', '932').replace("{'baseline','A50'}", "{'baseline'}")
    marker = "sum(Path(r['path']) in {ARCHIVE,LEGACY_ARCHIVE,A3_ARCHIVE} for r in p['candidates'])==2"
    assert s.count(marker) == 1
    s = s.replace(marker, marker[:-1]+'3')
    s = s.replace('inactive A/baseline RGB packs plus returned A and historical v1 execution home archives; full verified local backup',
        'inactive B baseline RGB packs plus returned B and superseded B2/A3 execution home archives; full verified local backup')
    worker = ROOT / 'scripts/cctv_dgp_bank_B_return_offload_v1_vm.py'
    assert not worker.exists(); ast.parse(s, feature_version=(3,10))
    worker.write_text(s, encoding='utf-8', newline='\n')
    source = ROOT / 'scripts/audit_cctv_dgp_bank_A_return_offload_v1.py'
    assert sha(source) == 'e384a177612dd9202b493b574bda03ce25d7670d2e3729047b8dfa5c684acbc8'
    s = source.read_text(encoding='utf-8').replace('bank_A_return_offload', 'bank_B_return_offload')
    s = s.replace('bank-A-return-offload', 'bank-B-return-offload')
    s = s.replace('cctv_dgp_bank_comparison_v1_r3_A_return', 'cctv_dgp_bank_comparison_v1_r2_B_return')
    s = s.replace('cctv_dgp_bank_comparison_v1_r3_A_vm', 'cctv_dgp_bank_comparison_v1_r2_B_vm').replace('1860', '932')
    checker = ROOT / 'scripts/audit_cctv_dgp_bank_B_return_offload_v1.py'
    assert not checker.exists(); ast.parse(s)
    checker.write_text(s, encoding='utf-8', newline='\n')
    print(dict(complete=True, exact_worker_count=932, home_archive_targets=3,
               no_VM_model_or_training_calls=True))


if __name__ == '__main__':
    main()
