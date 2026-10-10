"""Retain the bookkeeping failure and change only the prior handoff lookup."""
from pathlib import Path
import ast
import subprocess
import sys
from cctv_dgp_actual_step_review_v1_contract import read,write,sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/cctv_dgp_actual_step_review_v1_prepared_milestone/closure_recovery_r1'
OLD=ROOT/'scripts/verify_cctv_dgp_actual_step_review_v1_prepared_milestone.py'
NEW=OLD.with_name('verify_cctv_dgp_actual_step_review_v1_prepared_milestone_r1.py')


def main():
    assert not OUT.exists() and not NEW.exists();m=read(OUT.parent/'milestone.json')
    assert sha(OLD)==m['new_evidence_sha256'][OLD.relative_to(ROOT).as_posix()]
    OUT.mkdir();(OUT/'original_checker.py').write_bytes(OLD.read_bytes())
    result=subprocess.run([sys.executable,'-B',str(OLD)],cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode!=0 and "KeyError: 'PROJECT_HANDOFF.md'" in result.stderr
    (OUT/'original_failure.log').write_text(result.stdout+result.stderr,encoding='utf-8',newline='\n')
    text=OLD.read_text();changes=[
      ("assert parent['new_evidence_sha256']['PROJECT_HANDOFF.md']==sha(ROOT/m['previous_handoff_path'])",
       "assert parent['document']['after_sha256']==closed['additional_closure_bindings_sha256']['PROJECT_HANDOFF.md']==sha(ROOT/m['previous_handoff_path'])"),
      ("write(OUT/'independent_closure_audit.json'","write(OUT/'independent_closure_audit_r1.json'"),
      ("'new_bindings_verified':len(m['new_evidence_sha256'])",
       "'closure_failure_retained':True,'recovery_only_prior_document_hash_lookup':True,'additional_closure_bindings_sha256':{path.relative_to(ROOT).as_posix():sha(path) for path in (OUT/'closure_recovery_r1').rglob('*') if path.is_file()},'new_bindings_verified':len(m['new_evidence_sha256'])")]
    corrected=text
    for before,after in changes:assert corrected.count(before)==1;corrected=corrected.replace(before,after,1)
    inverse=corrected
    for before,after in reversed(changes):assert inverse.count(after)==1;inverse=inverse.replace(after,before,1)
    assert inverse==text;ast.parse(corrected,feature_version=(3,10))
    with NEW.open('x',encoding='utf-8',newline='\n') as stream:stream.write(corrected)
    write(OUT/'preparation.json',{'complete':True,'original_checker_sha256':sha(OLD),'corrected_checker_sha256':sha(NEW),
          'original_exit_code':result.returncode,'original_traceback_retained':True,'inverse_source_changes_exact':True,
          'only_document_hash_lookup_and_receipt_metadata_changed':True,'packet_or_protocol_or_model_or_gate_changed':False,
          'helper_sha256':sha(Path(__file__)),'VM_calls':0,'optimizer_updates':0})
    print({'complete':True,'original_failure_retained':True,'VM_calls':0},flush=True)


if __name__=='__main__':main()
