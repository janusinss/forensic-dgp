"""Add a verified local zero-reference reporter without changing the VM packet."""
from datetime import datetime,timezone
import re
from pathlib import Path

from import_cctv_dgp_appearance_input_response_v1_return import ROOT,OUT,NAME,STEM,PIN,read,sha,write
from publish_cctv_dgp_bank_saved_direction_forward_v1_r1_failure import link
from publish_cctv_dgp_png_group_gradients_v1_r2 import finish_prepend


def main():
    folder=OUT/'cctv_dgp_appearance_input_response_v1_checker_addendum'
    assert not (folder/'publication.json').exists()
    audit=read(folder/'audit.json');repair=read(folder/'repair.json');assert audit['complete'] and repair['complete']
    assert audit['VM_protocol_sha256']==PIN==sha(OUT/NAME/'protocol.json')
    assert audit['VM_archive_sha256']==sha(OUT/(STEM+'-execution.tar.gz'))
    original=ROOT/'scripts/audit_cctv_dgp_appearance_input_response_v1_return.py'
    corrected=ROOT/'scripts/audit_cctv_dgp_appearance_input_response_v1_return_r1.py'
    helper=ROOT/'scripts/cctv_dgp_RGB_gradient_parity_v1.py'
    assert sha(original)==repair['original_auditor_sha256'] and sha(corrected)==repair['replacement_auditor_sha256']
    assert sha(helper)==repair['zero_reference_reporting_helper_sha256']
    old=read(OUT/'cctv_dgp_appearance_input_response_v1_publication/publication.json')
    expected={'PROJECT_HANDOFF.md':old['documents']['PROJECT_HANDOFF.md']['after_sha256'],**old['new_documents_sha256']}
    (folder/'before').mkdir()
    prefix=f'''# Local return-checker addendum — verified, 14 October 2026

Use {link('prospective local return auditor R1','scripts/audit_cctv_dgp_appearance_input_response_v1_return_r1.py')}
after importing this diagnostic's eventual return. A synthetic zero-reference
fixture caught an undefined relative-gradient ratio in the original shared
reporter. R1 reports that ratio as null while retaining elementwise1e-4/1e-8
and norm1e-4/1e-10 bounds; nonzero-reference reports are exactly original.
Both old checker and RED failure remain. NaN/Inf inputs and genuine absolute
bound violations are still rejected. Saved five-case replay passes unchanged.
{link('Exact source change and verification','outputs/cctv_dgp_appearance_input_response_v1_checker_addendum/audit.json')}.

The VM archive,protocol,worker,200-query/40-forward limits,scientific image
gates and all five manual command blocks are unchanged. This is a local reporting
repair,not a training-recipe change or CUDA result. Diagnostic still UNRUN;
no model or more epochs qualified. Exact previous document follows.

'''
    bindings={n:finish_prepend(ROOT/n,prefix,folder/'before'/n,h) for n,h in expected.items()}
    guide=(ROOT/'CCTV_DGP_APPEARANCE_INPUT_RESPONSE_V1_VM.md').read_text(encoding='utf-8')
    assert re.findall(r'```(?:bat|bash)\n(.*?)\n```',guide,re.S)==list(old['exact_command_blocks'].values())
    for n,binding in bindings.items():
        before=Path(binding['backup']);assert sha(before)==binding['before_sha256']
        assert (ROOT/n).read_bytes()[binding['prefix_bytes']:]==before.read_bytes()
    protected=read(OUT/'cctv_dgp_conditioning_response_v1_analysis/local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    evidence={str(f.relative_to(ROOT).as_posix()):sha(f) for f in [folder/'repair.json',folder/'audit.json',original,corrected,helper,ROOT/'scripts/check_cctv_dgp_RGB_gradient_parity_v1.py']}
    write(folder/'publication.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),publisher_sha256=sha(__file__),
        protocol_sha256=PIN,archive_sha256=audit['VM_archive_sha256'],documents=bindings,evidence_sha256=evidence,
        original_checker_retained_unchanged=True,all_five_manual_command_blocks_unchanged=True,
        scientific_and_original_parity_bounds_unchanged=True,manual_diagnostic_unrun=True,
        local_neural_gradient_or_training_calls=0,model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,history_documents=3,VM_packet_and_manual_commands_unchanged=True,manual_UNRUN=True))


if __name__=='__main__':main()
