"""Independent archive/source/scope and pre-Torch launch-boundary checks."""
from pathlib import Path
import ast
import hashlib
import json
import subprocess
import sys
import tarfile
import time
from cctv_dgp_actual_step_tail_v1_vm import NAME, PARENT, STEM, read, write, sha, verify_transfer

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'outputs'/NAME
PARENT_BUNDLE=ROOT/'outputs'/PARENT
OUT=ROOT/'outputs/cctv_dgp_actual_step_tail_v1_preparation'


def main():
    start=time.monotonic()
    assert not (OUT/'independent_packet_audit.json').exists()
    p=read(BUNDLE/'protocol.json');pin=sha(BUNDLE/'protocol.json')
    prepared=read(OUT/'preparation.json')
    assert pin==prepared['protocol_sha256']
    archive=ROOT/'outputs'/(STEM+'-execution.tar.gz')
    assert sha(archive)==prepared['archive_sha256'] and archive.stat().st_size==prepared['archive_bytes']
    assert Path(str(archive)+'.sha256').read_text().split()==[sha(archive),archive.name]
    expected={NAME+'/protocol.json':pin,**{NAME+'/'+name:digest for name,digest in p['assets_sha256'].items()}}
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();assert len(members)==4 and {m.name for m in members}==set(expected)
        for member in members:
            assert member.isfile() and not member.issym() and not member.islnk() and member.size<=1024**2
            content=tar.extractfile(member).read();assert hashlib.sha256(content).hexdigest()==expected[member.name]
    for name in ['cctv_dgp_actual_step_tail_v1_vm.py','audit_cctv_dgp_actual_step_tail_v1_return.py']:
        assert sha(BUNDLE/'scripts'/name)==sha(ROOT/'scripts'/name)
        ast.parse((BUNDLE/'scripts'/name).read_text(),feature_version=(3,10))
    tail_p,parent_p,adapted=verify_transfer(BUNDLE,PARENT_BUNDLE,pin)
    assert len(tail_p['review_source_changes'])==8
    # No scientific formula, metric, model, proposal or optimizer statement is
    # among these eight source edits. Inspect exact changed statements here.
    before_statements=[r['before'] for r in tail_p['review_source_changes']]
    assert before_statements==["vm_scope(root);out=root/'outputs'",'Need6GiB free after installation',
        "for probe_index,probe in enumerate(p['probes']):","folder=root/output_prefix(probe['update'],proposal)",
        "'of':3150",'projected=elapsed*10*BUDGETS',"'remaining_probe_states':9",
        "assert progress['forward_slots']==3150 and progress['completed_conditions']==30"]
    assert len(tail_p['overlap_files_sha256'])==828
    original_import=read(ROOT/'outputs/cctv_dgp_actual_step_review_v1_return_import.json')
    assert tail_p['overlap_files_sha256']=={name:digest for name,digest in original_import['files_sha256'].items() if name.startswith('outputs/probes/update0045/')}
    assert tail_p['forward_slots']==315 and not tail_p['new_trained_checkpoint'] and not tail_p['app_promotion']
    driver=str(ROOT/'scripts/cctv_dgp_actual_step_tail_v1_vm.py')
    guards=[]
    for flag in ['--review','--export','--supervise']:
        arguments=[driver,'--root',str(BUNDLE),'--parent-root',str(PARENT_BUNDLE),'--protocol-sha',pin,flag]
        code="import sys,runpy\nsys.argv="+repr(arguments)+"\ntry:\n runpy.run_path(sys.argv[0],run_name='__main__')\nexcept AssertionError as e:\n assert 'Existing Linux VM only' in str(e)\n assert 'torch' not in sys.modules\n print('Windows launch rejected before Torch')\nelse:\n raise AssertionError('Local neural launch accepted')\n"
        result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,timeout=30)
        assert result.returncode==0,result.stderr
        guards.append({'flag':flag,'exit_code':0,'before_Torch':True,'stdout':result.stdout.strip()})
    # Shape-only finite storage ceiling: zero includes two RGB arrays; the
    # other two proposals include one. ZIP/NPY overhead is budgeted generously.
    raw_files=105*(2*256*256*3*4+16384)+210*(256*256*3*4+16384)
    PNG_files=630*256*1024
    ceiling=raw_files+PNG_files+16*1024**2
    assert ceiling<tail_p['budgets']['maximum_output_bytes']-16*1024**2
    app=read(ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_audit_recovery_r1/app_preservation_readback.json')
    assert app['complete'] and app['app_bindings_verified']==14
    for name,digest in app['files_sha256'].items():assert sha(ROOT/name)==digest,name
    write(OUT/'independent_packet_audit.json',{'complete':True,'protocol_sha256':pin,'execution_archive_sha256':sha(archive),
        'all4_archive_members_checked':True,'all3_packet_assets_checked':True,'original_parent_assets_checked':len(parent_p['assets_sha256']),
        'source_inverse_verified':True,'eight_source_changes_are_scope_storage_timing_only':True,'overlap_files_bound':828,
        'Windows_pre_Torch_stops':guards,'Python310_syntax_pass':True,'maximum_return_storage_ceiling_bytes':ceiling,
        'cap_bytes':tail_p['budgets']['maximum_output_bytes'],'app_bindings_verified':14,
        'model_forwards':0,'optimizer_updates':0,'gradient_queries':0,'VM_calls':0,'new_trained_checkpoint':False,
        'original_failure_retained':True,'app_promotion':False,'goal_complete':False,
        'seconds':time.monotonic()-start,'checker_sha256':sha(Path(__file__))})
    print({'complete':True,'members':4,'scope_slots':315,'Windows_launch_boundaries':3,'storage_ceiling_MiB':ceiling/1024**2,'model_forwards':0,'seconds':time.monotonic()-start},flush=True)


if __name__=='__main__':main()
