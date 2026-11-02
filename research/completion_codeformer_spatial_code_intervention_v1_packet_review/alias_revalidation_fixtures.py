"""Two actual post-tar retained-alias race fixtures, using tiny owned dummy files."""
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import contextlib, hashlib, importlib.util, io, json

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
FIXTURES=Path(r'C:\xampp\htdocs\YEAR 4\Testing\outputs\spcode_race_v1')

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
builder=load('alias_builder',ROOT/'scripts/prepare_completion_codeformer_spatial_code_intervention_v1.py')
exporter=load('alias_exporter',ROOT/'scripts/export_completion_codeformer_spatial_code_intervention_v1.py')

def sha_bytes(value):return hashlib.sha256(value).hexdigest()

def run(mode):
    home=FIXTURES/mode;home.mkdir();root=home/exporter.VM_NAME;root.mkdir();out=root/'outputs';out.mkdir()
    source=b'fixed-toy-source';(root/'source.txt').write_bytes(source)
    p={'format':builder.FORMAT,'manual_execution_only':True,'automatic_resume_or_promotion':False,'contract':{'output_bytes':1024**3,'stop_free_bytes':2*1024**3},'parity_cases':[{'id':'parity'}],'bridge_cases':[{'id':'bridge'}],'arms':[{'id':'H','filename_tag':'ORACLE_H_mask_touched_same_codebook'},{'id':'V','filename_tag':'ORACLE_V_visible_context_same_codebook'}],'monitor_cases':[{'id':'monitor'}],'challenge_cases':[{'id':'challenge'}],'files':[{'path':'source.txt','bytes':len(source),'sha256':sha_bytes(source)}],'export':{}}
    p['export']['allowed_output_paths']=builder.allowed_output_paths(p);body=(json.dumps(p)+'\n').encode();(root/'protocol.json').write_bytes(body)
    (root/'supervisor.json').write_text(json.dumps({'complete':True,'worker_exit':1,'timeout':False}));(root/'trainer.log').write_bytes(b'fixture\n');(out/'failure.json').write_text(json.dumps({'complete':False,'dummy_fixture':True}))
    state=out/'failed_actual_model_state.pth';alias=Path(str(state)+'.partial');dummy=b'NOT-A-CHECKPOINT actual bounded dummy alias fixture'
    state.write_bytes(dummy)
    import os
    os.link(state,alias)
    archive=home/(exporter.STEM+'-results.tar.gz');partial_archive=Path(str(archive)+'.partial')
    moved=home/'retained_original_alias_fixture.bin';changed=[];original_sha=exporter.sha
    def mutate_after_tar_hash(path,guard=None):
        value=original_sha(path,guard)
        if Path(path)==partial_archive and not changed:
            assert alias.resolve().is_relative_to(home.resolve()) and moved.resolve().is_relative_to(home.resolve()) and not moved.exists()
            original_inode=alias.stat().st_ino;alias.rename(moved)
            if mode=='same_bytes_new_inode':alias.write_bytes(dummy)
            changed.append({'archive_already_streamed_and_hashed':True,'old_inode':original_inode,'new_inode':alias.stat().st_ino if alias.exists() else None,'same_bytes_if_present':alias.read_bytes()==dummy if alias.exists() else None,'source_original_alias_preserved':str(moved)})
        return value
    error=None
    with patch.object(exporter,'sha',side_effect=mutate_after_tar_hash),patch.object(exporter.shutil,'disk_usage',return_value=SimpleNamespace(free=100*1024**3)),contextlib.redirect_stdout(io.StringIO()):
        try:exporter.export(root,sha_bytes(body),home,enforce_vm=False)
        except Exception as exc:error=repr(exc)
    assert error and len(changed)==1 and not archive.exists() and partial_archive.exists()
    assert not Path(str(archive)+'.sha256').exists() and not (home/(exporter.STEM+'-export.json')).exists()
    failure=json.loads((home/(exporter.STEM+'-export-failure.json')).read_text())
    assert failure['complete'] is False and failure['partial_retained'] is True and moved.read_bytes()==state.read_bytes()==dummy
    return {'case':mode,'passed':True,'actual_post_tar_mutation':changed[0],'error':error,'success_archive_and_receipt_absent':True,'bounded_archive_partial_retained':builder.pin(partial_archive),'source_payload_bytes_preserved':True,'dummy_file_never_deserialized':True}

def main():
    FIXTURES.mkdir(exist_ok=False);rows=[run(mode) for mode in ('same_bytes_new_inode','alias_path_removed')]
    report={'format':'completion-spatial-intervention-v1-author-post-tar-alias-revalidation-fixtures-v1','complete':True,'cases':rows,'checks':2,'exporter':builder.pin(ROOT/'scripts/export_completion_codeformer_spatial_code_intervention_v1.py'),'all_pass':True,'real_checkpoint_reads':0,'neural_calls':0,'VM_calls':0}
    path=OUT/'alias_revalidation_fixtures.json'
    with path.open('x',encoding='utf-8',newline='\n') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(builder.pin(path)))

if __name__=='__main__':main()
