"""Repair export namespaces in a fresh packet; retain the original bytes."""
import ast
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tarfile

from import_cctv_dgp_joint_raw_png_forward_v1_return import ROOT, OUT, ANALYSIS, read, sha, write


def remap(text):
    return text.replace('live_conditioning_v1','live_conditioning_v1_r1').replace(
        'cctv-dgp-bank-live-conditioning-v1','cctv-dgp-live-conditioning-v1').replace(
        'cctv-dgp-live-conditioning-v1','cctv-dgp-live-conditioning-v1-r1')


def constant(text,name):
    values=[n.value.value for n in ast.parse(text).body if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id==name for t in n.targets) and isinstance(n.value,ast.Constant)]
    assert len(values)==1;return values[0]


def main():
    original=OUT/'cctv_dgp_live_conditioning_v1_vm'
    name='cctv_dgp_live_conditioning_v1_r1_vm';stem='cctv-dgp-live-conditioning-v1-r1'
    packet=OUT/name;archive=OUT/(stem+'-execution.tar.gz')
    receipt=OUT/'cctv_dgp_live_conditioning_v1_r1_preparation.json'
    assert not any(p.exists() for p in [packet,archive,receipt])
    previous=read(OUT/'cctv_dgp_live_conditioning_v1_preparation.json')
    old=read(original/'protocol.json')
    assert previous['complete'] and previous['unrun'] and sha(original/'protocol.json')==previous['protocol_sha256']
    assert sha(OUT/'cctv-dgp-live-conditioning-v1-execution.tar.gz')==previous['archive_sha256']
    assert all(sha(original/n)==h for n,h in old['assets_sha256'].items())
    assert not (original/'outputs').exists() and not (original/'review.log').exists()
    packet.mkdir();(packet/'scripts').mkdir();changed={}
    for filename,digest in old['assets_sha256'].items():
        source=original/filename;newname=filename.replace('live_conditioning_v1','live_conditioning_v1_r1')
        target=packet/newname
        if source.suffix in ['.py','.sh']:
            before=source.read_text(encoding='utf-8');text=remap(before)
            if source.suffix=='.py':ast.parse(text,feature_version=(3,10))
            with target.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
            changed[newname]=dict(original_file=filename,original_sha256=digest,repaired_sha256=sha(target),
                exact_namespace_only_rewrite=True)
        else:shutil.copyfile(source,target);assert sha(target)==digest
    worker=packet/'scripts/cctv_dgp_live_conditioning_v1_r1_vm.py'
    assert constant(worker.read_text(encoding='utf-8'),'NAME')==name
    assert constant(worker.read_text(encoding='utf-8'),'STEM')==stem
    assert constant((packet/'cctv_dgp_live_conditioning_v1_r1_model.py').read_text(encoding='utf-8'),'NAME')==name
    p=dict(old);p.update(name=name,UTC=datetime.now(timezone.utc).isoformat(),
        namespace_only_repair=True,scientific_architecture_and_limits_unchanged=True,
        original_prepared_protocol_sha256=previous['protocol_sha256'],
        original_misnamed_export_stem='cctv-dgp-bank-live-conditioning-v1',correct_export_stem=stem,
        original_archive_sha256=previous['archive_sha256'],
        original_preparation_sha256=sha(OUT/'cctv_dgp_live_conditioning_v1_preparation.json'),
        original_prepared_packet_retained=True,original_human_VM_execution_unknown=True,
        unchanged_initial_architecture_state=old['initial_model_state'])
    p['assets_sha256']={f.relative_to(packet).as_posix():sha(f) for f in packet.rglob('*') if f.is_file()}
    assert len(p['assets_sha256'])==7
    write(packet/'protocol.json',p);pin=sha(packet/'protocol.json')
    importer_path=ROOT/'scripts/import_cctv_dgp_live_conditioning_v1_r1_return.py'
    importer=remap((ROOT/'scripts/import_cctv_dgp_live_conditioning_v1_return.py').read_text(encoding='utf-8')).replace(previous['protocol_sha256'],pin)
    assert constant(importer,'NAME')==name and constant(importer,'STEM')==stem and constant(importer,'PIN')==pin
    ast.parse(importer,feature_version=(3,10))
    with importer_path.open('x',encoding='utf-8',newline='\n') as f:f.write(importer)
    auditor_path=ROOT/'scripts/audit_cctv_dgp_live_conditioning_v1_r1_return.py'
    auditor=remap((ROOT/'scripts/audit_cctv_dgp_live_conditioning_v1_return.py').read_text(encoding='utf-8'))
    ast.parse(auditor,feature_version=(3,10))
    with auditor_path.open('x',encoding='utf-8',newline='\n') as f:f.write(auditor)
    for base in ['audit_cctv_dgp_live_conditioning_v1_preparation','audit_cctv_dgp_live_conditioning_v1_gate_fixtures']:
        source=ROOT/'scripts'/(base+'.py');target=ROOT/'scripts'/(base.replace('live_conditioning_v1','live_conditioning_v1_r1')+'.py')
        text=remap(source.read_text(encoding='utf-8'))
        ast.parse(text,feature_version=(3,10))
        with target.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
        for file in sorted(f for f in packet.rglob('*') if f.is_file()):tar.add(file,arcname=name+'/'+file.relative_to(packet).as_posix(),recursive=False)
    digest=sha(archive)
    with Path(str(archive)+'.sha256').open('x',encoding='ascii',newline='\n') as f:f.write(digest+'  '+archive.name+'\n')
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),preparer_sha256=sha(__file__),
        protocol_sha256=pin,archive_sha256=digest,archive_bytes=archive.stat().st_size,unrun=True,
        training_launch_ready=False,namespace_only_repair=True,architecture_unchanged=True,
        virtual_initial_state_sha256=old['initial_model_state'],original_packet_protocol_sha256=previous['protocol_sha256'],
        changes=changed,worker_and_importer_export_stem_agree=True,source_block_namespace_fingerprints_checked=True,
        future_importer_sha256=sha(importer_path),future_auditor_sha256=sha(auditor_path),
        no_local_neural_gradient_or_optimizer_calls=True,no_VM_connection_upload_or_launch=True,
        model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,protocol_sha256=pin,archive_sha256=digest,bytes=archive.stat().st_size,
        namespace_only_repair=True,original_packet_retained=True))


if __name__=='__main__':main()
