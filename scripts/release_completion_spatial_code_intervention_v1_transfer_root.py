"""Root release of verified, distinct manual-only zero-update transfer files."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os

ROOT=Path(__file__).resolve().parents[1]
DESIGN=ROOT/'research/completion_codeformer_spatial_code_intervention_v1_design'
INDEPENDENT=ROOT/'research/completion_codeformer_spatial_code_intervention_v1_independent_review'

def need(ok,message):
    if not ok:raise ValueError(message)
def physical(path):
    p=Path(path).resolve()
    return Path('\\\\?\\'+str(p)) if os.name=='nt' and not str(p).startswith('\\\\?\\') else p
def sha(path):
    h=hashlib.sha256()
    with physical(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def pin(path):return {'path':str(Path(path).resolve()),'bytes':physical(path).stat().st_size,'sha256':sha(path)}
def read(path):return json.loads(physical(path).read_text(encoding='utf-8-sig'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-id',required=True,choices=['draft_004'])
    parser.add_argument('--protocol-sha',required=True)
    parser.add_argument('--independent-source-admission',required=True,type=Path)
    parser.add_argument('--independent-source-sha',required=True)
    parser.add_argument('--independent-transport-review',required=True,type=Path)
    parser.add_argument('--independent-transport-sha',required=True)
    a=parser.parse_args();base=ROOT/'outputs/completion_codeformer_spatial_code_intervention_v1_packet'/a.stage_id
    stage=base/'cctv_completion_codeformer_spatial_code_intervention_v1_vm';pfile=stage/'protocol.json'
    need(sha(pfile)==a.protocol_sha,'Exact final protocol')
    p=read(pfile);need(p['format']=='completion-codeformer-spatial-code-intervention-v1' and p['manual_execution_only'] and p['diagnostic_only'] and not p['automatic_resume_or_promotion'],'Distinct manual-only diagnostic')
    need(len(p['files'])==166,'Exact reviewed166 payload scope')
    need(physical(ROOT/'research/completion_codeformer_spatial_code_intervention_v1/protocol.json').read_bytes()==physical(pfile).read_bytes(),'Current protocol pointer differs')
    for row in p['files']:
        q=stage/row['path'];need(physical(q).is_file() and not physical(q).is_symlink() and physical(q).stat().st_size==row['bytes'] and sha(q)==row['sha256'],'Staged source/data drift '+row['path'])
    actual={q.relative_to(physical(stage)).as_posix() for q in physical(stage).rglob('*') if q.is_file()}
    need(actual=={r['path'] for r in p['files']}|{'protocol.json'},'Closed regular staging namespace')
    need(sha(a.independent_source_admission)==a.independent_source_sha and sha(a.independent_transport_review)==a.independent_transport_sha,'Exact independent source/transport review pins')
    admission=read(a.independent_source_admission);transport=read(a.independent_transport_review)
    need(admission['complete'] is True and admission['independent'] is True and admission['protocol_sha256']==a.protocol_sha,'Final source/packet admission')
    need(transport['complete'] is True and transport['independent'] is True,'Final transport review')
    evidence={
        'root_design':(DESIGN/'root_design.json','1983f9fed9bede2078ffc6a1af0dd6ca08fe6ec532e5c3de9c4ce41d4db2c657'),
        'root_clarifications':(DESIGN/'pre_release_clarifications_v1.json','98c3521e5e7e0043479862923d03cc21c9b6b7c2e5371a0ffcd292d919ca1d78'),
        'root_installer_fixtures':(DESIGN/'root_installer_fixtures.json','610177441433cbb5f282c8264d35a0b63696e13b72cb4c3bcb627eb2c6b58e6e'),
        'independent_formal_design':(INDEPENDENT/'formal_design_admission.json','4f6ea7134e9c8628efb7deae4ca5dc137b35dc9f1d0471c796c8aa03692e63b5'),
        'independent_bridge_semantics':(INDEPENDENT/'independent_bridge_semantics.json','c1c022011c3c430753ccc0d600a0830ae1111a97de9d8e4ef28d4579f06121e1'),
        'independent_runtime_semantics':(INDEPENDENT/'independent_runtime_source_semantics.json','cb038d0e5746295eba36fe5313ee5a1006bc3634f8a5055523553ccb9df7b354'),
    }
    pins={}
    for key,(q,digest) in evidence.items():need(sha(q)==digest,'Design/verification drift '+key);pins[key]=pin(q)
    # External historical controls stay at their admitted source, not new uploads.
    controls=ROOT/'outputs/completion_codeformer_same_codebook_probe_v1_return/cctv_completion_codeformer_same_codebook_probe_v1_results/outputs'
    need(len(p['external_controls']['files'])==196,'All196 exact retained control files')
    for row in p['external_controls']['files']:
        q=controls/row['path'];need(physical(q).stat().st_size==row['bytes'] and sha(q)==row['sha256'],'Retained control drift '+row['path'])
    app=read(ROOT/'outputs/completion_codeformer_finetune_r1_app/source_admission.json')
    apppins=[*app['source_files'],app['candidate_manifest'],app['checkpoint']]
    for row in apppins:
        q=Path(row['path']);need(physical(q).stat().st_size==row['bytes'] and sha(q)==row['sha256'],'Working R1 model/app drift')
    need(len(apppins)==22,'All22 R1 app/model pins')
    release={'format':'completion-codeformer-spatial-code-intervention-v1-root-release-v1','created_UTC':datetime.now(timezone.utc).isoformat(),'author':pin(Path(__file__)),'released_for_manual_vm_transfer':True,'automatic_vm_launch':False,'protocol_sha256':a.protocol_sha,
        'independent_source_admission':pin(a.independent_source_admission),'independent_transport_review':pin(a.independent_transport_review),'evidence':pins,'staged_regular_members':167,'all166_payloads_root_rehashed':True,'all196_external_control_files_root_rehashed':True,'all22_R1_app_model_pins_exact':True,
        'payload_bytes_including_protocol':physical(pfile).stat().st_size+sum(r['bytes'] for r in p['files']),
        'scientific_choice':'Frozen497R1 geometric touchedH versus contextV oracle-code intervention;96newH/V+8ordinaryparity+4freshCbridge,108forwards,zerooptimizer/backward/teacher;oldA/C196files reused exactly,oraclesunavailablecleancontext/notdeployable.',
        'limits':{'worker_hard_seconds':900,'worker_cooperative_seconds':840,'maximum_new_optimizer_updates':0,'minimum_free_GiB':6,'stop_floor_GiB':2,'unexpected_failure_state_MiB':384,'output_GiB':1,'raw_tar_and_gzip_MiB_each':1280,'export_seconds':900},
        'whole_archive_and_published_guide_audit_still_required':True,'new_model_created':False,'actual_VM_preflight_or_execution':False,'neural_calls':0,'training_calls':0,'VM_calls':0,'model_qualified':False,'goal_complete':False}
    out=base/'root_manual_transfer_release.json'
    with physical(out).open('x',encoding='utf-8',newline='\n') as f:json.dump(release,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(pin(out)))

if __name__=='__main__':main()
