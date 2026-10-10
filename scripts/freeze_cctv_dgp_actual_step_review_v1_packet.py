"""Freeze a separate inference-only packet; do not launch any VM job."""
from pathlib import Path
import ast
import shutil
import tarfile
import time
from cctv_dgp_actual_step_review_v1_contract import NAME,STEM,read,write,sha,verify

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'outputs'/NAME


def main():
    start=time.monotonic();assert not (BUNDLE/'protocol.json').exists()
    p=read(BUNDLE/'protocol.draft.json')
    for name,digest in p['source_evidence_sha256'].items():
        assert sha(ROOT/name)==digest,name
        assert time.monotonic()-start<300
    excluded={'scripts/cctv_dgp_pcgrad_fit_v41_vm.py','cctv_dgp_pcgrad_v41.py','cctv_dgp_pcgrad_v41_evidence.py'}
    assets={name:digest for name,digest in p['copied_assets_sha256'].items() if name not in excluded}
    modules={'cctv_dgp_actual_step_review_v1_contract.py':'scripts/cctv_dgp_actual_step_review_v1_contract.py',
             'cctv_dgp_actual_step_review_v1_metrics.py':'scripts/cctv_dgp_actual_step_review_v1_metrics.py',
             'scripts/review_cctv_dgp_actual_steps_v1_vm.py':'scripts/review_cctv_dgp_actual_steps_v1_vm.py',
             'scripts/supervise_cctv_dgp_actual_steps_v1.py':'scripts/supervise_cctv_dgp_actual_steps_v1.py',
             'scripts/run_actual_step_review.sh':'scripts/run_cctv_dgp_actual_step_review_v1.sh'}
    for destination,source in modules.items():
        path=BUNDLE/destination;path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
        data=(ROOT/source).read_bytes()
        if path.suffix=='.py':ast.parse(data.decode('utf-8'),feature_version=(3,10))
        if path.suffix=='.sh':assert b'\r' not in data
        with path.open('xb') as stream:stream.write(data)
        assets[destination]=sha(path);p['source_evidence_sha256'][source]=sha(ROOT/source)
    for path in (BUNDLE/'evidence').rglob('*'):
        if path.is_file():assets[path.relative_to(BUNDLE).as_posix()]=sha(path)
    p['assets_sha256']=assets
    p['excluded_unused_historical_training_files']=sorted(excluded)
    p['return_raw_storage']='Every slot retains compressed float32 RGB and four embedding vectors; zero proposals also retain the original DGP RGB for independent raw-objective audit.'
    p['prospective_audit']={'CPU_current_batch_replay_cases':150,'raw_replay_maximum_error':1e-5,'PNG_replay_maximum_byte_difference':1,
       'embedding_replay_maximum_error':1e-4,'saved_pixel_metric_absolute_error':1e-10,
       'derived_group_and_comparison_absolute_error':1e-10,'categorical_preservation_decisions_exact':True,
       'raw_objective_term_absolute_error':[5e-5,5e-5,1e-5,1e-5,1e-5,5e-4,1e-5],
       'independent_audit_seconds':2400,'numeric_raw_objective_allowances_do_not_change_delivered_image_gates':True}
    for receipt in ['fixed_filter_arithmetic.json','contract_regressions_before_freeze.json','Bash_readonly_syntax.json']:
        assert read(ROOT/'outputs/cctv_dgp_actual_step_review_v1_preparation'/receipt)['complete']
    for source in ['scripts/audit_cctv_dgp_actual_step_review_v1_return.py','scripts/freeze_cctv_dgp_actual_step_review_v1_packet.py',
                   'scripts/verify_cctv_dgp_actual_step_review_v1_packet.py','scripts/validate_cctv_dgp_actual_step_review_v1_arithmetic.py',
                   'outputs/cctv_dgp_actual_step_review_v1_preparation/fixed_filter_arithmetic.json',
                   'outputs/cctv_dgp_actual_step_review_v1_preparation/contract_regressions_before_freeze.json',
                   'outputs/cctv_dgp_actual_step_review_v1_preparation/Bash_readonly_syntax.json']:
        p['source_evidence_sha256'][source]=sha(ROOT/source)
    p['primary_method_sources']=['https://docs.pytorch.org/docs/2.9/generated/torch.no_grad.html',
       'https://docs.pytorch.org/docs/2.9/generated/torch.load.html','https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html']
    write(BUNDLE/'protocol.json',p);pin=sha(BUNDLE/'protocol.json');verify(BUNDLE,pin)
    archive=ROOT/'outputs'/(STEM+'-execution.tar.gz');assert not archive.exists()
    with tarfile.open(archive,'w:gz',compresslevel=3) as tar:
        for name in sorted(set(assets)|{'protocol.json'}):
            assert time.monotonic()-start<300
            tar.add(BUNDLE/name,arcname=NAME+'/'+name,recursive=False)
    digest=sha(archive)
    assert time.monotonic()-start<300
    with Path(str(archive)+'.sha256').open('x',encoding='utf-8',newline='\n') as stream:stream.write(digest+'  '+archive.name+'\n')
    write(ROOT/'outputs/cctv_dgp_actual_step_review_v1_packet_preparation.json',{'complete':True,'protocol_sha256':pin,
         'archive_sha256':digest,'archive_bytes':archive.stat().st_size,'asset_files':len(assets),'cases':145,'references':29,
         'transient_proposals':30,'review_forward_slots':3150,'optimizer_updates':0,'gradient_queries':0,'new_trained_checkpoint':False,
         'VM_calls':0,'neural_calls_here':0,'training_started':False,'app_promotion':False,'seconds':time.monotonic()-start,'cap_seconds':300})
    print({'complete':True,'protocol_sha256':pin,'archive_sha256':digest,'bytes':archive.stat().st_size,'files':len(assets),
           'VM_calls':0,'optimizer_updates':0,'seconds':time.monotonic()-start},flush=True)


if __name__=='__main__':main()
