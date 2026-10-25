"""Independent actual packet and guide readback; no builder/worker/VM execution."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'outputs/codeformer_u64_png_surrogate_measurement_v1_preparation'
STAGE_NAME='cctv_codeformer_u64_png_surrogate_measurement_v1_vm'
STAGE=ROOT/'outputs'/STAGE_NAME
PARENT=ROOT/'outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm'
PACKAGE='research/codeformer_u64_png_surrogate_measurement_v1'
PREFIX='cctv-codeformer-u64-png-surrogate-measurement-v1'
QUERIES=['raw_mae','raw_adjacent','png_surrogate_mae','png_surrogate_adjacent']
OWN_SOURCES=[PACKAGE+'/'+n for n in('__init__.py','contract.py','surrogate.py','analysis.py','runtime.py','worker.py','run_png_surrogate.sh')]
OWN_SOURCES+=['research/__init__.py','research/codeformer_u64_role_gradients_v1/__init__.py',
              'research/codeformer_u64_role_gradients_v1/contract.py',
              'scripts/run_u64_png_surrogate_measurement_v1.py','scripts/build_u64_png_surrogate_measurement_v1.py']
REPORTS=['CCTV_CODEFORMER_U64_GATE_ALIGNED_PILOT_V1_RETURN_REVIEW.md',
         'CCTV_CODEFORMER_U64_PNG_SURROGATE_MEASUREMENT_V1_DESIGN.md',
         'CCTV_CODEFORMER_U64_ROLE_GRADIENTS_V1_RETURN_REVIEW.md']
PARENT_PIN={'bytes':279879,'sha256':'4db752a04222210a9fbccf176cf422c5fadeef443d39a15ae7b62086e9c41419'}
STANDARD_PIN={'bytes':197079,'sha256':'a9b7899cb23ada04779f350dfd37f11818533bf9bf462404e342b8d7e575294e'}
HELPERS={'research/codeformer_u64_role_gradients_v1/__init__.py':(82,'b13cc93cf4d889fde3d255356e1c9e7e483aa845de75e2ea4a94a2c562554baf'),
 'research/codeformer_u64_role_gradients_v1/contract.py':(11189,'90a11da6be2e531648e4c5051471946469e287d2bc151177006554932395074e')}


class BlockScientific(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in{'numpy','PIL','scipy','torch','torchvision','tensorflow','jax','basicsr'}:
            raise ImportError('Scientific import blocked in packet audit: '+fullname)
        return None


sys.meta_path.insert(0,BlockScientific())
SPEC=importlib.util.spec_from_file_location('packet_framing_parser',ROOT/'scripts/audit_u64_role_gradients_return_transport_v1.py')
F=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(F)
need,strict_json=F.need,F.strict_json


def pin(path,base=ROOT):
    path=Path(path).absolute();base=Path(base).absolute();before=path.lstat()
    need(path.resolve().is_relative_to(base.resolve())and stat.S_ISREG(before.st_mode)and
         before.st_nlink==1 and not path.is_symlink()and not getattr(before,'st_file_attributes',0)&1024,'Ordinary confined file required')
    for p in path.parents:
        s=p.lstat();need(stat.S_ISDIR(s.st_mode)and not p.is_symlink()and not getattr(s,'st_file_attributes',0)&1024,'Ordinary ancestry required')
    h=hashlib.sha256()
    with path.open('rb')as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    after=path.stat();need((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'Source changed during hash')
    return {'path':path.relative_to(base).as_posix(),'bytes':after.st_size,'sha256':h.hexdigest()}


def payload(value):return {k:value[k]for k in('bytes','sha256')}


def staged_help():
    code="""import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,n,path=None,target=None):
  if n.split('.')[0] in {'numpy','PIL','scipy','torch','torchvision','tensorflow','jax','basicsr'}:raise ImportError('blocked scientific import')
  return None
sys.meta_path.insert(0,Block())
script=sys.argv[1];sys.argv=[script,'--help'];runpy.run_path(script,run_name='__main__')
"""
    result=subprocess.run([sys.executable,'-B','-c',code,str(STAGE/'scripts/run_u64_png_surrogate_measurement_v1.py')],
        cwd=STAGE,capture_output=True,text=True,timeout=10,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    need(result.returncode==0 and not result.stderr,'Staged inert help failed')
    need(all('--'+n in result.stdout for n in('root','protocol-sha','verify-transfer','status','supervise','run','export')),'Staged CLI options differ')
    need(not any(STAGE.rglob('__pycache__')),'Staged help wrote bytecode')
    return {'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'scientific_imports_blocked':True}


def audit_archive(archive_sha,protocol_sha):
    consumed={};checks=[]
    def checked(path):
        value=pin(path);consumed[value['path']]=value;return value
    def doc(path):checked(path);return strict_json(path.read_bytes())
    def check(name,ok,evidence=None):
        need(ok,name);checks.append({'name':name,'pass':True,'evidence':evidence})
    build=doc(PREP/'build_receipt_v1.json');praw=(STAGE/'protocol.json').read_bytes();p=doc(STAGE/'protocol.json')
    ap=ROOT/'outputs'/(PREFIX+'-execution.tar.gz');cp=ap.with_name(ap.name+'.sha256')
    archive,checksum=checked(ap),checked(cp);protocol=pin(STAGE/'protocol.json',STAGE)
    check('independently_supplied_archive_protocol_pins',archive==build['archive']and protocol==build['protocol']and
          archive['sha256']==archive_sha and protocol['sha256']==protocol_sha,{'archive':archive,'protocol':protocol})
    check('exact_checksum_digest_basename_two_spaces_LF',checksum==build['checksum']and
          cp.read_bytes()==(archive_sha+'  '+ap.name+'\n').encode('ascii'),checksum)
    expected={'protocol.json':protocol,**{name:{'path':name,**value}for name,value in p['files'].items()}}
    with ap.open('rb')as stream:parsed=F.stream_archive(stream,expected,STAGE_NAME)
    check('single_gzip_CRC_ISIZE_full_USTAR_namespace_and_all22payloads',parsed['framing']['compressed_sha256']==archive_sha and
          parsed['framing']['compressed_bytes']==archive['bytes']and parsed['framing']['regular_members']==22 and
          len(p['files'])==21,parsed['framing'])
    expected_names=set(OWN_SOURCES)|{'scripts/run_png_surrogate.sh'}
    expected_names|={'evidence/review_'+str(i)+'_'+name for i,name in enumerate(REPORTS)}
    expected_names|={'evidence/source_review.json','evidence/source_tests.json','evidence/parent_protocol.json',
                     'evidence/standard_protocol.json','evidence/local_parent_readback.json'}
    check('independently_enumerated_exact_static_names',set(p['files'])==expected_names,sorted(expected_names))
    dirs=set();actual={};expected_dirs={x.as_posix()for name in expected for x in PurePosixPath(name).parents if x.as_posix()!='.'}
    for current,subdirs,names in os.walk(STAGE,followlinks=False):
        for name in subdirs:
            path=Path(current)/name;s=path.lstat();need(stat.S_ISDIR(s.st_mode)and not path.is_symlink()and
                 not getattr(s,'st_file_attributes',0)&1024,'Linked stage directory refused');dirs.add(path.relative_to(STAGE).as_posix())
        for name in names:
            path=Path(current)/name;rel=F.confined(path.relative_to(STAGE).as_posix());actual[rel]=pin(path,STAGE);checked(path)
    check('stage_files_and_only_expected_dirs_match_archive',actual==expected and dirs==expected_dirs,{'files':len(actual),'dirs':sorted(dirs)})
    review=doc(STAGE/'evidence/source_review.json');tests=doc(STAGE/'evidence/source_tests.json')
    check('current_complete_source_review_tests_no_blockers',review['complete']is True and review['source_review_pass']is True and
          review['unresolved_blockers']==[]and tests['complete']is True and tests['all_tests_pass']is True and
          tests['neural_gradient_optimizer_training_VM_calls']==0,{'review':pin(STAGE/'evidence/source_review.json'),
          'tests':pin(STAGE/'evidence/source_tests.json'),'test_count':tests.get('test_count',tests.get('tests_run'))})
    tables=[{r['path']:r for r in rows}for rows in(build['sources'],review['sources'],tests['sources'])]
    check('all12_source_admission_tables_exact',all(len(t)==12 for t in tables)and tables[0]==tables[1]==tables[2]and
          set(tables[0])==set(OWN_SOURCES),tables[0])
    for name in OWN_SOURCES:
        check('current_and_staged_source_'+name,checked(ROOT/name)==tables[0][name]==pin(STAGE/name,STAGE),tables[0][name])
        if name.endswith('.py'):ast.parse((STAGE/name).read_text(),feature_version=(3,10))
    for name,(size,digest)in HELPERS.items():check('exact_retained_helper_'+name,p['files'][name]=={'bytes':size,'sha256':digest})
    check('shell_alias_exact_reviewed_bytes',p['files']['scripts/run_png_surrogate.sh']==p['files'][PACKAGE+'/run_png_surrogate.sh'])
    for index,name in enumerate(REPORTS):
        target='evidence/review_'+str(index)+'_'+name
        check('current_copied_report_'+name,payload(checked(ROOT/name))==p['files'][target],checked(ROOT/name))
    parentraw=(STAGE/'evidence/parent_protocol.json').read_bytes();standardraw=(STAGE/'evidence/standard_protocol.json').read_bytes()
    parent,standard=strict_json(parentraw),strict_json(standardraw)
    check('exact_copied_parent_and_standard_protocols',p['files']['evidence/parent_protocol.json']==PARENT_PIN and
          p['files']['evidence/standard_protocol.json']==STANDARD_PIN and (PARENT/'protocol.json').read_bytes()==parentraw and
          (PARENT/parent['parent_protocol']['path']).read_bytes()==standardraw)
    checked(PARENT/'protocol.json');checked(PARENT/parent['parent_protocol']['path'])
    # Authentic stdlib-only contract import from the actual staged package.
    sys.path.insert(0,str(STAGE));contract=importlib.import_module('research.codeformer_u64_png_surrogate_measurement_v1.contract')
    for name in('research','research.codeformer_u64_png_surrogate_measurement_v1',
                'research.codeformer_u64_png_surrogate_measurement_v1.contract','research.codeformer_u64_role_gradients_v1.contract'):
        need(Path(sys.modules[name].__file__).resolve().is_relative_to(STAGE),'Wrong staged inert contract origin')
    selection=contract.select_cases(parentraw,standardraw);plan=contract.operation_plan(selection);diagnostic=contract.prospective_contract()
    check('staged_stdlib_contract_recomputes_exact_protocol',p['selection']==selection and p['plan']==plan and
          p['diagnostic']==diagnostic and p['limits']==contract.LIMITS)
    expected_plan=[]
    for phase in('historical_u64_parity','deterministic_same_u64_repeat'):
        for row in selection['monitors'][:4]:expected_plan.append((row,phase,[]))
    expected_plan.extend((row,'paired_TRAIN_gradient_diagnostic',QUERIES)for row in selection['optimization'])
    for slot,(source,phase,queries)in enumerate(expected_plan,1):
        role=source['role'];expected_row={'slot':slot,'id':source['id'],'reference_id':source['reference_id'],
            'role':role,'phase':phase,'metric_region':'central128'if role=='local_gaussian_1p2'else'whole','gradient_queries':queries}
        need(plan[slot-1]==expected_row,'Independently reconstructed plan differs')
    check('independent72forward256query328operation_plan',len(plan)==72 and sum(len(r['gradient_queries'])for r in plan)==256 and
          [r['reference_id']for r in selection['optimization'][::4]]==standard['learning']['optimization_reference_order'],
          {'forwards':72,'queries':256,'operation_pairs':328,'parity':4,'repeat':4,'optimization':64})
    zero_keys=('backward_attempts','backward_returns','optimizer_constructions','optimizer_update_attempts','optimizer_updates',
               'parameter_application_attempts','parameter_applications')
    check('zero_update_and16_pooled_surrogate_boundaries',all(p['limits'][k]==0 for k in zero_keys)and diagnostic['pooled_vectors']==16 and
          diagnostic['no_eight_vector_solver_on_16']is True and diagnostic['surrogate']['actual_PNG_derivative']is False and
          diagnostic['surrogate']['biased_estimator']is True and diagnostic['all_original_quality_gates_unchanged']is True and
          diagnostic['initial_and_closing575_arrays_unchanged_required']is True,{'zero_keys':zero_keys,'pooled_vectors':16,'true_PNG_derivative_claim':False})
    limits=p['limits'];check('finite_worker_export_resource_caps',
        [limits[k]for k in('worker_seconds','external_seconds','termination_grace_seconds','per_phase_seconds','export_seconds','export_external_seconds')]==[900,960,20,180,300,330]and
        limits['initial_free_disk_bytes']==8*1024**3 and limits['continuing_free_disk_bytes']==4*1024**3 and
        limits['maximum_RSS_bytes']==10*1024**3 and limits['maximum_CUDA_reserved_bytes']==18*1024**3 and
        limits['maximum_output_bytes']==limits['maximum_archive_bytes']==2*1024**3,limits)
    proof=doc(STAGE/'evidence/local_parent_readback.json');bindings={r['parent_relative']:r['actual_local']for r in proof['files']}
    check('parent469_readback_namespace_exact',len(parent['files'])==len(bindings)==469 and set(bindings)==set(parent['files']))
    parentrows=[];aliases=[]
    for name,value in sorted(parent['files'].items()):
        F.confined(name);source=PARENT/name
        if not source.is_file():
            need(name in{'weights/codeformer_r2.pth','weights/codeformer_u96_adapter.pth'},'Unexpected local parent alias')
            source=ROOT/F.confined(bindings[name]['path']);aliases.append({'parent_relative':name,'actual_local':bindings[name]})
        actual=checked(source);need(payload(actual)==value and actual==bindings[name],'Actual parent binding differs')
        parentrows.append({'parent_relative':name,'actual_local':actual})
    check('all469_parent_files_rehashed_2_model_only_local_aliases',len(aliases)==2 and
          proof['VM_requires_all469_at_actual_parent_stage']is True and proof['VM_inventory_performed']is False,
          {'rehashed':469,'direct':467,'local_model_aliases':aliases,'VM_substitution_permitted':False})
    check('manual_fresh_stage_no_promotion',p['manual_launch_only']is True and p['automatic_retry_resume']is False and
          p['model_qualified']is False and p['original_u64_is_retained']is True and p['app_incumbent_unchanged']is True and
          build['actual_parameter_applications']==0)
    help_value=staged_help();check('actual_staged_help_inert_exit0',help_value['exit_code']==0,help_value)
    for path,value in list(consumed.items()):need(pin(ROOT/path)==value,'Consumed bytes changed during archive audit')
    need({n:pin(STAGE/n,STAGE)for n in expected}==expected,'Stage changed after staged help')
    result={'complete':True,'archive_review_pass':True,'unresolved_blockers':[],'created_UTC':datetime.now(timezone.utc).isoformat(),
        'archive':archive,'checksum':checksum,'protocol':protocol,'stage_protocol':pin(STAGE/'protocol.json'),
        'build_receipt':pin(PREP/'build_receipt_v1.json'),'checks_pass':len(checks),'checks':checks,'framing':parsed['framing'],
        'actual_archive_file_pins':parsed['files'],'all12_sources_admitted_exact':list(tables[0].values()),
        'parent_files_rehashed':parentrows,'local_parent_model_aliases':aliases,'staged_help':help_value,
        'consumed_files':list(consumed.values()),'all_consumed_pins_unchanged_after':True,'source':pin(Path(__file__)),
        'runtime_real_VM_verify_transfer_pass_claimed':False,'runtime_static_scope':'All469actualVMparentpaths remain mandatory; two local model aliases are hash provenance only. No remote inventory/static admission performed.',
        'scientific_imports_blocked':True,'actual_worker_model_neural_parameter_optimizer_VM_browser_calls':0,
        'manual_guide_approved_here':False,'model_qualified':False}
    target=PREP/'archive_readback_v1.json'
    with target.open('x',encoding='utf-8',newline='\n')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'receipt':pin(target),'checks_pass':len(checks),'archive':archive,'protocol':protocol,'framing':parsed['framing'],'unresolved_blockers':[]},indent=2))


def audit_guide():
    path=ROOT/'CCTV_CODEFORMER_U64_PNG_SURROGATE_MEASUREMENT_V1_VM.md'
    raw=path.read_bytes();text=raw.decode('utf-8');flat=' '.join(text.replace('**','').replace('`','').split());compact=re.sub(r'\s+','',flat)
    archive_review=strict_json((PREP/'archive_readback_v1.json').read_bytes())
    need(archive_review['archive_review_pass']is True and archive_review['unresolved_blockers']==[],'Actual archive approval required')
    build=strict_json((PREP/'build_receipt_v1.json').read_bytes());sha=build['protocol']['sha256'];archive=ROOT/build['archive']['path']
    need(pin(archive)==build['archive']and pin(ROOT/build['checksum']['path'])==build['checksum']and pin(STAGE/'protocol.json',STAGE)==build['protocol'],'Guide archive/protocol consumed pins changed')
    checks=[]
    def check(name,ok,evidence=None):need(ok,name);checks.append({'name':name,'pass':True,'evidence':evidence})
    check('LF_only_guide_matches_actual_archive_protocol',b'\r'not in raw and sha in text and build['archive']['sha256']in text and
          (str(build['archive']['bytes'])in compact or format(build['archive']['bytes'],',')in compact),{'guide':pin(path),'archive':build['archive'],'protocol':build['protocol']})
    check('exactly5_numbered_steps',len(re.findall(r'^##\s+[1-5][. ]',text,re.M))==5)
    commands=[line.strip()for line in text.splitlines()if line.strip().startswith('gcloud compute scp ')]
    need(len(commands)==5,'Exactly2upload3download gcloud commands required')
    remote='janusdominic0@forensic-dgp-thesis';home='/home/janusdominic0/'
    names=[PREFIX+'-execution.tar.gz',PREFIX+'-execution.tar.gz.sha256',PREFIX+'-results.tar.gz',PREFIX+'-results.tar.gz.sha256',PREFIX+'-export.json']
    for index,command in enumerate(commands):
        tokens=shlex.split(command);need(tokens[:3]==['gcloud','compute','scp']and
              '--project=forensic-dgp-thesis'in tokens and '--zone=us-central1-a'in tokens,'Exact confirmed gcloud target required')
        operands=[x for x in tokens[3:]if not x.startswith('--')]
        expected=[names[index],remote+':'+home]if index<2 else[remote+':'+home+names[index],'.']
        need(operands==expected,'Exactly one remote source/destination per transfer command')
    check('5_gcloud_commands_exact_target_names_and_single_source',True,commands)
    check('install_checksum_parent_venv_fresh_stage_guards_and_tar_destination',all(s in compact for s in[
        'sha256sum-c'+PREFIX+'-execution.tar.gz.sha256',
        'test-d~/forensic-dgp/cctv_dgp_vm_bundle/.venv','test-d~/forensic-dgp/'+PARENT.name,
        'test!-e~/forensic-dgp/'+STAGE_NAME,'tar-xzf'+PREFIX+'-execution.tar.gz-C~/forensic-dgp']))
    bridge='scripts/run_u64_png_surrogate_measurement_v1.py --root . --protocol-sha '+sha+' --verify-transfer'
    check('exact_verify_transfer_bridge_and_protocol_SHA',flat.count(bridge)>=2 and '--verify-transfer'in archive_review['staged_help']['stdout'])
    check('manual_tmux_session_and_exact_new_stage_launch',('tmuxnew-session-A-spng_surrogate_measurement_v1'in compact or
          'tmuxnew-session-A-su64_png_surrogate_measurement_v1'in compact)and 'cd~/forensic-dgp/'+STAGE_NAME in compact and
          'source~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate'in compact and
          'bashscripts/run_png_surrogate.sh'+sha in compact)
    check('monitor_exact_new_output_journal','~/forensic-dgp/'+STAGE_NAME+'/outputs/u64_png_surrogate_measurement_v1/events.jsonl'in compact)
    check('72_256_575_and_zero_update_surrogate_boundaries',all(s in compact for s in['72','256','575'])and
          'surrogate'in text.lower()and ('biased'in text.lower())and ('zero updates'in flat.lower()or'zero-updates'in flat.lower())and
          ('not the true derivative'in flat.lower()or'not a true'in flat.lower()or'not the actual'in flat.lower()))
    check('900_960_20_180_and300_330_caps',all(str(n)in text for n in[900,960,20,180,300,330]))
    check('8_4_10_18_2GiB_resources_disclosed',all(str(n)in text for n in[8,4,10,18,2])and 'GiB'in text)
    check('no_retry_resume_and_failure_retention',all(s in text.lower()for s in['retry','resume','preserve'])and
          ('unknown'in text.lower())and ('cleanup'in text.lower()))
    check('status_model_and_original_failure_boundaries','U96'in text and 'original'in text.lower()and 'FINAL'in text and
          ('ACTIVE/incomplete'in text)and ('unpaired'in text.lower())and ('qualif'in text.lower()))
    result={'complete':True,'guide_review_pass':True,'manual_guide_ready_at_exact_consumed_pins':True,'unresolved_blockers':[],
        'checks_pass':len(checks),'checks':checks,'guide':pin(path),'archive_review':pin(PREP/'archive_readback_v1.json'),
        'archive':build['archive'],'checksum':build['checksum'],'protocol':build['protocol'],'source':pin(Path(__file__)),
        'actual_worker_model_neural_parameter_optimizer_VM_browser_calls':0,'model_qualified':False,
        'scope':'Actual byte-bound transfer/CLI/stage/tmux/resource/status guide readback only; no command in the guide executed.'}
    target=PREP/'guide_readback_v1.json'
    with target.open('x',encoding='utf-8',newline='\n')as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'receipt':pin(target),'checks_pass':len(checks),'unresolved_blockers':[]},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--archive-sha');parser.add_argument('--protocol-sha');parser.add_argument('--guide',action='store_true');args=parser.parse_args()
    if args.guide:audit_guide()
    else:
        need(args.archive_sha and args.protocol_sha,'Independently supplied sealed archive and protocol hashes required')
        audit_archive(args.archive_sha,args.protocol_sha)
