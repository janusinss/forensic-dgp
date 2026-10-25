"""Independent exact manual-guide readback. Never execute guide commands.

Only the staged --help path runs, in a child with scientific imports blocked.
Archive/dependency byte checks join the independent archive review receipt.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'outputs/codeformer_u64_png_checked_step_v1_preparation'
STAGE=ROOT/'outputs/cctv_codeformer_u64_png_checked_step_v1_vm'
PREFIX='cctv-codeformer-u64-png-checked-step-v1'
PARENT='cctv_codeformer_u64_cleananchors_continuation_v1_vm'
MEASUREMENT='cctv_codeformer_u64_png_surrogate_measurement_v1_vm'
GUIDE=ROOT/'CCTV_CODEFORMER_U64_PNG_CHECKED_STEP_V1_VM.md'
PROTOCOL_SHA='7f4546d4d911bea7100f42870514f08a42e349ec524a2b9795f8f3c7f55dcf60'
ARCHIVE_SHA='15ef783122ea90259aa8fc7fd86c912cb88bf3a425b2c85a1a86a41d0db126e7'
PACKAGE='research/codeformer_u64_png_checked_step_v1'
SPEC=importlib.util.spec_from_file_location('pure_guide_file_pins',ROOT/'scripts/audit_u64_role_gradients_return_transport_v1.py')
F=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(F)
need,pin,strict_json=F.need,F.pin,F.strict_json


def main():
    consumed={};checks=[]
    def checked(path):
        value=pin(path);key=value['path']
        need(key not in consumed or consumed[key]==value,'File changed during guide review')
        consumed[key]=value;return value
    def text(path):checked(path);return path.read_text(encoding='utf-8')
    def doc(path):return strict_json(text(path).encode('utf-8'))
    def check(name,ok,evidence=None):checks.append({'name':name,'pass':bool(ok),'evidence':evidence})
    guide=text(GUIDE);flat=' '.join(guide.replace('**','').replace('`','').split());compact=re.sub(r'\s+','',flat)
    a=doc(PREP/'archive_readback_v1.json');b=doc(PREP/'build_receipt_v1.json');p=doc(STAGE/'protocol.json')
    archive=checked(ROOT/b['archive']['path']);checksum=checked(ROOT/b['checksum']['path'])
    protocol=checked(STAGE/'protocol.json');limits=p['limits'];pilot=p['pilot']
    check('complete_independent_archive_review63checks_no_blockers',a['complete'] is True and a['archive_review_pass'] is True and
        a['unresolved_blockers']==[] and a['checks_pass']==63 and all(v['pass'] for v in a['checks']) and
        a['all_consumed_pins_unchanged_after'] and not a['builder_imported_or_executed'],checked(PREP/'archive_readback_v1.json'))
    check('exact798435_archive_and382027_protocol_joined_to_reviews',archive==a['archive']==b['archive'] and
        archive['bytes']==798435 and archive['sha256']==ARCHIVE_SHA and protocol['bytes']==382027 and
        protocol['sha256']==PROTOCOL_SHA and a['protocol']==b['protocol']=={'path':'protocol.json','bytes':382027,'sha256':PROTOCOL_SHA} and
        '798,435 bytes' in flat and ARCHIVE_SHA in guide and PROTOCOL_SHA in guide,
        {'archive':archive,'protocol':protocol})
    check('LF_checksum_exact_digest_two_spaces_basename',checksum==a['checksum']==b['checksum'] and
        (ROOT/checksum['path']).read_bytes()==(ARCHIVE_SHA+'  '+PREFIX+'-execution.tar.gz\n').encode('ascii'),checksum)
    # Current staged sources are rehashed; no helper, worker, or neural source
    # is imported while reviewing the instructions and its sealed CLI.
    sources=[]
    for binding in a['all16_sources_admitted_exact']:
        current=checked(STAGE/binding['path'])
        sources.append(current)
        need({k:current[k] for k in ('bytes','sha256')}=={k:binding[k] for k in ('bytes','sha256')},'Current staged approved source differs')
    check('all16_current_staged_sources_still_match_archive_admission',len(sources)==16,sources)
    shell=text(STAGE/'scripts/run_png_checked.sh');runtime=text(STAGE/PACKAGE/'runtime.py')
    worker=text(STAGE/PACKAGE/'worker.py');bridge=text(STAGE/'scripts/run_u64_png_checked_step_v1.py')
    check('five_steps_exact_one_through_five',re.findall(r'^## (\d+)\.',guide,re.M)==['1','2','3','4','5'])
    commands=[line.strip() for line in guide.splitlines() if line.strip().startswith('gcloud compute scp ')]
    remote='janusdominic0@forensic-dgp-thesis:/home/janusdominic0/'
    expected=[[PREFIX+'-execution.tar.gz',remote],[PREFIX+'-execution.tar.gz.sha256',remote],
        [remote+PREFIX+'-results.tar.gz','.'],[remote+PREFIX+'-results.tar.gz.sha256','.'],[remote+PREFIX+'-export.json','.']]
    parsed=[shlex.split(command) for command in commands]
    check('two_upload_three_single_source_download_commands_exact_target',len(parsed)==5 and all(
        row[:5]==['gcloud','compute','scp','--project=forensic-dgp-thesis','--zone=us-central1-a'] and row[5:]==want
        for row,want in zip(parsed,expected)),commands)
    check('both_transfer_sections_exact_Windows_output_folder',guide.count('cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"')==2)
    install='cd~&&sha256sum-c'+PREFIX+'-execution.tar.gz.sha256&&test-d~/forensic-dgp/cctv_dgp_vm_bundle/.venv&&test-d~/forensic-dgp/'+PARENT+'&&test-d~/forensic-dgp/'+MEASUREMENT+'&&test!-e~/forensic-dgp/'+STAGE.name+'&&tar-xzf'+PREFIX+'-execution.tar.gz-C~/forensic-dgp'
    check('install_chain_checksum_venv_both_dependencies_fresh_stage_before_tar',install in compact)
    check('partial_or_existing_stage_must_be_retained_no_deletion_command',
        'do not delete a previous or partially installed stage' in flat and not re.search(r'(?m)^\s*(rm|del|Remove-Item)\s',guide))
    check('manual_tmux_exact_session_detach_reattach','tmuxnew-session-A-scodeformer_u64_png_checked_v1' in compact and
        'Ctrl+B' in guide and 'Reattach using step 3' in guide)
    launch='cd~/forensic-dgp/'+STAGE.name+'&&source~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate&&python-B-uscripts/run_u64_png_checked_step_v1.py--root.--protocol-sha'+PROTOCOL_SHA+'--verify-transfer&&bashscripts/run_png_checked.sh'+PROTOCOL_SHA
    check('single_exact_protocol_transfer_verify_then_sealed_shell_launch',launch in compact and
        compact.count('--verify-transfer')==1 and compact.count('bashscripts/run_png_checked.sh')==1)
    check('shell_single_protocol_tmux_exact_stage_and_retained_python',all(s in shell for s in
        ('set -euo pipefail','"$#" -ne 1','^[0-9a-f]{64}$','${TMUX:-}','$(pwd -P)',
        'task_root="$HOME/forensic-dgp/'+STAGE.name+'"','$HOME/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python',
        'scripts/run_u64_png_checked_step_v1.py --root "$task_root" --protocol-sha "$1" --supervise')))
    check('bridge_verifies_protocol_and_own_sources_before_runtime_import',all(s in bridge for s in
        ("if '--help' not in sys.argv[1:]",'hashlib.sha256(raw).hexdigest() != args[index+1]',
        'Sealed bootstrap source differs',"importlib.import_module('research.codeformer_u64_png_checked_step_v1.runtime')")))
    check('runtime_static_verifies_both_dependencies_before_worker_import_and_parent_cwd',
        runtime.index('p = verify_static(root, parent, args.protocol_sha)')<runtime.index("worker = importlib.import_module(") and
        'verify_retained_measurement(root,p,guard)' in runtime and 'os.chdir(parent)' in runtime and
        "original parent's directory as a read-only model-loading context" in flat)
    check('actual_event_journal_path_and_no_duplicate_worker_monitor',
        'tail-F~/forensic-dgp/'+STAGE.name+'/outputs/u64_png_checked_step_v1/events.jsonl' in compact and
        'follow the journal without starting another worker' in flat and "store.append('events.jsonl'" in worker)
    first_write=next(i for i,row in enumerate(p['plan']) if row['kind']=='parameter_application')
    check('quiet_initial124_forwards_and_trial_action_progress_messages',first_write==124 and
        all(row['kind']=='forward' for row in p['plan'][:124]) and 'trial-action messages start after the initial 124 forwards' in flat and
        "print(json.dumps({'action': action" in worker)
    dependency=p['retained_measurement'];measurement_static=set(strict_json((STAGE/p['measurement_protocol_copy']['path']).read_bytes())['files'])
    check('real469parent_and288measurement_dependency_counts_no_VM_aliases',len(a['parent_files_rehashed'])==469 and
        len(a['local_parent_model_aliases'])==2 and len(dependency['files'])==288 and len(measurement_static)==21 and
        len(set(dependency['files'])-measurement_static)==267 and sum(name.endswith(('_raw_mae.npz','_raw_adjacent.npz')) for name in dependency['files'])==128 and
        all(s in flat for s in ('all 469 static files','21 sealed static files and 267 required runtime witnesses','128 RAW gradient files',
        'Both stages remain read-only','Local extraction/model aliases used for auditing are not accepted as substitute paths on the VM')),
        {'parent':469,'measurement':288,'measurement_static':21,'measurement_runtime':267,'RAW_gradients':128,'VM_substitutions_permitted':False})
    check('measurement_exact_prior_protocol_and_runtime_manifest_source_guard',dependency['stage_name']==MEASUREMENT and
        dependency['protocol']=={'path':'protocol.json','bytes':188285,'sha256':'0a61c2d6433399e9e748f3782e4d87bf9da3089848b2502a761fcd029690797c'} and
        'measurement=directory(root.parent/MEASUREMENT_STAGE)' in runtime and 'Actual retained VM measurement file differs:' in runtime)
    times={'worker_seconds':1200,'external_seconds':1260,'termination_grace_seconds':20,'per_phase_seconds':180,
        'export_seconds':300,'export_external_seconds':330}
    check('all_worker_phase_external_export_caps_match_protocol',all(limits[k]==v for k,v in times.items()) and all(s in flat for s in
        ('Worker stop: 1200 seconds','1260 seconds','20 seconds cleanup','180-second cap','300 seconds internally, 330 externally')),
        times)
    resources={'initial_free_disk_bytes':8*1024**3,'continuing_free_disk_bytes':4*1024**3,
        'maximum_RSS_bytes':10*1024**3,'maximum_CUDA_reserved_bytes':18*1024**3,
        'maximum_output_bytes':2*1024**3,'maximum_archive_bytes':2*1024**3}
    check('RAM_CUDA_disk_output_archive_caps_exact',all(limits[k]==v for k,v in resources.items()) and all(s in flat for s in
        ('8 GiB free','retain 4 GiB throughout','10 GiB RAM','18 GiB reserved CUDA','2 GiB new outputs','2 GiB return archive')),resources)
    check('concrete_8to16minute_estimate_explicitly_unmeasured','Estimate 8–16 minutes, unmeasured for this route' in flat)
    check('complete436forwards9writes0queries1selected_update_budget',len(p['plan'])==445 and
        sum(row['kind']=='forward' for row in p['plan'])==436 and sum(row['kind']=='parameter_application' for row in p['plan'])==9 and
        limits['gradient_query_attempts']==limits['gradient_query_returns']==limits['optimizer_constructions']==0 and
        limits['selected_training_updates']==1 and all(s in flat for s in ('436 full forwards, zero new gradient queries',
        'zero optimizer constructions and nine explicit parameter applications','four trial writes, four restores','one selected final commit',
        'one selected training update','nine writes are not epochs')))
    check('four_fixed_subscales_point99_and_original_actual_coordinate_bounds',
        pilot['candidate_maximum_absolute_coordinate_bounds']==[2e-5,1.5e-5,1e-5,5e-6] and pilot['fixed_numerical_headroom']==.99 and
        limits['original_maximum_absolute_coordinate_step']==2.5e-5 and limits['maximum_cumulative_absolute_coordinate_delta']==1e-4 and
        all(s in flat for s in ('Every trial starts independently from original u64','2e-5, 1.5e-5, 1e-5, 5e-6','fixed .99 numerical headroom',
        'old 2.5e-5 per-application and 1e-4 cumulative bounds','all 539 R2/U96 tensors stay exact','Every copy and restore is journaled')))
    no=pilot['no_eligible']
    check('noeligible380forward8writes388pairs_exact_original_restored',no['plan_rows']==388 and no['full_forward_returns']==380 and
        no['parameter_applications']==8 and p['plan'][387]['action']=='trial_restore' and p['plan'][388]['action']=='selected_commit' and
        all(s in flat for s in ('380 forwards and eight applications','with original u64 restored','If none qualifies, stop')),
        {'forwards':380,'applications':8,'completed_operation_pairs':388,'selected_commit':False})
    check('all_four_complete_TRAIN_banks_before_firsteligible_zero_tolerance_selection',all(s in flat for s in
        ('All four complete 64-case optimization TRAIN banks are saved before selection','first eligible candidate in the registered order',
        'preserve all 16 actual RAW/PNG means','strictly improve both blur PNG adjacent means')) and
        'zero tolerance' in pilot['selection'])
    check('no_monitor_selection_leakage_single56screen_terminal_failure',all(s in flat for s in
        ('Selection receives no monitor/guard records','one 56-case monitor/guard evaluation','inherited 1% structure',
        'A monitor failure rejects the entire test; no next candidate is tried')))
    check('invariant_failure_immediate_partial_write_unknown_no_restore_erasure',all(s in flat for s in
        ('Invariant, numerical, copy or resource failures stop immediately','A partial write stays unknown; no restore hides that outcome')) and
        'keep pending application unknown' in worker)
    check('worker_exit0_selected_screen_exit2_qualitystop_exit1_other_source_consistent',
        'return 0' in worker and 'return 2 if isinstance(error, QualityStop) else 1' in worker and
        all(s in flat for s in ('worker_exit: 0: the selected finite TRAIN screen passed','still need independent audit and visual review',
        'worker_exit: 2: no candidate qualified or the selected monitor screen failed','Do not resume or append updates',
        'worker_exit: 1 or another nonzero value: execution/resource/invariant failure','without relaunching')))
    check('export_success_distinct_quality_and_unknown_writer_skip',all(s in flat for s in
        ('export_exit: 0: evidence was exported; this does not establish model quality',
        'Unknown writer cleanup can skip export, leaving download files absent','Wait for the supervisor\'s export result',
        'do not rerun the test to manufacture an archive')) and "'active_writer_outcome_unknown': True" in runtime)
    check('return_review_all_states_metrics_selection_resources_and_visuals_required',all(s in flat for s in
        ('actual trial/restore/commit values','all saved RAW/PNG metrics, individual regressions','selection membership/order',
        'separate final screen, frozen tensors, RNG/sampler','resource records, operation accounting and every planned visual sheet')))
    check('prepared_unlaunched_manual_no_VM_inventory_or_retry_claim',p['manual_launch_only'] is True and p['automatic_retry_resume'] is False and
        not a['runtime_real_VM_verify_transfer_pass_claimed'] and all(s in flat for s in
        ('Prepared for one manual L4 test; not launched','No VM connection or current free-space check was performed',
        'No automatic retry/resume','newly sealed implementation','guide reviews must pass before PROJECT_HANDOFF.md')))
    check('no_promotion_no_native_FINAL_generalization_goal_incomplete',p['model_qualified'] is False and p['app_incumbent_unchanged'] is True and
        all(s in flat for s in ('A completed or passing screen does not qualify a new restoration model','U96 stays experimental app incumbent',
        'original u64 and all predecessor/failure states stay retained','does not qualify native CCTV, wild images, FINAL or Zamboanga performance',
        'Broader DEV failures remain binding','stay unpaired','FINAL is unopened and execution-unfrozen','Protect/Apply/bypass remains separate',
        'Automatic matching and later completion development remain deferred','ACTIVE/incomplete')))
    code="""import sys,importlib.abc,runpy
class BlockScientific(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'numpy','PIL','torch','scipy','tensorflow','jax','cv2'}:raise RuntimeError('Scientific import forbidden: '+fullname)
sys.meta_path.insert(0,BlockScientific());script=sys.argv[1];sys.argv=[script,'--help'];runpy.run_path(script,run_name='__main__')
"""
    help_result=subprocess.run([sys.executable,'-B','-c',code,str(STAGE/'scripts/run_u64_png_checked_step_v1.py')],cwd=STAGE,
        capture_output=True,text=True,timeout=10,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    check('actual_staged_help_exit0_all_scientific_imports_blocked',help_result.returncode==0 and not help_result.stderr and
        all('--'+s in help_result.stdout for s in ('root','protocol-sha','verify-transfer','status','supervise','run','export')) and
        not any(STAGE.rglob('__pycache__')),
        {'exit_code':help_result.returncode,'stdout':help_result.stdout,'stderr':help_result.stderr,'scientific_imports_blocked':True})
    checked(Path(__file__));checked(Path(F.__file__))
    for name,value in consumed.items():need(pin(ROOT/name)==value,'Consumed bytes changed after guide review: '+name)
    check('all_consumed_file_pins_exact_unchanged_after',True,{'files':len(consumed)})
    blockers=[x['name'] for x in checks if not x['pass']]
    result={'complete':True,'guide_review_pass':not blockers,'guide_source_review_pass':not blockers,
        'manual_guide_ready_at_exact_consumed_pins':not blockers,'unresolved_blockers':blockers,
        'created_UTC':datetime.now(timezone.utc).isoformat(),'guide':consumed[GUIDE.name],
        'archive_review':consumed[(PREP/'archive_readback_v1.json').relative_to(ROOT).as_posix()],
        'archive':archive,'checksum':checksum,'protocol':protocol,'source':pin(Path(__file__)),
        'checks_pass':sum(x['pass'] for x in checks),'checks':checks,'consumed_files':list(consumed.values()),
        'all_consumed_pins_unchanged_after':True,'guide_commands_executed':False,
        'only_actual_staged_help_executed_with_scientific_imports_blocked':True,
        'worker_model_gradient_parameter_optimizer_VM_browser_calls':0,'model_qualified':False,
        'runtime_real_VM_dependency_verification_claimed':False,
        'scope':'Exact command/path/protocol/tmux/cap/finite-plan/status guide review joined to independent actual sealed archive and local dependency audit. No guide command, model or VM launched.'}
    target=PREP/'guide_review_v1.json'
    with target.open('x',encoding='utf-8',newline='\n') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'receipt':pin(target),'guide':result['guide'],'archive_review':result['archive_review'],
        'checks_pass':result['checks_pass'],'checks_total':len(checks),'unresolved_blockers':blockers},indent=2))


if __name__=='__main__':main()
