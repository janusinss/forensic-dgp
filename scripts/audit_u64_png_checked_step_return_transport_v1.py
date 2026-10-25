"""Independent stdlib-only returned PNG-checked finite-step transport audit.

No extraction, arrays, worker/model import, neural execution or VM access.
The complete streaming pass authenticates every payload before JSON replay.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
STAGE = 'cctv_codeformer_u64_png_checked_step_v1_vm'
PARENT = 'cctv_codeformer_u64_cleananchors_continuation_v1_vm'
PREFIX = 'cctv-codeformer-u64-png-checked-step-v1'
OPREFIX = 'outputs/u64_png_checked_step_v1/'
PROTOCOL_SHA = '7f4546d4d911bea7100f42870514f08a42e349ec524a2b9795f8f3c7f55dcf60'
ARCHIVE_SHA = 'f2d8f2670c0c4b7fd9310317d1a020ffb9f0b1211e07b9d0ed92023e7832fafc'
ARCHIVE_BYTES = 1112595923
OUT = ROOT/'outputs/codeformer_u64_png_checked_step_v1_return_transport'
CAPS={'full_forward_attempts':436,'full_forward_returns':436,'gradient_query_attempts':0,'gradient_query_returns':0,
      'parameter_application_attempts':9,'parameter_applications':9,'trial_parameter_applications':4,
      'original_restoration_applications':4,'selected_parameter_commits':1,'initial_quality_forwards':56,
      'initial_repeat_forwards':4,'initial_optimization_forwards':64,'trial_optimization_forwards':256,'selected_quality_forwards':56}
ACTION_COUNTER={'trial_apply':'trial_parameter_applications','trial_restore':'original_restoration_applications',
                'selected_commit':'selected_parameter_commits'}
OPS = {'forward':('full_forward_attempts','full_forward_returns'),
       'parameter_application':('parameter_application_attempts','parameter_applications')}
HEAD='6629d8414321b0d2ba3a9cc476e415d31911fd4bd1fa121a1a8db9097ce235be'
FROZEN='f096108dc16dff56687a61a57d7ee367f8dd50accc385d2d72be8b53d4510cdf'
MEASUREMENT='cctv_codeformer_u64_png_surrogate_measurement_v1_vm'
POST_EXPORT = {'export.json','export.log','export_failure.json','supervisor.export.json',
               'supervisor.export.resources.jsonl','supervisor.export.started.json',
               'supervisor.json','supervisor_failure.json'}
SPEC = importlib.util.spec_from_file_location('pure_return_framing',ROOT/'scripts/audit_u64_role_gradients_return_transport_v1.py')
F = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(F)
need, strict_json, confined, pin = F.need, F.strict_json, F.confined, F.pin


def write_new(path, value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')


def capture():
    OUT.mkdir(exist_ok=True)
    archive = ROOT/'outputs'/(PREFIX+'-results.tar.gz')
    export_path = ROOT/'outputs'/(PREFIX+'-export.json')
    sidecar = archive.with_name(archive.name+'.sha256')
    export_pin,sidecar_pin = pin(export_path),pin(sidecar)
    export = strict_json(export_path.read_bytes())
    need(export['archive']['bytes']==ARCHIVE_BYTES and export['archive']['sha256']==ARCHIVE_SHA and
         export['protocol_sha256']==PROTOCOL_SHA and archive.stat().st_size==ARCHIVE_BYTES,
         'Exact independent archive/protocol identity required')
    need(sidecar.read_bytes()==(ARCHIVE_SHA+'  '+archive.name+'\n').encode('ascii'),'Exact LF checksum required')
    expected = export['files']; need(len(expected)==1469,'Exact1469 returned members required')
    for name,value in expected.items():
        confined(name); need(value['path']==name and set(value)=={'path','bytes','sha256'},'Exact export pin required')
    before = archive.stat()
    with archive.open('rb') as stream: parsed = F.stream_archive(stream,expected,STAGE)
    after = archive.stat()
    identity = lambda s:(s.st_ino,s.st_size,s.st_mtime_ns)
    need(identity(before)==identity(after),'Archive changed during streaming')
    framing=parsed['framing']
    need(framing['compressed_bytes']==ARCHIVE_BYTES and framing['compressed_sha256']==ARCHIVE_SHA and
         framing['payload_bytes']==1227524329,'Actual compressed and payload identities differ')
    need(pin(export_path)==export_pin and pin(sidecar)==sidecar_pin,'Export/sidecar changed during streaming')
    value={'complete':True,'framing':framing,'files':parsed['files'],
           'authenticated_small_text':{k:v.decode('utf-8') for k,v in parsed['metadata'].items()},
           'archive':{'path':archive.relative_to(ROOT).as_posix(),'bytes':ARCHIVE_BYTES,'sha256':ARCHIVE_SHA},
           'archive_stat':list(identity(after)),'export':export_pin,'sidecar':sidecar_pin,
           'framing_parser':pin(Path(F.__file__)),'actual_array_decodes_extractions_worker_model_VM_calls':0}
    target=OUT/'stream_capture_v1.json';write_new(target,value)
    print(json.dumps({'capture':pin(target),'framing':framing},indent=2))


def validate_plan(p):
    expected=[]
    def forward(source,phase,category,trial=None):
        expected.append({'slot':len(expected)+1,'kind':'forward','id':source['id'],'reference_id':source['reference_id'],
            'role':source['role'],'metric_region':'central128' if source['role']=='local_gaussian_1p2' else 'whole',
            'phase':phase,'category':category,'trial_index':trial,'gradient_queries':[]})
    def application(action,trial):
        expected.append({'slot':len(expected)+1,'kind':'parameter_application','id':action+'_'+('selected' if trial is None else 'trial_'+str(trial)),
            'action':action,'trial_index':trial,'phase':action,'gradient_queries':[],
            'source_state':'candidate_trial_'+str(trial) if action=='trial_restore' else 'original_u64',
            'destination_state':'original_u64' if action=='trial_restore' else
                'candidate_trial_'+str(trial) if action=='trial_apply' else 'selected_eligible_candidate',
            'destination_head_canonical_sha256':HEAD if action=='trial_restore' else None,
            'worker_must_record_exact_before_after_canonical':True})
    selection=p['selection'];bank=selection['monitors']+selection['guards'];optimization=selection['optimization']
    need(len(bank)==56 and len(optimization)==64,'Exact bank selection sizes required')
    for row in bank:forward(row,'initial_u64_quality_bank','initial_quality_forwards')
    for row in selection['monitors'][:4]:forward(row,'deterministic_same_u64_repeat','initial_repeat_forwards')
    for row in optimization:forward(row,'initial_optimization_bank','initial_optimization_forwards')
    for trial in range(4):
        application('trial_apply',trial)
        for row in optimization:forward(row,'trial_optimization_bank','trial_optimization_forwards',trial)
        application('trial_restore',trial)
    need(len(expected)==388,'Exact noeligible388-operation prefix required')
    application('selected_commit',None)
    for row in bank:forward(row,'selected_quality_bank','selected_quality_forwards')
    need(p['plan']==expected and len(expected)==445,'Independently reconstructed445-row plan differs')


def replay(raw,plan):
    need(type(raw) is bytes and raw.endswith(b'\n') and len(raw)<32*1024**2,'Complete bounded journal required')
    need(len(plan)==445 and [r['slot'] for r in plan]==list(range(1,446)) and all(r['gradient_queries']==[] for r in plan),
         'Exact445 noquery plan required')
    counts=dict.fromkeys(CAPS,0);position=token=applications=0;pending=None;current=HEAD;trials={};returns=[]
    previous=-1.;rows=[];kinds=Counter()
    for line in raw.splitlines():
        row=strict_json(line);rows.append(row);kind=row['kind'];kinds[kind]+=1;elapsed=row.get('elapsed_seconds')
        need(type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed>=previous,'Monotonic finite event time required');previous=elapsed
        reserve=kind in ('forward_reserved','parameter_application_reserved');returned=kind in ('forward_returned','parameter_application_returned')
        need(not kind.endswith(('_reserved','_returned')) or reserve or returned or kind=='phase_returned','Unknown operation event refused')
        need(not any(word in kind for word in ('gradient_query','optimizer','backward')),'New gradient/optimizer/backward event refused')
        if reserve:
            need(pending is None and position<len(plan),'Overlapping or excess operation reservation')
            call=plan[position];operation=call['kind'];token+=1
            need(operation in OPS and kind==operation+'_reserved' and type(row['token']) is int and row['token']==token,
                 'Exact operation/token order required')
            pending={'token':token,'plan_slot':call['slot'],'id':call['id'],'trial_index':call['trial_index'],
                'phase':'call_'+str(call['slot']).zfill(4) if operation=='forward' else 'application_'+str(applications+1).zfill(2)}
            if operation=='forward':pending['category']=call['category'];counts[call['category']]+=1
            else:pending['action']=call['action']
            counts[OPS[operation][0]]+=1
        elif returned:
            need(pending is not None,'Return without reserved operation');call=plan[position];operation=call['kind']
            need(kind==operation+'_returned' and row.get('synchronized_return') is True,'Known synchronized exact return required')
            counts[OPS[operation][1]]+=1
            if operation=='parameter_application':
                action=call['action'];trial=call['trial_index'];before=row.get('actual_before_head_canonical_sha256');after=row.get('actual_after_head_canonical_sha256')
                need(before==current and type(after) is str and re.fullmatch('[0-9a-f]{64}',after),'Broken actual head canonical chain')
                need(action in ACTION_COUNTER,'Known application action required');counts[ACTION_COUNTER[action]]+=1
                if action=='trial_restore':need(current==trials[trial] and after==HEAD,'Restore does not recover exact original heads')
                else:
                    need(current==HEAD and after!=HEAD,'Independent-original nonzero trial/commit required')
                    if action=='trial_apply':trials[trial]=after
                    else:need(after in trials.values(),'Selected commit differs from every registered candidate')
                applications+=1;current=after
                returns.append({'index':applications,'call':call,'before':before,'after':after,'counters':dict(counts)})
        if reserve or returned:
            need(all(row.get(k)==v and (v is None or type(row.get(k)) is type(v)) for k,v in pending.items()),'Operation identity/action/phase/slot differs')
            need(row['counters']==counts and all(type(v) is int and 0<=v<=CAPS[k] for k,v in counts.items()),'Exact bounded counter replay differs')
            if returned:position+=1;pending=None
        elif 'counters' in row:need(row['counters']==counts,'Nonoperation counters differ')
    return {'counters':counts,'journal_events_replayed':len(rows),'event_kinds':dict(kinds),'completed_operation_pairs':position,
        'maximum_operation_pairs':445,'last_token':token,'unknown_reservations':[] if pending is None else [pending],
        'operation_outcome_unknown':pending is not None,'full_exact_plan_completed':position==445 and pending is None,
        'valid_sealed_plan_prefix':True,'application_returns_by_action':dict(Counter(r['call']['action'] for r in returns)),
        'last_known_head_canonical_sha256':current,'no_eligible_original_restored_prefix':position==388 and pending is None and current==HEAD,
        'last_journal_elapsed_seconds':previous,'applications':returns},rows


def evidence_audit(docs,raw,p,journal,events,files,export):
    limits=p['limits'];status=docs[OPREFIX+'status.json'];failure=docs[OPREFIX+'failure.json']
    worker=docs['supervisor.worker.json'];spawned=docs['supervisor.worker.started.json'];start=docs['supervisor.start.json']
    need(journal['counters']==CAPS and journal['full_exact_plan_completed'] and len(events)==1341,'Full actual436/9 plan required')
    for value in (status,failure,export):
        need(value['protocol_sha256']==PROTOCOL_SHA and value['counters']==CAPS and value['unknown_reservations']==[] and
             value['operation_outcome_unknown'] is False and value['model_qualified'] is False,'Terminal/journal outcome differs')
    need(status['complete'] is False and status['label']=='retained_quality_stop' and status['active_application']==9 and
         status['active_slot']==445 and status['active_case']==p['plan'][-1]['id'] and status['active_query'] is None and
         status['phase']=='call_0445' and status['elapsed_seconds']<limits['worker_seconds'],'Actual retained stop status differs')
    need(failure['quality_stop'] is True and failure['error_type']=='QualityStop' and
         failure['error']=='Selected candidate failed unchanged terminal monitor screen; no fallback' and
         failure['active_slot']==445 and failure['active_application']==9 and OPREFIX+'result.json' not in files,
         'Authoritative failed monitor terminal classification differs')
    need(start['protocol_sha256']==PROTOCOL_SHA and start['worker_spawn_reserved'] is True and
         start['read_only_parent'].endswith('/'+PARENT),'Supervisor root/protocol reservation differs')
    need(spawned['child_pid']==spawned['process_group']==worker['child_pid'] and type(worker['child_pid']) is int and worker['child_pid']>0 and
         spawned['phase']==worker['phase']=='worker' and spawned['external_seconds']==worker['external_seconds']==limits['external_seconds'],
         'Actual known child/group/external cap differs')
    cleanup=worker['cleanup']
    need(worker['complete'] is True and worker['exit_code']==2 and worker['timeout'] is False and worker['spawn_returned'] is True and
         worker['error'] is None and cleanup['child_reaped'] is True and cleanup['process_group_absent'] is True and cleanup['actions']==[] and
         0<=cleanup['seconds']<=limits['termination_grace_seconds'] and 0<=worker['elapsed_seconds']<limits['external_seconds'],
         'Worker exit2 known bounded cleanup differs')
    initial=['idle_GPU_before_framework','dependencies_and_FP32_policy','exact_original_u64_initialization_no_forward',
             'cached_frozen_U96_controls_zero_neural','complete_initial_u64_state_and_checkpoint',
             'retained_RAW_gradients_and_original_direction_no_autograd']
    expected_events=[('static_VM_supervisor_guards_passed_before_Torch','static_guards')]
    expected_events.extend(('phase_returned',name) for name in initial);labels=['before','after']*len(initial);app=0
    for index,call in enumerate(p['plan']):
        operation=call['kind']
        if operation=='parameter_application':app+=1
        phase='call_'+str(call['slot']).zfill(4) if operation=='forward' else 'application_'+str(app).zfill(2)
        expected_events.extend([(operation+'_reserved',phase),(operation+'_returned',phase)])
        labels+=['before']+(['before_forward'] if operation=='forward' else [])
        if index<444:expected_events.append(('phase_returned',phase));labels.append('after')
    need([(row['kind'],row['phase']) for row in events]==expected_events and events[0]['Torch_imported'] is False,
         'Exact1341-event phase/interleaving differs')
    phases=[r for r in events if r['kind']=='phase_returned']
    need(len(phases)==450 and all(0<=r['phase_seconds']<limits['per_phase_seconds'] for r in phases),'Completed phase count/cap differs')
    resources=[strict_json(line) for line in raw[OPREFIX+'resources.jsonl'].splitlines()]
    watchdog=[strict_json(line) for line in raw['supervisor.worker.resources.jsonl'].splitlines()]
    need(raw[OPREFIX+'resources.jsonl'].endswith(b'\n') and raw['supervisor.worker.resources.jsonl'].endswith(b'\n') and
         [r['label'] for r in resources]==labels,'Complete exact1337 resource boundary samples required')
    previous=-1.
    for row in resources:
        need(type(row['elapsed_seconds']) in (int,float) and previous<=row['elapsed_seconds']<limits['worker_seconds'],
             'Worker measured elapsed cap/order differs');previous=row['elapsed_seconds']
        for key in ('current_RSS_bytes','peak_RSS_bytes','CUDA_reserved_bytes','CUDA_peak_reserved_bytes','output_bytes','free_disk_bytes'):
            need(type(row[key]) is int and row[key]>=0,'Nonnegative integer saved resource measurement required')
        need(row['free_disk_bytes']>=limits['continuing_free_disk_bytes'] and
             max(row['current_RSS_bytes'],row['peak_RSS_bytes'])<=limits['maximum_RSS_bytes'] and
             max(row['CUDA_reserved_bytes'],row['CUDA_peak_reserved_bytes'])<=limits['maximum_CUDA_reserved_bytes'] and
             row['output_bytes']<=limits['maximum_output_bytes'],'Saved worker resource bound exceeded')
    need(resources[0]['free_disk_bytes']>=limits['initial_free_disk_bytes'],'Initial sampled8GiB admission differs')
    previous=-1.
    for row in watchdog:
        need(row['child_pid']==worker['child_pid'] and row['phase']=='worker' and previous<=row['elapsed_seconds']<limits['external_seconds'],
             'Watchdog child/time order differs');previous=row['elapsed_seconds']
        need(row['free_disk_bytes']>=limits['continuing_free_disk_bytes'] and
             (row['current_RSS_bytes'] is None or 0<=row['current_RSS_bytes']<=limits['maximum_RSS_bytes']) and
             0<=row['output_bytes']<=limits['maximum_output_bytes'] and 0<=row['archive_bytes']<=limits['maximum_archive_bytes'],
             'Watchdog resource bound exceeded')
    need(worker['resource_samples']==len(watchdog) and worker['minimum_free_disk_bytes']==min(r['free_disk_bytes'] for r in watchdog),
         'Watchdog saved sample count/minimum differs')
    for key in worker['resource_maxima']:
        need(worker['resource_maxima'][key]==max((r[key] for r in watchdog if r[key] is not None),default=0),
             'Watchdog saved maximum differs')
    gate=docs[OPREFIX+'banks/selected_quality.json']['gate'];selection=docs[OPREFIX+'selection.json']
    need(gate['pass'] is False and gate['quality_stop'] is True and gate['additional_role_mean_checks_pass'] is False and
         gate['earlier_state_fallback_or_retry_authorized'] is False and gate['zero_added_tolerance'] is True and
         selection['selected_index']==0 and selection['all_four_trials_evaluated'] is True and
         selection['all_four_original_restores_completed'] is True and selection['monitor_records_supplied']==0,
         'Saved quality-stop/optimization-only selection declarations differ')
    own_bytes=sum(v['bytes'] for k,v in files.items() if k.startswith(OPREFIX))
    need(own_bytes<=limits['maximum_output_bytes'] and ARCHIVE_BYTES<=limits['maximum_archive_bytes'],'Actual output/archive2GiB cap exceeded')
    return {'complete':True,'terminal_classification':'QUALITY_STOP_SELECTED_MONITOR_SCREEN_FAILED_RETAINED_NOT_QUALIFIED',
        'worker':worker,'status':status,'failure':failure,'completed_phase_returns':450,
        'maximum_completed_phase_seconds':max(r['phase_seconds'] for r in phases),'worker_resource_samples':len(resources),
        'watchdog_samples':len(watchdog),'worker_resource_maxima':{k:max(r[k] for r in resources) for k in
          ('current_RSS_bytes','peak_RSS_bytes','CUDA_reserved_bytes','CUDA_peak_reserved_bytes','output_bytes')},
        'minimum_worker_sampled_free_disk_bytes':min(r['free_disk_bytes'] for r in resources),
        'actual_archived_own_output_bytes':own_bytes,'selected_candidate_index_declaration':0,
        'post_export_receipts_absent_by_design':sorted(POST_EXPORT),'final_export_exit_code_available':False,
        'scientific_gate_metrics_not_independently_recomputed_here':True,
        'scope':'Authenticated saved finite operation/resource/cleanup/failure records; not current VM monitoring or independent quality validation.'}


def metadata_joins(docs,p,files,journal):
    expected=set(p['files'])|{'protocol.json','supervisor.start.json','supervisor.worker.started.json','supervisor.worker.json',
        'supervisor.worker.resources.jsonl','run.log'}
    base=('started.json','operation_plan.json','events.jsonl','resources.jsonl','preflight.json','frozen_U96_control_metrics.json',
          'failure.json','status.json','selection.json','failure_values/RNG_sampler.json','full_states/initial.json',
          'full_states/initial.npz','full_states/failed_current.json','full_states/failed_current.npz',
          'direction/direction.npy','direction/pooled_RAW8.npy','direction/solver.json','direction/retained_RAW_dependencies.json',
          'banks/initial_quality.json','banks/initial_optimization.json','banks/selected_quality.json')
    expected.update(OPREFIX+name for name in base)
    for trial in range(4):expected.update(OPREFIX+n for n in
        ('banks/trial_'+str(trial)+'.json','trials/'+str(trial)+'/proposed_heads.npz','trials/'+str(trial)+'/rounded_delta.npy'))
    def bound(value):
        need(type(value) is dict and set(value)=={'path','bytes','sha256'} and files[confined(value['path'])]==value,
             'Saved metadata binding differs from actual payload')
    counts=dict.fromkeys(CAPS,0);head=HEAD;app=0
    source_rows={r['id']:r for r in p['selection']['monitors']+p['selection']['guards']+p['selection']['optimization']}
    apps={r['call']['slot']:r for r in journal['applications']}
    rng=docs[OPREFIX+'applications/a0000/resume.json']['RNG'];schema=None
    for index in range(10):
        stem=OPREFIX+'applications/a'+str(index).zfill(4)+'/'
        expected.update(stem+n for n in ('checkpoint.json','heads.npz','resume.json'))
        if index:expected.update(stem+n for n in ('commit.json','precommit.json','precommit_heads.npz'))
    for call in p['plan']:
        if call['kind']=='parameter_application':
            app+=1;actual=apps[call['slot']];head=actual['after'];stem=OPREFIX+'applications/a'+str(app).zfill(4)+'/'
            pre=docs[stem+'precommit.json'];commit=docs[stem+'commit.json']
            need(pre['call']==commit['call']==call and pre['before']==commit['actual_before']==actual['before'] and
                 commit['actual_after']==head and commit['counters']==actual['counters'],'Precommit/commit/journal canonical join differs')
            bound(pre['candidate']);bound(commit['candidate'])
            need(pre['candidate']==commit['candidate']==files[stem+'precommit_heads.npz'] and
                 pre['description']['canonical_sha256']==head,'Precommitted candidate metadata identity differs')
            need(pre['actual_maxabs_step']<=p['limits']['original_maximum_absolute_coordinate_step'] and
                 pre['actual_maxabs_cumulative']<=p['limits']['maximum_cumulative_absolute_coordinate_delta'],
                 'Saved displacement bound declaration exceeded')
            need(commit['selected_training_updates']==int(call['action']=='selected_commit') and
                 pre['restore_is_explicit_reversal']==(call['action']=='trial_restore'),'Application selected/reversal metadata differs')
        else:
            stem=OPREFIX+'calls/'+str(call['slot']).zfill(4)+'/'
            expected.update(stem+n for n in ('record.json','raw.npy','result.png'))
            parity=call['phase']=='deterministic_same_u64_repeat' or (call['phase']=='initial_u64_quality_bank' and call['id'] in
                {r['id'] for r in p['selection']['monitors'][:4]})
            if parity:expected.update(stem+n for n in ('native.npy','logits.npy','LQ.npy','unclipped.npy'))
            record=docs[stem+'record.json'];source=source_rows[call['id']]
            need(record['complete'] is True and record['call']==call and record['input']==source['input'] and record['target']==source.get('target') and
                 record['id']==call['id'] and record['reference_id']==call['reference_id'] and record['role']==call['role'] and
                 record['parameter_application_index']==app and record['head_before_sha256']==head and record['frozen539_canonical_sha256']==FROZEN,
                 'Forward plan/input/target/application/current-head metadata join differs')
            for value in record['outputs'].values():bound(value)
            for value in record['stock'].values():
                if 'file' in value:bound(value['file'])
    for index in range(10):
        stem=OPREFIX+'applications/a'+str(index).zfill(4)+'/'
        checkpoint=docs[stem+'checkpoint.json'];resume=docs[stem+'resume.json'];head=HEAD if index==0 else journal['applications'][index-1]['after']
        bound(checkpoint['heads']);bound(checkpoint['resume'])
        need(checkpoint['parameter_application_index']==resume['parameter_application_index']==index and
             checkpoint['head_description']['canonical_sha256']==resume['head_canonical_sha256']==head and
             resume['RNG']==rng and resume['optimizer'] is None and resume['optimizer_constructions']==0 and
             resume['initial_head_canonical_sha256']==HEAD and resume['frozen539_canonical_sha256']==FROZEN and
             resume['sampler']['automatic_resume_permitted'] is False,'Checkpoint/RNG/null-optimizer/head-chain metadata differs')
        if index==0:need(all(v==0 for v in resume['counters'].values()),'Initial checkpoint counters not zero')
        else:need(resume['counters']==journal['applications'][index-1]['counters'],'Checkpoint/action counters differ')
    need(set(files)==expected and len(expected)==1469,'Independently reconstructed full1469-member namespace differs')
    failure=docs[OPREFIX+'failure.json'];bound(failure['last_checkpoint']);bound(failure['last_complete_state'])
    bound(failure['failed_current_state']);bound(failure['failed_RNG_sampler'])
    failed=docs[OPREFIX+'failure_values/RNG_sampler.json']
    need(failed['RNG']==rng and failed['optimizer'] is None and failed['active_application']==9 and failed['counters']==CAPS,
         'Saved failure RNG/sampler/outcome differs')
    for label,index in (('initial',0),('failed_current',9)):
        value=docs[OPREFIX+'full_states/'+label+'.json'];bound(value['state'])
        need(value['parameter_application_index']==index and value['head36_canonical_sha256']==
             (HEAD if index==0 else journal['last_known_head_canonical_sha256']),'Full-state/head checkpoint metadata differs')
    return {'complete':True,'independently_reconstructed_file_names':1469,'forward_metadata_joins':436,'application_metadata_joins':9,
        'checkpoint_RNG_null_optimizer_metadata_joins':10,'full575_state_payload_joins':2,'all_recorded_restores_exact_original_head_canonical':True,
        'saved_run_RNG_equal_at_all10_checkpoints_and_failure':True,'array_image_gradient_values_not_decoded_or_recomputed':True}


def audit():
    consumed={}
    def checked(path):
        value=pin(path);consumed[value['path']]=value;return value
    def doc(path):checked(path);return strict_json(path.read_bytes())
    c=doc(OUT/'stream_capture_v1.json');archive=ROOT/c['archive']['path'];s=archive.stat()
    need([s.st_ino,s.st_size,s.st_mtime_ns]==c['archive_stat'],'Archive changed after independent full stream')
    export_path=ROOT/'outputs'/(PREFIX+'-export.json');export=doc(export_path)
    need(checked(export_path)==c['export'] and checked(ROOT/c['sidecar']['path'])==c['sidecar'] and
         checked(Path(F.__file__))==c['framing_parser'],'Stream-authenticated export/checksum/parser changed')
    files=c['files'];raw={k:v.encode('utf-8') for k,v in c['authenticated_small_text'].items()}
    need(files==export['files'],'Actual/export payload pins differ')
    protocol_path=ROOT/'outputs'/STAGE/'protocol.json';p=doc(protocol_path)
    need(raw['protocol.json']==protocol_path.read_bytes() and checked(protocol_path)['bytes']==382027 and
         consumed[protocol_path.relative_to(ROOT).as_posix()]['sha256']==PROTOCOL_SHA,'Returned/original sealed protocol differs')
    need(len(p['files'])==33,'Exact33 own static bindings required')
    for name,value in p['files'].items():
        actual=checked(ROOT/'outputs'/STAGE/name)
        need(files[name]=={'path':name,**value} and {k:actual[k] for k in ('bytes','sha256')}==value,'Returned/local sealed payload differs: '+name)
    admissions={}
    for name,flag in [('evidence/source_review.json','source_review_pass'),('evidence/source_tests.json','all_tests_pass')]:
        proof=strict_json(raw[name]);need(proof['complete'] is True and proof[flag] is True,'Returned source admission not passed')
        if 'review' in name:need(proof['unresolved_blockers']==[],'Source admission blockers')
        else:need(proof['neural_gradient_optimizer_training_VM_calls']==0,'Source tests not pure')
        approved={r['path']:{'bytes':r['bytes'],'sha256':r['sha256']} for r in proof['sources']}
        need(len(approved)==len(proof['sources'])==16,'Exactly16 approved current sources required')
        for name,value in p['files'].items():
            if name.startswith(('scripts/','research/')):
                key='research/codeformer_u64_png_checked_step_v1/run_png_checked.sh' if name=='scripts/run_png_checked.sh' else name
                need(approved[key]==value,'Source review/test binding differs: '+name)
        admissions[flag]={'sources_verified':16,'proof':files['evidence/source_review.json' if flag=='source_review_pass' else 'evidence/source_tests.json']}
    parent=strict_json(raw['evidence/parent_protocol.json']);proof=strict_json(raw['evidence/local_parent_readback.json'])
    actual_parent=checked(ROOT/'outputs'/PARENT/'protocol.json')
    need({k:actual_parent[k] for k in ('bytes','sha256')}=={k:p['parent_protocol'][k] for k in ('bytes','sha256')} and
         (ROOT/actual_parent['path']).read_bytes()==raw['evidence/parent_protocol.json'] and
         export['parent_consumed_bindings']==parent['files'] and len(parent['files'])==469 and
         export['parent_protocol']==p['parent_protocol'] and export['parent_payload_included'] is False,'Original469parent consumption differs')
    local={r['parent_relative']:r['actual_local'] for r in proof['files']};parentrows=[];parentaliases=[]
    need(len(local)==469 and set(local)==set(parent['files']),'Exact parent provenance namespace required')
    for name,value in parent['files'].items():
        confined(name);source=ROOT/'outputs'/PARENT/name
        if not source.is_file():
            need(name in ('weights/codeformer_r2.pth','weights/codeformer_u96_adapter.pth'),'Unexpected parent alias')
            source=ROOT/confined(local[name]['path']);parentaliases.append({'parent_relative':name,'actual_local':local[name]})
        actual=checked(source);need(actual==local[name] and {k:actual[k] for k in ('bytes','sha256')}==value,'Actual parent file mismatch')
        parentrows.append({'parent_relative':name,'actual_local':actual})
    need(len(parentaliases)==2,'Exactly two declared local model aliases required')
    dependency=p['retained_measurement'];mp=strict_json(raw['evidence/measurement_protocol.json']);mproof=strict_json(raw['evidence/local_measurement_readback.json'])
    need(export['retained_measurement']==dependency and export['retained_measurement_payload_included'] is False and
         dependency['stage_name']==MEASUREMENT and len(dependency['files'])==288 and len(mp['files'])==21 and
         strict_json(raw['evidence/retained_measurement_manifest.json'])==dependency,'Exact retained288 measurement manifest required')
    measurement_protocol=checked(ROOT/'outputs'/MEASUREMENT/'protocol.json')
    need({k:measurement_protocol[k] for k in ('bytes','sha256')}=={k:dependency['protocol'][k] for k in ('bytes','sha256')} and
         (ROOT/measurement_protocol['path']).read_bytes()==raw['evidence/measurement_protocol.json'],'Actual retained measurement protocol differs')
    mlocal={r['measurement_relative']:r['actual_local'] for r in mproof['files']};mrows=[];malias=[]
    need(len(mlocal)==288 and set(mlocal)==set(dependency['files']) and mproof['local_aliases_permitted_on_VM'] is False,
         'Exact explicit local-only retained measurement provenance required')
    for name,value in dependency['files'].items():
        confined(name);source=ROOT/'outputs'/MEASUREMENT/name
        if not source.is_file():
            need(name not in mp['files'] and name in mproof['local_runtime_extraction_aliases'],'Undeclared local retained alias')
            source=ROOT/confined(mlocal[name]['path']);malias.append(name)
        actual=checked(source);need(actual==mlocal[name] and {k:actual[k] for k in ('bytes','sha256')}==value,'Actual retained measurement witness mismatch')
        mrows.append({'measurement_relative':name,'actual_local':actual})
    need(len(malias)==267 and set(malias)==set(mproof['local_runtime_extraction_aliases']),'Exact267 runtime-only local extraction aliases required')
    validate_plan(p);docs={k:strict_json(v) for k,v in raw.items() if k.endswith('.json')}
    need(docs[OPREFIX+'operation_plan.json']['calls']==p['plan'],'Returned operation plan differs from seal')
    journal,events=replay(raw[OPREFIX+'events.jsonl'],p['plan'])
    for key in ('counters','journal_events_replayed','completed_operation_pairs','maximum_operation_pairs','last_token','unknown_reservations',
                'operation_outcome_unknown','full_exact_plan_completed','valid_sealed_plan_prefix','application_returns_by_action',
                'last_known_head_canonical_sha256','no_eligible_original_restored_prefix'):
        need(export[key]==journal[key],'Export versus independent journal replay differs: '+key)
    need(set(export['post_export_receipts_not_in_archive'])==POST_EXPORT and not POST_EXPORT.intersection(files),'Post-export receipt scope differs')
    runtime=evidence_audit(docs,raw,p,journal,events,files,export);joins=metadata_joins(docs,p,files,journal)
    logpath=ROOT/'outputs/codeformer_u64_png_checked_step_v1_return_audit/extracted/run.log';logpin=checked(logpath)
    need({k:logpin[k] for k in ('bytes','sha256')}=={k:files['run.log'][k] for k in ('bytes','sha256')},'Retained run.log differs from authenticated archive')
    log=logpath.read_text();lines=log.splitlines();progress=[strict_json(line) for line in lines if line.startswith('{')]
    need(len(progress)==9 and all(v['parameter_applications']==i and v['action']==journal['applications'][i-1]['call']['action'] and
         v['trial_index']==journal['applications'][i-1]['call']['trial_index'] for i,v in enumerate(progress,1)) and
         'Selected candidate failed unchanged terminal monitor screen; no fallback' in log,'Run log progress/quality stop differs')
    workerpath=ROOT/'outputs'/STAGE/'research/codeformer_u64_png_checked_step_v1/worker.py';tree=ast.parse(workerpath.read_text())
    def dotted(node):return node.id if isinstance(node,ast.Name) else dotted(node.value)+'.'+node.attr if isinstance(node,ast.Attribute) else ''
    calls=[dotted(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
    need(calls.count('parameter.copy_')==1 and not any(c.endswith(('.backward','.step')) or '.autograd.' in c or '.optim.' in c for c in calls),
         'Sealed worker source no-gradient/optimizer explicit application callsite scope differs')
    metadata_path=OUT/'authenticated_small_metadata_v1.json';write_new(metadata_path,{'archive_sha256':ARCHIVE_SHA,'documents':docs,'events':events})
    checked(metadata_path);checked(Path(__file__))
    for name,value in consumed.items():need(pin(ROOT/name)==value,'Consumed evidence changed after audit')
    consumed[c['archive']['path']]=c['archive']
    receipt={'complete':True,'transport_audit_pass':True,'status':runtime['terminal_classification'],'created_UTC':datetime.now(timezone.utc).isoformat(),
        'archive':c['archive'],'export':c['export'],'sidecar':c['sidecar'],'protocol':consumed[protocol_path.relative_to(ROOT).as_posix()],
        'framing':c['framing'],'all1469_actual_member_pins_match_export':True,'all_member_pins':files,
        'sealed_static_bindings_verified':33,'source_admissions':admissions,'parent_static_bindings_rehashed':469,
        'parent_local_model_aliases':parentaliases,'parent_bindings':parentrows,'retained_measurement_bindings_rehashed':288,
        'retained_measurement_local_runtime_aliases':malias,'retained_measurement_bindings':mrows,
        'journal':journal,'runtime':runtime,'metadata_joins':joins,'run_log_progress_entries_verified':9,
        'worker_source_AST':{'no_runtime_source_imported':True,'parameter_copy_callsites':1,'new_autograd_backward_optimizer_step_callsites':0,'source':pin(workerpath)},
        'unresolved_transport_blockers':[],'source':pin(Path(__file__)),'authenticated_metadata':pin(metadata_path),
        'consumed_files':list(consumed.values()),'all_consumed_file_pins_unchanged_after':True,
        'archive_fully_hashed_once_and_stat_unchanged_after':True,
        'scope':'Complete gzip/USTAR/payload/source/dependency integrity, full operation/canonical metadata chain and saved resource/cleanup/failure records. Actual scientific state/displacement/quality/pixel values require separate audits.',
        'array_image_decodes_worker_model_new_gradient_actual_parameter_VM_browser_calls':0,'model_qualified':False,
        'full_operation_completion_does_not_overrule_quality_stop':True,'final_export_exit_code_available':False}
    target=OUT/'receipt_v1.json';write_new(target,receipt)
    print(json.dumps({'receipt':pin(target),'status':receipt['status'],'framing':c['framing'],
        'journal':{k:v for k,v in journal.items() if k not in ('applications','event_kinds')},'runtime':runtime,
        'unresolved_transport_blockers':[]},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--capture',action='store_true');args=parser.parse_args()
    if args.capture:capture()
    else:audit()
