"""Independent saved-metadata chronology/state/response audit; no neural imports."""
import argparse
import ast
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path,PurePosixPath
import stat
import struct
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL='3ea4f1fe9ce90f3d29d29d6ace108eae7a809e6665cb770811f36773b888c089'
ARCHIVE='e60258356e148ae426dcc8863794ad62377559f858a2b5dbf75c3e0eee0d50ad'
PREFIX='outputs/fixed_state_response_v1/'
COUNTS={'CodeFormer_new_original':64,'CodeFormer_new_full_coarse':64,'controller_fitted_initial':200,
    'controller_fitted_final':200,'controller_heldout_initial':64,'controller_heldout_final':64,
    'gradient_queries':0,'backward_calls':0,'optimizer_constructions':0,'optimizer_steps':0,'committed_training_updates':0}
DIGESTS={'original':'6cb93cee937f67814edcf4625a68bae4bf8f1c2351f7306864ceefa63d4803bd',
    'base':'92ac81750b86ef1da6f1f7c0480b9e64773fafe07b094e4a5ad7e88b3f7a0c8b',
    'head0000':'9a3f78f85d1bc9b70878cabad61545ec808248566c7b13220052c3ed02918509',
    'head0050':'d6ec3096958fc53c3e067de3c44ff398252a2bd9bd25568dcf89997f68f3b20b'}


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def unique(rows):
        out={}
        for k,v in rows:require(k not in out,'Duplicate JSON key');out[k]=v
        return out
    return json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=unique,
        parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def f32(value):return struct.unpack('<f',struct.pack('<f',value))[0]


def close(a,b):require(type(a)in(int,float)and type(b)in(int,float)and math.isfinite(a)and math.isfinite(b)and
    abs(a-b)<=1e-14*max(1,abs(b)),'Independent saved-scalar arithmetic differs')


class Evidence:
    def __init__(self,receipt):
        self.receipt_path=Path(receipt).resolve(strict=True);self.receipt_sha=sha(self.receipt_path);self.receipt=read(self.receipt_path)
        r=self.receipt
        require(r['complete']is True and r['protocol_sha256']==PROTOCOL and r['archive_sha256']==ARCHIVE and
            r['regular_files']==2536 and r['payload_bytes']==583143654,'Exact completed inert import required')
        self.root=Path(r['import_root']).resolve(strict=True)
        require(self.root==(ROOT/'outputs/codeformer_fixed_state_response_v1_return').resolve(strict=True),'Exact imported root required')
        self.hashes,self.sizes=r['files_sha256'],r['files_bytes'];require(set(self.hashes)==set(self.sizes),'Closed import map required')
        self.pins={};self.external={};self.identities={}
        self.p=self.json('protocol.json');require(self.hashes['protocol.json']==PROTOCOL and self.p['call_caps']==COUNTS,'Protocol census differs')
        self.stage=ROOT/'outputs/cctv_codeformer_fixed_state_response_v1_vm'
        stage_receipt=ROOT/'outputs/codeformer_fixed_state_response_v1_preparation/stage_v1.json'
        self.stage_receipt=self.local(stage_receipt);self.resolution=self.stage_receipt['local_dependency_resolution']
        require(self.stage_receipt['protocol_sha256']==PROTOCOL,'Exact stage receipt required')
        for n,d in self.p['files_sha256'].items():
            require(self.hashes[n]==d and self.sizes[n]==self.p['files_bytes'][n]and sha(self.get(n))==sha(self.stage/n)==d,
                'Frozen returned/static source differs: '+n)
        m=self.json(PREFIX+'export_manifest.json');mn=PREFIX+'export_manifest.json'
        require(m['files_sha256']=={n:d for n,d in self.hashes.items()if n!=mn}and
            m['files_bytes']=={n:d for n,d in self.sizes.items()if n!=mn}and m['protocol_sha256']==PROTOCOL,'Exact manifest map differs')
    def get(self,name):
        p=PurePosixPath(name);require(name in self.hashes and p.as_posix()==name and not p.is_absolute()and '..'not in p.parts,'Unsafe/unbound return path')
        path=self.root.joinpath(*p.parts);require(path.resolve(strict=True).is_relative_to(self.root),'Return path escapes')
        info=path.lstat();require(stat.S_ISREG(info.st_mode)and info.st_nlink==1 and not path.is_symlink()and
            not getattr(info,'st_file_attributes',0)&1024 and info.st_size==self.sizes[name],'Exact regular evidence required')
        identity=(info.st_size,info.st_mtime_ns,info.st_ino)
        if name not in self.pins:require(sha(path)==self.hashes[name],'Returned evidence changed');self.pins[name]=self.hashes[name];self.identities[name]=identity
        else:require(identity==self.identities[name],'Read evidence changed')
        return path
    def json(self,name):return read(self.get(name))
    def local(self,path):
        path=Path(path);digest=sha(path);doc=read(path);require(sha(path)==digest,'Local evidence changed')
        self.external[path.relative_to(ROOT).as_posix()]=digest;return doc
    def parent_json(self,alias,name):
        pin=self.p['dependencies'][alias][name];path=Path(self.resolution[alias][name]);require(path.stat().st_size==pin['bytes']and sha(path)==pin['sha256'],'Bound parent metadata differs')
        self.external[path.relative_to(ROOT).as_posix()]=pin['sha256'];return read(path)
    def finish(self):
        require(sha(self.receipt_path)==self.receipt_sha,'Import receipt changed')
        for name,identity in self.identities.items():
            s=(self.root/name).stat();require((s.st_size,s.st_mtime_ns,s.st_ino)==identity,'Read evidence changed at close')
        for name,digest in self.external.items():require(sha(ROOT/name)==digest,'External metadata changed at close')
        return {'import_receipt_sha256':self.receipt_sha,'returned_metadata_and_static_files_freshly_hashed':self.pins,
            'external_metadata_sha256':self.external,'array_or_checkpoint_payloads_read':0}


def expected_events(fitted,heldout):
    counts={k:0 for k in COUNTS};rows=[]
    for cohort,source in [('fitted',fitted),('heldout',heldout)]:
        for item in source:
            names=(['CodeFormer_new_original','CodeFormer_new_full_coarse']if cohort=='heldout'else[])+[
                'controller_'+cohort+'_initial','controller_'+cohort+'_final']
            for name in names:
                counts[name]+=1;rows.append({'sequence':len(rows)+1,'name':name,'counts':dict(counts),'call_completion_proved':False})
    require(len(rows)==656 and counts==COUNTS,'Independent source chronology census differs')
    return rows


def summaries(records):
    groups=[]
    for cohort in ('fitted','heldout'):
        for role in ('dark','normal'):
            for head in ('final','initial'):
                rows=[r for r in records if(r['cohort'],r['role'],r['head'])==(cohort,role,head)]
                item={'cohort':cohort,'role':role,'head':head,'cases':len(rows)}
                for name in ('delta','loss','signed_residual'):
                    values=[r[name]for r in rows];item[name]={'minimum':min(values),'mean':math.fsum(values)/len(values),'maximum':max(values)}
                item.update(negative_offsets=sum(r['delta']<0 for r in rows),positive_offsets=sum(r['delta']>0 for r in rows),zero_offsets=sum(r['delta']==0 for r in rows))
                groups.append(item)
    return sorted(groups,key=lambda x:(x['cohort'],x['role'],x['head']))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--import-receipt',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/codeformer_fixed_state_response_v1_return_review/state_v1')
    a=parser.parse_args(argv);require(not a.out.exists(),'Preserve prior state audit');e=Evidence(a.import_receipt)
    a.out.mkdir();start=time.monotonic();report={'format':'codeformer-fixed-state-independent-metadata-state-audit-v1','complete':False,
        'archive_sha256':ARCHIVE,'protocol_sha256':PROTOCOL,'source_sha256':sha(__file__),'model_qualified':False}
    try:
        fitted=e.json('fitted_cases.json')['cases'];heldout=e.json('heldout_pairs.json')['cases']
        require(len(fitted)==200 and len(heldout)==64 and len({r['reference_id']for r in fitted})==100 and
            len({r['reference_id']for r in heldout})==32 and not{r['source_sha256']for r in fitted}&{r['source_sha256']for r in heldout},
            'Exact fitted/heldout photograph census or source disjointness differs')
        require(Counter(r['role']for r in fitted)=={'normal':100,'dark':100}and
            Counter(r['role']for r in heldout)=={'normal':32,'dark':32},'Declared role census differs')
        events=expected_events(fitted,heldout)
        for number,event in enumerate(events,1):require(e.json(PREFIX+'trace/%04d.json'%number)==event,'Actual reservation chronology differs')
        run=e.json(PREFIX+'run_results.json');require(run['complete']is run['measurement_complete']is True and
            run['counts']==COUNTS and run['finished_neural_calls']==656 and run['last_reserved_call']==events[-1]and
            all(type(run[k])is int and run[k]==0 for k in ('head_updates_this_run','restorer_updates_this_run','gradient_queries',
                'backward_calls','optimizer_constructions','optimizer_steps','committed_training_updates'))and
            run['new_checkpoints_written']is False and run['automatic_retry']is False and run['app_promoted']is False and
            run['model_qualified']is False and PREFIX+'failure.json'not in e.hashes,'Actual finite zero-update terminal differs')
        require(not any(Path(n).suffix.lower()in('.pt','.pth','.ckpt','.pkl','.pickle','.safetensors')for n in e.hashes),'Unexpected new neural state payload')
        state=e.json(PREFIX+'initial_state.json');final=e.json(PREFIX+'final_state.json');factory=e.json(PREFIX+'factory.json')
        require(state==final==run['frozen_state_verified']and state['complete']is True and
            state['canonical_tensor_state_sha256']==DIGESTS and state['all_eval_frozen_grad_None']is True and
            state['gradient_queries']==state['backward_calls']==state['optimizer_constructions_or_steps']==0,'Frozen model/head source witnesses differ')
        require(factory['complete']is True and factory['canonical_tensor_state_sha256']==DIGESTS and
            factory['asset_sha256']=={k:v['sha256']for k,v in e.p['assets'].items()}and
            factory['construction_RNG_preserved_exact']is True and factory['historical_RNG_installed']is False and
            factory['optimizer_constructed_or_state_installed']is False and factory['neural_forwards']==0 and
            factory['all_modules_FP32_CUDA_eval_requires_grad_false_grad_None']is True,'Actual factory/frozen/RNG source receipt differs')
        runtime=e.json(PREFIX+'runtime.json');parent_runtime=e.parent_json('tone_training','outputs/tone_control_training_v1/runtime.json')
        for key in ('python','torch','numpy','Pillow','OpenCV','CUDA','device','capability','deterministic_algorithms_enabled','cudnn_deterministic'):
            require(runtime[key]==parent_runtime[key],'Exact parent runtime/backend differs')
        require(sorted(runtime['pip_freeze'].splitlines())==sorted(parent_runtime['pip_freeze'].splitlines())and
            all(runtime[k]is False for k in ('TF32','autocast','cudnn_benchmark','universal_determinism_claim')),'Package/precision/source claims differ')
        resources=run['resources'];limits=e.p['limits']
        require(0<=resources['peak_RSS_bytes']<=limits['process_RSS_bytes']and
            0<=resources['peak_CUDA_reserved_bytes']<=limits['CUDA_reserved_bytes']and
            resources['minimum_sampled_free_bytes']>=limits['reserve_free_bytes']and resources['free_space_is_sampled']is True and
            0<=run['seconds']<=limits['worker_seconds']and set(run['phase_seconds'])=={'data','initialization','measurement'}and
            all(0<=v<=limits[k+'_seconds']for k,v in run['phase_seconds'].items()),'Reported resource/phase bounds differ')
        require(abs(sum(run['phase_seconds'].values())-run['seconds'])<.01,'Phase-time sum differs beyond auditreadout bound')
        indexed=e.json(PREFIX+'input_index.json');require(indexed['complete']is True and indexed['fitted_observations']==200 and
            indexed['heldout_observations']==64 and indexed['fitted_parent_RAW_arrays']==400 and indexed['new_DEV_FINAL_pixels_opened']==0 and
            indexed['neural_calls']==0 and indexed['reference_ids_are_not_proven_person_ids']is True,'Actual data index scope differs')
        records=[];paired=[]
        for cohort,source in [('fitted',fitted),('heldout',heldout)]:
            for row in source:
                before_after={}
                for head in ('initial','final'):
                    record=e.json(PREFIX+head+'/'+row['id']+'/record.json')
                    require(record['id']==row['id']and record['cohort']==cohort and record['head']==head and record['role']==row['role']and
                        record['reference_id']==row['reference_id']and record['source_sha256']==row['source_sha256']and
                        record['support_is_not_covering_mask']is True and record['clean_pixels_used_for_target']is False and
                        record['independent_FINAL_or_native_evaluation']is False and
                        record['fitted_parent_arrays_reused_without_new_CodeFormer_calls']is(cohort=='fitted'),'Actual response namespace/role/source scope differs')
                    require(-.25<=record['delta']<=.25 and all(math.isfinite(record[k])for k in ('delta','mean_y','loss','target_mean_y','signed_residual')),
                        'Finite response scalar required')
                    close(record['signed_residual'],record['mean_y']-record['target_mean_y'])
                    residual=f32(f32(record['mean_y'])-f32(record['target_mean_y']));require(record['loss']==f32(residual*residual),'Actual FP32 squared-mean loss differs')
                    for key,filename in [('RAW','raw.npy'),('PNG','native.png')]:
                        name=PREFIX+head+'/'+row['id']+'/'+filename;pin=record[key]
                        require(pin=={'path':name,'bytes':e.sizes[name],'sha256':e.hashes[name]},'Output record/physical map binding differs')
                    if cohort=='fitted':
                        cache=e.parent_json('tone_training',row['cache_path'])
                        require(record['target_mean_y']==cache['target_mean_y']and record['features']==cache['features']and
                            all(record['sourcebindings'].get(k)==v for k,v in cache['sourcebindings'].items()),
                            'Fitted target/feature/source witness differs')
                        require(record['target_policy']=='exact_retained_fitted_target_mean'and record['new_parent_RAW']is None,'Cached parent relabelled as fresh')
                    else:
                        require(record['sourcebindings']==e.json(row['pair_record_path'])and record['target_policy']==
                            ('fresh_frozen_original_mean'if row['role']=='normal'else'fresh_frozen_base_mean'),'New target/record witness differs')
                        for key,filename in [('original_raw','original_raw.npy'),('base_raw','base_raw.npy')]:
                            name=PREFIX+'new/'+row['id']+'/'+filename
                            require(record['new_parent_RAW'][key]=={'path':name,'sha256':e.hashes[name],'bytes':e.sizes[name]},'Fresh parent output map differs')
                    if head=='initial':require(record['delta']==0,'Actual initial head offset not zero')
                    records.append(record);before_after[head]=record
                x,y=before_after['initial'],before_after['final']
                require(x['target_mean_y']==y['target_mean_y']and x['features']==y['features']and x['pair_id']==y['pair_id'],
                    'Fixed-state comparison changes input/target/pair')
                paired.append({'id':row['id'],'reference_id':row['reference_id'],'cohort':cohort,'role':row['role'],
                    'initial_loss':x['loss'],'final_loss':y['loss'],'loss_change':y['loss']-x['loss'],
                    'initial_signed_residual':x['signed_residual'],'final_signed_residual':y['signed_residual'],
                    'final_delta':y['delta'],'target_mean_y':x['target_mean_y'],'desired_mean_change':x['target_mean_y']-x['mean_y']})
        saved=e.json(PREFIX+'response_summary.json');groups=summaries(records)
        require(saved['groups']==groups and saved['paired_fixed_state_comparisons']==sorted(paired,key=lambda r:r['id'])and
            saved['new_acceptance_threshold_or_selection']is False and saved['model_qualified']is False,'Independent fixed-state scalar summary differs')
        comparisons=[]
        for cohort in ('fitted','heldout'):
            for role in ('normal','dark'):
                selected=[r for r in paired if(r['cohort'],r['role'])==(cohort,role)]
                comparisons.append({'cohort':cohort,'role':role,'cases':len(selected),
                    'initial_MSE':math.fsum(r['initial_loss']for r in selected)/len(selected),
                    'final_MSE':math.fsum(r['final_loss']for r in selected)/len(selected),
                    'improved':sum(r['loss_change']<0 for r in selected),'worsened':sum(r['loss_change']>0 for r in selected),
                    'same':sum(r['loss_change']==0 for r in selected),'final_negative_offsets':sum(r['final_delta']<0 for r in selected),
                    'final_mean_offset':math.fsum(r['final_delta']for r in selected)/len(selected)})
        require(len(records)==528 and len(paired)==264,'Full response record census differs')
        old_audit=e.local(ROOT/'outputs/codeformer_tone_control_training_v1_return_review/state_v1/state_audit_v1.json')
        require(old_audit['complete']is old_audit['evidence_consistency_pass']is True and
            old_audit['sections']['checkpoints']['checkpoints']['stopped_composite']['head_tensor_sha256']==DIGESTS['head0050'],
            'Prior physical checkpoint state proof differs')
        report.update(complete=True,evidence_consistency_pass=True,UTC=datetime.now(timezone.utc).isoformat(),
            operation_census={'reservation_events':656,'finished_calls_recorded':656,'head_records':528,'fresh_parent_RAW_members':128,
                'counts':COUNTS,'individual_reservations_are_not_completion_or_profiler_evidence':True,
                'completed_call_scope':'Final counter, complete output bindings and frozen-source execution order; independent pixel decoding is separate'},
            frozen_state_receipts=state,factory=factory,runtime=runtime,reported_resources=resources,
            reported_worker_seconds=run['seconds'],reported_phase_seconds=run['phase_seconds'],
            all264_final_offsets_negative=all(r['final_delta']<0 for r in paired),fixed_state_comparisons=comparisons,
            independently_recomputed_groups=groups,paired_fixed_state_comparisons=paired,
            conclusion='Actual final fitted-source dark leakage rules out a DEV-only generalization explanation; heldout failure is similar. Unique optimizer/objective/capacity cause remains unresolved.',
            gradient_queries=0,backward_calls=0,optimizer_constructions=0,optimizer_steps=0,committed_training_updates=0,
            no_new_checkpoint_members=True,local_framework_imports=0,local_neural_or_manual_head_replay=0,
            evidence_binding=e.finish(),seconds=time.monotonic()-start,
            limits=['Frozen515/4digests and live mode/Nonegradient status are source-bound VM captures, linked to previously independently audited physical checkpoints. No tensors were returned or locally reloaded for a new tensor-equality proof.',
                'Zero optimization is supported by frozen source plus counters/records; no independent profiler/gradient execution trace exists.',
                'Runtime/resource values are reported by the VM with sampled disk minima, not locally measured CUDA peaks.',
                'Summary MSE measures behavioral parent-mean targets, not clean-reference restoration quality, geometry or identity.',
                'Fitted/heldout photographic sources are exact-file disjoint, not proved person-identity or pretraining disjoint.',
                'Actual RAW/PNG/feature/target pixel arithmetic is owned by a separate independent CPU audit.'])
    except BaseException as error:
        report.update(error_type=type(error).__name__,error=str(error),seconds=time.monotonic()-start)
        with(a.out/'failure.json').open('x',encoding='utf-8')as stream:json.dump(report,stream,indent=2);stream.write('\n')
        raise
    with(a.out/'audit.json').open('x',encoding='utf-8')as stream:json.dump(report,stream,indent=2);stream.write('\n')
    with(a.out/'STATE_FINDINGS.md').open('x',encoding='utf-8')as stream:
        stream.write('# Fixed-state metadata readback\n\nPASS. All656prospective reservations, the final656completed-call counter,528response records and128newparentRAW bindings agree with the frozen source census. Counts remainzero for gradients,backwards,optimizer construction/steps,head/restorer updates and committed training. No new checkpoints exist.\n\nFactory/initial/final source captures agree on the original/base515-tensor and two4-tensor head digests. Runtime/package/precision flags match the bound training parent; reportedworker/phase/resourcecaps pass. These are source-bound runtime captures linked to the prior physical checkpoint audit, not a new live tensor/profiler replay.\n\nAll264fixed-final offsets are negative. Dark preservation fails already on100fitted observations and similarly on32heldout sources; a purely DEV/generalization explanation is disconfirmed. Objective,optimization and representation causes remain unseparated. Behavioralmean MSE is not facial restoration quality. Independent savedpixel audit remains separate.\n\nNo neural framework,array/checkpoint load,manualhead replay,autograd,optimizer or VM run was performed by this auditor.\n')
    print(json.dumps({'complete':True,'audit_sha256':sha(a.out/'audit.json'),'fixed_state_comparisons':report['fixed_state_comparisons'],'seconds':report['seconds']}))


if __name__=='__main__':main()
