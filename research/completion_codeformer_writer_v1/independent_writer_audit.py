"""Independent pure guard/ledger review; tiny synthetic extents, no models."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,importlib.util,json,os,sys,time,uuid
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.completion_codeformer_coverage_v2 import contracts as old
from research.completion_codeformer_writer_v1 import optimized_budget as new
OUT=ROOT/'research/completion_codeformer_writer_v1'
FIX=Path(r'C:\xampp\htdocs\YEAR 4\Testing\outputs')/('writer_independent_'+uuid.uuid4().hex[:10]);FIX.mkdir()
def need(ok,label):
    if not ok:raise ValueError(label)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pin(path):path=Path(path);return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}
def seeded(cls,label,rows=(),external=0,free=100*1024**3,now=0,committed=False):
    folder=FIX/label;folder.mkdir();b=cls(folder,0,clock=lambda:now,disk_free=lambda:free)
    for relative,value in rows:
        p=folder/relative
        if cls is new.OutputBudget:
            category=b.classify(p);entry=new.Entry(p,category,value);b.entries[p]=entry;b.totals[category]+=value;b.extent_total+=value
        b.sizes[p]=value
    if external:
        ext=FIX/(label+'_external.log');ext.write_bytes(b'x'*external);b.metadata_external=[ext]
    b.terminal_committed=committed
    return b
def outcome(call):
    try:call();return {'allowed':True}
    except (ValueError,OSError) as exc:return {'allowed':False,'error_type':type(exc).__name__,'message':str(exc)}
def main():
    started=time.monotonic();source=Path(new.__file__);old_source=Path(old.__file__);initial_new=pin(source);initial_old=pin(old_source);need(initial_old['sha256']=='1822d73ebe57b922cd9e79817ba5c961582a93e65935ef442f60ae07576ec442','Immutableoldsource drift')
    contract=old.CONTRACT;checks=[];count=0
    scenarios=[
      ('ordinary_allowed','x.npz',64,{},True),
      ('deadline_normal_before','x.npz',1,{'now':1679.99},True),
      ('deadline_normal_at','x.npz',1,{'now':1680},False),
      ('deadline_terminal_before','terminal_training_state.pth.partial',1,{'now':1799.99,'terminal':True},True),
      ('deadline_terminal_at','terminal_training_state.pth.partial',1,{'now':1800,'terminal':True},False),
      ('deadline_emergency','failure.json',1,{'now':1799,'emergency':True},True),
      ('committed_postdeadline_allowed','results.json',1,{'now':1799,'committed':True},True),
      ('committed_deadline_at','results.json',1,{'now':1800,'committed':True},False),
      ('teacher_exact_cap','teacher_labels/x.npz',contract['teacher_bytes'],{},True),
      ('teacher_overcap','teacher_labels/x.npz',contract['teacher_bytes']+1,{},False),
      ('terminal_exact_cap','terminal_training_state.pth.partial',contract['terminal_bytes'],{'terminal':True},True),
      ('terminal_overcap','terminal_training_state.pth.partial',contract['terminal_bytes']+1,{'terminal':True},False),
      ('PNG_exact_cap','x.png',contract['PNG_bytes_each'],{},True),
      ('PNG_overcap','x.png',contract['PNG_bytes_each']+1,{},False),
      ('metadata_normal_exact','x.json',contract['metadata_bytes']-1024**2-512*1024,{},True),
      ('metadata_normal_over','x.json',contract['metadata_bytes']-1024**2-512*1024+1,{},False),
      ('metadata_emergency_exact','failure.json',contract['metadata_bytes']-1024**2,{'emergency':True},True),
      ('metadata_emergency_over','failure.json',contract['metadata_bytes']-1024**2+1,{'emergency':True},False),
      ('disk_preterminal_floor_exact','x.npz',0,{'free':contract['stop_free_bytes']+contract['terminal_bytes']+1024**2},True),
      ('disk_preterminal_floor_under','x.npz',0,{'free':contract['stop_free_bytes']+contract['terminal_bytes']+1024**2-1},False),
      ('disk_terminal_floor_exact','terminal_training_state.pth.partial',0,{'free':contract['stop_free_bytes'],'terminal':True},True),
      ('disk_terminal_growth_refused','terminal_training_state.pth.partial',1,{'free':contract['stop_free_bytes'],'terminal':True},False),
      ('whole_output_exact_terminal','terminal_training_state.pth.partial',contract['terminal_bytes'],{'terminal':True,'preexisting_array':contract['output_bytes']-contract['terminal_bytes']-1024**2},True),
      ('whole_output_over_terminal','terminal_training_state.pth.partial',contract['terminal_bytes'],{'terminal':True,'preexisting_array':contract['output_bytes']-contract['terminal_bytes']-1024**2+1},False),
      ('preterminal_reservation_exact','x.npz',1,{'preexisting_array':contract['output_bytes']-contract['terminal_bytes']-2*1024**2-1},True),
      ('preterminal_reservation_over','x.npz',1,{'preexisting_array':contract['output_bytes']-contract['terminal_bytes']-2*1024**2},False),
      ('postcommit_headroom_exact','x.npz',1,{'committed':True,'preexisting_array':contract['output_bytes']-2*1024**2-1},True),
      ('postcommit_headroom_over','x.npz',1,{'committed':True,'preexisting_array':contract['output_bytes']-2*1024**2},False),
      ('external_metadata_normalcap','x.json',contract['metadata_bytes']-1024**2-512*1024-10,{'external':10},True),
      ('external_metadata_overcap','x.json',contract['metadata_bytes']-1024**2-512*1024-9,{'external':10},False)]
    for name,relative,end,options,expected in scenarios:
        count+=1;results={}
        for cls,label in [(old.OutputBudget,'old'),(new.OutputBudget,'new')]:
            rows=[]
            if options.get('preexisting_array'):rows.append(('preexisting.npz',options['preexisting_array']))
            rows.append((relative,0))
            b=seeded(cls,label+'_'+name,rows,external=options.get('external',0),free=options.get('free',100*1024**3),now=options.get('now',0),committed=options.get('committed',False));target=b.out/relative
            results[label]=outcome(lambda:b.before_write(target,end,terminal=options.get('terminal',False),emergency=options.get('emergency',False)))
        need(results['old']['allowed']==results['new']['allowed']==expected,'Old/newpolicymismatch '+name);checks.append({'case':name,'expected_allowed':expected,'outcomes':results})
    # Tinyrealwrites exercise seek/headerpatch extent accounting without large files.
    b=seeded(new.OutputBudget,'real_tiny_ledger');target=b.out/'data.npz'
    with b.open(target) as stream:
        stream.write(b'abcdefgh');need(b.total()==8 and b.total('array')==8,'Initial extent')
        stream.seek(0);stream.write(b'XY');need(b.total()==8 and stream.tell()==2,'Headerpatch doublecount')
        stream.seek(12);stream.write(b'Z');need(b.total()==13 and stream.tell()==13,'Seekhole growth')
        stream.seek(4);stream.write(b'123456789');need(b.total()==13,'Overwriteextentmax')
        need(not hasattr(stream,'fileno'),'Nativefileno bypass exposed')
    need(target.read_bytes()==b'XYcd123456789','Seekoutputbytes');need(b.assert_ledger_consistent(),'Ledgerconsistency')
    checks.append({'case':'real_seek_overwrite_and_sparseextent','allowed':True,'bytes':13,'no_public_fileno':True})
    # Closed/repeated/unknown paths do not enter ledger orwritebytes.
    checks.append({'case':'closed_writer_write_refused','outcome':outcome(lambda:stream.write(b'bad'))});need(not checks[-1]['outcome']['allowed'],'Closedwrite allowed')
    for name,call in [('existing_owned_reopen',lambda:b.open(target)),('unknown_unopened_write',lambda:b.before_write(b.out/'unknown.npz',1)),('traversal',lambda:b.open(b.out/'../escape.npz')),('outside_root',lambda:b.open(FIX/'outside.npz')),('unknown_suffix',lambda:b.open(b.out/'x.txt'))]:
        result=outcome(call);need(not result['allowed'],'Pathguardfailed '+name);checks.append({'case':name,'outcome':result})
    before=b.total();rename_to=b.out/'renamed.npz';target.rename(rename_to);b.rename(target,rename_to);need(b.total()==before and b.sizes[rename_to]==13 and b.assert_ledger_consistent(),'Renameledger changedextent')
    result=outcome(lambda:b.rename(rename_to,b.out/'renamed.json'));need(not result['allowed'],'Crosscategoryrenameallowed');checks.append({'case':'rename_sametype_extent_exact_crosscategory_refused','outcome':result})
    term=b.out/'terminal_training_state.pth.partial'
    with b.open(term,terminal=True) as f:f.write(b'abc')
    committed=b.out/'terminal_training_state.pth';term.rename(committed);b.rename(term,committed);need(b.terminal_committed and b.total('terminal')==3 and b.assert_ledger_consistent(),'Terminalcommitledger')
    checks.append({'case':'terminal_rename_commits_without_recount','passed':True})
    # Simulated pathsymlink detection is deterministic underWindows withoutprivilege.
    bad=b.out/'linked.npz';orig=Path.is_symlink
    with patch.object(Path,'is_symlink',lambda path:True if Path(path)==bad else orig(path)):
        result=outcome(lambda:b.open(bad));need(not result['allowed'],'Symlinktargetallowed')
    checks.append({'case':'symlink_target_refused_mock','outcome':result,'scope':'Pathlinkflagfixture; noOSsymlinkcreated'})
    rootlink=FIX/'rootlink';rootlink.mkdir()
    with patch.object(Path,'is_symlink',lambda path:True if Path(path)==rootlink else orig(path)):
        result=outcome(lambda:new.OutputBudget(rootlink,0,clock=lambda:0,disk_free=lambda:100*1024**3));need(not result['allowed'],'Symlinkrootflagallowed')
    checks.append({'case':'symlink_root_flag_refused_mock','outcome':result,'scope':'Directrootflag fixture; a resolve-target symlinkpath needs separateactualtest'})
    # Complexityproof byoperation count, independent ofmachine timing.
    scale=[]
    for count in (1,20,620):
        rows=[('virtual/f%04d.npz'%i,1) for i in range(count)]+[('terminal_training_state.pth.partial',0)]
        ledger=seeded(new.OutputBudget,'scale_'+str(count),rows);entry=ledger.entries[ledger.out/'terminal_training_state.pth.partial'];before=dict(ledger.check_counts)
        original_resolve=Path.resolve
        def forbid_resolve(*args,**kwargs):raise RuntimeError('Perwrite pathresolve forbidden')
        class NoScan(dict):
            def items(self):raise RuntimeError('Perwrite namespaceitems scan forbidden')
            def values(self):raise RuntimeError('Perwrite namespacevalues scan forbidden')
            def __iter__(self):raise RuntimeError('Perwrite namespaceiteration forbidden')
        ledger.entries=NoScan(ledger.entries);ledger.sizes=NoScan(ledger.sizes)
        with patch.object(Path,'resolve',forbid_resolve):
            for _ in range(10):ledger.before_entry_write(entry,1,terminal=True)
        after=dict(ledger.check_counts);need(after['canonical_paths']==before['canonical_paths'],'Perwrite canonicalization')
        scale.append({'virtual_entries':count+1,'writechecks':10,'canonical_paths_added':after['canonical_paths']-before['canonical_paths'],'path_components_added':after['path_components']-before['path_components'],'guardcalls_added':after['guard_calls']-before['guard_calls'],'external_stats_added':after['external_stats']-before['external_stats'],'allnamespace_iteration_and_pathresolve_forbidden':True})
    benchmark=json.loads((OUT/'namespace_scale_benchmark_v2.json').read_text())
    policy=json.loads((OUT/'policy_tests_v2.json').read_text());tensor_fixture=json.loads((OUT/'tensor_32MiB_scale_fixture.json').read_text())
    need(policy['complete'] and policy['tests_run']==18 and policy['failures']==policy['errors']==0,'Author policy fixtures failed')
    need(tensor_fixture['complete'] and tensor_fixture['prototype_sha256']==initial_new['sha256'] and tensor_fixture['fixture_tensors']==996 and tensor_fixture['write_callbacks']==5014 and tensor_fixture['canonical_path_calls']==2 and tensor_fixture['all996tensors_safe_CPU_readback_equal'] and tensor_fixture['allZIP_CRC_pass'],'Bounded native serialization evidence failed')
    need(benchmark['prototype_source']['sha256']==initial_new['sha256'] and benchmark['immutable_old_source']['sha256']==initial_old['sha256'],'Benchmarksourcebinding differs')
    need(benchmark['actual_return_namespace']['tracked_output_entries']==620,'Benchmarkednamespace notactual620')
    need(len({r['output_sha256'] for r in benchmark['bounded8MiB_native_old_new_writes']})==1,'Native/old/newserialization-equivalent bytes differ')
    need(pin(source)==initial_new and pin(old_source)==initial_old,'Source changedduringaudit')
    need(not any(n=='torch' or n=='pretrained_completion' or n.startswith('third_party.codeformer') for n in sys.modules),'Modelruntimeimported')
    report={'format':'completion-optimized-writer-independent-admission-v1','complete':True,'independent':True,'audited_UTC':datetime.now(timezone.utc).isoformat(),'auditor':pin(Path(__file__)),'source':initial_new,'immutable_old_source':initial_old,'contract_unchanged':True,'old_new_guard_equivalence_checks':checks,'equivalence_boundary_scenarios':len(scenarios),'namespace_constant_operation_proof':scale,'benchmark_evidence':pin(OUT/'namespace_scale_benchmark_v2.json'),'author_policy_test_source':pin(OUT/'test_optimized_budget.py'),'author_policy_receipt':pin(OUT/'policy_tests_v2.json'),'author_32MiB_native_serialization_source':pin(OUT/'tensor_scale_fixture.py'),'author_32MiB_native_serialization_receipt':pin(OUT/'tensor_32MiB_scale_fixture.json'),'author_serialization_evidence_not_freshly_rerun_by_independent_auditor':True,'benchmark_local_report_not_GPU_save_guarantee':True,'newsource_pathvalidation_strengthened':True,'no_policy_weakening_observed':True,'integration_status':'Researchhelperonly; not used byhistoricalV2/anytrainingpacket orapp. Runtimewrapping/serializer source must be independently admitted before futuremanualpilot.','fixture_directory_retained':str(FIX),'limits':['Opening validatespathbeforeexclusiveownedhandle; writesusecachedEntryandownedhandle. No adversarialexternaldirectoryreplacement race certification.','Performanceislocalmechanismproof; solecause ofactualVMsave timeoutnotproven.','Thisadmission doesnot recoverincompleteV2state orqualifycompletionquality.'],'local_execution':{'Torch_imported':False,'model_loads':0,'neural_calls':0,'gradients':0,'optimizer_steps':0,'VM_calls':0},'model_qualified':False,'next_training_packet_qualified':False,'seconds':time.monotonic()-started}
    dest=OUT/'independent_writer_admission.json'
    with dest.open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'receipt':pin(dest),'boundarychecks':len(scenarios),'totalchecks':len(checks),'scale':scale,'seconds':report['seconds']}))
if __name__=='__main__':main()