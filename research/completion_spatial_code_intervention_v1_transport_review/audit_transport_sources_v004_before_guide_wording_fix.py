"""Independent transport004 delta: unchanged exporter reuses prior actual fixtures; new shell tested."""
import ast,hashlib,importlib.util,json,shlex,subprocess
from pathlib import Path
B=Path(r'C:\xampp\htdocs\YEAR 4\Testing_FACE_MASK_COMPLETION');H=B/'research/completion_spatial_code_intervention_v1_transport_review';BUILDER=B/'scripts/prepare_completion_codeformer_spatial_code_intervention_v1.py';EXPORT=B/'scripts/export_completion_codeformer_spatial_code_intervention_v1.py';RUN=B/'scripts/run_completion_codeformer_spatial_code_intervention_v1.sh';checks=[]
def pin(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def need(ok,label):
 if not ok:raise ValueError(label)
 checks.append(label)
prior=json.loads((H/'independent_transport_source_review.json').read_text());need(prior['complete'] is True and prior['checks_passed']==61 and len(prior['actual_export_fixtures'])==9,'Reuse exact admitted61checks+9actualfilefixtures')
oldexport=next(r for r in prior['source_pins'] if Path(r['path']).name==EXPORT.name);need(pin(EXPORT)==oldexport,'Exporter exactunchanged fd7; no unrelated fixture rerun')
spec=importlib.util.spec_from_file_location('independent_spatial004_builder',BUILDER);builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder);need(hashlib.sha256(builder.INSTALL_CODE.encode()).hexdigest()==prior['exact_installer_literal_sha256'],'Installer literal exactunchanged root9fixtures')
for p in (BUILDER,EXPORT):ast.parse(p.read_text(),feature_version=(3,10))
shell=RUN.read_text();need('command -v timeout >/dev/null || exit 1' in shell and shell.index('command -v timeout')<shell.index('--run'),'Timeout dependency checkedbeforework');need('timeout --signal=KILL 900s "$PY" -B -u scripts/export_completion_codeformer_spatial_code_intervention_v1.py --root "$ROOT" --protocol-sha "$EXPECTED"' in shell,'Owned GNUtimeout external900s exactexportcommand');need('export_exit=$?' in shell and 'if test "$export_exit" -ne 0; then exit "$export_exit"; fi' in shell,'Nonzero externalexport status propagated');need('--foreground' not in shell and '&' not in shell,'No unownedbackground orforegroundtimeout mode')
bash=Path(r'C:\Program Files\Git\bin\bash.exe');syntax=subprocess.run([str(bash),'-n',str(RUN).replace('\\','/')],capture_output=True,text=True,timeout=10);need(syntax.returncode==0,'ActualBash syntax passes')
toy=subprocess.run([str(bash),'--noprofile','--norc','-c','PATH=/usr/bin:/bin; timeout --signal=KILL 0.1s bash -c "sleep 3"; observed=$?; printf "owned_toy_timeout_exit=%s\\n" "$observed"; test "$observed" -eq 137'],capture_output=True,text=True,timeout=10);need(toy.returncode==0 and 'owned_toy_timeout_exit=137' in toy.stdout,'Actual localGNUtimeout kills onlyownedshortsleep andreturns137')
protocol='b27617eb4f2a2225e0969d463acc8048a148203488f0fcd5397e5c5f9a0a0123';guide=builder.manual_guide(protocol,'b'*64,123,166);need('\r' not in guide and '__PROTOCOL' not in guide and '__INSTALL' not in guide,'Final template LF no placeholders');need('900' in guide and 'SIGKILL' in guide and 'partial' in guide and 'quiet' in guide.lower(),'Guide externalhardkill/partial/quiet truth')
commands=[line for line in guide.splitlines() if line.startswith('gcloud compute scp ')];need(len(commands)==5,'FinalfiveSCPcommands')
for i,line in enumerate(commands):
 args=[v for v in shlex.split(line)[3:] if not v.startswith('--')];need(len(args)==2 and sum(v.startswith('janusdominic0@forensic-dgp-thesis:') for v in args)==1 and '--project=forensic-dgp-thesis' in line and '--zone=us-central1-a' in line,'Finalsingle-remote exactproject/zone')
 if i<2:need(args[1]=='janusdominic0@forensic-dgp-thesis:/home/janusdominic0/','Finalupload direction')
 else:need(args[1]=='.','Finaldownload direction')
need(guide.count('cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"')==2 and 'tmux new-session -A -s completion_spatial_code_intervention_v1' in guide,'Windowsfolders andtmux unchanged')
need("python3 - <<'PY'\n"+builder.INSTALL_CODE.replace('__PROTOCOL_SHA__',protocol)+'PY\n' in guide,'Finalguide exactpreviousroot9testedinstaller')
result={'format':'spatial-code-intervention-v1-independent-transport-source-review-v004','complete':True,'independent':True,'protocol_sha256':protocol,'source_pins':[pin(p) for p in (BUILDER,EXPORT,RUN)],'checks_passed':len(checks),'checks':checks,'prior_actual_source_review':pin(H/'independent_transport_source_review.json'),'unchanged_exporter_and_installer_actual_fixtures_reused':True,'new_Bash_syntax_fixture':{'exit':syntax.returncode,'stderr':syntax.stderr},'new_owned_timeout_fixture':{'exit':toy.returncode,'stdout':toy.stdout,'stderr':toy.stderr,'duration_seconds':0.1,'not_actual_VM900second_measurement':True},'new_VM_calls':0,'new_NN_calls':0,'actual_training_calls':0,'model_qualified':False,'archive_ready_claim':False,'final_archive_guide_review_still_pending':True}
p=H/'independent_transport_source_review_v004.json'
with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'receipt':pin(p),'checks':len(checks)},indent=2))
