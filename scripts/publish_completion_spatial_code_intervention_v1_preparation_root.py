"""Publish verified manual-diagnostic readiness, retaining exact prior documents."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/completion_codeformer_spatial_code_intervention_v1_packet/draft_004'
STAGE=BASE/'cctv_completion_codeformer_spatial_code_intervention_v1_vm'
TRANSFER=Path('C:/xampp/htdocs/YEAR 4/Testing/outputs')
GUIDE='COMPLETION_CODEFORMER_SPATIAL_CODE_INTERVENTION_V1_VM.md'
PROTOCOL='b27617eb4f2a2225e0969d463acc8048a148203488f0fcd5397e5c5f9a0a0123'
ARCHIVE='24f480add44e4fe8176a68a3b67321b11b4baee0ebb43d0a3790fefc434549a5'

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
def new(path,body):
    with physical(path).open('xb') as f:f.write(body)

PREPARATION='''# Spatial-code intervention V1: verified manual diagnostic

27 October 2026. **PREPARED, UNRUN; ZERO TRAINING UPDATES.** R1 remains the
selected experimental completion model. [Use the five-step guide](COMPLETION_CODEFORMER_SPATIAL_CODE_INTERVENTION_V1_VM.md).

The last [predictor training result](COMPLETION_CODEFORMER_PREDICTOR_CE_V1_RESULTS.md)
trained correctly but failed quality. Its saved state, all256images and20sheets
are audited and retained. The next test measures where a distinct training
change could help before spending more updates on the failed recipe.

## What the diagnostic does

Keep all497 exact R1 model tensors frozen. Use the same48 adaptive paired TRAIN
cases and reviewed masks. H substitutes existing clean-context same-codebook
indices only at geometric cells touched by removal; V substitutes only in the
visible complement. Reuse the previously audited ordinary A and full-oracle C
controls. H/V/C are deliberately unavailable-clean-information diagnostics;
they cannot be used as app outputs or claims of recovered hidden identity.

Four fresh ordinary/app parity pairs(8calls) and four fresh C bridge checks
precede96newH/V calls:108full forwards planned, zero optimizer objects, updates,
gradients, backwards or teacher-network calls. Separate quantizer and generator
witnesses join every call. No new learned checkpoint is made on normal success.
Save96H/VPNGs and104NPZs, with exact copied outside pixels. Return all3download
files for independent raw/PNG, state/code-selection and all48/four-condition
visual review. A completed diagnostic is not a qualified replacement.

The geometric rule is fixed: any reviewed removal pixel in each16x16original256
block marks that16x16code-grid cell touched. No mask enlargement, dilation or
threshold search. It is a geometric proxy, not a neural receptive-field map.
No new targets, data, incompatible codebook or clean RGB pasting is introduced.
Code agreement alone cannot establish anatomy/style; original quality gates
remain binding for any future trained candidate. FINAL identities remain unused.

## Files and manual execution

The archive, LFchecksum and guide are in C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs.
Archive:cctv-completion-codeformer-spatial-code-intervention-v1-execution.tar.gz,
13,610,328bytes. SHA256:
`24f480add44e4fe8176a68a3b67321b11b4baee0ebb43d0a3790fefc434549a5`.
Protocol SHA256:
`b27617eb4f2a2225e0969d463acc8048a148203488f0fcd5397e5c5f9a0a0123`.
The execution archive has167regular members(166pinned payloads plus protocol),
19,491,897uncompressed payload bytes. It reuses the exact original/R1 weights
and196old A/C control files already on the VM; those large files are not uploaded
again. Missing or changed files cause refusal rather than silent recomputation.

Require6GiBfree after install plus the exact payload before extraction. Preserve
2GiBfloor. Worker15minhard/14minwork/60sfinish reserve; allocatedCUDA20GiB.
Export has its own external15minSIGKILL cutoff plus internal time/byte checks.
Quiet trainer.log is expected during inference; use the guide's final supervisor
and export receipts. Do not start a second process because the log is quiet.

Output1GiB; rawtar/gzip1280MiBeach. Numeric arrays560MiB,PNGs24MiB,metadata8MiB
and optional unexpected-state failure384MiB total976MiB. Normal frozen success
writes no checkpoint. Unexpected mutation/nonfinite retains one actual497-tensor
CPUstate with RNG/progress within384MiB, including failedpartials; no optimizer.
Resource-bounded capture does not require the failed scientific check to pass.
IO/hardkill may prevent recovery; retain partials, report failure and never resume
or repeat automatically. No VM connection or actual diagnostic was run in preparation.

## Verification and current limits

The21author fixtures,350independent pure selector checks,2,988independent runtime
checks and1,669final source/packet checks passed. The actual unmodified transfer
CLI exited0 with scientific imports, network and process execution blocked;
all166payloads and196controls verified. These are source/transfer checks, not a
real model preflight, measured inference result or new quality evidence.

Root independently exercised the exact installer in9toy cases. Independent
transport review includes9export fixtures, alias-drift repair and a short owned
local GNUtimeout fixture. Bash-n checks execute no model commands. Both root
and independent final audits streamed all167archive member hashes and fullgzip
CRC, checked both published copies/LFchecksum and the exact five SCPcommands,
tested Python3.10installer literal and tmux/protocol binding. The independent
final archive/guide audit contains1,036checks. Its stopped earlier drafts,
source revisions and tests remain preserved and unreleased.

All22working R1 app/model pins remain exact. No RESTORATION or IMPLEMENTATION
change follows. The full goal remains ACTIVE: useful all-family reviewed-mask
outputs, separate automatic/assisted evaluation, user acceptance, blinded human
FINAL, Windows CPUminimum/fresh installation and accepted selective app port
remain unfinished. This source-verified diagnostic prepares the next learning
decision; it does not replace those requirements.

See [the frozen design](COMPLETION_CODEFORMER_SPATIAL_CODE_INTERVENTION_V1_DESIGN.md),
its root clarifications and independent admissions under
research/completion_codeformer_spatial_code_intervention_v1_independent_review
and research/completion_spatial_code_intervention_v1_transport_review.
The design's earlier source-unready status is historical; this preparation is current.
'''

HANDOFF='''# 27 October 2026: spatial-code diagnostic verified for manual transfer

PREPARED_UNRUN_ZERO_UPDATES_NO_NEW_MODEL. R1 remains selected.
[Five-step upload/install/tmux/download guide](COMPLETION_CODEFORMER_SPATIAL_CODE_INTERVENTION_V1_VM.md).
[Preparation, verification and limits](COMPLETION_CODEFORMER_SPATIAL_CODE_INTERVENTION_V1_PREPARATION.md).
Files are in original Testing/outputs. No VM diagnostic or training launched.

The closed predictor CE trial below trained correctly but failed quality.
Next: frozen497R1 masked-cell H versus visible-context V oracle substitution,
using same48pairedadaptiveTRAIN cases/16existing labels and exact196oldA/C
controlfiles. Fourordinaryparitypairs+fourCbridges before96H/Vcalls:108forwards,
zerooptimizer/backward/gradient/teacher. Alloraclesuseunavailablecleancontext,
notappoutputs orrecoveredidentity. No normalnewcheckpoint ormodelpromotion.

Archive13,610,328B SHA24f480add44e4fe8176a68a3b67321b11b4baee0ebb43d0a3790fefc434549a5.
Protocolb27617eb4f2a2225e0969d463acc8048a148203488f0fcd5397e5c5f9a0a0123.
All167members/fullgzipCRC,actualimportblockedtransferCLIexit0,independentsource/
transport androotguide/installfixtures pass.166payloads+196externalcontrols exact.
Require6GiBfree+installpayload,2GiBfloor,15minhardworker+separate15minexternal
exportcutoff.104NPZ/96PNG;1GiBoutputs/1280MiBtar+gzip;384MiBactualfailurestate
reservationonlyifunexpectedmutation. Quietlogs expected. Allpartialsretained;
neverautomaticrepeat/resume orunchangedfailedtraining. All22R1pins unchanged.

The nextstep is the user's manualtransfer/run, then independentstate/code/pixel
andall48/fourconditionvisualreview beforedecidingnewlearning. ActualL4behavior
andH/Vresults areunverified. FullgoalACTIVE:all-familyusefulness,automatic-assisted
evaluation,useracceptance,humanFINAL,WindowsCPUminimum/freshinstall and accepted
selectiveIMPLEMENTATIONport remain incomplete. RESTORATION unchanged. Earlier
source-unready/preparedtraining entries below are historical; failures retained.

---

'''

SPEC='''

## Spatial-code intervention V1: manual diagnostic ready

27 October 2026. Frozen497R1 H(mask-touchedclean-code) versusV(visible-context
clean-code) diagnostic now source/transfer/archive verified. Reuse exact196A/C
controlfiles andsame48adaptivepairedcases/16labels;108forwards planned,zero
training/optimizer/gradient/teacher. Oraclesareunavailablecleaninformation,not
deployableestimates. Fixedgeometricselector/reviewedmasks;104NPZ/96PNGs,allcase
independentRAW/PNG/code/state/visual review required aftermanualrun.

Follow COMPLETION_CODEFORMER_SPATIAL_CODE_INTERVENTION_V1_VM.md. Require6GiBplus
installpayload,2GiBfloor,15minhardworker andseparate15minexternalexportlimit;
1GiBoutput/1280MiBtar+gzip/384MiBoptionalactualmutationfailurestate. All167
archivemembers/fullCRC/sourceCLI verify. NoactualVMrun ornewmodelcreated.
R1app/model22pins remain exact;RESTORATION/IMPLEMENTATION unchanged. Theprior
32updatepredictortraining failedquality andremainsretained,neverrerununchanged.
FullgoalACTIVE:allfamilies/automatic-assisted/useracceptance/humanFINAL/Windows
CPUminimum/freshinstall/acceptedselectiveport remain incomplete.
'''

def main():
    dest=BASE/'preparation_milestone_v1.json';need(not dest.exists(),'Preserve prior preparation')
    evidence={
      'root_release':(BASE/'root_manual_transfer_release.json','140b74bf93bc31cab12482a4b4b144172ef7320b7cb1f695d51c32a53a494be7'),
      'root_final_archive':(BASE/'root_final_archive_audit.json','12d08c9e3152de5979b209c272b0825df770af2b534227907933edf80217bae7'),
      'independent_final_archive':(ROOT/'research/completion_spatial_code_intervention_v1_transport_review/independent_final_archive_guide_admission_v004.json','3015fb51b15302005b544fced147177c891d90195790983d461ee6c30c1825a7'),
      'independent_source_CLI':(ROOT/'research/completion_codeformer_spatial_code_intervention_v1_independent_review/independent_source_packet_admission_draft004.json','e9d6a0d0ffb6fd1e4fbd1e95260c8c0437031c93672ed4c43b15baa1435867fe'),
      'independent_runtime':(ROOT/'research/completion_codeformer_spatial_code_intervention_v1_independent_review/independent_runtime_source_semantics.json','cb038d0e5746295eba36fe5313ee5a1006bc3634f8a5055523553ccb9df7b354'),
      'root_installer':(ROOT/'research/completion_codeformer_spatial_code_intervention_v1_design/root_installer_fixtures.json','610177441433cbb5f282c8264d35a0b63696e13b72cb4c3bcb627eb2c6b58e6e'),
    }
    pins={}
    for key,(q,h) in evidence.items():need(sha(q)==h,'Evidence pin '+key);pins[key]=pin(q)
    need(sha(STAGE/'protocol.json')==PROTOCOL,'Final protocol')
    archive=TRANSFER/'cctv-completion-codeformer-spatial-code-intervention-v1-execution.tar.gz'
    need(physical(archive).stat().st_size==13610328 and sha(archive)==ARCHIVE,'Published archive')
    need(sha(ROOT/GUIDE)==sha(TRANSFER/GUIDE)=='e35e2618a7fb7d1192cc2a220e536b321a3e156e4e7fcd5eaf69e112d7c65644','Both guide copies')
    p=read(STAGE/'protocol.json');need(len(p['files'])==166,'Final payload scope')
    source=ROOT/'outputs/completion_codeformer_finetune_r1_app/source_admission.json';a=read(source)
    apppins=[*a['source_files'],a['candidate_manifest'],a['checkpoint']]
    for r in apppins:need(physical(r['path']).stat().st_size==r['bytes'] and sha(r['path'])==r['sha256'],'R1 current source/model')
    prep=ROOT/'COMPLETION_CODEFORMER_SPATIAL_CODE_INTERVENTION_V1_PREPARATION.md';new(prep,PREPARATION.encode('utf-8'))
    backup=BASE/'documents_before_preparation_milestone';need(not backup.exists(),'Preserve prior docs');physical(backup).mkdir()
    changes=[]
    for name,mode,text in [('PROJECT_HANDOFF.md','prepend',HANDOFF),('FACE_MASK_COMPLETION_SPEC_V1.md','append',SPEC)]:
        path=ROOT/name;before=physical(path).read_bytes();saved=backup/name;new(saved,before);extra=text.encode('utf-8');after=extra+before if mode=='prepend' else before+extra
        need(physical(path).read_bytes()==before,'Concurrent document change')
        physical(path).write_bytes(after);offset=len(extra) if mode=='prepend' else 0;need(physical(path).read_bytes()[offset:offset+len(before)]==before,'Previous bytes exact')
        changes.append({'before':pin(saved),'after':pin(path),'mode':mode,'old_bytes_offset':offset,'old_bytes_exact':True})
    record={'format':'completion-spatial-code-intervention-v1-manual-preparation-milestone-v1','complete':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'author':pin(Path(__file__)),'preparation':pin(prep),'manual_guide':pin(ROOT/GUIDE),'transfer_guide':pin(TRANSFER/GUIDE),'transfer_archive':pin(archive),'protocol':pin(STAGE/'protocol.json'),'evidence':pins,'documents':changes,'working_R1_all22_pins_unchanged':True,'status':'PREPARED_UNRUN_ZERO_UPDATE_DIAGNOSTIC_MANUAL_TRANSFER_READY','actual_VM_preflight_or_execution':False,'actual_training':False,'new_model_created':False,'neural_calls':0,'VM_calls':0,'RESTORATION_IMPLEMENTATION_edited':False,'full_goal_status':'ACTIVE_INCOMPLETE','goal_complete':False}
    new(dest,(json.dumps(record,indent=2,allow_nan=False)+'\n').encode('utf-8'))
    print(json.dumps({'milestone':pin(dest),'preparation':pin(prep),'goal_complete':False}))

if __name__=='__main__':main()
