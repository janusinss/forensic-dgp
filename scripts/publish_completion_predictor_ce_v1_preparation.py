"""Publish verified manual-pilot readiness while preserving previous handoffs."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/completion_codeformer_predictor_ce_v1_packet/draft_001'
TRANSFER=Path(r'C:\xampp\htdocs\YEAR 4\Testing\outputs')
STEM='cctv-completion-codeformer-predictor-ce-v1'
def need(ok,message):
 if not ok:raise ValueError(message)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def pin(p):return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p)}
def write_new(path,text):
 with path.open('xb') as f:f.write(text.encode('utf-8'))
def main():
 evidence={
 'root_release':(BASE/'root_manual_transfer_release.json','6186b33cc575262cc48555ffeb78adce87e33e64091b4f962e19c18576cbf499'),
 'root_archive_audit':(BASE/'root_final_archive_audit.json','59372c2b2fc78dd72db49b77cfa30aa043bbfd05cadc0166c9abce83611d511a'),
 'independent_archive_guide':(ROOT/'research/completion_codeformer_predictor_ce_source_audit_v1/independent_final_archive_guide_admission.json','afbde65480748c4a42edbe264d7a2be4b8233b89b05565a9ba007e59c6b03a8a'),
 'independent_source_packet':(ROOT/'research/completion_codeformer_predictor_ce_source_audit_v1/independent_source_packet_admission.json','5dd1568cda855474d31f4c84175b5cf89c4b6162a1911cbc1c746a77339dc6d3'),
 'actual_guarded_transfer_CLI':(ROOT/'research/completion_codeformer_predictor_ce_source_audit_v1/independent_pure_transfer_CLI_fixture_v1.json','1cabcacfebb8209971e34dee8b15c72730d875b94b23851fb3d06dafe1a6a8e6'),
 'author_guide_final':(BASE/'final_guide_publication_author_check.json','dc3fdb4671834eed7b3c686608b28d6439857de92f6ff4bb939158df832b7af7'),
 }
 for key,(path,digest) in evidence.items():need(sha(path)==digest,'Evidence drift '+key)
 archive=TRANSFER/(STEM+'-execution.tar.gz');checksum=TRANSFER/(archive.name+'.sha256')
 need(archive.stat().st_size==73748641 and sha(archive)=='1052d09627cdd69f5e652a6996cee9909e5498ba516cb2b5900981b066a458c3','Transfer archive drift')
 guide=ROOT/'COMPLETION_CODEFORMER_PREDICTOR_CE_V1_VM.md'
 need(sha(guide)==sha(TRANSFER/guide.name)=='0c35dd81e95dd96b72b127e7d58a91e40fec76ed04930113d9ca91f92b1e1749','Guide drift')
 report='''# Predictor-only completion V1: ready for a manual L4 pilot

27 October 2026. **PREPARED, UNRUN; NO NEW TRAINED MODEL.**
R1 remains the user-selected working completion model. The complete thesis goal
is active. [Use the verified five-step guide](COMPLETION_CODEFORMER_PREDICTOR_CE_V1_VM.md).

The previous [same-codebook diagnostic](COMPLETION_CODEFORMER_SAME_CODEBOOK_PROBE_V1_RESULTS.md)
showed that clean-reference codes can help at fixed R1 fusion, while removing
fusion with actual predicted codes worsens results. Those oracle improvements
use unavailable clean information. This pilot tests whether ordinary covered
inputs can learn better predictions; it does not supply oracle codes at inference.

## Exact prepared scope

Start from the retained complete R1 weights, mapped Adam moments at step32 and
parent RNG. Learn114 existing predictor tensors,19,450,880 scalars. Keep383
encoder/fusion/prior tensors frozen and preserve the original512-code prior.
Use CE only with LR1e-5, four one-case microbatches per step,32 new updates and
128 backwards maximum. No optimizer reset, encoder/fusion updates, latent MSE,
pixel-loss surrogate, new teacher generation or hidden clean input at evaluation.

The balanced128 observations are a fixed subset of the retained256 FIT pool,
from32 photograph reference IDs, using64 exact RGB/GRAY FIT label sets. Keep
all16 label sets from the eight adaptive monitor reference IDs out of training.
Photograph-ID separation does not prove distinct people or no pretraining overlap.
All original masks, targets, source attribution, terms, splits and failures remain.

All48 paired cases are read out at0/8/16/32, with32 unpaired public photographs
at0/32 (or the stopped terminal cycle if a catastrophe replaces32). Save256
evaluation outputs,128 FIT logits/labels and four fresh parity bundles, plus80
exact teacher-array copies. Complete execution plans384 network invocations:
128 code-only and256 full generation calls, with168 exact empty bypasses.

Four fresh parity pairs and all48 initial R1 RAW/PNG historical matches precede
updates. Intermediate8/16 readouts are safety-only. Final32 must pass separate
clear/challenge>=1% PNG hole-error reduction, all existing RAW/PNG hole/boundary
subgroup checks, exact outside pixels and a new eight-case gray-style nonworse
guard. Lower CE, code agreement and a valid checkpoint do not qualify a model.
No earlier-checkpoint rescue, automatic epochs, repeat or app promotion follows.

## Limits and recovery

Require8GiB free after installation; the installer additionally reserves its
exact payload before extraction. Retain a2GiB floor. Worker hard cap1800s,
cooperative deadline1680s and120s save reserve; CUDA allocation20GiB maximum.
Whole output2.5GiB, terminal/failed partial1GiB, raw streamed tar/gzip2.75GiB
each, export900s. These are enforced caps, not measured timing guarantees.

One bounded terminal attempt retains actual332 non-prior tensors, merged278
historical code-Adam slots with114 updated,54 frozen fusion slots, RNG, sampler,
progress and partial gradients. Unexpected mutated prior165 tensors are retained
as failure evidence within the reservation. Internal ZIP/pickle/CRC checks run
on the closed bounded partial before commit. Actual scientific failures are
recorded rather than blocking the resource-bounded failure save. Hardkill or IO
failure can still prevent recovery; preserve partials and never auto-resume.

## Verified files and evidence

Original Windows transfer folder: C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs.
Archive: cctv-completion-codeformer-predictor-ce-v1-execution.tar.gz,
73,748,641 bytes. SHA256:
`1052d09627cdd69f5e652a6996cee9909e5498ba516cb2b5900981b066a458c3`.
Protocol SHA256:
`15aa3e1478a67c120b2fcb6af50ecae58d0558eb4e65cef473d0339572bf084b`.
The LF checksum and matching guide copies are published there. The archive has
671 regular members, including protocol; payload79,488,873 bytes. Original and
R1 weight artifacts are reused by exact hash on the existing VM, not uploaded again.

Verification includes23 author source tests;4,014 plain CPU tensor fixtures;
2,036 independent source-semantic checks;2,839 independent source/data checks;
the actual unmodified import-blocked transfer CLI (exit0);root seven terminal
and nine installer toy-file fixtures; and3,398 independent final archive/guide
checks. Root separately streamed every archive payload and gzip CRC and checked
all five single-remote SCP commands, tested installer literal, exact tmux launch,
published copies/LF checksum and22 unchanged R1 app/model pins.

These are preparation checks. No neural model, actual checkpoint learning,
optimizer update, actual VM preflight or VM connection was run in preparation.
Plain-logit mathematical/autograd fixtures are distinct from model training.
An unused frozen legacy face_workflow.py snapshot has a newer-Python hash helper;
the actual Python3.10 VM execution path does not import it. All selected runtime
and transport sources were parsed and checked for the VM interface.

Read [the frozen design](COMPLETION_CODEFORMER_PREDICTOR_CE_V1_DESIGN.md), its
root pre-release clarifications, and source/transfer admissions under
research/completion_codeformer_predictor_ce_source_audit_v1. The design's earlier
runtime-unready wording is historical; this preparation record is current.

The next action is the user's manual Windows upload, VM install and tmux run.
Return all three download files for independent weights/Adam/RNG/sampler/logit/
RAW/PNG and all-case visual audit. No new app candidate exists yet. Useful real
covering-family results, automatic-versus-assisted evaluation, user acceptance,
blinded independent FINAL, minimum Windows CPU/fresh-install checks and accepted
selective IMPLEMENTATION integration remain incomplete. RESTORATION and the
current R1 app/checkpoint remain unchanged.
'''
 prep=ROOT/'COMPLETION_CODEFORMER_PREDICTOR_CE_V1_PREPARATION.md';write_new(prep,report)
 prefix='''# 27 October 2026: predictor-only completion pilot verified for manual transfer

PREPARED_UNRUN_MANUAL_L4_PILOT_NO_NEW_MODEL. R1 remains selected.
[Five-step upload/install/tmux/download guide](COMPLETION_CODEFORMER_PREDICTOR_CE_V1_VM.md).
[Preparation, evidence and limits](COMPLETION_CODEFORMER_PREDICTOR_CE_V1_PREPARATION.md).
Verified files are in original Testing/outputs. No actual VM training launched.

Distinct next test:114 predictor tensors learn CE while383 R1 encoder/fusion/
original-prior tensors stay fixed. ExactR1 Adam moments/RNG, balanced128 retained
FIT observations across32 photograph reference IDs/64FIT labels,32 new updates/
128backs maximum. All16 monitor labels stay out of optimization. This is not
an epoch over all256 cases or proof of human/pretraining identity separation.

All48 paired cases at0/8/16/32 and32 public unpaired cases at0/final;256 outputs,
128FIT logits and4parity arrays plus80 exactteacher copies.384 planned calls,
168empty bypasses, no teacherNN/oracle evaluation. Final32-only old hole/boundary
gates plus newgray-style guard; intermediate8/16 onlysafety. Oracle16%gain from
the closed probe is not a new trained gain or promised result of this pilot.

Require8GiB free plus installation payload,2GiB floor.30min hard/28min work,
120s save reserve,20GiBCUDA,1GiBterminal/2.5GiBoutput/2.75GiBtar+gzip,15minexport.
Preserve all actual failed-state/optimizer/RNG/sampler/partial evidence; no auto
retry/resume, epochappend, historical recipe rerun, checkpoint rescue or promotion.

Independent2036semantic/2839source-data/3398archive-guide checks, actual guarded
transferCLIexit0 and root full671member archive/hash/CRC/guide checks pass.
Archive73,748,641B SHA1052d09627cdd69f5e652a6996cee9909e5498ba516cb2b5900981b066a458c3.
Protocol15aa3e1478a67c120b2fcb6af50ecae58d0558eb4e65cef473d0339572bf084b.
R1app/model22pins exact. NoRESTORATION/IMPLEMENTATION changes. Manual user launch
and full independent returned evidence/all-case review precede any candidate.
Full goal ACTIVE:all-familyusefulness,automatic-assisted evaluation,useracceptance,
humanFINAL,WindowsCPUinstallation and acceptedselectiveport remain incomplete.
Earlier runtime-not-ready and probe-unrun entries below are historical.

---

'''
 appendix='''

## Predictor-only CE V1: manual-transfer ready, unrun

27 October 2026. The distinct predictor-only packet is fully source/transfer/
archive/guide verified. Follow COMPLETION_CODEFORMER_PREDICTOR_CE_V1_VM.md from
original Testing/outputs. ExactR1 weights/Adam/RNG;114predictor tensors learn,
383encoder/fusion/prior tensors frozen;balanced128existingFIT observations,
32newsteps/128backs. Photograph IDs are disjoint, not independently proven human
identities/pretraining separation.16monitor labelsets never train. Ordinaryw1
inference only;nooracle/newteacher/latentMSE orencoder/fusion learning.

Final32-only oldhole/boundary requirements plusgray-style nonworse guard.8/16
are safety checks only. Require8GiBfree+installpayload,2GiBfloor,30minhardworker,
28minwork+120ssave,finite32steps. Packet source tests and all671archive members
verify; noactualVM/modeltraining orqualifiedsuccessor yet. R1app remains exact.
Returnall3files for independentfullstate/logits/pixels/all-case review before
adoption. Noautomatictraining/resume/epochappend orhistorical relaunch.
All-familyusefulness,automatic-versus-assisted evaluation,useracceptance,
independentFINAL andWindowsCPU deployment/IMPLEMENTATION port remain incomplete.
'''
 backup=BASE/'documents_before_preparation_milestone';need(not backup.exists(),'Preserve previous milestone');backup.mkdir()
 changes=[]
 for name,mode,extra in [('PROJECT_HANDOFF.md','prepend',prefix),('FACE_MASK_COMPLETION_SPEC_V1.md','append',appendix)]:
  path=ROOT/name;before=path.read_bytes();old=backup/name
  with old.open('xb') as f:f.write(before)
  extra=extra.encode('utf-8');body=extra+before if mode=='prepend' else before+extra;path.write_bytes(body)
  offset=len(extra) if mode=='prepend' else 0;need(path.read_bytes()[offset:offset+len(before)]==before,'Prior doc bytes changed')
  changes.append({'before':pin(old),'after':pin(path),'old_bytes_offset':offset,'old_bytes_preserved_exact':True})
 milestone={'format':'completion-predictor-CE-v1-preparation-milestone-v1','complete':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'author':pin(Path(__file__)),'status':'PREPARED_UNRUN_MANUAL_TRANSFER_READY','preparation_report':pin(prep),'archive':pin(archive),'checksum':pin(checksum),'guide':pin(guide),'evidence':{key:pin(path) for key,(path,digest) in evidence.items()},'document_changes':changes,'R1_model_app_unchanged':True,'RESTORATION_IMPLEMENTATION_edited':False,'actual_VM_actions':0,'actual_training_steps':0,'new_model_created':False,'model_qualified':False,'goal_complete':False}
 out=BASE/'preparation_milestone_v1.json'
 with out.open('x',encoding='utf-8',newline='\n') as f:json.dump(milestone,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'milestone':pin(out),'preparation':pin(prep)}))
if __name__=='__main__':main()
