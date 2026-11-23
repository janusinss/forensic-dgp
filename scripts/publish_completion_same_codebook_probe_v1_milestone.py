"""Publish the completed probe milestone, preserving every prior document byte."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,os

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/completion_codeformer_same_codebook_probe_v1_return'

def physical(path):
 p=Path(path).resolve()
 return Path('\\\\?\\'+str(p)) if os.name=='nt' and not str(p).startswith('\\\\?\\') else p
def sha(path):
 h=hashlib.sha256()
 with physical(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def pin(path):return {'path':str(Path(path).resolve()),'bytes':physical(path).stat().st_size,'sha256':sha(path)}
def need(ok,message):
 if not ok:raise ValueError(message)
def read(path):return json.loads(physical(path).read_text(encoding='utf-8-sig'))

def main():
 evidence={
 'transport_audit.json':'ee4980dcd22a04f4315d7369a16c6fb2ce8ad8704532f6e1b284d35a45d24fa4',
 'independent_state_code_audit.json':'794fed73804eabfa081805cdd0d3db3dda89d5a1fbd6138df725b6b404ad922d',
 'independent_saved_output_audit.json':'4f5042edc19496bd3dee515dc052928c03e4f6d9e12b246efdcca4262a54d02f',
 'root_endpoint_and_preservation_closure.json':'7d2a94a61a50d9fbc5d4fae0ec07c95fafe807b12382c288f274cfa4d6802160',
 'root_visual_review.json':'3d2568cfd19614f7d8b98da71b027d471fc32c27753cee017ed512b394931a53',
 'visual_sheet_index.json':'13bb379941d6e333d2e4dbb754fc21aad08ca8ac066f26e23c83210743b68263',
 'review.html':'13b89f6fb9045776bec164113a5f69a5a3499b7d42f8ff9066a53b84b97fe4ac',
 'independent_HTML_artifact_closure.json':'7392b8ce411cc72247802be9ad96ed93fca1be4c6c5f339a1bd77e3f77f3f33e',
 'guide_retirement_receipt.json':'69fa72c186c054426768382f24f54b0f1727ebba93e84d049fe4849d97e04ce2',
 'results_MD_factual_scope_review_by_packet_v1.json':'0329e50cb5f19bd9d4f1768f9ff3618ab2aea8e3e75b7653d82ddea16b623dd5',
 }
 for name,digest in evidence.items():need(sha(BASE/name)==digest,'Evidence drift '+name)
 report=ROOT/'COMPLETION_CODEFORMER_SAME_CODEBOOK_PROBE_V1_RESULTS.md'
 need(sha(report)=='ef63309bd5ca7df4256a80c27d5379a06df5b2c5cec726de80c0df651799ae19','Results report drift')
 prefix='''# 26 October 2026: same-codebook probe audited; predictor-learning design follows

EXECUTED_ZERO_UPDATE_DIAGNOSTIC_NO_NEW_MODEL. R1 remains selected.
[Actual results](COMPLETION_CODEFORMER_SAME_CODEBOOK_PROBE_V1_RESULTS.md).
[All 12 sheets and independent audits](outputs/completion_codeformer_same_codebook_probe_v1_return/review.html).

Manual L4 diagnosis finished 200 forwards in 106.579 seconds without timeout,
with zero gradients, backwards or optimizer updates. All 497 R1 state tensors,
192 saved outputs, 196 NPZs, four fresh parity pairs and 48 historical R1 cases
verify. All 33,551,076 outside-mask RGB bytes and 22 R1 app/model pins are exact.
Root reviewed all 12 sheets / 48 cases / 192 outputs at original cell detail.

Clean-context SAME-codebook ORACLE codes with R1 fusion reduced paired PNG hole
error 16.488% clear / 16.364% challenge, with all declared aggregate subgroups
nonworse. These use unavailable clean information, NOT deployable app outputs,
training gains, restoration structure scores, hidden identity or native CCTV.
Three individual holes and ten boundaries worsen; softness and gray false color
remain. Turning fusion off with actual predictions worsens all 48 cases, so
retain R1 fusion at w=1. The oracle without fusion fails boundary requirements.

Next: prepare an isolated predictor-only CE test from exact R1, freezing encoder,
fusion and original prior (383 tensors) and learning only 114 predictor tensors.
A deterministic balanced 128-observation subset of the retained 256-case FIT
pool uses all 32 FIT identities and 64 FIT label sets; monitor identities/labels
never optimize. Review maximum 32 steps / 128 backwards; final step32 quality
endpoint, earlier checks safety-only. Runtime preparation is IN PROGRESS, not a
ready upload packet or actual training. All training remains manual on the L4.
Oracle improvement is not a promised achievable gain; all quality gates remain.

Both old probe launch guides are HISTORICAL / DO NOT RELAUNCH with their original
bytes preserved. All original checkpoints, failures, splits, sources and returns
remain. No app, RESTORATION or IMPLEMENTATION changes. Full goal ACTIVE: real
covering-family usefulness, automatic-versus-assisted evaluation, user acceptance,
blinded human FINAL, Windows CPU/fresh installation and accepted selective port
remain incomplete. Earlier prepared/unrun entries below are historical.

---

'''
 appendix='''

## Same-codebook probe actual return and predictor-only next design

26 October 2026. The zero-update manual probe is now executed and independently
closed: 200 forwards, all497 R1 state tensors exact, all192 outputs replayed,
all12 sheets reviewed. R1 remains selected. Clean-context SAME-codebook oracle
codes at fixed R1 fusion reduce paired hole error16.488%/16.364%, but use hidden
clean information unavailable in the app. Three individual holes worsen; gray
false color and softness persist. Removing fusion with actual codes is worse.
This is a representation diagnosis, not a new trained model or identity result.

Review a distinct predictor-only CE test:114 predictor tensors learn while the
other383 encoder/fusion/prior tensors remain fixed. Balanced128 existing FIT
observations across32 identities/64labels; keep all16 monitor-label sets out of
optimization. Final32-step endpoint,128backs maximum; earlier checks safety-only.
Runtime source preparation is underway. No execution packet or automatic launch
is ready; manual L4 transfer/source/state/output review remains required. All
original evidence and current R1 app remain unchanged. Automatic proposals,
all-family user acceptance, independent final review and CPU deployment remain
unfinished. See COMPLETION_CODEFORMER_SAME_CODEBOOK_PROBE_V1_RESULTS.md.
'''
 readiness='''# Current status: probe executed and independently reviewed

26 October 2026. The prospective diagnostic below has now run. Read
COMPLETION_CODEFORMER_SAME_CODEBOOK_PROBE_V1_RESULTS.md for actual results and
limits. R1 remains selected; oracle gains are not deployable model gains.
The probe manual guide is historical and must not be relaunched. A distinct
predictor-only learning design is being prepared; no successor packet is ready.
Exact prior design/document bytes are preserved below and in the return folder.

---

'''
 specs=[('PROJECT_HANDOFF.md','prepend',prefix),('FACE_MASK_COMPLETION_SPEC_V1.md','append',appendix),('COMPLETION_CODEFORMER_POST_V4_REPRESENTATION_DESIGN.md','prepend',readiness)]
 backup=BASE/'documents_before_probe_actual_closure';need(not backup.exists(),'Preserve earlier milestone');physical(backup).mkdir()
 changes=[]
 for name,mode,addition in specs:
  path=ROOT/name;before=physical(path).read_bytes();saved=backup/name
  with physical(saved).open('xb') as f:f.write(before)
  extra=addition.encode('utf-8');body=extra+before if mode=='prepend' else before+extra
  physical(path).write_bytes(body)
  actual=physical(path).read_bytes();offset=len(extra) if mode=='prepend' else 0
  need(actual[offset:offset+len(before)]==before,'Historical document bytes changed')
  changes.append({'before':pin(saved),'after':pin(path),'mode':mode,'old_bytes_offset':offset,'old_bytes_preserved_exact':True})
 record={'format':'same-codebook-probe-v1-actual-return-milestone-v1','complete':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'author':pin(Path(__file__)),'results_report':pin(report),'evidence':{name:pin(BASE/name) for name in evidence},'documents':changes,'status':'ACTUAL_DIAGNOSTIC_CLOSED_R1_RETAINED_PREDICTOR_RUNTIME_PREPARATION','actual_probe_training_updates':0,'new_trained_model_created':False,'R1_app_unchanged':True,'RESTORATION_IMPLEMENTATION_edited':False,'next_runtime_ready':False,'goal_complete':False}
 destination=BASE/'return_milestone_v1.json'
 with physical(destination).open('x',encoding='utf-8',newline='\n') as f:json.dump(record,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(pin(destination)))

if __name__=='__main__':main()
