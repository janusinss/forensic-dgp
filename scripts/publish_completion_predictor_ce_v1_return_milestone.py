"""Publish the actual completion return without altering retained research bytes."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/completion_codeformer_predictor_ce_v1_return'
ORIGINAL=Path('C:/xampp/htdocs/YEAR 4/Testing/outputs')
PINS={
 'transport_audit.json':'60bcc196d165e082010d67d1a62708792cf1f08dfa732921a58a1d65f13408c0',
 'independent_saved_output_audit.json':'d1c93643925154fbcdb6310614ebb507b51ddc66d54d649b3b3c03774d1496e0',
 'root_endpoint_and_preservation_closure.json':'a8f680bab3455ff2e459f01290caf8c46c948af97df5b855b3951eab561b29a3',
 'root_saved_code_geometry_diagnosis.json':'647bfa63a9f7aaa62d18cc7e494a5bba5ccf5468d224e3293d5370a469a7fca8',
 'root_visual_review.json':'58cd3c91c39754bc216d1bd924a498ec64f107188f8cc447c7cb095ffd569595',
 'visual_sheet_index.json':'c09f368601e54281e52dbe74bb26057ba5cd9c94c11339604e55bd07ddfca2e9',
 'review.html':'559a2eca986283e7fec6bfa25f5f74eeaaacec63cba3cc5fb59b712b355de398',
 'review_html_admission.json':'093135901add0c15b040121d62b661deab6e254fb75333d27313feda228f5963',
}
EXTRA={
 'research/completion_codeformer_predictor_ce_v1_state_review/independent_state_Adam_logits_audit.json':'281d966d28f49ca0001f33d78b567c7696a183a2db2946ed3e31f818a00ae88e',
 'research/completion_codeformer_predictor_ce_v1_state_review/independent_root_saved_diagnosis_QA.json':'057052290828aab39fdbdf50d9eeb97c860bf26e5507fcdfae91c3849da5bc83',
 'research/completion_codeformer_post_predictor_ce_v1_review/FIT_loss_geometric_decomposition.json':'052e200d0ac652c2b6a4f5afa88bf7f563860ec6f8668944221542d5adfc2f11',
 'research/completion_codeformer_post_predictor_ce_v1_review/independent_grid_QA_and_retained_gray.json':'ffda4805b10c72ce65417c1081a181f83a1d28c9125e9c09e694eba7190b3807',
 'research/completion_codeformer_post_predictor_ce_v1_review/independent_scientific_review.md':'c8e7f78ce4f9ab7f584d204060ea57b038134e46f2a3db1753954ef6643c2a34',
 'research/completion_codeformer_post_predictor_ce_v1_review/scientific_review_admission.json':'c649ca67f757942e214b9fb806717257b6af694aa3b11e62f926ec1942761305',
}

REPORT='''# Predictor-only CodeFormer completion V1: actual result

27 October 2026. **EXECUTED; VALID TRAINING STATE, FAILED QUALITY. R1 stays selected.**
The earlier prepared/unrun entries are historical. Do not relaunch this pilot,
append epochs or select its intermediate readouts as a replacement.

[Inspect every saved comparison](outputs/completion_codeformer_predictor_ce_v1_return/review.html).
[Independent design diagnosis](research/completion_codeformer_post_predictor_ce_v1_review/independent_scientific_review.md).

## What ran and what is preserved

The user manually completed all32 new predictor Adam updates and128 backwards
on the existing L4. The supervisor finished in240.763seconds without timeout;
the worker recorded232.360seconds. Exit2 is a quality rejection, not a Python
crash, storage failure or missing training. Export then completed in94.247seconds.

The1,615,107,092-byte archive SHA256 is
`75b28e11dce3cd15f04286763d8d61a71bcf073bdab815c77eef529968f6559e`.
All835 regular archive members, including834 manifest-listed payloads, passed
closed namespace/type/size/hash and full gzipCRC checks. All468 NPZs and256
evaluation PNGs were independently checked. The retained original/R1 weights,
teacher labels, fixed data/splits, previous failures and this failed return remain.

The complete701,933,925-byte terminal state SHA256 is
`bc7e1bd5db71ffffc55eff4cc2b4da364aa96b9316c099102cc1523ea010d5db`.
All497 model tensors are finite. All114 intended predictor tensors changed;
the other383 remain exactly R1, including all165 original prior tensors.
Named predictor Adam states advanced32to64; the164 frozen encoder and54 frozen
fusion Adam slots stayed exactly at32. Full RNG/sampler and all32 update SHA
chains verify. All114 actual gradients were finite/nonzero; clipping was used
on29of32steps. This was genuine learning with the intended optimizer.

All128 savedFIT CE/logit records replay with the once-only division by4.
All192 paired fixed-case whole-grid code comparisons verify. There were384
actual model calls:128code-only and256full generation calls;168empty bypasses.
No teacher neural inference or oracle code substitution occurred in this run.
Local verification used saved arrays and safe CPU state inspection; no new
neural inference, gradient, optimizer update or VM operation was performed.

## Delivered image result against working R1

Positive gains below mean less PNG hole-L1 error on fixed paired adaptive TRAIN
cases. They are not restoration feature-structure gains, PSNR/SSIM, identity
accuracy, native CCTV clean-reference evidence or independent FINAL results.

| Separate cohort | Final32 PNG hole-error reduction | Individual holes worse | Individual boundaries worse |
|---|---:|---:|---:|
|32 clear paired cases|0.307561%|14/32|15/32|
|16 synthetic challenges|0.446027%|5/16|6/16|

Both cohort gains miss the prospective1% requirement. Lowerface RAW/PNG hole
and boundary groups regress. Challenge boundaries also worsen at macro,
cheek and condition levels. These are independently reproduced image changes,
not an export-rounding explanation. Every44,001,540 outside-support RGB byte
is exact; all48 historical R1 baseline cases and four fresh parity pairs match.
Exact copied visible pixels do not establish plausible generated hidden anatomy.

The eight strict-gray cases also fail the independent style requirement:
mean PNG support channel spread rises from4.671184to4.816679. Raw/PNG macro
and three geometry pairs worsen; the eye-band mean is slightly better. Color
was not removed by postprocessing to conceal the failed training result.

Root inspected all20original-detail sheets:48paired cases at0/8/16/32 and
32previously-exposed public unpaired cases at0/32,256outputs/496exact256cells.
No consistent material final32 improvement was found. The puffed closed-mouth
example still becomes a smile; soft eyes, invented eyewear, false gray color,
asymmetric hidden eyes and a central orange/plastic scarf/glove defect persist.
Some public-mask/flower estimates are plausible, but the new version looks
very similar to R1. Public hidden regions have no aligned clean truth.

All input images and reviewed removal masks stayed fixed. Residual covering
outside support is a mask-footprint question; defects generated inside support
are model failures. The delivered mask was not expanded to conceal them.
All22 current R1 app/model/source pins remain exact. No candidate was installed
or adopted, and RESTORATION/IMPLEMENTATION were not edited.

## What this changes about the next improvement

The broken-optimizer/disconnected-gradient/save-failure explanations are ruled
out for this run. Hard code choices changed386of8192clear positions and251of4096
challenge positions. Whole-grid clean-label matches went5183to5182clear and
1331to1344challenge. Better code classification is not a guaranteed image gain.

The additional saved-only diagnosis separates16x16 geometric cells touched by
removal from fully visible cells. Clear touched-cell pooled agreement declines
21.4939%to21.1890%; challenge agreement improves10.8926%to11.3464%. These cells
are a geometric proxy, not actual neural receptive fields or identity accuracy.

Across128 distinct FIT observations, touched cells are15.741%of tokens and
contribute34.073%of uniform CE. Fully visible cells contribute65.927%. This
supports investigating how supervision is allocated; it does not prove visible
context is irrelevant. A generator mixes information across positions. The
different FIT observations cannot be presented as a before/after learning curve.

The previous full-grid clean-code oracle helped, but even its no-fusion output
retained gray falsecolor. Oracle-derived gains use hidden clean information
unavailable in the app; they are not a promised attainable training gain.

The next selected design distinction is a frozen-R1, zero-update comparison of
clean-code substitution only at geometrically touched cells versus only in
their visible complement, against the already audited ordinary/full-oracle
controls. This tests whether spatial supervision or wider context compatibility
deserves the next actual learning change. It is nondeployable diagnosis, with
no clean RGB pasted, no new targets or enlarged masks. A prospective108-call
outline exists; **no execution packet or launch guide is ready for that test**.
Finite source/resource/bridge checks must precede a manual launch. Do not repeat
the failed uniform-CE recipe unchanged or assume more epochs/LR fixes it.

## Evidence and remaining scope

The return folder links transport, independent state/Adam/logit, independent
all-pixel audits, root endpoint replay, complete visual review and geometric
diagnosis. Independent QA remeasured both geometric code counts and PNG edges.
The root replay field `payload_files_rehashed:835` means all regular files
including export_manifest.json; the precise manifest-listed payload count is834.
Its original bytes are retained and this label clarification is independently
recorded; no science numbers or artifact hashes changed.

The full goal remains ACTIVE. Useful results across all seven covering
families, separately tested automatic versus assisted masks, refusal behavior,
user acceptance, blinded human FINAL, minimum Windows CPU/RAM and fresh-install
evidence, and selective accepted IMPLEMENTATION integration remain incomplete.
This failed training/export, intact state and review page do not satisfy them.
R1 remains the working experimental model, not a full-scope qualified release.
'''

HANDOFF='''# 27 October 2026: predictor-only completion V1 audited; R1 retained

EXECUTED_32_UPDATES_VALID_STATE_FAILED_QUALITY_NO_PROMOTION.
[Actual results and diagnosis](COMPLETION_CODEFORMER_PREDICTOR_CE_V1_RESULTS.md).
[All20 sheets /80 cases /256 outputs](outputs/completion_codeformer_predictor_ce_v1_return/review.html).

Manual L4 run completed32newAdam/128backs in240.763s without timeout;worker2
is quality rejection,not crash.Export complete;835regular/834payloads verified.
Terminal701,933,925B SHA bc7e1bd5db71ffffc55eff4cc2b4da364aa96b9316c099102cc1523ea010d5db.
All114predictor tensorschanged,383R1frozen exact;activeAdam32to64,inactive
encoder164/fusion54unchanged32;fullRNG/sampler/128CE and192codecomparisons pass.

Independent PNGhole gain vsR1:0.307561%clear/0.446027%challenge,below1%,with
lowerface/boundary andGRAYstyle regressions.14/32clearholes and5/16challenge
holes worsen.All44,001,540outsideRGBbytes exact.All48historicalR1/fourparity
pairs pass.Root reviewedall20sheets/496exactcells:no materialconsistent gain;
expression/softness/inventedframes/falsecolor/scarfcentraldefects persist.

Additional saved-only diagnosis finds65.927%FITCE liesonfullyvisible geometric
cells;thisdoesnotprovetheircontextirrelevant.Nextselecteddesign distinguishes
masked-cell versus visible-context oraclecodes atfixedR1,zerooptimizer,using
the same48pairedadaptiveTRAINcases and auditedcontrols.Prospective108calls;
NO ready successorpacket orautomaticlaunch.Oraclesareunavailablecleancontext,
notappoutputs orpromisedgains.Moreepochs/LR orunchangeduniformCE isnotjustified.

Both manualguide copies retired HISTORICAL/DO_NOT_RELAUNCH;all priorbytes,
originalweights/teacherlabels/splits andfailedterminal evidence retained.All22
R1app/modelpins unchanged.NoRESTORATION/IMPLEMENTATION changes.FullgoalACTIVE:
all-familyusefulness,automatic-assisted evaluation,useracceptance,humanFINAL,
WindowsCPUminimum/freshinstall/selectiveacceptedport remain incomplete.
Earlier prepared/unrun entries below are historical.

---

'''

SPEC='''

## Predictor-only CE V1 actual return: retain R1; review spatial supervision

27 October 2026. The manual32-update/128-backward pilot executed successfully
but failed quality. All114 intended tensors changed;383 frozen tensors and
inactive Adam slots remained exact. Complete terminal/Adam/RNG/sampler,128saved
CE observations and all256image outputs independently verify. PNGhole gain is
0.307561%clear/0.446027%challenge with lowerface/boundary/GRAY regressions.
All44,001,540outsideRGBbytes exact;all20sheets/80cases/496cells root-reviewed.
No consistent material visible improvement;R1 and its22app/modelpins retained.

See COMPLETION_CODEFORMER_PREDICTOR_CE_V1_RESULTS.md and linkedallcase review.
Both old manualguides nowhistorical/donotrelaunch;exactpreviousbytes retained.
Nextdesign isolates geometric masked-code versus visible-context oracle effects
atfrozenR1 beforeanynewlearning.108forwards isprospectiveonly,no readyexecution
packet/guide,zeroupdates andnoautomaticlaunch.Wholegridoraclebenefit doesnot
provehole-onlyCEbest;65.927%FITlossvisibleallocation isdiagnosis,notcausalproof.
Allquality/preservation gates remain unchanged.Noepochappend orfailedrecipe
repeat,maskexpansion,checkpointrescue orcandidatepromotion.FullgoalACTIVE:
allfamilies/automatic-assisted/useracceptance/humanFINAL/CPUfreshinstall and
selectiveacceptedIMPLEMENTATION integration remain incomplete.
'''

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
def new(path,body):
    with physical(path).open('xb') as f:f.write(body)

def main():
    need(not (BASE/'return_milestone_v1.json').exists(),'Preserve prior milestone')
    for n,d in PINS.items():need(sha(BASE/n)==d,'Evidence drift '+n)
    for n,d in EXTRA.items():need(sha(ROOT/n)==d,'Evidence drift '+n)
    backup=BASE/'documents_before_actual_return_milestone';need(not backup.exists(),'Preserve prior backups');physical(backup).mkdir()
    changes=[]
    def change(path,mode,text,name):
        before=physical(path).read_bytes();new(backup/name,before);extra=text.encode('utf-8');after=extra+before if mode=='prepend' else before+extra
        need(physical(path).read_bytes()==before,'Concurrent document change; refuse overwrite')
        physical(path).write_bytes(after);offset=len(extra) if mode=='prepend' else 0
        need(physical(path).read_bytes()[offset:offset+len(before)]==before,'Prior bytes retained exactly')
        changes.append({'before':pin(backup/name),'after':pin(path),'old_bytes_offset':offset,'mode':mode,'previous_bytes_exact':True})
    report=ROOT/'COMPLETION_CODEFORMER_PREDICTOR_CE_V1_RESULTS.md';new(report,REPORT.encode('utf-8'))
    oldguide=ROOT/'COMPLETION_CODEFORMER_PREDICTOR_CE_V1_VM.md';oldpublished=ORIGINAL/oldguide.name
    need(sha(oldguide)==sha(oldpublished)=='0c35dd81e95dd96b72b127e7d58a91e40fec76ed04930113d9ca91f92b1e1749','Original launch-guide copies drift')
    retired='''# HISTORICAL COMPLETED PILOT — DO NOT RELAUNCH

27 October 2026. This exact manual pilot ran and failed final quality at32updates.
Read COMPLETION_CODEFORMER_PREDICTOR_CE_V1_RESULTS.md in FACE_MASK_COMPLETION.
R1 remains selected. Preserve the actual state, archive and original commands;
do not rerun, resume or append epochs. No successor is launched by this guide.
The original guide bytes follow unchanged as historical evidence.

---

'''
    change(oldguide,'prepend',retired,'guide_before_retirement_canonical.md')
    change(oldpublished,'prepend',retired,'guide_before_retirement_original_outputs.md')
    change(ROOT/'COMPLETION_CODEFORMER_PREDICTOR_CE_V1_PREPARATION.md','prepend','''# Current status: pilot executed; quality rejected

27 October 2026. The prospective preparation below is historical. All32 new
updates ran; state/output audits pass but quality fails. R1 stays selected.
Read COMPLETION_CODEFORMER_PREDICTOR_CE_V1_RESULTS.md for actual measurements.
Do not relaunch or append epochs. Original preparation bytes remain below.

---

''','preparation_before_actual_return.md')
    change(ROOT/'PROJECT_HANDOFF.md','prepend',HANDOFF,'PROJECT_HANDOFF.md')
    change(ROOT/'FACE_MASK_COMPLETION_SPEC_V1.md','append',SPEC,'FACE_MASK_COMPLETION_SPEC_V1.md')
    record={'format':'completion-codeformer-predictor-ce-v1-actual-return-milestone-v1','complete':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'author':pin(Path(__file__)),'status':'EXECUTED_32_UPDATES_VALID_STATE_FAILED_QUALITY_R1_RETAINED','results_report':pin(report),
            'return_evidence':{n:pin(BASE/n) for n in PINS},'independent_evidence':{n:pin(ROOT/n) for n in EXTRA},'documents':changes,
            'next_design_selected':'Frozen R1 geometric masked-code versus context-code oracle diagnostic with exact retained controls; prospective108calls only','next_packet_ready':False,'next_actual_training_launched':False,
            'R1_working_app_unchanged':True,'RESTORATION_IMPLEMENTATION_edited':False,'full_goal_status':'ACTIVE_INCOMPLETE','goal_complete':False,'local_neural_calls':0,'VM_operations':0,
            'precise_regular_return_count':835,'precise_manifest_payload_count':834,'root_replay_payload_files_rehashed_label_clarification':'835includes export_manifest.json; original receipt and science evidence unchanged'}
    new(BASE/'return_milestone_v1.json',(json.dumps(record,indent=2,allow_nan=False)+'\n').encode('utf-8'))
    print(json.dumps({'milestone':pin(BASE/'return_milestone_v1.json'),'report':pin(report),'documents_updated':len(changes),'next_packet_ready':False,'goal_complete':False}))

if __name__=='__main__':main()
