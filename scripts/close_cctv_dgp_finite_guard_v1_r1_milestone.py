"""Publish verified milestone status with exact-byte backups of canonical documents."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_closure'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''): value.update(block)
    return value.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    assert not OUT.exists(), 'Preserve every preceding closure'
    bundle = ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_vm'
    protocol = read(bundle/'protocol.json')
    assert sha(bundle/'protocol.json') == 'efdd62759213136114a56f0aaa256278753cac8bac4bc6c6a52f1f8b7550598f'
    for name,digest in protocol['local_sources_sha256'].items(): assert sha(ROOT/name) == digest,name
    audit_path = ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_independent_audit.json'
    visual_path = ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_return_review_v1/visual_review.json'
    qdir = ROOT/'outputs/cctv_dgp_finite_guard_quantization_v1'
    cdir = ROOT/'outputs/cctv_dgp_post_finite_guard_input_coverage_v1'
    a,v,q,c = read(audit_path),read(visual_path),read(qdir/'independent_audit.json'),read(cdir/'saved_arithmetic_audit.json')
    result = read(ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_vm_return/outputs/results.json')
    assert a['complete'] and a['accepted_training_changes_verified'] == 0 and a['CPU_replay_cases'] == 80
    assert v['complete'] and v['unique_model_outputs_reviewed'] == 400 and v['all20_pages_actually_viewed_at_original_resolution']
    assert q['complete'] and q['all_PNG_metric_rows_verified'] == 800 and q['original_floor_failure_decisions_reproduced']
    assert c['complete'] and c['input_records_verified'] == 124 and c['existing_24_native_usable_labels_preserved']
    assert result['optimizer_updates'] == result['epochs'] == 0 and result['states_before'] == result['states_after_restore']
    quant = read(qdir/'results.json')
    assert all(not d['pass'] for policies in quant['comparisons'].values() for cohorts in policies.values() for d in cohorts.values())
    for name,digest in read(qdir/'protocol.json')['source_bindings'].items(): assert sha(ROOT/name) == digest,name
    for name,digest in read(cdir/'plan.json')['source_bindings'].items(): assert sha(ROOT/name) == digest,name
    checkpoints = {
        'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth':'646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b',
        'checkpoints/dgp_zamboanga_final.pth':'b6376f56c4161ef1f26a8f9efb0bb309b3014f4ecdeaca5399a47c1c4607586c'}
    for name,digest in checkpoints.items(): assert sha(ROOT/name) == digest,name
    reports = ['CCTV_DGP_FINITE_GUARD_V1_R1_RESULTS.md','CCTV_DGP_FINITE_GUARD_QUANTIZATION_V1_RESULTS.md',
               'CCTV_DGP_POST_FINITE_GUARD_INPUT_COVERAGE_V1_RESULTS.md','CCTV_DGP_POST_FINITE_GUARD_TRAINING_REVIEW.md']
    for name in reports: assert (ROOT/name).is_file()
    common = '''**Latest verified status - 10 October 2026: finite-guard R1 returned and fully reviewed; conversion and input-coverage diagnostics closed. Full goal incomplete.**

The manual L4 study accepts zero parameter changes and zero epochs. Its frozen
return checker passes1,629 members,400 raw/PNG records,64 saved aggregate vectors,
24 comparisons and80 CPU replays. All20 exact-cell pages/400 model outputs have
primary-assistant development visual review. The proposals remain soft; highest
PNG structure gain is0.043497%, below1%. Every raw preservation comparison passes,
but delivered ArcFace regressions reject allthree proposals. Original checkpoints,
model states, split roles, caches, provenance, terms and failed gates remain.

A separately frozen local conversion check verifies400 rounded PNGs and800 metric
rows. Nearest rounding approximately halves conversion error, but qualifies no
proposal, retains recognition failures and introduces negative source gains.
Historical floor decisions are reproduced. No policy or model is adopted. New
rounded images are numerically audited, not separately visually qualified.

The input-only coverage review measures100 exposed TRAIN and all24 usable native
development crops. Sampled degraded TRAIN geometric eye spacing is5.38-21.40 grid
pixels; native annotation spacing is23.02-47.04 capture pixels. These are geometry,
not equivalent resolved detail or a full3,905-case corpus audit. Existing Auto
selects all10 asian_faces clear cases and none of10 FFHQ clear cases, so it cannot
guarantee a clear-image bypass. All24 native usable labels and source separation
remain. The older transitive-quality-source omission and preparation failure are
retained; current sources are prospectively bound. There are zero local DGP or
completion forwards, gradients, parameter updates or training epochs in these
two diagnostics; only the conversion audit uses frozen recognizer inference.

Next design: broaden resolution/framing/degradation coverage from genuine audited
HQ TRAIN references and investigate input-conditioned spatial learning in a
separate current-DGP copy. This is an evidence-based design hypothesis, not an
executable training packet. More epochs of the rejected recipe are not justified.
The current checkpoint remains Phase3 plus two selected fine-tuning epochs; its
total ancestral count is unestablished. Additional epochs1/2/5 remain a proposed
five-epoch study after finite qualification, preserving existing quality gates.

All actual training/autograd studies remain manually launched inside tmux on the
existing L4 until the user names another VM. This turn did not connect, start,
clean or train on the VM. Returned guest receipts show the human's completed run;
current power/free-space state was not checked. Earlier stopped-VM statements are
historical. Direct VM authorization remains inventory-first, hash-bound maintenance
preserving research and active work. No historical pilot is resumed automatically.

All five milestones remain required: native provenance/terms/overlap/input-only
criteria; same-input resize/Phase3/DGP/pretrained comparisons; justified finite
training; primary DGP/Auto/override integration preserving the existing design;
useful development outputs, independent final review, regressions and bundled
inline Playwright. Native CCTV remains unpaired, separate from paired synthetic
PSNR/SSIM. Prioritize Asian capture sources without ethnicity or Zamboanga claims.
All visible facial features remain in scope; request clearer/less-covered crops
when information is insufficient. Masks, sunglasses, strong lens glare, hands,
obstructing hair, scarves and objects require separate plausible completion,
automatic-area preview/correction, visible-appearance/clear-glasses/ordinary-hair
preservation, original/mask/result and PNG/bundle downloads, and separate automatic
versus assisted evidence. Exact hidden identity is not claimed. Goal active/incomplete.

Read CCTV_DGP_FINITE_GUARD_V1_R1_RESULTS.md,
CCTV_DGP_FINITE_GUARD_QUANTIZATION_V1_RESULTS.md,
CCTV_DGP_POST_FINITE_GUARD_INPUT_COVERAGE_V1_RESULTS.md and
CCTV_DGP_POST_FINITE_GUARD_TRAINING_REVIEW.md. This status supersedes historical
pending/prepared/stopped statements below without deleting their evidence.
Exact preceding bytes: outputs/cctv_dgp_finite_guard_v1_r1_closure/before/.


'''
    files = ['PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md']
    assert not set(files)&set(protocol['local_sources_sha256'])
    for name in files:
        assert name not in read(qdir/'protocol.json')['source_bindings']
        assert name not in read(cdir/'plan.json')['source_bindings']
    old = {name:(ROOT/name).read_bytes() for name in files}
    (OUT/'before').mkdir(parents=True)
    bindings = {}
    for name in files:
        backup = OUT/'before'/name
        with backup.open('xb') as stream: stream.write(old[name])
        assert (ROOT/name).read_bytes() == old[name] and backup.read_bytes() == old[name]
        prefix = common.encode('utf-8')
        with (ROOT/name).open('wb') as stream: stream.write(prefix+old[name])
        published = (ROOT/name).read_bytes()
        assert published == prefix+backup.read_bytes() and published[len(prefix):] == old[name]
        bindings[name] = {'before_sha256':sha(backup),'after_sha256':sha(ROOT/name),
            'prefix_bytes':len(prefix),'exact_preceding_suffix_preserved':True}
    for name,digest in protocol['local_sources_sha256'].items(): assert sha(ROOT/name) == digest,name
    for name,digest in checkpoints.items(): assert sha(ROOT/name) == digest,name
    evidence = [audit_path,visual_path,visual_path.parent/'gallery_independent_audit.json',
        qdir/'protocol.json',qdir/'results.json',qdir/'independent_audit.json',
        cdir/'plan.json',cdir/'results.json',cdir/'saved_arithmetic_audit.json']+[ROOT/name for name in reports]
    receipt = {'complete':True,'documents':bindings,'evidence_bindings':{p.relative_to(ROOT).as_posix():sha(p) for p in evidence},
        'current_and_Phase3_checkpoint_hashes_verified':checkpoints,'frozen_parent_local_sources_unchanged':53,
        'new_training_accepted_updates':0,'new_training_epochs':0,'original_visual_review_closed':True,
        'new_rounded_images_visually_qualified':False,'VM_connection_or_cleanup_or_start':False,
        'training_recipe_ready':False,'independent_final_review_complete':False,'goal_complete':False,
        'closed_UTC':datetime.now(timezone.utc).isoformat(),'closure_source_sha256':sha(Path(__file__))}
    with (OUT/'closure.json').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print({'complete':True,'status_documents':3,'exact_byte_backups':3,'source_and_checkpoint_hashes_verified':True,
        'training_recipe_ready':False,'goal_complete':False})


if __name__ == '__main__': main()
