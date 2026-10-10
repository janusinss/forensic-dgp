"""Close verified reviews and next manual preparation; preserve all historical bytes."""
from pathlib import Path
import time
from cctv_dgp_finite_guard_v1_r1_contract import read,write,sha,verified_assets

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/cctv_dgp_group_conflicts_v1_closure'
NEXT = ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_vm'
PREP = ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_preparation'
PIN = 'efdd62759213136114a56f0aaa256278753cac8bac4bc6c6a52f1f8b7550598f'
CHECKPOINTS = {
    'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth':'646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b',
    'checkpoints/dgp_zamboanga_final.pth':'b6376f56c4161ef1f26a8f9efb0bb309b3014f4ecdeaca5399a47c1c4607586c'}

HANDOFF = '''**DGP milestone - 10 October 2026: group-conflicts return fully audited/reviewed; finite-guard R1 verified for manual L4 execution. Full goal incomplete.**

The group-conflicts returned archive (608,243,422 bytes; SHA256
701cfd773d48f69cdbf5a933fd453db11bcd9ac3839de0fc88f7b50592e30dfb) passes
its frozen prospective checker: 1,624 regular members, 400 raw/PNG records,
32 saved aggregate gradients, three reset placements and 12 unchanged scientific
decisions. Eighty fresh CPU inference cases agree within declared tolerances
(raw maximum 2.272427e-6; PNG one byte; embedding 2.831221e-7). All 20 exact-cell
pages and 400 unique outputs were actually visually reviewed; the independent
pixel checker verifies 600 cells and 20 observations. This is primary-assistant
development visual review, not independent final review. Earlier audit flags
stating visual_review_pending are closed by the later bound visual review.

All 12 quality decisions fail. Smallest delivered gains are 0.034730% and
0.018856% on two exposed photographic TRAIN cohorts; the cross-cohort has four
delivered preservation failures. Larger steps worsen some finite losses despite
negative initial derivative predictions for every sampled group. The largest
loses structure in both cohorts and introduces tone/canvas-edge defects. These
findings justify finite-step checks and TRAIN coverage, not unchanged recipes
or more epochs. See CCTV_DGP_GROUP_CONFLICTS_V1_RESULTS.md. The 1% early quality
requirement and V42's full-corpus 0.00692364% failure remain binding. No model is
promoted and no trained epoch or parameter update occurred in that diagnostic.

The AGENTS circuit-breaker question was asked after three unsuccessful designs;
the user selected the best approach. The new finite-guard R1 study recomputes
64 separate cohort/source/profile objectives at each state, including both
50-case TRAIN cohorts (100 total). The formerly TRAIN cross-check cohort now
also guides fitting; its history is retained and no independent-generalization
claim is allowed. At most three actual accepted training changes, nine tested
placements and 960 autograd queries are permitted. Original and preceding-state
raw AND delivered preservation/structure/source/brightness checks determine
mechanics acceptance. The original 1% requirement is measured separately for
every state; microchange acceptance is not model qualification or an epoch.
Rejected outputs/decisions are retained. No resume, automatic continuation or
failed-pilot launch. A later 50-update/full-corpus experiment requires separate
justification and the existing preservation gates.

R1 is prepared and verified, not run. It contains 171 regular archive members,
53 local source bindings and 20 Python3.10-parsed sources. Four CPU parity cases
have maximum raw error 2.205372e-6; original/copy/buffer states are exact. Four
worker modes and three candidate local derivative/trial guards reject execution;
pure arithmetic and preservation-rejection fixtures pass. The prospective
return checker is frozen before launch. Individual autograd queries/aggregation
are not replayed locally; saved group-vector/certificate arithmetic, decisions,
trained-state contents and fresh CPU outputs will be independently checked.

Protocol: efdd62759213136114a56f0aaa256278753cac8bac4bc6c6a52f1f8b7550598f.
Execution: 190,367,690 bytes; SHA256
4a620dd41d96cedfef25b0ed7bbd82ac362f0e867c3ab4ca656642bb1c74870c.
Guide: CCTV_DGP_FINITE_GUARD_V1_R1_VM.md.
Design: CCTV_DGP_FINITE_GUARD_V1_R1_DESIGN.md.
Audit: outputs/cctv_dgp_finite_guard_v1_r1_preparation/independent_packet_audit.json.
The earlier V1 packet is superseded: a redundant derived-PNG storage projection
exceeded the 3GiB cap. Its packet, sources, audit and storage failure are retained.
R1 reconstructs only the preceding-anchor mean-only diagnostic PNG from exact
saved raw pixels; all primary raw/PNG outputs and both scientific anchor metrics
remain. R1's guide-publication filename collision is also retained; a distinct
guide was published without changing the frozen packet or prior guide.

Estimated R1 runtime 10-20 minutes plus 1-6 minutes export, extrapolated from
prior L4 timings; new 64-group timing is unmeasured. Require 7GiB free after
installation. Enforced worker1800s/external1830s, per-state derivatives240s,
solver120s/proposals180s, cache120s, export360s/external390s, 30s kill grace,
20GiB allocated VRAM, 3GiB retained return and 512MiB reserve. Dynamic timing
and conservative storage stops remain mandatory. No cleanup is bundled.

The user selected the existing NVIDIA L4 g2-standard-4 VM. A read-only Google
Cloud API check verifies instance identity in us-central1-a and reports
TERMINATED. No guest disk/workload inventory is available, no VM was started,
no files were deleted and no training was launched. Maintenance remains
authorized only through a live inventory and exact retained-backup/hash-bound
plan preserving research assets, active work, original checkpoints, splits and
gate failures. Start the existing VM manually before transfer; if space is
insufficient, obtain a live inventory before any deletion or launch. All actual
learning remains the user's manual tmux/verified-transfer workflow.

Separate covering-detector score review is also closed. The 36 review-mask calls
on 18 photographs and their fixed synthetic degradations run only the retained
segmenter; no DGP/completion-generator forwards or training occur. All 36 current
proposals are exact, 174 source bindings and 153 artifacts are audited, and all
nine pages/36 cases are actually reviewed. Four input-only exclusions stay held.
Post-hoc score witnesses show an inversion between one fixed covering-core pixel
and one protected-appearance pixel in each of 28 eligible covering fixtures.
A single pointwise threshold cannot fix those witnesses. Approximate assisted
footprints are not expert ground truth; the '_native' fixture suffix means
original photograph, not native CCTV. Automatic proposals remain unqualified;
no detector, threshold, margin or completion generator is promoted. See
CCTV_DGP_AUTOMATIC_PROPOSAL_SCORE_V1_RESULTS.md. The recorded numerical bounds
and later review audit close creation-time pending flags, not quality failures.

Retained DGP/Phase3 checkpoints and app design remain exact. Native QMUL evidence
remains unpaired: no convincing useful current-DGP gain on its six core cases;
seven insufficient cases remain insufficient and 32 reserved final pixels are
unviewed. Photographic/synthetic TRAIN findings are not native CCTV or Zamboanga
performance. Source labels do not establish ethnicity. All five milestones,
paired DEV preservation, useful native DEV outputs, independent final review,
seven automatic/assisted covering families and bundled inline Playwright app
flow remain required. Preserve visible appearance, clear glasses and ordinary
hair; request clearer/less-covered crops when insufficient; return original,
mask and one plausible estimate with PNG/bundle downloads and no hidden-identity
claim. Thesis deadline around/after 20 November 2026 and the user's manual VM/
credit-migration decisions remain unchanged. Full goal active/incomplete.
This prefix supersedes historical pending-return statements below; exact prior
bytes and status hashes are retained in outputs/cctv_dgp_group_conflicts_v1_closure/.


'''

SYSTEM = '''**Latest status - 10 October 2026: group-conflicts and covering-score reviews closed; finite-guard R1 prepared/verified, not run.**

All 12 group-conflicts quality decisions fail despite a certified initial
direction. All 400 unique outputs were reviewed: small changes remain soft,
larger changes regress finite preservation. Unchanged direction/epoch extensions
are not justified. The user selected the best approach after the diagnostic
question. A separate finite R1 mechanics study recomputes 64 objectives from
both sampled TRAIN cohorts and admits at most three accepted training changes,
nine proposals and 960 gradient queries. These are actual training changes,
not zero-update probes or completed epochs. Former cross-check TRAIN now fits;
no DEV/final role is changed. Original1% and all preservation thresholds remain.
Microchange acceptance does not qualify a model or resume any stopped recipe.
Manual commands: CCTV_DGP_FINITE_GUARD_V1_R1_VM.md. No automatic continuation.

The user chose the existing L4 VM. The last API check reports it stopped;
guest free space is unknown. No startup, deletion or training occurred. Require
7GiB free after installation and live inventory/hash-bound preservation before
maintenance. All learning stays manual in tmux; inference/audits can be local.
Original checkpoints, splits, gate failures and research assets are preserved.

The covering-score diagnostic is fully audited and all36 exposed development
inputs reviewed. Automatic proposals remain unqualified; 28 post-hoc fixed
score witnesses cannot be corrected by a single pointwise threshold. No
completion generator runs or app weights change. See the two latest results
reports and PROJECT_HANDOFF.md for numerical, visual and audit limits.

The full own-trained DGP-led 256x256 goal remains active/incomplete: all five
milestones, useful native DEV outputs, independent final identities/reviewer,
paired synthetic evidence kept separate, all seven automatic/assisted covering
families and bundled inline Playwright remain required. The goal and later user
decisions retain precedence over historical wording. Exact preceding bytes:
outputs/cctv_dgp_group_conflicts_v1_closure/before/.


'''

PRACTICAL = '''**Latest status - 10 October 2026: useful restoration and covering-family scope remains incomplete.**

Group-conflicts return is audited and all400 unique outputs reviewed. Every
quality decision fails; no output is adopted. Automatic covering proposals are
also audited/reviewed on all36 exposed fixtures and remain unqualified. These
are development findings, not independent final or native CCTV performance.
See CCTV_DGP_GROUP_CONFLICTS_V1_RESULTS.md and
CCTV_DGP_AUTOMATIC_PROPOSAL_SCORE_V1_RESULTS.md.

The user chose the existing L4 VM and the best evidence-based approach. The new
finite-guard R1 packet is verified for manual execution, not run: at most three
actual accepted parameter changes, with recomputed cohort/source/profile losses
and finite raw/PNG preservation checks. It establishes no completed epoch or
qualified model. The old V1 preparation is superseded and retained. Use only
CCTV_DGP_FINITE_GUARD_V1_R1_VM.md. The VM was stopped at inspection; disk space
is unknown. Require7GiB free after installation and live hash-bound inventory
before cleanup. No VM startup, deletion, training or app promotion occurred.

The existing design, current DGP and original checkpoints stay exact. All
visible facial features, insufficiency requests, original/mask/plausible-result,
PNG/bundle downloads, automatic proposal preview/correction, clear-glasses and
ordinary-hair preservation, seven covering families, separate automatic versus
assisted results and no exact hidden-identity claim remain required. All five
milestones, useful native development outputs, independent final review,
meaningful regressions and bundled inline Playwright app flow remain required.
Full goal active/incomplete. Exact preceding bytes:
outputs/cctv_dgp_group_conflicts_v1_closure/before/.


'''


def bindings():
    p = verified_assets(NEXT,PIN)
    for name,digest in p['local_sources_sha256'].items(): assert sha(ROOT/name) == digest,name
    old = read(ROOT/'outputs/cctv_dgp_group_conflicts_v1_vm/protocol.json')
    for name,digest in old['local_sources_sha256'].items(): assert sha(ROOT/name) == digest,name
    for name,digest in CHECKPOINTS.items(): assert sha(ROOT/name) == digest,name
    return p


def main():
    started = time.monotonic(); assert not OUT.exists(); p = bindings()
    audit_path = ROOT/'outputs/cctv_dgp_group_conflicts_v1_independent_audit.json'
    audit = read(audit_path); assert audit['complete'] and audit['members_verified'] == 1624
    review_path = ROOT/'outputs/cctv_dgp_group_conflicts_v1_return_review_v1/visual_review.json'
    review = read(review_path)
    assert review['complete'] and review['all20_pages_actually_viewed_at_original_resolution'] and review['unique_model_outputs_reviewed'] == 400
    assert not review['independent_final_reviewer']
    assert read(ROOT/'outputs/cctv_dgp_group_conflicts_v1_return_review_v1/gallery_independent_audit.json')['complete']
    result = read(ROOT/'outputs/cctv_dgp_group_conflicts_v1_vm_return/outputs/results.json')
    gates = [gate for trial in result['trial_summaries'] for stages in trial['comparisons'].values() for gate in stages.values()]
    assert len(gates) == 12 and all(not gate['pass'] for gate in gates) and result['optimizer_updates'] == 0
    proposal = read(ROOT/'outputs/automatic_proposal_score_v1/review_independent_audit.json')
    assert proposal['complete'] and proposal['observations_bound'] == 36 and proposal['score_witnesses_verified'] == 28
    assert proposal['input_exclusions_preserved'] == 4
    prepared,packet = read(PREP/'prepared.json'),read(PREP/'independent_packet_audit.json')
    assert packet['complete'] and prepared['protocol_sha256'] == packet['protocol_sha256'] == PIN
    assert packet['archive_sha256'] == prepared['archive_sha256'] and packet['archive_members_verified'] == 171
    assert packet['source_bindings_verified'] == 53 and packet['states_before'] == packet['states_after']
    assert packet['local_gradient_queries'] == packet['local_optimizer_updates'] == packet['local_trial_assignments'] == 0
    assert not packet['VM_training_launched']
    assert read(ROOT/'outputs/cctv_dgp_finite_guard_v1_preparation/storage_preflight_correction.json')['initial_packet_superseded_do_not_run']
    assert read(PREP/'publication_failure_retained.json')['complete']
    vm = read(ROOT/'outputs/cctv_dgp_finite_guard_v1_vm_inventory/availability.json')
    assert vm['status'] == 'TERMINATED' and not vm['VM_started'] and vm['files_removed'] == 0 and not vm['training_launched']
    docs = {'PROJECT_HANDOFF.md':HANDOFF,'SYSTEM_WORKFLOW_AND_GOAL.md':SYSTEM,'PRACTICAL_OUTPUT_SCOPE.md':PRACTICAL}
    assert not set(docs).intersection(p['local_sources_sha256'])
    originals = {name:(ROOT/name).read_bytes() for name in docs}
    OUT.mkdir(); (OUT/'before').mkdir(); published = {}
    guide = ROOT/'CCTV_DGP_FINITE_GUARD_V1_R1_VM.md'
    prior_guide = guide.read_bytes(); assert sha(guide) == prepared['manual_guide_sha256']
    (OUT/'before'/guide.name).write_bytes(prior_guide)
    text = prior_guide.decode('utf-8')
    before = 'Both\n100-case TRAIN cohorts now guide fitting; neither is independent DEV/final.'
    after = 'Both sampled TRAIN cohorts (50 cases each; 100 total)\nnow guide fitting; neither is independent DEV/final.'
    assert text.count(before) == 1; guide.write_text(text.replace(before,after),encoding='utf-8',newline='\n')
    for name,prefix in docs.items():
        (OUT/'before'/name).write_bytes(originals[name]); path = ROOT/name
        assert path.read_bytes() == originals[name]
        prefix_bytes = prefix.encode('utf-8'); path.write_bytes(prefix_bytes+originals[name])
        assert path.read_bytes() == prefix_bytes+(OUT/'before'/name).read_bytes()
        published[name] = {'before_sha256':sha(OUT/'before'/name),'published_sha256':sha(path),
            'prefix_bytes':len(prefix_bytes),'original_suffix_exact':True}
    bindings()
    write(PREP/'ready.json',{'complete':True,'prepared_status':'verified; not launched',
        'protocol_sha256':PIN,'archive_sha256':prepared['archive_sha256'],
        'independent_packet_audit_sha256':sha(PREP/'independent_packet_audit.json'),
        'creation_time_pending_flag_superseded':True,'manual_guide_sha256':sha(guide),
        'before_manual_guide_sha256':sha(OUT/'before'/guide.name),
        'guide_correction':'Clarify50 cases per cohort,100 total; commands and recipe exact.',
        'return_supervisor_sha256':sha(ROOT/'scripts/supervise_cctv_dgp_finite_guard_v1_r1_return_audit.py'),
        'VM_training_launched':False,'model_qualification':False,'goal_complete':False})
    write(OUT/'closure.json',{'complete':True,'reviewed_unique_restoration_outputs':400,
        'restoration_audit_sha256':sha(audit_path),'restoration_visual_review_sha256':sha(review_path),
        'all12_quality_decisions_failed':True,'covering_proposal_cases_reviewed':36,
        'covering_review_audit_sha256':sha(ROOT/'outputs/automatic_proposal_score_v1/review_independent_audit.json'),
        'latest_reports_sha256':{name:sha(ROOT/name) for name in ['CCTV_DGP_GROUP_CONFLICTS_V1_RESULTS.md','CCTV_DGP_AUTOMATIC_PROPOSAL_SCORE_V1_RESULTS.md']},
        'next_manual_packet_ready_sha256':sha(PREP/'ready.json'),'protocol_sha256':PIN,
        'published_documents':published,'original_checkpoints_sha256':CHECKPOINTS,
        'old_source_bindings_unchanged':True,'next_source_bindings_unchanged':True,
        'VM_status_at_last_inspection':'TERMINATED','guest_free_space_known':False,'files_removed':0,
        'VM_started':False,'VM_training_launched':False,'current_app_model_changed':False,
        'independent_final_review':False,'model_qualification':False,'goal_complete':False,
        'seconds':time.monotonic()-started,'closure_script_sha256':sha(Path(__file__))})
    print({'complete':True,'status_documents':3,'historical_suffixes_exact':True,
        'next_packet_ready':True,'VM_training_launched':False,'goal_complete':False})


if __name__ == '__main__': main()
