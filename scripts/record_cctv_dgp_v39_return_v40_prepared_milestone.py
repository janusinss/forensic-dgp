"""Bind the audited V39 return, distinct unrun V40 packet, and preserved history."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time
import sys
from cctv_dgp_spatial_fit_v40_contract import sha, read, write

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/cctv_dgp_v39_return_v40_prepared_milestone'
PRIOR = ROOT/'outputs/cctv_dgp_v39_prepared_milestone_v1'
PACKET = ROOT/'outputs/cctv_dgp_spatial_fit_vm_v40'
PREP = ROOT/'outputs/cctv_dgp_spatial_fit_v40_preparation'
DRAFT = ROOT/'outputs/cctv_dgp_spatial_fit_v40_preparation_draft_v1'


def main():
    start = time.monotonic(); assert not OUT.exists()
    old = read(PRIOR/'milestone.json'); old_audit = read(PRIOR/'independent_closure_audit.json')
    assert old_audit['complete'] and old_audit['milestone_sha256'] == sha(PRIOR/'milestone.json')
    for name, value in old['new_evidence_sha256'].items(): assert sha(ROOT/name) == value, name
    a = read(ROOT/'outputs/cctv_dgp_spatial_decoder_v39_independent_audit.json')
    v39 = read(ROOT/'outputs/cctv_dgp_spatial_decoder_v39_return/outputs/results.json')
    p = read(PACKET/'protocol.json'); ready = read(PREP/'prepared.json'); checked = read(PREP/'independent_packet_audit.json')
    assert a['complete'] and a['diagnostic_complete'] and a['local_gradient_calls'] == 0
    assert v39['all57_improvement_gradients_nonzero'] and v39['optimizer_updates'] == v39['parameter_updates'] == 0
    assert checked['complete'] and checked['all3905_first_epoch_cases_unique'] and checked['V39_architecture_identical_except_scope_and_name']
    assert not (PACKET/'outputs').exists() and not (ROOT/'outputs/cctv-dgp-spatial-fit-v40-results.tar.gz').exists()
    # Keep the initial unverified draft; the only worker change is truthful stop metadata.
    previous_worker = (DRAFT/'sources/cctv_dgp_spatial_fit_v40_vm.py').read_text()
    current_worker = (ROOT/'scripts/cctv_dgp_spatial_fit_v40_vm.py').read_text()
    addition = "            progress['new_trained_checkpoint'] = progress['optimizer_updates'] > 0\n"
    assert current_worker.count(addition) == 1 and current_worker.replace(addition, '') == previous_worker
    previous_prepare = (DRAFT/'sources/prepare_cctv_dgp_spatial_fit_v40.py').read_text()
    current_prepare = (ROOT/'scripts/prepare_cctv_dgp_spatial_fit_v40.py').read_text()
    assert current_prepare.replace('other3855 raw stages', 'remaining3905 raw stages') == previous_prepare
    write(DRAFT/'preparation_review.json', {'complete': True, 'status': 'Unverified draft retained; no VM run or recipe failure',
          'old_archive_sha256': sha(DRAFT/'cctv-dgp-spatial-fit-v40-execution.tar.gz'),
          'old_protocol_sha256': sha(DRAFT/'cctv_dgp_spatial_fit_vm_v40/protocol.json'),
          'current_archive_sha256': ready['archive_sha256'], 'current_protocol_sha256': ready['protocol_sha256'],
          'worker_change': 'Record a learned stopped checkpoint after any optimizer update, even before snapshot50',
          'description_change': 'Correct count of nonpreview raw stages from3905 to3855',
          'all_recipe_and_gate_and_optimizer_and_scope_logic_unchanged': True,
          'VM_calls': 0, 'local_gradient_calls': 0, 'local_optimizer_updates': 0})
    OUT.mkdir(); (OUT/'before_docs').mkdir()
    before = (ROOT/'PROJECT_HANDOFF.md').read_bytes()
    (OUT/'before_docs/PROJECT_HANDOFF.md').write_bytes(before)
    section = '''
**DGP milestone — 8 October 2026: V39 gradient proof independently passed; full-TRAIN V40 spatial learning packet verified and unrun.**

The human returned V39's78,588,659-byte archive. Its sidecar and export receipt
match SHA25622eb58d246891e946a6b3aeee71ae3e3923f6ff13717e679c460f07b6c1d777e.
The distinct L4 diagnostic takes33.64275s, with70 saved component queries,
zero optimizer/parameter updates, zero backwards and zero epochs. All57 new
decoder tensors receive finite nonzero improvement gradients; norms range from
7.331710e-7 to0.02951934. Initial raw/PNG output is exactly the original on50
exposed TRAIN cases. Four preservation values/gradients are exactly zero.
Original DGP, both decoder copies and recognizer states remain unchanged.
Peak allocated VRAM is2,116,979,200 bytes. No trained checkpoint is created.

The unchanged prospective independent checker takes23.54897s. It verifies273
returned regular files, ten7x17,952 float64 gradient matrices, their exact sum,
scalar sums, norms, Gram and all57 partitions, all50 initial raw/PNG pairs and
normalizer readback. Ten metadata-selected frozen CPU replays cover both sources
and five profiles: maximum raw error2.145767e-6, PNG difference one byte and
target-vector error2.095476e-7 within fixed compatibility limits. Saved VM
gradients are audited without independently differentiating locally. There are
zero local gradients, optimizer updates or VM calls. Connectivity supports a
separate capacity test; no useful-restoration or independent-final pass follows.

V40 learns this SAME spatial decoder across all781 existing TRAIN references
and3,905 cases:1,950 dataset/asian_faces cases and1,955 FFHQ-thumbnail cases.
Source labels are not ethnicity. The original CNN, fixed initial decoder and
recognizer are frozen. Only17,952 parameters in57 new tensors are optimized.
The fixed V31 metadata schedule pairs one clear and four degraded views in each
batch, covering every case once in781 updates and19 additional distinct reference
batches, for800 updates maximum. Same50 normalizer inputs/targets/support bytes,
seven original losses, filter and PNG metric definitions and quality thresholds.
AdamW3e-4/weight-decay0.01/clip1 is fixed for the NEW initialized decoder;
there is no rate sweep, continuation of V38, pretrained-restorer target, display
sharpening or target/source/person/profile-conditioned inference path.

At50, the1% structure requirement and all17 delivered-PNG preservation groups,
both-source nonregression and20% brightness-only limit must pass. Failure stops
and exports evidence. Final800 requires10% structure with unchanged preservation.
The thresholds are retained; checking preservation at50 also provides an earlier
stop. Complete snapshots0/50/800 save all3,905 PNGs, mean-only PNGs and vectors,
plus all50 prospective raw previews. The other3,855 raw stages are hashed but
not retained. The prospective independent return audit covers all PNG metrics,
all17 groups, exact raw compositions for those50 previews and50 frozen CPU
replays per complete snapshot. It does not claim every raw float is retained.
All50 preview outputs require visual review before separately frozen development
work. Necessary TRAIN capacity never overrides development or final failures.

The self-contained V40 packet has5,491 regular files and444,516,831 compressed
bytes, including265,702,338 bytes of fingerprinted input/target/support assets,
original frozen weights, code and the untrained seed. Existing venv only;
no deleted historical worker or cache is required or executed. Protocol SHA256:
b21597d7c50dc39ee7cb2bd70d7d0ad9f5cd967661b6f5e78413a7e4eb121010.
Archive SHA256:
b8113316de65ee965ffe71303adae6caf5446be7f853ab95c11242e776036b0f.

The41.6913s independent packet audit verifies5,490 assets, every tar stream,
all3,905 input files/781 references/800 batches, exact50 proof data, V39 architecture
equivalence except distinct manual root/name, identical filters/losses and AST-exact
PNG metric body. Five synthetic preservation failure cases, nine unsafe-return
archives, three scope rejections, two invalid schedules and actual pre-neural
Windows rejection pass. Python3.10 parsing, read-only Bash syntax and five
single-remote-source gcloud commands pass. Synthetic tests are not learning proof.

The first unverified draft is archived. Review corrected only the learned-stop
checkpoint metadata and nonpreview raw-stage count; recipe/gates did not change.
Git Bash's sandbox signal-pipe failure is retained. A read-only Bash -n parser
outside the sandbox passes in0.542123s without executing the launcher or any VM
operation. There was no automatic approval rejection.

V40 is PREPARED AND VERIFIED, NOT RUN. Follow the five manual upload/install/
tmux/launch/download steps on the existing idle L4/g2-standard-4. Require6GiB
free AFTER installation. Prospective estimate15–40 minutes learning and3–15
minutes export, not measured V40 timing. Cache900s, fit3600s, worker4500s/
external4800s+30s grace, export900s/external930s+30s grace, VRAM20GiB and3GiB
uncompressed return are bounded. Cache and update20 projections use1.25 safety
factors and two measured remaining snapshot allowances. No automatic repeat,
historical pilot, training, VM connection or cleanup is launched here.

V38's one paired-development ArcFace failure and weak native clarity remain.
All previous checkpoint/split/source/terms/provenance/cache/backup and gate
failures remain. The14 app/checkpoint bindings and design are unchanged, with
the original trained DGP primary, Auto/On/Off, mask review, original/mask/result,
PNG and bundle downloads. No new browser or app-quality claim is made. Separate
automatic/assisted completion is still unqualified across masks, sunglasses,
strong lens glare, hands, obstructing hair, scarves and objects. Preserve clear
glasses, non-obstructing hair and visible appearance, with only documented margin.
Request clearer/less-covered input only when usable information is insufficient.

Photographic TRAIN,520 paired development and24 native unpaired CCTV crops remain
separate. No hidden-identity, native PSNR/SSIM, ethnicity or Zamboanga-performance
claim is made. Real Zamboanga CCTV is absent;45 reserved identities/58 crops stay
unopened. Useful reviewed development outputs, independent final review, all
covering families and the full DGP-led app remain required. Goal active/incomplete.

[V39 independent results](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_SPATIAL_DECODER_V39_RESULTS.md>)
[V40 five manual steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_SPATIAL_FIT_V40_VM.md>)
[V40 independent packet audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_spatial_fit_v40_preparation/independent_packet_audit.json>)

The complete preceding handoff follows; its V39-unrun statements are historical.

'''
    at = before.index(b'\n')+1; addition_bytes = section.encode('utf-8')
    (ROOT/'PROJECT_HANDOFF.md').write_bytes(before[:at]+addition_bytes+before[at:])
    bindings = {}
    def merge(mapping):
        for name, digest in mapping.items():
            assert name not in bindings or bindings[name] == digest, name
            bindings[name] = digest
    merge(p['sources_sha256']); merge({(PACKET/name).relative_to(ROOT).as_posix(): digest for name, digest in p['assets_sha256'].items()})
    imported = read(ROOT/'outputs/cctv_dgp_spatial_decoder_v39_return_import.json')
    merge({'outputs/cctv_dgp_spatial_decoder_v39_return/'+name: value for name, value in imported['files_sha256'].items()})
    names = ['CCTV_DGP_SPATIAL_DECODER_V39_RESULTS.md', 'CCTV_DGP_SPATIAL_FIT_V40_VM.md', 'PROJECT_HANDOFF.md',
             'scripts/record_cctv_dgp_v39_return_v40_prepared_milestone.py', 'scripts/verify_cctv_dgp_v39_return_v40_prepared_milestone.py',
             'outputs/cctv-dgp-spatial-fit-v40-execution.tar.gz', 'outputs/cctv-dgp-spatial-fit-v40-execution.tar.gz.sha256',
             'outputs/cctv-dgp-spatial-decoder-v39-results.tar.gz', 'outputs/cctv-dgp-spatial-decoder-v39-results.tar.gz.sha256', 'outputs/cctv-dgp-spatial-decoder-v39-export.json',
             'outputs/cctv_dgp_spatial_decoder_v39_independent_audit.json', 'outputs/cctv_dgp_spatial_decoder_v39_return_import.json',
             'outputs/cctv_dgp_spatial_fit_v40_preparation_draft_v1/cctv-dgp-spatial-fit-v40-execution.tar.gz',
             'outputs/cctv_dgp_spatial_fit_v40_preparation_draft_v1/cctv_dgp_spatial_fit_vm_v40/protocol.json',
             'outputs/cctv_dgp_spatial_fit_v40_preparation_draft_v1/sources/cctv_dgp_spatial_fit_v40_vm.py',
             'outputs/cctv_dgp_spatial_fit_v40_preparation_draft_v1/sources/prepare_cctv_dgp_spatial_fit_v40.py',
             'outputs/cctv_dgp_spatial_fit_v40_preparation_draft_v1/preparation_review.json',
             'outputs/cctv_dgp_spatial_fit_vm_v40/protocol.json']
    names.extend(path.relative_to(ROOT).as_posix() for path in PREP.iterdir() if path.is_file())
    merge({name: sha(ROOT/name) for name in names})
    for name, digest in bindings.items(): assert sha(ROOT/name) == digest
    app = read(ROOT/'outputs/completion_conditioning_union_v1/protocol.json')['app_preservation_sha256']
    assert len(app) == 14
    for name, digest in app.items(): assert sha(ROOT/name) == digest
    write(OUT/'milestone.json', {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
          'previous_milestone_sha256': sha(PRIOR/'milestone.json'), 'previous_handoff_path': (OUT/'before_docs/PROJECT_HANDOFF.md').relative_to(ROOT).as_posix(),
          'document': {'before_sha256': hashlib.sha256(before).hexdigest(), 'addition_bytes': len(addition_bytes)},
          'new_evidence_sha256': bindings, 'V39_independent_return_pass': True, 'V39_optimizer_updates': 0,
          'V40_packet_verified_and_unrun': True, 'V40_updates_bound': 800, 'training_cases': 3905, 'training_references': 781,
          'all14_app_bindings_unchanged': True, 'draft_and_environment_failure_retained': True,
          'VM_calls_here': 0, 'local_gradient_calls': 0, 'local_optimizer_updates': 0, 'training_started_here': False,
          'app_promotion': False, 'automatic_quality_qualification': False, 'assisted_quality_qualification': False,
          'independent_final_review': False, 'goal_complete': False, 'seconds': time.monotonic()-start, 'cap_seconds': 600})
    print({'complete': True, 'bindings': len(bindings), 'seconds': time.monotonic()-start}, flush=True)


if __name__ == '__main__': main()
