"""Publish verified A/maintenance status while retaining every previous document byte."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r3_A_analysis'
RETURN = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r3_A_return'
MAINTENANCE = ROOT / 'outputs/cctv_dgp_bank_A_return_offload_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def link(name):
    return '[' + Path(name).name + '](<' + (ROOT / name).as_posix() + '>)'


def main():
    receipt = ANALYSIS / 'publication.json'
    assert not receipt.exists()
    imported = read(ANALYSIS / 'import.json')
    audit = read(ANALYSIS / 'independent_audit.json')
    execution = read(ANALYSIS / 'execution_audit.json')
    visual = read(ANALYSIS / 'visual_review.json')
    clean = read(MAINTENANCE / 'independent_audit.json')
    storage = read(MAINTENANCE / 'inventory_after.json')
    assert all(r['complete'] for r in [imported, audit, execution, visual, clean, storage])
    assert visual['viewed_sheets'] == 26 and visual['full_metric_audit_complete']
    assert audit['saved_paired_raw_PNG_cases_checked'] == 7810 and audit['native_compositions_checked'] == 48
    assert audit['optimizer_updates_in_VM'] == execution['actual_optimizer_updates'] == 50
    assert clean['exact_deletions_checked'] == 1860 and clean['retained_state_exactly_equal']
    assert storage['manual_B_space_requirement_met'] and storage['conservative_B_post_install_GiB'] >= 16
    assert not audit['model_qualified'] and not visual['independent_final_review']
    gate = read(RETURN / 'outputs/bank_comparison_v1/quality_stop_A.json')['outcome']['gates']
    assert not gate['raw']['pass'] and not gate['png']['pass']
    assert gate['raw']['minimum_gain'] == gate['png']['minimum_gain'] == .01
    assert len(gate['raw']['preservation_failures']) == 14 and len(gate['png']['preservation_failures']) == 15
    baseline = ROOT / 'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth'
    baseline_pin = '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'
    assert sha(baseline) == baseline_pin
    prepared = read(ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r3_preparation.json')
    route, = [r for r in prepared['routes'] if r['route'] == 'B']
    assert storage['research_usage']['exit_code'] == 0
    assert route['name'] not in storage['research_usage']['stdout'], 'Retain and audit an existing B root rather than relaunch it'
    packet = ROOT / 'outputs' / route['name']
    archive = ROOT / 'outputs' / (route['stem'] + '-execution.tar.gz')
    checksum = archive.with_name(archive.name + '.sha256')
    assert sha(packet / 'protocol.json') == route['protocol_sha256']
    assert archive.stat().st_size == route['archive_bytes'] and sha(archive) == route['archive_sha256']
    assert checksum.read_bytes() == (route['archive_sha256'] + '  ' + archive.name + '\n').encode('ascii')
    assert read(ROOT / 'outputs/cctv_dgp_bank_split_transfer_v2_audit.json')['complete']
    assert not (ROOT / 'outputs/cctv-dgp-bank-comparison-v1-r2-B-results.tar.gz').exists(), 'A/B report needs updating if B already returned'
    prior = read(ANALYSIS / 'documents_before/manifest.json')['files_sha256']
    for name, expected in prior.items():
        assert sha(ROOT / name) == sha(ANALYSIS / 'documents_before' / name) == expected
    raw_gain = gate['raw']['relative_feature_gain'] * 100
    png_gain = gate['png']['relative_feature_gain'] * 100
    raw_sources = gate['raw']['source_feature_gains']
    now = datetime.now(timezone.utc).isoformat()
    free = storage['free_GiB']
    projected = storage['conservative_B_post_install_GiB']
    allocated = clean['allocated_reclaim_bytes'] / 1024**3
    recovered = clean['measured_recovered_bytes'] / 1024**3
    worker = execution['worker_seconds'] / 60
    epoch = execution['exposures']['A']['fraction_of_781_reference_epoch']
    report_name = 'CCTV_DGP_BANK_A_R3_RETURN_REVIEW.md'
    assert not (ROOT / report_name).exists()
    report = f'''# Bank comparison A r3: returned result and next manual comparison

Published {now}. Route A completed a valid finite VM training run but failed
the unchanged quality requirements. It is retained as evidence, with no new
incumbent or app checkpoint. Route B is prepared and has not run.

## Execution and qualification

The original return archive was checked against both checksum/export receipts;
all 7,493 safe regular file members and prepared packet bindings were imported
without altering the archive or running the trainer locally. Archive SHA256:
`{imported['archive_sha256']}`. Protocol SHA256: `{audit['protocol_sha256']}`.
The separate execution audit checked the complete schedule, counters, finite
optimizer/scheduler state and retained CPU/CUDA RNG records.

Route A made 50 optimizer updates using 250 distinct TRAIN references, 125 from
each declared source, and 1,250 profile exposures. This is {epoch:.4f} of a
781-reference epoch: zero complete epochs. The worker took {worker:.2f} minutes,
including preflight and large output snapshots. Fitting time was not recorded
separately; export took approximately four minutes. Trainer exit code 2 records
a planned quality rejection. Export success preserves that rejection.

| Saved output | Relative feature-MSE improvement | Required improvement | Preservation failures |
|---|---:|---:|---:|
| Raw | {raw_gain:.6f}% | 1% | 14 |
| Delivered PNG | {png_gain:.6f}% | 1% | 15 |

The structure criterion uses the frozen recognizer's feature MSE; it is a proxy,
not a direct percentage of recovered facial anatomy. Failure counts are
overlapping group/metric checks, not counts of people. Raw degraded gains by
declared source were {raw_sources['dataset/asian_faces/degraded']*100:.6f}%
for `dataset/asian_faces` and
{raw_sources['dataset/thumbnails128x128/degraded']*100:.6f}% for the retained
FFHQ-derived references. Source labels do not establish ethnicity. The lowlight
and compound profiles account for important appearance/quality regressions;
better clear-image preservation cannot override those failures. The unchanged
gate decision is rejection for both raw and delivered images. No failed state
is approved for continuation or app promotion.

The independent checker recomputed 7,810 paired case records (baseline plus
A50), checked 48 native compositions, strict full-state loading and unchanged
frozen tensors, and replayed two limited model/recognizer batches on CPU.
It did not replay every model forward, run local training, or establish final
restoration quality. The retained A50 full-state SHA256 is
`{execution['states'][0]['full_state_sha256']}`.

## Viewed outputs

All 26 planned sheets were displayed at original resolution: 20 paired TRAIN
sheets covering 100 previews and six native DEV sheets covering 24 unpaired
ChokePoint crops. Clear faces retain more input eye, mouth, hair and ordinary
glasses detail than the original model's smoothing. Blur and compound outputs
still lack convincing central facial detail; dark outputs show darker or shifted
appearance. Later native crops lose less observed detail, while early, dark and
turned crops remain soft. These limited improvements do not qualify the model.

The sheet review was by the implementing assistant. The independent saved-output
audit is separate; independent final visual review remains pending. The 248
missing B/B-disabled tiles mean those routes were not run, not that B failed.
Native CCTV has no aligned clean reference; do not attach paired PSNR/SSIM or
Zamboanga performance claims to it. Reserved final identities were not used.
No completion family was trained or qualified by this restoration run.

## Backup-bound VM maintenance

After return audit and visual closure, authorized maintenance removed exactly
1,858 inactive baseline/A50 RGB output files and two home archives: the returned
A archive and an obsolete original-v1 execution upload. All 1,860 deletions
had complete hash-matched local backups and exact UID, inode, size and singly
linked-file checks. The local return/archive evidence remains.

Independent closure checked {clean['retained_metadata_entries']:,} retained
research metadata entries and {clean['retained_bank_byte_hashes']:,} retained A
byte hashes before/after deletion. Checkpoints, full states, training caches,
inputs, splits, provenance, instructions, metrics, logs and gate/failure records
were retained. All scientific-cache bytes were not rehashed. Local app bindings
and permanent original checkpoint SHA256 `{baseline_pin}` remain unchanged.
No training/gradient task or VM start/stop/switch was performed by the agent.

Allocated reclaim: {allocated:.3f} GiB; measured free-space increase:
{recovered:.3f} GiB. Fresh free space: {free:.3f} GiB; conservative B space after
upload/install: {projected:.3f} GiB, against the unchanged 16 GiB requirement.
This is a current snapshot; manual preflight must check again at launch.

## Manual route B and limits

Run the separate frozen B packet in tmux using
{link('CCTV_DGP_BANK_COMPARISON_B_V1_VM.md')}.
B starts from its own declared initializer. Its frozen external StyleGAN2
face-generating bank is a declared prior; our trained input conditioning and
256x256 reconstruction must earn acceptance. It is not a generator trained
entirely by us, an exact published DGP reproduction, or a pretrained restorer
substitution. Match A's data, schedule, 50 updates and preservation requirements.

The B return must undergo full saved-output/state checks, the bank-disabled
ablation, matched A/B data/baseline verification and all planned visual sheets.
The new comparison/import/review helpers are prepared sources, not completed B
validation. Additional epochs depend on useful preserved learning. Historical
failed recipes, automatic training and gate bypasses remain excluded.

The goal stays ACTIVE. No accepted new incumbent, app model change, independent
final evaluation or seven-family completion qualification is delivered here.
All five milestones, app Auto/override and bundled inline Playwright flow,
clearer/less-covered input handling, and seven separate automatic/assisted
covering-family evaluations remain required.

Evidence: {link('outputs/cctv_dgp_bank_comparison_v1_r3_A_analysis/independent_audit.json')},
{link('outputs/cctv_dgp_bank_comparison_v1_r3_A_analysis/execution_audit.json')},
{link('outputs/cctv_dgp_bank_comparison_v1_r3_A_analysis/visual_review.json')},
{link('outputs/cctv_dgp_bank_A_return_offload_v1/independent_audit.json')},
{link('outputs/cctv_dgp_bank_A_return_offload_v1/inventory_after.json')}.
'''
    header = f'''# Latest verified status - 11 October 2026: A quality stop retained; B manual comparison ready

Route A r3 made 50 updates, 250 distinct TRAIN references and 1,250 profile
exposures: {epoch:.4f} epoch, zero complete epochs. Worker: {worker:.2f} minutes.
Raw/PNG feature-MSE gains: {raw_gain:.6f}% / {png_gain:.6f}%, below the unchanged
1% requirement, with 14 / 15 overlapping group/metric preservation failures.
This was a quality stop, not an execution/storage failure. Preserve the failed
state and gates; no extra epoch run, new incumbent or app promotion is approved.

Independent return checks covered 7,810 paired records, 48 native compositions,
strict full-state/frozen-tensor checks and two limited CPU replay batches.
All26 visual sheets were viewed:100 paired TRAIN previews and24 native DEV
unpaired crops. Clear/higher-information faces lose less detail, while blur,
dark and compound results remain insufficient. Independent final review and
all seven completion families remain pending; final identities stayed unused.

Backup-verified authorized maintenance removed 1,858 inactive RGB files and
two home archive copies, preserving research/checkpoint/state/cache/split/
instruction/provenance/gate/log evidence. Independent closure checked
{clean['retained_metadata_entries']:,} metadata entries and
{clean['retained_bank_byte_hashes']:,} retained A byte hashes; all scientific
cache bytes were not rehashed. Reclaimed allocation: {allocated:.3f} GiB.
Fresh free space: {free:.3f} GiB; conservative B post-install: {projected:.3f}
GiB, meeting the unchanged16 GiB requirement at this snapshot. No actual
training, gradient task, VM start/stop/switch or app/checkpoint change was
performed by the agent. The launch preflight must check resources again.

Manual next: {link('CCTV_DGP_BANK_COMPARISON_B_V1_VM.md')}.
B remains unrun: a separate frozen external generative prior with our learned
conditioning/reconstruction, equal A exposures and unchanged preservation
criteria. The full comparative goal and five milestones remain ACTIVE;
historical sections below retain their original, dated meaning.
Detailed evidence and limits: {link(report_name)}.

---

'''
    (ROOT / report_name).write_text(report, encoding='utf-8', newline='\n')
    after = {}
    prefix = header.encode('utf-8')
    for name, expected in prior.items():
        old = (ANALYSIS / 'documents_before' / name).read_bytes()
        assert sha(ROOT / name) == expected
        (ROOT / name).write_bytes(prefix + old)
        assert (ROOT / name).read_bytes()[len(prefix):] == old
        after[name] = sha(ROOT / name)
    guide = ROOT / 'CCTV_DGP_BANK_COMPARISON_B_V1_VM.md'
    previous_guide = guide.read_bytes()
    (ANALYSIS / 'B_guide_before_storage_close.md').write_bytes(previous_guide)
    note = f'''Verified maintenance snapshot {now}: A independent return/visual audit
and exact backup-bound cleanup closed. VM free space {free:.3f} GiB;
conservative B post-install free space {projected:.3f} GiB. B is ready for the
user's manual transfer/preflight at this snapshot; runtime resources are checked
again by the fixed launch. Full review: {link(report_name)}.

'''
    first, rest = previous_guide.split(b'\n', 1)
    guide.write_bytes(first + b'\n\n' + note.encode('utf-8') + rest)
    output = dict(complete=True, UTC=now, original_document_sha256=prior,
        published_document_sha256=after, all_previous_document_bytes_retained=True,
        report_sha256=sha(ROOT / report_name), B_guide_sha256=sha(guide),
        A_audit_sha256=sha(ANALYSIS / 'independent_audit.json'),
        A_visual_review_sha256=sha(ANALYSIS / 'visual_review.json'),
        maintenance_audit_sha256=sha(MAINTENANCE / 'independent_audit.json'),
        live_storage_sha256=sha(MAINTENANCE / 'inventory_after.json'),
        original_app_checkpoint_sha256=sha(baseline),
        B_protocol_sha256=sha(packet / 'protocol.json'), B_archive_sha256=sha(archive),
        B_checksum_exact_LF=True, B_training_pending=True,
        goal_status='active', model_qualified=False, app_promotion=False,
        local_or_automatic_training_calls=0)
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(dict(complete=True, report=report_name, current_goal='active', B_manual_ready=True,
               free_GiB=free, conservative_B_post_install_GiB=projected))


if __name__ == '__main__':
    main()
