# Bank comparison A r3: returned result and next manual comparison

Published 2026-10-10T17:56:46.712372+00:00. Route A completed a valid finite VM training run but failed
the unchanged quality requirements. It is retained as evidence, with no new
incumbent or app checkpoint. Route B is prepared and has not run.

## Execution and qualification

The original return archive was checked against both checksum/export receipts;
all 7,493 safe regular file members and prepared packet bindings were imported
without altering the archive or running the trainer locally. Archive SHA256:
`2720c5838e6a5da748783ae0e476c2bf2210693434f8b7d25f7e7404dbb26222`. Protocol SHA256: `6ca05c6b87a4588bfea2bc5b80307927c3a2fd47fa419567ee35280b37a78d0b`.
The separate execution audit checked the complete schedule, counters, finite
optimizer/scheduler state and retained CPU/CUDA RNG records.

Route A made 50 optimizer updates using 250 distinct TRAIN references, 125 from
each declared source, and 1,250 profile exposures. This is 0.3201 of a
781-reference epoch: zero complete epochs. The worker took 20.64 minutes,
including preflight and large output snapshots. Fitting time was not recorded
separately; export took approximately four minutes. Trainer exit code 2 records
a planned quality rejection. Export success preserves that rejection.

| Saved output | Relative feature-MSE improvement | Required improvement | Preservation failures |
|---|---:|---:|---:|
| Raw | 0.471905% | 1% | 14 |
| Delivered PNG | 0.473663% | 1% | 15 |

The structure criterion uses the frozen recognizer's feature MSE; it is a proxy,
not a direct percentage of recovered facial anatomy. Failure counts are
overlapping group/metric checks, not counts of people. Raw degraded gains by
declared source were 1.014908%
for `dataset/asian_faces` and
0.343823% for the retained
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
`8ec8456955e6a2524de62bf00e2408f7f5115938f484a6f3b7ff6dd559389734`.

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

Independent closure checked 405,381 retained
research metadata entries and 5,636 retained A
byte hashes before/after deletion. Checkpoints, full states, training caches,
inputs, splits, provenance, instructions, metrics, logs and gate/failure records
were retained. All scientific-cache bytes were not rehashed. Local app bindings
and permanent original checkpoint SHA256 `646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b` remain unchanged.
No training/gradient task or VM start/stop/switch was performed by the agent.

Allocated reclaim: 9.336 GiB; measured free-space increase:
9.328 GiB. Fresh free space: 18.129 GiB; conservative B space after
upload/install: 16.523 GiB, against the unchanged 16 GiB requirement.
This is a current snapshot; manual preflight must check again at launch.

## Manual route B and limits

Run the separate frozen B packet in tmux using
[CCTV_DGP_BANK_COMPARISON_B_V1_VM.md](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_COMPARISON_B_V1_VM.md>).
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

Evidence: [independent_audit.json](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_comparison_v1_r3_A_analysis/independent_audit.json>),
[execution_audit.json](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_comparison_v1_r3_A_analysis/execution_audit.json>),
[visual_review.json](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_comparison_v1_r3_A_analysis/visual_review.json>),
[independent_audit.json](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_A_return_offload_v1/independent_audit.json>),
[inventory_after.json](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_A_return_offload_v1/inventory_after.json>).
