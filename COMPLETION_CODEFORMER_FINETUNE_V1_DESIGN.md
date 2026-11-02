# CodeFormer completion first fine-tuning pilot V1

25 October 2026. EXECUTION PACKET PREPARED AND INDEPENDENTLY AUDITED; NOT RUN. No local neural/gradient call or actual training has run for this pilot, and no automatic launch is permitted. The previous eight-case gradient diagnostic returned and passed its independent saved-output/receipt audit. That confirms usable gradient paths; it does not prove useful fine-tuning or hidden-face accuracy. The frozen execution protocol, source/data/archive audits and exact manual commands are now ready in COMPLETION_CODEFORMER_FINETUNE_V1_VM.md. Training remains manual and unlaunched.

## What changes first

Start from the unchanged **original CodeFormer inpainting** checkpoint and train its existing code-prediction and fusion paths together, with separate Adam optimizers. This avoids another inference-wrapper-only adjustment. Keep the codebook/quantizer and generator frozen byte-for-byte. Restoration/U96/u64 and IMPLEMENTATION remain unchanged. No additional model architecture, learning-rate sweep or scheduler is introduced in this first finite pilot.

The new source backend is `scripts/cctv_completion_codeformer_finetune_v1_vm.py`. `research/completion_codeformer_finetune_v1/designcontract.json` records the exact design,40 selected source/target pins, per-image terms and prior independent audit binding. The contract is a source proposal, not the final execution protocol.

## Deterministic TRAIN fit/monitor split

Reuse the audited391 genuine1024 source-reviewed FFHQ TRAIN registry, not the legacy80,000 lower-resolution completion pool. Sort zero-padded reference IDs: first32 are FIT, next8 are MONITOR. All four gradient-calibration references and six earlier HQ probe references remain in FIT. The two subsets have distinct source IDs and target256 byte hashes. This is source-disjoint **adaptive TRAIN monitoring**, not proof of identity disjointness or an independent final test. Keep original53 VAL-role references and reserved FINAL untouched. Official pretraining overlap remains unknown.

The selected40 originals total51,873,288bytes; their exact genuine256 targets total4,239,290bytes. The frozen packet transfers the exact256 targets and newly derived512 targets for this small subset; the original1024 images remain locally retained with their verified source hashes/attributions. Do not transfer all391 originals or duplicate the original370,786,611-byte checkpoint unnecessarily. The execution packet can hash-bind a read-only original checkpoint already installed under ~/forensic-dgp. Do not change or delete prior packets/states.

FIT uses four fixed masks per source: lower face, eye band, image-left lens and cheek/object,128 active cases. MONITOR uses the same four shapes on eight different sources,32 active cases. All40 sources are exact empty-mask/uncovered controls at0/8/16/32; controls take the real app wrapper's bypass path, with a forward hook verifying zero model calls. Visible clear-glasses frames and non-obstructing hair outside removal support remain exact. Never enlarge masks to hide a preservation failure. Automatic proposals are not trained or assessed by this pilot.

## Correct the input-resolution mismatch before learning

The previous gradient diagnostic formed masked input directly at512; the app accepts prepared256. This pilot must replicate the actual app path: start from the exact clean256 target, multiply by visible support, bilinear-resize RGB/support to512, divide by support.clamp_min(1e-8), nearest-resize the binary mask, fill the hole white, normalize to[-1,1], and call w1/noAdaIN. The clean genuine1024 source is downsampled directly to512 only for supervision. A high-resolution target does not silently become a high-resolution input.

Before any update, four fresh original-checkpoint cases compare the pilot delivery path with the retained `CodeFormerCompletion` app wrapper on the same CUDA device. Maximum raw256 mismatch must be <=1e-6, and the full original logical state must remain unchanged. Cross-device CPU/GPU bit equality is not asserted. If fresh parity fails, stop without training.

Clean latent and nearest-code targets for all40 references are cached from the **same original frozen inpainting encoder/codebook** before updates. These are training labels, not an independent clean generative prior or deployment access to hidden ground truth. The clean HQ512 target and delivered256 reference are retained separately.

## Finite optimizer schedule

Run32 paired cycles. Each cycle has four sequential microbatches of batch size1 for each route; loss/4 accumulates across them before each optimizer step. A deterministic seeded Latin schedule covers every128 FIT case exactly once, balances the four geometries per cycle and exposes each32 reference to all four masks. There are32 code-optimizer steps plus32 fusion-optimizer steps: **64 actual optimizer updates**, not32. Backward calls are256. This is one bounded pass over these FIT/mask pairs, not a claim of another full-dataset epoch.

Code route enables encoder, position/feature embeddings, transformer and prediction head only. Loss is CE over all256 clean-nearest-code labels plus MSE to original clean HQ512 latent. Fusion route then freezes the code route and enables existing fusion32/64/128 only. Loss is unclamped masked RGB512 L1 plus masked inner-boundary RGB512 L1; the inner boundary is mask-erode(mask) using a seven-pixel pooling kernel. The architecture's selected-code and encoder-to-fusion detach behavior remains unchanged.

Both optimizers are Adam, LR1e-5, betas(0.9,0.99), eps1e-8, weight_decay0, global gradient-norm clip1 after four microbatches. No LR changes, sweeps, scheduler or EMA. Record component losses, each intended group's connected/finite gradient norms and clip norms; exactly-zero individual tensors do not falsely invalidate an otherwise connected route. Parameters must remain finite after updates; quantizer/generator state and original checkpoint hashes must remain unchanged.

## Prospective monitoring and quality boundaries

At0/8/16/32, save MONITOR delivered256 raw arrays, exact rounded PNGs, fixed masks and hidden-support metrics. Compare normalized mean absolute RGB error inside the hole and in the fixed inner boundary, both RAW and PNG, as macro averages and separately for all four geometries. Keep error-derived metrics separate from visual usefulness. No PSNR/SSIM is computed for unpaired public covering photos.

After8 or16, stop if any monitor macro or per-geometry hole/boundary endpoint worsens more than10% from the original baseline; retain the terminal state and gate failure. At32, the prospective numeric screen requires at least **1% relative reduction of macro delivered-PNG hole L1**, with every RAW/PNG macro and geometry hole/boundary endpoint nonworsening. This1% is a declared small-signal screen for this pilot; it is **not the old restoration “structure gain” metric**, a universal completion threshold or a hidden-identity claim. An apparent1% score change does not qualify anatomy.

Run the32 fixed public covering-photo cases(28 active,4 empty) only at original baseline and terminal state. Retain original-photo versus synthetically degraded-photo conditions separately and compare all seven families and clear/uncovered controls with the exact prior reviewed masks. Historical filenames containing “native” describe original public photos here; they do not turn them into CCTV or Zamboanga evidence. These are unpaired visual development checks, not additional training or update-by-update feedback.

The first pilot's FIT/MONITOR images have procedural missing support on otherwise clear inputs. The degraded public-photo terminal comparison exposes that limitation; successful clear-context training does not establish degraded CCTV generalization. If it fails there, a later independently justified degradation-aware training design is needed. Do not expand this pilot's monitor after seeing results.

Every active output must preserve prepared256 outside-mask pixels exactly in RAW and PNG. Empty bypass preserves its source exactly without a model call. This compositing guarantee and a clear-frame bypass are functional preservation, not evidence of plausible generated anatomy. User review of every planned sheet, useful completion on covered/degraded development examples, and later independent final review remain required. No automatic app promotion follows a numeric pass.

## Timing, storage and migration

Only manually run on the existing GCE g2-standard-4/NVIDIA L4 at ~/forensic-dgp/cctv_completion_codeformer_finetune_v1_vm. Require4GiB initial free storage; stop ordinary work below2GiB free or beyond1536MiB outputs. Hard runtime cap1800seconds; cooperative work stop1680seconds reserves up to120seconds for terminal-state retention. CUDA allocated memory cap20GiB is an operational stop, not a measured minimum requirement. OOM, nonfinite values, source drift, parity failure, exact-preservation failure or exceeded counters stop the run. No rerun or extension is automatic.

Expected maximum:80 clean teacher component calls(40 encoder+40 quantizer),256 training whole-model forwards,128 monitor whole-model forwards,56 public-development whole-model forwards and8 fresh-parity whole-model forwards =528 declared slots, below the600-slot cap. The forward hook independently counts448 whole-model calls; clean teacher component calls are not mislabeled as full network forwards. Controls use no neural forward. Actual typed counters are retained.

Snapshots8/16/32 retain **images, metrics and logical-state hashes only**. They are not recoverable checkpoints and cannot be cherry-picked, rescued or resumed. Save **one reconstructable terminal training state** on normal32-cycle completion, early quality stop or controlled exception. It binds the original checkpoint SHA, all trainable parameter tensors, frozen-state hash, both full Adam states with ordered parameter names, Torch CPU/CUDA/Python/primitive NumPy RNG, exact sampler schedule, completed-cycle counts, actual route/microbatch progress and any partial accumulated gradients. Mid-backward or mid-step failure is explicitly partial; automatic resume is forbidden and its returned state needs audit/new protocol. Terminal serialization writes an exclusive .partial file, flushes/fsyncs it, checks bounded time/storage, then atomically publishes the final name. Failed partial files remain export evidence and are not resumable states. A hard OS kill can prevent a terminal save and must be reported honestly.

The source proof estimated233,839,360 trainable-state bytes plus467,678,720 Adam-moment bytes, at least701,518,080bytes for one terminal state before metadata/serialization. Three such states would waste over2.1GB; this design deliberately saves one. Partial-gradient failure retention can add bytes. Output/export ceilings remain enforced on actual files, and root's separate export wrapper must attempt return packaging after every exit while retaining failed status. Original installed weights, prior gate failures and VM migration backups remain protected.

The supervisor owns a new process group, handles Ctrl+C/SIGTERM by bounded cooperative termination then kills only that owned group if needed, and always records its terminal supervisor receipt. Its external hard cap may prevent saving model state; export/audit must not report training success merely because an archive exists.

## Next concrete action

Root independently reviews this source, prepares the small frozen input packet and exact transfer/tmux/return commands, then the user runs the32-cycle pilot manually. Audit returned source/checkpoint states, optimizer/RNG/sampler, RAW/PNG metrics and every visual sheet before exposing a selected terminal candidate for local user comparison. Never replace the original inpainting baseline or change restoration/IMPLEMENTATION during this research pilot. The completion goal remains active.
