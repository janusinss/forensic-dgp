# Recomputed directions with finite preservation checks

Design review, 10 October 2026. The user selected the best approach after the
group-conflicts return was audited and all outputs reviewed. Prioritize finite
step behavior and TRAIN coverage before another epoch run. This document is a
design, not a prepared executable or a launched study.

## Evidence for changing the experiment

The group-conflicts diagnostic finds a strong common initial direction, but
larger placements reverse some sampled scientific losses and the smallest
placement still fails the other TRAIN cohort. A larger constant displacement
cannot be substituted for longer training. Six sampled losses also oppose the
previous mean-balanced direction. Current weights, targets, masks, geometry,
normalization and all scientific requirements remain fixed.

The next experiment should test a short **recomputed, finite-checked sequence**
on an isolated current-DGP copy. It must count accepted parameter changes as
actual training, even if it uses an explicit descent assignment instead of a
torch optimizer. A serial trajectory must never be labeled a zero-update probe.
All gradients and parameter-learning operations remain manual on the L4 VM.

## Intended finite mechanics study

1. Retain all 100 TRAIN cases and their two named cohorts. Include both cohorts
   separately in the derivative constraints: 64 cohort/source/profile losses,
   not a pooled 32-group mean. This explicitly changes the second cohort's role
   from TRAIN cross-check to sampled fitting evidence. It was never DEV or final;
   retain its earlier role/history and make no independent-generalization claim.
2. Recompute the common direction at every accepted state using all 100 cases,
   frozen normalization and recognizer, and the same declared losses. No momentum
   or stale reuse of the initial direction. A failed bounded direction certificate
   stops and exports; it does not prove global impossibility.
3. Test a bounded small-step backtracking list before accepting a change. Compare
   actual raw and delivered metrics against both the original baseline and the
   preceding accepted state. Preserve MSE, SSIM, recognition, source structure
   and brightness requirements. Reject and restore a placement that violates
   them; preserve its gate receipt and exact output evidence. A derivative
   prediction cannot replace those finite checks.
4. Initially bound the mechanics study to three accepted changes, at most three
   proposals per change and 960 gradient queries. Freeze exact step lengths,
   numerical definitions, timing, storage/VRAM limits and export rules before
   execution. Store the full saved group vectors and directions needed to audit
   each accepted state; budget their roughly one-GiB uncompressed 64-group
   matrix per state before choosing the transfer and storage plan.
5. Measure the original 1% early requirement and all raw/PNG preservation decisions
   at every saved state. A three-change mechanics study does not qualify a model,
   resume V42 or establish a completed epoch. It cannot launch a later run.
   Only verified evidence of stable small-step progress can justify a separate
   finite 50-update/full-corpus study; useful DEV/native and final review remain
   mandatory before an epoch extension or app adoption.

No numeric loss allowance may be widened to make this succeed. Rejecting a
proposal before committing is distinct from allowing a below-1% diagnostic
state to become a qualified model. The 1% stop at 50 updates remains binding.
If progress stays too small, fails preservation or cannot fit the finite storage
and timing budget, retain the stop and reconsider supervision/reconstruction
capacity rather than preparing an automatic continuation.

## Research rationale and limits

[Liu et al., Conflict-Averse Gradient Descent](https://papers.nips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html)
studies how an averaged gradient can harm individual objectives. This supports
checking individual losses; it supplies no CCTV restoration or identity evidence.
[Chen, Tang and Yang, A Barzilai-Borwein Descent Method for Multiobjective Optimization Problems](https://arxiv.org/abs/2204.08616)
describes line-search step-size limitations in multiobjective descent. We infer
that direction selection and finite step-size acceptance should be evaluated
separately here. Neither paper guarantees success for our nonsmooth clamped,
quantized face pipeline, and this design does not claim to implement either
paper's complete algorithm.

## Required preparation before commands exist

Preparation must first resolve the exact three-step trace size, free-space
projection, runtime from the returned L4 timings, independent prospective return
checker and Windows local gradient/training denials. CPU verification may check
saved arithmetic and inference only. Preserve originals, splits and failures.
Do not launch an unverified packet or the historical group-conflicts diagnostic.
The current app continues to use the retained DGP. All five milestones and all
seven automatic/assisted covering families remain required.
