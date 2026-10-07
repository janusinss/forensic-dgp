# DGP design discussion after the audited V32 r2 stop

The next model change follows a fresh design discussion. V30, V31 and V32 have
missed the same early structure requirement. All original and stopped checkpoints,
splits, provenance, source receipts and failed gates remain binding. None is
resumed or selected for the application.

| Finite stopped50 recipe | Degraded TRAIN structure gain |1% requirement |
| --- | ---: | --- |
| V30: wider corpus, shuffled five-case batches | 0.805717% | Failed |
| V31: paired clear/four-degradation reference batches | 0.694525% | Failed |
| V32 r2: original fusion plus active decoder, same paired batches | 0.970672% | Failed |

R1's root mismatch was a separate pre-optimizer packaging failure. R2 fixes that
guard and actually trains. It is inappropriate to treat R2's structure failure
as another install error or remove its assertion because the gain is close.

The invalid assumption is that favourable initial descent measurements and
release of the original fusion weights would meet the finite whole-face
requirement under the unchanged training design. The result is weak despite
nonzero gradients, changed weights and passing group-average preservation.
Greater connectivity and a slightly better patch score are insufficient evidence
of useful eyelid, nose, lip and outline reconstruction.

The saved original-to-stopped parameter path changes fusion and decoder by about
0.339% of their initial L2 norms. The three initial improvement-gradient dot
products with that finite displacement are negative, and the negative-total-
gradient cosine is0.122446. These observations do not isolate a cause or predict
AdamW steps. In particular, zero initial preservation gradients follow initial
baseline equality; they do not show that preservation terms stay inactive after
learning. The recorded whole-face previews remain soft.

The recommended next review concerns **why the present DGP learns weak structure**:
inspect raw versus delivered losses, normalization, the actual constrained output
path, preservation activation and effective update magnitudes. Start with the
audited saved arrays and source. If an unresolved quantity requires a VM probe,
define a separate finite diagnostic prospectively; keep the original/stopped
states read-only, explicitly count gradient/optimizer calls and retain timing,
stops and returned evidence. No claimed reconstruction of unsaved optimizer
history and no unchanged continuation of V32.

A new spatial decoder remains an alternative within the user's earlier spatial/
feature-path decision. It must access the observed256 crop and the original
multiscale features, preserve appearance and introduce a genuinely different,
justified forward path. Its initial parity/connectivity would be preconditions,
not quality evidence. Choosing it now without the learning review risks repeating
the same unsuccessful adjustment pattern.

Both directions retain all visible facial features, clear glasses and
non-obstructing hair, accepting softness only where useful structure remains.
Neither substitutes a pretrained restoration model for the user's own DGP.
Neither qualifies mask/sunglasses/glare/hand/hair/scarf/object completion; that
component and automatic-versus-assisted quality remain separate requirements.

The workspace [AGENTS.md](<C:/xampp/htdocs/YEAR 4/Testing/AGENTS.md:16>) requires:
"If an issue remains unresolved after 3 consecutive attempts, STOP modifying code.
Name the invalid assumption and ask one diagnostic question."
The applied [systematic-debugging skill](<C:/xampp/htdocs/YEAR 4/Testing/.codex/skills/systematic-debugging/SKILL.md>)
also says "DON'T attempt Fix #4 without architectural discussion" and "Discuss
with your human partner before attempting more fixes."
These are explicit discussion requirements before another model fix. The user
answered the diagnostic question on7October2026: **"apply the best approach and
do research also if needed"**. This authorizes the agent to choose the design
direction and research supporting evidence. It does not authorize automatic
training or waive any failed gate.

The discussion question is: **Should the next review diagnose the present DGP's
weak learning before another model change, or design a new spatial decoder?**
Selected direction: **diagnose the current learning design first**, using the
saved loss and weight-change evidence and primary research. This is the agent's
interpretation of the user's best-approach instruction, rather than a claim that
the user selected the option's exact wording. The discussion is satisfied; the
scientific cause remains under investigation. Any new VM experiment still needs
a distinct justified finite protocol and the user's manual launch.

The exact reply and this interpretation are recorded in
[the decision receipt](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_post_v32_architecture_review_v1/user_decision.json>).

Evidence: [audited V32 r2 findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FEATURE_FUSION_V32_R2_RESULTS.md>),
[saved parameter path](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_feature_fusion_v32_r2_displacement_v1/analysis.json>),
[all50 development observations](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_feature_fusion_v32_r2_failure_review_v1/visual_review.json>).
No new native or reserved pixels are viewed. Actual training stays the human's
manual existing-L4 workflow. Goal active/incomplete.
