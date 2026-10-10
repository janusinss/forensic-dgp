# Current DGP: original parameter ownership review

This read-only review establishes a constraint on any future original-DGP
training design: **the checkpoint's 622 keys are not 622 independent trainable
tensors, and its stored backbone includes layers outside the retained forward.**
No learning recipe, optimizer, gradient query, neural forward, new checkpoint or
VM operation occurs in this review. The architecture question remains pending.

The current retained checkpoint matches SHA256
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
Its loaded state stays unchanged at
`d89f4e52777b5260e94a3e533aa21a0204cbd79b578137881fc6a4c67f1bb0c3`.
The inventory reads the frozen V42 local source and current checkpoint; it does
not execute returned training code.

## Stored objects and forward ownership

| Scope | Unique parameter tensors | Parameters |
| --- | ---: | ---: |
| Complete stored DGP | 181 | 3,312,707 |
| Original reconstruction convolutions | 14 | 609,219 |
| Stored encoder/FPN | 167 | 2,703,488 |
| Backbone tail outside the current forward | 21 | 1,206,080 |
| Structurally reachable in the retained forward | 160 | 2,106,627 |

There are 316 parameter names because some names alias the same objects. The
checkpoint also contains 306 buffer names for 171 unique buffer objects.
Together those 622 names include shared parameters and stored normalization
statistics. All alias values agree with the serialized checkpoint. The unique
parameter count agrees with the earlier original-decoder review.

The stored MobileNet backbone has 19 blocks. The actual `enc0`–`enc4` containers
share blocks 0–15. FPN.forward calls these containers and never calls the full
stored `features` container. Consequently blocks 16–18 are outside that forward.
This conclusion uses the unchanged source AST and actual module ownership, not
a sampled activation trace. Those weights must stay in the preserved checkpoint;
this finding does not authorize deletion or a changed network.

A future optimizer must enumerate unique declared parameter objects. It must
distinguish the structurally unused tail from tensors that participate in a
forward but happen to have zero derivatives for particular data. The reachable
count is not a gradient-connectivity, capacity or restoration-quality result.

## Numerical and historical limits remain relevant

Only two complete parameter tensors consist entirely of nonzero float32
subnormal values: `head4.block0.weight` and `head4.block1.weight`. Their 110,592
values match the previously diagnosed inactive branch. Ninety-eight parameter
tensors contain at least one subnormal value; that alone does not establish
that the other branches are inactive. None of the parameter tensors is entirely
zero.

The retained original-decoder R2 diagnostic has exact zero head4 gradients in
all seven components of all ten batches. The independently audited historical
branch trace has nonzero upstream maps but zero float32 direct-head outputs in
all 50 cases. The fourth map still contributes through the FPN top-down path;
the entire coarse representation must not be described as absent. No new
activation or derivative experiment is performed here.

V28 already trained the other twelve original reconstruction tensors. It
improved landmark structure on its small photographic TRAIN cohort but failed
clear appearance preservation and the brightness requirement. V29's broader
development failure and subsequent finite original-path failures remain binding.
Re-enabling these twelve tensors unchanged would repeat a rejected direction.
The new inventory does not override any failed gate or prove joint feature
training will solve preservation.

## Consequence for the pending design discussion

An original-feature/reconstruction study needs explicit forward ownership,
initial output parity and unchanged normalization, then a finite manual L4
connectivity and learning diagnostic. It must separate missing graph connections
from numerical inactivity and finite appearance regressions. It must retain the
existing scientific thresholds and expose only declared photographic TRAIN
cases. No original encoder/decoder learning code is implemented while the
three-attempt architecture question is pending.

The app, selected checkpoint, datasets, splits and failed states remain intact.
No native CCTV, development or final pixels are opened. Useful development and
independent final outputs, all five restoration milestones and all seven
automatic/assisted covering families remain required. The full goal stays active.

Evidence: `outputs/cctv_dgp_post_v42_original_path_inventory_v1/inventory.json`
and `forward_ownership.json`, the retained original-decoder R2 gradient summary,
and `outputs/cctv_dgp_original_decoder_head4_review_v1/independent_readback.json`.
