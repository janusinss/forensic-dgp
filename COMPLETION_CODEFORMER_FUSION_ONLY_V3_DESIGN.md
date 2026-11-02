# Completion fusion-only V3: distinct R1 continuation

26 October 2026. Prospective design; no new training has launched.

V2 helped some new challenge cases but regressed on clear eyes/lenses and failed
the fixed gates. It also failed terminal saving. Keep trained R1. The next test
freezes R1 encoder/code prediction and original generator/codebook (443 tensors),
adapting only54 existing fusion tensors with the same styled data and fusion loss.
This is a route isolation experiment; fusion itself may still cause regressions.

Retained V2 u64 outputs serve as historical joint-route evidence. They are not
rerun. Restore full R1 Adam/RNG state, keep code Adam exactly at32, and perform
at most64 fusion updates/256 backwards. All256 fixed FIT pairs occur once in the
same64x4 schedule. No code CE, latent objective, teacher encoder or quantizer calls.

Before updates, four canonical R1 wrapper pairs must meet the existing1e-6 gate.
Initial R1 raw256 must match all48 historical clear/challenge cases within1e-6,
and all48 PNG files must match byte-for-byte. Historical arrays are compact,
exact saved result256 floats with original NPZ/audit/archive pins. Stop if this
compatibility does not hold; no threshold relaxation or mismatched pooling.

Keep original32-clear and16-challenge cohorts separate, requiring >=1% mean
PNG hole-error reduction versus R1 and every declared raw/PNG subgroup hole/
boundary endpoint nonworse. Preserve exact visible pixels, empty/refusal controls,
all224 outputs and all20 planned comparison pages. Public32 cases remain unpaired.
Final human review and covering-family acceptance are still incomplete.

Planned480 whole/typed forwards:256 fusionFIT +128 clear +32 challenge +56 public
+8 parity, capped600. At most64 new Adam steps/256 backwards. Cumulative from R1:
128 Adam updates,512 backwards,1008typed/928whole; code step32/fusion step96.

Integrate the separately admitted O(1) writer into new sources only, preserving
V2 history, all byte/time/disk policies and failed partials. Instrument pre-save,
CPU copy/assembly, serialization, fsync and commit. Retain full332-tensor R1 delta,
both Adam/RNG states and sampler/counters. Keep1800s hard/1680s cooperative limits,
7GiB start/2GiB diskfloor,2304MiB output+raw/compressed export and900s export.
Local fixtures are source/serialization only; all actual training remains manual L4.

The controlling machine-readable contract and evidence pins are in
research/completion_codeformer_fusion_only_v3_design_v1/root_design.json.
Source, independent admission and transfer archive are required before launch.
A prepared packet will not establish quality. Current app and other branches stay fixed.
