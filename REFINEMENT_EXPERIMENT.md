# Image-conditioned residual refinement: preparation

30 September 2026. Architecture implemented; VM runner/data packaging not ready.
No optimization or improvement claim yet. Application and generator unchanged.

Hypothesis: supplying image detail alongside frozen semantic features can improve
spatial discrimination beyond a shallow semantic-only classifier. Existing audits
show distant synthetic false positives and boundary errors; they do not prove this
architecture will solve them. This is a bounded architectural experiment, not a
new loss-weight sweep or a claim of a state-of-the-art design.

`refinement_head.py` freezes the unweighted control epoch-10 parent, projects
64x64 semantic features to 16 channels, and refines at 128x128 with semantic
features, parent logits and RGB. A zero-initialized output layer contributes a
residual to unchanged 256x256 parent logits. Trainable parameters: 14,545.

Compare identical initialized structures with RGB enabled versus zeroed RGB input.
Both receive the same semantic features and parent logits. This holds tensor
shapes/nominal parameter count fixed; RGB-channel weights in the zero-input arm
receive no data gradient, so do not describe effective capacity as identical.
Use the same data/order, fresh optimizer and final-only checkpoint policy. Keep
the parent and existing presence gate frozen to isolate residual refinement.

Parent SHA256:
`152735d14c6b1b84cc873401e99c4c8861a0fdefe9a7b83198a3737c5a44d2bc`
at local `outputs/downloaded_expanded_border/control_epoch_10.pth`, or VM
`~/forensic-dgp/expanded_feature_bundle/outputs/expanded_border_training/control_epoch_10.pth`.

Four local unit tests passed: exact initial parent output/freeze behavior,
zero-input control RGB invariance, active RGB influence/gradient isolation, invalid
RGB rejection. The real parent plus one cached training example also verified
identical initial states and exact output equality for both arms. Evidence:
`outputs/refinement_preparation.json`. No optimizer or training ran locally.

Next implementation: VM-only runner with verified RGB regeneration for every
cached training input (including real images), unchanged targets, equal budgets,
CPU refusal, GPU preflight and output/checkpoint provenance. Establish this data
path before issuing training commands. No extra encoder extraction is required.
Actual GPU memory/runtime remains unverified. Known frozen-gate failures and glare
transfer still need separate resolution; this experiment alone cannot qualify a
full pipeline. Original real/synthetic safeguards and completion review still apply.

Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM root: `~/forensic-dgp/expanded_feature_bundle/`.
