# Original DGP feature-fusion diagnostic: audited findings

The human VM diagnostic completed 280 component-gradient queries in 40.769 seconds, with zero optimizer updates, backwards or epochs. No new checkpoint was created.

The independent local checker verifies all 756 returned files and 301,298,844 saved gradient values. It replays 200 outputs with CPU inference only: raw maximum error 2.29477882e-06, PNG maximum difference 1 byte, vector maximum error 5.25265932e-07. Returned code is not executed and no local gradients are computed.

| State | TRAIN cohort | Fusion improvement norm | Decoder improvement norm | Fusion/decoder ratio | All 23 connected |
| --- | --- | ---: | ---: | ---: | --- |
| 0 | exposed | 0.272977 | 0.302483 | 0.902 | True |
| 0 | unexposed | 0.617877 | 0.503593 | 1.227 | True |
| 50 | exposed | 0.295048 | 0.278725 | 1.059 | True |
| 50 | unexposed | 0.699609 | 0.547107 | 1.279 | True |

The five lateral and three top-down convolutions comprise 11 original feature-fusion tensors with 479,616 parameters. Their values stayed fixed in V31. The comparison decoder has 12 active tensors with 498,627 parameters. All are connected to improvement losses in both measured cohorts at both states.

At the original state, a negative total-objective direction restricted to fusion decreases all three improvement terms in both sampled cohorts.
These are Euclidean first-order derivative observations on matched TRAIN subsets. Gradient norms depend on parameterization; larger norms do not prove more capacity, explain a unique cause or predict finite AdamW behavior. The stopped-state preservation terms remain necessary; they must not be removed to force a structure gain.

V31 remains closed at 0.694525% against the unchanged 1% early requirement. Original checkpoints, stopped weights, source hashes, splits and every failed gate remain retained. No stopped run is resumed.

The next justified design tests one parameter-partition change on a fresh original DGP copy: enable the 11 measured fusion tensors alongside the 12 active decoder tensors. Keep backbone, inactive head4, evaluation buffers, paired clear/degraded batches, full 3,905-case TRAIN corpus, seven losses, initial normalizers, AdamW settings and all gates. Release a finite manually launched L4 packet only after independent source/data/transfer verification.

This diagnostic qualifies no restoration or completion output. Cohorts are synthetic photographic TRAIN data, not native CCTV or independent evaluation. Source labels do not establish ethnicity. Reserved-final pixels remain unopened; useful native development, all covering families and independent final review are still outstanding. The app primary checkpoint is unchanged. Goal active/incomplete.
