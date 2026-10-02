# Frozen detector complementarity — verified 2 October 2026

The completed training-only diagnostic rejects fixed union/intersection as a
repair. Both frozen binary masks miss 8,205 of 21,853 reviewed lens pixels
(37.55%). A selector restricted to these two binary predictions cannot create
that missing coverage. Native covering learning is needed before another routing
experiment; this result does not establish that all learned routing is infeasible.

Windows root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM repository: `~/forensic-dgp/`; latest executed workspace:
`~/forensic-dgp/coverage_vm_bundle/`. This diagnostic ran locally with CPU
inference only. No new VM upload, optimizer update or completion generation occurred.

## Fixed execution and independent audit

Protocol: `FROZEN_COMPLEMENTARITY_PROTOCOL.md` and
`outputs/frozen_complementarity_protocol_v1/protocol.json` under the Windows root.
The latter was frozen before prediction. There are 973 training cases: 83 supported
real images, 610 unique safe replay cases from the executed core schedule, and 280
reviewed reflection fixtures. Existing source quarantine and held-out membership
remain unchanged. The fourteen pending native annotations remain excluded.

The original Gated U-Net parent and final-budget reflective epoch42 branch stayed
frozen. There were 1,593 new CPU image predictions and 353 reused candidate masks
whose pixels equal the independently reproduced VM masks. All 1,946 saved head
masks, seven systems and 6,811 case/system recounts pass the independent audit.
Source checkpoint tensor fingerprints match the logged pre/post states. No
identical-RGB contradictory labels were found on common supervised support.
All eleven diagnostic, protocol and independent-audit tests pass.

Completed evidence: `outputs/frozen_complementarity_v1/`.
Independent verification: `outputs/frozen_complementarity_validation_v1/verification.json`.
The old `partial.json` is an historical snapshot, not the current process status.
Execution completed with exit0; no diagnostic process needs restarting.

## Training evidence

| Fixed system | Real83 IoU | Replay610 IoU | Reflection280 IoU |
| --- | ---: | ---: | ---: |
| Original parent | 0.072010 | 0.968492 | 0.217556 |
| Face-specific candidate | 0.915044 | 0.950661 | 0.809366 |
| Union | 0.826500 | 0.968231 | 0.649972 |
| Intersection | 0.074687 | 0.950793 | 0.255080 |
| Conservative dominance oracle | 0.503702 | 0.979381 | 0.516712 |
| Pixel oracle | 0.932612 | 0.989570 | 0.843376 |

Union adds real visible false positives and nine clear-image errors, with replay
retention regressions. Intersection increases real empty-mask cases from17 to25
and misses most native coverings. Both fail the predeclared training checks.

The dominance oracle chooses the candidate only when that image's TP does not
fall and FP does not rise, with a strict gain; otherwise it uses the parent. It is
a conservative non-tradeoff diagnostic, not globally optimal image routing or a
general upper bound. It switches34/83 real,11/610 replay and143/280 fixture cases.
The pool oracle uses known source membership. The pixel oracle uses target labels
to choose OR on covered pixels and AND on visible pixels. None is deployable.

| Lens cohort | Labelled pixels | Parent TP | Candidate TP | Both miss |
| --- | ---: | ---: | ---: | ---: |
| Previous two glare labels | 2,031 | 0 | 1,369 | 662 |
| Native source171 | 5,666 | 0 | 2,484 | 3,182 |
| Native source216 | 5,284 | 0 | 3,247 | 2,037 |
| Native source348 | 4,828 | 2,057 | 3,682 | 812 |
| Native source374 | 4,044 | 0 | 2,532 | 1,512 |

Candidate lens recovery is13,314/21,853 (60.93%); union/pixel oracle reaches only
13,648/21,853 (62.45%). On the ten native additions, candidate IoU is0.594782
versus0.925360 on the original73 training examples. All six new transparent
controls and26 original controls remain empty. These are approximate reviewed
training labels, not hidden anatomy ground truth or generalization evidence.

## Fixed ten-row visual review

The bound 1,152×2,256 preview was inspected: four new lens cases, two previous
glare cases, two transparent controls and two fixed degraded replay examples.
Green denotes TP, red FP, yellow FN and purple ignored support. Oracle columns
explicitly disclose target use. The face-specific branch covers inner lens areas
but omits broad boundaries, especially source171's left lens. The union adds
little new coverage except source348 and also adds false positives. The two
transparent controls stay empty. The parent better preserves the irregular replay
boundary. Source171's unresolved lower band is correctly excluded from scoring.

This preview contains detector masks, not restored or completed facial output.
No end-to-end completion improvement has been demonstrated.

## Recheck and next action

The returned649,143,021-byte `outputs/reflection-coverage-results.tar.gz` still
matches its LF checksum:
`a5714641943da67137a51866e4614b2b2841b2447145c816a51a39e7c52b032b`.
No repeat download is required.

Prepare a separate bounded VM native-expert pilot using the supported real labels
and existing reviewed fixtures. Retain the original parent separately. Predeclare
fit/clear-control checks, exact budget and inherited optimizer moments before
execution. Changing sampling/objective creates a new expert experiment, not an
isolated causal test of native data. A later learned input-only selector needs its
own protocol and unchanged development gates; useful expert fit alone is insufficient.

The original real gate and all five synthetic gates remain unchanged, including
synthetic IoU≥0.9746899906463458, missed fraction≤0.016476187160583314,
visible FP≤0.0013327836915128428, empty masks≤1 and clear errors0/80.
No `best_detector.pth` or application model is promoted. Actual fitting stays on
the VM. The full goal remains active; hidden facial features remain estimates.

| Artifact | SHA256 |
| --- | --- |
| Frozen protocol | `2a42c084a969beb47613e87cca088e9fb599254335a47f76ebffcc01e72a4529` |
| Completed results | `2880c85f067c013f6998faf1744177be79b8ed0a50d949d6e35ae5da14563928` |
| Independent verification | `449ddc0f85908f653099cb229bfdbce16d152df1eb0ec7a7f4c881bd18d8134b` |
| Reviewed preview | `6861ae280d77fc6929aab58e67e344f31f56acc383cda3bbae4fc65cd98cb7a7` |
| Independent auditor | `53775b73305776ab82d7e8e6f1ea2320763253a78f34cbe31b7fb198b6bd3ca9` |
