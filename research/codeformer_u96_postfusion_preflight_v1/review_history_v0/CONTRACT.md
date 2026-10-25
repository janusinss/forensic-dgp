# Complete-U96 post-fusion preflight V1

Prepared only. Actual execution is manual on the existing NVIDIA L4 in tmux.
This diagnostic performs no optimizer construction or updates, no fitting,
no parameter perturbations and no trained-checkpoint creation. A successful
preflight is not useful restoration or permission to train automatically.

## Files and ownership

Root owns `packet.py`, stage/transfer builder, entry CLI, shell supervisor,
export/guide and returned-output auditor. Runtime author owns `worker.py`.
Stage name: `cctv_codeformer_u96_postfusion_preflight_v1_vm` under ~/forensic-dgp.
Package imports and --verify-transfer are inert: zero Torch/model calls.

`packet.verify(root,expected_sha)` verifies exact `protocol.json` hash plus
every static `files` path/bytes/SHA, returns protocol. Protocol itself is not
in its recursive file map. Only `outputs/`, `run.log`, `supervisor.json` and
`export.json` can be extra runtime paths. Reject any prior run marker before
VM work; no automatic resume/retry. `packet.require_vm(root,p)` checks Linux,
exact home/stage path, current cwd and TMUX, and initial free disk >=6GiB
before Torch import; returns host facts. It does not start a VM or download.
`packet.pin(root,path)` and `packet.sha256(path)` stream regular files.
`packet.regular(path)` rejects links/reparse/nonregular/multi-link paths.

Worker API `worker.run(root,protocol,expected_sha)` owns runtime output writing
exclusively below `outputs/u96_postfusion_preflight_v1/` (O). It verifies before
Torch import and calls require_vm. Writer must refuse preexisting O/started.json
and use exclusive writes, finite JSON and no pickled arrays. Save failure.json
on error and propagate failure so supervisor/export preserve it. Do not alter
static files. No worker entrypoint executes on import.

## Protocol fields

`format`, `stage_name`, `files`, `model` with `r2` and `u96` exact file pins,
`prototype` source/design/review pins, `data` provenance and input-admission
pins, `witnesses` exactly8 rows in four fixed TRAIN references with clear then
localGaussian1.2 each: `{id,reference_id,role,input,target,input_RGB8_sha256,
target_RGB8_sha256,gradient_witness}`. Only4 local rows have gradient_witness.
GlobalGaussian inputs were reviewed as auxiliary input-only controls, not
selected for these forwards. All witnesses are RGB256, full observed support.
No DEV/FINAL/native/wild inputs and no preprocessing in this diagnostic.

`limits`: full_forward_attempts20, full_forward_returns20,
reference_forwards8, candidate_parity_forwards8, candidate_gradient_forwards4,
gradient_queries4, optimizer_constructions0, optimizer_updates0,
internal_seconds1200, external_seconds1260, per_phase_seconds180,
initial_free_disk_bytes6442450944, continuing_disk_reserve4294967296,
maximum_RSS_bytes10737418240, maximum_CUDA_reserved_bytes19327352832,
maximum_output_bytes134217728. Check before/after each reserved operation and
write phase timing/resources. A full call includes four frozen-U96 adapter
sites; candidate calls also include four new head sites. No cache profiling
or extra hidden baseline/teacher calls. All errors retain partial evidence.

## Runtime graph and initial states

Load exact R2 `params_ema`515 and U96 `adapter_state`24, canonical digests fixed
by the independently reviewed prototype. An independently implemented frozen
reference contains R2 plus exact U96 pre-SFT adapters, same argument formula
as the existing app; it must not use the CPU-only app inference wrapper.
Candidate is the new prototype factory with all539 incumbent tensors frozen
and36 new heads, zero final projections. Fixed singleton CUDA FP32,
w1/detach16/AdaIN-on, no autocast, eval, prepared256 bilinear512[-1,1].

Preserve factory caller CPU/CUDA RNG. Check both frozen states/candidate575
and new heads before/after run; no parameters may change. Frozen grads remain
None. Avoid external hooks on candidate; its owned lifecycle requires a clean
namespace. Source model factory and all numerical claims remain unproven until
real returned evidence is independently audited.

For each8inputs: reference no_grad then candidate no_grad, ledger before/after
each base forward; verify all4sites and cleanup, compare native512 and stock
returned logits/LQ, delivered float32RAW256 and floorRGB8 PNG. Require exact
value/native/RAW/PNG parity with this same-device reference; log bit hashes
and signed-zero/maximum error diagnostics. Do not relax a mismatch. This is
not bit-exact CPU/L4 parity or proof against an independent implementation.

Delivery MUST remain connected when used for gradients:
`F.interpolate(((native.float()+1)/2).clamp(0,1),(256,256),mode='bilinear',align_corners=False)`.
Do not reuse historical delivered_rgb, which detaches under no_grad. Save
unclipped256 separately from clipped RAW and floorPNG; no display enhancement.

## Four combined-loss gradient witnesses

Each local input receives one additional candidate autograd forward and one
`torch.autograd.grad` query on the36 heads. Objective: equal mean of imageL1
on central128 and outside, plus0.25 equal mean of adjacent-residual L1 inside
each region. Regions are known synthetic TRAIN supervision, never deployment
input, automatic masks or completion masks. Log terms separately and clipping.
No optimizer, backward accumulation, updates, line search or perturbed weights.

Require finite joint gradients and a nonzero final-output projection gradient
at each of4sites. Earlier layer gradients may be zero at zero initialization;
do not require all36 tensors nonzero. Record per-tensor norm/max/count and
all539 frozen grad fields None. Anchor losses are zero at exact U96 identity;
this preflight does not test anchor optimization, actual-step preservation,
learned improvement or residual/output sensitivity after perturbation. Those
remain separate requirements before any learning pilot. Record per-site
structural execution and existing bound limits without adding diagnostic
hooks forbidden by the prototype.

O outputs: started.json, preflight.json, events.jsonl, resources.jsonl,
parity/{id}/{reference|candidate}-raw.npy,-unclipped.npy,.png,parity.json;
parity/{id}/{reference|candidate}-native512.npy,-logits.npy,-LQ.npy;
gradient/{id}/candidate-raw.npy,.png,gradient.json;
initial_heads.pt and final_heads.pt (36 new untrained tensors only),
state_invariants.json, result.json or failure.json. Save actual counters,
timing/RNG/state hashes and explicit no-training/quality status. Return archive
excludes large baseline weights but includes all source/protocol/data evidence.
Stage exporter uses explicit gzip.GzipFile plus streaming tar mode (not the
historically broken compresslevel argument to tarfile.open(mode='w|gz')).

Only the user launches the exact verified packet in tmux. Stop on parity,
gradient, source, resource, state or infrastructure failure. Independently
audit all returned arrays/states/PNGs and inspect every planned sheet before
choosing one distinct finite training experiment. Keep the original U96 and
all failure records. No new model is qualified by preflight completion.
