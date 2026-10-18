# Unsealed neural-module interface

`execute(root, p, original_path, stopped_path, callbacks)` is manual Linux/L4 only.
Importing the module or running its pure CPU fixtures constructs no model.
The module is preparation only; no protocol or execution packet is sealed.

The literal protocol keys are:

- `seed: 20261015`; `call_caps: {codeformer_forwards:170, fusion_block_evaluations:960, hq_teacher_forwards:0, behavior_teacher_forwards:0, lpips_forwards:0, gradient_queries:0, backward_calls:0, optimizer_attempts:0, optimizer_updates:0}`.
- `gate: {q10:0.21007671428419458, q90:0.5546973290483728, beta:0.1, epsilon:1e-6, gate_floor:0.25, gate_ceiling:1.0, central_support:[32,224]}`.
- `cases`: exactly32 selected rows, in frozen update/slot order. Each has safe `id`, `update`, `slot`, `reference_id`, `input_path`, `input_sha256`, `target512_path`, `target512_sha256`, `target256_path`, `target256_sha256`; include `source_path/source_sha256` and exact `input256_rgb8_sha256/input256_float32_sha256` when assembling the packet. Targets/source are only file-hashed here, never decoded or passed to the network/gate.
- `calibration_inputs`: exactly200 rows containing `update`, `slot`, installed `input_path`, PNG `input_sha256`; optional exact pixel hashes are checked when supplied. They must cover updates1..50 × slots1..4. `canary_ids` contains exactly the frozen two selected case IDs.
- `assets.original` and `assets.stopped`: exact `sha256` and integer `bytes`, plus caller's parent bindings; `runtime_expected` has exactly `torch,numpy,CUDA,GPU,capability` from retained actual FP32 runtime. The caller verifies source bytes, full selected-cohort algorithm, parent directories/protocols and all asset namespaces before any heavy import.

`callbacks` may be a dict or object exposing all of:

| Callback | Contract |
|---|---|
|`guard(root,p)`|Before heavy imports, verify manual exact-root installation, immutable sources/protocol/dependencies/parents/assets, no prior run, fresh external workload checks and complete input/cohort bindings. Neural independently checks Linux/home/tmux/L4 as well.|
|`require(condition,message)`|Raise on false; never log-and-continue.|
|`clock(stage)`|Return finite monotonic elapsed seconds; check whole/stage limits, disk reserve, current/peak CUDA/host accounting and output byte/file limits. Stages are `initialization`, then `measurement`; never reset a stage clock.|
|`reserve(kind,context)`|Reserve before each actual full network or fusion-block call, trace context/sequence and refuse cap overrun. Kinds use the exact call-cap keys above.|
|`write_json(relative,payload)`|Exclusive, regular-file, finite JSON write with fsync and metadata/file/payload caps.|
|`write_array(relative,array)`|Exclusive allow_pickle=False NPY for contiguous finite float32/int64 arrays; strict per-file/payload/file-count cap; reserve room for temporary write. Largest output is site64 feature:4,194,304 data bytes plus NPY header.|
|`write_png(relative,RGB8)`|Exclusive exact RGB256 PNG (no enhancement), strict finite PNG file cap.|

Every write returns a JSON-serializable `{path,sha256,bytes}` receipt. Passed
paths are portable relative paths **under the new output directory**; they do
not include `outputs/`. The writer owns exclusive mkdir, temporary accounting,
atomic finalization, no links/traversal, result inventory and failure retention.
The caller owns process-group supervision/terminal checks/robust failed export.

The neural result `run_outputs.json` is not the authoritative overall run or
quality receipt. On successful measurements, return flags are `complete`,
`state_before_after_equal`, `parameter_grad_fields_all_none`,
`content_preservation_verified`, `repeat_zero_identity_canaries_pass`, with exact
counts and `optimizer_updates=gradient_queries=0`, `model_qualified=false`.
On failure the caller preserves its exception, partial reservations/artifacts
and terminal status. A best-effort neural failure record never overrides that.

Native artifacts are HWC RGB float32: `native512.npy` remains unbounded;
`raw256.npy` uses unchanged clamp-before-bilinear delivery, `native.png` uses
floor(raw×255); actual hard-selected tokens are int64 `(256,)`. All170 paths,
including repeats/zero/identity canaries, are written. The four canary bounded
paths retain F0/FS/used-delta at32/64 only (24 feature files total). Per-path
`fusion_stats.json` aggregates site records so reserved inner calls do not
multiply uncontrolled output files. Other content is captured as shape/dtype/
byte hashes and finite statistics; it is not a full feature-vector replay claim.

`bounded_numpy_reference` exists only as a model-free specification for returned
canary tensors. It cannot accept an RGB face or generate a model output and is
never called as a restoration fallback. NumPy versus Torch reduction/tanh
rounding must be prospectively bounded by the independent audit, not confused
with a loosened quality requirement. Source/tests need independent readback
before any packet or VM instructions are published.
