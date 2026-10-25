# Implementation contract — spatial objectives V1

Fresh self-contained stage: `outputs/cctv_codeformer_spatial_objectives_v1_vm`.
VM root: `~/forensic-dgp/cctv_codeformer_spatial_objectives_v1_vm`.
Import package: `research.codeformer_spatial_objectives_v1` (source here copied into stage).
No Torch/NN/optimizer locally. Only worker.run after exact Linux/tmux/L4 guard.

## Entry points / ownership

- Root: `packet.py`, stage builder, entry CLI, supervisor/export/guide.
- Runtime author: `model.py`, `worker.py`. `worker.run(root, protocol, expected_sha)`.
- Independent author: `audit.py`: `objective_fixtures()`, `audit_static(root, protocol)`, `audit_return(root, protocol, result=None)`; return JSON-serializable dictionaries and raise on integrity failure. No Torch.
- All imports inert. Runtime may import Torch only through guarded model.activate_runtime.

## packet.py API

`require(condition,message)`, `sha256(path)`, `pin(root,path)` returns relative path/bytes/sha256.
`regular(path)` rejects symlinks/reparse/hardlinked/nonregular paths.
`read_json(path)`, `json_bytes(value)` strict finite JSON/duplicate rejection.
`verify(root, expected_sha)` reads exact protocol.json, verifies closed static namespace/files and contracts (no Torch), returns protocol.
`Writer(root, protocol)` with `json(relative,value)`, `array(relative,np_array)`, `npz(relative,dict_of_arrays)`, `png(relative,rgb8)`, `checkpoint(relative,torch_value,torch)` (exclusive atomic within allowed namespace), `append(relative,value)`; each returns pin except append. No overwrite; byte/file/per-file bounds; 4GiB disk reserve before physical writes. Writer owns no NN.
`Ledger(writer)` with `reserve(kind, **context) -> token`, `returned(token, **context)`; events durably appended before/after actual calls; `.counts`, `.returns`, `.pending` dictionaries. Caps: prefix10, adapter_train576, adapter_stats240, full_forward172, adapter_full688, backward144, optimizer48. Initial stats use arithmetic identity, not adapter calls. Exact starts/returns of full-site adapters recorded via caller parent reservation grouping; no hidden calls.
`Resources(root, writer)` with `.check(context)` uses Linux self VmRSS and resource.ru_maxrss, CUDA allocated/reserved if torch attached via `.torch=torch`; 10GiB RSS/peak, 18GiB CUDA reserved, 4GiB disk continuing reserve, 2700s worker cap. Whole-worker outer supervisor adds wall enforcement. `.stage(name, seconds)` context manager alarm; parent cap checked nested. `.project(name, elapsed, completed, total, cap)` raises before excess projected work; no extra calls/updates to profile. All return resource rows source-bound, no guarantee from Windows anomaly.

## Sealed protocol schema (root constructing)

`format`, `vm_name`, `design_sha256`, `files` maps every static relative pathname to `{bytes,sha256}` excluding protocol itself; no runtime cache/output inside static namespace. All data pins point to content-addressed `assets/` files. `r2_asset` = weights/codeformer_r2.pth.
`train_rows`: exact10 readable TRAIN rows `{id,reference_id,profile,input,input_RGB8_sha256,target,target_RGB8_sha256,...}`.
`clean_features`: ref -> `{features:{site:pin},scales:{site:float},...}` for6refs/24arrays.
`sampler`: exact72 rows; use `.id` field (root normalizes design sampler). Updates1..24, slots1..3.
`guards`:24 rows `{id,reference_id,profile,role,split,input,input_RGB8_sha256,target,target_RGB8_sha256,metrics,outputs,upstream,behavior_distance?,behavior_teacher?}`. `outputs` is cached identity R2 RAW/PNG; `behavior_teacher` clear/dark has `raw`,`png`,`kind`. Guard0 reuses these values after static audit and parity.
`parity_ids`: exact4 existing guard IDs (design unchanged).
`strong`:4 rows same schema with cached initial R2 outputs/upstream, profile, target; no behavior teacher.
`initial_adapter_sha256`, `base515_sha256`, `max_calls`, `limits`, `allowed_outputs`, `required_static_summaries`, `provenance` pins / explicit new strong4 training admission.
Root will notify when data_spec.json/stage present; runtime can rely on these keys.

## Runtime outputs (all relative stage root)

Prefix `outputs/spatial_objectives_v1/` (abbrev O):
- O/started.json, preflight.json, initial.json, result.json, failure.json; events.jsonl, resources.jsonl.
- O/parity/{guard_id}.npy,.png,.json for4 declared witnesses only.
- O/shared_features0/{train_id}.npz,.json for10 initial identity arithmetic readouts.
- O/checkpoints/{U|S}/u{0000|0006|0012|0024}.pt (adapter24/optimizer/RNG/counters only; never base515 copied).
- O/stats/{U|S}/u{0006|0012|0024}/{train_id}.npz,.json (10 each).
- O/guards/{U|S}/u{0006|0012|0024}/{guard_id}.npy,.png,.json and bank.json,gates.json.
- O/strong/{U|S}/u{0006|0012|0024}/{strong_id}.npy,.png,.json and bank.json.
- O/branches/{U|S}.json terminal even on branch stop, initial RNG/state reset receipts and actual optimizer committed count.
- Cache `cache/observed/{train_id}/encoder{32|64|128|256}.npy` VM-only, 450MiB + headers, excluded from return. Root Writer supports declared cache names with separate512MiB cap.
- Root supervisor/export files outside O: `launch.json`, `supervisor.json`, `trainer.log`, `export.json`; returned archive includes O plus these and the complete sealed static packet EXCEPT bulky weight/clean feature assets? Root export will use explicit return subset + protocol/source/pins; no implicit duplication. Final exact return subset declared before sealing.

## Feature statistics compression

Do NOT save full-resolution moment maps for all240site rows: they exceed return cap. For each case NPZ, keys `{site}_means` FP64(3,C) ordered observed,wanted=t-x,correction=z-x; `{site}_moments` FP64(12,C), each spatial mean, ordered:
`wanted_abs,final_abs,wanted_AC_abs,final_AC_abs,wanted_sq,correction_sq,final_sq,wanted_spatial_sq,correction_spatial_sq,spatial_dot,observed_sq,target_sq`.
Here final=correction-wanted; AC means remove each channel mean. `{site}_pooled` FP64(3,C,8,8), same observed,wanted,correction order, uniform exact spatial8 pooling. Compute full channel moments in finite chunks; normalization is unchanged cached total-clean RMS. JSON stores scales, scalar reductions/decompositions, source/case/site/state bindings. Auditor recomputes scalars from channel moments/means and checks pooled/DC consistency, but does NOT claim full-activation replay or independently observed prefix inference. Clean target means/RMS may be checked against physically retained arrays.
Store actual per-training FP32 scalar losses, branch/checkpoint links, grad norms (expected zeros allowed), lr/Adam counts in events. A separate local Torch content auditor after real return must inspect checkpoint tensors/Adam/RNG; pure array auditor explicitly leaves this unproven.

## Semantics

U and S are controlled learning screens, not accepted models. Same exact initial adapter digest/seed and72 sampler rows; each starts new Adam/RNG. At6/12/24 save state/stats, run unchanged24 guards, stop failed branch BEFORE4 strong probes/more updates. Still try other arm only its original finite schedule after a quality-only guard stop; infrastructure/resource/NaN/parity errors stop whole worker. Report all branches/failures, no automatic promotion or selecting from DEV. No extra full forward for gradient witness; use first reserved training microbatch for finite gradient/expected-zero reporting.
