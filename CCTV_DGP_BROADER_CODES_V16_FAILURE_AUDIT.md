# V16 cache timing failure: independent return audit, 5 October 2026

The9,069,418-byte failure archive is verified and safely imported. Export receipt
`complete:true` means the failure export completed. The trainer failed before
backward preflight or any optimizer update; this is not a learned restoration
result. The original execution archive, protocol, partial checkpoint and failure
remain unchanged. No checkpoint is selected or adopted.

Archive SHA256: `b32e3b39f400c800ba364ff08f44d9a7a4c97da5cb9d164fc3f221041c44a39d`.
Protocol SHA256: `4331c28c97a659846eab76ee0dee9150811a00da3c6139d8d24b7905173a6681`.
Local returned directory: `outputs/cctv_dgp_broader_codes_failure_return_v16/`.
VM original directory: `~/forensic-dgp/cctv_dgp_broader_codes_vm_v16/`.

## Verified evidence

| Evidence | Independent finding |
| --- | --- |
| Export/bootstrap | Archive hash/size and checksum match; original23 pinned execution members verified. Original path error and corrected preflight verified; recovery source matches uploaded SHA256489e85d001b3cca74fdeca3c9fbd50edc2e5bf1fa7aa30c2b45db36bce99bb17 |
| Learning/supervisor |0 backward calls,0 optimizer updates, no update trace/backward receipt/success result; child exit1,39.27954218199989s supervisor duration, no resume |
| Teacher labels |20 arrays match the20 training-prefix references; all int64,256 tokens, range0–1023; no validation teacher arrays |
| Cache timing |20 references,22.75441105899995s elapsed;1,874.5294464701833s projected,900s cap; arithmetic matches frozen source |
| Partial checkpoint |9,695,145 bytes, SHA256f351693258c11fe3f1c629aa009288cadd1e26231fc49134f0bdfa95a1942d2e; retained as zero-update failure evidence |

Core failure receipt: `outputs/cctv_dgp_broader_codes_failure_return_v16/local_failure_import.json`.
Additional timing/provenance receipt:
`outputs/cctv_dgp_broader_codes_failure_return_v16/local_cache_failure_audit.json`.
The additional audit took0.072s. Both audits perform no neural inference,
backward or optimization. The cache remains on the VM and was excluded from the
failure export; these audits do not replay its arrays or frozen neural forwards.
The partial checkpoint is fingerprinted without model loading or a usefulness claim.

## Demonstrated processing limitation

The cache timer starts before model loading and state fingerprints. At reference20,
the old expression is elapsed×81.0625+30s. It multiplies fixed setup together with
per-reference work and uses training-batch cost to approximate different validation
batching. Its15-minute gate correctly rejected its own31.24-minute estimate.
Startup and steady cost were not recorded separately, so this evidence does not
establish whether the actual full cache would fit900s.

## Separate correction

V16 r2 keeps all learning parameters, data/splits, optimizer schedule, previews,
quality guards, float32 cache contract and finite caps. It measures initialization
once and samples10 train/5 validation references per source. Each source/role's
first reference is cached once as warmup and excluded from its steady rate; its
actual elapsed cost remains counted once. It projects remaining references using
four measured source/role rates with the same1.25 safety factor and30s reserve.
It still stops for a projection above900s, actual cache deadline, fit deadline,
25-update timing, epoch4 fitting, nonfinite values, parity/state failures or VRAM.

The cache processing order changes; all885 references and4,425 cases are cached
once, and teacher labels remain restricted to781 training roles. The original
3,128-update training schedule and head source are byte-identical. R2 starts a
seeded reset head in a new directory; it does not load the failed partial head or
reuse its cache. The separate Path-typed bootstrap records visible child errors.
No local training or automatic VM execution occurs.

Exact verified files and manual commands: `CCTV_DGP_BROADER_CODES_V16_R2_VM.md`.
The main DGP-led application, useful development/covering-family outputs and final
independent review remain incomplete. Synthetic photographic camera proxies and
their paired metrics do not establish native CCTV or Zamboanga performance.
