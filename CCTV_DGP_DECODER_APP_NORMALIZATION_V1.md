# Original decoder: actual app input normalization clarification

The separate original-decoder copy exactly matches its retained-model reference
under the **actual local app input encoder** for all 50 exposed photographic
TRAIN cases. A finite CPU audit takes 32.730 seconds: 100 original-model forwards
and 50 candidate forwards. The original checkpoint and complete model state stay
unchanged. No gradient, backward, optimizer update or VM action occurs.

| Source | Byte-to-float conversion |
| --- | --- |
| App `dgp_face_workflow_v3.canonical_tensor` | NumPy float32 division by float32 255 before device transfer |
| Released R2 diagnostic `dgp_face_restoration.as_tensor` | Torch float32 division by 255 after device transfer |

Every byte value from 0 through 255 produces exactly the same float32 tensor with
both encoders on the current CPU. All 50 case inputs, raw outputs and delivered
PNG bytes are identical. Every earlier legacy initial raw/PNG receipt is replayed.
The measured maximum difference is zero. **No CPU normalization defect or quality
improvement is found.** Different source policies do not establish different
numerical values on CUDA; CUDA equivalence has not been measured here.

An independent input-only readback checks the 256 byte values against a rational
division oracle, all 252 source bindings, the complete 50-case receipt set, original
state, timing, forward counts and historical raw/PNG hashes. It makes no neural or
gradient call. This is an evidence audit, not the required independent final human
review or a new full application-flow verification.

The released **original-decoder gradient proof V1 R2** remains unchanged. Its
earlier “canonical” wording refers to its declared legacy adapter and same-batch
fresh reference; it does not establish the actual app encoder on the L4. The
four-file packet, protocol, commands, prospective return auditor and old failures
remain immutable. The diagnostic can establish whether the 14 original decoder
tensors receive finite connected improvement gradients at initialization. It has
70 gradient queries, zero optimizer updates and no quality or identity acceptance.
The actual returned L4 proof is still pending.

Any later, distinct finite training protocol must use the actual app's NumPy
float32 conversion before transfer. It must prove initial candidate/reference
parity in that input and batch context before optimization. The existing structure,
preservation, timing and stop gates remain in force. Do not change R2 in place,
retrain an old failed recipe or treat this CPU equality as a CUDA result.

These are exposed photographic TRAIN inputs, not native CCTV or reserved final
evaluation inputs. No native or reserved pixels are opened. No source label is
interpreted as ethnicity, and no Zamboanga performance or hidden identity is
inferred. Useful native restoration, all visible facial features, the seven
automatic/assisted covering families and independent final review still need
qualification. The goal remains active and incomplete.

[Forward audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_decoder_app_normalization_v1/results.json>) ·
[Independent readback](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_decoder_app_normalization_v1/independent_readback.json>) ·
[Unchanged five manual R2 steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_ORIGINAL_DECODER_GRADIENT_V1_R2_VM.md>)
