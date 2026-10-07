# Separate original DGP reconstruction-decoder review

The user's selected direction is implemented as a separate in-memory copy of the
retained own-DGP reconstruction network. Exact initial raw and PNG output parity
passes on all 50 exposed photographic TRAIN cases. The original checkpoint,
complete encoder/FPN, normalization statistics and local app remain protected.
No local differentiation, optimizer update or new checkpoint occurs.

The copy enables these existing reconstruction weights for a future VM experiment:

| Original module | Parameters |
| --- | ---: |
| Each of `head1`, `head2`, `head3`, `head4` | 110,592 |
| `smooth` | 147,520 |
| `smooth2` | 18,464 |
| `final` | 867 |

Together these are 609,219 parameters in 14 tensors; the remaining 2,703,488 unique
encoder/FPN parameters stay frozen. The original implementation has shared FPN
parameter aliases. Deep-copying the complete model retains those aliases inside
the candidate while sharing no parameter or buffer with the original model.
Every original buffer stays in evaluation mode, including all five corrected
InstanceNorm layers and the backbone's stored BatchNorm statistics.

The candidate calls the unchanged `DGPSynthesizer.forward` directly. It uses the
four pyramid heads, nearest-neighbor fusion, original smoothing blocks and RGB
residual/tanh/clamp. Observed padding outside the supplied support is preserved.
No profile, source, person label, target or pretrained restorer output conditions
this forward. The historical app wrapper forces inference mode and is retained
for the reference only. Candidate differentiation has a separate guard requiring
the existing Linux NVIDIA L4/g2-standard-4 VM under `~/forensic-dgp`.

The CPU review takes 21.631 seconds: 50 original forwards, 50 parity forwards and
one additional partial-support forward. Every initial raw array and delivered
PNG exactly equals its fresh retained-model reference. Historical GPU-cache
compatibility is separately bounded at 1e-5; its measured maximum is
0.000002384185791. This does not claim exact byte parity with a different legacy
normalization/kernel context or establish canonical app qualification on the VM.
Actual local derivative attempts, nonfinite inputs, wrong canvases and empty
support are rejected before another neural call. Partial-support pixels are exact.
An independent source/layout/history readback verifies 251 bindings and all 14
parameter shapes. No candidate weight file is written.

Forward-only inspection finds pre-clamp saturation in 0–3.1993% of RGB component
values across these 50 cases. This records the original clamp behavior. It does
not measure a derivative, identify the cause of earlier failures or prove that
the original decoder can learn useful structure.

V25–V27 trained added heads while the original reconstruction weights remained
frozen. Their audited gains remain below the unchanged 1% early requirement:
0.0282235%, 0.0335636% and 0.2210886%. Their reviewed outputs lack convincing added
whole-face structure. This selected direction changes the learned path itself;
it is not evidence of native CCTV generalization or usefulness. The implementation
is our retained MobileNet/FPN reconstruction path. Do not adopt the old source
docstring's unsupported “optimal,” artifact-elimination or published-GAN claims.
Checkpoint lineage and inherited/pretrained parts remain disclosed.

The next released packet is **original-decoder gradient proof V1 R2**, a
zero-update L4 diagnostic, not another 800-update training attempt. It uses the
same 50 exposed TRAIN cases, unchanged seven-term loss functions and original
CPU-built fixed Gaussian filter. The baseline is freshly computed in the same
canonical normalization and batch as the candidate. Legacy cached evidence stays
separate. All 14 decoder tensors must receive finite nonzero improvement gradients;
all four initial preservation values and all their gradients must remain exactly
zero in every batch. Save all ten 7×609,219 gradient matrices and their sum.
No optimizer, epochs, new checkpoint or automatic follow-on is allowed.

Local packet verification caught two unissued-draft differences: GPU-side fixed
kernel construction and a shell substring replacement that changed 660s to 6120s.
The drafts are preserved and were never launched. R2 retains the original CPU
initializer and verifies whole-token shell deadlines against the original shell.
R2 passes Python 3.10 parsing, actual Windows host rejection, Bash syntax, complete
packet/source/readback and eight malformed scientific-return regressions.
Synthetic test arrays are not VM evidence.

R2 is 13,816 bytes/four regular files and uploads no data or weights. Existing VM
assets remain read-only. Require 2 GiB free, an idle L4, a 600s internal worker,
660s external supervisor plus 30s grace, 90s export/120s external plus 10s grace
and at most 20 GiB torch-allocated VRAM. Retain any failure. The manual runbook
uses Google Cloud SDK uploads, tmux and three separate PuTTY-compatible downloads.
An independent prospective safe importer/matrix/CPU forward/recognizer/cohort audit
is prepared. It never executes returned Python or reruns local derivatives.

Only an independently audited returned proof can justify defining a separate
finite training pilot. The existing early/final structural requirements, 17-group
pixel/SSIM/identity bounds, source nonregression and brightness limit stay intact.
All eyes, nose, mouth, outline and visible appearance require subsequent review
together; the five-patch metric alone cannot accept the face outline. No native
or reserved-final pixels enter this review. No ethnicity, Zamboanga performance
or hidden-identity claim follows. The original DGP remains the main app model;
pretrained restorers remain comparisons. Useful development outputs, all seven
automatic/assisted covering families and independent final review remain required.
The full goal remains active and incomplete.

[Initial review receipt](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_original_decoder_review_v1/review.json>) ·
[Independent readback](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_original_decoder_review_v1/independent_readback.json>) ·
[Five manual VM steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_ORIGINAL_DECODER_GRADIENT_V1_R2_VM.md>)
