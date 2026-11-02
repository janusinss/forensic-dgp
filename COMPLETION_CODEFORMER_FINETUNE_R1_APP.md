# Test our trained CodeFormer completion R1

25 October 2026. Experimental manual-mask user-review app; full-family qualification is pending.

- **Trained R1:** http://127.0.0.1:64529/
- **Original CodeFormer comparison:** http://127.0.0.1:51239/

Both are local Windows CPU pages. Their ports are temporary; the runtime files in scratch record the current URLs. The old baseline runtime record was preserved before reopening its previously stopped service.

## Use the same face and mask

1. Upload one cropped frontal or mildly turned face to the trained page.
2. Paint only the intended covering, or import an opaque black/white 256Ã—256 mask. Orange marks allowed generation; clear glasses, ordinary hair and visible appearance stay outside support.
3. Review the area and confirm sufficient visible facial structure. Any image/mask change resets both confirmations. Empty masks return the prepared input unchanged; insufficient visible support asks for a clearer or less-covered image.
4. Generate one plausible estimate. Review the mask edges and the eyes, nose, mouth and jaw inside the generated region.
5. Download the input and mask PNGs. Use those same files on the original comparison page, so both models see identical prepared pixels and support. Compare the two result PNGs. Each generation remains separately retained.
6. Download the original/mask/result evidence bundle when needed.

This page uses our fine-tuned completion model. It does not run restoration or automatic covering detection. It does not establish the actual hidden face. The 6.19% result is a reduction in paired TRAIN-monitor pixel error inside synthetic masks; six individual monitor cases worsened. Scarves, obstructing hair, grayscale appearance and some glasses completions still need work. The original baseline remains selected while this candidate is reviewed.

## Model file and evidence

`outputs/completion_codeformer_finetune_v1_r1_candidate/codeformer_completion_finetune_r1.pth`

SHA256:7ae2486739de0f3efa99cf721a0fcb755f8f56cf69e295412f74190d2fda37e0.
It contains inference weights only. Keep the separate 701,952,165-byte terminal state for research recovery; it includes both Adam/RNG/schedule states and must not be replaced by this deployment file.

[Training/return findings](COMPLETION_CODEFORMER_FINETUNE_V1_R1_RESULTS.md).
CPU smoke request 12.5 seconds, one forward; empty bypass 0.48 seconds. Browser checks cover the editor, confirmation, downloads, refusal and three responsive widths. These are prototype/reference checks, not a fresh-device installer or certified minimum hardware. User acceptance and independent human final review remain pending.

## Relaunch after closing this test

From C:\xampp\htdocs\YEAR 4\Testing_FACE_MASK_COMPLETION in PowerShell:

```powershell
& 'C:\xampp\htdocs\YEAR 4\Testing\venv\Scripts\python.exe' -B completion_codeformer_finetune_r1_web.py --admission-sha 7370ff282df6570823deb4e45a3afb4fca3c462356f43796f17fe73f2288eb28 --worker-sha acf3e2a65099dc6117ff4f5e2079e4fcf15aeba0f000772f4f663a12304eccc4 --guard-sha 1277b0fde51c35337a0391d4089b193aa837f21c7e00ecc9ee4ed7ecd5f687c0 --candidate-manifest-sha e8bc39c321d490ee335b683ea6947bc24df47c92e16db30a1412d6fc5380ea77
```

Read the new URL printed in the terminal. Reuse an existing live instance rather than starting duplicate services. This prototype uses the retained original Testing Python environment and this worktree's SHA-bound candidate/source assets. Fresh Windows installation and selective IMPLEMENTATION integration remain later gates after useful completion and user acceptance.
