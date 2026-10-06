# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users
School staff and thesis researchers manually reviewing already cropped frontal or mildly turned face images.

## Product Purpose
Restore visible blur/noise when needed and generate one plausible estimate of facial regions hidden by a covering. Users review and optionally correct the removal area before generation. The generated image is not proof of the person's hidden anatomy or identity.

## Positioning
One 256×256 estimate in the existing terminal interface. The main research route uses our retained trained DGP for visible restoration and a separately declared pretrained inpainting component for covering completion. Useful native CCTV restoration and full covering-family acceptance remain unverified; historical benchmarks remain available.

## Operating Context
Stationary forensic workstation and lab environment. The interface operates in an OLED dark mode to eliminate eye strain during extended observation, using high-density tabular telemetry and keyboard-accessible cockpit controls.

## Capabilities and Constraints
- Upload one PNG/JPEG face crop up to 10 MB; confirm readable frontal/mild-turn facial structure from the input. Preview automatic covering detection and correct it with paint/erase, undo, reset or a matching mask. Automatic detection is unqualified and needs review.
- Generate one estimate beside the original and binary removal mask. Clear glasses and non-obstructing hair remain outside the reviewed area. Request a less-covered image when the reviewed area hides almost all central facial evidence.
- Observed-support blur/noise Auto/Off/On override. Off preserves prepared 256 input bytes outside removal exactly; On applies retained DGP to visible support while retaining completed pixels. These signals do not automatically establish crop usability.
- Download the final PNG or ZIP containing native original/mask, 256 input/mask/result, metadata and optional raw DGP float array. Automatic-reviewed and assisted-reviewed masks are reported separately. Inference is local; training remains L4 VM-only.
- Preserve the established dark terminal layout and tokens, with keyboard mask painting and responsive verification at 375/768/1280 pixels. Frontal/mild turns are the initial scope; covering-family usefulness requires fixed-gallery review.

## Brand Commitments
- Project: ZCPO // FORENSICS
- System ID: DGP_MATRIX_V1
- Swiss Utilitarian precision aesthetic: high data density, no frivolous decorations or colored halos, restrained emerald phosphor indicator (`#10B981`) on OLED black (`#020617`).

## Evidence on Hand
- Confirmed behavior and Goal: `SYSTEM_WORKFLOW_AND_GOAL.md` and `PRACTICAL_OUTPUT_SCOPE.md`.
- Baseline comparisons: `PRACTICAL_FOOTPRINT_RESULTS.md`, `PRACTICAL_RESTORATION_RESULTS.md`, `PRACTICAL_FACE_RESTORATION_RESULTS.md`.
- Current runtime: `dgp_face_workflow_v3.py`, `face_workflow_web.py`, `templates/face_workflow.html`; historical engines and original DGP benchmark remain intact.
- Current evidence: `CCTV_DGP_APP_V3_INTEGRATION.md`, `outputs/dgp_app_v3_inference_audit/`, `outputs/dgp_app_v3_visual_review.json`, and git-ignored `scratch/dgp-v3-*.json` browser/download checks.

## Product Principles
1. **Evidence Integrity Over Flourish:** Every visual element serves investigative clarity and telemetry provenance.
2. **Neutral Dominance (10% Rule):** Deep OLED neutrals and slate architecture perform the visual work; emerald marks operational status and the removal preview.
3. **Cockpit Ergonomics:** Zero unwanted page scrolling on desktop viewports; self-contained panel scrolling with responsive single-column collapse on mobile.
4. **Accessible Operational Speed:** Every interactive trigger is keyboard-operable with explicit focus rings and accessible labels.

## Accessibility & Inclusion
Use semantic labels, visible focus, keyboard upload and brush controls, 44px button height, screen-reader status/error messages and the existing reduced-motion styles. Browser checks do not constitute a full accessibility certification.
