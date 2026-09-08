# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users
Forensic digital investigators, biometric verification examiners, computer vision researchers, and cybersecurity analysts inspecting low-resolution surveillance footage and CCTV facial crops.

## Product Purpose
The Forensic DGP Terminal restores, enhances, and synthesizes facial morphological priors from degraded, motion-blurred, and low-resolution CCTV evidence, generating probabilistic Top-K reconstructions ranked by Facial Alignment Network (FAN) loss metrics.

## Positioning
Dual-pipeline generative face restoration combining direct native sharpness upscaling with a sub-32x32 super-resolution benchmark, utilizing deep learning (MobileNetV2 + Feature Pyramid Network) with facial landmark geometry preservation.

## Operating Context
Stationary forensic workstation and lab environment. The interface operates in an OLED dark mode to eliminate eye strain during extended observation, using high-density tabular telemetry and keyboard-accessible cockpit controls.

## Capabilities and Constraints
- Dual reconstruction modes: Direct Restoration (native crop) and Sub-32x32 Super-Resolution Simulation.
- Canonical 256x256 normalized synthesis space.
- FAN loss metric calculation and identity fit scoring.
- High-resolution PNG evidence export with rank-specific provenance naming.
- Client-side drag-and-drop ingestion with MIME verification (PNG/JPG) and 10MB memory safety bounds.
- Full keyboard operability and zero horizontal viewport overflow across all device viewports.

## Brand Commitments
- Project: ZCPO // FORENSICS
- System ID: DGP_MATRIX_V1
- Swiss Utilitarian precision aesthetic: high data density, no frivolous decorations or colored halos, restrained emerald phosphor indicator (`#10B981`) on OLED black (`#020617`).

## Evidence on Hand
- Test data: `test_images/test_cctv_blurred_1.png` and `test_images/test_cctv_blurred_2.png`.
- Pre-trained models and checkpoints: `checkpoints/dgp_improved_epoch_10.pth` and `weights/mapped_deblurgan.pth`.
- Automated Playwright verification suite: `scratch/verify_ui.js`.

## Product Principles
1. **Evidence Integrity Over Flourish:** Every visual element serves investigative clarity and telemetry provenance.
2. **Neutral Dominance (10% Rule):** Deep OLED neutrals and slate architecture perform the visual work; emerald color is reserved for operational status and affirmative forensic ranking.
3. **Cockpit Ergonomics:** Zero unwanted page scrolling on desktop viewports; self-contained panel scrolling with responsive single-column collapse on mobile.
4. **Accessible Operational Speed:** Every interactive trigger is keyboard-operable with explicit focus rings and accessible labels.

## Accessibility & Inclusion
Target standard: WCAG 2.1 AA. Contrast exceeds 15:1 for primary text, minimum touch target dimensions exceed 44x44px, all decorative iconography is hidden from screen readers (`aria-hidden="true"`), and `prefers-reduced-motion` is strictly respected.
