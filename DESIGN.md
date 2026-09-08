---
title: "ZCPO // Forensic DGP Terminal Design System"
description: "Utilitarian / Swiss Precision High-Density Forensic Stylesheet and Layout for Digital Evidence & Reconstruction"
last_updated: "2026-09-08"
version: "1.0.0"
---

# ZCPO // Forensic DGP Terminal Design System

## Design System

The Forensic DGP Terminal is a high-density, mission-critical interface engineered for deepfake detection, forensic face super-resolution, and morphological reconstruction from degraded CCTV surveillance footage. 

The design philosophy fuses **Swiss International Typographic Style** with **Utilitarian Terminal Electronics**:
- **OLED Pure Blacks and Muted Grays:** Dark mode baseline designed for forensic laboratory environments, eliminating eye strain during long inspection shifts.
- **Rigid 1px Spatial Grid:** Every section, header, and module is demarcated by explicit 1px boundary lines (`--border-color`) rather than soft drop shadows.
- **Calibrated Data Density:** Information-packed layouts featuring monospace telemetry, tabular numerals, status badges, and explicit metadata metrics (`FAN_LOSS`, `IDENTITY_FIT`, `RESOLUTION`).
- **Forensic Reticle Accents:** Functional alignment brackets, crosshairs, and corner indicators representing computer vision bounding boxes and facial landmark crops.
- **High-Contrast Telemetry:** Utilitarian amber, cyan, and emerald indicators providing instant situational awareness without visual noise.

### Normative Tokens & CSS Custom Properties

```css
:root {
    /* Surface & Background Hierarchy */
    --bg-base: #070709;
    --bg-surface: #0f1013;
    --bg-elevated: #15161b;
    --bg-hover: #1c1e24;

    /* Structural Borders */
    --border-color: #23252d;
    --border-subtle: #191a21;
    --border-highlight: #383a45;

    /* Typography & Contrast */
    --text-primary: #f4f4f6;
    --text-secondary: #9093a2;
    --text-muted: #565969;

    /* Forensic Accents */
    --accent: #f59e0b; /* Forensic Amber Accent */
    --accent-glow: rgba(245, 158, 11, 0.18);
    --accent-hover: #d97706;

    /* Semantic Status Indicators */
    --status-green: #10b981;
    --status-red: #ef4444;

    /* Font Stacks */
    --font-sans: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --font-mono: 'IBM Plex Mono', monospace;
}
```

---

## Typography

The interface employs a dual-font pairing: **IBM Plex Sans** for interface labels, instructions, and structural headings, and **IBM Plex Mono** for coordinates, model specifications, telemetry values, and file metadata.

### Font Families
- **Primary Interface (Sans):** `'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`
- **Telemetry & Data (Mono):** `'IBM Plex Mono', monospace`

### Typographic Hierarchy & Scale

| Role | Font Family | Size | Weight | Line Height | Case / Tracking | Example Usage |
|---|---|---|---|---|---|---|
| **Terminal Title (`h1`)** | Sans | `1.05rem` (16.8px) | `600` | 1.2 | Uppercase, `0.05em` | `ZCPO // FORENSICS` |
| **Panel Header (`h2`)** | Sans | `0.85rem` (13.6px) | `600` | 1.2 | Uppercase, `0.05em` | `01 // TARGET_EXTRACTION` |
| **Telemetry Label (`.mono-label`)** | Mono | `0.72rem` (11.5px) | `500` / `600` | 1.3 | Uppercase, `0.05em`, Tabular | `ENGINE_STATUS: STANDBY` |
| **Metric Value (`.data-value`)** | Mono | `1.15rem` (18.4px) | `500` / `600` | 1.1 | Tabular Numerals | `0.0421`, `256x256` |
| **Body & Subtext** | Sans | `0.82rem` (13.1px) | `400` | 1.4 | Sentence Case | Empty state description |
| **Mode Subtext** | Sans / Mono | `0.68rem` (10.9px) | `400` | 1.3 | Uppercase, `0.04em` | `NATIVE SCALE // MAXIMUM DETAIL` |
| **Specification Matrix** | Mono | `0.70rem` (11.2px) | `400` / `500` | 1.2 | Uppercase, Tabular | `MODEL: MOBILENETV2 + FPN` |
| **Pill / Badge** | Mono | `0.65rem` - `0.75rem` | `600` | 1.0 | Uppercase, `0.05em` | `RANK 01 // BEST_FIT` |

---

## Color System

The baseline color palette enforces low luminescence with high contrast ratios, calibrated to prevent eye fatigue while highlighting forensic discrepancies.

### Color Palette Roles

| Token | Hex / Value | Role & Usage | Contrast vs `#070709` |
|---|---|---|---|
| `--bg-base` | `#070709` | Deepest root layer, terminal background, header, footer | Baseline |
| `--bg-surface` | `#0f1013` | Primary panel surface, layout columns | 1.15:1 |
| `--bg-elevated` | `#15161b` | Interactive tiles, buttons, card background | 1.35:1 |
| `--bg-hover` | `#1c1e24` | Hover states, active dragover feedback | 1.55:1 |
| `--border-color` | `#23252d` | Primary structural division, panel borders, card borders | 1.8:1 |
| `--border-subtle` | `#191a21` | Inner boundaries, sub-dividers | 1.4:1 |
| `--border-highlight` | `#383a45` | Hovered card borders, active focus accents | 2.6:1 |
| `--text-primary` | `#f4f4f6` | High-contrast body text, headings, metric values | 17.5:1 (AAA) |
| `--text-secondary` | `#9093a2` | Monospace labels, inactive icons, auxiliary metadata | 5.8:1 (AA) |
| `--text-muted` | `#565969` | Dimmed status descriptions, corner reticles, disabled text | 3.2:1 |
| `--accent` | `#f59e0b` | Forensic amber primary CTA, selection glow, rank 1 pill | 8.2:1 (AAA) |
| `--accent-hover` | `#d97706` | Primary button hover state | 6.4:1 (AA) |
| `--accent-glow` | `rgba(245, 158, 11, 0.18)` | Active button glow, reticle drop shadow | - |
| `--status-green` | `#10b981` | Engine online status, verified verification badge | 7.6:1 (AAA) |
| `--status-red` | `#ef4444` | System error notifications, critical validation alerts | 4.8:1 (AA) |

### Candidate Differentiator Colors
- **Rank 01 (Best Fit):** `#f59e0b` (Amber - Global Prior)
- **Rank 02 (Edge Focus):** `#38bdf8` (Cyan - High-Frequency Texture)
- **Rank 03 (Natural Tone):** `#c084fc` (Purple - Bilinear Smoothing)

---

## Elevation & Surfaces

In accordance with Swiss utilitarian principles, elevation is communicated through **tonal layering and 1px borders** rather than diffuse shadows.

### Surface Elevation Levels
- **Level 0 (Base / Recessed):** `#070709` (`--bg-base`) — Used for drop zone background, terminal header, terminal footer, and outer margins.
- **Level 1 (Surface):** `#0f1013` (`--bg-surface`) — Applied to the primary left/right workbench panels.
- **Level 2 (Elevated):** `#15161b` (`--bg-elevated`) — Used for mode toggle buttons, spec cards, and results containers.
- **Level 3 (Interactive / Hover):** `#1c1e24` (`--bg-hover`) — Activated upon pointer hover or selection.

### Borders & Dividing Lines
- **Primary Grid Lines:** `1px solid var(--border-color)` (`#23252d`) separating all major structural quadrants.
- **Reticle Markings:** 8×8px L-shaped precision crosshairs positioned in the four corners of extraction zones.
- **Focus Rings:** `outline: 2px solid var(--accent); outline-offset: 1px;` ensuring unambiguous keyboard navigation.

### Shadows & Lighting
- **Result Card Hover:** `box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);`
- **Execute Button Glow:** `box-shadow: 0 0 15px var(--accent-glow);`

---

## Component Styles

### Component Breakdown Table

| Component | Description | Default State | Hover/Active State | Accessibility Notes |
|---|---|---|---|---|
| **Header Logo Block** | Branding quadrant with crosshair icon, system title, and SYS_ID | `#070709` bg, 1px right border, `#f4f4f6` title | Static branding | Focusable link/home semantics; descriptive icon |
| **Status Indicator** | Live operational heartbeat indicator | `#10b981` green dot with pulsing radial wave | Continuous 2s pulse animation | `aria-label="Engine Status: Standby"`, respects reduced-motion |
| **Target Drop Zone** | Ingestion container for CCTV face crops with reticle bounds | `#070709` bg, 1px `#23252d` border, centered icon & prompt | Dragover: `#1c1e24` bg, `#f59e0b` border | Keyboard focusable (`Enter`/`Space`), hidden file input with matching label |
| **Forensic Reticles** | Corner alignment brackets indicating spatial detection boundaries | 8×8px L-shapes in `#565969` | Highlighted amber (`#f59e0b`) when active | Purely decorative; tagged `aria-hidden="true"` |
| **Mode Toggle Button** | Segmented button selecting reconstruction pipeline | `#15161b` bg, `#9093a2` text, 1px border | Active: `#f59e0b` accent border, `#f4f4f6` text; Hover: `#1c1e24` | `role="radio"` / `aria-checked`, keyboard navigable |
| **Initiate Button** | Primary execution trigger to synthesize priors | Amber `#f59e0b` bg, black text, disabled: 0.5 opacity | Hover: `#d97706` bg; Active: `#b45309` bg with glow | Disabled state prevents premature trigger; clear `:focus-visible` ring |
| **Loading Spinner** | Morphological prior processing state | Rotating dual-border 24px spinner with blinking mono text | 0.8s continuous 360° spin | `role="status"`, `aria-live="polite"`, disabled in reduced-motion |
| **Result Card** | Diagnostic container for reconstructed candidate | `#070709` base bg, 1px `#23252d` border | Hover: `#383a45` border, subtle shadow lift | Descriptive `alt` text on images; semantic card structure |
| **Rank Pill** | Prioritized ranking indicator (01/02/03) | Monospace text in amber, cyan, or purple | Static metadata pill | Tabular numerals, contrast ratio exceeds 5:1 |
| **Mode Pill** | Reconstruction mode badge on result | `#15161b` bg, `#9093a2` text, 1px border | Static metadata pill | Monospace tag for provenance verification |
| **Export Button** | Download high-res reconstruction PNG | `#0f1013` bg, 1px top border, `#9093a2` text | Hover: `#1c1e24` bg, `#f4f4f6` text, amber top border | Clear download attribute, `aria-label="Export Rank X Image as PNG"` |
| **Empty State Crosshair** | Waiting acquisition placeholder with model matrix | Centered crosshair icon with `#565969` muted text | Static layout | Informative text for screen readers |
| **Terminal Footer** | Institutional telemetry, copyright, memory, latency info | 34px height, `#070709` bg, 1px top border | Static status bar | High contrast tabular monospace data |

---

## Layout & Spacing

The terminal is structured as a full-viewport application cockpit (`100vh`) with zero window scrolling on desktop, utilizing internal panel scroll containers.

### Master Layout Grid
- **Container Max-Width:** `1600px` (centered horizontally with left/right 1px borders)
- **Header Height:** `64px` fixed
- **Footer Height:** `34px` fixed
- **Main Body:** `calc(100vh - 98px)`, 2-column flexbox:
  - **Left Column (`input-panel`):** `420px` to `460px` fixed-width operational configuration sidebar.
  - **Right Column (`results-panel`):** Flexible space `flex: 1` displaying Top-3 candidate cards.

### Spacing Scale
- `4px` (`0.25rem`): Micro-padding, inline pill padding, spec cell gaps.
- `8px` (`0.5rem`): Icon gaps, internal button element spacing.
- `12px` (`0.75rem`): Form controls, mode button padding.
- `16px` (`1.0rem`): Panel headers, result data matrix padding.
- `24px` (`1.5rem`): Major panel inner margins, drop zone internal padding.
- `32px` (`2.0rem`): Section gaps and horizontal gutters.

---

## Motion & Animation

Motion is reserved strictly for functional feedback, processing telemetry, and interactive hover confirmation.

### Keyframe Animations
- **Pulse Animation (`@keyframes pulse`):**
  - Used for the engine status indicator (`.status-indicator.online`).
  - 2.0s duration, `ease-in-out`, infinite loop.
- **Spinner Animation (`@keyframes spin`):**
  - Used for `.spinner` during neural reconstruction computation.
  - 0.8s duration, `linear`, infinite 360° rotation.
- **Blink Animation (`@keyframes blink`):**
  - Used for `.loading-block span` to communicate live background processing.
  - 1.2s duration, opacity oscillates between `0.3` and `1.0`.

### Interaction Transitions
- Buttons, cards, and interactive pills: `transition: all 0.2s ease` or `transition: border-color 0.2s, box-shadow 0.2s`.
- Image hover: Smooth transition without layout shift.

---

## Accessibility & Compatibility

### WCAG Target
- Target standard: **WCAG 2.1 Level AA** compliance across all views and interactive controls.

### Keyboard Navigation
- Complete keyboard operability across all workflows:
  - Drop zone accepts `Enter` and `Space` to open native file explorer.
  - Mode selector buttons are focusable and switchable via arrow keys and `Tab`.
  - Reconstruction action and Export download links support standard keyboard activation.
  - Visible focus rings with high-visibility contrast (`outline: 2px solid var(--accent); outline-offset: 1px;`).

### Color Contrast Ratios
- Primary text on dark backgrounds: **17.5:1** (Exceeds AAA requirement).
- Secondary metadata and labels: **5.8:1** (Exceeds AA requirement of 4.5:1).
- Accent buttons (`#000000` text on `#f59e0b` amber): **9.6:1** (Exceeds AAA requirement).
- Status indicator greens and reds maintain > 4.5:1 contrast against surface backgrounds.

### Device Support
- Modern evergreen desktop and workstation browsers:
  - Chrome / Chromium 110+
  - Mozilla Firefox 110+
  - Safari 16+
  - Microsoft Edge 110+
- Calibrated for high-resolution displays (1080p, 1440p, 4K) with pixel-accurate 1:1 image rendering.

### Responsive Breakpoints
- **Desktop Large (≥ 1440px):** Full dual-panel layout, 3-column horizontal candidate cards.
- **Desktop Standard (1024px – 1439px):** Dual-panel layout, 2-column or stacked candidate cards.
- **Tablet (768px – 1023px):** Stacked vertical columns with natural vertical scrolling.
- **Mobile (< 768px):** Single-column vertical layout with touch targets expanded to minimum 44×44px.
- **Reduced Motion:** Fully honors `@media (prefers-reduced-motion: reduce)` by disabling infinite spin, blink, and pulse loops.
