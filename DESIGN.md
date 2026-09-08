---
name: "ZCPO // Forensic DGP Terminal"
description: "Utilitarian Swiss precision high-density forensic terminal for CCTV face restoration"
colors:
  primary: "#0F172A"
  on-primary: "#FFFFFF"
  secondary: "#1E293B"
  on-secondary: "#FFFFFF"
  accent: "#10B981"
  accent-hover: "#059669"
  accent-dim: "rgba(16, 185, 129, 0.08)"
  on-accent: "#020617"
  background: "#020617"
  foreground: "#F8FAFC"
  card: "#0A0E1A"
  card-foreground: "#F8FAFC"
  muted: "#111728"
  muted-foreground: "#8B9BB4"
  slate: "#94A3B8"
  border: "#1E293B"
  border-subtle: "#141B28"
  border-highlight: "#334155"
  destructive: "#EF4444"
  ring: "#10B981"
  white: "#FFFFFF"
typography:
  display:
    fontFamily: "'Fira Code', monospace"
    fontSize: "1.125rem"
    fontWeight: 500
    lineHeight: 1.2
    letterSpacing: "0.03em"
  heading:
    fontFamily: "'Fira Code', monospace"
    fontSize: "1.125rem"
    fontWeight: 500
    lineHeight: 1.2
    letterSpacing: "0.03em"
  icon-xl:
    fontSize: "2rem"
  icon-lg:
    fontSize: "1.85rem"
  icon-md:
    fontSize: "1.65rem"
  icon-sm:
    fontSize: "1.05rem"
  icon-xs:
    fontSize: "1rem"
  body:
    fontFamily: "'Fira Sans', sans-serif"
    fontSize: "0.875rem"
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: "normal"
  button:
    fontFamily: "'Fira Code', monospace"
    fontSize: "0.82rem"
    fontWeight: 500
  label:
    fontFamily: "'Fira Code', monospace"
    fontSize: "0.75rem"
    fontWeight: 500
    lineHeight: 1.4
    letterSpacing: "0.03em"
  caption:
    fontFamily: "'Fira Code', monospace"
    fontSize: "0.7rem"
    fontWeight: 400
    lineHeight: 1.35
  badge:
    fontFamily: "'Fira Code', monospace"
    fontSize: "0.66rem"
    fontWeight: 500
    lineHeight: 1.0
rounded:
  sm: "4px"
  md: "6px"
  lg: "10px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "16px"
  lg: "24px"
  xl: "32px"
  "2xl": "48px"
components:
  button-execute:
    backgroundColor: "{colors.card}"
    textColor: "{colors.foreground}"
    rounded: "{rounded.sm}"
    padding: "14px 20px"
    height: "48px"
  button-execute-hover:
    backgroundColor: "rgba(16, 185, 129, 0.1)"
    textColor: "#FFFFFF"
    rounded: "{rounded.sm}"
  card-result:
    backgroundColor: "{colors.background}"
    textColor: "{colors.foreground}"
    rounded: "{rounded.md}"
    padding: "0"
---

# ZCPO // Forensic DGP Terminal Design System

## Overview

The Forensic DGP Terminal is a high-density, mission-critical forensic workstation interface engineered for deepfake detection, super-resolution, and morphological prior reconstruction from degraded surveillance CCTV footage.

The design philosophy fuses **Swiss International Typographic Style** with **Utilitarian Terminal Electronics**:
- **OLED Midnight Baseline (`#020617`):** Minimizes eye strain during prolonged visual evidence analysis in low-light forensic laboratories.
- **Rigid 1px Spatial Grid:** Interfaces, telemetry headers, and input controls are partitioned by explicit structural 1px dividers rather than soft decorative drop shadows.
- **Neutral Dominance (10% Rule):** Deep slate neutrals handle the operational surface; emerald phosphor (`#10B981`) is reserved strictly for operational status and affirmative forensic ranking.

### Target Redesign Specification (via `taste-skill`)

- **Aesthetic Archetype:** **Technical / Swiss Utilitarian**
- **Core Design Dials:**
  - **`DESIGN_VARIANCE`**: **Low**
    - Rigid 1px grid architecture.
    - Asymmetric dual-panel cockpit layout (420px configuration column + flexible results canvas).
    - Uniform spatial alignment with zero arbitrary card offsets.
  - **`VISUAL_DENSITY`**: **Compact**
    - High data density optimized for laboratory investigators.
    - Monospace telemetry tags at 12px (`0.75rem`), metric values at 18px (`1.15rem`), section headings at 18px (`1.125rem`).
    - Tight spacing rhythm (`--space-xs: 4px`, `--space-sm: 8px`, `--space-md: 16px`, `--space-lg: 24px`).
    - Interactive touch target bounds (≥ 44×44px) strictly maintained.
  - **`MOTION_INTENSITY`**: **Subtle**
    - Purposeful micro-transitions (150ms ease).
    - No continuous visual distractions or peripheral radar wave loops; steady status indicator.
    - Gentle breathing cadence for processing indicators (0.55 to 0.95 opacity).
    - Strict `@media (prefers-reduced-motion: reduce)` support.

### Surface & Viewport Overrides
- **Desktop Cockpit (≥ 1024px):**
  - `VISUAL_DENSITY`: **Compact**
  - `DESIGN_VARIANCE`: **Low**
  - Full-height viewport containment (`100vh`) with independent scrollable panels and zero window-level scrollbars.
- **Mobile Viewport (< 768px):**
  - `VISUAL_DENSITY`: **Balanced**
  - `DESIGN_VARIANCE`: **Low**
  - Natural single-column vertical stack with touch-expanded controls (≥ 44×44px) and zero horizontal overflow.

---

## Colors

| Role | Hex | CSS Variable | Usage |
|---|---|---|---|
| Background | `#020617` | `--color-background` | Deepest root layer, terminal backdrop, canvas |
| Surface / Card | `#0A0E1A` | `--color-card` | Panels, input modules, inactive tiles |
| Elevated / Muted | `#111728` | `--color-muted` | Drop zone, spec matrix, status blocks |
| Secondary | `#1E293B` | `--color-secondary` | Interactive hover surfaces, button backgrounds |
| Border | `#1E293B` | `--color-border` | Structural 1px grid lines, panel separators |
| Border Subtle | `#141B28` | `--color-border-subtle` | Sub-dividers, inner container lines |
| Border Highlight | `#334155` | `--color-border-highlight` | Hovered card borders, active focus accents |
| Foreground | `#F8FAFC` | `--color-foreground` | Primary text, titles, prominent numeric data |
| Muted Foreground | `#8B9BB4` | `--color-muted-foreground` | Labels, inactive icons, auxiliary metadata |
| Accent (Phosphor) | `#10B981` | `--color-accent` | Primary trigger border, online status, rank 1 badge |
| Accent Dim | `rgba(16, 185, 129, 0.08)` | `--color-accent-dim` | Active mode button tint, subtle badge fills |
| Destructive | `#EF4444` | `--color-destructive` | Error messages, failed integrity alerts |

---

## Typography

- **Heading & Telemetry:** `'Fira Code', monospace` (Google Fonts)
- **Body & Descriptive:** `'Fira Sans', sans-serif` (Google Fonts)

### Typographic Hierarchy

| Role | Font Stack | Size | Weight | Line Height | Case / Tracking |
|---|---|---|---|---|---|
| **Terminal Title (`h1`)** | Fira Code | `1.125rem` (18px) | `500` | 1.2 | Uppercase, `0.03em` |
| **Panel Header (`h2`)** | Fira Code | `1.125rem` (18px) | `500` | 1.2 | Uppercase, `0.03em` |
| **Telemetry Label (`.mono-label`)** | Fira Code | `0.75rem` (12px) | `400` / `500` | 1.4 | Uppercase, `0.03em`, Tabular |
| **Metric Value (`.data-value`)** | Fira Code | `1.15rem` (18.4px) | `400` / `500` | 1.2 | Tabular Numerals |
| **Body & Descriptions** | Fira Sans | `0.875rem` (14px) | `400` | 1.5 | Sentence Case |
| **Mode Subtext** | Fira Code | `0.70rem` (11.2px) | `400` | 1.35 | Uppercase, `0.01em` |

---

## Layout

- **Desktop Cockpit Grid:**
  - `max-width: 1600px`, `height: 100vh`, flexbox container.
  - Header: `64px` fixed height.
  - Main Body: `calc(100vh - 98px)` with two columns:
    - Left Column (`input-panel`): `420px` fixed-width operational configuration sidebar.
    - Right Column (`results-panel`): Flexible space `flex: 1` displaying synthesized candidate cards.
  - Footer: `34px` fixed height status telemetry bar.
- **Responsive Layout:**
  - Tablet (`768px – 1024px`): Responsive wrap with internal vertical scrolling.
  - Mobile (`< 768px`): Single-column vertical layout, drop zone 240×200px, 100% width mode toggles, zero horizontal overflow.

---

## Elevation & Depth

- The interface rejects heavy blurry shadows in favor of crisp 1px structural boundaries.
- **Elevation Vocabulary:**
  - `--shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.2)`: Result cards default state.
  - `--shadow-md: 0 2px 5px rgba(0, 0, 0, 0.25)`: Hovered cards.
  - Zero zero-offset colored blur halos (`dark-glow` prohibited).

---

## Shapes

- **Radius Language:**
  - `--radius-sm: 4px`: Buttons, pills, form inputs, drop-zone reticles.
  - `--radius-md: 6px`: Result cards, empty state containers.
  - `--radius-lg: 10px`: Modals (if present).
  - Status indicator: `50%` round (7px).

---

## Components

### Initiate Reconstruction Button (`.execute-btn`)
- **Visual Style:** Neutral dominance (`#0A0E1A` background, 1px solid `#10B981` border, `#F8FAFC` text, `#10B981` icon).
- **Interactive State:** Hover adds 10% emerald background tint (`rgba(16, 185, 129, 0.1)`).
- **Disabled State:** Transparent background, muted slate border (`#1E293B`), 0.5 opacity.
- **A11y:** Min-height 48px, `:focus-visible` ring.

### Drop Zone (`#drop-zone`)
- **Visual Style:** 220×220px centered square with corner alignment reticles.
- **Interactive State:** Hover/dragover shifts background to `#1E293B`, reticles brighten to 1.0 opacity.
- **A11y:** Keyboard focusable via `tabindex="0"`, triggers file input on `Enter` or `Space`.

### Mode Toggle Selector (`.mode-btn`)
- **Visual Style:** Segmented radio buttons with monospace title and descriptive subtext.
- **Active State:** 1px `#10B981` border with subtle 8% background tint.
- **A11y:** `role="radiogroup"`, `role="radio"`, `aria-checked="true/false"`.

### Results Matrix Grid (`#results-grid`)
- Auto-fit grid (`minmax(320px, 1fr)`) displaying top probabilistic reconstruction cards.
- Each card displays candidate rank badge, source resolution, tabular FAN loss metric, and identity fit score.

---

## Do's and Don'ts

### Do
- Do tint all neutrals toward the cool navy/slate hue (`#020617`, `#0A0E1A`, `#1E293B`, `#8B9BB4`).
- Do use tabular numerals (`font-variant-numeric: tabular-nums;`) on all telemetry, coordinates, and metrics.
- Do enforce minimum touch dimensions (≥ 44×44px) across all interactive triggers.
- Do maintain a >1.25× typography hierarchy step between body, sub-headings, and titles.
- Do include `aria-hidden="true"` on all decorative SVG/Phosphor icons.

### Don't
- Don't use zero-offset colored blur halos or neon glows on dark backgrounds.
- Don't use decorative side-stripe borders (`box-shadow: inset 3px 0 0`).
- Don't use card-inside-card nesting.
- Don't use gradient text (`background-clip: text`).
- Don't use Unicode emojis as UI icons.
- Don't use pure raw `#000000` or untinted `#808080`.
- Don't allow horizontal scrolling on mobile viewports.
