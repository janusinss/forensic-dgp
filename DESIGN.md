# First-stage workbench design

## Current State (Extracted)

- Active workbench surfaces use warm light `#f8f7f1`, deep green-black `#101d1d`, ink `#172725`, muted ink `#4b6056`, and primary green `#176b52` / `#176c56`. Borders include `#a7b7ab` and `#b3c6b9`; review errors use `#912d29`.
- The active workbench variables use Segoe UI/system sans and Consolas/monospace. Panel headings use the local Loader Inter face; the older base heading rule uses Georgia. The first-stage redesign uses the active sans variable for headings and controls.
- Controls use 44px minimum height, roughly 4–12px spacing within groups, 4px control radii, 1px borders, green focus rings, and subdued green selected states. The preview uses contain sizing against a dark surface.
- The first stage uses a 42/58 desktop split with controls on the left and preview on the right; a media query at 800px places the preview in the active control flow. Existing crop controls support pointer and keyboard editing, a precise adjustment dialog, and a full-frame view.

## Target Redesign Specification

**Visitor mode:** Operate. The image is the focal element, with the next valid action close to it. Use the existing warm-and-forest palette and sans control typography; preserve app identity and visual clarity.

- `DESIGN_VARIANCE: 4` — practical, asymmetric use of available space.
- `MOTION_INTENSITY: 2` — short control feedback, no distracting movement in the preview.
- `VISUAL_DENSITY: 5` — compact contextual controls with enough spacing for review.

### Composition

Use a desktop split: Picture and Video controls vertically centered as a complete group on the left. Keep source actions close, separate face selection with a modest gap, and place crop dimensions beside precise adjustment above the full-width Use this face action. Place the contain-fit picture or video preview on the right. Uploaded video uses native browser controls with supplemental quarter-step speed and source dimensions beside the preview; browser menus and appearance may vary. Keep Capture and Review as separate phases: face selection and crop adjustment appear during Capture; human review choices and continuation appear after Use this face. Keep status and errors readable. The selected image stays visible beside review controls wherever space allows.

At standard desktop sizes, target a viewport-fitting primary workflow without nested scroll. At narrow widths or enlarged text, permit natural scrolling while keeping the preview, focused control, and actions reachable. Preserve 44px targets, visible focus, reduced-motion behavior, and the original aspect ratio. Preview zoom never changes source pixels or evidence.

The approved split uses a `#f7f9f5` left control surface and a `#101d1d` right preview surface. The desktop grid allocates 42% to controls and 58% to preview, with a 370px minimum control column. Keep ink `#172725`, muted ink `#4b6056`, accent `#176b52`, and visible borders; controls use 4px radii. At 800px and below, place the preview within the control flow for the active phase without clipping; allow overflow when needed.

### Video control refinements

Prioritize the desktop web browser. Match Picture's light surface and green primary actions: Video uses `#f7f9f5`, white fields, `#18352c` ink, `#53685d` muted text, `#c6d6ca` borders, and `#176b52` accent. Controls have 9px corners; the file input uses 10px, while the tinted source summary uses 12px. Selected faces use the green accent. Decorative icons accompany text and stay hidden in Picture mode. The dark preview adds a compact heading and an unboxed playback row with a transparent background; speed controls remain distinct. Retain the existing narrow-screen fallback without separate mobile feature work. Native player and source-pixel crop behavior remain unchanged.

### Verification

Inspect empty, loaded, frozen, selected, review, error, and permission-denied states at desktop sizes 1280×720, 1366×768, and 1920×1080. Confirm no accidental horizontal overflow, clipped actions, missing focus, or hidden status. Verify the real crop and review behavior separately from visual presentation.

### Shared dropdowns and crop actions

Visible dropdowns across the workspace and dialogs share 8px control corners, a consistent chevron, 44px targets, explicit hover/focus/disabled states and the existing light/forest palette. Playback speed uses the matching dark palette. Supporting browsers theme the native picker and option states; other browsers retain the native menu. Keep hidden select backing fields hidden and preserve their radio wiring.

The dedicated Draw crop button is removed. Dragging on the frozen preview still creates a manual crop. When no crop exists, Enter creates a centered starting rectangle for keyboard adjustment; Enter preserves any existing selected crop. The existing full-view action is labeled Full frame crop; it resets preview zoom without replacing the selected source-pixel crop.
