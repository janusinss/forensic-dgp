# First-stage input workbench

## Product and scope

This local experimental face-restoration app supports an operator reviewing one face. The first stage selects a still picture or captures an unaligned face crop from a video source. This specification covers that stage only. Preserve the existing HTML, CSS, JavaScript, API, and worker stack.

## P0: approved work

| Surface | Required behavior |
| --- | --- |
| Input controls | Use a split layout: Picture and Video control groups are vertically centered on the left. Source actions are grouped closely, face tabs keep crop tools on their right, and Use this face remains the main crop action. Video offers file, webcam, or LAN RTSP. Opening a chosen file is automatic. Starting a live source requires an explicit action. |
| Preview | Show the current picture, live video, or frozen frame on the right. Expose source and operation status. |
| File transport | Use the browser's native video controls for uploaded files, with a supplemental speed selector from 0.25× through 4× in 0.25× increments. Keep supported native choices within that range visible even when they fall between quarter steps; the step buttons move to the neighboring quarter step. Preserve the selected speed through playback, seeking, Freeze and Resume. Default to 1× on a new file. Space plays or pauses in the active file capture view without intercepting form or dialog interaction. Native controls and menus vary by browser. |
| Source quality | Show the original uploaded video's dimensions. A single uploaded file has no alternate quality levels; do not transcode or change the source crop. |
| Capture | Freeze a clean frame, select a detected box, drag a rectangle on the preview, or press Enter there to create a keyboard-adjustable crop; adjust the crop, and explicitly choose Use this face. Selecting a face may zoom the preview; Full frame crop returns to the full view without replacing the selected crop. No separate Draw crop button is shown. |
| Review | Show the chosen Picture, require the existing human assessment of one sufficiently clear face, and retain the warning and continuation gate. Back to capture and return to an earlier uploaded picture remain available when applicable. |

## State and evidence

Video uses the same light and forest-green control language as Picture. Its source summary, face controls, preview heading, and file playback area remain aligned with the existing centered desktop split workflow; visual refinements do not change the review or capture behavior. The requested design priority is the desktop web browser; existing narrow-screen fallback is retained without separate mobile feature work.

Capture and Review are separate states. Source changes stop the old source and invalidate pending operations. Freeze is unavailable while a video is seeking. Starting a live camera, changing speed, or selecting a crop does not run restoration. A new accepted Picture clears prior review and masks; the current Picture must pass the existing inspection flow.

The full clean frozen frame and integer source-pixel crop coordinates are authoritative. Preview zoom is visual only; it must not alter the crop or its evidence. Preserve the existing source-frame, crop, and metadata evidence binding. Detection boxes are suggestions; no tracking, alignment, FAN inference, quality ranking, or identity assertion is introduced.

## States and verification

Provide empty, opening, ready, frozen, selected, reviewing, failure, permission-denied, and stopped states. Keep errors visible and provide recovery. Keep keyboard operation, labeled controls, and accessible focus. Fit normal primary desktop flows within the viewport; retain accessible overflow for large text and exceptional states rather than clipping controls.

Check picture upload, all three source choices, explicit live start and stop, video speed and seek, frozen-frame selection, manual crop, zoom/full-frame switching, review reset, back navigation, and source-to-crop evidence. Prioritize desktop browser checks at 1280×720, 1366×768, and 1920×1080. Physical camera and RTSP results require separate actual-device checks.

## Deferred

P1–P3: no additional features are specified for this redesign. Do not introduce a dependency migration or new processing pipeline.

### Uploaded-file session lifetime

Input-mode switches, accepted-crop Review, temporary navigation away from Input, and document hiding pause an opened local file without discarding its playback position, speed, frozen crop or preview zoom. Returning does not autoplay. Native seeking or buffering does not make an opened file an incomplete session. Freeze waits for a decoded frame. Explicit Stop, file/source replacement and page unload release the file; incomplete openings and live cameras retain full teardown. Suspension invalidates stale capture responses.
