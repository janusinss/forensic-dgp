# Native removal-mask fidelity repair: functional verification

Location:`static/face_workflow.js`. Cause:an untouched native mask was reduced
to a≤512px editor and enlarged for export. Chromium changed768pixels on the
1024×768 fixture,720 on1280×720 and2,616 on the1024² thin-feature fixture.
The256²/512² controls were exact. This could change removal areas and correctly
make the server classify an untouched automatic proposal as assisted.

The repair keeps native mask pixels through import,review,editing and export.
The displayed editor dimensions/design stay unchanged. Pointer/keyboard strokes
map display coordinates into native support; only the edited support is
normalized. Hover reads the preview raster. Undo retains at most12frames and
96MiB, evicting before allocation; reset retains the automatic proposal.

All five Chromium exports now have zero changed mask pixels.21 named interaction
checks plus zero page/console errors cover paint/erase,exactundo/reset/clear,
delayed/invalid/superseded imports, review gating and submitted native mask bytes.
The4096×3906 fixture(15,998,976pixels) independently exports all40,001 selected
pixels exactly. The maximum-size undo ceiling and reset passed. A harness
self-comparison was discarded and replaced with a distinct-source targeted
browser check; two coordinate/focus harness failures are retained separately.

The11 backend tests include1024×768/1280×720 native automatic→one-pixel-assisted
→exact-reset transitions, even when the delivered256mask/output stay unchanged.
They use mocked components. Browser endpoints were intercepted synthetic
fixtures; no trained detector,completion or restorer forward was executed.
The local server was started for these checks; loading its existing components
does not constitute training or output-quality evidence.

Root inspected375/768/1280 responsive screenshots:three result cards, editor
and controls render with no horizontal overflow/page/console errors. CSS,
template and both backend source hashes remain unchanged. SourceJS is
`366d4a32a8d2c0b643ba8f8bf6274310090b09cb88f87b2f384bc8ca6738d080`; the exact original source and tests are retained.

[Independent 79-file/pixel binding audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/dgp_native_mask_roundtrip_v1_analysis/browser_binding_audit.json>).
[Root inline browser artifacts](<C:/xampp/htdocs/YEAR 4/Testing/scratch/dgp-native-mask-v1/after-layout.json>).
[Backend regression evidence](<C:/xampp/htdocs/YEAR 4/Testing/outputs/dgp_native_mask_roundtrip_v1_analysis/backend_tests_after.json>).
This verifies processing fidelity only. Automatic detector quality, useful
completion, all seven covering families and independent final review remain
unqualified. It does not show restoration or hidden-identity recovery.
