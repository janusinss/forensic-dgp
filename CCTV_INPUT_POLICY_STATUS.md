# CCTV input-policy audit: preparation, not a deployed gate

Update 5 October 2026: the new `dgp_face_workflow_v3.py` route removes the 32-pixel
upload gate and uses observed-support quality signals. Mandatory input-only
operator review routes insufficient/unsupported crops to a clearer/suitable
image request; it is not an automatic facial-structure classifier. Exactly flat
native visible regions and near-total reviewed covering are rejected before
generation. Six frozen native core cases match cached DGP raws exactly; no useful
native improvement follows. See `CCTV_DGP_APP_V3_INTEGRATION.md`. The original
input audit below and its sources/results remain historical and unchanged.

Verified 4 October 2026. Local workspace `C:\xampp\htdocs\YEAR 4\Testing\`;
intended VM document copy `~/forensic-dgp/CCTV_INPUT_POLICY_STATUS.md` after an
explicit transfer. This input-only audit made zero model forwards, backward
calls or optimizer updates and read none of the 32 reserved native crops.

Evidence: local `outputs\cctv_input_policy_audit_v1\results.json`, SHA256
`224ec8c5b316008152f2fee4cf02eab182ffceab7c52403784edd6d433b5634c`,
and `outputs\cctv_input_policy_checks_v1\review_receipt.json`.
Optional VM counterparts are `~/forensic-dgp/outputs/cctv_input_policy_audit_v1/`
and `~/forensic-dgp/outputs/cctv_input_policy_checks_v1/` after transfer; these
local audits do not require a VM upload or training.

The current decoder rejects 18 of 24 frozen development crops below its 32-pixel
minimum. This includes five of the six input-reviewed coarse frontal/mild faces.
Conversely, the larger 49×59 `dev_ge40_05` is accepted even though input review
judged its facial structure insufficient. Pixel dimensions alone cannot qualify
usable facial structure.

Controlled native-crop padding/interpolation checks used the same gray-127
padding for both interpolation choices. Excluding padding from AREA quality
scores changed seven of 24 automatic restoration suggestions. Changing AREA to
bilinear on full-square support changed eight; changing interpolation with
observed-only support changed one. These are suggestion changes, not demonstrated
restoration improvements. Frozen DGP gray-128 preparation suggested restoration
for all 24 inputs, including insufficient cases; it cannot serve as a usable-face
classifier. The report contains 120 distinct signal outputs and 24 additional
legacy control replays, for 144 actual signal evaluations.

`C:\xampp\htdocs\YEAR 4\Testing\cctv_input_quality.py` prepares a 256×256 crop
using the frozen DGP geometry and excludes uncaptured padding, removal areas
and their scoring halo from blur/noise signals. It explicitly makes no structure
qualification claim. Six tests pass for padding/covering invariance, supported
detail and blur, small-crop geometry, unavailable support and invalid masks;
eight existing workflow tests pass. The current app route, UI, decoder and
CodeFormer default were not modified. This helper has not been deployed.

Next: after a useful DGP comparison supports route integration, version and
validate the insufficient-input decision separately from blur/noise suggestions.
Keep development calibration and independent final assessment separate. The
audit exposes a product limitation; it does not establish a safe automatic gate.
