# R1 completion integration verified

28 October 2026. This selective integration adds optional reviewed covering
completion to the existing Conditioned SFT V2 restoration application. It is
based on IMPLEMENTATION commit320ec20542e49a133d2708530748a3c60f5ac0e7.

The user selected restoration first, then an automatic covering proposal,
optional correction and explicit confirmation, followed by trained R1 completion.
Uncovered faces can finish after restoration; empty reviewed masks skip R1.
Original, restoration, reviewed mask and final estimate remain separate, with
PNG and optional evidence downloads.

## Restoration preservation

Eighteen protected restoration source, dependency and vendor files remain
byte-for-byte equal to the prior IMPLEMENTATION checkout. Restoration weights,
loading, preprocessing, normalization, Auto/On/Off selection, model inference
and PNG quantization are unchanged. The launcher selects the combined ASGI
entrypoint; a passive receipt binds its finished output to the new stage.
Completion never reruns restoration or applies it to generated hidden pixels.

R1 uses a separate inpainting namespace and exact verified inference weights.
Every pixel outside the reviewed removal area is copied from the supplied
restoration result. That result can differ from the original upload because
restoration itself may change visible appearance. Completion adds no margin.

## Application verification

The bundled inline Playwright workflow passed upload, inspection, restoration,
automatic proposal, mask import/edit/undo, confirmation, CPU generation and
actual PNG/ZIP downloads. Editing invalidates approval. Successful generation
consumes it. Invalid masks and excessive coverage refuse before R1. New uploads
clear prior completion state. Viewports375/768/1280 show no horizontal overflow.
There were no page exceptions or unexpected failed requests; the negative
coverage test intentionally returned HTTP422.

Nine fixed development cases cover seven covering families and two clear
controls. Independent saved-output audit passed1212checks and verified1,521,399
exact outside-mask RGB bytes, all ZIP CRCs/manifest hashes, frozen input/mask
pixels, raw-to-PNG replay, model provenance and worker receipts. All completed
workers exited with zero surviving children within120seconds. Every planned
case and three exact256 comparison sheets were visually reviewed.

All69 unit tests ran in both the isolated integration environment and a clean
clone:68passed, one Windows symlink fixture skipped due to host privileges.
The unchanged restoration regression tests are included.

## Clean Windows CPU installation

A new ordinary clone began with no models or environment. install.bat created
its own short-path CPU environment and verified all six model assets. Three
restoration assets came from the existing public release; V2 heads were staged
from the committed bundled asset. R1 and detector were downloaded from the new
public completion-r1-models-v1 prerelease and passed complete size/SHA checks.
All installed model files have independent single-link inodes.

start.bat launched that clean installation. One actual browser request with
restoration On, automatic proposal, fixed reviewed mask and R1 passed every
stage and real PNG/ZIP downloads without browser errors. R1 took9.9seconds;
the earlier remaining-family requests took8.2-8.7seconds and empty bypasses
under one second. These are observations on the i5-11260H/15.69GiB reference
laptop, not arbitrary-hardware timing or minimum-device certification.

## Known limits and retained evidence

The public development photographs are unpaired; synthetic degradation does
not supply clean hidden-face truth. All nine automatic proposals were empty,
so all seven covering cases required assistance. Automatic detection is not
qualified. The hair example retains blue false color and uncertain gaze; other
hidden anatomy and expression remain plausible estimates. Exact preservation
and successful downloads do not qualify full-family usefulness or identity.

Earlier automated clients lost response bodies after completed model calls.
All failed ledgers and completed outputs remain retained locally. Completed
inferences were recovered rather than repeated. Saved-response controls isolated
the transfer issue. Bounded response writes preserve identical bytes, and exact
browser/Python JSON numeric meaning binds reviews without numerical tolerance
or model changes. The user-selected actual browser workflow passed afterward.

Private inputs, outputs, raw arrays, masks, source pins, reports, screenshots
and failed ledgers stay in ignored scratch folders. They are not published in
the application source. Training optimizer states and research archives are
excluded from deployment. No new training, VM operation, research-tree merge,
checkpoint overwrite or RESTORATION branch change occurred.

Full-family acceptance, useful automatic proposals, independent blinded human
FINAL and minimum-hardware certification remain incomplete. The thesis goal
stays active. This is an experimental local application integration.
