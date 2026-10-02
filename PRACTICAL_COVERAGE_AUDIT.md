# Practical output coverage audit — 2 October 2026

This is the first evidence milestone of `SYSTEM_WORKFLOW_AND_GOAL.md`. It is a
source-only audit, not output-quality validation or a new training result.
Windows: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM repository counterpart after transfer: `~/forensic-dgp/`.

## Verified inventory

Inspected all 83 training records in
`dataset/detector_supported_review_v1/manifest.json`: 51 covered and 32 uncovered.
Compared the original native source with the accepted crop on five contact sheets.
Verified each training source/image/mask/support fingerprint and the unchanged
source manifest. Metadata from all 115 records confirms 83 train / 25 validation /
7 test, with no cross-split source-group or exact-source-byte overlap. This does
not prove identities are disjoint. No validation/test image was opened by this
audit, and previously inspected test cases remain previously inspected.

| Visually identified family | Training records | Qualification |
| --- | ---: | --- |
| Mouth masks / respirators | 44 | Includes one full side profile outside the initial pose scope |
| Sunglasses / opaque or reflective lenses | 6 | Two also have strong scene reflections |
| Strong lens glare | 3 | Includes the two reflective-lens cases; one additional existing reflection label is visually ambiguous for the new practical scope |
| Hand over a mouth mask | 1 | Does not establish a standalone hand-over-face completion case |
| Uncovered controls | 32 | Includes 10 ordinary clear-glasses controls |
| Hair obstructing facial features | 0 observed | Not confirmed in this reviewed subset |
| Scarf over the face | 0 observed | Visible headscarf with an uncovered face is a preservation control |
| Other objects over facial features | 0 observed | Not confirmed in this reviewed subset |

These are descriptive, overlapping tags from assistant visual review, not new
pixel annotations or independent expert labels. Counts do not add to 83 because
some cases have multiple covering types. This audit does not establish absence
of the missing families in the full downloaded datasets.

## Important cases

1. Record 70 has an existing reflection annotation but retains partly visible
   eyes. Preserve that annotation and report its practical strong-glare ambiguity
   separately; do not silently change historical supervision or scores.
2. Record 103 is a full side profile. Keep its existing training membership but
   exclude it from the primary frontal/mild-turn practical gallery.
3. Record 104 has a hand over a mouth mask. It tests a combined covering, not
   standalone hand removal.
4. Record 105 has an unknown lower contextual band in the supported native crop.
   Do not treat that band as observed facial evidence or hidden-face ground truth.
5. Record 96 has a headscarf with visible facial features. Keeping the scarf is
   consistent with the user's request; it cannot demonstrate scarf removal.

## Evidence and reproduction

Local evidence: `C:\xampp\htdocs\YEAR 4\Testing\outputs\practical_coverage_v1\`.
If transferred, VM counterpart: `~/forensic-dgp/outputs/practical_coverage_v1/`.
The audit was run locally; there is no new VM execution artifact.

Source manifest SHA256:
`860ea1dfba8fa442cab19bf3ab41f6a678109dcdb067c38ec0bd79ce43b51ace`.
The descriptive visual review sidecar is `visual_review.json`, SHA256:
`7431c55292bc18004708466a96a4a8948d0ccb760a179ddb3dc6221611b2bf6e`.
`inventory.json` contains source hashes and contact-sheet membership/hashes.
The five sheets are named `training_sources_01.png` through
`training_sources_05.png`. There were zero model forwards, optimizer updates or
source label/split mutations.

Reproduce the source inventory into a new directory from the Windows root:

```powershell
.\venv\Scripts\python.exe scripts/audit_practical_coverage.py --output_dir outputs/practical_coverage_recheck
```

After the repository and data are transferred, equivalent VM command from
`~/forensic-dgp/` with its environment activated:

```bash
python scripts/audit_practical_coverage.py --output_dir outputs/practical_coverage_recheck
```

The script deliberately refuses an existing output directory. Visual tags are
recorded separately after inspecting the sheets; automatic inventory does not
reproduce a human visual judgement.

## Next step

Check the existing unlabelled source pool for standalone hands, obstructing hair,
scarves and other objects before adding data. Prepare additional suitable public
research examples if these are missing. Freeze case membership, hashes, pose,
removal policy and five review criteria before new output comparisons. Compare
automatic and reviewed masks through pretrained completion on the same inputs;
choose a VM experiment only after identifying which stage limits useful output.

The complete practical gallery is not frozen yet, and model output has not been
evaluated under the new protocol. The Goal is active and unmet.
