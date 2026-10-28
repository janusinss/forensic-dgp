# Separate R1 completion assets

Run `install.bat` for the normal Windows setup. The existing restoration installer
runs first without changes. The added completion setup then uses the validated
checkout Python environment and downloads only these two SHA-256-pinned files
from the `completion-r1-models-v1` GitHub release:

- `codeformer_completion_r1.pth`
- `covering_detector_segmenter_v1.pth`

The downloader verifies the actual release response; this document does not
assert release publication. Existing mismatched files and unsuccessful partials
are preserved. There is no automatic retry or checkpoint substitution.

For offline setup after obtaining both exact files, run from this checkout:

```powershell
python -B -m implementation.completion_install --local-r1 "C:\verified-assets\codeformer_completion_r1.pth" --local-detector "C:\verified-assets\covering_detector_segmenter_v1.pth"
```

Explicit local mode performs no download. A peer asset can be omitted only if its
installed copy already passes verification. To verify existing files without
changing them:

```powershell
python -B -m implementation.completion_install --verify-only
```

Successful new acquisition saves a separate installation receipt under
`models/completion/`, recording exact model hashes and whether each file came
from a verified local copy, HTTPS download or an existing verified installation.
The scientific asset manifest, restoration assets, and research evidence remain
unchanged. Installation success is not completion-quality qualification.
