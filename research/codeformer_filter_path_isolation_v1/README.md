# Frozen CodeFormer path comparison

This targeted inference comparison checks five already reviewed synthetic inputs
across the original CodeFormer, retained AdaIN parent and R2 states. It changes
only the released global fusion weight (0 or 1) and AdaIN switch (off or on).
It performs no optimization, training, target scoring or application changes.

The frozen protocol is
`outputs/codeformer_filter_path_isolation_v1_preparation/protocol_v1.json`
with SHA256
`23772988d16e11f52351365e66ce1b6d8ef77694f873966db98e24512b0aa86c`.
The cases were chosen after observing failures. They are exploratory examples,
not unseen validation or a representative replacement for the earlier 72 cases.

Fifteen existing weight-1/AdaIN-on outputs are reused. Each of three models
first makes one fresh clear baseline call whose RAW and PNG arrays must exactly
match its cached baseline before any intervention. The other three settings
require 45 new outputs, giving 48 total calls and 60 factorial cells. Runtime is
estimated at 4-7 minutes with a hard 600-second stop. The worker limits its RSS
to 3 GiB, keeps at least 768 MiB available RAM and 1 GiB disk free, and caps new
output at 170 files and 64 MiB. Partial results are retained and cannot resume.

Read-only hooks count the actual top-level call and four fusion sites. Returned
logit and low-quality-feature hashes must match across the new settings for a
given input/model. The 256 token indices are recomputed with the released
softmax/top-k rule from returned logits; the internal codebook lookup itself is
not directly captured. Fresh clear parity checks hook noninterference. Full 515
tensor states remain unchanged, and the exact 165 generator/codebook tensors
must be identical across all three retained models.

Disabling all four fusion sites is one global nonlinear intervention, not an
additive explanation of individual sites. AdaIN changes latent statistics and
is not a pure brightness control. Neither a successful technical comparison nor
any selected setting qualifies a model, changes existing acceptance thresholds,
or changes the application's defaults. All prior 96 inputs, 24 insufficiency
decisions and separate appearance-flag evidence remain preserved.

CPU-only source fixtures:

```powershell
& '.\venv\Scripts\python.exe' -X utf8 -B -m unittest discover -s tests -p 'test_codeformer_filter_path_isolation_v1.py' -v
```

The parent workflow launches the hash-reviewed frozen worker only after source
review. Saved outputs receive a separate artifact/metric audit and visual review.
