# Fixed filter-pair audit data

This CPU-only preparation creates twelve prospectively fixed input corruptions
for four already approved TRAIN photographs and four existing development
photographs. It does not train or select a restoration model, calibrate an input
threshold, add new identities, or use the five user-supplied unknown images.

Every source must be an exact, hash-verified RGB1024 photograph already recorded
in the audited reference manifest. Clean512 is direct LANCZOS1024→512; clean256
must match the retained direct LANCZOS1024→256 pixels exactly. The observation is
direct BILINEAR1024→256 followed by one fixed filter. There is no face alignment,
crop, warp, colour/contrast adjustment, or clean-target upscaling.

The local rectangle `[64,64,192,192]` is a synthetic corruption footprint. It
does not identify facial features, a covering, or a completion/removal area. Hard
rectangle boundaries are intentional stress artifacts and are recorded.

All images require input-only review before any use in restoration training.
The strong controls (Gaussian6, repeated rows, global mosaic16, local blur10,
local mosaic16) are initially excluded from restoration training. They can show
refusal coverage; they are not automatically declared usable or insufficient.
Other TRAIN profiles remain pending review. All DEV examples remain excluded
from optimization regardless of usability review. TRAIN and development roles stay separate;
the eight photographs are a small controlled diagnostic cohort, not a complete
model-training dataset or independent final evaluation.

The generator refuses existing output directories, verifies source and output
hashes, records versions/parameters/order/seed, writes exclusively and keeps a
finite file/byte bound. No corrected restoration images or trained checkpoint
are created. Root must review source and approve execution before generation.

Input-only review uses twelve profile-first sheets with eight exact256 panels
each. Strong controls are reviewed first and the clear profile last. Severe
judgments must be locked before viewing clear or clean-reference images. This
does not claim blinding to other degraded versions of the same photograph.
