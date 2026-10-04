# Face-prior data and evaluation limits

Recorded 4 October 2026. Windows:
`C:\xampp\htdocs\YEAR 4\Testing\CCTV_FACE_PRIOR_DATA_LIMITS.md`.
Intended VM counterpart: `~/forensic-dgp/CCTV_FACE_PRIOR_DATA_LIMITS.md`
after explicit document sync.

CodeFormer's author training instructions use FFHQ. The released checkpoint
does not provide a verified list of individual training images in this workspace.
Our FFHQ holdout is therefore held out from **our conditioning training**, not
proved unseen by the pretrained prior or clean-code teacher. Possible pretraining
overlap must be reported; photographic proxy improvements alone cannot establish
independent CCTV generalization.
[Author training instructions](https://github.com/sczhou/CodeFormer/blob/master/docs/train.md).

Asian prior images retain their source provenance. Their subject/exposure overlap
with other datasets and prior pretraining is unknown. Upscaling a small source
photograph does not create high-quality detail or aligned clean CCTV truth.
Source categories are dataset provenance, not inferred ethnicity.

V11's clean-code oracle consumes clean training references. V12 fits those ten
training faces and fixed synthetic camera profiles. Neither supplies validation,
unseen-identity restoration, native CCTV fidelity or Zamboanga-specific evidence.
A clean reconstruction that changes facial structure is a prior limitation,
even when its image looks coherent.

The 24 public native development crops remain separate unpaired evidence; 32
reserved native cases remain unrendered/unforwarded. Development previews and
recognizer similarity are not final identification accuracy. No real Zamboanga
CCTV samples or independent final human review are available yet. The active
Goal requires useful structure-preserving local workflow outputs, beyond fitting
losses or a completed training run.
