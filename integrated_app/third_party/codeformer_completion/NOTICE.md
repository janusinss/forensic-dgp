# CodeFormer inference architecture

Source: https://github.com/sczhou/CodeFormer

Pinned revision: `b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b`.

Files copied: `basicsr/archs/codeformer_arch.py`, `basicsr/archs/vqgan_arch.py`, and `basicsr/utils/registry.py`. The architecture files use relative imports and the standard-library logger instead of importing the full BasicSR training environment. Network operations are unchanged. This avoids unrelated training dependencies and package-wide registration/import side effects.

CodeFormer is Copyright 2022 S-Lab, under the non-commercial S-Lab License 1.0, retained in LICENSE. The registry and VQGAN source comments retain their original upstream attributions. This component is used as a research benchmark; its inclusion does not establish permission for every deployment use.

The app uses the trained continuation asset `codeformer_completion_r2.pth` from the separate immutable `completion-r2-models-v1` release. The upstream inpainting checkpoint, retained R1 parent and R2 protocol are documented in [the completion model card](../../COMPLETION_MODEL_CARD.md) and [R2 provenance](R2_PROVENANCE.json). Do not substitute the restoration checkpoint `codeformer.pth` or rename R1 weights as R2. Retain the unchanged [40-source training attribution](TRAINING_ATTRIBUTION.json); its original R1 protocol binding remains historical, and the R2 protocol uses the same attributed references. The unchanged covering detector is a separate asset. Prior R1 downloads are preserved.
