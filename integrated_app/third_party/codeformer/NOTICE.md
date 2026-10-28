# CodeFormer inference architecture

Source: https://github.com/sczhou/CodeFormer

Pinned revision: `b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b`.

Files copied: `basicsr/archs/codeformer_arch.py`, `basicsr/archs/vqgan_arch.py`, and `basicsr/utils/registry.py`. The architecture files use relative imports and the standard-library logger instead of importing the full BasicSR training environment. Network operations are unchanged. This avoids unrelated training dependencies and package-wide registration/import side effects.

CodeFormer is Copyright 2022 S-Lab, under the non-commercial S-Lab License 1.0, retained in LICENSE. The registry and VQGAN source comments retain their original upstream attributions. This application uses trained adaptations of the architecture; the license conditions remain applicable to use and distribution.

This application downloads its frozen R2 base, U96 adapter and optional u64 heads from this repository's model release. Exact URLs, sizes and SHA-256 values are pinned in `implementation/models.json`; the installer and runtime verify them before use. These application assets are separate from the upstream inpainting checkpoint.
