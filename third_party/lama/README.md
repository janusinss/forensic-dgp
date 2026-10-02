# Vendored LaMa generator

Apache-2.0 source from https://github.com/advimman/lama at 786f5936b27fb3dacd2b1ad799e4de968ea697e7. Copyright 2021 Samsung Research. Retain LICENSE. FFC source also credits https://github.com/pkumivision/FFC.

The pinned Big-LaMa generator/FFT operations are unchanged. Imports use minimal local helpers; discriminator and unused get_shape import are removed. Spatial-transform variants are rejected because the pinned config does not use them. Training code/dependencies are not bundled. Preparation provenance and upstream hashes are saved in outputs/lama_pretrained_v1/provenance.json. The prepared generator tensors derive from the original archive; no training occurs.
