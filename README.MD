# Forensic face restoration — IMPLEMENTATION

Windows application for one already cropped frontal or mildly turned face, producing a 256×256 result. This branch packages the trained CodeFormer-based U96 adaptation, optional reviewed Protect/Apply preparation, and original bypass in the existing local interface.

## Install

Install 64-bit Python 3.12 or 3.13 with the Python launcher or Python on PATH. Initial setup requires internet access.

```bat
git clone --branch IMPLEMENTATION https://github.com/janusinss/forensic-dgp.git
cd forensic-dgp
install.bat
```

The installer creates venv, installs CPU PyTorch and the application dependencies, and downloads three assets from this repository's [model release](https://github.com/janusinss/forensic-dgp/releases/tag/implementation-models-v1). Every model file must match its size and SHA-256. Existing mismatched files are preserved and reported.

## Start

```bat
start.bat
```

The browser opens after the local health check passes, normally at http://127.0.0.1:8000. Keep the terminal open; press Ctrl+C to stop. The launcher never kills another application's port. To select an available port:

```bat
start.bat --port 0
```

The actual address is printed and recorded in scratch/implementation_launch.json. After setup, inference runs locally without model-download requests.

## Use

1. Upload one PNG/JPEG crop and inspect visible facial structure. Request clearer or less-covered input when too little information remains.
2. Use U96 by default; u64 is an experimental comparison. Choose Auto, On or Off. Off retains the original resized input and performs no inference.
3. Bypass preparation, or paint Protect over existing facial blur and soft edges. Paint Apply only where added surroundings blur is permitted.
4. Preview and confirm manual preparation. Edits invalidate confirmation; every preview rebuilds from the original upload. Protected native pixels stay unchanged during preparation.
5. Restore the selected input. Review original, selected input and result; download PNGs or the optional evidence ZIP with available RAW output.

Preparation is optional: it did not consistently help the paired study. This branch does not provide an automatic blur matcher or always-on surroundings blur. Input protection does not guarantee identical generated appearance. Completion and covering removal remain separate, deferred work.

## Model choice

A saved-output comparison recomputed 642 outputs across 321 paired DEV cases. u64 had slightly better aggregate scores but worsened normal-blur PNG error in 46/48 cases. U96 is the conservative functional default because visible structure and preservation take priority over average scores. Both retain preservation failures; a runnable app is not independent FINAL qualification or proof of recovered identity. See [MODEL_CARD.md](MODEL_CARD.md).

Models are defined in implementation/models.json and loaded only in disposable CPU workers. No training, optimizer, VM, historical pilot or research output directory is required. Uploaded images and retained request records stay on this computer under scratch/implementation_requests. Remove your own request records when they are no longer needed.

## Troubleshooting

- Missing dependency or model: rerun install.bat and inspect the reported failing step.
- Low memory: close unused applications. The worker requires 2.5 GiB available RAM and has a 6 GiB RSS limit.
- Worker failure: inspect the retained request directory before restarting start.bat. There is no automatic retry; each request has a 360-second limit.
- Busy port: close your own earlier server or use start.bat --port 0.

## Attribution

CodeFormer is an existing S-Lab architecture; this thesis uses trained adaptations. Retain [the S-Lab license](third_party/codeformer/LICENSE) and [notices](third_party/codeformer/NOTICE.md). Its non-commercial terms remain applicable. This implementation includes no training dataset or reserved FINAL face set.
