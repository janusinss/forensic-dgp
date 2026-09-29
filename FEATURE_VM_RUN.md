# Mixed detector pilot on the VM — 29 September 2026

Training runs only on the Google Cloud GPU VM. This is a bounded detector
experiment, not generator training or evidence of improved completion output.
Local preparation and evaluation remain in `C:\xampp\htdocs\YEAR 4\Testing\`.

## Upload and start

Upload `C:\xampp\htdocs\YEAR 4\Testing\outputs\feature-mixed-vm-bundle.tar.gz`
and its `.sha256` companion through Google Cloud SSH's Upload File button.
These commands assume uploads arrive in your home directory.

```bash
cd ~
sha256sum -c feature-mixed-vm-bundle.tar.gz.sha256
mkdir -p ~/forensic-dgp
test ! -e ~/forensic-dgp/feature_vm_bundle && tar -xzf ~/feature-mixed-vm-bundle.tar.gz -C ~/forensic-dgp
tmux new-session -A -s dgp_training
```

Inside tmux, with no other training running:

```bash
cd ~/forensic-dgp/feature_vm_bundle
if [ -d ../venv ]; then source ../venv/bin/activate; fi
bash scripts/setup_feature_mixed_vm.sh && source .venv/bin/activate && bash scripts/run_feature_mixed_vm.sh
```

Setup reuses the existing CUDA PyTorch installation, requires torch >=2.5.1
with compatible torchvision, and installs SAM helper dependencies in an isolated
environment. It does not replace PyTorch. If this check or the new environment's
GPU check fails, stop and send the error; do not start CPU training.
Allow at least 6 GiB free VRAM, 8 GiB available host RAM and 5 GiB free disk.
The GPU execution path cannot be validated on this CPU-only local machine.

Preflight checks CUDA/free VRAM, every bundled file hash, the original split hash,
training-source membership, reviewed split integrity, and one frozen encoder/head
forward pass. No optimizer updates occur in preflight. The real manifest includes
all splits for integrity checking; only its 68 training records enter fitting.
No validation embeddings are packaged or used. The exact original split snapshot
is bundled, so this isolated run does not depend on the VM's other split files.

The run regenerates 68 real and 400 synthetic embeddings on CUDA, then trains
only the two heads for 20 epochs / 1,600 updates. Each batch has two cases from
each of six groups. Encoder weights stay frozen. GPU float32 embeddings can
differ numerically from earlier CPU caches; runtime versions are recorded.
There is no automatic restart/resume or deployment. Preserve an interrupted
directory and use a fresh extraction for a rerun; do not delete old evidence.

Detach with Ctrl+B, then D. Reattach with `tmux attach -t dgp_training`.

## Return results

Successful completion prints `COMPLETE` and creates:

```text
~/forensic-dgp/feature_vm_bundle/feature-mixed-vm-results.tar.gz
```

Use SSH's Download File button with the full absolute path printed by the script.
Place the archive in `C:\xampp\htdocs\YEAR 4\Testing\outputs\`.
The result includes `final_epoch_20.pth`, training history, protocol, source
manifest, cache provenance and environment log. Large feature caches stay on VM.

Next: evaluate the final checkpoint locally on fixed 25 real and 400 synthetic
cases. Report glare, mannequin and source strata. Keep the original real/synthetic
safeguards, thresholds and final-only selection. A passing detector still needs
an end-to-end completion comparison and visual review before application use.
No `best.pth` is selected from training loss.

SAM2 source: facebookresearch/sam2 revision
`2b90b9f5ceec907a1c18123530e92e794ad901a4`, bundled with license notices.
Weights: SAM2.1 Hiera tiny, SHA256
`7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`.
