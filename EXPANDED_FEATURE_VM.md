# Matched placement experiment on the existing VM

The standalone bundle uses the already working CUDA environment at
`~/forensic-dgp/feature_vm_bundle/.venv/`. It creates a separate checkout at
`~/forensic-dgp/expanded_feature_bundle/`; previous experiments remain intact.
Local source workspace: `C:\xampp\htdocs\YEAR 4\Testing\`.
No local fitting is permitted. No existing application checkpoint is replaced.

## Upload and verify

Upload these two files with Google Cloud SSH's Upload File button:

- `C:\xampp\htdocs\YEAR 4\Testing\outputs\expanded-feature-vm-bundle.tar.gz`
- `C:\xampp\htdocs\YEAR 4\Testing\outputs\expanded-feature-vm-bundle.tar.gz.sha256`

Paste in SSH:

```bash
cd ~ &&
sha256sum -c expanded-feature-vm-bundle.tar.gz.sha256 &&
test ! -e ~/forensic-dgp/expanded_feature_bundle &&
tar -xzf expanded-feature-vm-bundle.tar.gz -C ~/forensic-dgp &&
tmux new-session -A -s dgp_training
```

Inside tmux:

```bash
cd ~/forensic-dgp/expanded_feature_bundle &&
source ../feature_vm_bundle/.venv/bin/activate &&
bash scripts/run_expanded_feature_vm.sh
```

This bundle contains the required code, source photos, reviewed masks, parent
checkpoints and frozen encoder. No `git pull` is required for this snapshot.
Checksum file uses LF line endings. Do not repeat extraction over an existing
bundle or delete an existing experiment to bypass its refusal.

## What runs

Preflight requires CUDA, at least 6 GiB free GPU memory and 35 GiB free disk.
It checks inventory/parent hashes, split membership, identical arm membership,
current matched-data code hashes, and one frozen-encoder/head forward. It makes
zero optimizer updates. GPU compatibility remains unverified until this passes
on the VM. The existing environment must provide SAM2's required dependencies.

After preflight: 7,080 frozen-encoder feature extractions across two disk-backed
caches, followed by 1,600 optimizer updates for each arm. Cache progress prints
every 50 examples; training reports every epoch. The fixed and anatomical arms
share sources, texture RNG, camera recipe, initialization and sampling schedule.
Both context pixel and spatial presence heads train; SAM2 stays frozen.
See `EXPANDED_FEATURE_DATA.md` for the predeclared design and limitations.

Only final epoch-20 checkpoints are saved. Training-fit metrics are diagnostic;
passing them does not select a deployable model. Held-out evaluation and preview
review happen locally after return, with the existing real/synthetic safeguards.
Known mannequin and glare cases must be reported separately. Completion quality
still requires end-to-end evidence before any promotion.

Partial caches/output directories are preserved and refused for automatic reruns.
If execution fails, retain the error and current output; inspect before deciding
whether a cache can be recovered. No blind retry or automatic restart is built in.

## Download after DONE

Paste this absolute path into Google Cloud SSH's Download File dialog:

```text
/home/janusdominic0/forensic-dgp/expanded_feature_bundle/expanded-feature-results.tar.gz
```

Save it locally as:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\expanded-feature-results.tar.gz`.
It contains both heads' final weights, protocol, metrics, cache-row provenance,
source manifest and inventory. Large feature arrays stay on the VM.
Keep `expanded_training.log` and `expanded_environment.txt` on failure for diagnosis.
