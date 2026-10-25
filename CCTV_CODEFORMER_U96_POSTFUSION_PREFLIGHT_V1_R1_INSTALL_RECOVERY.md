# R1 installation recovery: preserve the import cache

21 October 2026. This fixes the reported installer directory stop only. It does
not launch the preflight or train a model. The sealed R1 packet stays unchanged.

The user confirmed `research/codeformer_u96_postfusion_preflight_v1_r1/__pycache__`.
The old install command omitted `-B`; its dynamic packet import could write that
cache before the strict namespace check rejected it. Weight copying is after
that check. The cause was reproduced locally using the same packet source;
that fixture used Python3.13, not the VM's Python3.10 runtime.
[Python -B documentation](https://docs.python.org/3.10/using/cmdline.html#cmdoption-B),
[CPython3.10 import implementation](https://raw.githubusercontent.com/python/cpython/3.10/Lib/importlib/_bootstrap_external.py).

`-B` prevents new bytecode writes; it does not remove an existing cache. The
9,173-byte recovery helper validates the exact R1 protocol and 43 nonweight
static files, refuses any copied weights or runtime state, and accepts only one
ordinary packet cache file for the running interpreter. It never loads that
cached code. It records size/hash/inode/owner and moves the cache into a fresh
named sibling evidence directory without deletion. It then invokes only the
original installer with `-B`, with a180-second cap and retained stdout/stderr.
Any unexpected file, partial state or changed source stops the recovery.

The reviewed helper and local filesystem fixture passed. The fixture verified
an extra-file refusal before moving anything, hash/inode preservation, unchanged
sealed files and no repeated cache after a `-B` import. No VM installation or
recovery has run here; these checks are not Linux/L4 feasibility or model quality.

## 1. Upload in Windows Google Cloud SDK Shell

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "recover_u96_r1_install_bytecode_v1.py" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Recover the installation in VM SSH

```bash
python3 -B ~/recover_u96_r1_install_bytecode_v1.py \
  --expected-sha 85cd0c018db8a8d396e1c65bf1ac572bbe523185f54b34d2beff200399c286b0 \
  --recover
```

Expected success includes `"weights_copied":2`, `"installer_exit":0` and
`"preflight_launched":false`. The cache and recovery logs remain in
`~/forensic-dgp/cctv_codeformer_u96_postfusion_preflight_v1_r1_vm_install_bytecode_recovery_v1/`.
Stop and retain output if recovery fails; do not delete or overwrite that evidence.

## 3. Continue the manual preflight

After successful recovery, use steps3 and4 in the
[corrected full VM guide](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_U96_POSTFUSION_PREFLIGHT_V1_R1_CORRECTED_VM.md>)
to open tmux and launch. Step5 has the unchanged three single-source downloads.
Do not re-extract R1 or repeat its old step2. All scientific limits, input/model
bindings, parity/preservation checks and manual-only training rules remain.
The diagnostic still has20 forwards,4 gradient queries and0 optimizer updates.
U96 is unchanged and the full goal remains active and incomplete.
