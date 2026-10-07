# V26 — thin upload, installation, tmux and download

**Latest research milestone — 6 October 2026: V26 return audited; identity-reference correction proved, structure failure retained.**

The downloaded 325,764,761-byte V26 archive matches SHA256
`9ea6afad52b70a27681630b8b09f96dfe560f6756664b627037198fe9e47ae53`.
Safe import verifies 631 regular files. Independent CPU inference and saved-source/
arithmetic checks pass: 240 assets, two snapshots at 0/50, 100 raw/PNG pairs and
metric rows, 50 original own-DGP forwards and 250 feature arrays. All original
numerical and quality bounds remain unchanged. Twenty-five tamper regressions pass.

Actual L4 preflight confirms exact zero matched identity penalty and all 26 matched
parameter-gradient assertions on all 50 cases, before optimization. The legacy
discrepancy is reproduced. The reference-processing fix therefore works, but it
does not resolve the structure failure: **0.0335635587% delivered degraded gain
against the unchanged 1% requirement at update 50**. Training stops after 50 updates/
51 backwards; final 800 never runs. Export complete:true packages the retained
failure and does not imply training success. Do not repeat V26 unchanged or relax
the stop. Original V22–V25 failures and the separate processing evidence stay intact.

All ten original-size sheets/50 paired photographic TRAIN cases are reviewed;
200 exact cells are independently checked. Eyes, nose, mouth, face outline and
overall visible appearance show no convincing V26 gain. All five own-feature
projection gradients are active and all 26 learned tensors change. Corrections
reach output, but the median degraded raw change is only 0.1348669090 byte level.
Raw degraded structure gain is 0.0151795995%, below even the delivered PNG gain;
quantization alone does not explain the failure. Worker 29.077s, fit 9.197s and
supervisor 35.001s pass timing bounds. Torch allocated peak VRAM is 1,806,989,312
bytes. There is no time-cap or CUDA failure behind the reported structure stop.

The corrected identity calculation is insufficient to establish useful restoration.
Review the corrected objective and spatial response at saved states before choosing
another training recipe. The complete optimizer-trajectory cause is not proved.
No V27 packet, unchanged retry, local gradients/optimization or assistant VM/cloud
action occurs. The user-selected own-DGP spatial/feature direction remains active.

The previous 299 bindings, deeper 697/513 bindings, twelve complete document bodies,
app 22 bindings and concurrent completed VM maintenance are preserved. No candidate
is promoted. Native CCTV stays unpaired; exposed paired TRAIN metrics are separate.
No native/reserved-final/new covering pixels or ethnicity/Zamboanga performance
claim enters this milestone. Previously useful native inputs remain usable despite
model softness. Useful native development output, input-only insufficient-information
handling, candidate app parity/full flow, all seven automatic/assisted covering
families and independent final review remain required. Goal active/incomplete.

[V26 verified result](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BATCHMATCHED_IDENTITY_V26_RESULTS.md>) ·
[Milestone](<C:/xampp/htdocs/YEAR 4/Testing/outputs/dgp_batchmatched_identity_v26_audit_milestone/milestone.json>)

Previous bodies below are preserved history. Their V26-pending statements and manual
V22–V26/diagnostic launch commands are historical. V26 is closed as a failure;
do not use those commands to repeat immutable historical pilots.


6 October 2026. **Use the new49 KB packet below. Keep failed V25 and its files.**
V26 changes the identity reference calculation. Training starts only after its
L4 preflight proves zero identity penalty and gradient on identical baseline outputs.
Original structure/preservation/time stops remain. No model weights need uploading.

Protocol SHA256: `f9ccf86ee8e87ca87d43e849df7c0deae8e511e0db1a175b86ab279ec6964994`.
Archive SHA256: `2a8a8f343fe6a2a93bfc479fccca924e56bc3f54bb42140c373ff01827a51864`.

1. Upload from **Windows Google Cloud SDK Shell**:

```cmd
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-batchmatched-identity-v26-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-batchmatched-identity-v26-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-batchmatched-identity-v26-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test -d ~/forensic-dgp/cctv_dgp_spatial_features_vm_v25 &&
test ! -e ~/forensic-dgp/cctv_dgp_batchmatched_identity_vm_v26 &&
tar -xzf cctv-dgp-batchmatched-identity-v26-execution.tar.gz -C ~/forensic-dgp &&
python3 ~/forensic-dgp/cctv_dgp_batchmatched_identity_vm_v26/scripts/install_v26.py --root ~/forensic-dgp/cctv_dgp_batchmatched_identity_vm_v26 --protocol-sha f9ccf86ee8e87ca87d43e849df7c0deae8e511e0db1a175b86ab279ec6964994 --install
```

3. Open **tmux** from a regular VM SSH prompt:

```bash
tmux new-session -A -s dgp_batchmatched_identity_v26
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_batchmatched_identity_vm_v26 &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_batchmatched_identity_v26_vm.py --root . --protocol-sha f9ccf86ee8e87ca87d43e849df7c0deae8e511e0db1a175b86ab279ec6964994 --verify-transfer &&
bash scripts/run_v26.sh f9ccf86ee8e87ca87d43e849df7c0deae8e511e0db1a175b86ab279ec6964994
```

5. Download **after the export receipt appears**, from **Windows Google Cloud SDK Shell**:

```cmd
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-batchmatched-identity-v26-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-batchmatched-identity-v26-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-batchmatched-identity-v26-export.json" "."
```

Downloads go to **C:\xampp\htdocs\YEAR 4\Testing\outputs**. Keep the three separate
remote-source calls; Windows PuTTY rejects a combined call.

`Ctrl+B`, then `D` safely detaches. Step3 reattaches. The training log is
~/forensic-dgp/cctv_dgp_batchmatched_identity_vm_v26/trainer.log.
The supervised run includes one actual neural preflight, capped at5 minutes;
its new identity proof has a3-minute sub-limit. Installation is capped at60s,
fitting25 minutes, worker30, supervisor35 plus30s grace, export2 minutes internally/
2.5 externally plus30s grace. An unchanged failed directory is preserved.

Export complete:true means packaging. The process exit, proof, presence flags and
independent review determine what passed. A stop before training can have no
outputs/failure.json: the traceback/log, supervisor exit and missing completed result
still retain that failure. Download stop evidence too; do not rerun it unchanged.
No successful preflight, completed archive or capacity test alone accepts the model.

Actual V26 L4 proof/training, independent return audit and whole-face review remain
pending. Goal active/incomplete.

[Finite design](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BATCHMATCHED_IDENTITY_V26_PLAN.md>) ·
[Audited V25 diagnostic](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V25_GRADIENT_DIAGNOSTIC_V1_RESULTS.md>)
