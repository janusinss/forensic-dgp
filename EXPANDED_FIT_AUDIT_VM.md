# Existing-cache fit audit — inference only

The full expanded feature arrays remain on the VM; the returned archive contains
cache metadata only. Do not infer per-stratum fit from aggregate scores. This
script reads both completed caches and final heads, performs inference at the
unchanged .5 thresholds, and reports covering/degradation/source groups.

It verifies checkpoint/code/cache provenance, per-row checksums, unchanged head
weights and exact aggregate count agreement with the original VM report. No
optimizer is created, no encoder feature extraction or training occurs, and no
validation/test data is read. If aggregate counts differ, it stops for inspection;
do not relax this check or rerun training to work around it.

Two local tests passed: aggregation/count denominators and CPU refusal before
workspace access. Python syntax passes. Actual GPU audit remains pending.

Upload local file `C:\xampp\htdocs\YEAR 4\Testing\scripts\audit_expanded_fit_vm.py`
through Google Cloud SSH. It should land at `~/audit_expanded_fit_vm.py`.
Run:

```bash
cd ~/forensic-dgp/expanded_feature_bundle &&
source ../feature_vm_bundle/.venv/bin/activate &&
python -u ~/audit_expanded_fit_vm.py --root "$PWD"
```

After DONE, download using Google Cloud SSH's Download File dialog:

```text
/home/janusdominic0/forensic-dgp/expanded_feature_bundle/expanded-fit-audit-results.tar.gz
```

Save locally to `C:\xampp\htdocs\YEAR 4\Testing\outputs\expanded-fit-audit-results.tar.gz`.
The output folder `~/forensic-dgp/expanded_feature_bundle/outputs/expanded_fit_audit/`
must be new; partial outputs remain for diagnosis. Existing training outputs and
application checkpoints are untouched. Do not delete caches before this audit.

Next: compare training fit across predeclared strata to distinguish poor fit from
generalization failure. This cannot itself qualify a model: fixed real/synthetic
selection safeguards and reviewed end-to-end completion improvement still apply.
The separate tiny-highlight scope question remains pending; proposed labels stay
disabled. No new training is authorized by this diagnostic command.
