# New-account L4 VM setup — 12 October 2026

Status: manual setup plan only. The user reports $300 credit in another Google
account and has not selected a new project. No new project, billing activation,
VM, transfer, inference, gradient query or training run was performed by the agent.
Google Drive setup is deferred while the training host is prepared.

## Billing prerequisite

Google blocks GPU VMs and quota increases on non-billable Free Trial accounts.
Activating Paid billing retains eligible unused Welcome credit until its original
expiry. Usage not covered by remaining credit becomes billable. Review the
account's credit balance, expiry and billing status yourself before activation.
Google's Welcome-credit eligibility requires no previous trial; an additional
email address does not by itself establish new eligibility.

Sources: [Free Trial restrictions and eligibility](https://docs.cloud.google.com/free/docs/free-cloud-features),
[Billing activation and retained credit](https://cloud.google.com/signup-faqs).

## Prepare the project

1. Sign into the new account in a normal Chrome or Edge window at
   [Google Cloud console](https://console.cloud.google.com/).
2. Review Billing and activate Paid billing if the account is still Free Trial
   and you accept its billing terms. Payment details and confirmations stay with
   the human user.
3. Create a project with display name `Forensic DGP Training`. Record its actual
   generated project ID; the display name is not the ID.
4. Enable [Compute Engine API](https://console.cloud.google.com/apis/library/compute.googleapis.com)
   with the new project selected and linked to the intended billing account.
5. Review [Quotas & System Limits](https://console.cloud.google.com/iam-admin/quotas).
   Require at least one available standard NVIDIA L4 GPU in `us-central1` and
   at least one available `GPUs (all regions)` quota. The current G2 standard
   quota is listed as `GPU_FAMILY:NVIDIA_L4`; request a quota increase where needed.

Source: [Current GPU quotas](https://docs.cloud.google.com/compute/resource-usage#gpu_quota).
Quota is not a guarantee of available physical capacity.

## Create the proposed host

In Compute Engine > VM instances > Create instance, use these proposed settings.
They are not a claim that a new VM already exists.

| Compute setting | Proposed value |
|---|---|
| Instance name | `forensic-dgp-l4-v2` |
| Region / zone | `us-central1` / `us-central1-a` |
| Machine type | `g2-standard-4` — 4 vCPU, 16 GB host RAM |
| GPU | One NVIDIA L4 — 24 GB GPU memory |
| Provisioning | Standard, with terminate-on-host-maintenance policy |

| Storage / OS setting | Proposed value |
|---|---|
| OS | Plain Ubuntu 22.04 LTS, x86/64 |
| Boot disk | Balanced Persistent Disk, 200 GiB |
| Boot-disk deletion rule | Keep disk if the VM is deleted |
| HTTP / HTTPS firewall | Leave unchecked; training uses SSH |
| Startup script | Empty; do not paste a historical training launcher |

The larger persistent disk is a proposed allocation for retained research caches,
selected code/checkpoints/data and future bounded outputs. It is not a promise
that every local output directory should be mirrored to the VM. Review the
console's cost estimate before clicking Create. Retained disks continue to incur
storage charges; alerts-only budgets do not automatically stop the VM.

Sources: [G2 hardware and supported disks](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines#g2_series),
[G2 creation](https://docs.cloud.google.com/compute/docs/gpus/create-gpu-vm-g-series),
[GPU zones](https://docs.cloud.google.com/compute/docs/regions-zones/gpu-regions-zones),
[Budget alerts](https://docs.cloud.google.com/billing/docs/how-to/budgets).

Use the plain Ubuntu image here. G2 instances do not support Deep Learning VM
boot images according to the current [GPU creation overview](https://docs.cloud.google.com/compute/docs/gpus/create-vm-with-gpus).
Driver installation is a separate step after host identity and security settings
are known. No protection should be disabled merely to make an installer run.

## First SSH verification

Once running, open its browser SSH connection. Run these read-only commands:

```bash
whoami
```

```bash
python3 --version
```

```bash
df -h / "$HOME"
```

```bash
nvidia-smi
```

Send the actual project ID, instance name, zone and these command outputs.
On a plain Ubuntu image, a missing `nvidia-smi` indicates driver setup is still
needed; it does not establish a failed training run.

Then prepare the driver/runtime and restoration transfer for that verified host.
Use [Google's driver installer instructions](https://docs.cloud.google.com/compute/docs/gpus/install-drivers-gpu)
and the retained environment metadata rather than replacing dependencies blindly.
The Windows virtual environment is not a Linux runtime.

## Migration and training boundary

The existing host and original research assets are retained. The historical local
cache backup is a cache backup, not a complete VM image. New transfers must bind
the exact new account/project/instance/zone/SSH user and restore preserved source,
data, checkpoints, scientific caches, split membership and failure evidence.
Require fresh hash and runtime verification before any pilot.

New training remains human-manual in tmux on an independently verified NVIDIA L4
at `~/forensic-dgp`, with finite protocol/timing/stop rules. The pending signed
diagnostic must not be launched automatically or assumed portable without auditing
its root bindings. No historical pilot is selected by this setup guide.
The DGP research goal and all unfinished quality/app/covering requirements remain.
