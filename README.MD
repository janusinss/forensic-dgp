# Face restoration app

This app helps you inspect and restore a picture of one face on your Windows computer. It can also make an estimate of a part of the face hidden by a covering, such as a mask. The finished images are 256 × 256 pixels.

**Please read:** This is experimental research software. A sharper image or a generated face is not proof of someone's identity or what was hidden. If the face is too blurry or covered to see its features, use a clearer or less-covered picture.

## What you need

- A **64-bit Windows** computer. A graphics card is not required.
- **64-bit Python 3.12 or 3.13** and **Git for Windows**. Installation links are below.
- An internet connection for the first setup. After setup, the app processes images on your computer.
- At least **2.5 GiB of free memory (RAM)** when starting a restoration. Close other programs if needed. This is a required check, not a guarantee that every computer with that amount will run the app.
- Free storage space for the app, its programs, and model files. The model files alone use about **720 MiB**.

## Set up the app for the first time

### 1. Install Python and Git

1. Download and install [Python for Windows](https://www.python.org/downloads/windows/). Choose **Python 3.12 or 3.13, 64-bit**. During installation, enable the Python launcher or choose **Add Python to PATH**.
2. Download and install [Git for Windows](https://git-scm.com/download/win). Its default installation options are suitable.

### 2. Download the app

1. Open the Windows **Start** menu, type **PowerShell**, and open it.
2. Copy the following lines into PowerShell. Press **Enter** after each line:

   ```powershell
   git clone --depth 1 --single-branch --branch IMPLEMENTATION https://github.com/janusinss/forensic-dgp.git
   cd forensic-dgp
   ```

This creates a folder named `forensic-dgp` and opens it in PowerShell.

### 3. Install the app's files

In the same PowerShell window, run:

```powershell
.\install.bat
```

Keep the window open and wait for the installer to finish. It downloads the programs and model files needed for both restoration and the optional covering-removal feature. Follow any prompt to press a key when it finishes. If you see an error, read the message and check [Common problems](#common-problems).

## Open the app

In the same PowerShell window, run:

```powershell
.\start.bat
```

The app should open in your browser. Its usual address is [http://127.0.0.1:8000](http://127.0.0.1:8000). If the browser does not open, copy the **Application URL** shown in PowerShell into your browser.

**Keep the PowerShell window open while using the app.** To close the app, return to that window, press **Ctrl+C**, and wait for it to stop.

### Open it again later

Open PowerShell, change to the `forensic-dgp` folder you downloaded, and run `.\start.bat`. For example, if you downloaded the folder in your Windows user folder:

```powershell
cd "$env:USERPROFILE\forensic-dgp"
.\start.bat
```

If you placed it somewhere else, use that folder's path after `cd` instead.

## Prepare a picture

Choose a picture that contains **one face**, facing forward or turned slightly. Crop a larger picture so that it contains just that face before uploading. You should be able to see the person's facial features.

The picture must be a PNG or JPEG, no larger than **10 MiB**, and no wider or taller than **1600 pixels**. It must contain no more than **2,560,000 pixels** in total and must not be animated.

## Restore a face

1. In **Inspect the original**, upload the picture and look at it. If you can clearly make out the face, select **One face with readable visible facial structure**. If you cannot, use a clearer picture.
2. Under **Choose restoration**, select **Auto** for the app's suggestion, **On** to request restoration, or **Off** to keep a resized copy without running restoration. Auto can decide to keep the original instead of restoring it.
3. For the easiest start, leave input preparation on **Bypass**. This uses the picture as uploaded. If you want to adjust the surroundings first, see [Optional: prepare the picture](#optional-prepare-the-picture).
4. Click **Restore selected input**. Compare the original, the selected input, and the result. Use **Download PNG** under an image to save it. You can also download the evidence ZIP, which contains the original and available records and results.

Restoration can change visible facial features. Compare the result carefully with the original.

### Optional: prepare the picture

Choose **Optional manual Protect + Apply** if you want to add blur to selected surroundings before restoration:

1. Paint **Protect** in green over the face and its existing soft edges.
2. Paint **Apply** in blue only over surroundings where added blur is acceptable.
3. Click **Preview permitted surroundings**. Review the preview and check the confirmation box. If you edit either area, preview and confirm again.

During this preparation step, protected areas and areas outside Apply stay unchanged. Restoration itself can still change the face. There is no automatic blur matching.

## Optional: review a covering

Use this after restoration only if something such as a mask, sunglasses, hand, or hair is hiding part of the face. If there is no covering to remove, keep the restoration result.

1. Click **Review covering removal**. Orange marks the area the app proposes to replace.
2. Check the area carefully. Paint over any missed covering and erase any orange covering visible facial features, clear glasses, or hair that is not blocking the face. The automatic suggestion can miss a covering entirely.
3. Read and check both confirmation boxes if they are true, then click **Generate one R1 estimate**. If too much of the face is hidden, use a clearer or less-covered picture instead.
4. Compare the original, restoration, selected area, and estimate. Download the PNG images or the evidence ZIP if needed.

The app changes only the selected area during this covering step. Everything outside it stays the same as the **restoration result**, which may already differ from the original picture. Hidden features are guesses and cannot be verified from the covered picture. An empty selected area skips the completion model.

## Common problems

- **Python was not found:** Install 64-bit Python 3.12 or 3.13, enable the Python launcher or PATH, and run `.\install.bat` again.
- **Installation failed:** Check the error in PowerShell, your internet connection, and available storage. Run `.\install.bat` again. Existing model files that do not match are kept for inspection. For local completion model files, see [completion installation details](COMPLETION_INSTALL.md).
- **The app says there is not enough memory:** Close other programs and try again.
- **The browser did not open:** Open the **Application URL** printed in PowerShell. If port 8000 is already in use, run `.\start.bat --port 0` and use the printed address.

## Your pictures and updates

The app runs on your computer. It saves restoration records in `scratch/implementation_requests` and covering-removal records in `scratch/completion_requests` inside the app folder. Downloaded images and evidence ZIPs can contain private pictures. Delete your own records and downloads when you no longer need them.

To update the app, stop it first. In PowerShell, open the app folder and run:

```powershell
git pull --ff-only origin IMPLEMENTATION
.\install.bat
.\start.bat
```

Save any changes you made to the app's files before updating. If the update reports a conflict, resolve it before running the installer.

## More information and license

The restoration and covering models are experimental and not qualified for identifying people. See the [restoration model card](MODEL_CARD.md) and [completion model card](COMPLETION_MODEL_CARD.md) for their limits. The app uses adaptations of S-Lab's CodeFormer; its **non-commercial terms** apply. Keep the [CodeFormer license](third_party/codeformer/LICENSE), [notices](third_party/codeformer/NOTICE.md), [completion license](third_party/codeformer_completion/LICENSE), and [completion notices](third_party/codeformer_completion/NOTICE.md). Training datasets are not included.
