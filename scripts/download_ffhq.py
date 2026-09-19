import os
import sys
import shutil

def ensure_dependencies():
    try:
        import kagglehub
    except ImportError:
        print("Installing kagglehub for automated dataset retrieval...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])

def download_and_extract_ffhq(output_dir="dataset/thumbnails128x128", limit=None):
    """
    Downloads the GREATGAMEDOTA FFHQ Face Dataset from Kaggle and standardizes
    it in the target directory for training and evaluation.
    """
    os.makedirs(output_dir, exist_ok=True)
    existing = [f for f in os.listdir(output_dir) if f.endswith(('.png', '.jpg'))]
    if len(existing) >= 5000:
        print(f"FFHQ dataset already present: {len(existing)} images found in {output_dir}")
        return len(existing)

    ensure_dependencies()
    import kagglehub

    print("Downloading GREATGAMEDOTA FFHQ Face Dataset (greatgamedota/ffhq-face-data-set)...")
    downloaded_path = kagglehub.dataset_download("greatgamedota/ffhq-face-data-set")
    print(f"Downloaded to cache: {downloaded_path}")

    # Search for thumbnails128x128 directory or images
    source_candidates = [
        os.path.join(downloaded_path, "thumbnails128x128"),
        os.path.join(downloaded_path, "images1024x1024"),
        downloaded_path
    ]

    source_dir = None
    for cand in source_candidates:
        if os.path.exists(cand) and any(f.endswith(('.png', '.jpg')) for f in os.listdir(cand)):
            source_dir = cand
            break

    if not source_dir:
        print(f"Error: Could not find face images inside {downloaded_path}")
        return 0

    files = [f for f in os.listdir(source_dir) if f.endswith(('.png', '.jpg'))]
    if limit:
        files = files[:limit]

    print(f"Transferring {len(files)} images to {output_dir}...")
    for idx, fname in enumerate(files):
        src_file = os.path.join(source_dir, fname)
        dst_file = os.path.join(output_dir, fname)
        if not os.path.exists(dst_file):
            shutil.copyfile(src_file, dst_file)
        if (idx + 1) % 5000 == 0 or (idx + 1) == len(files):
            print(f"Transferred {idx + 1}/{len(files)} images...")

    print(f"SUCCESS: {len(files)} images ready in {output_dir}")
    return len(files)

if __name__ == "__main__":
    out = "dataset/thumbnails128x128"
    if len(sys.argv) > 1:
        out = sys.argv[1]
    download_and_extract_ffhq(out)
