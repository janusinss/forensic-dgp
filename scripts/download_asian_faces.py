import os
import sys
import io
import urllib.request

def ensure_dependencies():
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("Installing pyarrow for fast parquet extraction...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow", "pillow"])

def download_and_extract_asian_faces(output_dir="dataset/asian_faces", limit=10000):
    ensure_dependencies()
    import pyarrow.parquet as pq
    from PIL import Image
    
    os.makedirs(output_dir, exist_ok=True)
    existing = [f for f in os.listdir(output_dir) if f.endswith(('.jpg', '.png'))]
    if len(existing) >= limit:
        print(f"Dataset already extracted: {len(existing)} images found in {output_dir}")
        return len(existing)
        
    parquet_url = "https://huggingface.co/datasets/hiennguyen9874/face-age-gender-asian/resolve/main/data/train-00000-of-00001.parquet"
    temp_parquet = os.path.join(output_dir, "temp_faces.parquet")
    
    print(f"Downloading Asian Face Dataset (14,760 faces, ~125MB) from Hugging Face...")
    urllib.request.urlretrieve(parquet_url, temp_parquet)
    print("Download complete! Extracting images...")
    
    table = pq.read_table(temp_parquet)
    image_col = table["image"]
    
    count = 0
    total = min(len(image_col), limit)
    
    for idx in range(total):
        item = image_col[idx].as_py()
        # item is either bytes or dict with 'bytes'
        img_bytes = item['bytes'] if isinstance(item, dict) and 'bytes' in item else item
        if isinstance(img_bytes, bytes):
            out_path = os.path.join(output_dir, f"asian_face_{idx:05d}.jpg")
            with open(out_path, "wb") as f:
                f.write(img_bytes)
            count += 1
            if count % 1000 == 0 or count == total:
                print(f"Extracted {count}/{total} Asian face images...")
                
    # Clean up parquet file to save disk space
    if os.path.exists(temp_parquet):
        os.remove(temp_parquet)
        
    print(f"SUCCESS: Successfully saved {count} Asian face images to {output_dir}")
    return count

if __name__ == "__main__":
    out = "dataset/asian_faces"
    if len(sys.argv) > 1:
        out = sys.argv[1]
    download_and_extract_asian_faces(out)
