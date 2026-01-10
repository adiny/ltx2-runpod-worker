#!/usr/bin/env python3
"""Download LTX-2 model files directly from HuggingFace"""

import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL_PATH = "/models/LTX-2"
HF_REPO = "Lightricks/LTX-2"


def download_file(args):
    url, dest_path = args
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        response = requests.get(url, stream=True, timeout=600)
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):
                f.write(chunk)
        return True, os.path.basename(dest_path)
    except Exception as e:
        return False, str(e)


def get_all_files(path=""):
    url = f"https://huggingface.co/api/models/{HF_REPO}/tree/main"
    if path:
        url = f"{url}/{path}"
    
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return []
    except:
        return []
    
    files = []
    for item in resp.json():
        if item['type'] == 'file':
            if not item['path'].endswith('.md') and '.git' not in item['path']:
                files.append(item['path'])
        elif item['type'] == 'directory':
            files.extend(get_all_files(item['path']))
    return files


def main():
    print(f"📥 Downloading {HF_REPO}...")
    
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    files = get_all_files()
    print(f"   Found {len(files)} files")
    
    base_url = f"https://huggingface.co/{HF_REPO}/resolve/main"
    downloads = [(f"{base_url}/{f}", os.path.join(MODEL_PATH, f)) for f in files]
    
    completed = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download_file, args): args for args in downloads}
        
        for future in as_completed(futures):
            success, result = future.result()
            if success:
                completed += 1
            else:
                failed += 1
                print(f"   ⚠️ Failed: {result}")
            
            if completed % 10 == 0:
                print(f"   Progress: {completed}/{len(downloads)}")
    
    print(f"✅ Done! ({completed} ok, {failed} failed)")


if __name__ == "__main__":
    main()
