# 1DataAcquisition.py
import synapseclient
import synapseutils
import os
import zipfile
import tarfile
import json

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

cache_dir = "1_.synapsecache"
dataset_dir = "2_BraTS2024_dataset"

def download():
    """Download BraTS dataset"""
    if os.path.exists(cache_dir):
        print("Already downloaded")
        return
    
    print("Downloading...")
    syn = synapseclient.Synapse()
    syn.login(authToken=config["synapse_auth_token"])
    synapseutils.syncFromSynapse(syn, config["synapse_dataset_id"], path=cache_dir)
    print("Download complete")

def extract():
    """Extract archives"""
    if os.path.exists(dataset_dir):
        print("Already extracted")
        return
    
    print("Extracting...")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Common archive extensions
    archive_extensions = [".zip", ".tar", ".tar.gz", ".tar.bz2", ".gz", ".7z", ".rar"]
    
    for root, _, files in os.walk(cache_dir):
        for file in files:
            filepath = os.path.join(root, file)
            
            # Check if it's an archive file
            if any(file.lower().endswith(ext) for ext in archive_extensions):
                try:
                    if file.endswith(".zip"):
                        with zipfile.ZipFile(filepath, 'r') as z:
                            z.extractall(dataset_dir)
                        print(f"Extracted: {file}")
                    elif file.endswith((".tar", ".tar.gz", ".tar.bz2")):
                        with tarfile.open(filepath, 'r:*') as t:
                            t.extractall(dataset_dir)
                        print(f"Extracted: {file}")
                    else:
                        print(f"Skipped unsupported format: {file}")
                except Exception as e:
                    print(f"Failed to extract {file}: {e}")
    
    print("Extraction complete")

if __name__ == "__main__":
    download()
    extract()