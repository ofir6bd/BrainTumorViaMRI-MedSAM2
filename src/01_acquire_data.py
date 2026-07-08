import synapseclient
import synapseutils
import os
import zipfile
import tarfile
import json
import yaml

with open('secrets.json', 'r') as f:
    config = json.load(f)

with open('config.yaml', 'r') as f:
    _cfg = yaml.safe_load(f)

cache_dir = _cfg["paths"]["raw"]
dataset_dir = _cfg["paths"]["extract_to"]

def download():
    if os.path.exists(cache_dir):
        print("Already downloaded")
        return

    print("Downloading...")
    syn = synapseclient.Synapse()
    syn.login(authToken=config["synapse_auth_token"])
    synapseutils.syncFromSynapse(syn, config["synapse_dataset_id"], path=cache_dir)
    print("Download complete")

def extract():
    if os.path.exists(dataset_dir):
        print("Already extracted")
        return

    print("Extracting...")
    os.makedirs(dataset_dir, exist_ok=True)

    archive_extensions = [".zip", ".tar", ".tar.gz", ".tar.bz2", ".gz", ".7z", ".rar"]

    for root, _, files in os.walk(cache_dir):
        for file in files:
            filepath = os.path.join(root, file)

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
