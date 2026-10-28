import argparse
import tarfile
import urllib.request
import zipfile
from pathlib import Path
import kagglehub
import numpy as np

def download_and_extract(url: str | None, kaggle_name: str | None, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = (url.split("/")[-1] if url else kaggle_name.split("/")[-1]) or "dataset.zip"
    filepath = dest_dir / filename
    if url:
        urllib.request.urlretrieve(url, filepath)
    else:
        kagglehub.dataset_download(kaggle_name, output_dir=str(filepath))

    if filepath.name.endswith(".zip"):
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            zip_ref.extractall(dest_dir)
    elif filepath.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(filepath, "r:gz") as tar_ref:
            tar_ref.extractall(dest_dir)

def main() -> None:
    parser = argparse.ArgumentParser(description="Aether Dataset Downloader and Generator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", type=str, help="URL of archive to download")
    group.add_argument("--kagglehub", type=str, help="Dataset identifier from Kaggle")
    parser.add_argument("--dest", type=str, default="data/dataset", help="Destination folder")
    args = parser.parse_args()

    dest_path = Path.cwd() / args.dest
    download_and_extract(args.url, args.kagglehub, dest_path)

if __name__ == "__main__":
    main()