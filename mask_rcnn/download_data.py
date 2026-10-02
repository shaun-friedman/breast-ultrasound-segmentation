"""Download the pinned BUSI release from Kaggle into data/ and verify it.

kagglehub downloads and extracts into its own cache (~/.cache/kagglehub, or
$KAGGLEHUB_CACHE). The Dataset_BUSI_with_GT folder is then copied to --dest and
checked: every file must decode and the content fingerprint must match the
pinned release. Public datasets need no Kaggle credentials.

    python mask_rcnn/download_data.py
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import kagglehub

import busi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dest", type=Path, default=busi.DEFAULT_DATA_ROOT)
    parser.add_argument("--force", action="store_true",
                        help="replace an existing copy at --dest (e.g. a damaged one)")
    return parser.parse_args()


def find_dataset_dir(download: Path) -> Path:
    """The folder that directly contains the benign/malignant/normal class folders."""
    for candidate in [download, *sorted(p for p in download.rglob("*") if p.is_dir())]:
        if all((candidate / cls).is_dir() for cls in busi.CLASSES):
            return candidate
    raise FileNotFoundError(f"No folder with {busi.CLASSES} subfolders under {download}")


def verify(data_root: Path) -> None:
    records = busi.discover(data_root)
    busi.verify_decodable(data_root, records)
    fingerprint = busi.dataset_fingerprint(data_root, records)
    if fingerprint != busi.KAGGLE_SHA256:
        raise RuntimeError(
            f"{data_root} does not match {busi.KAGGLE_HANDLE}\n"
            f"  expected sha256 {busi.KAGGLE_SHA256}\n  found    sha256 {fingerprint}\n"
            "Re-run with --force to replace it."
        )
    print(f"Verified {data_root} matches {busi.KAGGLE_HANDLE}")


def main() -> None:
    args = parse_args()
    dest = args.dest.resolve()
    if dest.exists() and not args.force:
        print(f"{dest} already exists; verifying it (use --force to replace)")
        verify(dest)
        return

    download = Path(kagglehub.dataset_download(busi.KAGGLE_HANDLE))
    source = find_dataset_dir(download)

    # Copy to a staging folder first so a failed copy never leaves a partial dataset at dest.
    staging = dest.with_name(dest.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    shutil.copytree(source, staging)
    verify(staging)
    if dest.exists():
        shutil.rmtree(dest)
    staging.rename(dest)
    print(f"Dataset ready at {dest}")


if __name__ == "__main__":
    main()
