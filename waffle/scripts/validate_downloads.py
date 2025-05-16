#!/usr/bin/env python3
"""
Validate downloaded data files using MD5 checksums.
"""

import argparse
from pathlib import Path

import rootutils
from loguru import logger

# Setup root for imports
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from waffle.utils.file_validation import validate_downloaded_files


def main():
    parser = argparse.ArgumentParser(description="Validate downloaded data files using MD5 checksums")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Base directory where data is stored (e.g., /path/to/data)"
    )
    parser.add_argument(
        "--dataset",
        default="AbRank",
        help="Dataset name (default: AbRank)"
    )
    parser.add_argument(
        "--md5sum-file",
        help="Path to custom md5sum.txt file. If not provided, will use <data-dir>/<dataset>/raw/md5sum.txt"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    dataset_raw_dir = data_dir / args.dataset / "raw"

    if not dataset_raw_dir.exists():
        logger.error(f"Dataset raw directory not found: {dataset_raw_dir}")
        exit(1)

    md5sum_path = args.md5sum_file if args.md5sum_file else dataset_raw_dir / "md5sum.txt"

    if not Path(md5sum_path).exists():
        logger.error(f"MD5 checksum file not found: {md5sum_path}")
        exit(1)

    logger.info(f"Validating files in {dataset_raw_dir} using {md5sum_path}")

    validated, missing, corrupted = validate_downloaded_files(
        download_dir=str(dataset_raw_dir),
        md5sum_path=str(md5sum_path)
    )

    # Print summary
    logger.info(f"Validation summary:")
    logger.info(f"  - {len(validated)} files validated successfully")

    if missing:
        logger.warning(f"  - {len(missing)} files missing:")
        for file in missing:
            logger.warning(f"    - {file}")

    if corrupted:
        logger.error(f"  - {len(corrupted)} files corrupted:")
        for file in corrupted:
            logger.error(f"    - {file}")

    # Exit with error code if issues found
    if missing or corrupted:
        logger.error("Validation failed. Please re-download the problematic files.")
        exit(1)
    else:
        logger.info("All files validated successfully!")
        exit(0)


if __name__ == "__main__":
    main()