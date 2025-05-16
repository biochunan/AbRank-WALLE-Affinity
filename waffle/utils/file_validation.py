import hashlib
import os
from pathlib import Path
from typing import Dict, List, Tuple

from loguru import logger


def calculate_md5(file_path: str) -> str:
    """
    Calculate MD5 hash for a file.

    Args:
        file_path (str): Path to the file

    Returns:
        str: MD5 hash of the file
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def read_md5sum_file(md5sum_path: str) -> Dict[str, str]:
    """
    Read md5sum.txt file and return a dictionary of filenames and their expected MD5 hashes.

    Args:
        md5sum_path (str): Path to md5sum.txt file

    Returns:
        Dict[str, str]: Dictionary mapping filenames to MD5 hashes
    """
    md5_dict = {}
    with open(md5sum_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                # Standard md5sum format: <md5> <filename>
                md5_hash = parts[0]
                filename = parts[1]
                # Remove any leading characters like '*' or spaces
                if filename.startswith("*"):
                    filename = filename[1:]
                md5_dict[filename] = md5_hash
    return md5_dict


def validate_downloaded_files(
    download_dir: str, md5sum_path: str
) -> Tuple[List[str], List[str], List[str]]:
    """
    Validate downloaded files using MD5 checksums from md5sum.txt.

    Args:
        download_dir (str): Directory containing the downloaded files
        md5sum_path (str): Path to md5sum.txt file

    Returns:
        Tuple[List[str], List[str], List[str]]:
            Tuple of (validated_files, missing_files, corrupted_files)
    """
    download_dir = Path(download_dir)
    expected_md5s = read_md5sum_file(md5sum_path)

    validated_files = []
    missing_files = []
    corrupted_files = []

    logger.info(f"Validating {len(expected_md5s)} files in {download_dir}")

    for filename, expected_md5 in expected_md5s.items():
        file_path = download_dir / filename

        if not file_path.exists():
            logger.warning(f"File not found: {filename}")
            missing_files.append(filename)
            continue

        logger.info(f"Calculating MD5 for {filename}...")
        actual_md5 = calculate_md5(str(file_path))

        if actual_md5 == expected_md5:
            logger.info(f"✅ {filename} validated successfully")
            validated_files.append(filename)
        else:
            logger.error(f"❌ {filename} is corrupted or tampered")
            logger.error(f"  Expected: {expected_md5}")
            logger.error(f"  Got:      {actual_md5}")
            corrupted_files.append(filename)

    logger.info(f"Validation complete: {len(validated_files)} valid, "
                f"{len(missing_files)} missing, {len(corrupted_files)} corrupted")

    return validated_files, missing_files, corrupted_files


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate downloaded files using MD5 checksums")
    parser.add_argument("--download-dir", required=True, help="Directory containing downloaded files")
    parser.add_argument("--md5sum-file", required=True, help="Path to md5sum.txt file")

    args = parser.parse_args()

    validated, missing, corrupted = validate_downloaded_files(
        args.download_dir, args.md5sum_file
    )

    if missing or corrupted:
        exit(1)  # Error exit code if any files are missing or corrupted
    else:
        exit(0)  # Success exit code if all files validate