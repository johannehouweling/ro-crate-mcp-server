from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from typing import BinaryIO


def _safe_extract_member(zf: zipfile.ZipFile, member: str, dest_dir: str) -> None|str:
    """Extract a single member from the ZipFile into dest_dir safely (prevents path traversal).

    Returns the absolute path to the extracted file, or None on failure or if the member is unsafe.
    """
    # Build normalized target path and normalized destination directory
    target_path = os.path.normpath(os.path.join(dest_dir, member))
    dest_dir_norm = os.path.normpath(dest_dir)

    # Prevent path traversal: the target must reside inside dest_dir
    if not (target_path == dest_dir_norm or target_path.startswith(dest_dir_norm + os.sep)):
        return None

    # If the member is a directory, create it and return
    if member.endswith("/") or member.endswith("\\"):
        os.makedirs(target_path, exist_ok=True)
        return os.path.abspath(target_path)

    # Ensure parent directory exists
    parent = os.path.dirname(target_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # Stream extract the file to disk
    try:
        with zf.open(member, "r") as src, open(target_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return os.path.abspath(target_path)
    except Exception:
        return None


def extract_files_from_zip_stream(stream: BinaryIO, target_names: list[str]) -> list[str]:
    """Stream a zip archive from a binary stream to disk, extract matching target files into a temp folder,
    and return the absolute file paths of extracted targets.

    Behavior:
    - The incoming stream is written to a temporary file on disk (streaming; avoids loading the entire archive into memory).
    - A temporary directory is created to hold both the written archive and the extracted files. The directory is
      intentionally left in place so callers can access the extracted files; callers are responsible for cleanup.
    - Matching is attempted first by exact entry name, then by case-insensitive suffix match.
    - The returned list preserves discovery order and may be empty if no matches were found.
    """
    temp_dir = tempfile.mkdtemp(prefix="rocrate_zip_")
    temp_zip_path = os.path.join(temp_dir, "archive.zip")

    # Write incoming stream to a temporary zip file on disk
    try:
        with open(temp_zip_path, "wb") as out_file:
            shutil.copyfileobj(stream, out_file)
    except Exception:
        # Attempt cleanup if write fails, then re-raise
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass
        raise

    extracted_paths: list[str] = []
    try:
        with zipfile.ZipFile(temp_zip_path, "r") as zf:
            namelist = zf.namelist()

            # Exact name matches first
            found: set[str] = set()
            for target in target_names:
                if target in namelist:
                    path = _safe_extract_member(zf, target, temp_dir)
                    if path:
                        extracted_paths.append(path)
                        found.add(os.path.basename(target).lower())

            # If we've already found all targets, return early to avoid extra extraction
            if len(found) == len({os.path.basename(t).lower() for t in target_names}):
                return extracted_paths

            # Case-insensitive suffix fallback for targets not yet found
            remaining = [
                t for t in target_names
                if os.path.basename(t).lower() not in found
            ]
            if remaining:
                for entry in namelist:
                    for target in list(remaining):
                        if entry.lower().endswith(target.lower()):
                            path = _safe_extract_member(zf, entry, temp_dir)
                            if path:
                                extracted_paths.append(path)
                                remaining.remove(target)
                    if not remaining:
                        break
    except Exception:
        # Leave temp_dir in place for debugging and re-raise so caller knows extraction failed
        raise

    return extracted_paths

