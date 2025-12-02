from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import BinaryIO, Optional


def _safe_extract_member(zf: zipfile.ZipFile, member: str, dest_dir: str) -> Optional[str]:
    """Extract a single member from the ZipFile into dest_dir safely (prevents path traversal).

    Returns the absolute path to the extracted file, or None on failure or if the member is unsafe.
    """
    dest_dir_path = Path(dest_dir)
    target_path = dest_dir_path.joinpath(member)

    try:
        target_resolved = target_path.resolve()
        dest_resolved = dest_dir_path.resolve()
    except Exception:
        return None

    # Prevent path traversal: the target must reside inside dest_dir
    if not (target_resolved == dest_resolved or dest_resolved in target_resolved.parents):
        return None

    # If the member is a directory, create it and return
    if member.endswith("/") or member.endswith("\\"):
        target_path.mkdir(parents=True, exist_ok=True)
        return str(target_path.resolve())

    # Ensure parent directory exists
    parent = target_path.parent
    if parent:
        parent.mkdir(parents=True, exist_ok=True)

    # Stream extract the file to disk
    try:
        with zf.open(member, "r") as src, open(target_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return str(target_path.resolve())
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
    temp_dir = Path(tempfile.mkdtemp(prefix="rocrate_zip_"))
    temp_zip_path = temp_dir / "archive.zip"

    # Write incoming stream to a temporary zip file on disk
    try:
        with temp_zip_path.open("wb") as out_file:
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
        with zipfile.ZipFile(str(temp_zip_path), "r") as zf:
            namelist = zf.namelist()

            # Exact name matches first
            found: set[str] = set()
            for target in target_names:
                if target in namelist:
                    path = _safe_extract_member(zf, target, temp_dir)
                    if path:
                        extracted_paths.append(path)
                        found.add(Path(target).name.lower())

            # If we've already found all targets, return early to avoid extra extraction
            if len(found) == len({Path(t).name.lower() for t in target_names}):
                return extracted_paths

            # Case-insensitive suffix fallback for targets not yet found
            remaining = [
                t for t in target_names
                if Path(t).name.lower() not in found
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
