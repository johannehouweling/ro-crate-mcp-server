from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from typing import BinaryIO


def _write_stream_to_tempfile(stream: BinaryIO, prefix: str = "rocrate_zip_") -> str:
    """Write incoming binary stream to a temporary file and return its path.

    The caller is responsible for removing the temporary directory if they need to.
    """
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    temp_path = os.path.join(temp_dir, "archive.zip")
    with open(temp_path, "wb") as out_file:
        shutil.copyfileobj(stream, out_file)
    return temp_path


def list_zip_contents(stream: BinaryIO) -> list[dict]:
    """Return a list of zip entry metadata for a zip archive provided as a binary stream.

    Each entry is a dict with keys: name (str), size (int), is_dir (bool).
    The archive is written to a temporary directory which is cleaned up before returning.
    """
    temp_path = _write_stream_to_tempfile(stream)
    entries: list[dict] = []
    try:
        with zipfile.ZipFile(temp_path, "r") as zf:
            for info in zf.infolist():
                entries.append({
                    "name": info.filename,
                    "size": info.file_size,
                    "is_dir": info.is_dir(),
                })
    finally:
        # Remove the temporary directory and file
        try:
            shutil.rmtree(os.path.dirname(temp_path))
        except Exception:
            pass
    return entries


def read_file_from_zip_stream(stream: BinaryIO, entry_name: str, max_bytes: int | None = None) -> bytes | None:
    """Read and return the contents of a single entry from a zip archive stream.

    - entry_name is matched exactly against the archive's members. If not found, None is returned.
    - If max_bytes is provided the read will be limited to that many bytes (useful for previews).
    """
    temp_path = _write_stream_to_tempfile(stream)
    try:
        with zipfile.ZipFile(temp_path, "r") as zf:
            try:
                with zf.open(entry_name, "r") as f:
                    data = f.read(max_bytes) if max_bytes is not None else f.read()
                    return data
            except KeyError:
                # Exact entry not found
                return None
    finally:
        try:
            shutil.rmtree(os.path.dirname(temp_path))
        except Exception:
            pass
