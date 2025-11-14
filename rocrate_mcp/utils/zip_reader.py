from __future__ import annotations
from typing import BinaryIO, List, Optional
import zipfile
import io


def find_file_in_zip_stream(stream: BinaryIO, target_names: List[str]) -> Optional[bytes]:
    """Read a zip file from a binary stream and return bytes of the first matching target file name.

    This loads the zip into memory; for very large blobs a streaming approach would be needed.
    """
    # Ensure stream is at start
    try:
        data = stream.read()
    except Exception:
        return None
    bio = io.BytesIO(data)
    with zipfile.ZipFile(bio) as zf:
        for name in target_names:
            if name in zf.namelist():
                with zf.open(name) as f:
                    return f.read()
    # fallback: case-insensitive search
    with zipfile.ZipFile(bio) as zf:
        for entry in zf.namelist():
            for name in target_names:
                if entry.lower().endswith(name.lower()):
                    with zf.open(entry) as f:
                        return f.read()
    return None
