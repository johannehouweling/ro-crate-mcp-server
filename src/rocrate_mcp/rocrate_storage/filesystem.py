from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from rocrate_mcp.utils.zip_reader import extract_files_from_zip_stream

from .base import ResourceInfo, StorageBackend


class FilesystemStorageBackend(StorageBackend):
    """
    Filesystem-backed storage backend.

    Args:
        root_dir: Base directory on the server that holds resources. This directory
            will be created if it does not exist. Accepts either a str or pathlib.Path.
        root_prefix: Optional subpath inside root_dir that will act as the logical root
            for all operations (useful to restrict the backend to a subtree). Accepts
            str or pathlib.Path.
        default_suffixes: Default suffix filter used when list_resources is called
            with suffixes=None. Pass an empty list to default to no filtering.
    Behavior:
        - list_resources yields ResourceInfo entries for files under the combined root.
        - get_resource_stream returns an open binary file object for the given locator.
        - Path traversal is prevented by resolving absolute paths and ensuring they
          remain inside the configured root directory.
    """

    def __init__(
        self,
        root_dir: str | Path,
        root_prefix: None | str | Path = None,
        default_suffixes: list[str] | None = None,
    ):
        # store resolved Path for the root directory
        self.root_dir: Path = Path(root_dir).resolve()
        # normalize root_prefix to a string without leading separators
        rp = root_prefix or ""
        if isinstance(rp, Path):
            rp = rp.as_posix()
        self.root_prefix = str(rp).lstrip("/\\")

        # default_suffixes: None means use no default, otherwise a list (may be empty)
        if default_suffixes is None:
            self.default_suffixes: list[str] | None = [".zip"]
        else:
            self.default_suffixes = [
                (s if s.startswith(".") else f".{s}") for s in default_suffixes
            ]

        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, relative: str | Path) -> Path:
        """Resolve a relative locator (may include nested subdirs) to an absolute Path
        under self.root_dir + self.root_prefix. Raises FileNotFoundError on unsafe locators.
        """
        rel = Path(self.root_prefix) / relative if self.root_prefix else Path(relative or "")
        full = (self.root_dir / rel).resolve()

        try:
            # will raise ValueError if full is not inside root_dir
            full.relative_to(self.root_dir)
        except Exception:
            raise FileNotFoundError(f"Unsafe locator or path traversal attempt: {relative!r}")

        return full

    def list_resources(
        self, prefix: str | None = None, suffixes: list[str] | None = None
    ) -> Iterator[ResourceInfo]:
        """
        List resources under the optional prefix. The prefix is interpreted as a
        relative path under the configured root_prefix (if any). By default this
        method filters results to the backend's default_suffixes. Pass an empty
        list to disable filtering and list all files.

        Yields ResourceInfo with locator set to a posix-style path relative to root_dir.
        """
        # Determine effective suffixes: None -> use class default_suffixes; [] -> no filter
        if suffixes is None:
            suffixes = self.default_suffixes
        elif len(suffixes) == 0:
            suffixes = None
        else:
            suffixes = [(s if s.startswith(".") else f".{s}") for s in suffixes]

        try:
            start = self._resolve(prefix or "")
        except FileNotFoundError:
            # Unsafe prefix -> yield nothing
            return

        if not start.exists():
            return

        def _matches(name: str) -> bool:
            if suffixes is None:
                return True
            for s in suffixes:
                if name.lower().endswith(s.lower()):
                    return True
            return False

        if start.is_file():
            rel = start.relative_to(self.root_dir).as_posix()
            if _matches(rel):
                try:
                    size = start.stat().st_size
                except OSError:
                    size = None
                try:
                    mtime = datetime.fromtimestamp(start.stat().st_mtime)
                except OSError:
                    mtime = None
                yield ResourceInfo(locator=rel, size=size, last_modified=mtime)
            return

        for p in start.rglob("*"):
            if not p.is_file():
                continue
            name = p.name
            if not _matches(name):
                continue
            rel = p.relative_to(self.root_dir).as_posix()
            try:
                size = p.stat().st_size
            except OSError:
                size = None
            try:
                mtime = datetime.fromtimestamp(p.stat().st_mtime)
            except OSError:
                mtime = None
            yield ResourceInfo(locator=rel, size=size, last_modified=mtime)

    def get_resource_stream(self, locator: str | Path) -> Any:
        """
        Return a readable binary stream for the resource identified by locator.
        The caller is responsible for closing the returned file-like object.

        Raises FileNotFoundError if the locator is invalid or the file does not exist.
        """
        full = self._resolve(locator)
        if not full.is_file():
            raise FileNotFoundError(locator)
        return full.open("rb")

    def get_extracted_file_paths(
        self, locator: str | Path, filenames_to_extract: Sequence[str] = ("ro-crate-metadata.json",)
    ) -> list[Path] |None:
        """
        Return a pathlib.Path pointing to the JSON file identified by locator.

        Raises FileNotFoundError if the locator is invalid or the file does not exist.
        """
        stream = self.get_resource_stream(locator)
        roc_json_paths = extract_files_from_zip_stream(
            stream, list(filenames_to_extract)
        )
        # look for canonical ro-crate metadata filenames (json and jsonld)
        if not roc_json_paths:
            try:
                if hasattr(stream, "close"):
                    stream.close()
            except Exception:
                pass
            return None
        return roc_json_paths
