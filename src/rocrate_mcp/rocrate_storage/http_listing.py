from __future__ import annotations

import io
from collections.abc import Iterator, Sequence
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests

from rocrate_mcp.utils.zip_reader import extract_files_from_zip_stream

from .base import ResourceInfo, StorageBackend


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str]]) -> None:
        if tag.lower() != "a":
            return
        for k, v in attrs:
            if k.lower() == "href" and v:
                self.links.append(v)


def _normalize_suffixes(suffixes: list[str] | None, default_suffixes: list[str] | None) -> list[str] | None:
    if suffixes is None:
        return default_suffixes
    if len(suffixes) == 0:
        return None
    return [(s if s.startswith(".") else f".{s}") for s in suffixes]


def _is_parent_link(href: str) -> bool:
    return href in ("..", "../")


class HTTPStorageBackend(StorageBackend):
    """HTTP directory-index based storage backend.

    This backend expects a base_url (e.g. https://example.org/) and will
    interpret locators as posix-style paths relative to an optional root_prefix.
    It attempts to parse HTML directory listings (typical Apache/Nginx style)
    to discover files and subdirectories. For file downloads it performs
    streamed GET requests and returns a file-like object suitable for
    extract_files_from_zip_stream.
    """

    def __init__(
        self,
        base_url: str,
        root_prefix: None | str = None,
        default_suffixes: list[str] | None = None,
        timeout: float | None = 10,
        session: requests.Session | None = None,
        backend_id: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        rp = root_prefix or ""
        self.root_prefix = str(rp).lstrip("/\\")
        if self.root_prefix and not self.root_prefix.endswith("/"):
            # keep as a path fragment without leading slash, but ensure trailing slash
            self.root_prefix = self.root_prefix.rstrip("/") + "/"

        if default_suffixes is None:
            self.default_suffixes: list[str] | None = [".zip"]
        else:
            self.default_suffixes = [
                (s if s.startswith(".") else f".{s}") for s in default_suffixes
            ]

        self.timeout = timeout
        self.session = session or requests.Session()

        # Optional backend identifier; indexer will read this to compute stable crate IDs
        self.backend_id: str | None = backend_id

    def _root_url(self) -> str:
        return urljoin(self.base_url, self.root_prefix or "")

    def _resolve_url(self, locator: str | None = None) -> str:
        """Resolve a posix locator (relative path) to an absolute URL under base_url+root_prefix.

        Ensures the resulting URL starts with the configured root URL to
        avoid accidental traversal outside the configured root.
        """
        root = self._root_url()
        if not locator:
            return root
        # Ensure no leading slash on locator
        loc = str(locator).lstrip("/")
        candidate = urljoin(root, loc)
        # Verify candidate is under root (by comparing path prefixes)
        rp = urlparse(root)
        cp = urlparse(candidate)
        if not cp.path.startswith(rp.path):
            raise FileNotFoundError(f"Locator resolves outside configured root: {locator}")
        return candidate

    def _locator_from_url(self, url: str) -> str | None:
        """Return the locator (path relative to configured root) for an absolute URL, or None if outside root."""
        root = self._root_url()
        if not url.startswith(root):
            return None
        loc = url[len(root):]
        return loc.lstrip("/")

    def _get_listing(self, url: str) -> list[str]:
        """Fetch an HTML listing page and return discovered hrefs as absolute URLs.

        If the request fails (non-200) return an empty list.
        """
        try:
            r = self.session.get(url, timeout=self.timeout)
        except Exception:
            return []
        if r.status_code != 200:
            return []
        html = r.text
        parser = _LinkParser()
        parser.feed(html)
        links: list[str] = []
        for href in parser.links:
            if _is_parent_link(href):
                continue
            abs_url = urljoin(url, href)
            # Only include links under configured root
            if abs_url.startswith(self._root_url()):
                links.append(abs_url)
        return links

    def list_resources(self, prefix: str | None = None, suffixes: list[str] | None = None) -> Iterator[ResourceInfo]:
        """List resources under the optional prefix. Mirrors Filesystem semantics for suffix filtering."""
        eff_suffixes = _normalize_suffixes(suffixes, self.default_suffixes)

        # Start URL to traverse
        try:
            start_url = self._resolve_url(prefix)
        except FileNotFoundError:
            return

        # BFS traversal
        seen: set[str] = set()
        queue: list[str] = [start_url]
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)

            # If the URL points to a file (heuristic: endswith file extension), yield if matches
            path = urlparse(current).path
            if path and not path.endswith("/"):
                # it's a file-like URL
                locator = self._locator_from_url(current)
                if locator is None:
                    continue
                name = locator.split("/")[-1]
                if eff_suffixes is not None:
                    matched = False
                    for s in eff_suffixes:
                        if name.lower().endswith(s.lower()):
                            matched = True
                            break
                    if not matched:
                        continue
                # Attempt to get size via HEAD (best-effort)
                size = None
                last_modified = None
                try:
                    head = self.session.head(current, timeout=self.timeout)
                    if head.status_code == 200:
                        cl = head.headers.get("Content-Length")
                        if cl and cl.isdigit():
                            size = int(cl)
                        lm = head.headers.get("Last-Modified")
                        if lm:
                            try:
                                # best-effort parse, may raise
                                from email.utils import parsedate_to_datetime

                                last_modified = parsedate_to_datetime(lm)
                            except Exception:
                                last_modified = None
                except Exception:
                    pass

                yield ResourceInfo(locator=locator, size=size, last_modified=last_modified)
                continue

            # Otherwise treat as directory: fetch listing and enqueue children
            children = self._get_listing(current)
            for child in children:
                if child not in seen:
                    queue.append(child)

    def get_resource_stream(self, locator: str | Any) -> Any:
        """Return a readable binary stream for the resource identified by locator.

        Raises FileNotFoundError if the locator is invalid or the resource is not accessible.
        The returned stream is an io.BytesIO containing the full content of the resource.
        """
        url = self._resolve_url(str(locator))
        try:
            r = self.session.get(url, stream=True, timeout=self.timeout)
        except Exception:
            raise FileNotFoundError(locator)
        if r.status_code != 200:
            raise FileNotFoundError(locator)
        try:
            data = r.content
        finally:
            try:
                r.close()
            except Exception:
                pass
        return io.BytesIO(data)

    def get_extracted_file_paths(
        self, locator: str | Any, filenames_to_extract: Sequence[str] = ("ro-crate-metadata.json",)
    ) -> list[Path] | None:
        stream = self.get_resource_stream(locator)
        roc_json_paths = extract_files_from_zip_stream(stream, list(filenames_to_extract))
        try:
            if hasattr(stream, "close"):
                stream.close()
        except Exception:
            pass
        if not roc_json_paths:
            return None
        return roc_json_paths
