from __future__ import annotations

import io
from collections.abc import Iterator
from typing import Any

from azure.storage.blob import ContainerClient

from .base import ResourceInfo, StorageBackend


class AzureBlobStorageBackend(StorageBackend):
    def __init__(self, connection_string: str, container: str, root_prefix: str|None = None, default_suffixes: None|[list[str]] = None):
        self.client = ContainerClient.from_connection_string(connection_string, container_name=container)
        self.root_prefix = root_prefix
        if default_suffixes is None:
            self.default_suffixes: list[str]|None = ['.zip']
        else:
            self.default_suffixes = [(s if s.startswith('.') else f'.{s}') for s in default_suffixes]

    def list_resources(self, prefix: str|None = None, suffixes: list[str]|None = None) -> Iterator[ResourceInfo]:
        """
        List blobs under the optional prefix, optionally filtering by suffixes.

        Args:
            prefix: Optional prefix to narrow listing.
            suffixes: List of suffix strings (e.g. ['.zip']). If None, defaults to backend's default_suffixes.
                If an empty list is provided, no filtering is applied.
        """
        # Normalize suffixes similar to filesystem backend
        if suffixes is None:
            suffixes = self.default_suffixes
        elif len(suffixes) == 0:
            suffixes = None
        else:
            suffixes = [(s if s.startswith('.') else f'.{s}') for s in suffixes]

        full_prefix = self.root_prefix + (prefix or "") if self.root_prefix else (prefix or "")
        for blob in self.client.list_blobs(name_starts_with=full_prefix):
            name = blob.name
            if suffixes is not None:
                matched = False
                for s in suffixes:
                    if name.lower().endswith(s.lower()):
                        matched = True
                        break
                if not matched:
                    continue
            yield ResourceInfo(locator=blob.name, size=blob.size, last_modified=blob.last_modified)

    def get_resource_stream(self, locator: str) -> Any:
        blob_client = self.client.get_blob_client(locator)
        stream = io.BytesIO()
        downloader = blob_client.download_blob()
        downloader.readinto(stream)
        stream.seek(0)
        return stream
