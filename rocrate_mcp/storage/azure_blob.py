from __future__ import annotations
from typing import Iterator, Any, Optional
from datetime import datetime
from azure.storage.blob import ContainerClient, BlobClient
from .base import StorageBackend, ResourceInfo
import io


class AzureBlobStorageBackend(StorageBackend):
    def __init__(self, connection_string: str, container: str, root_prefix: Optional[str] = None):
        self.client = ContainerClient.from_connection_string(connection_string, container_name=container)
        self.root_prefix = root_prefix

    def list_resources(self, prefix: Optional[str] = None) -> Iterator[ResourceInfo]:
        full_prefix = self.root_prefix + (prefix or "") if self.root_prefix else (prefix or "")
        for blob in self.client.list_blobs(name_starts_with=full_prefix):
            yield ResourceInfo(locator=blob.name, size=blob.size, last_modified=blob.last_modified)

    def get_resource_stream(self, locator: str) -> Any:
        blob_client = self.client.get_blob_client(locator)
        stream = io.BytesIO()
        downloader = blob_client.download_blob()
        downloader.readinto(stream)
        stream.seek(0)
        return stream
