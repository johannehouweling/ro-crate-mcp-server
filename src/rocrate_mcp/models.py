from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class IndexEntry(BaseModel):
    crate_id: str
    # Basic fields rocrate 
    name: list = ""
    description: list = ""
    date_published: list[datetime]|None = None
    license: list[str]|None = None
    # Additional indexed fields
    resource_locator: str
    resource_size: int|None = None
    resource_last_modified: datetime|None = None
    metadata_path: str
    top_level_metadata: dict[str, Any] = Field(default_factory=dict)
    extracted_fields: dict[str, Any] = Field(default_factory=dict)
    checksum: str|None = None
    version: str|None = None
    storage_backend_id: str|None = None
    indexed_at: datetime
    validation_status: Literal["unknown", "valid", "invalid"] = "unknown"
    embedding: list[float]|None = None  # dense vector for semantic search


class StorageBackendConfig(BaseModel):
    backend_id: str
    type: Literal["azure_blob", "s3", "filesystem", "generic_http"]
    config: dict[str, Any] = Field(default_factory=dict)
    root_prefix: None|str = None

class SearchFilter(BaseModel):
    q: None|str = None
    field_filters: None|dict[str, str] = None
    limit: int = 50
    offset: int = 0


class FileInfo(BaseModel):
    """Lightweight description of a file inside a crate archive."""
    name: str
    size: int
    is_dir: bool = False


class CrateDownloadResponse(BaseModel):
    """Response for a crate download request returning metadata and optional stream URL.

    In this MCP server we'll stream directly for filesystem backend; for other backends
    the response may include a pre-signed URL in stream_url.
    """
    crate_id: str
    resource_locator: str
    resource_size: int|None = None
    stream_url: None|str = None
    files: list[FileInfo] = []
