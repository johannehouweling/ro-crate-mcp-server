from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class IndexEntry(BaseModel):
    crate_id: str
    title: str = ""
    description: str = ""
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
