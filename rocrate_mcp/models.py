from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict, Literal
from datetime import datetime


class IndexEntry(BaseModel):
    crate_id: str
    resource_locator: str
    resource_size: Optional[int] = None
    resource_last_modified: Optional[datetime] = None
    metadata_path: str
    top_level_metadata: Dict[str, Any] = Field(default_factory=dict)
    extracted_fields: Dict[str, Any] = Field(default_factory=dict)
    checksum: Optional[str] = None
    version: Optional[str] = None
    storage_backend_id: Optional[str] = None
    indexed_at: datetime
    validation_status: Literal["unknown", "valid", "invalid"] = "unknown"


class StorageBackendConfig(BaseModel):
    backend_id: str
    type: Literal["azure_blob", "s3", "filesystem", "generic_http"]
    config: Dict[str, Any] = Field(default_factory=dict)
    root_prefix: Optional[str] = None


class SearchFilter(BaseModel):
    q: Optional[str] = None
    field_filters: Optional[Dict[str, str]] = None
    limit: int = 50
    offset: int = 0
