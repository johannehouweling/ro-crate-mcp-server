from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Text, Integer, JSON
from pydantic import BaseModel, Field

# SQLAlchemy base for ORM models
Base = declarative_base()


class IndexEntry(Base):
    __tablename__ = "entries"

    crate_id = Column(String, primary_key=True)
    # lists and complex fields are stored as JSON
    name = Column(JSON)
    description = Column(JSON)
    date_published = Column(JSON)
    license = Column(JSON)
    resource_locator = Column(Text)
    resource_size = Column(Integer)
    resource_last_modified = Column(Text)
    metadata_path = Column(Text)
    top_level_metadata = Column(JSON)
    extracted_fields = Column(JSON)
    checksum = Column(Text)
    version = Column(Text)
    storage_backend_id = Column(Text)
    indexed_at = Column(Text)
    validation_status = Column(Text)
    embedding = Column(JSON)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the entry."""
        d = {
            "crate_id": self.crate_id,
            "name": self.name or [],
            "description": self.description or [],
            "date_published": self.date_published or None,
            "license": self.license or [],
            "resource_locator": self.resource_locator,
            "resource_size": self.resource_size,
            "resource_last_modified": self.resource_last_modified,
            "metadata_path": self.metadata_path,
            "top_level_metadata": self.top_level_metadata or {},
            "extracted_fields": self.extracted_fields or {},
            "checksum": self.checksum,
            "version": self.version,
            "storage_backend_id": self.storage_backend_id,
            "indexed_at": self.indexed_at,
            "validation_status": self.validation_status,
            "embedding": self.embedding,
        }
        return d


# Keep other Pydantic models (unchanged)
class StorageBackendConfig(BaseModel):
    backend_id: str
    type: Literal["azure_blob", "s3", "filesystem", "generic_http"]
    config: dict[str, Any] = Field(default_factory=dict)
    root_prefix: None | str = None


class SearchFilter(BaseModel):
    q: None | str = None
    field_filters: None | dict[str, str] = None
    limit: int = 50
    offset: int = 0


class FileInfo(BaseModel):
    """Lightweight description of a file inside a crate archive."""
    name: str
    size: int
    is_dir: bool = False


