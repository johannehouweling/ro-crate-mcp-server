from __future__ import annotations

import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import declarative_base

# SQLAlchemy base for ORM models
Base = declarative_base()


class IndexEntry(Base):
    __tablename__ = "entries"

    crate_id = Column(String, primary_key=True)
    # lists and complex fields are stored as JSON
    name = Column(Text)
    description = Column(Text)
    date_published = Column(DateTime, default=datetime.datetime.now(datetime.UTC))
    license = Column(Text)
    resource_locator = Column(Text)
    resource_size = Column(Integer)
    resource_last_modified = Column(DateTime, default=datetime.datetime.now(datetime.UTC))
    metadata_path = Column(Text)
    top_level_metadata = Column(JSON)
    extracted_fields = Column(JSON)
    checksum_metadata_json = Column(String(64), unique=True, nullable=False)
    version = Column(Text)
    storage_backend_id = Column(Text)
    indexed_at = Column(DateTime, default=datetime.datetime.now(datetime.UTC))
    validation_status = Column(Text)
    embeddings = Column(JSON)
    
    # Keep uniqueness on checksum for integrity, and add a locator-based
    # uniqueness constraint so a given backend+locator maps to a single entry.
    # crate_id is derived as "{storage_backend_id}:{resource_locator}" by the indexer.
    __table_args__ = (
        UniqueConstraint("checksum_metadata_json", name="uq_entries_checksum"),
        UniqueConstraint("storage_backend_id", "resource_locator", name="uq_entries_backend_locator"),
    )   

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the entry."""
        d = {
            "crate_id": self.crate_id,
            "name": self.name or "",
            "description": self.description or "",
            "date_published": self.date_published or None,
            "license": self.license or "",
            "resource_locator": self.resource_locator,
            "resource_size": self.resource_size,
            "resource_last_modified": self.resource_last_modified,
            "metadata_path": self.metadata_path,
            "top_level_metadata": self.top_level_metadata or {},
            "extracted_fields": self.extracted_fields or {},
            "checksum_metadata_json": self.checksum_metadata_json,
            "version": self.version,
            "storage_backend_id": self.storage_backend_id,
            "indexed_at": self.indexed_at,
            "validation_status": self.validation_status,
            "embeddings": self.embeddings,
        }
        return d

    @property
    def combined_text(self) -> str:
        """Return a combined text representation for FTS indexing."""
        parts = [
            self.name or "",
            self.description or "",
        ]
        # Include extracted fields as well
        if self.extracted_fields:
            for value in self.extracted_fields.values():
                if isinstance(value, str):
                    parts.append(value)
                elif isinstance(value, list):
                    parts.extend(str(v) for v in value if isinstance(v, str))
        return " ".join(parts)

# Materialized entity tables for fast property queries
class EntityGlobal(Base):
    __tablename__ = "entities_global"
    id = Column(Integer, primary_key=True, autoincrement=True)
    type_name = Column(Text, nullable=False)
    entity_id = Column(Text)
    label = Column(Text)
    raw_json = Column(Text)
    created_at = Column(Text)
    updated_at = Column(Text)
    __table_args__ = (UniqueConstraint("type_name", "entity_id", name="uq_entities_type_id"),)


class EntityInCrate(Base):
    __tablename__ = "entity_in_crate"
    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_global_id = Column(Integer, ForeignKey("entities_global.id", ondelete="CASCADE"), nullable=False)
    crate_id = Column(Text, nullable=False)
    crate_metadata_path = Column(Text)
    occurrence_json = Column(Text)
    created_at = Column(Text)
    __table_args__ = (UniqueConstraint("entity_global_id", "crate_id", name="uq_entity_in_crate"),)


class EntityProperty(Base):
    __tablename__ = "entity_properties"
    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_global_id = Column(Integer, ForeignKey("entities_global.id", ondelete="CASCADE"), nullable=False)
    prop_path = Column(Text, nullable=False)
    prop_value = Column(Text, nullable=False)
    prop_value_json = Column(Text)
    created_at = Column(Text)


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
