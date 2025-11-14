from __future__ import annotations
from typing import Iterator, Protocol, runtime_checkable, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ResourceInfo:
    locator: str
    size: Optional[int]
    last_modified: Optional[datetime]


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for pluggable storage backends."""

    def list_resources(self, prefix: Optional[str] = None) -> Iterator[ResourceInfo]:
        ...

    def get_resource_stream(self, locator: str) -> Any:
        """Return a readable binary stream for the resource identified by locator."""
        ...
