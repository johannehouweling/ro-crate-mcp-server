from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@dataclass
class ResourceInfo:
    locator: str
    size: int|None
    last_modified: datetime|None


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for pluggable storage backends."""

    def list_resources(self, prefix: str|None = None, suffixes: list[str]|None = None) -> Iterator[ResourceInfo]:
        """List resources under the optional prefix, optionally filtering by file suffixes.

        Args:
            prefix: Optional relative path prefix to list under.
            suffixes: List of suffix strings (e.g. ['.zip']). If None, the caller
                should be treated as defaulting to ['.zip']. If an empty list is
                provided, no filtering is applied (all resources are listed).
        """
        ...

    def get_resource_stream(self, locator: str) -> Any:
        """Return a readable binary stream for the resource identified by locator."""
        ...
