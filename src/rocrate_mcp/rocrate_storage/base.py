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
    """Protocol for pluggable storage backends.

    Implementations MUST expose a backend_id attribute which identifies the
    configured backend instance. The indexer will use this backend_id together
    with the resource locator to form a stable crate_id in the format
    "{backend_id}:{resource_locator}".

    The attribute may be None for backward compatibility, but in production a
    non-empty backend_id is recommended.
    """

    backend_id: str | None

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
