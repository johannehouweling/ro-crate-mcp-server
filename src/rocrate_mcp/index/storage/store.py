from __future__ import annotations

"""Provide a simple export for the canonical index store implementation.

This module intentionally exposes the sqlite-backed implementation under the
legacy name `IndexStore` so other parts of the code can import from
`rocrate_mcp.index.storage.store` without needing to change import sites.
"""

from .sqlite_store import SqliteFTSIndexStore as IndexStore

# Re-export types for convenience (if needed elsewhere)
__all__ = ["IndexStore"]
