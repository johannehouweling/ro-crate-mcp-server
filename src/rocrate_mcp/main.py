import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from rocrate_mcp.config import Settings
from rocrate_mcp.index.indexer import Indexer
from rocrate_mcp.index.storage.sqlite_store import SqliteFTSIndexStore
from rocrate_mcp.models import IndexEntry, SearchFilter
from rocrate_mcp.rocrate_storage.azure_blob import AzureBlobStorageBackend
from rocrate_mcp.rocrate_storage.filesystem import FilesystemStorageBackend

# ----- Initialization: settings, store, backend, indexer --------------------

settings = Settings()

# Resolve store path and create sqlite-backed index store
# If no indexed_db_path is configured, default to a local file to avoid passing None
_db_path = settings.indexed_db_path or "rocrate_index.sqlite"
store = SqliteFTSIndexStore(_db_path)



# Configure backend
backend: Any
if (settings.backend or "").lower() == "azure":
    if not settings.azure_connection_string or not settings.azure_container:
        raise RuntimeError(
            "ROC_MCP_BACKEND=azure requires ROC_MCP_AZURE_CONNECTION_STRING and ROC_MCP_AZURE_CONTAINER"
        )
    backend = AzureBlobStorageBackend(settings.azure_connection_string, settings.azure_container)
elif (settings.backend or "").lower() == "filesystem":
    if not settings.filesystem_root:
        raise RuntimeError("ROC_MCP_BACKEND=filesystem requires ROC_MCP_FILESYSTEM_ROOT to be set")
    backend = FilesystemStorageBackend(
        settings.filesystem_root,
        root_prefix=settings.filesystem_root_prefix or None,
        default_suffixes=[
            s.strip() for s in (settings.filesystem_default_suffixes or "").split(",") if s.strip()
        ],
    )
else:
    backend = None

print("Using backend:", backend.__dict__ if backend else None)
print("Storage backend instance:", store.__dict__ if store else None)
print("Indexing mode:", settings.index_mode)

# Create indexer (but do NOT run it yet – that happens in the lifespan)
indexer = Indexer(backend=backend, store=store, mode=settings.index_mode)

# ----- Server lifespan: run eager indexing on startup -----------------------

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[None]:
    """Server-level lifespan. Runs once when the MCP server starts."""

    # 1) Ensure DB initialized
    print("Initializing SQLite/FTS store...")
    try:
        await store.init_db()
        print("SQLite/FTS store initialized.")
    except Exception as e:
        print("Store init failed:", e)

    # 2) Optional eager indexing
    if settings.index_mode == "eager" and backend is not None:
        print("Starting eager index build (lifespan)...")
        try:
            await indexer.build_index(
                roc_type_or_fields_to_index=settings.get_fields_to_index()
            )
            print("Eager index build finished (lifespan)")
        except Exception as e:
            print("Eager index build failed (lifespan):", e)
    else:
        print(
            "Skipping eager index build (lifespan). "
            f"index_mode={settings.index_mode!r}, backend={backend!r}"
        )

    # Let the server run
    yield
    # If you ever need shutdown cleanup, put it after yield

# ----- FastMCP app and tools ------------------------------------------------

# IMPORTANT: pass lifespan=app_lifespan so `mcp dev` uses it
mcp = FastMCP(name="rocrate-mcp",lifespan=app_lifespan)


# attach state container for tools to access
mcp.state = SimpleNamespace()
mcp.state.settings = settings
mcp.state.store = store
mcp.state.backend = backend
mcp.state.indexer = indexer



# Compatibility helper used by tests: coroutine that returns indexed crate metadata
async def get_crate(crate_id: str) -> dict[str, Any]:
    entry = await mcp.state.store.get(crate_id)
    if entry is None:
        return {}

    d = entry.to_dict()
    # Convert datetime fields to isoformat
    if "indexed_at" in d and d["indexed_at"] is not None:
        try:
            d["indexed_at"] = d["indexed_at"].isoformat()
        except Exception:
            d["indexed_at"] = str(d["indexed_at"])
    if "resource_last_modified" in d and d["resource_last_modified"] is not None:
        try:
            d["resource_last_modified"] = d["resource_last_modified"].isoformat()
        except Exception:
            d["resource_last_modified"] = str(d["resource_last_modified"])
    return d


@mcp.tool()
async def list_all_indexed_crates(
    limit: int = 100,
    offset: int = 0,
    ctx: Context | None = None,
) -> dict[str, Any]:
    settings_local = mcp.state.settings
    try:
        requested_limit = max(1, int(limit or 0))
    except Exception:
        requested_limit = 100
    max_limit = max(1, int(getattr(settings_local, "max_list_limit", 1000)))
    hard_cap = 10000
    effective_limit = min(requested_limit, max_limit, hard_cap)

    # call async listentries and await result
    entries = await mcp.state.store.listentries(offset=offset, limit=effective_limit)
    result = {}
    crate_ids = [ent.crate_id for ent in entries if ent is not None]
    crate_names = [e.name for e in entries if e is not None]
    total = len(crate_ids)
    page = crate_ids[offset : offset + effective_limit]
    truncated = offset + effective_limit < total

    meta = {
        "requested_limit": requested_limit,
        "effective_limit": effective_limit,
        "offset": offset,
        "total": total,
        "truncated": truncated,
    }
    return {"count": len(page), "crate_ids": page, "meta": meta}


@mcp.resource(uri="rocrate://{crate_id}",mime_type="application/zip")
async def get_crate_metadata(crate_id: str) -> dict[str, Any]:
    """Return the indexed crate metadata for the given crate_id.

    Ensures datetimes are serialized to ISO 8601 strings and returns a stable JSON-serializable dict.
    """
    entry = await mcp.state.store.get(crate_id)
    if entry is None:
        return {}

    d = entry.to_dict()
    # Convert known datetime fields to isoformat where present
    if "indexed_at" in d and d["indexed_at"] is not None:
        try:
            d["indexed_at"] = d["indexed_at"].isoformat()
        except Exception:
            d["indexed_at"] = str(d["indexed_at"])
    if "resource_last_modified" in d and d["resource_last_modified"] is not None:
        try:
            d["resource_last_modified"] = d["resource_last_modified"].isoformat()
        except Exception:
            d["resource_last_modified"] = str(d["resource_last_modified"])

    return d


@mcp.resource(uri="rocrate://{crate_id}/metadata")
async def get_crate_metadata(crate_id: str) -> dict[str, Any]:
    """Return the top-level ro-crate-metadata.json content for the given crate_id.

    This is a convenience endpoint to retrieve the main metadata without downloading
    and extracting the entire crate.
    """
    entry = await mcp.state.store.get(crate_id)
    if entry is None:
        return {}
    return entry.top_level_metadata or {}


@mcp.tool()
async def storage_list_resources(
    prefix: str | None = None,
    suffixes: list[str] | None = None,
    limit: int = 100,
    offset: int = 0,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """List raw storage resources from the configured backend (read-only)."""
    backend_local = mcp.state.backend
    if backend_local is None:
        return {"count": 0, "items": []}

    # enforce sensible bounds
    requested_limit = max(1, int(limit or 0))
    max_limit = max(1, int(getattr(mcp.state.settings, "max_list_limit", 1000)))
    hard_cap = 10000
    max_limit = min(max_limit, hard_cap)

    effective_limit = min(requested_limit, max_limit)

    it = backend_local.list_resources(prefix=prefix, suffixes=suffixes)
    # NOTE: don't convert `it` to list() here or you'll exhaust the iterator
    items: list[dict[str, Any]] = []
    skipped = 0
    total_seen = 0
    for res in it:
        total_seen += 1
        if skipped < offset:
            skipped += 1
            continue
        if len(items) >= effective_limit:
            break
        items.append(
            {
                "locator": res.locator,
                "size": res.size,
                "last_modified": str(res.last_modified),
            }
        )

    meta = {
        "requested_limit": requested_limit,
        "effective_limit": effective_limit,
        "max_limit": max_limit,
        "truncated": requested_limit > effective_limit,
        "total_seen": total_seen,
    }
    return {"count": len(items), "items": items, "meta": meta}


# Optional: run as a standalone server (not needed for `mcp dev main.py`)
if __name__ == "__main__":
    # This will also use app_lifespan because it's attached to FastMCP
    mcp.run()
