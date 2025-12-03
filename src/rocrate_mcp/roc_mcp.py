import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

from mcp.server.fastmcp import FastMCP

from rocrate_mcp.config import Settings
from rocrate_mcp.index.indexer import Indexer
from rocrate_mcp.index.storage.sqlite_store import SqliteFTSIndexStore
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
