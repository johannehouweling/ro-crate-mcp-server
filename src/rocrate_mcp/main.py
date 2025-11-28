from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.routes import router as api_router
from .config import Settings
from .index.indexer import Indexer
from .index.storage.sqlite_store import SqliteFTSIndexStore
from .index.store import IndexStore
from .rocrate_storage.azure_blob import AzureBlobStorageBackend


def create_app() -> FastAPI:
    settings = Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # prepare store and backend
        if settings.indexed_db_path:
            store = SqliteFTSIndexStore(settings.indexed_db_path)
        else:
            store = IndexStore()
        app.state.store = store

        # setup backend from settings
        if settings.azure_connection_string and settings.azure_container:
            backend = AzureBlobStorageBackend(
                settings.azure_connection_string, settings.azure_container
            )
        else:
            backend = None
        app.state.backend = backend

        # indexer
        indexer = Indexer(backend=backend, store=store, mode=settings.index_mode)
        app.state.indexer = indexer

        # eager index build on startup (before the app begins serving)
        if settings.index_mode == "eager":
            await indexer.build_index(
                roc_type_or_fields_to_index=settings.roc_fields_to_index.split(",")
            )

        try:
            yield
        finally:
            # best-effort cleanup: call close/shutdown/aclose if provided
            for obj in (indexer, store, backend):
                if obj is None:
                    continue
                for method in ("close", "shutdown", "aclose"):
                    fn = getattr(obj, method, None)
                    if callable(fn):
                        res = fn()
                        if hasattr(res, "__await__"):
                            await res
                        break

    app = FastAPI(title="rocrate-mcp", lifespan=lifespan)
    app.include_router(api_router, prefix="/api/v1")

    return app


app = create_app()
