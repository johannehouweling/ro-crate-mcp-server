import asyncio

from fastapi import FastAPI

from .api.routes import router as api_router
from .config import Settings
from .index.indexer import Indexer
from .index.store import IndexStore
from .storage.azure_blob import AzureBlobStorageBackend


def create_app() -> FastAPI:
    settings = Settings()
    app = FastAPI(title="rocrate-mcp")
    app.include_router(api_router, prefix="/api")

    # prepare store and backend
    store = IndexStore()
    app.state.store = store

    # setup backend from settings
    if settings.azure_connection_string and settings.azure_container:
        backend = AzureBlobStorageBackend(settings.azure_connection_string, settings.azure_container)
    else:
        backend = None
    app.state.backend = backend

    # indexer
    indexer = Indexer(backend=backend, store=store, mode=settings.index_mode)
    app.state.indexer = indexer

    @app.on_event("startup")
    async def startup_index():
        if indexer and settings.index_mode == "eager":
            await indexer.build_index()

    return app

app = create_app()
