"""CLI for running the indexer manually."""
from __future__ import annotations

import asyncio

from rocrate_mcp.config import Settings
from rocrate_mcp.index.indexer import Indexer
from rocrate_mcp.index.store import IndexStore
from rocrate_mcp.storage.azure_blob import AzureBlobStorageBackend


async def main():
    settings = Settings()
    store = IndexStore()
    backend = AzureBlobStorageBackend(settings.azure_connection_string, settings.azure_container)
    indexer = Indexer(backend=backend, store=store, mode=settings.index_mode)
    await indexer.build_index()
    print(f"Indexed {len(store._by_id)} entries")


if __name__ == "__main__":
    asyncio.run(main())
