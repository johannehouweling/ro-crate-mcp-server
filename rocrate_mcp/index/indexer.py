from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import List, Optional

from ..index.store import IndexStore
from ..models import IndexEntry
from ..storage.base import ResourceInfo, StorageBackend
from ..utils.zip_reader import find_file_in_zip_stream


class Indexer:
    def __init__(self, backend: StorageBackend, store: IndexStore, mode: str = "eager", concurrency: int = 8):
        self.backend = backend
        self.store = store
        self.mode = mode
        self.concurrency = concurrency

    async def build_index(self) -> None:
        loop = asyncio.get_running_loop()
        sem = asyncio.Semaphore(self.concurrency)
        tasks: List[asyncio.Task] = []

        async def worker(res: ResourceInfo):
            async with sem:
                # run blocking IO in threadpool
                def read_and_parse():
                    stream = self.backend.get_resource_stream(res.locator)
                    data = find_file_in_zip_stream(stream, ["ro-crate-metadata.jsonld", "ro-crate.jsonld", "ro-crate.json"])
                    if not data:
                        return None
                    try:
                        obj = json.loads(data)
                    except Exception:
                        return None
                    # compute (mock) embedding for now — replace with pluggable provider later
                    def compute_embedding(text: str) -> list[float]:
                        # deterministic simple hash-based mock embedding for tests
                        h = sum(ord(c) for c in text) % 100
                        return [float((h + i) % 10) / 10.0 for i in range(8)]

                    title = obj.get("name") or obj.get("label") or ""
                    combined_text = title if isinstance(title, str) else (title[0] if title else "")

                    entry = IndexEntry(
                        crate_id=obj.get("@id") or res.locator,
                        resource_locator=res.locator,
                        resource_size=res.size,
                        resource_last_modified=res.last_modified,
                        metadata_path="ro-crate-metadata.jsonld",
                        top_level_metadata=obj,
                        extracted_fields={
                            "title": title,
                        },
                        embedding=compute_embedding(combined_text),
                        indexed_at=datetime.utcnow(),
                    )

                    return entry

                entry = await loop.run_in_executor(None, read_and_parse)
                if entry:
                    self.store.insert(entry)

        for res in self.backend.list_resources():
            tasks.append(asyncio.create_task(worker(res)))

        if tasks:
            await asyncio.gather(*tasks)
