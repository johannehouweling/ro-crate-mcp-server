import asyncio
import io
import json
import pytest
from rocrate_mcp.index.indexer import Indexer
from rocrate_mcp.index.store import IndexStore
from rocrate_mcp.storage.base import ResourceInfo, StorageBackend


class DummyBackend(StorageBackend):
    def __init__(self, blobs):
        self.blobs = blobs

    def list_resources(self, prefix=None):
        for b in self.blobs:
            yield ResourceInfo(locator=b['name'], size=len(b['data']), last_modified=None)

    def get_resource_stream(self, locator):
        for b in self.blobs:
            if b['name'] == locator:
                return io.BytesIO(b['data'])
        raise FileNotFoundError(locator)


def make_rocrate_json(name, id_):
    return json.dumps({"@id": id_, "name": name}).encode('utf-8')


def test_indexer_basic():
    import zipfile, asyncio

    async def inner():
        # create dummy zip blobs
        blobs = []
        for i in range(3):
            bio = io.BytesIO()
            with zipfile.ZipFile(bio, 'w') as zf:
                zf.writestr('ro-crate-metadata.jsonld', make_rocrate_json(f"crate{i}", f"id{i}"))
            blobs.append({'name': f'crate{i}.zip', 'data': bio.getvalue()})

        backend = DummyBackend(blobs)
        store = IndexStore()
        indexer = Indexer(backend=backend, store=store)
        await indexer.build_index()
        assert store.get('id0') is not None
        assert store.get('id1') is not None
        assert store.get('id2') is not None

    asyncio.run(inner())
