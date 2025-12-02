import io, zipfile, json, asyncio
from rocrate_mcp.index.indexer import Indexer
from rocrate_mcp.index.storage.store import IndexStore
from rocrate_mcp.rocrate_storage.base import ResourceInfo, StorageBackend

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

async def main():
    blobs = []
    for i in range(3):
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, 'w') as zf:
            zf.writestr('ro-crate-metadata.jsonld', make_rocrate_json(f"crate{i}", f"id{i}"))
        blobs.append({'name': f'crate{i}.zip', 'data': bio.getvalue()})
    backend = DummyBackend(blobs)
    store = IndexStore()
    print('Using DB at', store._db_path)
    indexer = Indexer(backend=backend, store=store)
    await indexer.build_index()
    # query DB
    import sqlite3
    conn = sqlite3.connect(store._db_path)
    cur = conn.execute('SELECT crate_id, resource_locator, top_level_metadata FROM entries')
    rows = cur.fetchall()
    print('rows:', rows)

if __name__ == '__main__':
    asyncio.run(main())
