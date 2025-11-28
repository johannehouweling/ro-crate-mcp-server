import os
import io
import zipfile
import json
import tempfile
import pytest
from rocrate_mcp.index.storage.sqlite_store import SqliteFTSIndexStore
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


def make_rocrate_zip(name, id_, persons=None):
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, 'w') as zf:
        obj = {"@id": id_, "name": name}
        if persons is not None:
            obj['Person'] = persons
        zf.writestr('ro-crate-metadata.json', json.dumps(obj))
    return bio.getvalue()


def test_materialization_and_query(tmp_path):
    # create two crates: one with Scholze, one without
    blobs = []
    persons1 = [{"@id": "https://orcid.org/0000-0002-9569-7562", "@type": "Person", "familyName": "Scholze", "name": "Martin Scholze"}]
    blobs.append({'name': 'crate1.zip', 'data': make_rocrate_zip('crate1', 'crate1', persons=persons1)})
    persons2 = [{"@id": "https://orcid.org/0000-0003-4766-7358", "@type": "Person", "familyName": "Wagenaars", "name": "F.M.A. Wagenaars"}]
    blobs.append({'name': 'crate2.zip', 'data': make_rocrate_zip('crate2', 'crate2', persons=persons2)})

    backend = DummyBackend(blobs)
    # use sqlite store in temp file
    db_path = str(tmp_path / 'test.db')
    store = SqliteFTSIndexStore(db_path)

    indexer = Indexer(backend=backend, store=store)
    import asyncio
    asyncio.run(indexer.build_index(['Person']))

    # query via helper
    crates = store.find_crates_by_entity_property('Person', 'familyName', 'Scholze', exact=True)
    assert 'crate1' in crates
    assert 'crate2' not in crates

    # test combined query with title
    # ensure titles were saved
    assert store.get('crate1').title != ''

    res = store.find_crates_by_entity_and_entry('Person', 'familyName', 'Scholze', 'title', 'crate1', exact=False)
    assert 'crate1' in res
