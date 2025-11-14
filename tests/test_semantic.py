import io
import json
import zipfile
from datetime import datetime

import numpy as np

from rocrate_mcp.index.store import IndexStore
from rocrate_mcp.models import IndexEntry


def make_rocrate_blob(name, id_):
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, 'w') as zf:
        zf.writestr('ro-crate-metadata.json', json.dumps({"@id": id_, "name": name}).encode('utf-8'))
    return bio.getvalue()


def test_semantic_search_basic():
    store = IndexStore()
    blobs = [
        ("alpha crate", "id1"),
        ("beta crate", "id2"),
        ("gamma crate", "id3"),
    ]
    for name, id_ in blobs:
        entry = IndexEntry(
            crate_id=id_,
            resource_locator=f"{id_}.zip",
            metadata_path="ro-crate-metadata.jsonld",
            top_level_metadata={"name": name},
            extracted_fields={"title": name},
            indexed_at=datetime.utcnow(),
            embedding=[float(i)/10.0 for i in range(8)],
        )
        store.insert(entry)

    # query similar to 'alpha'
    # use semantic mode
    results = store.search(filter=type('F', (), {'q': 'alpha', 'field_filters': None, 'limit': 10, 'offset': 0})(), mode='semantic')
    assert len(results) >= 1
    assert results[0].crate_id in ('id1','id2','id3')
