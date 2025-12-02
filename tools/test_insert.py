import asyncio
from rocrate_mcp.index.storage.store import IndexStore
from rocrate_mcp.models import IndexEntry
from datetime import datetime, timezone

store = IndexStore()
print('DB:', store._db_path)
# create ORM IndexEntry instance
entry = IndexEntry(
    crate_id='manual1',
    name=['manual'],
    description=['manual desc'],
    date_published=[datetime.now().isoformat()],
    license=['MIT'],
    resource_locator='manual.zip',
    resource_size=123,
    resource_last_modified=None,
    metadata_path='ro-crate-metadata.json',
    top_level_metadata={'name':'manual'},
    extracted_fields={'title':'manual'},
    checksum=None,
    version=None,
    storage_backend_id=None,
    indexed_at=datetime.now(timezone.utc).isoformat(),
    validation_status='unknown',
    embedding=[0.1]*8,
)
print('Inserting entry')
store.insert(entry)
print('Inserted. Query DB')
import sqlite3
conn = sqlite3.connect(store._db_path)
cur = conn.execute('SELECT crate_id, resource_locator, top_level_metadata FROM entries')
rows = cur.fetchall()
print('rows:', rows)
