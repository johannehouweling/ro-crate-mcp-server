SQLite + FTS (Full-Text Search) — Overview and how it applies to IndexStore

What is SQLite + FTS?
- SQLite is an embedded SQL database engine (single-file, no server). FTS (Full-Text Search) is a SQLite extension (FTS3/FTS4/FTS5) that provides fast, indexed full-text search over text columns using an inverted index.

Why use it for the MCP index?
- Persistence: store the crate index on-disk in a compact SQLite file instead of only in memory.
- Fast search: FTS provides tokenization, ranking, and efficient text queries across large numbers of documents (good for thousands to millions of small records).
- Simplicity: SQLite requires no external service and is easy to ship and maintain.

Recommended schema (high-level)
- entries table (metadata, primary key crate_id)
  - crate_id TEXT PRIMARY KEY
  - resource_locator TEXT
  - resource_size INTEGER
  - resource_last_modified TEXT
  - metadata_path TEXT
  - top_level_metadata JSON  -- store as TEXT; SQLite supports JSON functions in modern builds
  - checksum TEXT
  - version TEXT
  - storage_backend_id TEXT
  - indexed_at TEXT
  - validation_status TEXT

- entries_fts virtual table (FTS5)
  - content='entries', content_rowid='rowid' or standalone FTS table
  - columns: crate_id, title, authors, keywords, combined_text
  - combined_text should contain the text you want searchable (title + abstract + keywords + author names)

Example SQL (FTS5, embedding combined text):

CREATE TABLE IF NOT EXISTS entries (
  crate_id TEXT PRIMARY KEY,
  resource_locator TEXT,
  resource_size INTEGER,
  resource_last_modified TEXT,
  metadata_path TEXT,
  top_level_metadata TEXT,
  checksum TEXT,
  version TEXT,
  storage_backend_id TEXT,
  indexed_at TEXT,
  validation_status TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
  crate_id UNINDEXED,
  combined_text,
  content=''
);

When inserting/updating an entry:
- INSERT OR REPLACE INTO entries (...) VALUES (...)
- INSERT OR REPLACE INTO entries_fts(rowid, crate_id, combined_text) VALUES (last_insert_rowid(), ?, ?)
  or use triggers to keep FTS synchronized.

Querying examples:
- Full-text search: SELECT crate_id FROM entries_fts WHERE entries_fts MATCH 'virus NEAR 3 sequencing';
- Combine with filters: SELECT e.* FROM entries e JOIN entries_fts f ON e.rowid=f.rowid WHERE f.combined_text MATCH ? AND e.storage_backend_id=? LIMIT ? OFFSET ?

Implementation notes for Python
- Use the standard library sqlite3 module with row_factory and text_factory; enable detect_types if needed.
- FTS5 is built into recent SQLite versions; ensure the Python runtime’s SQLite supports FTS5 (sqlite3.sqlite_version and sqlite3.sqlite_version_info).
- Use parameterized queries to avoid injection.
- For JSON fields, store the JSON as text. If JSON functions are available, you can query inside them, but a simpler approach is to extract searchable fields at index time into combined_text.

Concurrency and durability
- SQLite supports concurrent readers and a single writer; for frequent writes, use a write queue or a single writer thread/process.
- Use WAL (PRAGMA journal_mode=WAL) for better concurrency (readers won’t block writers as much).
- Commit in batches (e.g., batch inserts in transactions of N entries) to improve throughput.

Pros/Cons
- Pros: zero-dependency, fast for text search, simple to back up and move, transactional.
- Cons: single-writer constraint (mitigated with queue), limited to local disk (not a distributed search engine like Elasticsearch), FTS features depend on SQLite build.

Suggested next steps to implement in IndexStore
1. Add optional sqlite path configuration (RMP_INDEXED_DB_PATH). If set, initialize DB and FTS tables on startup.
2. On insert/bulk_insert, write to entries and entries_fts (batch inside transactions).
3. Implement search(filter) to use FTS when available and fall back to in-memory scanning otherwise.
4. Add load()/persist() helpers to rebuild in-memory caches if desired.

If you’d like, I can implement these changes now: modify IndexStore to add sqlite persistence + FTS (with WAL and batched writes), update tests, and add a small integration test. Reply "implement sqlite+FTS" to proceed.
