from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ...models import IndexEntry, SearchFilter


class SqliteFTSIndexStore:
    """
    SQLite-backed index store using FTS5 for keyword search.

    - Creates an `entries` table and a standalone `entries_fts` FTS5 table
      containing (crate_id, combined_text).
    - Stores extracted_fields and embedding as JSON text in the entries table.
    - Materializes top-level entities into a small generic relational schema
      (entities_global, entity_in_crate, entity_properties) to allow fast
      filtering by entity properties without requiring bespoke tables.
    - Designed for a single-writer pattern (caller guarantees writes are
      serialized) — bulk_insert is batched inside a single transaction.
    """

    def _resolve_db_path(self, db_path: str) -> str:
        p = Path(db_path)

        def _unique_path(candidate: Path) -> Path:
            if not candidate.exists():
                return candidate
            stem = candidate.stem
            suffix = candidate.suffix
            parent = candidate.parent
            i = 1
            while True:
                new_name = f"{stem}_{i}{suffix}"
                new_candidate = parent / new_name
                if not new_candidate.exists():
                    return new_candidate
                i += 1

        if p.exists():
            if p.is_dir():
                candidate = p / "rocrate_index.db"
                return str(_unique_path(candidate))
            if p.is_file():
                try:
                    with p.open("rb") as fh:
                        header = fh.read(16)
                    if header.startswith(b"SQLite format 3"):
                        return str(p)
                except Exception:
                    pass
                candidate = p.with_suffix(".db")
                return str(_unique_path(candidate))

        s = str(db_path)
        if s.endswith(os.sep) or p.suffix == "":
            parent = p
            parent.mkdir(parents=True, exist_ok=True)
            candidate = parent / "rocrate_index.db"
            return str(_unique_path(candidate))

        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)

    def __init__(self, db_path: str, timeout: float = 5.0, materialize_entities: bool = True) -> None:
        resolved = self._resolve_db_path(db_path)
        self._db_path = resolved
        self._lock = threading.RLock()
        self._fts_available = True
        self._materialize_entities_enabled = materialize_entities
        self._conn = sqlite3.connect(self._db_path, timeout=timeout, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL;")
        self._conn.execute("PRAGMA synchronous = NORMAL;")
        self._conn.execute("PRAGMA busy_timeout = 5000;")
        try:
            self._conn.execute("PRAGMA foreign_keys = ON;")
        except Exception:
            pass
        self._create_tables()

    def _create_tables(self) -> None:
        with self._conn:
            # entries table now includes title and description for direct querying
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entries (
                    crate_id TEXT PRIMARY KEY,
                    resource_locator TEXT,
                    resource_size INTEGER,
                    resource_last_modified TEXT,
                    metadata_path TEXT,
                    top_level_metadata TEXT,
                    extracted_fields TEXT,
                    title TEXT,
                    description TEXT,
                    checksum TEXT,
                    version TEXT,
                    storage_backend_id TEXT,
                    indexed_at TEXT,
                    validation_status TEXT,
                    embedding TEXT
                );
                """
            )
            try:
                self._conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
                        crate_id UNINDEXED,
                        combined_text,
                        content=''
                    );
                    """
                )
            except sqlite3.OperationalError:
                self._fts_available = False

            # ensure older DBs get the new columns if missing
            try:
                cur = self._conn.execute("PRAGMA table_info(entries);")
                cols = {r[1] for r in cur.fetchall()}  # name is at index 1
                if 'title' not in cols:
                    try:
                        self._conn.execute("ALTER TABLE entries ADD COLUMN title TEXT;")
                    except Exception:
                        pass
                if 'description' not in cols:
                    try:
                        self._conn.execute("ALTER TABLE entries ADD COLUMN description TEXT;")
                    except Exception:
                        pass
            except Exception:
                pass

            # Create generic entity materialization tables using single-statement executes
            try:
                stmts = [
                    (
                        "CREATE TABLE IF NOT EXISTS entities_global ("
                        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                        "type_name TEXT NOT NULL,"
                        "entity_id TEXT,"
                        "label TEXT,"
                        "raw_json TEXT,"
                        "created_at TEXT,"
                        "updated_at TEXT,"
                        "UNIQUE(type_name, entity_id) ON CONFLICT IGNORE"
                        ");"
                    ),
                    (
                        "CREATE TABLE IF NOT EXISTS entity_in_crate ("
                        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                        "entity_global_id INTEGER NOT NULL REFERENCES entities_global(id) ON DELETE CASCADE,"
                        "crate_id TEXT NOT NULL,"
                        "crate_metadata_path TEXT,"
                        "occurrence_json TEXT,"
                        "created_at TEXT,"
                        "UNIQUE(entity_global_id, crate_id) ON CONFLICT IGNORE"
                        ");"
                    ),
                    (
                        "CREATE TABLE IF NOT EXISTS entity_properties ("
                        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                        "entity_global_id INTEGER NOT NULL REFERENCES entities_global(id) ON DELETE CASCADE,"
                        "prop_path TEXT NOT NULL,"
                        "prop_value TEXT NOT NULL,"
                        "prop_value_json TEXT,"
                        "created_at TEXT"
                        ");"
                    ),
                    ("CREATE INDEX IF NOT EXISTS idx_entities_global_type ON entities_global(type_name);"),
                    ("CREATE INDEX IF NOT EXISTS idx_entity_in_crate_crate ON entity_in_crate(crate_id);"),
                    ("CREATE INDEX IF NOT EXISTS idx_entity_properties_path_value ON entity_properties(prop_path, prop_value);"),
                    ("CREATE INDEX IF NOT EXISTS idx_entity_properties_entity ON entity_properties(entity_global_id);"),
                ]
                for s in stmts:
                    self._conn.execute(s)
            except sqlite3.OperationalError:
                self._materialize_entities_enabled = False

    def _entry_to_row(self, entry: IndexEntry) -> dict[str, Any]:
        return {
            "crate_id": entry.crate_id,
            "name": entry.name,
            "description": entry.description,
            "date_published": entry.date_published,
            "license": entry.license,
            "resource_locator": entry.resource_locator,
            "resource_size": entry.resource_size,
            "resource_last_modified": entry.resource_last_modified.isoformat() if entry.resource_last_modified else None,
            "metadata_path": entry.metadata_path,
            "top_level_metadata": json.dumps(entry.top_level_metadata or {}),
            "extracted_fields": json.dumps(entry.extracted_fields or {}),
            "checksum": entry.checksum,
            "version": entry.version,
            "storage_backend_id": entry.storage_backend_id,
            "indexed_at": entry.indexed_at.astimezone(timezone.utc).isoformat(),
            "validation_status": entry.validation_status,
            "embedding": json.dumps(entry.embedding) if entry.embedding is not None else None,
        }

    def _row_to_entry(self, row: sqlite3.Row) -> IndexEntry:
        data: dict[str, Any] = {
            "crate_id": row["crate_id"],
            "name": row["name"] or [],
            "description": row["description"] or [],
            "date_published": row["date_published"] or None,
            "license": row["license"] or None,
            "resource_locator": row["resource_locator"],
            "resource_size": row["resource_size"],
            "resource_last_modified": None,
            "metadata_path": row["metadata_path"],
            "top_level_metadata": {},
            "extracted_fields": {},
            "checksum": row["checksum"],
            "version": row["version"],
            "storage_backend_id": row["storage_backend_id"],
            "indexed_at": datetime.fromisoformat(row["indexed_at"]) if row["indexed_at"] else datetime.now(timezone.utc),
            "validation_status": row["validation_status"] or "unknown",
            "embedding": None,
        }
        if row["resource_last_modified"]:
            try:
                data["resource_last_modified"] = datetime.fromisoformat(row["resource_last_modified"])
            except Exception:
                data["resource_last_modified"] = None
        if row["top_level_metadata"]:
            try:
                data["top_level_metadata"] = json.loads(row["top_level_metadata"])
            except Exception:
                data["top_level_metadata"] = {}
        if row["extracted_fields"]:
            try:
                data["extracted_fields"] = json.loads(row["extracted_fields"])
            except Exception:
                data["extracted_fields"] = {}
        if row["embedding"]:
            try:
                data["embedding"] = json.loads(row["embedding"])
            except Exception:
                data["embedding"] = None
        return IndexEntry.parse_obj(data)

    def insert(self, entry: IndexEntry) -> None:
        with self._lock:
            row = self._entry_to_row(entry)
            with self._conn:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO entries (
                        crate_id, resource_locator, resource_size, resource_last_modified,
                        metadata_path, top_level_metadata, extracted_fields, title, description,
                        checksum, version, storage_backend_id, indexed_at,
                        validation_status, embedding
                    ) VALUES (
                        :crate_id, :resource_locator, :resource_size, :resource_last_modified,
                        :metadata_path, :top_level_metadata, :extracted_fields, :title, :description,
                        :checksum, :version, :storage_backend_id, :indexed_at,
                        :validation_status, :embedding
                    );
                    """,
                    row,
                )
                if self._fts_available:
                    combined_text = self._make_combined_text(entry)
                    self._conn.execute("DELETE FROM entries_fts WHERE crate_id = ?;", (entry.crate_id,))
                    self._conn.execute(
                        "INSERT INTO entries_fts (crate_id, combined_text) VALUES (?, ?);",
                        (entry.crate_id, combined_text),
                    )

                if self._materialize_entities_enabled:
                    try:
                        self._materialize_entry(entry)
                    except Exception:
                        pass

    def bulk_insert(self, entries: Iterable[IndexEntry]) -> None:
        with self._lock:
            entries_list = list(entries)
            rows = [self._entry_to_row(e) for e in entries_list]
            if not rows:
                return
            with self._conn:
                cur = self._conn.cursor()
                cur.executemany(
                    """
                    INSERT OR REPLACE INTO entries (
                        crate_id, resource_locator, resource_size, resource_last_modified,
                        metadata_path, top_level_metadata, extracted_fields, title, description,
                        checksum, version, storage_backend_id, indexed_at,
                        validation_status, embedding
                    ) VALUES (
                        :crate_id, :resource_locator, :resource_size, :resource_last_modified,
                        :metadata_path, :top_level_metadata, :extracted_fields, :title, :description,
                        :checksum, :version, :storage_backend_id, :indexed_at,
                        :validation_status, :embedding
                    );
                    """,
                    rows,
                )
                if self._fts_available:
                    for entry_row in rows:
                        crate_id = entry_row["crate_id"]
                        try:
                            extracted_fields = json.loads(entry_row["extracted_fields"] or "{}")
                        except Exception:
                            extracted_fields = {}
                        combined_text = self._make_combined_text_from_json(extracted_fields)
                        self._conn.execute("DELETE FROM entries_fts WHERE crate_id = ?;", (crate_id,))
                        self._conn.execute("INSERT INTO entries_fts (crate_id, combined_text) VALUES (?, ?);", (crate_id, combined_text))

                if self._materialize_entities_enabled:
                    for e in entries_list:
                        try:
                            self._materialize_entry(e)
                        except Exception:
                            continue

    def get(self, crate_id: str) -> IndexEntry|None:
        cur = self._conn.execute("SELECT * FROM entries WHERE crate_id = ?;", (crate_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    def listentries(self) -> list[None|IndexEntry]:
        cur = self._conn.execute("SELECT * FROM entries;")
        rows = cur.fetchall()
        return [self._row_to_entry(r) for r in rows]
    
    def _make_combined_text(self, entry: IndexEntry) -> str:
        # include title/description along with flattened extracted_fields
        parts: list[str] = []
        if entry.title:
            parts.append(str(entry.title))
        if entry.description:
            parts.append(str(entry.description))
        parts.append(self._make_combined_text_from_json(entry.extracted_fields or {}))
        return " ".join([p for p in parts if p])

    def _make_combined_text_from_json(self, obj: Any) -> str:
        parts: list[str] = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                parts.append(str(k))
                if isinstance(v, (list, tuple)):
                    for item in v:
                        parts.append(str(item))
                else:
                    parts.append(str(v))
        else:
            parts.append(str(obj))
        return " ".join(parts)

    def search(self, filter: SearchFilter, mode: str = "keyword") -> list[IndexEntry]:
        q = (filter.q or "").strip()
        candidate_ids: list[str] = []

        if q and mode == "keyword" and self._fts_available:
            cur = self._conn.execute("SELECT crate_id FROM entries_fts WHERE entries_fts MATCH ?;", (q,))
            candidate_ids = [r["crate_id"] for r in cur.fetchall()]
        else:
            cur = self._conn.execute("SELECT crate_id FROM entries;")
            candidate_ids = [r["crate_id"] for r in cur.fetchall()]

        results: list[IndexEntry] = []
        for cid in candidate_ids:
            entry = self.get(cid)
            if entry is None:
                continue
            match = True
            if filter.field_filters:
                for k, v in filter.field_filters.items():
                    if str(entry.extracted_fields.get(k, "")).lower().find(v.lower()) == -1:
                        match = False
                        break
            if not match:
                continue
            if mode == "keyword" or not q:
                if q and not self._fts_available:
                    hay = " ".join(str(x) for x in (entry.extracted_fields or {}).values()).lower()
                    if q.lower() not in hay:
                        continue
                results.append(entry)
            elif mode == "semantic":
                results.append(entry)
            else:
                results.append(entry)

        start = filter.offset
        end = start + filter.limit

        if mode == "semantic" and q:
            def compute_embedding(text: str) -> list[float]:
                h = sum(ord(c) for c in text) % 100
                return [float((h + i) % 10) / 10.0 for i in range(8)]

            qvec = compute_embedding(q)
            candidates = []
            for e in results:
                if e.embedding is None:
                    continue
                try:
                    emb = list(e.embedding)
                    score = sum(float(a) * float(b) for a, b in zip(qvec, emb))
                    candidates.append((e, score))
                except Exception:
                    continue
            candidates.sort(key=lambda x: x[1], reverse=True)
            return [c[0] for c in candidates[start:end]]

        return results[start:end]

    # ----------------------
    # Convenience query helpers
    # ----------------------
    def find_crates_by_entity_property(self, type_name: str, prop_path: str, prop_value: str, exact: bool = True) -> list[str]:
        if exact:
            q = (
                "SELECT DISTINCT eic.crate_id"
                " FROM entities_global eg"
                " JOIN entity_properties ep ON eg.id = ep.entity_global_id"
                " JOIN entity_in_crate eic ON eg.id = eic.entity_global_id"
                " WHERE eg.type_name = ? AND ep.prop_path = ? AND LOWER(ep.prop_value) = LOWER(?)"
            )
            args = (type_name, prop_path, prop_value)
        else:
            q = (
                "SELECT DISTINCT eic.crate_id"
                " FROM entities_global eg"
                " JOIN entity_properties ep ON eg.id = ep.entity_global_id"
                " JOIN entity_in_crate eic ON eg.id = eic.entity_global_id"
                " WHERE eg.type_name = ? AND ep.prop_path = ? AND LOWER(ep.prop_value) LIKE '%' || LOWER(?) || '%'"
            )
            args = (type_name, prop_path, prop_value)
        cur = self._conn.execute(q, args)
        return [r["crate_id"] for r in cur.fetchall()]

    def find_crates_by_entry_field(self, field: str, value: str, exact: bool = True) -> list[str]:
        # field expected to be one of the top-level entry columns, e.g. 'title' or 'description'
        if exact:
            q = "SELECT crate_id FROM entries WHERE LOWER({}) = LOWER(?)".format(field)
            args = (value,)
        else:
            q = "SELECT crate_id FROM entries WHERE LOWER({}) LIKE '%' || LOWER(?) || '%'".format(field)
            args = (value,)
        cur = self._conn.execute(q, args)
        return [r[0] for r in cur.fetchall()]

    def find_crates_by_entity_and_entry(self, type_name: str, prop_path: str, prop_value: str, entry_field: str, entry_value: str, exact: bool = True) -> list[str]:
        # combine joins against materialized entities and entries table
        if exact:
            q = (
                "SELECT DISTINCT eic.crate_id"
                " FROM entities_global eg"
                " JOIN entity_properties ep ON eg.id = ep.entity_global_id"
                " JOIN entity_in_crate eic ON eg.id = eic.entity_global_id"
                " JOIN entries en ON en.crate_id = eic.crate_id"
                " WHERE eg.type_name = ? AND ep.prop_path = ? AND LOWER(ep.prop_value) = LOWER(?)"
                f" AND LOWER(en.{entry_field}) = LOWER(?)"
            )
            args = (type_name, prop_path, prop_value, entry_value)
        else:
            q = (
                "SELECT DISTINCT eic.crate_id"
                " FROM entities_global eg"
                " JOIN entity_properties ep ON eg.id = ep.entity_global_id"
                " JOIN entity_in_crate eic ON eg.id = eic.entity_global_id"
                " JOIN entries en ON en.crate_id = eic.crate_id"
                " WHERE eg.type_name = ? AND ep.prop_path = ? AND LOWER(ep.prop_value) LIKE '%' || LOWER(?) || '%'"
                f" AND LOWER(en.{entry_field}) LIKE '%' || LOWER(?) || '%'"
            )
            args = (type_name, prop_path, prop_value, entry_value)
        cur = self._conn.execute(q, args)
        return [r[0] for r in cur.fetchall()]

    # ----------------------
    # Entity materialization helpers
    # ----------------------
    def _materialize_entry(self, entry: IndexEntry) -> None:
        try:
            extracted = entry.extracted_fields or {}
        except Exception:
            extracted = {}

        crate_id = entry.crate_id
        metadata_path = entry.metadata_path
        now = datetime.now(timezone.utc).isoformat()

        for sel, val in (extracted.items() if isinstance(extracted, dict) else []):
            if not isinstance(val, list):
                continue
            if not any(isinstance(x, dict) for x in val):
                continue

            type_name = sel
            for entity in val:
                if not isinstance(entity, dict):
                    continue
                entity_id = entity.get("@id") or str(uuid.uuid4())
                label = entity.get("name") or entity.get("title") or None
                raw_json = json.dumps(entity)

                cur = self._conn.execute(
                    "SELECT id FROM entities_global WHERE type_name = ? AND entity_id = ?;",
                    (type_name, entity_id),
                )
                row = cur.fetchone()
                if row:
                    egid = row["id"]
                    try:
                        self._conn.execute(
                            "UPDATE entities_global SET label = ?, raw_json = ?, updated_at = ? WHERE id = ?;",
                            (label, raw_json, now, egid),
                        )
                    except Exception:
                        pass
                else:
                    cur = self._conn.execute(
                        "INSERT INTO entities_global (type_name, entity_id, label, raw_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?);",
                        (type_name, entity_id, label, raw_json, now, now),
                    )
                    egid = cur.lastrowid

                cur = self._conn.execute(
                    "SELECT id FROM entity_in_crate WHERE entity_global_id = ? AND crate_id = ?;",
                    (egid, crate_id),
                )
                if not cur.fetchone():
                    self._conn.execute(
                        "INSERT INTO entity_in_crate (entity_global_id, crate_id, crate_metadata_path, occurrence_json, created_at) VALUES (?, ?, ?, ?, ?);",
                        (egid, crate_id, metadata_path, raw_json, now),
                    )

                try:
                    self._conn.execute("DELETE FROM entity_properties WHERE entity_global_id = ?;", (egid,))
                except Exception:
                    pass

                props = self._extract_primitive_properties(entity)
                for prop_path, value in props:
                    try:
                        self._conn.execute(
                            "INSERT INTO entity_properties (entity_global_id, prop_path, prop_value, prop_value_json, created_at) VALUES (?, ?, ?, ?, ?);",
                            (egid, prop_path, str(value), json.dumps(value), now),
                        )
                    except Exception:
                        continue

    def _extract_primitive_properties(self, obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
        results: list[tuple[str, Any]] = []

        def _recurse(o: Any, path: str) -> None:
            if isinstance(o, dict):
                for k, v in o.items():
                    new_path = f"{path}.{k}" if path else k
                    _recurse(v, new_path)
            elif isinstance(o, list):
                for item in o:
                    _recurse(item, path)
            else:
                results.append((path, o))

        _recurse(obj, prefix)
        return results
