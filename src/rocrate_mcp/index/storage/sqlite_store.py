from __future__ import annotations

import json
import logging
import threading
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from ...models import (
    Base as ORMBase,
)
from ...models import (
    EntityGlobal,
    EntityInCrate,
    EntityProperty,
    IndexEntry,
    SearchFilter,
)

logger = logging.getLogger(__name__)

Base = ORMBase
Entry = IndexEntry


class SqliteFTSIndexStore:
    """
    SQLAlchemy ORM-backed SQLite index store using FTS5 for keyword search.

    Uses a declarative Entry model and sessionmaker for DB access. FTS5
    virtual table creation is run via raw DDL. Materialization of entities is
    implemented using ORM models (EntityGlobal, EntityInCrate, EntityProperty).
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
        if s.endswith("/") or s.endswith("\\") or p.suffix == "":
            parent = p
            parent.mkdir(parents=True, exist_ok=True)
            candidate = parent / "rocrate_index.db"
            return str(_unique_path(candidate))

        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)

    def __init__(
        self, db_path: str | None = None, timeout: float = 5.0, materialize_entities: bool = True
    ) -> None:
        if db_path is None:
            db_path = str(Path.cwd())
        resolved = self._resolve_db_path(db_path)
        self._db_path = resolved
        self._lock = threading.RLock()
        self._fts_available = True
        self._materialize_entities_enabled = materialize_entities

        # create SQLAlchemy engine and sessionmaker
        self._engine: Engine = create_engine(
            f"sqlite:///{self._db_path}", connect_args={"check_same_thread": False}, future=True
        )
        self._Session = sessionmaker(bind=self._engine, future=True)

        # create tables and fts
        self._create_tables()

    def _create_tables(self) -> None:
        with self._engine.begin() as conn:
            try:
                conn.execute(text("PRAGMA journal_mode = WAL;"))
                conn.execute(text("PRAGMA synchronous = NORMAL;"))
                conn.execute(text("PRAGMA busy_timeout = 5000;"))
                conn.execute(text("PRAGMA foreign_keys = ON;"))
            except Exception:
                logger.debug("failed to set pragmas", exc_info=True)

            # create ORM-backed tables via metadata
            try:
                Base.metadata.create_all(self._engine)
            except Exception:
                # fallback to explicit DDL for the entries table
                logger.debug("failed to create fallback entries table", exc_info=True)

            # create FTS5 virtual table (raw DDL)
            try:
                conn.execute(
                    text(
                        "CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5( crate_id UNINDEXED, combined_text, content='' );"
                    )
                )
            except Exception:
                logger.info("FTS5 not available - falling back to non-FTS search")
                self._fts_available = False

            # create materialization tables via ORM metadata
            try:
                Base.metadata.create_all(self._engine)
            except OperationalError:
                self._materialize_entities_enabled = False

    # ----------------------
    # Serialization helpers
    # ----------------------


    def _entry_to_mapping(self, entry: IndexEntry) -> dict[str, Any]:
        """Return a mapping of values suitable for constructing/updating an IndexEntry ORM object."""

        return {
            "crate_id": entry.crate_id,
            "name": entry.name,
            "description": entry.description,
            "date_published": entry.date_published,
            "license": entry.license,
            "resource_locator": entry.resource_locator,
            "resource_size": entry.resource_size,
            "resource_last_modified": entry.resource_last_modified
            if entry.resource_last_modified
            else None,
            "metadata_path": entry.metadata_path,
            "top_level_metadata": entry.top_level_metadata or {},
            "extracted_fields": entry.extracted_fields or {},
            "checksum_metadata_json": entry.checksum_metadata_json,
            "version": entry.version,
            "storage_backend_id": entry.storage_backend_id,
            "indexed_at": (
                entry.indexed_at.astimezone(UTC)
                if hasattr(entry.indexed_at, "astimezone")
                else (
                    entry.indexed_at
                    if entry.indexed_at is not None
                    else datetime.now(UTC)
                )
            ),
            "validation_status": entry.validation_status,
            "embedding": entry.embedding if entry.embedding is not None else None,
        }

    # ----------------------
    # CRUD
    # ----------------------
    def insert(self, entry: IndexEntry) -> None:
        """Insert or update an IndexEntry using the ORM.

        This will also update the FTS index and optionally run entity
        materialization.
        """
        mapping = self._entry_to_mapping(entry)
        logger.debug("insert mapping: %s", mapping)
        with self._lock:
            with self._Session() as session:
                obj = session.get(Entry, entry.crate_id)
                if obj is None:
                    obj = Entry(**mapping)
                    session.add(obj)
                else:
                    for k, v in mapping.items():
                        setattr(obj, k, v)
                try:
                    session.commit()
                except Exception:
                    logger.exception("commit failed")
                    raise

            # maintain FTS index using a separate connection
            if self._fts_available:
                combined_text = self._make_combined_text(entry)
                with self._engine.begin() as conn:
                    conn.execute(
                        text("DELETE FROM entries_fts WHERE crate_id = :cid"),
                        {"cid": entry.crate_id},
                    )
                    conn.execute(
                        text(
                            "INSERT INTO entries_fts (crate_id, combined_text) VALUES (:cid, :ct)"
                        ),
                        {"cid": entry.crate_id, "ct": combined_text},
                    )

            # materialize entities using ORM inside a session transaction
            if self._materialize_entities_enabled:
                try:
                    with self._Session() as session:
                        with session.begin():
                            self._materialize_entry(entry, session)
                except Exception:
                    logger.debug(
                        "entity materialization failed for %s", entry.crate_id, exc_info=True
                    )

    def bulk_insert(self, entries: Iterable[IndexEntry]) -> None:
        with self._lock:
            entries_list = list(entries)
            if not entries_list:
                return
            mappings = [self._entry_to_mapping(e) for e in entries_list]
            with self._Session() as session:
                for m in mappings:
                    obj = session.get(Entry, m["crate_id"]) if m.get("crate_id") else None
                    if obj is None:
                        session.add(Entry(**m))
                    else:
                        for k, v in m.items():
                            setattr(obj, k, v)
                session.commit()

            if self._fts_available:
                with self._engine.begin() as conn:
                    for e_row in mappings:
                        crate_id = e_row["crate_id"]
                        try:
                            extracted_fields = e_row["extracted_fields"] or {}
                        except Exception:
                            extracted_fields = {}
                        combined_text = self._make_combined_text_from_json(extracted_fields)
                        conn.execute(
                            text("DELETE FROM entries_fts WHERE crate_id = :cid"), {"cid": crate_id}
                        )
                        conn.execute(
                            text(
                                "INSERT INTO entries_fts (crate_id, combined_text) VALUES (:cid, :ct)"
                            ),
                            {"cid": crate_id, "ct": combined_text},
                        )

            if self._materialize_entities_enabled:
                with self._Session() as session:
                    with session.begin():
                        for e in entries_list:
                            try:
                                self._materialize_entry(e, session)
                            except Exception:
                                logger.debug(
                                    "materialize failed for entry %s",
                                    getattr(e, "crate_id", None),
                                    exc_info=True,
                                )

    def get(self, crate_id: str) -> IndexEntry | None:
        """Return an ORM IndexEntry instance with sensible Python defaults."""
        with self._Session() as session:
            orm = session.get(Entry, crate_id)
            if orm is None:
                return None

            # Normalize values to predictable Python types
            orm.name = orm.name or []
            orm.description = orm.description or []
            orm.date_published = orm.date_published or None
            orm.license = orm.license or []
            orm.top_level_metadata = orm.top_level_metadata or {}
            orm.extracted_fields = orm.extracted_fields or {}
            orm.embedding = orm.embedding or None
            return orm

    def listentries(self) -> list[None | IndexEntry]:
        with self._Session() as session:
            rows = session.query(Entry).all()
            results: list[IndexEntry] = []
            for orm in rows:
                results.append(self.get(orm.crate_id))
            return results

    # ----------------------
    # Search helpers
    # ----------------------
    def _make_combined_text(self, entry: IndexEntry) -> str:
        parts: list[str] = []
        if entry.name:
            if isinstance(entry.name, (list, tuple)):
                parts.append(" ".join(str(x) for x in entry.name))
            else:
                parts.append(str(entry.name))
        if entry.description:
            if isinstance(entry.description, (list, tuple)):
                parts.append(" ".join(str(x) for x in entry.description))
            else:
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
            with self._engine.connect() as conn:
                cur = conn.execute(
                    text("SELECT crate_id FROM entries_fts WHERE entries_fts MATCH :q"), {"q": q}
                )
                candidate_ids = [r[0] for r in cur.fetchall()]
        else:
            with self._Session() as session:
                rows = session.query(Entry.crate_id).all()
                candidate_ids = [r[0] for r in rows]

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
    def find_crates_by_entity_property(
        self, type_name: str, prop_path: str, prop_value: str, exact: bool = True
    ) -> list[str]:
        # Keep raw SQL for this multi-join query for simplicity/performance
        if exact:
            q = (
                "SELECT DISTINCT eic.crate_id"
                " FROM entities_global eg"
                " JOIN entity_properties ep ON eg.id = ep.entity_global_id"
                " JOIN entity_in_crate eic ON eg.id = eic.entity_global_id"
                " WHERE eg.type_name = :type_name AND ep.prop_path = :prop_path AND LOWER(ep.prop_value) = LOWER(:prop_value)"
            )
            params = {"type_name": type_name, "prop_path": prop_path, "prop_value": prop_value}
        else:
            q = (
                "SELECT DISTINCT eic.crate_id"
                " FROM entities_global eg"
                " JOIN entity_properties ep ON eg.id = ep.entity_global_id"
                " JOIN entity_in_crate eic ON eg.id = eic.entity_global_id"
                " WHERE eg.type_name = :type_name AND ep.prop_path = :prop_path AND LOWER(ep.prop_value) LIKE '%' || LOWER(:prop_value) || '%'"
            )
            params = {"type_name": type_name, "prop_path": prop_path, "prop_value": prop_value}
        with self._engine.connect() as conn:
            cur = conn.execute(text(q), params)
            return [r[0] for r in cur.fetchall()]

    def find_crates_by_entry_field(self, field: str, value: str, exact: bool = True) -> list[str]:
        # field expected to be one of the top-level entry columns, e.g. 'name' or 'description'
        if exact:
            q = f"SELECT crate_id FROM entries WHERE LOWER({field}) = LOWER(:val)"
            args = {"val": value}
        else:
            q = f"SELECT crate_id FROM entries WHERE LOWER({field}) LIKE '%' || LOWER(:val) || '%'"
            args = {"val": value}
        with self._engine.connect() as conn:
            cur = conn.execute(text(q), args)
            return [r[0] for r in cur.fetchall()]

    def find_crates_by_entity_and_entry(
        self,
        type_name: str,
        prop_path: str,
        prop_value: str,
        entry_field: str,
        entry_value: str,
        exact: bool = True,
    ) -> list[str]:
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
        with self._engine.connect() as conn:
            cur = conn.execute(text(q), args)
            return [r[0] for r in cur.fetchall()]

    # ----------------------
    # Entity materialization helpers
    # ----------------------
    def _materialize_entry(self, entry: IndexEntry, session) -> None:
        """Materialize entities found in an entry's extracted_fields using ORM models.

        Expects an active transactional session (sessionmaker Session) and will
        create/update EntityGlobal, EntityInCrate and EntityProperty rows.
        """
        try:
            extracted = entry.extracted_fields or {}
        except Exception:
            extracted = {}

        crate_id = entry.crate_id
        metadata_path = entry.metadata_path
        now = datetime.now(timezone.utc).isoformat()

        for sel, val in extracted.items() if isinstance(extracted, dict) else []:
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

                eg = (
                    session.query(EntityGlobal)
                    .filter(EntityGlobal.type_name == type_name)
                    .filter(EntityGlobal.entity_id == entity_id)
                    .one_or_none()
                )
                if eg is not None:
                    eg.label = label
                    eg.raw_json = raw_json
                    eg.updated_at = now
                else:
                    eg = EntityGlobal(
                        type_name=type_name,
                        entity_id=entity_id,
                        label=label,
                        raw_json=raw_json,
                        created_at=now,
                        updated_at=now,
                    )
                    session.add(eg)
                    session.flush()  # ensure eg.id is populated

                # ensure entity_in_crate exists
                eic = (
                    session.query(EntityInCrate)
                    .filter(EntityInCrate.entity_global_id == eg.id)
                    .filter(EntityInCrate.crate_id == crate_id)
                    .one_or_none()
                )
                if eic is None:
                    eic = EntityInCrate(
                        entity_global_id=eg.id,
                        crate_id=crate_id,
                        crate_metadata_path=metadata_path,
                        occurrence_json=raw_json,
                        created_at=now,
                    )
                    session.add(eic)

                # replace properties
                try:
                    session.query(EntityProperty).filter(
                        EntityProperty.entity_global_id == eg.id
                    ).delete()
                except Exception:
                    logger.debug("failed to delete old properties for %s", eg.id, exc_info=True)

                props = self._extract_primitive_properties(entity)
                for prop_path, value in props:
                    try:
                        ep = EntityProperty(
                            entity_global_id=eg.id,
                            prop_path=prop_path,
                            prop_value=str(value),
                            prop_value_json=json.dumps(value),
                            created_at=now,
                        )
                        session.add(ep)
                    except Exception:
                        logger.debug(
                            "failed to add property %s for entity %s",
                            prop_path,
                            entity_id,
                            exc_info=True,
                        )

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
