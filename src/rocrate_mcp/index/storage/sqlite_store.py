from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from rocrate_mcp.index.embeddings import get_embeddings

from ...config import Settings
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

settings = Settings()

logger = logging.getLogger(__name__)
Base = ORMBase


class SqliteFTSIndexStore:
    """Async SQLAlchemy SQLite store using aiosqlite and FTS5.

    All public methods are async. This class creates the async engine and
    async_sessionmaker. On initialization it runs synchronous table/FTS
    creation via the engine.sync_engine for convenience.
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
        self._lock = None  # async code should not use thread locks
        self._fts_available = True
        self._materialize_entities_enabled = materialize_entities

        # async engine and sessionmaker
        self._engine: AsyncEngine = create_async_engine(
            f"sqlite+aiosqlite:///{self._db_path}",
            connect_args={"check_same_thread": False, "autocommit": False},
            future=True,
        )
        self._Session = async_sessionmaker(
            bind=self._engine, expire_on_commit=False, class_=AsyncSession
        )

        # create tables and fts using the sync engine for convenience

    async def init_db(self) -> None:
        async with self._engine.begin() as conn:
            # PRAGMAs
            try:
                await conn.exec_driver_sql("PRAGMA journal_mode = WAL;")
                await conn.exec_driver_sql("PRAGMA synchronous = NORMAL;")
                await conn.exec_driver_sql("PRAGMA busy_timeout = 5000;")
                await conn.exec_driver_sql("PRAGMA foreign_keys = ON;")
            except Exception:
                logger.debug("failed to set pragmas", exc_info=True)

            # ORM tables
            try:
                await conn.run_sync(Base.metadata.create_all)
            except Exception:
                logger.debug("failed to create ORM tables via metadata", exc_info=True)

            # FTS table
            try:
                await conn.exec_driver_sql(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts "
                    "USING fts5( crate_id UNINDEXED, combined_text, content='' );"
                )
            except Exception:
                logger.info("FTS5 not available - falling back to non-FTS search")
                self._fts_available = False

    # ----------------------
    # Serialization helpers
    # ----------------------
    def _ensure_datetime(self, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, (list, tuple)) and v:
            v = v[0]
        if isinstance(v, datetime):
            try:
                return v.astimezone(UTC)
            except Exception:
                return v.replace(tzinfo=UTC)
        if isinstance(v, str):
            try:
                dt = datetime.fromisoformat(v)
                try:
                    return dt.astimezone(UTC)
                except Exception:
                    return dt.replace(tzinfo=UTC)
            except Exception:
                return None
        return None

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

    def _entry_to_mapping(self, e: IndexEntry) -> dict[str, Any]:
        """Convert IndexEntry to a plain mapping suitable for ORM construction/update."""
        return {
            "crate_id": e.crate_id,
            "name": e.name,
            "description": e.description,
            "date_published": self._ensure_datetime(e.date_published),
            "license": e.license,
            "resource_locator": e.resource_locator,
            "resource_size": e.resource_size,
            "resource_last_modified": self._ensure_datetime(e.resource_last_modified),
            "metadata_path": e.metadata_path,
            "top_level_metadata": e.top_level_metadata or {},
            "extracted_fields": e.extracted_fields or {},
            "checksum_metadata_json": getattr(e, "checksum_metadata_json", None),
            "version": e.version,
            "storage_backend_id": e.storage_backend_id,
            "indexed_at": self._ensure_datetime(e.indexed_at) or datetime.now(UTC),
            "validation_status": e.validation_status,
            "embeddings": e.embeddings if e.embeddings is not None else [],
        }

    async def _upsert_entry_in_session(
        self,
        session: AsyncSession,
        mapping: dict[str, Any],
        crate_id: str,
    ) -> IndexEntry:
        """Upsert a single entry in an existing session, using checksum first, then crate_id."""
        checksum = mapping.get("checksum_metadata_json")
        obj: IndexEntry | None = None

        # 1) Try to find by checksum if available
        if checksum:
            res = await session.execute(
                select(IndexEntry).where(IndexEntry.checksum_metadata_json == checksum)
            )
            obj = res.scalars().first()

        # 2) If not found, fall back to primary key (crate_id)
        if obj is None:
            obj = await session.get(IndexEntry, crate_id)

        # 3) Insert or update
        if obj is None:
            obj = IndexEntry(**mapping)
            session.add(obj)
        else:
            for k, v in mapping.items():
                setattr(obj, k, v)

        return obj

    async def _update_fts_for_entries(
        self,
        items: Iterable[tuple[str, str]],
    ) -> None:
        """Update FTS index for a set of (crate_id, combined_text)."""
        if not self._fts_available:
            return

        async with self._engine.begin() as conn:
            for crate_id, combined_text in items:
                await conn.execute(
                    text("DELETE FROM entries_fts WHERE crate_id = :cid"),
                    {"cid": crate_id},
                )
                await conn.execute(
                    text("INSERT INTO entries_fts (crate_id, combined_text) VALUES (:cid, :ct)"),
                    {"cid": crate_id, "ct": combined_text},
                )

    async def _materialize_entries(
        self,
        entries: Iterable[IndexEntry],
    ) -> None:
        """Materialize entities for a batch of entries in a single session."""
        if not self._materialize_entities_enabled:
            return

        entries_list = list(entries)
        if not entries_list:
            return

        try:
            async with self._Session() as session:
                async with session.begin():
                    for e in entries_list:
                        try:
                            await self._materialize_entry(e, session)
                        except Exception:
                            logger.debug(
                                "entity materialization failed for %s",
                                getattr(e, "crate_id", None),
                                exc_info=True,
                            )
        except Exception:
            # Don't blow up if materialization fails
            logger.debug("materialization batch failed", exc_info=True)

    # ----------------------
    # CRUD (async)
    # ----------------------
    async def insert(self, entry: IndexEntry) -> None:
        """Upsert a single entry, deduplicating by checksum if present."""
        mapping = self._entry_to_mapping(entry)

        # Upsert in ORM table
        async with self._Session() as session:
            async with session.begin():
                await self._upsert_entry_in_session(
                    session=session,
                    mapping=mapping,
                    crate_id=entry.crate_id,
                )

        # FTS maintenance
        combined_text = self._make_combined_text(entry)
        await self._update_fts_for_entries([(entry.crate_id, combined_text)])

        # Entity materialization
        await self._materialize_entries([entry])

    async def bulk_insert(self, entries: Iterable[IndexEntry]) -> None:
        """Upsert multiple entries, deduplicating by checksum if present."""
        entries_list = list(entries)
        if not entries_list:
            return

        # Upsert all ORM rows in a single transaction
        async with self._Session() as session:
            async with session.begin():
                for e in entries_list:
                    mapping = self._entry_to_mapping(e)
                    await self._upsert_entry_in_session(
                        session=session,
                        mapping=mapping,
                        crate_id=e.crate_id,
                    )

        # FTS updates for all entries
        fts_items: list[tuple[str, str]] = [
            (e.crate_id, self._make_combined_text(e)) for e in entries_list
        ]
        await self._update_fts_for_entries(fts_items)

        # Materialize all entries
        await self._materialize_entries(entries_list)

    async def get(self, crate_id: str) -> IndexEntry | None:
        async with self._Session() as session:
            orm = await session.get(IndexEntry, crate_id)
            if orm is None:
                return None
            orm.name = orm.name or ""
            orm.description = orm.description or ""
            orm.date_published = orm.date_published or None
            orm.license = orm.license or ""
            orm.top_level_metadata = orm.top_level_metadata or {}
            orm.extracted_fields = orm.extracted_fields or {}
            orm.embeddings = orm.embeddings or []
            return orm

    async def listentries(
        self, offset: int = 0, limit: int = 50, order_by=None, return_total: bool = False
    ):
        if order_by is None:
            try:
                order_by = IndexEntry.indexed_at.desc()
            except Exception:
                order_by = IndexEntry.crate_id

        async with self._Session() as session:
            q = (
                select(IndexEntry)
                .order_by(order_by)
                .offset(max(0, int(offset)))
                .limit(max(1, int(limit)))
            )
            res = await session.execute(q)
            rows = res.scalars().all()

            results: list[IndexEntry] = []
            for orm in rows:
                orm.name = orm.name or ""
                orm.description = orm.description or ""
                orm.date_published = orm.date_published or None
                orm.license = orm.license or ""
                orm.top_level_metadata = orm.top_level_metadata or {}
                orm.extracted_fields = orm.extracted_fields or {}
                orm.embeddings = orm.embeddings or []
                results.append(orm)

        if return_total:
            async with self._engine.connect() as conn:
                total = (await conn.execute(text("SELECT COUNT(*) FROM entries"))).scalar() or 0
            return results, int(total)

        return results

    # ----------------------
    # Search helpers
    # ----------------------
    async def search(self, filter: SearchFilter, mode: str = "keyword") -> list[IndexEntry]:
        q = (filter.q or "").strip()
        candidate_ids: list[str] = []

        if q and mode == "keyword" and self._fts_available:
            async with self._engine.connect() as conn:
                cur = await conn.execute(
                    text("SELECT crate_id FROM entries_fts WHERE entries_fts MATCH :q"), {"q": q}
                )
                candidate_ids = [r[0] for r in cur.fetchall()]
        else:
            async with self._Session() as session:
                res = await session.execute(select(IndexEntry.crate_id))
                candidate_ids = [r[0] for r in res.fetchall()]

        results: list[IndexEntry] = []
        for cid in candidate_ids:
            entry = await self.get(cid)
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
                    parts.append(" ".join(str(x) for x in (entry.extracted_fields or {}).values()))
                    hay = " ".join([p for p in parts if p]).lower()
                    if q.lower() not in hay:
                        continue
                results.append(entry)
            elif mode == "semantic":
                results.append(entry)
            else:
                results.append(entry)

        start = filter.offset
        end = start + filter.limit

        return [await self.get(crate_id) for crate_id in candidate_ids[start:end]]

    async def semantic_search(
        self, query: str, limit: int = 10, offset: int = 0, mode="semantic"
    ) -> list[str]:
        """Perform a semantic search over indexed crates using embeddings."""
        if mode == "semantic" and query and settings.embeddings_provider != "none":
            from sentence_transformers.util import semantic_search, dot_score

            query_embeddings = await get_embeddings(query, prompt_name="query")
            # 2) Load all entries that have embeddings, plus extracted_fields for filtering
            async with self._Session() as session:
                res = await session.execute(
                    select(IndexEntry.crate_id, IndexEntry.embeddings).where(
                        IndexEntry.embeddings.isnot(None)
                    )
                )
                rows = res.fetchall()
                print(rows)
            print(rows)
            corpus_embeddings = [emb for _, emb in rows][0]
            crate_ids_per_embedding = [cid for cid, emb in rows for _ in emb]

            hits_per_query = semantic_search(
                np.array(query_embeddings), np.array(corpus_embeddings), score_function=dot_score
            )
            crate_max_scores: dict[str, float] = {}
            for q_hits in hits_per_query:
                for h in q_hits:
                    idx = h["corpus_id"]
                    score = h["score"]
                    cid = crate_ids_per_embedding[idx]
                    prev = crate_max_scores.get(cid)
                    if prev is None or score > prev:
                        crate_max_scores[cid] = score

            ranked_crates = [
                cid
                for cid, _ in sorted(crate_max_scores.items(), key=lambda kv: kv[1], reverse=True)
            ]
            return ranked_crates
        else:
            return ["Semantic search not enabled"]

    # ----------------------
    # Convenience query helpers (async)
    # ----------------------
    async def find_crates_by_entity_property(
        self, type_name: str, prop_path: str, prop_value: str, exact: bool = True
    ) -> list[str]:
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
        async with self._engine.connect() as conn:
            cur = await conn.execute(text(q), params)
            return [r[0] for r in cur.fetchall()]

    async def find_crates_by_entry_field(
        self, field: str, value: str, exact: bool = True
    ) -> list[str]:
        if exact:
            q = f"SELECT crate_id FROM entries WHERE LOWER({field}) = LOWER(:val)"
            args = {"val": value}
        else:
            q = f"SELECT crate_id FROM entries WHERE LOWER({field}) LIKE '%' || LOWER(:val) || '%'"
            args = {"val": value}
        async with self._engine.connect() as conn:
            cur = await conn.execute(text(q), args)
            return [r[0] for r in cur.fetchall()]

    async def find_crates_by_entity_and_entry(
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
                " WHERE eg.type_name = :tn AND ep.prop_path = :pp AND LOWER(ep.prop_value) = LOWER(:pv)"
                f" AND LOWER(en.{entry_field}) = LOWER(:ev)"
            )
            params = {"tn": type_name, "pp": prop_path, "pv": prop_value, "ev": entry_value}
        else:
            q = (
                "SELECT DISTINCT eic.crate_id"
                " FROM entities_global eg"
                " JOIN entity_properties ep ON eg.id = ep.entity_global_id"
                " JOIN entity_in_crate eic ON eg.id = eic.entity_global_id"
                " JOIN entries en ON en.crate_id = eic.crate_id"
                " WHERE eg.type_name = :tn AND ep.prop_path = :pp AND LOWER(ep.prop_value) LIKE '%' || LOWER(:pv) || '%'"
                f" AND LOWER(en.{entry_field}) LIKE '%' || LOWER(:ev) || '%'"
            )
            params = {"tn": type_name, "pp": prop_path, "pv": prop_value, "ev": entry_value}
        async with self._engine.connect() as conn:
            cur = await conn.execute(text(q), params)
            return [r[0] for r in cur.fetchall()]

    # ----------------------
    # Entity materialization helpers
    # ----------------------
    async def _materialize_entry(self, entry: IndexEntry, session: AsyncSession) -> None:
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

                res = await session.execute(
                    select(EntityGlobal).where(
                        EntityGlobal.type_name == type_name, EntityGlobal.entity_id == entity_id
                    )
                )
                eg = res.scalars().one_or_none()
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
                    await session.flush()

                res = await session.execute(
                    select(EntityInCrate).where(
                        EntityInCrate.entity_global_id == eg.id, EntityInCrate.crate_id == crate_id
                    )
                )
                eic = res.scalars().one_or_none()
                if eic is None:
                    eic = EntityInCrate(
                        entity_global_id=eg.id,
                        crate_id=crate_id,
                        crate_metadata_path=metadata_path,
                        occurrence_json=raw_json,
                        created_at=now,
                    )
                    session.add(eic)

                try:
                    await session.execute(
                        text("DELETE FROM entity_properties WHERE entity_global_id = :egid"),
                        {"egid": eg.id},
                    )
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
