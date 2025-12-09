from __future__ import annotations

import json
import logging
import math
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

from rocrate_mcp.config import Settings
from rocrate_mcp.index.search.embeddings import get_embeddings
from rocrate_mcp.index.search.query_parser import FieldNode, PhraseNode, TermNode, parse_query
from rocrate_mcp.models import (
    Base as ORMBase,
)
from rocrate_mcp.models import (
    EntityGlobal,
    EntityInCrate,
    EntityProperty,
    IndexEntry,
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
        self.last_updated_at: datetime | None = None

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
                # Create a normal FTS5 table (no external content= '') so we can
                # insert combined_text rows directly and avoid contentless behavior
                await conn.exec_driver_sql(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts "
                    "USING fts5(crate_id UNINDEXED, combined_text);"
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
        """Flatten nested JSON-like structures into a space-separated token string.

        This produces plain-word tokens instead of embedding raw JSON strings which
        improves FTS indexing and matching for materialized properties.
        """
        tokens: list[str] = []

        def _collect(o: Any) -> None:
            if o is None:
                return
            if isinstance(o, dict):
                for k, v in o.items():
                    # include the key name as a token and recurse on the value
                    try:
                        tokens.append(str(k))
                    except Exception:
                        pass
                    _collect(v)
            elif isinstance(o, (list, tuple)):
                for item in o:
                    _collect(item)
            else:
                try:
                    tokens.append(str(o))
                except Exception:
                    pass

        _collect(obj)
        # filter out empty tokens and join with spaces
        return " ".join(t for t in tokens if t)

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
        """Upsert a single entry in an existing session, preferring lookup by
        storage_backend_id + resource_locator, then checksum, then crate_id."""
        checksum = mapping.get("checksum_metadata_json")
        obj: IndexEntry | None = None

        # 1) Try to find by storage_backend_id + resource_locator if available
        sbid = mapping.get("storage_backend_id")
        locator = mapping.get("resource_locator")
        if sbid and locator:
            res = await session.execute(
                select(IndexEntry).where(
                    IndexEntry.storage_backend_id == sbid,
                    IndexEntry.resource_locator == locator,
                )
            )
            obj = res.scalars().first()

        # 2) Try to find by checksum if available
        if obj is None and checksum:
            res = await session.execute(
                select(IndexEntry).where(IndexEntry.checksum_metadata_json == checksum)
            )
            obj = res.scalars().first()

        # 3) If not found, fall back to primary key (crate_id)
        if obj is None:
            obj = await session.get(IndexEntry, crate_id)

        # Insert or update
        if obj is None:
            obj = IndexEntry(**mapping)
            session.add(obj)
        else:
            for k, v in mapping.items():
                setattr(obj, k, v)

        return obj

    async def find_by_locator(
        self, storage_backend_id: str, resource_locator: str
    ) -> IndexEntry | None:
        """Return IndexEntry for the given backend_id + locator, or None if not found."""
        async with self._Session() as session:
            res = await session.execute(
                select(IndexEntry).where(
                    IndexEntry.storage_backend_id == storage_backend_id,
                    IndexEntry.resource_locator == resource_locator,
                )
            )
            orm = res.scalars().first()
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

    async def delete_by_locator(self, storage_backend_id: str, resource_locator: str) -> None:
        """Hard delete index entry and associated FTS and materialized entities by locator."""
        async with self._engine.begin() as conn:
            # Find crate_id
            res = await conn.execute(
                text(
                    "SELECT crate_id FROM entries WHERE storage_backend_id = :sbid AND resource_locator = :rl"
                ),
                {"sbid": storage_backend_id, "rl": resource_locator},
            )
            row = res.fetchone()
            if not row:
                return
            crate_id = row[0]

            # Remove entity_in_crate rows referencing this crate
            await conn.execute(
                text("DELETE FROM entity_in_crate WHERE crate_id = :cid"), {"cid": crate_id}
            )
            # Remove orphaned entity_properties
            await conn.execute(
                text(
                    "DELETE FROM entity_properties WHERE entity_global_id NOT IN (SELECT entity_global_id FROM entity_in_crate)"
                )
            )
            # Remove orphaned entities_global
            await conn.execute(
                text(
                    "DELETE FROM entities_global WHERE id NOT IN (SELECT entity_global_id FROM entity_in_crate)"
                )
            )
            # Remove FTS entry
            if self._fts_available:
                await conn.execute(
                    text("DELETE FROM entries_fts WHERE crate_id = :cid"), {"cid": crate_id}
                )
            # Finally remove the entry row
            await conn.execute(text("DELETE FROM entries WHERE crate_id = :cid"), {"cid": crate_id})

    async def _update_fts_for_entries(
        self,
        items: Iterable[tuple[str, str]],
    ) -> None:
        """Update FTS index for a set of (crate_id, combined_text)."""
        if not self._fts_available:
            return

        async with self._engine.begin() as conn:
            for crate_id, combined_text in items:
                # ensure combined_text is a plain string for sqlite binding
                ct = combined_text if isinstance(combined_text, str) else json.dumps(combined_text)
                try:
                    await conn.execute(
                        text("DELETE FROM entries_fts WHERE crate_id = :cid"),
                        {"cid": crate_id},
                    )
                    # Use explicit casting to TEXT to avoid sqlite parameter binding issues
                    await conn.execute(
                        text("INSERT INTO entries_fts (crate_id, combined_text) VALUES (:cid, CAST(:ct AS TEXT))"),
                        {"cid": crate_id, "ct": ct},
                    )
                except Exception:
                    logger.exception("failed to update entries_fts for %s", crate_id)

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

        # Record last update time so external callers can observe when the store changed
        self.last_updated_at = datetime.now(UTC)

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
        self.last_updated_at = datetime.now(UTC)

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
            total = await self.count()
            return results, int(total)

        return results

    async def count(self) -> int:
        async with self._engine.connect() as conn:
            total = (await conn.execute(text("SELECT COUNT(*) FROM entries"))).scalar() or 0
            return int(total)

    # ----------------------
    # Search helpers
    # ----------------------
    async def _candidate_ids_for_term(self, term: str, is_phrase: bool = False, fuzzy: bool = False) -> list[str]:
        # Use FTS match for single token/phrase; fallback to simple LIKE scan if FTS
        # returns no usable results (covers cases where entries_fts combined_text is
        # empty or FTS not available).
        if is_phrase:
            fts_query = f'"{term}"'
        else:
            # for fuzzy terms, try a prefix wildcard match in FTS (term*) which
            # gives broader results; otherwise use the exact token
            fts_query = f"{term}*" if fuzzy else term

        if self._fts_available:
            async with self._engine.connect() as conn:
                cur = await conn.execute(
                    text("SELECT crate_id FROM entries_fts WHERE entries_fts MATCH :q"), {"q": fts_query}
                )
                rows = cur.fetchall()
                ids = [r[0] for r in rows if r and r[0] is not None]
                if ids:
                    return ids
                # if FTS returned nothing useful, fall through to LIKE fallback
        # Fallback: perform a case-insensitive LIKE search across name, description,
        # and serialized extracted_fields. Use the raw term (no surrounding quotes).
        like_val = term
        async with self._Session() as session:
            q = select(IndexEntry.crate_id).where(
                text(
                    f"LOWER(name || ' ' || description || ' ' || json_extract(extracted_fields, '$')) LIKE '%' || LOWER(:val) || '%'"
                )
            )
            res = await session.execute(q, {"val": like_val})
            return [r[0] for r in res.fetchall()]

    async def _field_variants(self, prop_path: str) -> list[str]:
        """Return plausible variants for a property name (e.g. fullName -> full_name).
        """
        def to_snake(s: str) -> str:
            import re

            s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', s)
            s2 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1)
            return s2.replace('-', '_').lower()

        def to_camel(s: str) -> str:
            parts = s.split('_')
            if not parts:
                return s
            return parts[0] + ''.join(p.capitalize() for p in parts[1:])

        variants: list[str] = []
        if prop_path not in variants:
            variants.append(prop_path)
        snake = to_snake(prop_path)
        if snake not in variants:
            variants.append(snake)
        camel = to_camel(prop_path)
        if camel not in variants:
            variants.append(camel)
        lower = prop_path.lower()
        if lower not in variants:
            variants.append(lower)
        return variants

    async def _candidate_ids_for_field(
        self, field: str, value: str, exact: bool = False, fuzzy: bool = False
    ) -> list[str]:
        # field may be an entry field like 'description' or a materialized entity path like 'Person.familyName'
        # fuzzy=True indicates the parser requested fuzzy matching (tilde). We prefer FTS lookups
        # when available; otherwise fall back to LIKE queries.
        if "." in field:
            # entity field: Type.prop
            type_name, prop_path = field.split(".", 1)
            # try variants of prop_path
            for variant in await self._field_variants(prop_path):
                # if fuzzy, use LIKE-based search for entity properties
                if fuzzy:
                    ids = await self.find_crates_by_entity_property(type_name, variant, value, exact=False)
                else:
                    ids = await self.find_crates_by_entity_property(type_name, variant, value, exact=exact)
                if ids:
                    return ids
            return []
        else:
            # entry-level field
            if fuzzy:
                return await self.find_crates_by_entry_field(field, value, exact=False)
            return await self.find_crates_by_entry_field(field, value, exact=exact)

    async def list_searchable_fields(self) -> dict:
        """Return available searchable fields: entry columns and materialized entity prop paths.

        Returns: {"entries": [..], "entities": {type_name: [prop_paths...]}}
        """
        result: dict = {"entries": ["crate_id", "name", "description", "license", "resource_locator"], "entities": {}}
        async with self._engine.connect() as conn:
            rows = (await conn.execute(text(
                "SELECT eg.type_name, ep.prop_path FROM entities_global eg JOIN entity_properties ep ON eg.id = ep.entity_global_id"
            ))).fetchall()
            for type_name, prop_path in rows:
                if type_name not in result["entities"]:
                    result["entities"][type_name] = []
                if prop_path not in result["entities"][type_name]:
                    result["entities"][type_name].append(prop_path)
        return result

    async def search(
        self,
        query: None | str = None,
        limit: int = 10,
        offset: int = 0,
        mode: str = "keyword",
    ) -> list[IndexEntry]:
        """Search index using a canonical query string with Lucene-subset parsing.

        Supports: field:value, field:"phrase", quoted phrases, and bare terms.
        Multiple tokens are combined with AND semantics.
        """
        q = (query or "").strip()

        nodes = parse_query(q) if q else []

        # For parsed queries, compute candidate id sets per token and intersect
        candidate_sets: list[set[str]] = []
        for n in nodes:
            if isinstance(n, TermNode):
                ids = await self._candidate_ids_for_term(n.term, is_phrase=False, fuzzy=getattr(n, 'fuzzy', False))
                candidate_sets.append(set(ids))
            elif isinstance(n, PhraseNode):
                ids = await self._candidate_ids_for_term(n.phrase, is_phrase=True, fuzzy=False)
                candidate_sets.append(set(ids))
            elif isinstance(n, FieldNode):
                if isinstance(n.child, TermNode):
                    ids = await self._candidate_ids_for_field(n.field, n.child.term, exact=False, fuzzy=getattr(n.child, 'fuzzy', False))
                    candidate_sets.append(set(ids))
                elif isinstance(n.child, PhraseNode):
                    ids = await self._candidate_ids_for_field(n.field, n.child.phrase, exact=False, fuzzy=False)
                    candidate_sets.append(set(ids))

        if not candidate_sets:
            # nothing matched
            return []

        # intersection
        common = set.intersection(*candidate_sets)
       

        return {"success":True, "hits":[{"crate_id":cid} for cid in common]}

    async def semantic_search(
        self, query: str, limit: int = 10, offset: int = 0, mode="semantic"
    ) -> list[dict]:
        """Perform a semantic search over indexed crates using embeddings."""
        if mode == "semantic" and query and settings.embeddings_provider != "none":
            from sentence_transformers.util import semantic_search, dot_score

            query_embeddings = np.array(await get_embeddings(query, prompt_name="query"))
            # 2) Load all entries that have embeddings, plus extracted_fields for filtering
            async with self._Session() as session:
                res = await session.execute(
                    select(IndexEntry.crate_id, IndexEntry.embeddings).where(
                        IndexEntry.embeddings.isnot(None)
                    )
                )
                rows = res.fetchall()
                logger.debug("semantic_search: fetched rows for embeddings: %s", rows)
            logger.debug("semantic_search: rows outer: %s", rows)

            flat_embeddings: list[np.ndarray] = []
            crate_ids_per_embedding: list[str] = []

            for crate_id, emb in rows:
                # Convert to array so we can reason about ndim
                emb_arr = np.asarray(emb)

                if emb_arr.ndim == 1:
                    # Single embedding (1024,)
                    flat_embeddings.append(emb_arr)
                    crate_ids_per_embedding.append(crate_id)
                elif emb_arr.ndim == 2:
                    # Multiple embeddings (k, 1024)
                    flat_embeddings.extend(emb_arr)  # appends each row
                    crate_ids_per_embedding.extend([crate_id] * emb_arr.shape[0])
                else:
                    [dict(success=False, error_msg="Unexpeced embedding shape in index.")]

            if not flat_embeddings:
                return [dict(success=False, error_msg="No embeddings available in index.")]

            corpus_embeddings = np.stack(flat_embeddings, axis=0)  # (total_embeddings, 1024)

            # Perform semantic search with increased limit to account for multiple embeddings per crate
            factored_limit = query_embeddings.shape[0] * math.ceil(
                corpus_embeddings.shape[0] / len(set(crate_ids_per_embedding)) * limit
            )

            hits_per_query = semantic_search(
                query_embeddings, corpus_embeddings, score_function=dot_score, top_k=factored_limit
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
            ][offset : offset + limit]
            return {"success": True, "limit": limit,"offset":offset, "hits":[{"crate_id":cid,"score":crate_max_scores[cid]} for cid in ranked_crates]}
        elif mode == "semantic" and settings.embeddings_provider != "none":
            return [dict(success=False, error_msg="No query string provided for semantic search.")]
        else:
            return [dict(success=False, error_msg="Semantic search not available")]

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
