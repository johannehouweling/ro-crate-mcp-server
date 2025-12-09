from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
from collections.abc import Iterable
from typing import Any

from rocrate.rocrate import ROCrate

from ..config import Settings
from ..models import IndexEntry
from ..rocrate_storage.base import ResourceInfo, StorageBackend
from .search.embeddings import get_embeddings
from .storage.store import IndexStore

logger = logging.getLogger(__name__)


# assuming these types exist in your codebase
# from your_module import StorageBackend, IndexStore, ResourceInfo, IndexEntry
# from rocrate.rocrate import ROCrate

type json_leaf = str | int | float | bool | None


def compute_embedding(text: str) -> list[float]:
    # deterministic simple hash-based mock embedding for tests
    h = sum(ord(c) for c in text) % 100
    return [float((h + i) % 10) / 10.0 for i in range(8)]


def get_nested_values(obj: Any, path: list[str]) -> set[json_leaf]:
    """
    Traverse a nested JSON-like structure (dicts/lists/primitives)
    following the given path (list of keys). Returns a set of
    primitive leaf values.
    """
    if not path:
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return {obj}
        return set()

    key, *rest = path
    results: set[json_leaf] = set()

    if isinstance(obj, dict):
        if key not in obj:
            return set()
        return get_nested_values(obj[key], rest)

    if isinstance(obj, list):
        for item in obj:
            results |= get_nested_values(item, path)
        return results

    # primitive but still path left
    return set()


def extract_fields(
    roc: ROCrate,
    selections: list[str],
) -> dict[str, set[json_leaf] | list[dict[str, Any]]]:
    """
    Extract fields from an ROCrate JSON-LD graph.

    selections examples:
        - "Person"                      -> full table of Person entities
        - "Person.name"                 -> names of Persons
        - "Person.affiliation.name"     -> nested traversal
        - "Dataset.license.@id"         -> license ids for datasets

    Semantics:
        * The part before the first '.' is treated as @type
            for roc.get_by_type(type_name).

        * If there is no '.' (i.e. only "Type"), we return
            a *table* of full JSON-LD rows:
            key: "Person" -> list[dict[str, Any]]

        * If there is a path following the type, we return
            a set of deduplicated primitive values:
            key: "Person.name" -> set[json_leaf]
    """
    extracted: dict[str, set[json_leaf] | list[dict[str, Any]]] = {}

    for selection in selections:
        selection = selection.strip()
        if not selection:
            continue

        parts = selection.split(".")
        type_name = parts[0]
        field_parts = parts[1:]  # may be empty

        if not type_name:
            continue

        entities = roc.get_by_type(type_name)
        if entities is None:
            continue

        if isinstance(entities, Iterable) and not isinstance(entities, (str, bytes)):
            entity_iter = entities
        else:
            entity_iter = [entities]

        # mode 1: full table for this type ("Person")
        if not field_parts:
            rows: list[dict[str, Any]] = extracted.get(  # type: ignore[assignment]
                selection, []
            )  # reuse if already started from earlier crates

            # make sure rows is actually a list
            if not isinstance(rows, list):
                rows = []

            for entity in entity_iter:
                data = entity.as_jsonld() if hasattr(entity, "as_jsonld") else entity
                if isinstance(data, dict):
                    rows.append(data)

            extracted[selection] = rows
            continue

        # mode 2: nested field values ("Person.name", etc.)
        value_set: set[json_leaf] = set()

        for entity in entity_iter:
            data = entity.as_jsonld() if hasattr(entity, "as_jsonld") else entity
            values = get_nested_values(data, field_parts)
            if values:
                value_set.update(values)

        if not value_set:
            continue

        current = extracted.get(selection)
        if current is None:
            extracted[selection] = value_set
        else:
            # merge with existing set if there is one
            if isinstance(current, set):
                current.update(value_set)
            else:
                # if existing is a list (should not happen here),
                # we just overwrite to keep types consistent.
                extracted[selection] = value_set

    return extracted


class Indexer:
    def __init__(
        self,
        backend: StorageBackend,
        store: IndexStore,
        mode: str = "eager",
        concurrency: int = 8,
    ) -> None:
        self.backend = backend
        self.store = store
        self.mode = mode
        self.concurrency = concurrency

    async def build_index(
        self,
        roc_type_or_fields_to_index: list[str] = [],
    ) -> None:
        """
        Build an index from all resources exposed by the backend.

        roc_type_or_fields_to_index:
            List of selection strings with two behaviours:

            - "Type.field.subfield" -> collect deduplicated primitive leaves
              from entity.as_jsonld() along that path.

            - "Type" (no dot) -> collect the *full* JSON-LD of all entities
              of that type as a "table" (list[dict[str, Any]]).

        Example:
            [
                "Person",                  # table of all Person entities
                "Person.name",             # all person names
                "Dataset.license.@id",     # license IRIs for datasets
            ]
        """

        loop = asyncio.get_running_loop()
        sem = asyncio.Semaphore(self.concurrency)
        tasks: list[asyncio.Task[Any]] = []

        async def worker(res: ResourceInfo) -> None:
            async with sem:
                # Determine backend_id to build stable crate_id. To keep crate_id
                # length and characters safe, hash the locator portion using sha256.
                backend_id = getattr(self.backend, "backend_id", None) or Settings().backend_id
                locator_hash = hashlib.sha256(str(res.locator).encode("utf-8")).hexdigest()
                crate_id = f"{backend_id}:{locator_hash}"

                # cheap pre-check: if entry exists and size/mtime match, skip extraction
                try:
                    existing = await self.store.find_by_locator(backend_id, res.locator)
                except Exception:
                    existing = None

                if existing is not None:
                    # compare sizes and last_modified
                    size_matches = (existing.resource_size == res.size) or (
                        existing.resource_size is None and res.size is None
                    )
                    lm_matches = (existing.resource_last_modified == res.last_modified) or (
                        existing.resource_last_modified is None and res.last_modified is None
                    )
                    if size_matches and lm_matches:
                        logger.info("[indexer] skipping unchanged resource: %s", res.locator)
                        return None

                def read_and_parse() -> IndexEntry | None:
                    roc_json_paths = self.backend.get_extracted_file_paths(res.locator)
                    if roc_json_paths is None:
                        return None
                    # debug
                    logger.debug("[indexer] extracted paths: %s", roc_json_paths)
                    # If the extractor didn't find a metadata file, skip

                    # roc_meta_path is the temp folder containing ro-crate-metadata.json
                    roc_meta_path = roc_json_paths[0]

                    roc = None
                    try:
                        roc = ROCrate(roc_meta_path.parent)
                    except Exception:
                        # Fallback: attempt to parse the metadata JSON directly and
                        # create a minimal ROC-like object that supports the
                        # attributes and methods used by extract_fields.
                        try:
                            import json

                            data = json.loads(roc_meta_path.read_bytes())

                            class SimpleMetadata:
                                def __init__(self, obj: dict):
                                    self._obj = obj

                                def as_jsonld(self):
                                    return self._obj

                            class SimpleRoc:
                                def __init__(self, obj: dict):
                                    self._obj = obj
                                    # normalize to lists where appropriate
                                    self.name = obj.get("name") if isinstance(obj.get("name"), list) else ([obj.get("name")] if obj.get("name") else [])
                                    self.description = obj.get("description") if isinstance(obj.get("description"), list) else ([obj.get("description")] if obj.get("description") else [])
                                    self.root_dataset = obj
                                    self.metadata = SimpleMetadata(obj)

                                def get_by_type(self, type_name: str):
                                    # Expect top-level key with type_name mapping to a list of entities
                                    val = self._obj.get(type_name)
                                    if val is None:
                                        return None
                                    return val

                            roc = SimpleRoc(data)
                        except Exception:
                            return None

                    # Parse metadata JSON-LD file directly (preferred for basic fields)
                    extracted_fields = extract_fields(roc, roc_type_or_fields_to_index)

                    # Derive human-friendly title/description to build embeddings

                    # Construct ORM IndexEntry mapped object
                    entry = IndexEntry(
                        crate_id=crate_id,
                        storage_backend_id=backend_id,
                        name=" ".join(roc.name) if getattr(roc, "name", None) else "",
                        # Prefer explicit root_dataset description if present; fall back to roc.description
                        description=(" ".join(roc.root_dataset.get("description", [])) if roc is not None and roc.root_dataset.get("description", []) else (" ".join(roc.description) if getattr(roc, "description", None) else "")),
                        license=" ".join(roc.root_dataset.get("license", [])),
                        date_published=(
                            datetime.datetime.fromisoformat(
                                roc.root_dataset.get("datePublished", [])[0]
                            )
                            if roc.root_dataset.get("datePublished", [])
                            else None
                        ),
                        resource_locator=res.locator,
                        resource_size=res.size,
                        resource_last_modified=res.last_modified,
                        metadata_path="ro-crate-metadata.json",
                        checksum_metadata_json=f"sha256:{hashlib.sha256(roc_meta_path.read_bytes()).hexdigest()}",
                        top_level_metadata=roc.metadata.as_jsonld()
                        if getattr(roc, "metadata", None)
                        else {},
                        extracted_fields=extracted_fields,
                        embeddings=[[]],  # will be filled later
                        indexed_at=datetime.datetime.now(datetime.UTC),
                    )
                    text_to_embed = f"{roc.name}\n{roc.description}"
                    # cleanup temporary extracted folder
                    [path.unlink(missing_ok=True) for path in roc_json_paths]

                    return entry, text_to_embed

                logger.info("[indexer] submitting read_and_parse to executor for %s", res.locator)
                result = await loop.run_in_executor(None, read_and_parse)
                if result is None:
                    entry = None
                    text_to_embedd = None
                else:
                    entry, text_to_embedd = result

                logger.debug("[indexer] read_and_parse returned: %s", entry)
                if entry is not None:
                    if text_to_embedd:
                        entry.embeddings = await get_embeddings(text_to_embedd)
                    logger.info("[indexer] inserting entry: %s", getattr(entry, "crate_id", None))
                    try:
                        await self.store.insert(entry)
                    except Exception as e:
                        logger.exception("[indexer] store.insert raised")
                        raise
                else:
                    logger.debug("[indexer] no entry parsed for %s", res.locator)

        for res in self.backend.list_resources():
            tasks.append(asyncio.create_task(worker(res)))

        if tasks:
            await asyncio.gather(*tasks)
