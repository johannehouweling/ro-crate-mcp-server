from __future__ import annotations

import asyncio
import datetime
import os
import shutil
from collections.abc import Iterable
from typing import Any

from rocrate.rocrate import ROCrate

from ..models import IndexEntry
from ..rocrate_storage.base import ResourceInfo, StorageBackend
from ..utils.zip_reader import extract_files_from_zip_stream
from .storage.store import IndexStore

# assuming these types exist in your codebase
# from your_module import StorageBackend, IndexStore, ResourceInfo, IndexEntry
# from rocrate.rocrate import ROCrate

type json_leaf = str | int | float | bool | None


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
                def read_and_parse() -> IndexEntry | None:
                    stream = self.backend.get_resource_stream(res.locator)

                    # look for canonical ro-crate metadata filenames (json and jsonld)
                    roc_json_paths = extract_files_from_zip_stream(
                        stream, ["ro-crate-metadata.json", "ro-crate-metadata.jsonld"]
                    )
                    # If the extractor didn't find a metadata file, skip
                    if not roc_json_paths:
                        try:
                            if hasattr(stream, "close"):
                                stream.close()
                        except Exception:
                            pass
                        return None

                    # roc_dummy_path is the temp folder containing ro-crate-metadata.json
                    roc_meta_path = roc_json_paths[0]
                    roc_dummy_path = os.path.dirname(roc_meta_path)

                    roc = None
                    try:
                        roc = ROCrate(roc_dummy_path)
                    except Exception:
                        roc = None

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

                            if isinstance(entities, Iterable) and not isinstance(
                                entities, (str, bytes)
                            ):
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
                                    data = (
                                        entity.as_jsonld()
                                        if hasattr(entity, "as_jsonld")
                                        else entity
                                    )
                                    if isinstance(data, dict):
                                        rows.append(data)

                                extracted[selection] = rows
                                continue

                            # mode 2: nested field values ("Person.name", etc.)
                            value_set: set[json_leaf] = set()

                            for entity in entity_iter:
                                data = (
                                    entity.as_jsonld()
                                    if hasattr(entity, "as_jsonld")
                                    else entity
                                )
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

                    # Parse metadata JSON-LD file directly (preferred for basic fields)
                    title = roc.name[0] if hasattr(roc, "name") else ""
                    description = roc.description[0] if hasattr(roc, "description") else ""

                    extracted_fields = extract_fields(roc, roc_type_or_fields_to_index)

                    entry = IndexEntry(
                        crate_id=roc.get("@id") or res.locator,
                        title = title,
                        description = description,
                        resource_locator=res.locator,
                        resource_size=res.size,
                        resource_last_modified=res.last_modified,
                        metadata_path="ro-crate-metadata.json",
                        top_level_metadata={k:v for k,v in roc.metadata.__dict__.items() if k not in ['crate']},
                        extracted_fields=extracted_fields,
                        embedding=compute_embedding(description or title),
                        indexed_at=datetime.datetime.now(datetime.timezone.utc),
                    )

                    # cleanup temporary extracted folder
                    try:
                        shutil.rmtree(roc_dummy_path)
                    except Exception:
                        pass

                    try:
                        if hasattr(stream, "close"):
                            stream.close()
                    except Exception:
                        pass

                    return entry

                entry = await loop.run_in_executor(None, read_and_parse)
                if entry is not None:
                    self.store.insert(entry)

        for res in self.backend.list_resources():
            tasks.append(asyncio.create_task(worker(res)))

        if tasks:
            await asyncio.gather(*tasks)
