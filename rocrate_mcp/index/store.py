from __future__ import annotations
from typing import Dict, Iterable, List, Optional
from ..models import IndexEntry, SearchFilter
import threading


class IndexStore:
    def __init__(self):
        self._lock = threading.RLock()
        self._by_id: Dict[str, IndexEntry] = {}

    def insert(self, entry: IndexEntry) -> None:
        with self._lock:
            self._by_id[entry.crate_id] = entry

    def bulk_insert(self, entries: Iterable[IndexEntry]) -> None:
        with self._lock:
            for e in entries:
                self._by_id[e.crate_id] = e

    def get(self, crate_id: str) -> Optional[IndexEntry]:
        return self._by_id.get(crate_id)

    def search(self, filter: SearchFilter) -> List[IndexEntry]:
        # naive search: full-text over extracted_fields values and simple field_filters exact match
        q = (filter.q or "").lower()
        results: List[IndexEntry] = []
        for e in self._by_id.values():
            match = True
            if filter.field_filters:
                for k, v in filter.field_filters.items():
                    if str(e.extracted_fields.get(k, "")).lower().find(v.lower()) == -1:
                        match = False
                        break
            if not match:
                continue
            if q:
                hay = " ".join(str(x) for x in e.extracted_fields.values()).lower()
                if q not in hay:
                    continue
            results.append(e)
        start = filter.offset
        end = start + filter.limit
        return results[start:end]
