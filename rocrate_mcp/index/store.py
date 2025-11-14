from __future__ import annotations

import threading
from typing import Dict, Iterable, List, Optional

import numpy as np

from ..models import IndexEntry, SearchFilter


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

    def search(self, filter: SearchFilter, mode: str = "keyword") -> List[IndexEntry]:
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
            if mode == "keyword" or not q:
                if q:
                    hay = " ".join(str(x) for x in e.extracted_fields.values()).lower()
                    if q not in hay:
                        continue
                results.append(e)
            elif mode == "semantic":
                # collect for later semantic ranking
                results.append(e)
            else:
                results.append(e)
        start = filter.offset
        end = start + filter.limit

        if mode == "semantic" and q:
            # compute query embedding using same mock used in indexer (for now)
            def compute_embedding(text: str) -> np.ndarray:
                h = sum(ord(c) for c in text) % 100
                vec = np.array([float((h + i) % 10) / 10.0 for i in range(8)], dtype=np.float32)
                return vec

            qvec = compute_embedding(q)
            candidates = []
            for e in results:
                if e.embedding is None:
                    continue
                candidates.append((e, np.dot(qvec, np.array(e.embedding, dtype=np.float32))))
            candidates.sort(key=lambda x: x[1], reverse=True)
            return [c[0] for c in candidates[start:end]]

        return results[start:end]
