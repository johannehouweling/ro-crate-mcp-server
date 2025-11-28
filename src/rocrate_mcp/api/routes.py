from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..index.storage.store import IndexStore
from ..models import IndexEntry, SearchFilter

router = APIRouter()

@router.get("/health")
async def health():
    return {"status": "ok"}

@router.post("/search")
async def search(filter: SearchFilter, store: IndexStore = Depends()):
    results = store.search(filter)
    return {"count": len(results), "results": [r.dict() for r in results]}

@router.get("/crate/{crate_id}")
async def get_crate(crate_id: str, store: IndexStore = Depends()):
    entry = store.get(crate_id)
    if not entry:
        raise HTTPException(status_code=404, detail="crate not found")
    return entry.dict()

@router.get("/search/by-entity")
async def search_by_entity(type_name: str, prop_path: str, prop_value: str, title_contains: str | None = None, store: IndexStore = Depends()):
    """
    Find crate ids for crates that contain an entity of the given type with prop_path matching prop_value.
    Optionally filter by title substring.
    """
    # If the store is the sqlite-backed store it implements the convenience helper; fall back to scanning otherwise
    if hasattr(store, 'find_crates_by_entity_property'):
        crate_ids = store.find_crates_by_entity_property(type_name, prop_path, prop_value, exact=True)
        if title_contains and hasattr(store, 'find_crates_by_entry_field'):
            # filter crate_ids by title_contains
            filtered = []
            for cid in crate_ids:
                entry = store.get(cid)
                if not entry:
                    continue
                if title_contains.lower() in (entry.title or '').lower():
                    filtered.append(cid)
            crate_ids = filtered
        return {"count": len(crate_ids), "crate_ids": crate_ids}
    # generic fallback: scan
    results = []
    for entry in store.search(SearchFilter()):
        persons = entry.extracted_fields.get(type_name, [])
        if not isinstance(persons, list):
            continue
        for ent in persons:
            if str(ent.get(prop_path)) == prop_value:
                if title_contains:
                    if title_contains.lower() in (entry.title or '').lower():
                        results.append(entry.crate_id)
                        break
                else:
                    results.append(entry.crate_id)
                    break
    return {"count": len(results), "crate_ids": results}
