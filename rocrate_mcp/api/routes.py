from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from ..models import SearchFilter, IndexEntry
from ..index.store import IndexStore

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
